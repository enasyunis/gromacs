#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "statutil.h"
#include "mdrun.h"
#include "md_logging.h"
#include "md_support.h"
#include "network.h"
#include "pull.h"
#include "names.h"
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "partdec.h"
#include "coulomb.h"
#include "mvdata.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "tpxio.h"
#include "txtdump.h"
#include "gmx_detect_hardware.h"
#include "gmx_omp_nthreads.h"
#include "pull_rotation.h"
#include "calc_verletbuf.h"
#include "../mdlib/nbnxn_search.h"
#include "../mdlib/nbnxn_consts.h"
#include "gmx_fatal_collective.h"
#include "types/membedt.h"
#include "gmx_omp.h"
#include "gmx_thread_affinity.h"

#include "tmpi.h"


#include "gpu_utils.h"
#include "nbnxn_cuda_data_mgmt.h"


/* The array should match the eI array in include/types/enums.h */
gmx_large_int_t     deform_init_init_step_tpx;
matrix              deform_init_box_tpx;
tMPI_Thread_mutex_t deform_init_box_mutex = TMPI_THREAD_MUTEX_INITIALIZER;


struct mdrunner_arglist
{
    gmx_hw_opt_t   *hw_opt;
    FILE           *fplog;
    t_commrec      *cr;
    int             nfile;
    const t_filenm *fnm;
    output_env_t    oenv;
    gmx_bool        bVerbose;
    gmx_bool        bCompact;
    int             nstglobalcomm;
    ivec            ddxyz;
    int             dd_node_order;
    real            rdd;
    real            rconstr;
    const char     *dddlb_opt;
    real            dlb_scale;
    const char     *ddcsx;
    const char     *ddcsy;
    const char     *ddcsz;
    const char     *nbpu_opt;
    int             nsteps_cmdline;
    int             nstepout;
    int             resetstep;
    int             nmultisim;
    int             repl_ex_nst;
    int             repl_ex_nex;
    int             repl_ex_seed;
    real            pforce;
    real            cpt_period;
    real            max_hours;
    const char     *deviceOptions;
    unsigned long   Flags;
    int             ret; /* return value */
};




static int get_tmpi_omp_thread_division(const gmx_hw_info_t *hwinfo,
                                        const gmx_hw_opt_t  *hw_opt,
                                        int                  nthreads_tot,
                                        int                  ngpu)
{
    int nthreads_tmpi;

    /* There are no separate PME nodes here, as we ensured in
     * check_and_update_hw_opt that nthreads_tmpi>0 with PME nodes
     * and a conditional ensures we would not have ended up here.
     * Note that separate PME nodes might be switched on later.
     */
    {
        /* TODO choose nthreads_omp based on hardware topology
           when we have a hardware topology detection library */
        /* In general, when running up to 4 threads, OpenMP should be faster.
         * Note: on AMD Bulldozer we should avoid running OpenMP over two dies.
         * On Intel>=Nehalem running OpenMP on a single CPU is always faster,
         * even on two CPUs it's usually faster (but with many OpenMP threads
         * it could be faster not to use HT, currently we always use HT).
         * On Nehalem/Westmere we want to avoid running 16 threads over
         * two CPUs with HT, so we need a limit<16; thus we use 12.
         * A reasonable limit for Intel Sandy and Ivy bridge,
         * not knowing the topology, is 16 threads.
         */
        const int nthreads_omp_always_faster             =  4;
        const int nthreads_omp_always_faster_Nehalem     = 12;
        const int nthreads_omp_always_faster_SandyBridge = 16;
        const int first_model_Nehalem                    = 0x1A;
        const int first_model_SandyBridge                = 0x2A;
        gmx_bool  bIntel_Family6;

        bIntel_Family6 =
            (gmx_cpuid_vendor(hwinfo->cpuid_info) == GMX_CPUID_VENDOR_INTEL &&
             gmx_cpuid_family(hwinfo->cpuid_info) == 6);

        if (nthreads_tot <= nthreads_omp_always_faster ||
            (bIntel_Family6 &&
             ((gmx_cpuid_model(hwinfo->cpuid_info) >= nthreads_omp_always_faster_Nehalem && nthreads_tot <= nthreads_omp_always_faster_Nehalem) ||
              (gmx_cpuid_model(hwinfo->cpuid_info) >= nthreads_omp_always_faster_SandyBridge && nthreads_tot <= nthreads_omp_always_faster_SandyBridge))))
        {
            printf("\n---- runner.c Using pure OpenMP ---- \n");
            /* Use pure OpenMP parallelization */
            nthreads_tmpi = 1;
        }
        else
        {
            printf("\n----- runner.c  Not using OpenMP -----\n");
            /* Don't use OpenMP parallelization */
            nthreads_tmpi = nthreads_tot;
        }
    }

    return nthreads_tmpi;
}


/* Get the number of threads to use for thread-MPI based on how many
 * were requested, which algorithms we're using,
 * and how many particles there are.
 * At the point we have already called check_and_update_hw_opt.
 * Thus all options should be internally consistent and consistent
 * with the hardware, except that ntmpi could be larger than #GPU.
 */
static int get_nthreads_mpi(gmx_hw_info_t *hwinfo,
                            gmx_hw_opt_t *hw_opt,
                            t_inputrec *inputrec, gmx_mtop_t *mtop,
                            const t_commrec *cr,
                            FILE *fplog)
{
// GETTING CALLED 
    int      nthreads_hw, nthreads_tot_max, nthreads_tmpi, nthreads_new;
    int      min_atoms_per_mpi_thread;
    char    *env;
    char     sbuf[STRLEN];


    nthreads_hw = hwinfo->nthreads_hw_avail;

    nthreads_tot_max = nthreads_hw;


    nthreads_tmpi =
    get_tmpi_omp_thread_division(hwinfo, hw_opt, nthreads_tot_max, 0);
    min_atoms_per_mpi_thread = MIN_ATOMS_PER_MPI_THREAD;


    return nthreads_tmpi;
}


/* Environment variable for setting nstlist */
static const char*  NSTLIST_ENVVAR          =  "GMX_NSTLIST";
/* Try to increase nstlist when using a GPU with nstlist less than this */
static const int    NSTLIST_GPU_ENOUGH      = 20;
/* Increase nstlist until the non-bonded cost increases more than this factor */
static const float  NBNXN_GPU_LIST_OK_FAC   = 1.25;
/* Don't increase nstlist beyond a non-bonded cost increases of this factor */
static const float  NBNXN_GPU_LIST_MAX_FAC  = 1.40;


static void prepare_verlet_scheme(FILE             *fplog,
                                  gmx_hw_info_t    *hwinfo,
                                  t_commrec        *cr,
                                  gmx_hw_opt_t     *hw_opt,
                                  const char       *nbpu_opt,
                                  t_inputrec       *ir,
                                  const gmx_mtop_t *mtop,
                                  matrix            box,
                                  gmx_bool         *bUseGPU)
{
    /* Here we only check for GPU usage on the MPI master process,
     * as here we don't know how many GPUs we will use yet.
     * We check for a GPU on all processes later.
     */

        /* Update the Verlet buffer size for the current run setup */
        verletbuf_list_setup_t ls;
        real                   rlist_new;

        /* Here we assume CPU acceleration is on. But as currently
         * calc_verlet_buffer_size gives the same results for 4x8 and 4x4
         * and 4x2 gives a larger buffer than 4x4, this is ok.
         */
        verletbuf_get_list_setup(*bUseGPU, &ls);

        calc_verlet_buffer_size(mtop, det(box), ir,
                                ir->verletbuf_drift, &ls,
                                NULL, &rlist_new);
}


static void check_and_update_hw_opt(gmx_hw_opt_t *hw_opt,
                                    int           cutoff_scheme,
                                    gmx_bool      bIsSimMaster)
{
    gmx_omp_nthreads_read_env(&hw_opt->nthreads_omp, bIsSimMaster);

}



/* Data structure set by SIMMASTER which needs to be passed to all nodes
 * before the other nodes have read the tpx file and called gmx_detect_hardware.
 */
typedef struct {
    int      cutoff_scheme; /* The cutoff scheme from inputrec_t */
    gmx_bool bUseGPU;       /* Use GPU or GPU emulation          */
} master_inf_t;

int mdrunner(gmx_hw_opt_t *hw_opt,
             FILE *fplog, t_commrec *cr, int nfile,
             const t_filenm fnm[], const output_env_t oenv, gmx_bool bVerbose,
             gmx_bool bCompact, int nstglobalcomm,
             ivec ddxyz, int dd_node_order, real rdd, real rconstr,
             const char *dddlb_opt, real dlb_scale,
             const char *ddcsx, const char *ddcsy, const char *ddcsz,
             const char *nbpu_opt,
             int nsteps_cmdline, int nstepout, int resetstep,
             int nmultisim, int repl_ex_nst, int repl_ex_nex,
             int repl_ex_seed, real pforce, real cpt_period, real max_hours,
             const char *deviceOptions, unsigned long Flags)
{

    double          nodetime = 0, realtime;
    t_inputrec     *inputrec;
    t_state        *state = NULL;
    matrix          box;
    gmx_ddbox_t     ddbox = {0};
    int             npme_major, npme_minor;
    real            tmpr1, tmpr2;
    t_nrnb         *nrnb;
    gmx_mtop_t     *mtop       = NULL;
    t_mdatoms      *mdatoms    = NULL;
    t_forcerec     *fr         = NULL;
    t_fcdata       *fcd        = NULL;
    gmx_pme_t      *pmedata    = NULL;
    int             i, m, nChargePerturbed = -1, status, nalloc;
    char           *gro;
    gmx_wallcycle_t wcycle;
    gmx_bool        bReadRNG, bReadEkin;
    int             list;
    gmx_runtime_t   runtime;
    int             rc;
    gmx_large_int_t reset_counters;
    gmx_edsam_t     ed           = NULL;
    t_commrec      *cr_old       = cr;
    int             nthreads_pme = 1;
    int             nthreads_pp  = 1;
    gmx_hw_info_t  *hwinfo       = NULL;
    master_inf_t    minf         = {-1, FALSE};

    /* CAUTION: threads may be started later on in this function, so
       cr doesn't reflect the final parallel state right now */
    snew(inputrec, 1);
    snew(mtop, 1);

    snew(state, 1);
    /* Read (nearly) all data required for the simulation */
    read_tpx_state(ftp2fn(efTPX, nfile, fnm), inputrec, state, NULL, mtop);

    /* Detect hardware, gather information. With tMPI only thread 0 does it
     * and after threads are started broadcasts hwinfo around. */
    snew(hwinfo, 1);
    gmx_detect_hardware(fplog, hwinfo, cr,
                            0,0, hw_opt->gpu_id);

    minf.cutoff_scheme = inputrec->cutoff_scheme;
    minf.bUseGPU       = FALSE;

    prepare_verlet_scheme(fplog, hwinfo, cr, hw_opt, nbpu_opt,
                                  inputrec, mtop, state->box,
                                  &minf.bUseGPU);

    /* Check for externally set OpenMP affinity and turn off internal
     * pinning if any is found. We need to do this check early to tell
     * thread-MPI whether it should do pinning when spawning threads.
     * TODO: the above no longer holds, we should move these checks down
     */
    gmx_omp_check_thread_affinity(fplog, cr, hw_opt);

    gmx_omp_nthreads_read_env(&hw_opt->nthreads_omp, 1);


        /* Early check for externally set process affinity. Can't do over all
         * MPI processes because hwinfo is not available everywhere, but with
         * thread-MPI it's needed as pinning might get turned off which needs
         * to be known before starting thread-MPI. */
        gmx_check_thread_affinity_set(fplog,
                                      NULL,
                                      hw_opt, hwinfo->nthreads_hw_avail, FALSE);



        /* NOW the threads will be started: */
        hw_opt->nthreads_tmpi = get_nthreads_mpi(hwinfo,
                                                 hw_opt,
                                                 inputrec, mtop,
                                                 cr, fplog);

    /* END OF CAUTION: cr is now reliable */

    pr_inputrec(fplog, 0, "Input Parameters", inputrec, FALSE);


    /* now make sure the state is initialized and propagated */
    set_state_entries(state, inputrec, cr->nnodes);



    /* NMR restraints must be initialized before load_checkpoint,
     * since with time averaging the history is added to t_state.
     * For proper consistency check we therefore need to extend
     * t_state here.
     * So the PME-only nodes (if present) will also initialize
     * the distance restraints.
     */
    snew(fcd, 1);

    /* This needs to be called before read_checkpoint to extend the state */
    init_disres(fplog, mtop, inputrec, cr, Flags & MD_PARTDEC, fcd, state, repl_ex_nst > 0);


    copy_mat(state->box, box);


    /* PME, if used, is done on all nodes with 1D decomposition */
    cr->npmenodes = 0;
    cr->duty      = (DUTY_PP | DUTY_PME);
    npme_minor    = 1;
    npme_major = cr->nnodes;


    /* Initialize per-physical-node MPI process/thread ID and counters. */
    gmx_init_intranode_counters(cr);


    md_print_info(cr, fplog, "Using %d MPI %s\n",
                  cr->nnodes,
                  cr->nnodes == 1 ? "thread" : "threads"
                  );
    fflush(stderr);

    gmx_omp_nthreads_init(fplog, cr,
                          hwinfo->nthreads_hw_avail,
                          hw_opt->nthreads_omp,
                          hw_opt->nthreads_omp_pme,
                          0,
                          TRUE);

    gmx_check_hw_runconf_consistency(fplog, hwinfo, cr, hw_opt->nthreads_tmpi, minf.bUseGPU);

    /* getting number of PP/PME threads
       PME: env variable should be read only on one node to make sure it is
       identical everywhere;
     */
    /* TODO nthreads_pp is only used for pinning threads.
     * This is a temporary solution until we have a hw topology library.
     */
    nthreads_pp  = gmx_omp_nthreads_get(emntNonbonded);
    nthreads_pme = gmx_omp_nthreads_get(emntPME);

    wcycle = wallcycle_init(fplog, resetstep, cr, nthreads_pp, nthreads_pme);


    snew(nrnb, 1);
        /* For domain decomposition we allocate dynamically
         * in dd_partition_system.
         */

        /* Initiate forcerecord */
        fr         = mk_forcerec();
        fr->hwinfo = hwinfo;
        init_forcerec(fplog, oenv, fr, fcd, inputrec, mtop, cr, box, FALSE,
                      opt2fn("-table", nfile, fnm),
                      opt2fn("-tabletf", nfile, fnm),
                      opt2fn("-tablep", nfile, fnm),
                      opt2fn("-tableb", nfile, fnm),
                      nbpu_opt,
                      FALSE, pforce);

        /* version for PCA_NOT_READ_NODE (see md.c) */
        /*init_forcerec(fplog,fr,fcd,inputrec,mtop,cr,box,FALSE,
           "nofile","nofile","nofile","nofile",FALSE,pforce);
         */
        fr->bSepDVDL = TRUE; // ((Flags & MD_SEPPOT) == MD_SEPPOT);


        /* Initialize the mdatoms structure.
         * mdatoms is not filled with atom data,
         * as this can not be done now with domain decomposition.
         */
        mdatoms = init_mdatoms(fplog, mtop, inputrec->efep != efepNO);


	/*** ENAS TODO SPEAK TO RIO ABOUT MASSIVE EFFECT OF CALC_SHIFTS */
        calc_shifts(box, fr->shift_vec);


        /* With periodic molecules the charge groups should be whole at start up
         * and the virtual sites should not be far from their proper positions.
         */
        /* Make molecules whole at start of run */
        do_pbc_first_mtop(fplog, inputrec->ePBC, box, mtop, state->x);
        if (EEL_PME(fr->eeltype)) // PME HERE
        { 
            pmedata    = &fr->pmedata;
        }
        else // EWALD HERE
        {
            pmedata = NULL;
        }

        /* Before setting affinity, check whether the affinity has changed
         * - which indicates that probably the OpenMP library has changed it
         * since we first checked).
         */
        gmx_check_thread_affinity_set(fplog, cr,
                                      hw_opt, hwinfo->nthreads_hw_avail, TRUE);

        /* Set the CPU affinity */
        gmx_set_thread_affinity(fplog, cr, hw_opt, nthreads_pme, hwinfo, inputrec);

    /* Initiate PME if necessary,
     * either on all nodes or on dedicated PME nodes only. */
    if (EEL_PME(inputrec->coulombtype)) // PME 1, EWALD 0
    {
        nChargePerturbed = mdatoms->nChargePerturbed;

        status = gmx_pme_init(pmedata, cr, npme_major, npme_minor, inputrec,
                                  mtop ? mtop->natoms : 0, nChargePerturbed,
                                  (Flags & MD_REPRODUCIBLE), nthreads_pme);
        if (status != 0)
        {
            gmx_fatal(FARGS, "Error %d initializing PME", status);
         }
    }

    /* Turn on signal handling on all nodes */
    /*
     * (A user signal from the PME nodes (if any)
     * is communicated to the PP nodes.
     */
    signal_handler_install();


    do_md(fplog, cr, nfile, fnm,
                                      oenv, bVerbose, bCompact,
                                      nstglobalcomm,
                                      NULL, NULL,
                                      nstepout, inputrec, mtop,
                                      fcd, state,
                                      mdatoms, nrnb, wcycle, ed, fr,
                                      repl_ex_nst, repl_ex_nex, repl_ex_seed,
                                      NULL,
                                      cpt_period, max_hours,
                                      deviceOptions,
                                      Flags,
                                      &runtime);



    runtime.proc = runtime.real;

    wallcycle_stop(wcycle, ewcRUN);

    /* Finish up, write some stuff
     * if rerunMD, don't write last frame again
     */
    finish_run(fplog, cr, ftp2fn(efSTO, nfile, fnm),
               inputrec, nrnb, wcycle, &runtime,
               fr != NULL && fr->nbv != NULL && fr->nbv->bUseGPU ?
               nbnxn_cuda_get_timings(fr->nbv->cu_nbv) : NULL,
               nthreads_pp,
               EI_DYNAMICS(inputrec->eI) && !MULTISIM(cr));

    /* Does what it says */
    print_date_and_time(fplog, cr->nodeid, "Finished mdrun", &runtime);


    rc = (int)gmx_get_stop_condition();


    return rc;
}
