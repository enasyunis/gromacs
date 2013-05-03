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
#include "../mdlib/nbnxn_search.h"
#include "../mdlib/nbnxn_consts.h"
#include "gmx_fatal_collective.h"
#include "types/membedt.h"
#include "gmx_omp.h"
#include "gmx_thread_affinity.h"

#include "tmpi.h"


/* The array should match the eI array in include/types/enums.h */
gmx_large_int_t     deform_init_init_step_tpx;
matrix              deform_init_box_tpx;
tMPI_Thread_mutex_t deform_init_box_mutex = TMPI_THREAD_MUTEX_INITIALIZER;



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



/* Data structure set by SIMMASTER which needs to be passed to all nodes
 * before the other nodes have read the tpx file and called gmx_detect_hardware.
 */
typedef struct {
    int      cutoff_scheme; /* The cutoff scheme from inputrec_t */
    gmx_bool bUseGPU;       /* Use GPU or GPU emulation          */
} master_inf_t;

int mdrunner(gmx_hw_opt_t *hw_opt,
             FILE *fplog, t_commrec *cr, int nfile,
             const t_filenm fnm[],  
             const char *deviceOptions, unsigned long Flags)
{

    double          nodetime = 0, realtime;
    t_inputrec     *inputrec;
    t_state        *state = NULL;
    matrix          box;
    gmx_ddbox_t     ddbox = {0};
    int             npme_major, npme_minor;
    real            tmpr1, tmpr2;
    gmx_mtop_t     *mtop       = NULL;
    t_mdatoms      *mdatoms    = NULL;
    t_forcerec     *fr         = NULL;
    t_fcdata       *fcd        = NULL;
    gmx_pme_t      *pmedata    = NULL;
    int             i, m, nChargePerturbed = -1, status, nalloc;
    char           *gro;
    gmx_bool        bReadRNG, bReadEkin;
    int             list;
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

    //prepare_verlet_scheme(fplog, hwinfo, cr, hw_opt, "cpu",
    //                              inputrec, mtop, state->box,
   //                               &minf.bUseGPU);

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
    init_disres(fplog, mtop, inputrec, cr, Flags & MD_PARTDEC, fcd, state, FALSE);


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




        /* Initiate forcerecord */
        fr         = mk_forcerec();
        fr->hwinfo = hwinfo;
        init_forcerec(fplog, fr, fcd, inputrec, mtop, cr, box, FALSE,
                      opt2fn("-table", nfile, fnm),
                      opt2fn("-tabletf", nfile, fnm),
                      opt2fn("-tablep", nfile, fnm),
                      opt2fn("-tableb", nfile, fnm),
                      "cpu",
                      FALSE, -1);

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



    do_md(fplog, cr, nfile, fnm,
          inputrec, mtop,
          fcd, state,
          mdatoms, ed, fr,
          deviceOptions,
          Flags
          );





    /* Does what it says */

    return 0;
}
