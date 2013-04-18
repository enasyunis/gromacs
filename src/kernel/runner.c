/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
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
#include "repl_ex.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "partdec.h"
#include "coulomb.h"
#include "constr.h"
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
#include "membed.h"
#include "gmx_omp.h"
#include "gmx_thread_affinity.h"

#include "tmpi.h"


#include "gpu_utils.h"
#include "nbnxn_cuda_data_mgmt.h"

typedef struct {
    gmx_integrator_t *func;
} gmx_intp_t;

/* The array should match the eI array in include/types/enums.h */
const gmx_intp_t    integrator[eiNR] = { {do_md}, {do_steep}, {do_cg}, {do_md}, {do_md}, {do_nm}, {do_lbfgs}, {do_tpi}, {do_tpi}, {do_md}, {do_md}, {do_md}};

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
    if (ngpu > 0)
    {
        nthreads_tmpi = ngpu;
        if (nthreads_tot > 0 && nthreads_tot < nthreads_tmpi)
        {
            nthreads_tmpi = nthreads_tot;
        }
    }
    else if (hw_opt->nthreads_omp > 0)
    {
        /* Here we could oversubscribe, when we do, we issue a warning later */
        nthreads_tmpi = max(1, nthreads_tot/hw_opt->nthreads_omp);
    }
    else
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
            /* Use pure OpenMP parallelization */
            nthreads_tmpi = 1;
        }
        else
        {
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
printf("**** inside of get_nthreads_mpi ****\n");
    int      nthreads_hw, nthreads_tot_max, nthreads_tmpi, nthreads_new, ngpu;
    int      min_atoms_per_mpi_thread;
    char    *env;
    char     sbuf[STRLEN];
    gmx_bool bCanUseGPU;

    if (hw_opt->nthreads_tmpi > 0)
    {
        /* Trivial, return right away */
        return hw_opt->nthreads_tmpi;
    }

    nthreads_hw = hwinfo->nthreads_hw_avail;

    /* How many total (#tMPI*#OpenMP) threads can we start? */
    if (hw_opt->nthreads_tot > 0)
    {
        nthreads_tot_max = hw_opt->nthreads_tot;
    }
    else
    {
        nthreads_tot_max = nthreads_hw;
    }

    bCanUseGPU = (hwinfo->bCanUseGPU);
    if (bCanUseGPU)
    {
        ngpu = hwinfo->gpu_info.ncuda_dev_use;
    }
    else
    {
        ngpu = 0;
    }

    nthreads_tmpi =
        get_tmpi_omp_thread_division(hwinfo, hw_opt, nthreads_tot_max, ngpu);

    if (inputrec->eI == eiNM || EI_TPI(inputrec->eI))
    {
        /* Steps are divided over the nodes iso splitting the atoms */
        min_atoms_per_mpi_thread = 0;
    }
    else
    {
        if (bCanUseGPU)
        {
            min_atoms_per_mpi_thread = MIN_ATOMS_PER_GPU;
        }
        else
        {
            min_atoms_per_mpi_thread = MIN_ATOMS_PER_MPI_THREAD;
        }
    }

    /* Check if an algorithm does not support parallel simulation.  */
    if (nthreads_tmpi != 1 &&
        ( inputrec->eI == eiLBFGS ||
          inputrec->coulombtype == eelEWALD ) )
    {
        nthreads_tmpi = 1;

        md_print_warn(cr, fplog, "The integration or electrostatics algorithm doesn't support parallel runs. Using a single thread-MPI thread.\n");
        if (hw_opt->nthreads_tmpi > nthreads_tmpi)
        {
            gmx_fatal(FARGS, "You asked for more than 1 thread-MPI thread, but an algorithm doesn't support that");
        }
    }
    else if (mtop->natoms/nthreads_tmpi < min_atoms_per_mpi_thread)
    {
        /* the thread number was chosen automatically, but there are too many
           threads (too few atoms per thread) */
        nthreads_new = max(1, mtop->natoms/min_atoms_per_mpi_thread);

        /* Avoid partial use of Hyper-Threading */
        if (gmx_cpuid_x86_smt(hwinfo->cpuid_info) == GMX_CPUID_X86_SMT_ENABLED &&
            nthreads_new > nthreads_hw/2 && nthreads_new < nthreads_hw)
        {
            nthreads_new = nthreads_hw/2;
        }

        /* Avoid large prime numbers in the thread count */
        if (nthreads_new >= 6)
        {
            /* Use only 6,8,10 with additional factors of 2 */
            int fac;

            fac = 2;
            while (3*fac*2 <= nthreads_new)
            {
                fac *= 2;
            }

            nthreads_new = (nthreads_new/fac)*fac;
        }
        else
        {
            /* Avoid 5 */
            if (nthreads_new == 5)
            {
                nthreads_new = 4;
            }
        }

        nthreads_tmpi = nthreads_new;

        fprintf(stderr, "\n");
        fprintf(stderr, "NOTE: Parallelization is limited by the small number of atoms,\n");
        fprintf(stderr, "      only starting %d thread-MPI threads.\n", nthreads_tmpi);
        fprintf(stderr, "      You can use the -nt and/or -ntmpi option to optimize the number of threads.\n\n");
    }

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
    *bUseGPU = hwinfo->bCanUseGPU || (getenv("GMX_EMULATE_GPU") != NULL);

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
        if (rlist_new != ir->rlist)
        {
            if (fplog != NULL)
            {
                fprintf(fplog, "\nChanging rlist from %g to %g for non-bonded %dx%d atom kernels\n\n",
                        ir->rlist, rlist_new,
                        ls.cluster_size_i, ls.cluster_size_j);
            }
            ir->rlist     = rlist_new;
            ir->rlistlong = rlist_new;
        }
}


static void check_and_update_hw_opt(gmx_hw_opt_t *hw_opt,
                                    int           cutoff_scheme,
                                    gmx_bool      bIsSimMaster)
{
    gmx_omp_nthreads_read_env(&hw_opt->nthreads_omp, bIsSimMaster);


    if (hw_opt->nthreads_tot > 0 && hw_opt->nthreads_omp_pme <= 0)
    {
        /* We have the same number of OpenMP threads for PP and PME processes,
         * thus we can perform several consistency checks.
         */
        if (hw_opt->nthreads_tmpi > 0 &&
            hw_opt->nthreads_omp > 0 &&
            hw_opt->nthreads_tot != hw_opt->nthreads_tmpi*hw_opt->nthreads_omp)
        {
            gmx_fatal(FARGS, "The total number of threads requested (%d) does not match the thread-MPI threads (%d) times the OpenMP threads (%d) requested",
                      hw_opt->nthreads_tot, hw_opt->nthreads_tmpi, hw_opt->nthreads_omp);
        }

        if (hw_opt->nthreads_tmpi > 0 &&
            hw_opt->nthreads_tot % hw_opt->nthreads_tmpi != 0)
        {
            gmx_fatal(FARGS, "The total number of threads requested (%d) is not divisible by the number of thread-MPI threads requested (%d)",
                      hw_opt->nthreads_tot, hw_opt->nthreads_tmpi);
        }

        if (hw_opt->nthreads_omp > 0 &&
            hw_opt->nthreads_tot % hw_opt->nthreads_omp != 0)
        {
            gmx_fatal(FARGS, "The total number of threads requested (%d) is not divisible by the number of OpenMP threads requested (%d)",
                      hw_opt->nthreads_tot, hw_opt->nthreads_omp);
        }

        if (hw_opt->nthreads_tmpi > 0 &&
            hw_opt->nthreads_omp <= 0)
        {
            hw_opt->nthreads_omp = hw_opt->nthreads_tot/hw_opt->nthreads_tmpi;
        }
    }


    if (hw_opt->nthreads_omp_pme > 0 && hw_opt->nthreads_omp <= 0)
    {
        gmx_fatal(FARGS, "You need to specify -ntomp in addition to -ntomp_pme");
    }

    if (hw_opt->nthreads_tot == 1)
    {
        hw_opt->nthreads_tmpi = 1;

        if (hw_opt->nthreads_omp > 1)
        {
            gmx_fatal(FARGS, "You requested %d OpenMP threads with %d total threads",
                      hw_opt->nthreads_tmpi, hw_opt->nthreads_tot);
        }
        hw_opt->nthreads_omp = 1;
    }

    if (hw_opt->nthreads_omp_pme <= 0 && hw_opt->nthreads_omp > 0)
    {
        hw_opt->nthreads_omp_pme = hw_opt->nthreads_omp;
    }

    if (debug)
    {
        fprintf(debug, "hw_opt: nt %d ntmpi %d ntomp %d ntomp_pme %d gpu_id '%s'\n",
                hw_opt->nthreads_tot,
                hw_opt->nthreads_tmpi,
                hw_opt->nthreads_omp,
                hw_opt->nthreads_omp_pme,
                hw_opt->gpu_id != NULL ? hw_opt->gpu_id : "");

    }
}


/* Override the value in inputrec with value passed on the command line (if any) */
static void override_nsteps_cmdline(FILE            *fplog,
                                    int              nsteps_cmdline,
                                    t_inputrec      *ir,
                                    const t_commrec *cr)
{
    assert(ir);
    assert(cr);

    /* override with anything else than the default -2 */
    if (nsteps_cmdline > -2)
    {
        char stmp[STRLEN];

        ir->nsteps = nsteps_cmdline;
        if (EI_DYNAMICS(ir->eI))
        {
            sprintf(stmp, "Overriding nsteps with value passed on the command line: %d steps, %.3f ps",
                    nsteps_cmdline, nsteps_cmdline*ir->delta_t);
        }
        else
        {
            sprintf(stmp, "Overriding nsteps with value passed on the command line: %d steps",
                    nsteps_cmdline);
        }

        md_print_warn(cr, fplog, "%s\n", stmp);
    }
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
printf("****** inside of mdrunner *******\n");

    gmx_bool        bForceUseGPU, bTryUseGPU;
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
    real            ewaldcoeff = 0;
    gmx_pme_t      *pmedata    = NULL;
    gmx_vsite_t    *vsite      = NULL;
    gmx_constr_t    constr;
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


    bForceUseGPU = (strncmp(nbpu_opt, "gpu", 3) == 0);
    bTryUseGPU   = (strncmp(nbpu_opt, "auto", 4) == 0) || bForceUseGPU;

    snew(state, 1);
    /* Read (nearly) all data required for the simulation */
    read_tpx_state(ftp2fn(efTPX, nfile, fnm), inputrec, state, NULL, mtop);

    /* Detect hardware, gather information. With tMPI only thread 0 does it
     * and after threads are started broadcasts hwinfo around. */
    snew(hwinfo, 1);
    gmx_detect_hardware(fplog, hwinfo, cr,
                            bForceUseGPU, bTryUseGPU, hw_opt->gpu_id);

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

        check_and_update_hw_opt(hw_opt, minf.cutoff_scheme, 1);

        /* Early check for externally set process affinity. Can't do over all
         * MPI processes because hwinfo is not available everywhere, but with
         * thread-MPI it's needed as pinning might get turned off which needs
         * to be known before starting thread-MPI. */
        gmx_check_thread_affinity_set(fplog,
                                      NULL,
                                      hw_opt, hwinfo->nthreads_hw_avail, FALSE);

        if (cr->npmenodes > 0 && hw_opt->nthreads_tmpi <= 0)
        {
            gmx_fatal(FARGS, "You need to explicitly specify the number of MPI threads (-ntmpi) when using separate PME nodes");
        }

        if (hw_opt->nthreads_omp_pme != hw_opt->nthreads_omp &&
            cr->npmenodes <= 0)
        {
            gmx_fatal(FARGS, "You need to explicitly specify the number of PME nodes (-npme) when using different number of OpenMP threads for PP and PME nodes");
        }

        /* NOW the threads will be started: */
        hw_opt->nthreads_tmpi = get_nthreads_mpi(hwinfo,
                                                 hw_opt,
                                                 inputrec, mtop,
                                                 cr, fplog);
        if (hw_opt->nthreads_tot > 0 && hw_opt->nthreads_omp <= 0)
        {
            hw_opt->nthreads_omp = hw_opt->nthreads_tot/hw_opt->nthreads_tmpi;
        }

printf("***** (hw_opt->nthreads_tmpi > 1) %d ****\n", (hw_opt->nthreads_tmpi > 1)); 
    /* END OF CAUTION: cr is now reliable */


    if (fplog != NULL)
    {
        pr_inputrec(fplog, 0, "Input Parameters", inputrec, FALSE);
    }

    /* With tMPI we detected on thread 0 and we'll just pass the hwinfo pointer
     * to the other threads  -- slightly uncool, but works fine, just need to
     * make sure that the data doesn't get freed twice. */
    if (cr->nnodes > 1)
    {
        gmx_bcast(sizeof(&hwinfo), &hwinfo, cr);
    }

    /* now make sure the state is initialized and propagated */
    set_state_entries(state, inputrec, cr->nnodes);

    /* A parallel command line option consistency check that we can
       only do after any threads have started. */
    if (
        (ddxyz[XX] > 1 || ddxyz[YY] > 1 || ddxyz[ZZ] > 1 || cr->npmenodes > 0))
    {
        gmx_fatal(FARGS,
                  "The -dd or -npme option request a parallel simulation, "
                  "but the number of threads (option -nt) is 1"
                  , ShortProgram()
                  );
    }

    if ((Flags & MD_RERUN) &&
        (EI_ENERGY_MINIMIZATION(inputrec->eI) || eiNM == inputrec->eI))
    {
        gmx_fatal(FARGS, "The .mdp file specified an energy mininization or normal mode algorithm, and these are not compatible with mdrun -rerun");
    }


    if (!EEL_PME(inputrec->coulombtype) || (Flags & MD_PARTDEC))
    {
        if (cr->npmenodes > 0)
        {
            if (!EEL_PME(inputrec->coulombtype))
            {
                gmx_fatal_collective(FARGS, cr, NULL,
                                     "PME nodes are requested, but the system does not use PME electrostatics");
            }
            if (Flags & MD_PARTDEC)
            {
                gmx_fatal_collective(FARGS, cr, NULL,
                                     "PME nodes are requested, but particle decomposition does not support separate PME nodes");
            }
        }

        cr->npmenodes = 0;
    }


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

    if (gmx_mtop_ftype_count(mtop, F_ORIRES) > 0)
    {
        /* Orientation restraints */
        if (MASTER(cr))
        {
            init_orires(fplog, mtop, state->x, inputrec, cr->ms, &(fcd->orires),
                        state);
        }
    }

    if (DEFORM(*inputrec))
    {
        /* Store the deform reference box before reading the checkpoint */
            copy_mat(state->box, box);
        /* Because we do not have the update struct available yet
         * in which the reference values should be stored,
         * we store them temporarily in static variables.
         * This should be thread safe, since they are only written once
         * and with identical values.
         */
        tMPI_Thread_mutex_lock(&deform_init_box_mutex);
        deform_init_init_step_tpx = inputrec->init_step;
        copy_mat(box, deform_init_box_tpx);
        tMPI_Thread_mutex_unlock(&deform_init_box_mutex);
    }

    if (opt2bSet("-cpi", nfile, fnm))
    {
        /* Check if checkpoint file exists before doing continuation.
         * This way we can use identical input options for the first and subsequent runs...
         */
        if (gmx_fexist_master(opt2fn_master("-cpi", nfile, fnm, cr), cr) )
        {
            load_checkpoint(opt2fn_master("-cpi", nfile, fnm, cr), &fplog,
                            cr, Flags & MD_PARTDEC, ddxyz,
                            inputrec, state, &bReadRNG, &bReadEkin,
                            0,
                            0);

            if (bReadRNG)
            {
                Flags |= MD_READ_RNG;
            }
            if (bReadEkin)
            {
                Flags |= MD_READ_EKIN;
            }
        }
    }


    /* override nsteps with value from cmdline */
    override_nsteps_cmdline(fplog, nsteps_cmdline, inputrec, cr);


        copy_mat(state->box, box);


    /* Essential dynamics */
    if (opt2bSet("-ei", nfile, fnm))
    {
        /* Open input and output files, allocate space for ED data structure */
        ed = ed_open(mtop->natoms, &state->edsamstate, nfile, fnm, Flags, oenv, cr);
    }

    /* PME, if used, is done on all nodes with 1D decomposition */
    cr->npmenodes = 0;
    cr->duty      = (DUTY_PP | DUTY_PME);
    npme_major    = 1;
    npme_minor    = 1;
    if (!EI_TPI(inputrec->eI))
    {
         npme_major = cr->nnodes;
    }

    if (inputrec->ePBC == epbcSCREW)
    {
       gmx_fatal(FARGS,
                      "pbc=%s is only implemented with domain decomposition",
                      epbc_names[inputrec->ePBC]);
    }


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
                          (cr->duty & DUTY_PP) == 0,
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
    if (cr->duty & DUTY_PP)
    {
        /* For domain decomposition we allocate dynamically
         * in dd_partition_system.
         */
        if (DOMAINDECOMP(cr))
        {
            bcast_state_setup(cr, state);
        }

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
        fr->bSepDVDL = ((Flags & MD_SEPPOT) == MD_SEPPOT);

        /* Initialize QM-MM */
        if (fr->bQMMM)
        {
            init_QMMMrec(cr, box, mtop, inputrec, fr);
        }

        /* Initialize the mdatoms structure.
         * mdatoms is not filled with atom data,
         * as this can not be done now with domain decomposition.
         */
        mdatoms = init_mdatoms(fplog, mtop, inputrec->efep != efepNO);

        /* Initialize the virtual site communication */
        vsite = init_vsite(mtop, cr, FALSE);

        calc_shifts(box, fr->shift_vec);

        /* With periodic molecules the charge groups should be whole at start up
         * and the virtual sites should not be far from their proper positions.
         */
        if (!inputrec->bContinuation && MASTER(cr) &&
            !(inputrec->ePBC != epbcNONE && inputrec->bPeriodicMols))
        {
            /* Make molecules whole at start of run */
            if (fr->ePBC != epbcNONE)
            {
                do_pbc_first_mtop(fplog, inputrec->ePBC, box, mtop, state->x);
            }
            if (vsite)
            {
                /* Correct initial vsite positions are required
                 * for the initial distribution in the domain decomposition
                 * and for the initial shell prediction.
                 */
                construct_vsites_mtop(fplog, vsite, mtop, state->x);
            }
        }

        if (EEL_PME(fr->eeltype))
        {
            ewaldcoeff = fr->ewaldcoeff;
            pmedata    = &fr->pmedata;
        }
        else
        {
            pmedata = NULL;
        }
    }
    else
    {
        /* This is a PME only node */

        /* We don't need the state */
        done_state(state);

        ewaldcoeff = calc_ewaldcoeff(inputrec->rcoulomb, inputrec->ewald_rtol);
        snew(pmedata, 1);
    }

    if (hw_opt->thread_affinity != threadaffOFF)
    {
        /* Before setting affinity, check whether the affinity has changed
         * - which indicates that probably the OpenMP library has changed it
         * since we first checked).
         */
        gmx_check_thread_affinity_set(fplog, cr,
                                      hw_opt, hwinfo->nthreads_hw_avail, TRUE);

        /* Set the CPU affinity */
        gmx_set_thread_affinity(fplog, cr, hw_opt, nthreads_pme, hwinfo, inputrec);
    }

    /* Initiate PME if necessary,
     * either on all nodes or on dedicated PME nodes only. */
    if (EEL_PME(inputrec->coulombtype))
    {
        if (mdatoms)
        {
            nChargePerturbed = mdatoms->nChargePerturbed;
        }
        if (cr->npmenodes > 0)
        {
            /* The PME only nodes need to know nChargePerturbed */
            gmx_bcast_sim(sizeof(nChargePerturbed), &nChargePerturbed, cr);
        }

        if (cr->duty & DUTY_PME)
        {
            status = gmx_pme_init(pmedata, cr, npme_major, npme_minor, inputrec,
                                  mtop ? mtop->natoms : 0, nChargePerturbed,
                                  (Flags & MD_REPRODUCIBLE), nthreads_pme);
            if (status != 0)
            {
                gmx_fatal(FARGS, "Error %d initializing PME", status);
            }
        }
    }


    if (integrator[inputrec->eI].func == do_md)
    {
        /* Turn on signal handling on all nodes */
        /*
         * (A user signal from the PME nodes (if any)
         * is communicated to the PP nodes.
         */
        signal_handler_install();
    }

    if (cr->duty & DUTY_PP)
    {
        if (inputrec->ePull != epullNO)
        {
            /* Initialize pull code */
            init_pull(fplog, inputrec, nfile, fnm, mtop, cr, oenv, inputrec->fepvals->init_lambda,
                      EI_DYNAMICS(inputrec->eI) && MASTER(cr), Flags);
        }

        if (inputrec->bRot)
        {
            /* Initialize enforced rotation code */
            init_rot(fplog, inputrec, nfile, fnm, cr, state->x, box, mtop, oenv,
                     bVerbose, Flags);
        }

        constr = init_constraints(fplog, mtop, inputrec, ed, state, cr);

        if (DOMAINDECOMP(cr))
        {
            dd_init_bondeds(fplog, cr->dd, mtop, vsite, constr, inputrec,
                            Flags & MD_DDBONDCHECK, fr->cginfo_mb);

            set_dd_parameters(fplog, cr->dd, dlb_scale, inputrec, fr, &ddbox);

            setup_dd_grid(fplog, cr->dd);
        }

        /* Now do whatever the user wants us to do (how flexible...) */
        integrator[inputrec->eI].func(fplog, cr, nfile, fnm,
                                      oenv, bVerbose, bCompact,
                                      nstglobalcomm,
                                      vsite, constr,
                                      nstepout, inputrec, mtop,
                                      fcd, state,
                                      mdatoms, nrnb, wcycle, ed, fr,
                                      repl_ex_nst, repl_ex_nex, repl_ex_seed,
                                      NULL,
                                      cpt_period, max_hours,
                                      deviceOptions,
                                      Flags,
                                      &runtime);

        if (inputrec->ePull != epullNO)
        {
            finish_pull(fplog, inputrec->pull);
        }

        if (inputrec->bRot)
        {
            finish_rot(fplog, inputrec->rot);
        }

    }
    else
    {
        /* do PME only */
        gmx_pmeonly(*pmedata, cr, nrnb, wcycle, ewaldcoeff, FALSE, inputrec);
    }

    if (EI_DYNAMICS(inputrec->eI) || EI_TPI(inputrec->eI))
    {
        /* Some timing stats */
            if (runtime.proc == 0)
            {
                runtime.proc = runtime.real;
            }
    }

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

    if ((cr->duty & DUTY_PP) && fr->nbv != NULL && fr->nbv->bUseGPU)
    {
        char gpu_err_str[STRLEN];

        /* free GPU memory and uninitialize GPU (by destroying the context) */
        nbnxn_cuda_free(fplog, fr->nbv->cu_nbv);

        if (!free_gpu(gpu_err_str))
        {
            gmx_warning("On node %d failed to free GPU #%d: %s",
                        cr->nodeid, get_current_gpu_device_id(), gpu_err_str);
        }
    }



    /* Does what it says */
    print_date_and_time(fplog, cr->nodeid, "Finished mdrun", &runtime);


    rc = (int)gmx_get_stop_condition();


    return rc;
}
