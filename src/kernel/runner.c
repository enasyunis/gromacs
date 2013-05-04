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
    gmx_detect_hardware(fplog, hwinfo, cr, 0,0, hw_opt->gpu_id);

    minf.cutoff_scheme = inputrec->cutoff_scheme;
    minf.bUseGPU       = FALSE;
    hw_opt->nthreads_tmpi = 1; 


    copy_mat(state->box, box);


    /* PME, if used, is done on all nodes with 1D decomposition */
    cr->npmenodes = 0;
    cr->duty      = (DUTY_PP | DUTY_PME);
    npme_minor    = 1;
    npme_major = cr->nnodes;


    gmx_omp_nthreads_init(fplog, cr,
                          hwinfo->nthreads_hw_avail,
                          hw_opt->nthreads_omp,
                          hw_opt->nthreads_omp_pme,
                          0,
                          TRUE);


    // Both numbers equal to 12 - the number of OMP_NUM_THREADS
    nthreads_pp  = gmx_omp_nthreads_get(emntNonbonded);
    nthreads_pme = gmx_omp_nthreads_get(emntPME);



    /* Initiate forcerecord */
    snew(fr, 1);
    fr->hwinfo = hwinfo;
    init_forcerec(fplog, fr, inputrec, mtop, cr, box, FALSE,
                      opt2fn("-table", nfile, fnm),
                      opt2fn("-tabletf", nfile, fnm),
                      opt2fn("-tablep", nfile, fnm),
                      opt2fn("-tableb", nfile, fnm),
                      "cpu",
                      FALSE, -1);

     fr->bSepDVDL = TRUE; 


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
          state,
          mdatoms, ed, fr,
          deviceOptions,
          Flags
          );


    return 0;
}
