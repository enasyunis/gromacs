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
#include "coulomb.h"
#include "mvdata.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "tpxio.h"
#include "txtdump.h"
#include "gmx_omp_nthreads.h"
#include "pull_rotation.h"
#include "../mdlib/nbnxn_search.h"
#include "../mdlib/nbnxn_consts.h"
#include "gmx_fatal_collective.h"
#include "types/membedt.h"
#include "gmx_omp.h"

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
             unsigned long Flags)
{

    t_inputrec     *inputrec;
    t_state        *state = NULL;
    matrix          box;
    gmx_mtop_t     *mtop       = NULL;
    t_mdatoms      *mdatoms    = NULL;
    t_forcerec     *fr         = NULL;
    gmx_pme_t      *pmedata    = NULL;
    master_inf_t    minf         = {-1, FALSE};

    /* CAUTION: threads may be started later on in this function, so
       cr doesn't reflect the final parallel state right now */
    snew(inputrec, 1);
    snew(mtop, 1);

    snew(state, 1);

    /* Read (nearly) all data required for the simulation */
    read_tpx_state(ftp2fn(efTPX, nfile, fnm), inputrec, state, NULL, mtop);


    minf.cutoff_scheme = inputrec->cutoff_scheme;
    minf.bUseGPU       = FALSE;
    hw_opt->nthreads_tmpi = 1; 


    copy_mat(state->box, box);


    /* PME, if used, is done on all nodes with 1D decomposition */
    cr->npmenodes = 0;
    cr->duty      = (DUTY_PP | DUTY_PME);


    gmx_omp_nthreads_init(fplog, cr,
                          hw_opt->nthreads_omp,
                          hw_opt->nthreads_omp_pme,
                          0,
                          TRUE);



    /* Initiate forcerecord */
    snew(fr, 1);
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
    //mdatoms = init_mdatoms(fplog, mtop);
    snew(mdatoms, 1);

    mdatoms->nenergrp = mtop->groups.grps[egcENER].nr;
    mdatoms->bVCMgrps = FALSE;
    mdatoms->tmassA = 3024.0; // SUM over all t_atom->m
    mdatoms->tmassB = 3024.0; // SUM over all t_atom->mB
    mdatoms->bOrires = 0;



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
        // cr->nnodes=1, natoms=3000,gmx_omp_nthreads_get(emntPME)=12
        gmx_pme_init(pmedata, cr,  cr->nnodes, 1, inputrec,
                                  mtop->natoms, 0,
                                  0, gmx_omp_nthreads_get(emntPME));
    }



    do_md(fplog, cr, nfile, fnm,
          inputrec, mtop,
          state,
          mdatoms, fr,
          Flags
          );


    return 0;
}
