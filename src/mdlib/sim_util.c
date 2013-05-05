#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_CRAY_XT3
#include <catamount/dclock.h>
#endif


#include <stdio.h>
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#include <math.h>
#include "visibility.h"
#include "typedefs.h"
#include "string2.h"
#include "gmxfio.h"
#include "smalloc.h"
#include "names.h"
#include "txtdump.h"
#include "pbc.h"
#include "vec.h"
#include <time.h>
#include "nrnb.h"
#include "mdrun.h"
#include "physics.h"
#include "main.h"
#include "mdatoms.h"
#include "force.h"
#include "bondf.h"
#include "pme.h"
#include "orires.h"
#include "network.h"
#include "trnio.h"
#include "mpelogging.h"
#include "nbnxn_atomdata.h"
#include "nbnxn_search.h"
#include "nbnxn_kernels/nbnxn_kernel_ref.h"
#include "tmpi.h"
#include "types/simple.h"
#include "typedefs.h"
#include "qmmm.h"


static void do_nb_verlet(t_forcerec *fr,
                         interaction_const_t *ic,
                         gmx_enerdata_t *enerd,
                         int flags, 
                         int clearF
                         )
{
    nonbonded_verlet_group_t  *nbvg = &fr->nbv->grp[eintLocal];
    nbnxn_kernel_ref(&nbvg->nbl_lists,
                             nbvg->nbat, ic,
                             fr->shift_vec,
                             flags,
                             clearF,
                             fr->fshift[0],
                             enerd->grpp.ener[egCOULSR],
                             enerd->grpp.ener[egLJSR]);
}

void do_force(FILE *fplog, t_commrec *cr,
              t_inputrec *inputrec,
              gmx_large_int_t step, 
              gmx_localtop_t *top,
              gmx_mtop_t *mtop,
              gmx_groups_t *groups,
              matrix box, rvec x[], history_t *hist,
              rvec f[],
              t_mdatoms *mdatoms,
              gmx_enerdata_t *enerd, 
              real *lambda,
              t_forcerec *fr,
              double t,  
              int flags)
{
    int                 i, end, start, homenr;
    double              mu[2*DIM];
    rvec                vzero, box_diag;
    real                e, v;
    nonbonded_verlet_t *nbv;

    nbv            = fr->nbv;

    start  = mdatoms->start;
    homenr = mdatoms->homenr;
    end    = start+homenr;

    put_atoms_in_box_omp(fr->ePBC, box, homenr, x);




    nbnxn_atomdata_copy_shiftvec(0, fr->shift_vec, nbv->grp[0].nbat);


    clear_rvec(vzero);
    box_diag[XX] = box[XX][XX];
    box_diag[YY] = box[YY][YY];
    box_diag[ZZ] = box[ZZ][ZZ];

    nbnxn_put_on_grid(nbv->nbs, fr->ePBC, box,
                     0, vzero, box_diag,
                     0, mdatoms->homenr, -1, fr->cginfo, x,
                     0, NULL,
                     nbv->grp[eintLocal].kernel_type,
                     nbv->grp[eintLocal].nbat);

    nbnxn_atomdata_set(nbv->grp[eintLocal].nbat, eatAll,
                               nbv->nbs, mdatoms, fr->cginfo);


    /* do local pair search */
    nbnxn_make_pairlist(nbv->nbs, nbv->grp[eintLocal].nbat,
                       &top->excls,
                       fr->ic->rlist,
                       nbv->min_ci_balanced,
                       &nbv->grp[eintLocal].nbl_lists,
                       eintLocal,
                       nbv->grp[eintLocal].kernel_type
                       );



    /* Reset energies */
    reset_enerdata(&(inputrec->opts), fr, 1, enerd, MASTER(cr));
    clear_rvecs(SHIFTS, fr->fshift);


    /* Reset forces for which the virial is calculated separately:
     * PME/Ewald forces if necessary */
    fr->f_novirsum = fr->f_novirsum_alloc;
    GMX_BARRIER(cr->mpi_comm_mygroup);
    clear_rvecs(homenr, fr->f_novirsum+start);
    GMX_BARRIER(cr->mpi_comm_mygroup);

    /* Clear the short- and long-range forces */
    clear_rvecs(fr->natoms_force_constr, f);


    GMX_BARRIER(cr->mpi_comm_mygroup);



    /* Compute the bonded and non-bonded energies and optionally forces */
    do_force_lowlevel(fplog, step, fr, inputrec, &(top->idef),
                      cr, mdatoms, &(inputrec->opts),
                      x, hist, f, f, enerd,  mtop, top,
                      &(top->atomtypes),  box,
                      inputrec->fepvals, lambda, &(top->excls), 
                      flags);


    /* Maybe we should move this into do_force_lowlevel */
    do_nb_verlet(fr, fr->ic, enerd, flags, enbvClearFYes
                     );



   /* Add all the non-bonded force to the normal force array.
    * This can be split into a local a non-local part when overlapping
    * communication with calculation with domain decomposition.
    */
    nbnxn_atomdata_add_nbat_f_to_f(nbv->nbs, eatAll, nbv->grp[eintLocal].nbat, f);

    /* if there are multiple fshift output buffers reduce them */
    nbnxn_atomdata_add_nbat_fshift_to_fshift(nbv->grp[eintLocal].nbat,
                                                     fr->fshift);

    GMX_BARRIER(cr->mpi_comm_mygroup);

    // accomulating the forces.
    for (i = start; (i < end); i++)
    {
        rvec_inc(f[i], fr->f_novirsum[i]); // f will be incremented by fr->f_novirsum data
    }

    /* Sum the potential energy terms from group contributions */
    enerd->term[F_COUL_SR]  = enerd->grpp.ener[egCOULSR][0];
    enerd->term[F_EPOT]= enerd->term[F_COUL_SR] + enerd->term[F_COUL_RECIP];
    // NO NEED FOR A LOG FILE AT THIS POINT :)
    // correct answer - only works when numthreads = 12!!!
    // ** Potential Coul. SR -9.017649e+05, Recip 3.117281e+02, Tot -9.014532e+05
    printf("\n** Potential Coul. SR %e, Recip %e, Tot %e\n", enerd->term[F_COUL_SR], enerd->term[F_COUL_RECIP], enerd->term[F_EPOT]);
    fprintf(stderr,"\n** Potential Coul. SR %e, Recip %e, Tot %e\n", enerd->term[F_COUL_SR], enerd->term[F_COUL_RECIP], enerd->term[F_EPOT]);
}


