/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <math.h>
#include "macros.h"
#include "main.h"
#include "smalloc.h"
#include "futil.h"
#include "tgroup.h"
#include "vec.h"
#include "network.h"
#include "smalloc.h"
#include "mtop_util.h"
#include "gmx_omp_nthreads.h"

static void init_grptcstat(int ngtc, t_grp_tcstat tcstat[])
{ // called
    tcstat[0].T = 0;
    clear_mat(tcstat[0].ekinh);
    clear_mat(tcstat[0].ekinh_old);
    clear_mat(tcstat[0].ekinf);
}

static void init_grpstat(FILE *log,
                         gmx_mtop_t *mtop, int ngacc, t_grp_acc gstat[])
{// called
    gmx_groups_t           *groups;
    gmx_mtop_atomloop_all_t aloop;
    int                     i, grp;
    t_atom                 *atom;
    groups = &mtop->groups;
    aloop  = gmx_mtop_atomloop_all_init(mtop);
    while (gmx_mtop_atomloop_all_next(aloop, &i, &atom)) // 3000
    { 
        grp = ggrpnr(groups, egcACC, i);
        gstat[grp].nat++;
        /* This will not work for integrator BD */
        gstat[grp].mA += atom->m;
        gstat[grp].mB += atom->mB;
    }
}

void init_ekindata(FILE *log, gmx_mtop_t *mtop, t_grpopts *opts,
                   gmx_ekindata_t *ekind)
{// called
    int i;
    int nthread, thread;

    /* bNEMD tells if we should remove remove the COM velocity
     * from the velocities during velocity scaling in T-coupling.
     * Turn this on when we have multiple acceleration groups
     * or one accelerated group.
     */
    ekind->bNEMD = (opts->ngacc > 1 || norm(opts->acc[0]) > 0); // false

    ekind->ngtc = opts->ngtc;
    snew(ekind->tcstat, opts->ngtc);
    init_grptcstat(opts->ngtc, ekind->tcstat);
    /* Set Berendsen tcoupl lambda's to 1,
     * so runs without Berendsen coupling are not affected.
     */
   ekind->tcstat[0].lambda         = 1.0;
   ekind->tcstat[0].vscale_nhc     = 1.0;
   ekind->tcstat[0].ekinscaleh_nhc = 1.0;
   ekind->tcstat[0].ekinscalef_nhc = 1.0;

    nthread = gmx_omp_nthreads_get(emntUpdate);

    snew(ekind->ekin_work_alloc, nthread);
    snew(ekind->ekin_work, nthread);
#pragma omp parallel for num_threads(nthread) schedule(static)
    for (thread = 0; thread < nthread; thread++)
    {
        /* Allocate 2 elements extra on both sides,
         * so in single precision we have 2*3*3*4=72 bytes buffer
         * on both sides to avoid cache pollution.
         */
        snew(ekind->ekin_work_alloc[thread], ekind->ngtc+4);
        ekind->ekin_work[thread] = ekind->ekin_work_alloc[thread] + 2;
    }

    ekind->ngacc = opts->ngacc;
    snew(ekind->grpstat, opts->ngacc);
    init_grpstat(log, mtop, opts->ngacc, ekind->grpstat);
}

real sum_ekin(t_grpopts *opts, gmx_ekindata_t *ekind, real *dekindlambda,
              gmx_bool bEkinAveVel, gmx_bool bSaveEkinOld, gmx_bool bScaleEkin)
{// called
    int           i, j, m, ngtc;
    real          T, ek;
    t_grp_tcstat *tcstat;
    real          nrdf, nd, *ndf;

    ngtc = opts->ngtc;
    ndf  = opts->nrdf;

    T    = 0;
    nrdf = 0;

    clear_mat(ekind->ekin);
    i = 0;

    nd     = ndf[i];
    tcstat = &ekind->tcstat[i];
    /* Sometimes a group does not have degrees of freedom, e.g.
     * when it consists of shells and virtual sites, then we just
     * set the temperatue to 0 and also neglect the kinetic
     * energy, which should be  zero anyway.
     */

     /* Calculate the full step Ekin as the average of the half steps */
     for (j = 0; (j < DIM); j++)
     {
        for (m = 0; (m < DIM); m++)
        {
             tcstat->ekinf[j][m] =
                 0.5*(tcstat->ekinh[j][m]*tcstat->ekinscaleh_nhc + tcstat->ekinh_old[j][m]);
        }
     }
     m_add(tcstat->ekinf, ekind->ekin, ekind->ekin);

     tcstat->Th = calc_temp(trace(tcstat->ekinh), nd);
     tcstat->T  = calc_temp(trace(tcstat->ekinf), nd);

    /* after the scaling factors have been multiplied in, we can remove them */
    tcstat->ekinscaleh_nhc = 1.0;
    T    += nd*tcstat->T;
    nrdf += nd;
    T /= nrdf;
    *dekindlambda = 0.5*(ekind->dekindl + ekind->dekindl_old);
    return T;
}
