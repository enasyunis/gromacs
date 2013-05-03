#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <math.h>

#include "types/commrec.h"
#include "sysstuff.h"
#include "smalloc.h"
#include "typedefs.h"
#include "nrnb.h"
#include "physics.h"
#include "macros.h"
#include "vec.h"
#include "main.h"
#include "confio.h"
#include "update.h"
#include "gmx_random.h"
#include "futil.h"
#include "mshift.h"
#include "tgroup.h"
#include "force.h"
#include "names.h"
#include "txtdump.h"
#include "mdrun.h"
#include "copyrite.h"
#include "constr.h"
#include "edsam.h"
#include "pull.h"
#include "disre.h"
#include "orires.h"
#include "gmx_wallcycle.h"
#include "gmx_omp_nthreads.h"
#include "gmx_omp.h"

typedef struct {
    double gdt;
    double eph;
    double emh;
    double em;
    double b;
    double c;
    double d;
} gmx_sd_const_t;

typedef struct {
    real V;
    real X;
    real Yv;
    real Yx;
} gmx_sd_sigma_t;

typedef struct {
    /* The random state for ngaussrand threads.
     * Normal thermostats need just 1 random number generator,
     * but SD and BD with OpenMP parallelization need 1 for each thread.
     */
    int             ngaussrand;
    gmx_rng_t      *gaussrand;
    /* BD stuff */
    real           *bd_rf;
    /* SD stuff */
    gmx_sd_const_t *sdc;
    gmx_sd_sigma_t *sdsig;
    rvec           *sd_V;
    int             sd_V_nalloc;
    /* andersen temperature control stuff */
    gmx_bool       *randomize_group;
    real           *boltzfac;
} gmx_stochd_t;

typedef struct gmx_update
{
    gmx_stochd_t *sd;
    /* xprime for constraint algorithms */
    rvec         *xp;
    int           xp_nalloc;

    /* variable size arrays for andersen */
    gmx_bool *randatom;
    int      *randatom_list;
    gmx_bool  randatom_list_init;

    /* Variables for the deform algorithm */
    gmx_large_int_t deformref_step;
    matrix          deformref_box;
} t_gmx_update;



static gmx_stochd_t *init_stochd(FILE *fplog, t_inputrec *ir, int nthreads)
{// called
    gmx_stochd_t   *sd;
    gmx_sd_const_t *sdc;
    int             ngtc, n, th;
    real            y;

    snew(sd, 1);
    sd->ngaussrand = 1;
    snew(sd->gaussrand, sd->ngaussrand);

    /* Initialize the first random generator */
    sd->gaussrand[0] = gmx_rng_init(ir->ld_seed);

    return sd;
}


gmx_update_t init_update(FILE *fplog, t_inputrec *ir)
{ // called
    t_gmx_update *upd;

    snew(upd, 1);

    upd->sd = init_stochd(fplog, ir, gmx_omp_nthreads_get(emntUpdate));

    upd->xp                 = NULL;
    upd->xp_nalloc          = 0;
    upd->randatom           = NULL;
    upd->randatom_list      = NULL;
    upd->randatom_list_init = FALSE; /* we have not yet cleared the data structure at this point */

    return upd;
}


static void calc_ke_part_normal(rvec v[], t_grpopts *opts, t_mdatoms *md,
                                gmx_ekindata_t *ekind, t_nrnb *nrnb, gmx_bool bEkinAveVel,
                                gmx_bool bSaveEkinOld)
{ // called 
    t_grp_tcstat *tcstat  = ekind->tcstat;
    t_grp_acc    *grpstat = ekind->grpstat;
    int           nthread, thread;

    /* three main: VV with AveVel, vv with AveEkin, leap with AveEkin.  Leap with AveVel is also
       an option, but not supported now.  Additionally, if we are doing iterations.
       bEkinAveVel: If TRUE, we sum into ekin, if FALSE, into ekinh.
       bSavEkinOld: If TRUE (in the case of iteration = bIterate is TRUE), we don't copy over the ekinh_old.
       If FALSE, we overrwrite it.
     */

    /* group velocities are calculated in update_ekindata and
     * accumulated in acumulate_groups.
     * Now the partial global and groups ekin.
     */
    copy_mat(tcstat[0].ekinh, tcstat[0].ekinh_old);
    clear_mat(tcstat[0].ekinh);
    ekind->dekindl_old = ekind->dekindl;

    nthread = gmx_omp_nthreads_get(emntUpdate); // 12

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (thread = 0; thread < nthread; thread++)
    {
        int     start_t, end_t, n;
        int     ga, gt;
        rvec    v_corrt;
        real    hm;
        int     d, m;
        matrix *ekin_sum;
        real   *dekindl_sum;

        start_t = md->start + ((thread+0)*md->homenr)/nthread;
        end_t   = md->start + ((thread+1)*md->homenr)/nthread;

        ekin_sum    = ekind->ekin_work[thread];
        dekindl_sum = &ekind->ekin_work[thread][opts->ngtc][0][0];

        clear_mat(ekin_sum[0]);

        ga = 0;
        gt = 0;
        for (n = start_t; n < end_t; n++)
        {
            hm   = 0.5*md->massT[n];

            for (d = 0; (d < DIM); d++)
            {
                v_corrt[d]  = v[n][d]  - grpstat[ga].u[d];
            }
            for (d = 0; (d < DIM); d++)
            {
                for (m = 0; (m < DIM); m++)
                {
                    /* if we're computing a full step velocity, v_corrt[d] has v(t).  Otherwise, v(t+dt/2) */
                    ekin_sum[gt][m][d] += hm*v_corrt[m]*v_corrt[d];
                }
            }
            if (md->nMassPerturbed && md->bPerturbed[n])
            {
                *dekindl_sum -=
                    0.5*(md->massB[n] - md->massA[n])*iprod(v_corrt, v_corrt);
            }
        }
    }

    ekind->dekindl = 0;
    for (thread = 0; thread < nthread; thread++)
    {
        m_add(tcstat[0].ekinh, ekind->ekin_work[thread][0],
                      tcstat[0].ekinh);

        ekind->dekindl += ekind->ekin_work[thread][opts->ngtc][0][0];
    }

    inc_nrnb(nrnb, eNR_EKIN, md->homenr);
}

void calc_ke_part(t_state *state, t_grpopts *opts, t_mdatoms *md,
                  gmx_ekindata_t *ekind, t_nrnb *nrnb, gmx_bool bEkinAveVel, gmx_bool bSaveEkinOld)
{ // called
        calc_ke_part_normal(state->v, opts, md, ekind, nrnb, bEkinAveVel, bSaveEkinOld);
}

extern void init_ekinstate(ekinstate_t *ekinstate, const t_inputrec *ir)
{ // called
    ekinstate->ekin_n = ir->opts.ngtc;
    snew(ekinstate->ekinh, ekinstate->ekin_n);
    snew(ekinstate->ekinf, ekinstate->ekin_n);
    snew(ekinstate->ekinh_old, ekinstate->ekin_n);
    snew(ekinstate->ekinscalef_nhc, ekinstate->ekin_n);
    snew(ekinstate->ekinscaleh_nhc, ekinstate->ekin_n);
    snew(ekinstate->vscale_nhc, ekinstate->ekin_n);
    ekinstate->dekindl = 0;
    ekinstate->mvcos   = 0;
}


