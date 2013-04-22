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

/*For debugging, start at v(-dt/2) for velolcity verlet -- uncomment next line */
/*#define STARTFROMDT2*/

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


static void do_update_md(int start, int nrend, double dt,
                         t_grp_tcstat *tcstat,
                         double nh_vxi[],
                         gmx_bool bNEMD, t_grp_acc *gstat, rvec accel[],
                         ivec nFreeze[],
                         real invmass[],
                         unsigned short ptype[], unsigned short cFREEZE[],
                         unsigned short cACC[], unsigned short cTC[],
                         rvec x[], rvec xprime[], rvec v[],
                         rvec f[], matrix M,
                         gmx_bool bNH, gmx_bool bPR)
{ // called
    double imass, w_dt;
    int    gf = 0, ga = 0, gt = 0;
    rvec   vrel;
    real   vn, vv, va, vb, vnrel;
    real   lg, vxi = 0, u;
    int    n, d;

    if (bNH || bPR)
    {
        /* Update with coupling to extended ensembles, used for
         * Nose-Hoover and Parrinello-Rahman coupling
         * Nose-Hoover uses the reversible leap-frog integrator from
         * Holian et al. Phys Rev E 52(3) : 2338, 1995
         */
        for (n = start; n < nrend; n++)
        {
            imass = invmass[n];
            if (cFREEZE)
            {
                gf   = cFREEZE[n];
            }
            if (cACC)
            {
                ga   = cACC[n];
            }
            if (cTC)
            {
                gt   = cTC[n];
            }
            lg   = tcstat[gt].lambda;
            if (bNH)
            {
                vxi   = nh_vxi[gt];
            }
            rvec_sub(v[n], gstat[ga].u, vrel);

            for (d = 0; d < DIM; d++)
            {
                if ((ptype[n] != eptVSite) && (ptype[n] != eptShell) && !nFreeze[gf][d])
                {
                    vnrel = (lg*vrel[d] + dt*(imass*f[n][d] - 0.5*vxi*vrel[d]
                                              - iprod(M[d], vrel)))/(1 + 0.5*vxi*dt);
                    /* do not scale the mean velocities u */
                    vn             = gstat[ga].u[d] + accel[ga][d]*dt + vnrel;
                    v[n][d]        = vn;
                    xprime[n][d]   = x[n][d]+vn*dt;
                }
                else
                {
                    v[n][d]        = 0.0;
                    xprime[n][d]   = x[n][d];
                }
            }
        }
    }
    else if (cFREEZE != NULL ||
             nFreeze[0][XX] || nFreeze[0][YY] || nFreeze[0][ZZ] ||
             bNEMD)
    {
        /* Update with Berendsen/v-rescale coupling and freeze or NEMD */
        for (n = start; n < nrend; n++)
        {
            w_dt = invmass[n]*dt;
            if (cFREEZE)
            {
                gf   = cFREEZE[n];
            }
            if (cACC)
            {
                ga   = cACC[n];
            }
            if (cTC)
            {
                gt   = cTC[n];
            }
            lg   = tcstat[gt].lambda;

            for (d = 0; d < DIM; d++)
            {
                vn             = v[n][d];
                if ((ptype[n] != eptVSite) && (ptype[n] != eptShell) && !nFreeze[gf][d])
                {
                    vv             = lg*vn + f[n][d]*w_dt;

                    /* do not scale the mean velocities u */
                    u              = gstat[ga].u[d];
                    va             = vv + accel[ga][d]*dt;
                    vb             = va + (1.0-lg)*u;
                    v[n][d]        = vb;
                    xprime[n][d]   = x[n][d]+vb*dt;
                }
                else
                {
                    v[n][d]        = 0.0;
                    xprime[n][d]   = x[n][d];
                }
            }
        }
    }
    else
    {
        /* Plain update with Berendsen/v-rescale coupling */
        for (n = start; n < nrend; n++)
        {
            if ((ptype[n] != eptVSite) && (ptype[n] != eptShell))
            {
                w_dt = invmass[n]*dt;
                if (cTC)
                {
                    gt = cTC[n];
                }
                lg = tcstat[gt].lambda;

                for (d = 0; d < DIM; d++)
                {
                    vn           = lg*v[n][d] + f[n][d]*w_dt;
                    v[n][d]      = vn;
                    xprime[n][d] = x[n][d] + vn*dt;
                }
            }
            else
            {
                for (d = 0; d < DIM; d++)
                {
                    v[n][d]        = 0.0;
                    xprime[n][d]   = x[n][d];
                }
            }
        }
    }
}

static gmx_stochd_t *init_stochd(FILE *fplog, t_inputrec *ir, int nthreads)
{// called
    gmx_stochd_t   *sd;
    gmx_sd_const_t *sdc;
    int             ngtc, n, th;
    real            y;

    snew(sd, 1);

    /* Initiate random number generator for langevin type dynamics,
     * for BD, SD or velocity rescaling temperature coupling.
     */
    if (ir->eI == eiBD || EI_SD(ir->eI))
    {
        sd->ngaussrand = nthreads;
    }
    else
    {
        sd->ngaussrand = 1;
    }
    snew(sd->gaussrand, sd->ngaussrand);

    /* Initialize the first random generator */
    sd->gaussrand[0] = gmx_rng_init(ir->ld_seed);


    ngtc = ir->opts.ngtc;

    if (ir->eI == eiBD)
    {
        snew(sd->bd_rf, ngtc);
    }
    else if (EI_SD(ir->eI))
    {
        snew(sd->sdc, ngtc);
        snew(sd->sdsig, ngtc);

        sdc = sd->sdc;
        for (n = 0; n < ngtc; n++)
        {
            if (ir->opts.tau_t[n] > 0)
            {
                sdc[n].gdt = ir->delta_t/ir->opts.tau_t[n];
                sdc[n].eph = exp(sdc[n].gdt/2);
                sdc[n].emh = exp(-sdc[n].gdt/2);
                sdc[n].em  = exp(-sdc[n].gdt);
            }
            else
            {
                /* No friction and noise on this group */
                sdc[n].gdt = 0;
                sdc[n].eph = 1;
                sdc[n].emh = 1;
                sdc[n].em  = 1;
            }
            if (sdc[n].gdt >= 0.05)
            {
                sdc[n].b = sdc[n].gdt*(sdc[n].eph*sdc[n].eph - 1)
                    - 4*(sdc[n].eph - 1)*(sdc[n].eph - 1);
                sdc[n].c = sdc[n].gdt - 3 + 4*sdc[n].emh - sdc[n].em;
                sdc[n].d = 2 - sdc[n].eph - sdc[n].emh;
            }
            else
            {
                y = sdc[n].gdt/2;
                /* Seventh order expansions for small y */
                sdc[n].b = y*y*y*y*(1/3.0+y*(1/3.0+y*(17/90.0+y*7/9.0)));
                sdc[n].c = y*y*y*(2/3.0+y*(-1/2.0+y*(7/30.0+y*(-1/12.0+y*31/1260.0))));
                sdc[n].d = y*y*(-1+y*y*(-1/12.0-y*y/360.0));
            }
            if (debug)
            {
                fprintf(debug, "SD const tc-grp %d: b %g  c %g  d %g\n",
                        n, sdc[n].b, sdc[n].c, sdc[n].d);
            }
        }
    }
    else if (ETC_ANDERSEN(ir->etc))
    {
        int        ngtc;
        t_grpopts *opts;
        real       reft;

        opts = &ir->opts;
        ngtc = opts->ngtc;

        snew(sd->randomize_group, ngtc);
        snew(sd->boltzfac, ngtc);

        /* for now, assume that all groups, if randomized, are randomized at the same rate, i.e. tau_t is the same. */
        /* since constraint groups don't necessarily match up with temperature groups! This is checked in readir.c */

        for (n = 0; n < ngtc; n++)
        {
            reft = max(0.0, opts->ref_t[n]);
            if ((opts->tau_t[n] > 0) && (reft > 0))  /* tau_t or ref_t = 0 means that no randomization is done */
            {
                sd->randomize_group[n] = TRUE;
                sd->boltzfac[n]        = BOLTZ*opts->ref_t[n];
            }
            else
            {
                sd->randomize_group[n] = FALSE;
            }
        }
    }
    return sd;
}


gmx_update_t init_update(FILE *fplog, t_inputrec *ir)
{ // called
    t_gmx_update *upd;

    snew(upd, 1);

    if (ir->eI == eiBD || EI_SD(ir->eI) || ir->etc == etcVRESCALE || ETC_ANDERSEN(ir->etc))
    {
        upd->sd = init_stochd(fplog, ir, gmx_omp_nthreads_get(emntUpdate));
    }

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
    int           g;
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
    for (g = 0; (g < opts->ngtc); g++)
    {

        if (!bSaveEkinOld)
        {
            copy_mat(tcstat[g].ekinh, tcstat[g].ekinh_old);
        }
        if (bEkinAveVel)
        {
            clear_mat(tcstat[g].ekinf);
        }
        else
        {
            clear_mat(tcstat[g].ekinh);
        }
        if (bEkinAveVel)
        {
            tcstat[g].ekinscalef_nhc = 1.0; /* need to clear this -- logic is complicated! */
        }
    }
    ekind->dekindl_old = ekind->dekindl;

    nthread = gmx_omp_nthreads_get(emntUpdate);

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

        for (gt = 0; gt < opts->ngtc; gt++)
        {
            clear_mat(ekin_sum[gt]);
        }

        ga = 0;
        gt = 0;
        for (n = start_t; n < end_t; n++)
        {
            if (md->cACC)
            {
                ga = md->cACC[n];
            }
            if (md->cTC)
            {
                gt = md->cTC[n];
            }
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
        for (g = 0; g < opts->ngtc; g++)
        {
            if (bEkinAveVel)
            {
                m_add(tcstat[g].ekinf, ekind->ekin_work[thread][g],
                      tcstat[g].ekinf);
            }
            else
            {
                m_add(tcstat[g].ekinh, ekind->ekin_work[thread][g],
                      tcstat[g].ekinh);
            }
        }

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


void update_tcouple(FILE             *fplog,
                    gmx_large_int_t   step,
                    t_inputrec       *inputrec,
                    t_state          *state,
                    gmx_ekindata_t   *ekind,
                    gmx_wallcycle_t   wcycle,
                    gmx_update_t      upd,
                    t_extmass        *MassQ,
                    t_mdatoms        *md)

{ // called
    gmx_bool   bTCouple = FALSE;
    real       dttc;
    int        i, start, end, homenr, offset;

    /* if using vv with trotter decomposition methods, we do this elsewhere in the code */
    if (inputrec->etc != etcNO &&
        !(IR_NVT_TROTTER(inputrec) || IR_NPT_TROTTER(inputrec) || IR_NPH_TROTTER(inputrec)))
    {
        /* We should only couple after a step where energies were determined (for leapfrog versions)
           or the step energies are determined, for velocity verlet versions */

        if (EI_VV(inputrec->eI))
        {
            offset = 0;
        }
        else
        {
            offset = 1;
        }
        bTCouple = (inputrec->nsttcouple == 1 ||
                    do_per_step(step+inputrec->nsttcouple-offset,
                                inputrec->nsttcouple));
    }

    if (bTCouple)
    {
        dttc = inputrec->nsttcouple*inputrec->delta_t;

        switch (inputrec->etc)
        {
            case etcNO:
                break;
            case etcBERENDSEN:
                berendsen_tcoupl(inputrec, ekind, dttc);
                break;
            case etcNOSEHOOVER:
                nosehoover_tcoupl(&(inputrec->opts), ekind, dttc,
                                  state->nosehoover_xi, state->nosehoover_vxi, MassQ);
                break;
            case etcVRESCALE:
                vrescale_tcoupl(inputrec, ekind, dttc,
                                state->therm_integral, upd->sd->gaussrand[0]);
                break;
        }
        /* rescale in place here */
        if (EI_VV(inputrec->eI))
        {
            rescale_velocities(ekind, md, md->start, md->start+md->homenr, state->v);
        }
    }
    else
    {
        /* Set the T scaling lambda to 1 to have no scaling */
        for (i = 0; (i < inputrec->opts.ngtc); i++)
        {
            ekind->tcstat[i].lambda = 1.0;
        }
    }
}

void update_pcouple(FILE             *fplog,
                    gmx_large_int_t   step,
                    t_inputrec       *inputrec,
                    t_state          *state,
                    matrix            pcoupl_mu,
                    matrix            M,
                    gmx_wallcycle_t   wcycle,
                    gmx_update_t      upd,
                    gmx_bool          bInitStep)
{ // called
    gmx_bool   bPCouple = FALSE;
    real       dtpc     = 0;
    int        i;

    /* if using Trotter pressure, we do this in coupling.c, so we leave it false. */
    if (inputrec->epc != epcNO && (!(IR_NPT_TROTTER(inputrec) || IR_NPH_TROTTER(inputrec))))
    {
        /* We should only couple after a step where energies were determined */
        bPCouple = (inputrec->nstpcouple == 1 ||
                    do_per_step(step+inputrec->nstpcouple-1,
                                inputrec->nstpcouple));
    }

    clear_mat(pcoupl_mu);
    for (i = 0; i < DIM; i++)
    {
        pcoupl_mu[i][i] = 1.0;
    }

    clear_mat(M);

    if (bPCouple)
    {
        dtpc = inputrec->nstpcouple*inputrec->delta_t;

        switch (inputrec->epc)
        {
            /* We can always pcoupl, even if we did not sum the energies
             * the previous step, since state->pres_prev is only updated
             * when the energies have been summed.
             */
            case (epcNO):
                break;
            case (epcBERENDSEN):
                if (!bInitStep)
                {
                    berendsen_pcoupl(fplog, step, inputrec, dtpc, state->pres_prev, state->box,
                                     pcoupl_mu);
                }
                break;
            case (epcPARRINELLORAHMAN):
                parrinellorahman_pcoupl(fplog, step, inputrec, dtpc, state->pres_prev,
                                        state->box, state->box_rel, state->boxv,
                                        M, pcoupl_mu, bInitStep);
                break;
            default:
                break;
        }
    }
}

static rvec *get_xprime(const t_state *state, gmx_update_t upd)
{ // called
    if (state->nalloc > upd->xp_nalloc)
    {
        upd->xp_nalloc = state->nalloc;
        srenew(upd->xp, upd->xp_nalloc);
    }

    return upd->xp;
}

void update_constraints(FILE             *fplog,
                        gmx_large_int_t   step,
                        real             *dvdlambda, /* the contribution to be added to the bonded interactions */
                        t_inputrec       *inputrec,  /* input record and box stuff	*/
                        gmx_ekindata_t   *ekind,
                        t_mdatoms        *md,
                        t_state          *state,
                        gmx_bool          bMolPBC,
                        t_graph          *graph,
                        rvec              force[],   /* forces on home particles */
                        t_idef           *idef,
                        tensor            vir_part,
                        tensor            vir,       /* tensors for virial and ekin, needed for computing */
                        t_commrec        *cr,
                        t_nrnb           *nrnb,
                        gmx_wallcycle_t   wcycle,
                        gmx_update_t      upd,
                        gmx_constr_t      constr,
                        gmx_bool          bInitStep,
                        gmx_bool          bFirstHalf,
                        gmx_bool          bCalcVir,
                        real              vetanew)
{ // called
    gmx_bool             bExtended, bLastStep, bLog = FALSE, bEner = FALSE, bDoConstr = FALSE;
    double               dt;
    real                 dt_1;
    int                  start, homenr, nrend, i, n, m, g, d;
    tensor               vir_con;
    rvec                *vbuf, *xprime = NULL;
    int                  nth, th;

    if (constr)
    {
        bDoConstr = TRUE;
    }
    if (bFirstHalf && !EI_VV(inputrec->eI))
    {
        bDoConstr = FALSE;
    }

    /* for now, SD update is here -- though it really seems like it
       should be reformulated as a velocity verlet method, since it has two parts */

    start  = md->start;
    homenr = md->homenr;
    nrend  = start+homenr;

    dt   = inputrec->delta_t;
    dt_1 = 1.0/dt;

    /*
     *  Steps (7C, 8C)
     *  APPLY CONSTRAINTS:
     *  BLOCK SHAKE

     * When doing PR pressure coupling we have to constrain the
     * bonds in each iteration. If we are only using Nose-Hoover tcoupling
     * it is enough to do this once though, since the relative velocities
     * after this will be normal to the bond vector
     */


    where();
    if ((inputrec->eI == eiSD2) && !(bFirstHalf))
    {
        xprime = get_xprime(state, upd);

        nth = gmx_omp_nthreads_get(emntUpdate);

        inc_nrnb(nrnb, eNR_UPDATE, homenr);

    }

    /* We must always unshift after updating coordinates; if we did not shake
       x was shifted in do_force */

    if (!(bFirstHalf)) /* in the first half of vv, no shift. */
    {
        if (graph && (graph->nnodes > 0))
        {
            unshift_x(graph, state->box, state->x, upd->xp);
            if (TRICLINIC(state->box))
            {
                inc_nrnb(nrnb, eNR_SHIFTX, 2*graph->nnodes);
            }
            else
            {
                inc_nrnb(nrnb, eNR_SHIFTX, graph->nnodes);
            }
        }
        else
        {
#pragma omp parallel for num_threads(gmx_omp_nthreads_get(emntUpdate)) schedule(static)
            for (i = start; i < nrend; i++)
            {
                copy_rvec(upd->xp[i], state->x[i]);
            }
        }

    }
/* ############# END the update of velocities and positions ######### */
}

void update_box(FILE             *fplog,
                gmx_large_int_t   step,
                t_inputrec       *inputrec,  /* input record and box stuff	*/
                t_mdatoms        *md,
                t_state          *state,
                t_graph          *graph,
                rvec              force[],   /* forces on home particles */
                matrix           *scale_tot,
                matrix            pcoupl_mu,
                t_nrnb           *nrnb,
                gmx_wallcycle_t   wcycle,
                gmx_update_t      upd,
                gmx_bool          bInitStep,
                gmx_bool          bFirstHalf)
{ // called
    gmx_bool             bExtended, bLastStep, bLog = FALSE, bEner = FALSE;
    double               dt;
    real                 dt_1;
    int                  start, homenr, nrend, i, n, m, g;
    tensor               vir_con;

    start  = md->start;
    homenr = md->homenr;
    nrend  = start+homenr;

    bExtended =
        (inputrec->etc == etcNOSEHOOVER) ||
        (inputrec->epc == epcPARRINELLORAHMAN) ||
        (inputrec->epc == epcMTTK);

    dt = inputrec->delta_t;

    where();

    /* now update boxes */
    switch (inputrec->epc)
    {
        case (epcNO):
            break;
        case (epcBERENDSEN):
            berendsen_pscale(inputrec, pcoupl_mu, state->box, state->box_rel,
                             start, homenr, state->x, md->cFREEZE, nrnb);
            break;
        case (epcPARRINELLORAHMAN):
            /* The box velocities were updated in do_pr_pcoupl in the update
             * iteration, but we dont change the box vectors until we get here
             * since we need to be able to shift/unshift above.
             */
            for (i = 0; i < DIM; i++)
            {
                for (m = 0; m <= i; m++)
                {
                    state->box[i][m] += dt*state->boxv[i][m];
                }
            }
            preserve_box_shape(inputrec, state->box_rel, state->box);

            /* Scale the coordinates */
            for (n = start; (n < start+homenr); n++)
            {
                tmvmul_ur0(pcoupl_mu, state->x[n], state->x[n]);
            }
            break;
        case (epcMTTK):
            switch (inputrec->epct)
            {
                case (epctISOTROPIC):
                    /* DIM * eta = ln V.  so DIM*eta_new = DIM*eta_old + DIM*dt*veta =>
                       ln V_new = ln V_old + 3*dt*veta => V_new = V_old*exp(3*dt*veta) =>
                       Side length scales as exp(veta*dt) */

                    msmul(state->box, exp(state->veta*dt), state->box);

                    /* Relate veta to boxv.  veta = d(eta)/dT = (1/DIM)*1/V dV/dT.
                       o               If we assume isotropic scaling, and box length scaling
                       factor L, then V = L^DIM (det(M)).  So dV/dt = DIM
                       L^(DIM-1) dL/dt det(M), and veta = (1/L) dL/dt.  The
                       determinant of B is L^DIM det(M), and the determinant
                       of dB/dt is (dL/dT)^DIM det (M).  veta will be
                       (det(dB/dT)/det(B))^(1/3).  Then since M =
                       B_new*(vol_new)^(1/3), dB/dT_new = (veta_new)*B(new). */

                    msmul(state->box, state->veta, state->boxv);
                    break;
                default:
                    break;
            }
            break;
        default:
            break;
    }

    if ((!(IR_NPT_TROTTER(inputrec) || IR_NPH_TROTTER(inputrec))) && scale_tot)
    {
        /* The transposes of the scaling matrices are stored,
         * therefore we need to reverse the order in the multiplication.
         */
        mmul_ur0(*scale_tot, pcoupl_mu, *scale_tot);
    }

    where();
}

void update_coords(FILE             *fplog,
                   gmx_large_int_t   step,
                   t_inputrec       *inputrec,  /* input record and box stuff	*/
                   t_mdatoms        *md,
                   t_state          *state,
                   gmx_bool          bMolPBC,
                   rvec             *f,    /* forces on home particles */
                   gmx_bool          bDoLR,
                   rvec             *f_lr,
                   t_fcdata         *fcd,
                   gmx_ekindata_t   *ekind,
                   matrix            M,
                   gmx_wallcycle_t   wcycle,
                   gmx_update_t      upd,
                   gmx_bool          bInitStep,
                   int               UpdatePart,
                   t_commrec        *cr, /* these shouldn't be here -- need to think about it */
                   t_nrnb           *nrnb,
                   gmx_constr_t      constr,
                   t_idef           *idef)
{ // called
    gmx_bool          bNH, bPR, bLastStep, bLog = FALSE, bEner = FALSE;
    double            dt, alpha;
    real             *imass, *imassin;
    rvec             *force;
    real              dt_1;
    int               start, homenr, nrend, i, j, d, n, m, g;
    int               blen0, blen1, iatom, jatom, nshake, nsettle, nconstr, nexpand;
    int              *icom = NULL;
    tensor            vir_con;
    rvec             *vcom, *xcom, *vall, *xall, *xin, *vin, *forcein, *fall, *xpall, *xprimein, *xprime;
    int               nth, th;

    /* Running the velocity half does nothing except for velocity verlet */
    if ((UpdatePart == etrtVELOCITY1 || UpdatePart == etrtVELOCITY2) &&
        !EI_VV(inputrec->eI))
    {
        gmx_incons("update_coords called for velocity without VV integrator");
    }

    start  = md->start;
    homenr = md->homenr;
    nrend  = start+homenr;

    xprime = get_xprime(state, upd);

    dt   = inputrec->delta_t;
    dt_1 = 1.0/dt;

    /* We need to update the NMR restraint history when time averaging is used */
    if (state->flags & (1<<estDISRE_RM3TAV))
    {
        update_disres_history(fcd, &state->hist);
    }
    if (state->flags & (1<<estORIRE_DTAV))
    {
        update_orires_history(fcd, &state->hist);
    }


    bNH = inputrec->etc == etcNOSEHOOVER;
    bPR = ((inputrec->epc == epcPARRINELLORAHMAN) || (inputrec->epc == epcMTTK));

    force = f;

    /* ############# START The update of velocities and positions ######### */
    where();

    if (EI_RANDOM(inputrec->eI))
    {
        /* We still need to take care of generating random seeds properly
         * when multi-threading.
         */
        nth = 1;
    }
    else
    {
        nth = gmx_omp_nthreads_get(emntUpdate);
    }

#pragma omp parallel for num_threads(nth) schedule(static) private(alpha)
    for (th = 0; th < nth; th++)
    {
        int start_th, end_th;

        start_th = start + ((nrend-start)* th   )/nth;
        end_th   = start + ((nrend-start)*(th+1))/nth;
        do_update_md(start_th, end_th, dt,
                                 ekind->tcstat, state->nosehoover_vxi,
                                 ekind->bNEMD, ekind->grpstat, inputrec->opts.acc,
                                 inputrec->opts.nFreeze,
                                 md->invmass, md->ptype,
                                 md->cFREEZE, md->cACC, md->cTC,
                                 state->x, xprime, state->v, force, M,
                                 bNH, bPR);
    }

}

