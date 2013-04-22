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
        /* Plain update with Berendsen/v-rescale coupling */
        for (n = start; n < nrend; n++) // the code is sent across the OpenMP threads
        {
                w_dt = invmass[n]*dt;
                lg = tcstat[gt].lambda;
                for (d = 0; d < DIM; d++)
                {
                    vn           = lg*v[n][d] + f[n][d]*w_dt;
                    v[n][d]      = vn;
                    xprime[n][d] = x[n][d] + vn*dt;
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
    /* Set the T scaling lambda to 1 to have no scaling */
    ekind->tcstat[0].lambda = 1.0;
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
    int        i;

    clear_mat(pcoupl_mu);
    for (i = 0; i < DIM; i++)
    {
        pcoupl_mu[i][i] = 1.0;
    }

    clear_mat(M);
}

static rvec *get_xprime(const t_state *state, gmx_update_t upd)
{ // called
    upd->xp_nalloc = state->nalloc;
    srenew(upd->xp, upd->xp_nalloc);

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
    int                  start, homenr, nrend, i;

    /* for now, SD update is here -- though it really seems like it
       should be reformulated as a velocity verlet method, since it has two parts */

    start  = md->start;
    homenr = md->homenr;
    nrend  = start+homenr;

    where();


    /* We must always unshift after updating coordinates; if we did not shake
       x was shifted in do_force */

#pragma omp parallel for num_threads(gmx_omp_nthreads_get(emntUpdate)) schedule(static)
    for (i = start; i < nrend; i++) // 3000
    {
       copy_rvec(upd->xp[i], state->x[i]);
    }

/* ############# END the update of velocities and positions ######### */
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
    int               start, homenr, nrend, i, j, d, n, m, g;
    int               blen0, blen1, iatom, jatom, nshake, nsettle, nconstr, nexpand;
    int              *icom = NULL;
    tensor            vir_con;
    rvec             *vcom, *xcom, *vall, *xall, *xin, *vin, *forcein, *fall, *xpall, *xprimein, *xprime;
    int               nth, th;

    start  = md->start;
    homenr = md->homenr;
    nrend  = start+homenr;

    xprime = get_xprime(state, upd);

    dt   = inputrec->delta_t;

    bNH = inputrec->etc == etcNOSEHOOVER;
    bPR = ((inputrec->epc == epcPARRINELLORAHMAN) || (inputrec->epc == epcMTTK));

    force = f;

    /* ############# START The update of velocities and positions ######### */
    where();

    nth = gmx_omp_nthreads_get(emntUpdate);


#pragma omp parallel for num_threads(nth) schedule(static) private(alpha)
    for (th = 0; th < nth; th++) // 12
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

