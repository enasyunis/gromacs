#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "string2.h"
#include "smalloc.h"
#include "mdrun.h"
#include "domdec.h"
#include "mtop_util.h"
#include "gmx_wallcycle.h"
#include "vcm.h"
#include "nrnb.h"
#include "md_logging.h"
#include "md_support.h"

/* Is the signal in one simulation independent of other simulations? */
gmx_bool gs_simlocal[eglsNR] = { TRUE, FALSE, FALSE, TRUE };

void init_global_signals(globsig_t *gs, const t_commrec *cr,
                         const t_inputrec *ir, int repl_ex_nst)
{ // called
    int i;

    gs->nstms = 1;

    for (i = 0; i < eglsNR; i++)
    {
        gs->sig[i] = 0;
        gs->set[i] = 0;
    }
}


void compute_globals(FILE *fplog, gmx_global_stat_t gstat, t_commrec *cr, t_inputrec *ir,
                     t_forcerec *fr, gmx_ekindata_t *ekind,
                     t_state *state, t_state *state_global, t_mdatoms *mdatoms,
                     t_nrnb *nrnb, t_vcm *vcm, gmx_wallcycle_t wcycle,
                     gmx_enerdata_t *enerd, tensor force_vir, tensor shake_vir, tensor total_vir,
                     tensor pres, rvec mu_tot, gmx_constr_t constr,
                     globsig_t *gs, gmx_bool bInterSimGS,
                     matrix box, gmx_mtop_t *top_global, real *pcurr,
                     int natoms, gmx_bool *bSumEkinhOld, int flags)
{ //called
    int      i, gsi;
    real     gs_buf[eglsNR];
    tensor   corr_vir, corr_pres, shakeall_vir;
    gmx_bool bEner, bPres, bTemp, bVV;
    gmx_bool bRerunMD, bStopCM, bGStat, bIterate,
             bFirstIterate, bReadEkin, bEkinAveVel, bScaleEkin, bConstrain;
    real     ekin, temp, prescorr, enercorr, dvdlcorr;

    /* translate CGLO flags to gmx_booleans */
    bRerunMD = flags & CGLO_RERUNMD;
    bStopCM  = flags & CGLO_STOPCM;
    bGStat   = flags & CGLO_GSTAT;

    bReadEkin     = (flags & CGLO_READEKIN);
    bScaleEkin    = (flags & CGLO_SCALEEKIN);
    bEner         = flags & CGLO_ENERGY;
    bTemp         = flags & CGLO_TEMPERATURE;
    bPres         = (flags & CGLO_PRESSURE);
    bConstrain    = (flags & CGLO_CONSTRAINT);
    bIterate      = (flags & CGLO_ITERATE);
    bFirstIterate = (flags & CGLO_FIRSTITERATE);

    /* we calculate a full state kinetic energy either with full-step velocity verlet
       or half step where we need the pressure */

    bEkinAveVel = (ir->eI == eiVV || (ir->eI == eiVVAK && bPres) || bReadEkin);

    /* in initalization, it sums the shake virial in vv, and to
       sums ekinh_old in leapfrog (or if we are calculating ekinh_old) for other reasons */

    /* ########## Kinetic energy  ############## */

    if (bTemp)
    {
        /* Non-equilibrium MD: this is parallellized, but only does communication
         * when there really is NEMD.
         */

        if (PAR(cr) && (ekind->bNEMD))
        {
            accumulate_u(cr, &(ir->opts), ekind);
        }
        debug_gmx();
        {

            calc_ke_part(state, &(ir->opts), mdatoms, ekind, nrnb, bEkinAveVel, bIterate);
        }

        debug_gmx();
    }

    /* Calculate center of mass velocity if necessary, also parallellized */
    if (bStopCM)
    {
        calc_vcm_grp(fplog, mdatoms->start, mdatoms->homenr, mdatoms,
                     state->x, state->v, vcm);
    }

    if (bTemp || bStopCM || bPres || bEner || bConstrain)
    {
        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            *bSumEkinhOld = TRUE;

        }
        else
        {
            if (gs != NULL)
            {
                for (i = 0; i < eglsNR; i++)
                {
                    gs_buf[i] = gs->sig[i];
                }
            }
            if (gs != NULL)
            {
                if (MULTISIM(cr) && bInterSimGS)
                {
                    if (MASTER(cr))
                    {
                        /* Communicate the signals between the simulations */
                        gmx_sum_sim(eglsNR, gs_buf, cr->ms);
                    }
                    /* Communicate the signals form the master to the others */
                    gmx_bcast(eglsNR*sizeof(gs_buf[0]), gs_buf, cr);
                }
                for (i = 0; i < eglsNR; i++)
                {
                    if (bInterSimGS || gs_simlocal[i])
                    {
                        /* Set the communicated signal only when it is non-zero,
                         * since signals might not be processed at each MD step.
                         */
                        gsi = (gs_buf[i] >= 0 ?
                               (int)(gs_buf[i] + 0.5) :
                               (int)(gs_buf[i] - 0.5));
                        if (gsi != 0)
                        {
                            gs->set[i] = gsi;
                        }
                        /* Turn off the local signal */
                        gs->sig[i] = 0;
                    }
                }
            }
            *bSumEkinhOld = FALSE;
        }
    }


    /* Do center of mass motion removal */
    if (bStopCM)
    {
        check_cm_grp(fplog, vcm, ir, 1);
        do_stopcm_grp(fplog, mdatoms->start, mdatoms->homenr, mdatoms->cVCM,
                      state->x, state->v, vcm);
        inc_nrnb(nrnb, eNR_STOPCM, mdatoms->homenr);
    }

    if (bEner)
    {
        /* Calculate the amplitude of the cosine velocity profile */
        ekind->cosacc.vcos = ekind->cosacc.mvcos/mdatoms->tmass;
    }

    if (bTemp)
    {
        /* Sum the kinetic energies of the groups & calc temp */
        /* compute full step kinetic energies if vv, or if vv-avek and we are computing the pressure with IR_NPT_TROTTER */
        /* three maincase:  VV with AveVel (md-vv), vv with AveEkin (md-vv-avek), leap with AveEkin (md).
           Leap with AveVel is not supported; it's not clear that it will actually work.
           bEkinAveVel: If TRUE, we simply multiply ekin by ekinscale to get a full step kinetic energy.
           If FALSE, we average ekinh_old and ekinh*ekinscale_nhc to get an averaged half step kinetic energy.
           bSaveEkinOld: If TRUE (in the case of iteration = bIterate is TRUE), we don't reset the ekinscale_nhc.
           If FALSE, we go ahead and erase over it.
         */
        enerd->term[F_TEMP] = sum_ekin(&(ir->opts), ekind, &(enerd->term[F_DKDL]),
                                       bEkinAveVel, bIterate, bScaleEkin);

        enerd->term[F_EKIN] = trace(ekind->ekin);
    }

    /* ##########  Long range energy information ###### */

    if (bEner || bPres || bConstrain)
    {
        calc_dispcorr(fplog, ir, fr, 0, top_global->natoms, box, state->lambda[efptVDW],
                      corr_pres, corr_vir, &prescorr, &enercorr, &dvdlcorr);
    }

    if (bEner && bFirstIterate)
    {
        enerd->term[F_DISPCORR]  = enercorr;
        enerd->term[F_EPOT]     += enercorr;
        enerd->term[F_DVDL_VDW] += dvdlcorr;
    }

    /* ########## Now pressure ############## */
    if (bPres || bConstrain)
    {

        m_add(force_vir, shake_vir, total_vir);

        /* Calculate pressure and apply LR correction if PPPM is used.
         * Use the box from last timestep since we already called update().
         */

        enerd->term[F_PRES] = calc_pres(fr->ePBC, ir->nwall, box, ekind->ekin, total_vir, pres);

        /* Calculate long range corrections to pressure and energy */
        /* this adds to enerd->term[F_PRES] and enerd->term[F_ETOT],
           and computes enerd->term[F_DISPCORR].  Also modifies the
           total_vir and pres tesors */

        m_add(total_vir, corr_vir, total_vir);
        m_add(pres, corr_pres, pres);
        enerd->term[F_PDISPCORR] = prescorr;
        enerd->term[F_PRES]     += prescorr;
        *pcurr                   = enerd->term[F_PRES];
    }
}


static void min_zero(int *n, int i)
{ // called
    if (i > 0 && (*n == 0 || i < *n))
    {
        *n = i;
    }
}

static int lcd4(int i1, int i2, int i3, int i4)
{ // called
    int nst;

    nst = 0;
    min_zero(&nst, i1);
    min_zero(&nst, i2);
    min_zero(&nst, i3);
    min_zero(&nst, i4);
    if (nst == 0)
    {
        gmx_incons("All 4 inputs for determininig nstglobalcomm are <= 0");
    }

    while (nst > 1 && ((i1 > 0 && i1 % nst != 0)  ||
                       (i2 > 0 && i2 % nst != 0)  ||
                       (i3 > 0 && i3 % nst != 0)  ||
                       (i4 > 0 && i4 % nst != 0)))
    {
        nst--;
    }

    return nst;
}

int check_nstglobalcomm(FILE *fplog, t_commrec *cr,
                        int nstglobalcomm, t_inputrec *ir)
{ // called

        if (!(ir->nstcalcenergy > 0 ||
              ir->nstlist > 0 ||
              ir->etc != etcNO ||
              ir->epc != epcNO))
        {
            nstglobalcomm = 10;
            if (ir->nstenergy > 0 && ir->nstenergy < nstglobalcomm)
            {
                nstglobalcomm = ir->nstenergy;
            }
        }
        else
        {
            /* Ensure that we do timely global communication for
             * (possibly) each of the four following options.
             */
            nstglobalcomm = lcd4(ir->nstcalcenergy,
                                 ir->nstlist,
                                 ir->etc != etcNO ? ir->nsttcouple : 0,
                                 ir->epc != epcNO ? ir->nstpcouple : 0);
        }

    if (ir->comm_mode != ecmNO && ir->nstcomm < nstglobalcomm)
    {
        md_print_warn(cr, fplog, "WARNING: Changing nstcomm from %d to %d\n",
                      ir->nstcomm, nstglobalcomm);
        ir->nstcomm = nstglobalcomm;
    }

    return nstglobalcomm;
}


