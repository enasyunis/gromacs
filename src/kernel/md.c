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

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "statutil.h"
#include "vcm.h"
#include "mdebin.h"
#include "nrnb.h"
#include "calcmu.h"
#include "index.h"
#include "vsite.h"
#include "update.h"
#include "ns.h"
#include "trnio.h"
#include "xtcio.h"
#include "mdrun.h"
#include "md_support.h"
#include "md_logging.h"
#include "confio.h"
#include "network.h"
#include "pull.h"
#include "xvgr.h"
#include "physics.h"
#include "names.h"
#include "xmdrun.h"
#include "ionize.h"
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "repl_ex.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "domdec_network.h"
#include "partdec.h"
#include "topsort.h"
#include "coulomb.h"
#include "constr.h"
#include "shellfc.h"
#include "compute_io.h"
#include "mvdata.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "txtdump.h"
#include "string2.h"
#include "pme_loadbal.h"
#include "bondf.h"
#include "membed.h"
#include "types/nlistheuristics.h"
#include "types/iteratedconstraints.h"
#include "nbnxn_cuda_data_mgmt.h"

#ifdef GMX_LIB_MPI
#include <mpi.h>
#endif
#ifdef GMX_THREAD_MPI
#include "tmpi.h"
#endif

#ifdef GMX_FAHCORE
#include "corewrap.h"
#endif


double do_md(FILE *fplog, t_commrec *cr, int nfile, const t_filenm fnm[],
             const output_env_t oenv, gmx_bool bVerbose, gmx_bool bCompact,
             int nstglobalcomm,
             gmx_vsite_t *vsite, gmx_constr_t constr,
             int stepout, t_inputrec *ir,
             gmx_mtop_t *top_global,
             t_fcdata *fcd,
             t_state *state_global,
             t_mdatoms *mdatoms,
             t_nrnb *nrnb, gmx_wallcycle_t wcycle,
             gmx_edsam_t ed, t_forcerec *fr,
             int repl_ex_nst, int repl_ex_nex, int repl_ex_seed, gmx_membed_t membed,
             real cpt_period, real max_hours,
             const char *deviceOptions,
             unsigned long Flags,
             gmx_runtime_t *runtime)
{
    gmx_mdoutf_t   *outf;
    gmx_large_int_t step, step_rel;
    double          run_time;
    double          t, t0, lam0[efptNR];
    gmx_bool        bGStatEveryStep, bGStat, bCalcVir, bCalcEner;
    gmx_bool        bNS, bNStList, bSimAnn, bStopCM, bNotLastFrame = FALSE;
    gmx_bool        bFirstStep, bStateFromCP, bStateFromTPX, bInitStep, bLastStep;
    gmx_bool        bBornRadii, bStartingFromCpt;
    gmx_bool          bDoDHDL = FALSE, bDoFEP = FALSE, bDoExpanded = FALSE;
    gmx_bool          do_ene, do_log, do_verbose, bRerunWarnNoV = TRUE,
                      bForceUpdate = FALSE, bCPT;
    int               mdof_flags;
    gmx_bool          bMasterState;
    int               force_flags, cglo_flags;
    tensor            force_vir, shake_vir, total_vir, tmp_vir, pres;
    int               i, m;
    t_trxstatus      *status;
    rvec              mu_tot;
    t_vcm            *vcm;
    t_state          *bufstate = NULL;
    matrix           *scale_tot, pcoupl_mu, M, ebox;
    gmx_nlheur_t      nlh;
    t_trxframe        rerun_fr;
    gmx_repl_ex_t     repl_ex = NULL;
    int               nchkpt  = 1;
    gmx_localtop_t   *top;
    t_mdebin         *mdebin = NULL;
    df_history_t      df_history;
    t_state          *state    = NULL;
    rvec             *f_global = NULL;
    int               n_xtc    = -1;
    rvec             *x_xtc    = NULL;
    gmx_enerdata_t   *enerd;
    rvec             *f = NULL;
    gmx_global_stat_t gstat;
    gmx_update_t      upd   = NULL;
    t_graph          *graph = NULL;
    globsig_t         gs;
    gmx_rng_t         mcrng = NULL;
    gmx_groups_t     *groups;
    gmx_ekindata_t   *ekind, *ekind_save;
    gmx_shellfc_t     shellfc;
    int               count, nconverged = 0;
    real              timestep = 0;
    double            tcount   = 0;
    gmx_bool          bConverged = TRUE, bOK, bSumEkinhOld, bExchanged;
    gmx_bool          bResetCountersHalfMaxH = FALSE;
    gmx_bool          bIterativeCase, bFirstIterate, bTemp, bPres, bTrotter;
    gmx_bool          bUpdateDoLR;
    real              mu_aver = 0, dvdl;
    int               a0, a1, gnx = 0, ii;
    atom_id          *grpindex = NULL;
    char             *grpname;
    t_coupl_rec      *tcr     = NULL;
    rvec             *xcopy   = NULL, *vcopy = NULL, *cbuf = NULL;
    matrix            boxcopy = {{0}}, lastbox;
    tensor            tmpvir;
    real              fom, oldfom, veta_save, pcurr, scalevir, tracevir;
    real              vetanew = 0;
    int               lamnew  = 0;
    /* for FEP */
    int               nstfep;
    real              rate;
    double            cycles;
    real              saved_conserved_quantity = 0;
    real              last_ekin                = 0;
    int               iter_i;
    t_extmass         MassQ;
    int             **trotter_seq;
    char              sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];
    int               handled_stop_condition = gmx_stop_cond_none; /* compare to get_stop_condition*/
    gmx_iterate_t     iterate;
    gmx_large_int_t   multisim_nsteps = -1;                        /* number of steps to do  before first multisim
                                                                      simulation stops. If equal to zero, don't
                                                                      communicate any more between multisims.*/
    /* PME load balancing data for GPU kernels */
    pme_load_balancing_t pme_loadbal = NULL;
    double               cycles_pmes;
#ifdef GMX_FAHCORE
    /* Temporary addition for FAHCORE checkpointing */
    int chkpt_ret;
#endif

    /* Check for special mdrun options */
    if (Flags & MD_RESETCOUNTERSHALFWAY)
    {
        if (ir->nsteps > 0)
        {
            /* Signal to reset the counters half the simulation steps. */
            wcycle_set_reset_counters(wcycle, ir->nsteps/2);
        }
        /* Signal to reset the counters halfway the simulation time. */
        bResetCountersHalfMaxH = (max_hours > 0);
    }

    /* md-vv uses averaged full step velocities for T-control
       md-vv-avek uses averaged half step velocities for T-control (but full step ekin for P control)
       md uses averaged half step kinetic energies to determine temperature unless defined otherwise by GMX_EKIN_AVE_VEL; */
    /* all the iteratative cases - only if there are constraints */
    bIterativeCase = ((IR_NPH_TROTTER(ir) || IR_NPT_TROTTER(ir)) && (constr) );
    gmx_iterate_init(&iterate, FALSE); /* The default value of iterate->bIterationActive is set to
                                          false in this step.  The correct value, true or false,
                                          is set at each step, as it depends on the frequency of temperature
                                          and pressure control.*/
    bTrotter = 0; 


    check_ir_old_tpx_versions(cr, fplog, ir, top_global);

    nstglobalcomm   = check_nstglobalcomm(fplog, cr, nstglobalcomm, ir);
    bGStatEveryStep = (nstglobalcomm == 1);

    if (!bGStatEveryStep && ir->nstlist == -1 && fplog != NULL)
    {
        fprintf(fplog,
                "To reduce the energy communication with nstlist = -1\n"
                "the neighbor list validity should not be checked at every step,\n"
                "this means that exact integration is not guaranteed.\n"
                "The neighbor list validity is checked after:\n"
                "  <n.list life time> - 2*std.dev.(n.list life time)  steps.\n"
                "In most cases this will result in exact integration.\n"
                "This reduces the energy communication by a factor of 2 to 3.\n"
                "If you want less energy communication, set nstlist > 3.\n\n");
    }

    groups = &top_global->groups;

    /* Initial values */
    init_md(fplog, cr, ir, oenv, &t, &t0, state_global->lambda,
            &(state_global->fep_state), lam0,
            nrnb, top_global, &upd,
            nfile, fnm, &outf, &mdebin,
            force_vir, shake_vir, mu_tot, &bSimAnn, &vcm, state_global, Flags);

    clear_mat(total_vir);
    clear_mat(pres);
    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda,
                  enerd);
        snew(f, top_global->natoms);

    /* lambda Monte carlo random number generator  */
    if (ir->bExpanded)
    {
        mcrng = gmx_rng_init(ir->expandedvals->lmc_seed);
    }
    /* copy the state into df_history */
    copy_df_history(&df_history, &state_global->dfhist);

    /* Kinetic energy data */
    snew(ekind, 1);
    init_ekindata(fplog, top_global, &(ir->opts), ekind);
    /* needed for iteration of constraints */
    snew(ekind_save, 1);
    init_ekindata(fplog, top_global, &(ir->opts), ekind_save);
    /* Copy the cos acceleration to the groups struct */
    ekind->cosacc.cos_accel = ir->cos_accel;

    gstat = global_stat_init(ir);
    debug_gmx();

    /* Check for polarizable models and flexible constraints */
    shellfc = init_shell_flexcon(fplog,
                                 top_global, n_flexible_constraints(constr),
                                 (ir->bContinuation 
                                  ) ?
                                 NULL : state_global->x);

    if (DEFORM(*ir))
    {
#ifdef GMX_THREAD_MPI
        tMPI_Thread_mutex_lock(&deform_init_box_mutex);
#endif
        set_deform_reference_box(upd,
                                 deform_init_init_step_tpx,
                                 deform_init_box_tpx);
#ifdef GMX_THREAD_MPI
        tMPI_Thread_mutex_unlock(&deform_init_box_mutex);
#endif
    }

    {
        double io = compute_io(ir, top_global->natoms, groups, mdebin->ebin->nener, 1);
        if ((io > 2000) && MASTER(cr))
        {
            fprintf(stderr,
                    "\nWARNING: This run will generate roughly %.0f Mb of data\n\n",
                    io);
        }
    }

        if (PAR(cr))
        {
            /* Initialize the particle decomposition and split the topology */
            top = split_system(fplog, top_global, ir, cr);

            pd_cg_range(cr, &fr->cg0, &fr->hcg);
            pd_at_range(cr, &a0, &a1);
        }
        else
        {
            top = gmx_mtop_generate_local_top(top_global, ir);

            a0 = 0;
            a1 = top_global->natoms;
        }

        forcerec_set_excl_load(fr, top, cr);

        state    = partdec_init_local_state(cr, state_global);
        f_global = f;

        atoms2md(top_global, ir, 0, NULL, a0, a1-a0, mdatoms);

        if (ir->ePBC != epbcNONE && !fr->bMolPBC)
        {
            graph = mk_graph(fplog, &(top->idef), 0, top_global->natoms, FALSE, FALSE);
        }


        init_bonded_thread_force_reduction(fr, &top->idef);

        if (ir->pull && PAR(cr))
        {
            dd_make_local_pull_groups(NULL, ir->pull, mdatoms);
        }


    update_mdatoms(mdatoms, state->lambda[efptMASS]);

    if (opt2bSet("-cpi", nfile, fnm))
    {
        bStateFromCP = gmx_fexist_master(opt2fn_master("-cpi", nfile, fnm, cr), cr);
    }
    else
    {
        bStateFromCP = FALSE;
    }

    if (MASTER(cr))
    {
        if (bStateFromCP)
        {
            /* Update mdebin with energy history if appending to output files */
            if (Flags & MD_APPENDFILES)
            {
                restore_energyhistory_from_state(mdebin, &state_global->enerhist);
            }
            else
            {
                /* We might have read an energy history from checkpoint,
                 * free the allocated memory and reset the counts.
                 */
                done_energyhistory(&state_global->enerhist);
                init_energyhistory(&state_global->enerhist);
            }
        }
        /* Set the initial energy history in state by updating once */
        update_energyhistory(&state_global->enerhist, mdebin);
    }

    if ((state->flags & (1<<estLD_RNG)) && (Flags & MD_READ_RNG))
    {
        /* Set the random state if we read a checkpoint file */
        set_stochd_state(upd, state);
    }

    if (state->flags & (1<<estMC_RNG))
    {
        set_mc_state(mcrng, state);
    }

    /* Initialize constraints */
    if (constr)
    {
            set_constraints(constr, top, ir, mdatoms, cr);
    }

    if (repl_ex_nst > 0)
    {
        /* We need to be sure replica exchange can only occur
         * when the energies are current */
        check_nst_param(fplog, cr, "nstcalcenergy", ir->nstcalcenergy,
                        "repl_ex_nst", &repl_ex_nst);
        /* This check needs to happen before inter-simulation
         * signals are initialized, too */
    }
    if (repl_ex_nst > 0 && MASTER(cr))
    {
        repl_ex = init_replica_exchange(fplog, cr->ms, state_global, ir,
                                        repl_ex_nst, repl_ex_nex, repl_ex_seed);
    }


    if (!ir->bContinuation )
    {
        if (mdatoms->cFREEZE && (state->flags & (1<<estV)))
        {
            /* Set the velocities of frozen particles to zero */
            for (i = mdatoms->start; i < mdatoms->start+mdatoms->homenr; i++)
            {
                for (m = 0; m < DIM; m++)
                {
                    if (ir->opts.nFreeze[mdatoms->cFREEZE[i]][m])
                    {
                        state->v[i][m] = 0;
                    }
                }
            }
        }

    }

    debug_gmx();

    /* set free energy calculation frequency as the minimum of nstdhdl, nstexpanded, and nstrepl_ex_nst*/
    nstfep = ir->fepvals->nstdhdl;
    if (ir->bExpanded && (nstfep > ir->expandedvals->nstexpanded))
    {
        nstfep = ir->expandedvals->nstexpanded;
    }
    if (repl_ex_nst > 0 && nstfep > repl_ex_nst)
    {
        nstfep = repl_ex_nst;
    }

    /* I'm assuming we need global communication the first time! MRS */
    cglo_flags = (CGLO_TEMPERATURE | CGLO_GSTAT
                  | ((ir->comm_mode != ecmNO) ? CGLO_STOPCM : 0)
                  | ((Flags & MD_READ_EKIN) ? CGLO_READEKIN : 0));

    bSumEkinhOld = FALSE;
    compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                    NULL, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                    constr, NULL, FALSE, state->box,
                    top_global, &pcurr, top_global->natoms, &bSumEkinhOld, cglo_flags);
    if (ir->eI == eiVVAK)
    {
        /* a second call to get the half step temperature initialized as well */
        /* we do the same call as above, but turn the pressure off -- internally to
           compute_globals, this is recognized as a velocity verlet half-step
           kinetic energy calculation.  This minimized excess variables, but
           perhaps loses some logic?*/

        compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                        NULL, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                        constr, NULL, FALSE, state->box,
                        top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                        cglo_flags &~(CGLO_STOPCM | CGLO_PRESSURE));
    }

    /* Calculate the initial half step temperature, and save the ekinh_old */
    if (!(Flags & MD_STARTFROMCPT))
    {
        for (i = 0; (i < ir->opts.ngtc); i++)
        {
            copy_mat(ekind->tcstat[i].ekinh, ekind->tcstat[i].ekinh_old);
        }
    }
    if (ir->eI != eiVV)
    {
        enerd->term[F_TEMP] *= 2; /* result of averages being done over previous and current step,
                                     and there is no previous step */
    }

    /* if using an iterative algorithm, we need to create a working directory for the state. */
    if (bIterativeCase)
    {
        bufstate = init_bufstate(state);
    }

    /* need to make an initiation call to get the Trotter variables set, as well as other constants for non-trotter
       temperature control */
    trotter_seq = init_npt_vars(ir, state, &MassQ, bTrotter);

    if (MASTER(cr))
    {
        if (constr && !ir->bContinuation && ir->eConstrAlg == econtLINCS)
        {
            fprintf(fplog,
                    "RMS relative constraint deviation after constraining: %.2e\n",
                    constr_rmsd(constr, FALSE));
        }
        if (EI_STATE_VELOCITY(ir->eI))
        {
            fprintf(fplog, "Initial temperature: %g K\n", enerd->term[F_TEMP]);
        }
            char tbuf[20];
            fprintf(stderr, "starting mdrun '%s'\n",
                    *(top_global->name));
            if (ir->nsteps >= 0)
            {
                sprintf(tbuf, "%8.1f", (ir->init_step+ir->nsteps)*ir->delta_t);
            }
            else
            {
                sprintf(tbuf, "%s", "infinite");
            }
            if (ir->init_step > 0)
            {
                fprintf(stderr, "%s steps, %s ps (continuing from step %s, %8.1f ps).\n",
                        gmx_step_str(ir->init_step+ir->nsteps, sbuf), tbuf,
                        gmx_step_str(ir->init_step, sbuf2),
                        ir->init_step*ir->delta_t);
            }
            else
            {
                fprintf(stderr, "%s steps, %s ps.\n",
                        gmx_step_str(ir->nsteps, sbuf), tbuf);
            }
        fprintf(fplog, "\n");
    }

    /* Set and write start time */
    runtime_start(runtime);
    print_date_and_time(fplog, cr->nodeid, "Started mdrun", runtime);
    wallcycle_start(wcycle, ewcRUN);
    if (fplog)
    {
        fprintf(fplog, "\n");
    }

    /* safest point to do file checkpointing is here.  More general point would be immediately before integrator call */
#ifdef GMX_FAHCORE
    chkpt_ret = fcCheckPointParallel( cr->nodeid,
                                      NULL, 0);
    if (chkpt_ret == 0)
    {
        gmx_fatal( 3, __FILE__, __LINE__, "Checkpoint error on step %d\n", 0 );
    }
#endif

    debug_gmx();


    /* loop over MD steps or if rerunMD to end of input trajectory */
    bFirstStep = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bStateFromTPX    = !bStateFromCP;
    bInitStep        = bFirstStep && (bStateFromTPX);
    bStartingFromCpt = (Flags & MD_STARTFROMCPT) && bInitStep;
    bLastStep        = FALSE;
    bSumEkinhOld     = FALSE;
    bExchanged       = FALSE;

    init_global_signals(&gs, cr, ir, repl_ex_nst);

    step     = ir->init_step;
    step_rel = 0;

    if (MULTISIM(cr) && (repl_ex_nst <= 0 ))
    {
        /* check how many steps are left in other sims */
        multisim_nsteps = get_multisim_nsteps(cr, ir->nsteps);
    }


    /* and stop now if we should */
    bLastStep = ((ir->nsteps >= 0 && step_rel > ir->nsteps) ||
                 ((multisim_nsteps >= 0) && (step_rel >= multisim_nsteps )));
    while (!bLastStep )
    {

        wallcycle_start(wcycle, ewcSTEP);

        GMX_MPE_LOG(ev_timestep1);

            bLastStep = (step_rel == ir->nsteps);
            t         = t0 + step*ir->delta_t;

        if (ir->efep != efepNO || ir->bSimTemp)
        {
            /* find and set the current lambdas.  If rerunning, we either read in a state, or a lambda value,
               requiring different logic. */

            set_current_lambdas(step, ir->fepvals, 0, &rerun_fr, state_global, state, lam0);
            bDoDHDL      = do_per_step(step, ir->fepvals->nstdhdl);
            bDoFEP       = (do_per_step(step, nstfep) && (ir->efep != efepNO));
            bDoExpanded  = (do_per_step(step, ir->expandedvals->nstexpanded) && (ir->bExpanded) && (step > 0));
        }

        if (bSimAnn)
        {
            update_annealing_target_temp(&(ir->opts), t);
        }


        /* Stop Center of Mass motion */
        bStopCM = (ir->comm_mode != ecmNO && do_per_step(step, ir->nstcomm));

        /* Copy back starting coordinates in case we're doing a forcefield scan */
            bNStList = (ir->nstlist > 0  && step % ir->nstlist == 0);

            bNS = (bFirstStep || bExchanged || bNStList || bDoFEP ||
                   (ir->nstlist == -1 && nlh.nabnsb > 0));


        /* check whether we should stop because another simulation has
           stopped. */
        if (MULTISIM(cr))
        {
            if ( (multisim_nsteps >= 0) &&  (step_rel >= multisim_nsteps)  &&
                 (multisim_nsteps != ir->nsteps) )
            {
                if (bNS)
                {
                    if (MASTER(cr))
                    {
                        fprintf(stderr,
                                "Stopping simulation %d because another one has finished\n",
                                cr->ms->sim);
                    }
                    bLastStep         = TRUE;
                    gs.sig[eglsCHKPT] = 1;
                }
            }
        }

        /* < 0 means stop at next step, > 0 means stop at next NS step */
        if ( (gs.set[eglsSTOPCOND] < 0 ) ||
             ( (gs.set[eglsSTOPCOND] > 0 ) && ( bNS || ir->nstlist == 0)) )
        {
            bLastStep = TRUE;
        }

        /* Determine whether or not to update the Born radii if doing GB */
        bBornRadii = bFirstStep;
        if (ir->implicit_solvent && (step % ir->nstgbradii == 0))
        {
            bBornRadii = TRUE;
        }

        do_log     = do_per_step(step, ir->nstlog) || bFirstStep || bLastStep;
        do_verbose = bVerbose &&
            (step % stepout == 0 || bFirstStep || bLastStep);

        if (bNS && !(bFirstStep && ir->bContinuation))
        {
                bMasterState = FALSE;
                /* Correct the new box if it is too skewed */
                if (DYNAMIC_BOX(*ir))
                {
                    if (correct_box(fplog, step, state->box, graph))
                    {
                        bMasterState = TRUE;
                    }
                }

        }

        if (MASTER(cr) && do_log )
        {
            print_ebin_header(fplog, step, t, state->lambda[efptFEP]); /* can we improve the information printed here? */
        }

        if (ir->efep != efepNO)
        {
            update_mdatoms(mdatoms, state->lambda[efptMASS]);
        }

        if ( bExchanged)
        {

            /* We need the kinetic energy at minus the half step for determining
             * the full step kinetic energy and possibly for T-coupling.*/
            /* This may not be quite working correctly yet . . . . */
            compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                            wcycle, enerd, NULL, NULL, NULL, NULL, mu_tot,
                            constr, NULL, FALSE, state->box,
                            top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                            CGLO_RERUNMD | CGLO_GSTAT | CGLO_TEMPERATURE);
        }
        clear_mat(force_vir);

        GMX_MPE_LOG(ev_timestep2);

        /* We write a checkpoint at this MD step when:
         * either at an NS step when we signalled through gs,
         * or at the last step (but not when we do not want confout),
         * but never at the first step or with rerun.
         */
        bCPT = (((gs.set[eglsCHKPT] && (bNS || ir->nstlist == 0)) ||
                 (bLastStep && (Flags & MD_CONFOUT))) &&
                step > ir->init_step );
        if (bCPT)
        {
            gs.set[eglsCHKPT] = 0;
        }

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */
        bCalcEner = do_per_step(step, ir->nstcalcenergy);
        bCalcVir  = bCalcEner ||
                (ir->epc != epcNO && do_per_step(step, ir->nstpcouple));

        /* Do we need global communication ? */
        bGStat = (bCalcVir || bCalcEner || bStopCM ||
                  do_per_step(step, nstglobalcomm) ||
                  (ir->nstlist == -1  && step >= nlh.step_nscheck));

        do_ene = (do_per_step(step, ir->nstenergy) || bLastStep);

        if (do_ene || do_log)
        {
            bCalcVir  = TRUE;
            bCalcEner = TRUE;
            bGStat    = TRUE;
        }

        /* these CGLO_ options remain the same throughout the iteration */
        cglo_flags = (
                      (bGStat ? CGLO_GSTAT : 0)
                      );

        force_flags = (GMX_FORCE_STATECHANGED |
                       ((DYNAMIC_BOX(*ir)) ? GMX_FORCE_DYNAMICBOX : 0) |
                       GMX_FORCE_ALLFORCES |
                       GMX_FORCE_SEPLRF |
                       (bCalcVir ? GMX_FORCE_VIRIAL : 0) |
                       (bCalcEner ? GMX_FORCE_ENERGY : 0) |
                       (bDoFEP ? GMX_FORCE_DHDL : 0)
                       );

        if (fr->bTwinRange)
        {
            if (do_per_step(step, ir->nstcalclr))
            {
                force_flags |= GMX_FORCE_DO_LR;
            }
        }

            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too.
             * Check comments in sim_util.c
             */
            do_force(fplog, cr, ir, step, nrnb, wcycle, top, top_global, groups,
                     state->box, state->x, &state->hist,
                     f, force_vir, mdatoms, enerd, fcd,
                     state->lambda, graph,
                     fr, vsite, mu_tot, t, outf->fp_field, ed, bBornRadii,
                     (bNS ? GMX_FORCE_NS : 0) | force_flags);

        GMX_BARRIER(cr->mpi_comm_mygroup);




        /* ########  END FIRST UPDATE STEP  ############## */
        /* ########  If doing VV, we now have v(dt) ###### */
        if (bDoExpanded)
        {
            /* perform extended ensemble sampling in lambda - we don't
               actually move to the new state before outputting
               statistics, but if performing simulated tempering, we
               do update the velocities and the tau_t. */

            lamnew = ExpandedEnsembleDynamics(fplog, ir, enerd, state, &MassQ, &df_history, step, mcrng, state->v, mdatoms);
        }
        /* ################## START TRAJECTORY OUTPUT ################# */

        /* Now we have the energies and forces corresponding to the
         * coordinates at time t. We must output all of this before
         * the update.
         * for RerunMD t is read from input trajectory
         */
        GMX_MPE_LOG(ev_output_start);

        mdof_flags = 0;
        if (do_per_step(step, ir->nstxout))
        {
            mdof_flags |= MDOF_X;
        }
        if (do_per_step(step, ir->nstvout))
        {
            mdof_flags |= MDOF_V;
        }
        if (do_per_step(step, ir->nstfout))
        {
            mdof_flags |= MDOF_F;
        }
        if (do_per_step(step, ir->nstxtcout))
        {
            mdof_flags |= MDOF_XTC;
        }
        if (bCPT)
        {
            mdof_flags |= MDOF_CPT;
        }
        ;

#if defined(GMX_FAHCORE) || defined(GMX_WRITELASTSTEP)
        if (bLastStep)
        {
            /* Enforce writing positions and velocities at end of run */
            mdof_flags |= (MDOF_X | MDOF_V);
        }
#endif
#ifdef GMX_FAHCORE
        if (MASTER(cr))
        {
            fcReportProgress( ir->nsteps, step );
        }

        /* sync bCPT and fc record-keeping */
        if (bCPT && MASTER(cr))
        {
            fcRequestCheckPoint();
        }
#endif

        if (mdof_flags != 0)
        {
            wallcycle_start(wcycle, ewcTRAJ);
            if (bCPT)
            {
                if (state->flags & (1<<estLD_RNG))
                {
                    get_stochd_state(upd, state);
                }
                if (state->flags  & (1<<estMC_RNG))
                {
                    get_mc_state(mcrng, state);
                }
                if (MASTER(cr))
                {
                    if (bSumEkinhOld)
                    {
                        state_global->ekinstate.bUpToDate = FALSE;
                    }
                    else
                    {
                        update_ekinstate(&state_global->ekinstate, ekind);
                        state_global->ekinstate.bUpToDate = TRUE;
                    }
                    update_energyhistory(&state_global->enerhist, mdebin);
                    if (ir->efep != efepNO || ir->bSimTemp)
                    {
                        state_global->fep_state = state->fep_state; /* MRS: seems kludgy. The code should be
                                                                       structured so this isn't necessary.
                                                                       Note this reassignment is only necessary
                                                                       for single threads.*/
                        copy_df_history(&state_global->dfhist, &df_history);
                    }
                }
            }
            write_traj(fplog, cr, outf, mdof_flags, top_global,
                       step, t, state, state_global, f, f_global, &n_xtc, &x_xtc);
            if (bCPT)
            {
                nchkpt++;
                bCPT = FALSE;
            }
            debug_gmx();
            if (bLastStep && step_rel == ir->nsteps &&
                (Flags & MD_CONFOUT) && MASTER(cr)
                )
            {
                /* x and v have been collected in write_traj,
                 * because a checkpoint file will always be written
                 * at the last step.
                 */
                fprintf(stderr, "\nWriting final coordinates.\n");
                if (fr->bMolPBC)
                {
                    /* Make molecules whole only for confout writing */
                    do_pbc_mtop(fplog, ir->ePBC, state->box, top_global, state_global->x);
                }
                write_sto_conf_mtop(ftp2fn(efSTO, nfile, fnm),
                                    *top_global->name, top_global,
                                    state_global->x, state_global->v,
                                    ir->ePBC, state->box);
                debug_gmx();
            }
            wallcycle_stop(wcycle, ewcTRAJ);
        }
        GMX_MPE_LOG(ev_output_finish);

        /*  ################## END TRAJECTORY OUTPUT ################ */

        /* Determine the wallclock run time up till now */
        run_time = gmx_gettime() - (double)runtime->real;

        /* Check whether everything is still allright */
        if (((int)gmx_get_stop_condition() > handled_stop_condition)
#ifdef GMX_THREAD_MPI
            && MASTER(cr)
#endif
            )
        {
            /* this is just make gs.sig compatible with the hack
               of sending signals around by MPI_Reduce with together with
               other floats */
            if (gmx_get_stop_condition() == gmx_stop_cond_next_ns)
            {
                gs.sig[eglsSTOPCOND] = 1;
            }
            if (gmx_get_stop_condition() == gmx_stop_cond_next)
            {
                gs.sig[eglsSTOPCOND] = -1;
            }
            /* < 0 means stop at next step, > 0 means stop at next NS step */
            if (fplog)
            {
                fprintf(fplog,
                        "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                        gmx_get_signal_name(),
                        gs.sig[eglsSTOPCOND] == 1 ? "NS " : "");
                fflush(fplog);
            }
            fprintf(stderr,
                    "\n\nReceived the %s signal, stopping at the next %sstep\n\n",
                    gmx_get_signal_name(),
                    gs.sig[eglsSTOPCOND] == 1 ? "NS " : "");
            fflush(stderr);
            handled_stop_condition = (int)gmx_get_stop_condition();
        }
        else if (MASTER(cr) && (bNS || ir->nstlist <= 0) &&
                 (max_hours > 0 && run_time > max_hours*60.0*60.0*0.99) &&
                 gs.sig[eglsSTOPCOND] == 0 && gs.set[eglsSTOPCOND] == 0)
        {
            /* Signal to terminate the run */
            gs.sig[eglsSTOPCOND] = 1;
            if (fplog)
            {
                fprintf(fplog, "\nStep %s: Run time exceeded %.3f hours, will terminate the run\n", gmx_step_str(step, sbuf), max_hours*0.99);
            }
            fprintf(stderr, "\nStep %s: Run time exceeded %.3f hours, will terminate the run\n", gmx_step_str(step, sbuf), max_hours*0.99);
        }

        if (bResetCountersHalfMaxH && MASTER(cr) &&
            run_time > max_hours*60.0*60.0*0.495)
        {
            gs.sig[eglsRESETCOUNTERS] = 1;
        }

        if (ir->nstlist == -1 )
        {
            /* When bGStatEveryStep=FALSE, global_stat is only called
             * when we check the atom displacements, not at NS steps.
             * This means that also the bonded interaction count check is not
             * performed immediately after NS. Therefore a few MD steps could
             * be performed with missing interactions.
             * But wrong energies are never written to file,
             * since energies are only written after global_stat
             * has been called.
             */
            if (step >= nlh.step_nscheck)
            {
                nlh.nabnsb = natoms_beyond_ns_buffer(ir, fr, &top->cgs,
                                                     nlh.scale_tot, state->x);
            }
            else
            {
                /* This is not necessarily true,
                 * but step_nscheck is determined quite conservatively.
                 */
                nlh.nabnsb = 0;
            }
        }

        /* In parallel we only have to check for checkpointing in steps
         * where we do global communication,
         *  otherwise the other nodes don't know.
         */
        if (MASTER(cr) && ((bGStat || !PAR(cr)) &&
                           cpt_period >= 0 &&
                           (cpt_period == 0 ||
                            run_time >= nchkpt*cpt_period*60.0)) &&
            gs.set[eglsCHKPT] == 0)
        {
            gs.sig[eglsCHKPT] = 1;
        }


        if (bIterativeCase && do_per_step(step, ir->nstpcouple))
        {
            gmx_iterate_init(&iterate, TRUE);
            /* for iterations, we save these vectors, as we will be redoing the calculations */
            copy_coupling_state(state, bufstate, ekind, ekind_save, &(ir->opts));
        }

        bFirstIterate = TRUE;
        while (bFirstIterate || iterate.bIterationActive)
        {
            /* We now restore these vectors to redo the calculation with improved extended variables */
            if (iterate.bIterationActive)
            {
                copy_coupling_state(bufstate, state, ekind_save, ekind, &(ir->opts));
            }

            /* We make the decision to break or not -after- the calculation of Ekin and Pressure,
               so scroll down for that logic */

            /* #########   START SECOND UPDATE STEP ################# */
            GMX_MPE_LOG(ev_update_start);
            /* Box is changed in update() when we do pressure coupling,
             * but we should still use the old box for energy corrections and when
             * writing it to the energy file, so it matches the trajectory files for
             * the same timestep above. Make a copy in a separate array.
             */
            copy_mat(state->box, lastbox);

            bOK = TRUE;
                wallcycle_start(wcycle, ewcUPDATE);
                dvdl = 0;
                /* UPDATE PRESSURE VARIABLES IN TROTTER FORMULATION WITH CONSTRAINTS */
                if (bTrotter)
                {
                    if (iterate.bIterationActive)
                    {
                        if (bFirstIterate)
                        {
                            scalevir = 1;
                        }
                        else
                        {
                            /* we use a new value of scalevir to converge the iterations faster */
                            scalevir = tracevir/trace(shake_vir);
                        }
                        msmul(shake_vir, scalevir, shake_vir);
                        m_add(force_vir, shake_vir, total_vir);
                        clear_mat(shake_vir);
                    }
                    trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ3);
                    /* We can only do Berendsen coupling after we have summed
                     * the kinetic energy or virial. Since the happens
                     * in global_state after update, we should only do it at
                     * step % nstlist = 1 with bGStatEveryStep=FALSE.
                     */
                }
                else
                {
                    update_tcouple(fplog, step, ir, state, ekind, wcycle, upd, &MassQ, mdatoms);
                    update_pcouple(fplog, step, ir, state, pcoupl_mu, M, wcycle,
                                   upd, bInitStep);
                }

                /* Above, initialize just copies ekinh into ekin,
                 * it doesn't copy position (for VV),
                 * and entire integrator for MD.
                 */

                if (ir->eI == eiVVAK)
                {
                    copy_rvecn(state->x, cbuf, 0, state->natoms);
                }
                bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

                update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC, f,
                              bUpdateDoLR, fr->f_twin, fcd,
                              ekind, M, wcycle, upd, bInitStep, etrtPOSITION, cr, nrnb, constr, &top->idef);
                wallcycle_stop(wcycle, ewcUPDATE);

                update_constraints(fplog, step, &dvdl, ir, ekind, mdatoms, state,
                                   fr->bMolPBC, graph, f,
                                   &top->idef, shake_vir, force_vir,
                                   cr, nrnb, wcycle, upd, constr,
                                   bInitStep, FALSE, bCalcVir, state->veta);

                if (ir->eI == eiVVAK)
                {
                    /* erase F_EKIN and F_TEMP here? */
                    /* just compute the kinetic energy at the half step to perform a trotter step */
                    compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                    wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                    constr, NULL, FALSE, lastbox,
                                    top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                    cglo_flags | CGLO_TEMPERATURE
                                    );
                    wallcycle_start(wcycle, ewcUPDATE);
                    trotter_update(ir, step, ekind, enerd, state, total_vir, mdatoms, &MassQ, trotter_seq, ettTSEQ4);
                    /* now we know the scaling, we can compute the positions again again */
                    copy_rvecn(cbuf, state->x, 0, state->natoms);

                    bUpdateDoLR = (fr->bTwinRange && do_per_step(step, ir->nstcalclr));

                    update_coords(fplog, step, ir, mdatoms, state, fr->bMolPBC, f,
                                  bUpdateDoLR, fr->f_twin, fcd,
                                  ekind, M, wcycle, upd, bInitStep, etrtPOSITION, cr, nrnb, constr, &top->idef);
                    wallcycle_stop(wcycle, ewcUPDATE);

                    /* do we need an extra constraint here? just need to copy out of state->v to upd->xp? */
                    /* are the small terms in the shake_vir here due
                     * to numerical errors, or are they important
                     * physically? I'm thinking they are just errors, but not completely sure.
                     * For now, will call without actually constraining, constr=NULL*/
                    update_constraints(fplog, step, &dvdl, ir, ekind, mdatoms,
                                       state, fr->bMolPBC, graph, f,
                                       &top->idef, tmp_vir, force_vir,
                                       cr, nrnb, wcycle, upd, NULL,
                                       bInitStep, FALSE, bCalcVir,
                                       state->veta);
                }
                if (!bOK)
                {
                    gmx_fatal(FARGS, "Constraint error: Shake, Lincs or Settle could not solve the constrains");
                }

                if (fr->bSepDVDL && fplog && do_log)
                {
                    fprintf(fplog, sepdvdlformat, "Constraint dV/dl", 0.0, dvdl);
                }
                enerd->term[F_DVDL_BONDED] += dvdl;

            GMX_BARRIER(cr->mpi_comm_mygroup);
            GMX_MPE_LOG(ev_update_finish);



            /* ############## IF NOT VV, Calculate globals HERE, also iterate constraints  ############ */
            /* With Leap-Frog we can skip compute_globals at
             * non-communication steps, but we need to calculate
             * the kinetic energy one step before communication.
             */

            if (bGStat || ( do_per_step(step+1, nstglobalcomm)))
            {
                if (ir->nstlist == -1 && bFirstIterate)
                {
                    gs.sig[eglsNABNSB] = nlh.nabnsb;
                }
                compute_globals(fplog, gstat, cr, ir, fr, ekind, state, state_global, mdatoms, nrnb, vcm,
                                wcycle, enerd, force_vir, shake_vir, total_vir, pres, mu_tot,
                                constr,
                                bFirstIterate ? &gs : NULL,
                                (step_rel % gs.nstms == 0) &&
                                (multisim_nsteps < 0 || (step_rel < multisim_nsteps)),
                                lastbox,
                                top_global, &pcurr, top_global->natoms, &bSumEkinhOld,
                                cglo_flags
                                | CGLO_ENERGY
                                | (bStopCM ? CGLO_STOPCM : 0)
                                | CGLO_TEMPERATURE
                                | CGLO_PRESSURE
                                | (iterate.bIterationActive ? CGLO_ITERATE : 0)
                                | (bFirstIterate ? CGLO_FIRSTITERATE : 0)
                                | CGLO_CONSTRAINT
                                );
                if (ir->nstlist == -1 && bFirstIterate)
                {
                    nlh.nabnsb         = gs.set[eglsNABNSB];
                    gs.set[eglsNABNSB] = 0;
                }
            }
            /* bIterate is set to keep it from eliminating the old ekin kinetic energy terms */
            /* #############  END CALC EKIN AND PRESSURE ################# */

            /* Note: this is OK, but there are some numerical precision issues with using the convergence of
               the virial that should probably be addressed eventually. state->veta has better properies,
               but what we actually need entering the new cycle is the new shake_vir value. Ideally, we could
               generate the new shake_vir, but test the veta value for convergence.  This will take some thought. */

            if (iterate.bIterationActive &&
                done_iterating(cr, fplog, step, &iterate, bFirstIterate,
                               trace(shake_vir), &tracevir))
            {
                break;
            }
            bFirstIterate = FALSE;
        }

        /* only add constraint dvdl after constraints */
        enerd->term[F_DVDL_BONDED] += dvdl;
        /* sum up the foreign energy and dhdl terms for md and sd. currently done every step so that dhdl is correct in the .edr */
        sum_dhdl(enerd, state->lambda, ir->fepvals);
        update_box(fplog, step, ir, mdatoms, state, graph, f,
                   ir->nstlist == -1 ? &nlh.scale_tot : NULL, pcoupl_mu, nrnb, wcycle, upd, bInitStep, FALSE);

        /* ################# END UPDATE STEP 2 ################# */
        /* #### We now have r(t+dt) and v(t+dt/2)  ############# */

        if (!bGStat)
        {
            /* We will not sum ekinh_old,
             * so signal that we still have to do it.
             */
            bSumEkinhOld = TRUE;
        }

        

        /* #########  BEGIN PREPARING EDR OUTPUT  ###########  */

        /* use the directly determined last velocity, not actually the averaged half steps */
        if (bTrotter && ir->eI == eiVV)
        {
            enerd->term[F_EKIN] = last_ekin;
        }
        enerd->term[F_ETOT] = enerd->term[F_EPOT] + enerd->term[F_EKIN];

        enerd->term[F_ECONSERVED] = enerd->term[F_ETOT] + compute_conserved_from_auxiliary(ir, state, &MassQ);
        /* #########  END PREPARING EDR OUTPUT  ###########  */

        /* Time for performance */
        if (((step % stepout) == 0) || bLastStep)
        {
            runtime_upd_proc(runtime);
        }

        /* Output stuff */
        if (MASTER(cr))
        {
            gmx_bool do_dr, do_or;

            if (fplog && do_log && bDoExpanded)
            {
                /* only needed if doing expanded ensemble */
                PrintFreeEnergyInfoToFile(fplog, ir->fepvals, ir->expandedvals, ir->bSimTemp ? ir->simtempvals : NULL,
                                          &df_history, state->fep_state, ir->nstlog, step);
            }
            if (bCalcEner)
            {
                upd_mdebin(mdebin, bDoDHDL, TRUE,
                               t, mdatoms->tmass, enerd, state,
                               ir->fepvals, ir->expandedvals, lastbox,
                               shake_vir, force_vir, total_vir, pres,
                               ekind, mu_tot, constr);
            }
            else
            {
                upd_mdebin_step(mdebin);
            }

            do_dr  = do_per_step(step, ir->nstdisreout);
            do_or  = do_per_step(step, ir->nstorireout);

            print_ebin(outf->fp_ene, do_ene, do_dr, do_or, do_log ? fplog : NULL,
                           step, t,
                           eprNORMAL, bCompact, mdebin, fcd, groups, &(ir->opts));
            if (ir->ePull != epullNO)
            {
                pull_print_output(ir->pull, step, t);
            }

            if (do_per_step(step, ir->nstlog))
            {
                if (fflush(fplog) != 0)
                {
                    gmx_fatal(FARGS, "Cannot flush logfile - maybe you are out of disk space?");
                }
            }
        }
        if (bDoExpanded)
        {
            /* Have to do this part after outputting the logfile and the edr file */
            state->fep_state = lamnew;
            for (i = 0; i < efptNR; i++)
            {
                state_global->lambda[i] = ir->fepvals->all_lambda[i][lamnew];
            }
        }
        /* Remaining runtime */
        if (MULTIMASTER(cr) && (do_verbose || gmx_got_usr_signal()) )
        {
            print_time(stderr, runtime, step, ir, cr);
        }

        /* Replica exchange */
        bExchanged = FALSE;
        if ((repl_ex_nst > 0) && (step > 0) && !bLastStep &&
            do_per_step(step, repl_ex_nst))
        {
            bExchanged = replica_exchange(fplog, cr, repl_ex,
                                          state_global, enerd,
                                          state, step, t);

        }

        bFirstStep       = FALSE;
        bInitStep        = FALSE;
        bStartingFromCpt = FALSE;

        /* #######  SET VARIABLES FOR NEXT ITERATION IF THEY STILL NEED IT ###### */
        /* With all integrators, except VV, we need to retain the pressure
         * at the current step for coupling at the next step.
         */
        if ((state->flags & (1<<estPRES_PREV)) &&
            (bGStatEveryStep ||
             (ir->nstpcouple > 0 && step % ir->nstpcouple == 0)))
        {
            /* Store the pressure in t_state for pressure coupling
             * at the next MD step.
             */
            copy_mat(pres, state->pres_prev);
        }

        /* #######  END SET VARIABLES FOR NEXT ITERATION ###### */

            /* increase the MD step number */
            step++;
            step_rel++;

        cycles = wallcycle_stop(wcycle, ewcSTEP);

    }
    /* End of main MD loop */
    debug_gmx();

    /* Stop the time */
    runtime_end(runtime);


    if (!(cr->duty & DUTY_PME))
    {
        /* Tell the PME only node to finish */
        gmx_pme_send_finish(cr);
    }

    if (MASTER(cr))
    {
        if (ir->nstcalcenergy > 0 )
        {
            print_ebin(outf->fp_ene, FALSE, FALSE, FALSE, fplog, step, t,
                       eprAVER, FALSE, mdebin, fcd, groups, &(ir->opts));
        }
    }

    done_mdoutf(outf);

    debug_gmx();

    if (ir->nstlist == -1 && nlh.nns > 0 && fplog)
    {
        fprintf(fplog, "Average neighborlist lifetime: %.1f steps, std.dev.: %.1f steps\n", nlh.s1/nlh.nns, sqrt(nlh.s2/nlh.nns - sqr(nlh.s1/nlh.nns)));
        fprintf(fplog, "Average number of atoms that crossed the half buffer length: %.1f\n\n", nlh.ab/nlh.nns);
    }

    if (pme_loadbal != NULL)
    {
        pme_loadbal_done(pme_loadbal, fplog);
    }


    if (repl_ex_nst > 0 && MASTER(cr))
    {
        print_replica_exchange_statistics(fplog, repl_ex);
    }

    runtime->nsteps_done = step_rel;

    return 0;
}
