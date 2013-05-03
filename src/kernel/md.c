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
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "domdec.h"
#include "domdec_network.h"
#include "partdec.h"
#include "topsort.h"
#include "coulomb.h"
#include "constr.h"
#include "mvdata.h"
#include "checkpoint.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "txtdump.h"
#include "string2.h"
#include "bondf.h"
#include "types/membedt.h"
#include "types/nlistheuristics.h"
#include "types/iteratedconstraints.h"
#include "nbnxn_cuda_data_mgmt.h"

#include "tmpi.h"



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
    gmx_bool        bCalcVir;
    gmx_bool        bSimAnn, bStopCM, bNotLastFrame = FALSE;
    gmx_bool        bFirstStep, bStateFromTPX, bInitStep, bLastStep;
    gmx_bool        bBornRadii, bStartingFromCpt;
    gmx_bool          bDoDHDL = FALSE, bDoFEP = FALSE;
    gmx_bool          do_ene, do_verbose, bRerunWarnNoV = TRUE,
                      bForceUpdate = FALSE;
    int               mdof_flags;
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
    int               nchkpt  = 1;
    gmx_localtop_t   *top;
    t_mdebin         *mdebin = NULL;
    t_state          *state    = NULL;
    rvec             *f_global = NULL;
    int               n_xtc    = -1;
    rvec             *x_xtc    = NULL;
    gmx_enerdata_t   *enerd;
    rvec             *f = NULL;
    gmx_global_stat_t gstat;
    t_graph          *graph = NULL;
    globsig_t         gs;
    gmx_rng_t         mcrng = NULL;
    gmx_groups_t     *groups;
    int               count, nconverged = 0;
    real              timestep = 0;
    double            tcount   = 0;
    gmx_bool          bConverged = TRUE, bOK, bSumEkinhOld;
    gmx_bool          bResetCountersHalfMaxH = FALSE;
    gmx_bool          bTemp, bPres;
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
    char              sbuf[STEPSTRSIZE], sbuf2[STEPSTRSIZE];
    int               handled_stop_condition = gmx_stop_cond_none; /* compare to get_stop_condition*/
    gmx_iterate_t     iterate;
    gmx_large_int_t   multisim_nsteps = -1;                        /* number of steps to do  before first multisim
                                                                      simulation stops. If equal to zero, don't
                                                                      communicate any more between multisims.*/
    /* PME load balancing data for GPU kernels */
    double               cycles_pmes;

    /* md-vv uses averaged full step velocities for T-control
       md-vv-avek uses averaged half step velocities for T-control (but full step ekin for P control)
       md uses averaged half step kinetic energies to determine temperature unless defined otherwise by GMX_EKIN_AVE_VEL; */
    /* all the iteratative cases - only if there are constraints */



    /* The default value of iterate->bIterationActive is set to
       false in this step.  The correct value, true or false,
       is set at each step, as it depends on the frequency of temperature
       and pressure control.*/
    iterate.iter_i           = 0;
    iterate.bIterationActive = FALSE;
    iterate.num_close        = 0;
    for (i = 0; i < MAXITERCONST+2; i++)
    {
        iterate.allrelerr[i] = 0;
    }


    nstglobalcomm   = ir->nstlist;


    groups = &top_global->groups;

    /* Initial values */
    init_md(fplog, cr, ir, oenv, &t, &t0, state_global->lambda,
            &(state_global->fep_state), lam0,
            nrnb, top_global, 
            nfile, fnm, &outf, &mdebin,
            force_vir, shake_vir, mu_tot, &bSimAnn, &vcm, state_global, Flags);

    clear_mat(total_vir);
    clear_mat(pres);
    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda,
                  enerd);
        snew(f, top_global->natoms);




    gstat = global_stat_init(ir);
    debug_gmx();



    top = gmx_mtop_generate_local_top(top_global, ir);

    a0 = 0;
    a1 = top_global->natoms;

    forcerec_set_excl_load(fr, top, cr);

    state    = partdec_init_local_state(cr, state_global);
    f_global = f;

    atoms2md(top_global, ir, 0, NULL, a0, a1-a0, mdatoms);



    update_mdatoms(mdatoms, state->lambda[efptMASS]);





    debug_gmx();

    /* set free energy calculation frequency as the minimum of nstdhdl, nstexpanded, and nstrepl_ex_nst*/
    nstfep = ir->fepvals->nstdhdl;

    bSumEkinhOld = FALSE;



    enerd->term[F_TEMP] *= 2; /* result of averages being done over previous and current step,
                                     and there is no previous step */


    {
        char tbuf[20];
        fprintf(stderr, "starting mdrun '%s'\n",
                    *(top_global->name));

        sprintf(tbuf, "%8.1f", (ir->init_step+ir->nsteps)*ir->delta_t);


        fprintf(stderr, "%s steps, %s ps.\n",
                        gmx_step_str(ir->nsteps, sbuf), tbuf);
    }


    debug_gmx();


    /* loop over MD steps or if rerunMD to end of input trajectory */
    bFirstStep = TRUE;
    /* Skip the first Nose-Hoover integration when we get the state from tpx */
    bStateFromTPX    = TRUE;
    bInitStep        = bFirstStep && (bStateFromTPX);
    bStartingFromCpt = (Flags & MD_STARTFROMCPT) && bInitStep;
    bLastStep        = FALSE;
    bSumEkinhOld     = FALSE;


    step     = ir->init_step;
    step_rel = 0;



    bLastStep = ((ir->nsteps >= 0 && step_rel > ir->nsteps) ||
                 ((multisim_nsteps >= 0) && (step_rel >= multisim_nsteps )));



        GMX_MPE_LOG(ev_timestep1);

            bLastStep = (step_rel == ir->nsteps);
            t         = t0 + step*ir->delta_t;



        /* Stop Center of Mass motion */
        bStopCM = (ir->comm_mode != ecmNO && do_per_step(step, ir->nstcomm));


        /* Determine whether or not to update the Born radii if doing GB */
        bBornRadii = TRUE;

        do_verbose = bVerbose &&
            (step % stepout == 0 || bFirstStep || bLastStep);


        clear_mat(force_vir);

        GMX_MPE_LOG(ev_timestep2);

        /* We write a checkpoint at this MD step when:
         * either at an NS step when we signalled through gs,
         * or at the last step (but not when we do not want confout),
         * but never at the first step or with rerun.
         */

        /* Determine the energy and pressure:
         * at nstcalcenergy steps and at energy output steps (set below).
         */
        bCalcVir  = (do_per_step(step, ir->nstcalcenergy)) ||
                (ir->epc != epcNO && do_per_step(step, ir->nstpcouple));


        do_ene = (do_per_step(step, ir->nstenergy) || bLastStep);

        bCalcVir  = TRUE;

        /* these CGLO_ options remain the same throughout the iteration */
        cglo_flags = (
                      CGLO_GSTAT 
                      );
        force_flags = (
                       GMX_FORCE_ALLFORCES |
                       GMX_FORCE_SEPLRF 
                       );


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
                     /*GMX_FORCE_NS |*/ force_flags);

        GMX_BARRIER(cr->mpi_comm_mygroup);


    /* End of main MD loop */
    debug_gmx();

    /* Stop the time */

    done_mdoutf(outf);

    debug_gmx();



    return 0;
}
