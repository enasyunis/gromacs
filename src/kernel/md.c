#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "statutil.h"
#include "mdebin.h"
#include "nrnb.h"
#include "index.h"
#include "ns.h"
#include "trnio.h"
#include "mdrun.h"
#include "md_support.h"
#include "md_logging.h"
#include "network.h"
#include "pull.h"
#include "physics.h"
#include "names.h"
#include "disre.h"
#include "orires.h"
#include "pme.h"
#include "mdatoms.h"
#include "qmmm.h"
#include "mpelogging.h"
#include "topsort.h"
#include "coulomb.h"
#include "mtop_util.h"
#include "sighandler.h"
#include "txtdump.h"
#include "string2.h"
#include "bondf.h"
#include "types/membedt.h"
#include "types/nlistheuristics.h"
#include "types/iteratedconstraints.h"

#include "tmpi.h"



double do_md(FILE *fplog, t_commrec *cr, int nfile, const t_filenm fnm[],
             t_inputrec *ir,
             gmx_mtop_t *top_global,
             t_state *state_global,
             t_mdatoms *mdatoms,
             t_forcerec *fr,
             unsigned long Flags
              )
{


    gmx_large_int_t step, step_rel;
    double          t, t0, lam0[efptNR];
    int               force_flags;
    int               i, m;
    t_trxstatus      *status;
    gmx_localtop_t   *top;
    t_state          *state    = NULL;
    gmx_enerdata_t   *enerd;
    rvec             *f = NULL;
    globsig_t         gs;
    int               a0, a1, gnx = 0, ii;
    /* for FEP */
    char              sbuf[STEPSTRSIZE];

    /* Initial values */
    t = t0       = ir->init_t;

    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda,
                  enerd);
        snew(f, top_global->natoms);




    debug_gmx();



    top = gmx_mtop_generate_local_top(top_global, ir);

    a0 = 0;
    a1 = top_global->natoms;

    forcerec_set_excl_load(fr, top, cr);

    // initialize the local state
    snew(state, 1);
    /* Copy all the contents */
    *state = *state_global;
    snew(state->lambda, efptNR);

    /* local storage for lambda */
    for (i = 0; i < efptNR; i++) // 7
    {
        state->lambda[i] = state_global->lambda[i];
    }


    atoms2md(top_global, ir, 0, NULL, a0, a1-a0, mdatoms);

    mdatoms->tmass = mdatoms->tmassA; // total system mass 3024.0


    debug_gmx();


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


    step     = ir->init_step;
    step_rel = 0;






        GMX_MPE_LOG(ev_timestep1);

            t         = t0 + step*ir->delta_t;








        GMX_MPE_LOG(ev_timestep2);

        /* We write a checkpoint at this MD step when:
         * either at an NS step when we signalled through gs,
         * or at the last step (but not when we do not want confout),
         * but never at the first step or with rerun.
         */


        /* these CGLO_ options remain the same throughout the iteration */
        force_flags = (
                       GMX_FORCE_ALLFORCES |
                       GMX_FORCE_SEPLRF 
                       );


            /* The coordinates (x) are shifted (to get whole molecules)
             * in do_force.
             * This is parallellized as well, and does communication too.
             * Check comments in sim_util.c
             */ 
            do_force(fplog, cr, ir, step, top, top_global, &top_global->groups, 
                     state->box, state->x, &state->hist,
                     f, mdatoms, enerd, 
                     state->lambda, 
                     fr, t,  
                     force_flags);

        GMX_BARRIER(cr->mpi_comm_mygroup);


    /* End of main MD loop */
    debug_gmx();

    /* Stop the time */


    debug_gmx();



    return 0;
}
