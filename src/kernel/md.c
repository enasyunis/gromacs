#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "smalloc.h"
#include "sysstuff.h"
#include "vec.h"
#include "ns.h"
#include "mdrun.h"
#include "md_logging.h"
#include "physics.h"
#include "names.h"
#include "pme.h"
#include "mdatoms.h"
#include "topsort.h"
#include "coulomb.h"
#include "mtop_util.h"
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
    int               force_flags;
    gmx_localtop_t   *top;
    t_state          *state    = NULL;
    gmx_enerdata_t   *enerd;
    rvec             *f = NULL;

    /* Energy terms and groups */
    snew(enerd, 1);
    init_enerdata(top_global->groups.grps[egcENER].nr, ir->fepvals->n_lambda, enerd);
    snew(f, top_global->natoms);

    top = gmx_mtop_generate_local_top(top_global, ir);

    // initialize the local state
    snew(state, 1);

    /* Copy all the contents */
    *state = *state_global; // global lambda's are all zeros
    snew(state->lambda, efptNR); // automatically all 7 lambda's initialized to 0.0

    atoms2md(top_global, ir, 0, NULL, 0, top_global->natoms, mdatoms);

    mdatoms->tmass = mdatoms->tmassA; // total system mass 3024.0

    // used to print to screen:: starting mdrun 'Water q only'

    /* Expands to: (GMX_FORCE_BONDED | GMX_FORCE_NONBONDED | GMX_FORCE_FORCES | GMX_FORCE_SEPLRF) */
    force_flags = (GMX_FORCE_ALLFORCES | GMX_FORCE_SEPLRF);


    /* The coordinates (x) are shifted (to get whole molecules)
     * in do_force.
     * This is parallellized as well, and does communication too.
     * Check comments in sim_util.c
     */ 
    do_force(fplog, cr, ir, ir->init_step, top, top_global, &top_global->groups, 
            state->box, state->x, &state->hist,
            f, mdatoms, enerd, 
            state->lambda, 
            fr, ir->init_t,  
            force_flags);


    return 0;
}
