#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include "typedefs.h"
#include "tpxio.h"
#include "smalloc.h"
#include "vec.h"
#include "main.h"
#include "mvdata.h"
#include "gmx_fatal.h"
#include "symtab.h"
#include "txtdump.h"
#include "mdatoms.h"
#include "mdrun.h"
#include "statutil.h"
#include "names.h"
#include "calcgrid.h"
#include "gmx_random.h"
#include "mdebin.h"

#define BUFSIZE 256

#define NOT_FINISHED(l1, l2) \
    printf("not finished yet: lines %d .. %d in %s\n", l1, l2, __FILE__)


void set_state_entries(t_state *state, const t_inputrec *ir, int nnodes)
{ // called
    int nnhpres;

    /* The entries in the state in the tpx file might not correspond
     * with what is needed, so we correct this here.
     */
    state->flags = 0;
    state->flags |= (1<<estX);
    state->flags |= (1<<estV);
    state->nrng  = gmx_rng_n();
    state->nrngi = 1;
    state->flags |= ((1<<estLD_RNG) | (1<<estLD_RNGI));
    snew(state->ld_rng, state->nrng);
    snew(state->ld_rngi, state->nrngi);

    state->nnhpres = 0;
    state->flags |= (1<<estBOX);
    state->flags |= (1<<estTC_INT);

    init_gtc_state(state, state->ngtc, state->nnhpres, ir->opts.nhchainlength); /* allocate the space for nose-hoover chains */

    init_energyhistory(&state->enerhist);
    init_df_history(&state->dfhist, ir->fepvals->n_lambda, ir->expandedvals->init_wl_delta);
}


