#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "macros.h"
#include "maths.h"
#include "vec.h"
#include "nsgrid.h"
#include "force.h"
#include "nonbonded.h"
#include "ns.h"
#include "pbc.h"
#include "names.h"
#include "gmx_fatal.h"
#include "txtdump.h"
#include "mtop_util.h"

#include "types/simple.h"
#include "typedefs.h"

typedef void
    put_in_list_t (gmx_bool              bHaveVdW[],
                   int                   ngid,
                   t_mdatoms     *       md,
                   int                   icg,
                   int                   jgid,
                   int                   nj,
                   atom_id               jjcg[],
                   atom_id               index[],
                   t_excl                bExcl[],
                   int                   shift,
                   t_forcerec     *      fr,
                   gmx_bool              bLR,
                   gmx_bool              bDoVdW,
                   gmx_bool              bDoCoul,
                   int                   solvent_opt);


/****************************************************
 *
 *    F A S T   N E I G H B O R  S E A R C H I N G
 *
 *    Optimized neighboursearching routine using grid
 *    at least 1x1x1, see GROMACS manual
 *
 ****************************************************/


static void get_cutoff2(t_forcerec *fr, gmx_bool bDoLongRange,
                        real *rvdw2, real *rcoul2,
                        real *rs2, real *rm2, real *rl2)
{ // called
    *rs2 = (fr->rlist)*(fr->rlist);
    *rvdw2  = *rs2;
    *rcoul2 = *rs2;
    *rm2 = *rs2; 
    *rl2 = *rs2;
}

static void init_nsgrid_lists(t_forcerec *fr, int ngid, gmx_ns_t *ns)
{ // called
    real rvdw2, rcoul2, rs2, rm2, rl2;
    int  j;

    get_cutoff2(fr, TRUE, &rvdw2, &rcoul2, &rs2, &rm2, &rl2);

    /* Short range buffers */
    snew(ns->nl_sr, ngid);
    /* Counters */
    snew(ns->nsr, ngid);
    snew(ns->nlr_ljc, ngid);
    snew(ns->nlr_one, ngid);

    /* Always allocate both list types, since rcoulomb might now change with PME load balancing */
    /* Long range VdW and Coul buffers */
    snew(ns->nl_lr_ljc, ngid);
    /* Long range VdW or Coul only buffers */
    snew(ns->nl_lr_one, ngid);
    snew(ns->nl_sr[0], MAX_CG);
    snew(ns->nl_lr_ljc[0], MAX_CG);
    snew(ns->nl_lr_one[0], MAX_CG);
}


void ns_realloc_natoms(gmx_ns_t *ns, int natoms)
{ // called
    int i;

    ns->nra_alloc = over_alloc_dd(natoms);
    srenew(ns->bexcl, ns->nra_alloc);
    for (i = 0; i < ns->nra_alloc; i++) // 3000
    {
        ns->bexcl[i] = 0;
    }
}

void init_ns(FILE *fplog, const t_commrec *cr,
             gmx_ns_t *ns, t_forcerec *fr,
             const gmx_mtop_t *mtop,
             matrix box)
{ // called
    int  mt, icg, nr_in_cg, jcg, ngid, ncg;
    t_block *cgs;
    char *ptr;

    /* Compute largest charge groups size (# atoms) */
    nr_in_cg = 1;
    cgs = &mtop->moltype[0].cgs;
    for (icg = 0; (icg < cgs->nr); icg++) // cgs->nr = 3
    {
           nr_in_cg = max(nr_in_cg, (int)(cgs->index[icg+1]-cgs->index[icg]));
    }

    /* Verify whether largest charge group is <= max cg.
     * This is determined by the type of the local exclusion type
     * Exclusions are stored in bits. (If the type is not large
     * enough, enlarge it, unsigned char -> unsigned short -> unsigned long)
     */

    ngid = mtop->groups.grps[egcENER].nr;
    snew(ns->bExcludeAlleg, ngid);

    ns->bExcludeAlleg[0] = FALSE;
    /* Grid search */
    snew(ns->grid, 1);
    ns->grid->npbcdim = 3;
    /* The ideal number of cg's per ns grid cell seems to be 10 */
    ns->grid->ncg_ideal = 10;


    init_nsgrid_lists(fr, ngid, ns);

    /* Create array that determines whether or not atoms have VdW */
    snew(ns->bHaveVdW, fr->ntype);
    ns->bHaveVdW[0] = FALSE;

    ns->nra_alloc = 0;
    ns->bexcl     = NULL;
    /* This could be reduced with particle decomposition */
    ns_realloc_natoms(ns, mtop->natoms);

    ns->nblist_initialized = FALSE;

    /* nbr list debug dump */
    ns->dump_nl = 0;
}


