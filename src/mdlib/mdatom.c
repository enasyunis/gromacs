#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "mdatoms.h"
#include "smalloc.h"
#include "qmmm.h"
#include "mtop_util.h"
#include "gmx_omp_nthreads.h"

#define ALMOST_ZERO 1e-30

void atoms2md(gmx_mtop_t *mtop, t_inputrec *ir,
              int nindex, int *index,
              int start, int homenr,
              t_mdatoms *md)
{ 
    gmx_mtop_atomlookup_t alook;
    int                   i;
    t_grpopts            *opts;
    gmx_groups_t         *groups;
    gmx_molblock_t       *molblock;

    opts = &ir->opts;

    groups = &mtop->groups;

    molblock = mtop->molblock;

    md->nr = mtop->natoms;

    md->nalloc = over_alloc_dd(md->nr);

    srenew(md->massT, md->nalloc);
    srenew(md->invmass, md->nalloc);
    srenew(md->chargeA, md->nalloc);
    srenew(md->typeA, md->nalloc);
    srenew(md->ptype, md->nalloc);
    srenew(md->cENER, md->nalloc);

    alook = gmx_mtop_atomlookup_init(mtop);

#pragma omp parallel for num_threads(gmx_omp_nthreads_get(emntDefault)) schedule(static)
    for (i = 0; i < md->nr; i++)
    {
        int      g, ag, molb;
        real     mA, mB, fac;
        t_atom  *atom;

        ag = i;
        gmx_mtop_atomnr_to_atom(alook, ag, &atom);

        mA = atom->m;
        mB = atom->mB;
        md->massT[i]    = mA;
        md->invmass[i]    = 1.0/mA;
        md->chargeA[i]  = atom->q;
        md->typeA[i]    = atom->type;
        md->ptype[i]    = atom->ptype;
        md->cENER[i]    =
            (groups->grpnr[egcENER] ? groups->grpnr[egcENER][ag] : 0);

    } 

    gmx_mtop_atomlookup_destroy(alook);

    md->start  = start;
    md->homenr = homenr;
    md->lambda = 0;
}

