#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "mdatoms.h"
#include "smalloc.h"
#include "main.h"
#include "qmmm.h"
#include "mtop_util.h"
#include "gmx_omp_nthreads.h"

#define ALMOST_ZERO 1e-30

t_mdatoms *init_mdatoms(FILE *fp, gmx_mtop_t *mtop, gmx_bool bFreeEnergy)
{ 
    int                     mb, a, g, nmol;
    double                  tmA, tmB;
    t_atom                 *atom;
    t_mdatoms              *md;
    gmx_mtop_atomloop_all_t aloop;
    t_ilist                *ilist;

    snew(md, 1);

    md->nenergrp = mtop->groups.grps[egcENER].nr;
    md->bVCMgrps = FALSE;
    tmA          = 0.0;
    tmB          = 0.0;

    aloop = gmx_mtop_atomloop_all_init(mtop);
    while (gmx_mtop_atomloop_all_next(aloop, &a, &atom))
    {
        tmA += atom->m;
        tmB += atom->mB;
    } //Final result tmA 3.024000e+03, tmB 3.024000e+03

    md->tmassA = tmA;
    md->tmassB = tmB;
    md->bOrires = gmx_mtop_ftype_count(mtop, F_ORIRES);

    return md;
}

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

void update_mdatoms(t_mdatoms *md, real lambda)
{ 
    md->tmass = md->tmassA;
    md->lambda = lambda;
}
