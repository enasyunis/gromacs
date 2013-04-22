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

#include <math.h>
#include <string.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "macros.h"
#include "maths.h"
#include "vec.h"
#include "network.h"
#include "nsgrid.h"
#include "force.h"
#include "nonbonded.h"
#include "ns.h"
#include "pbc.h"
#include "names.h"
#include "gmx_fatal.h"
#include "nrnb.h"
#include "txtdump.h"
#include "mtop_util.h"

#include "domdec.h"
#include "adress.h"


/*
 *    E X C L U S I O N   H A N D L I N G
 */

#ifdef DEBUG
static void SETEXCL_(t_excl e[], atom_id i, atom_id j)
{
    e[j] = e[j] | (1<<i);
}
static void RMEXCL_(t_excl e[], atom_id i, atom_id j)
{
    e[j] = e[j] & ~(1<<i);
}
static gmx_bool ISEXCL_(t_excl e[], atom_id i, atom_id j)
{
    return (gmx_bool)(e[j] & (1<<i));
}
static gmx_bool NOTEXCL_(t_excl e[], atom_id i, atom_id j)
{
    return !(ISEXCL(e, i, j));
}
#else
#define SETEXCL(e, i, j) (e)[((atom_id) (j))] |= (1<<((atom_id) (i)))
#define RMEXCL(e, i, j)  (e)[((atom_id) (j))] &= (~(1<<((atom_id) (i))))
#define ISEXCL(e, i, j)  (gmx_bool) ((e)[((atom_id) (j))] & (1<<((atom_id) (i))))
#define NOTEXCL(e, i, j) !(ISEXCL(e, i, j))
#endif

static int
round_up_to_simd_width(int length, int simd_width)
{
    int offset, newlength;

    offset = (simd_width > 0) ? length % simd_width : 0;

    return (offset == 0) ? length : length-offset+simd_width;
}
/************************************************
 *
 *  U T I L I T I E S    F O R    N S
 *
 ************************************************/

static void reallocate_nblist(t_nblist *nl)
{
    if (gmx_debug_at)
    {
        fprintf(debug, "reallocating neigborlist (ielec=%d, ivdw=%d, igeometry=%d, type=%d), maxnri=%d\n",
                nl->ielec, nl->ivdw, nl->igeometry, nl->type, nl->maxnri);
    }
    srenew(nl->iinr,   nl->maxnri);
    if (nl->igeometry == GMX_NBLIST_GEOMETRY_CG_CG)
    {
        srenew(nl->iinr_end, nl->maxnri);
    }
    srenew(nl->gid,    nl->maxnri);
    srenew(nl->shift,  nl->maxnri);
    srenew(nl->jindex, nl->maxnri+1);
}


static void init_nblist(FILE *log, t_nblist *nl_sr, t_nblist *nl_lr,
                        int maxsr, int maxlr,
                        int ivdw, int ivdwmod,
                        int ielec, int ielecmod,
                        int igeometry, int type)
{
    t_nblist *nl;
    int       homenr;
    int       i, nn;

    for (i = 0; (i < 2); i++)
    {
        nl     = (i == 0) ? nl_sr : nl_lr;
        homenr = (i == 0) ? maxsr : maxlr;

        if (nl == NULL)
        {
            continue;
        }


        /* Set coul/vdw in neighborlist, and for the normal loops we determine
         * an index of which one to call.
         */
        nl->ivdw        = ivdw;
        nl->ivdwmod     = ivdwmod;
        nl->ielec       = ielec;
        nl->ielecmod    = ielecmod;
        nl->type        = type;
        nl->igeometry   = igeometry;

        if (nl->type == GMX_NBLIST_INTERACTION_FREE_ENERGY)
        {
            nl->igeometry  = GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE;
        }

        /* This will also set the simd_padding_width field */
        gmx_nonbonded_set_kernel_pointers( (i == 0) ? log : NULL, nl);

        /* maxnri is influenced by the number of shifts (maximum is 8)
         * and the number of energy groups.
         * If it is not enough, nl memory will be reallocated during the run.
         * 4 seems to be a reasonable factor, which only causes reallocation
         * during runs with tiny and many energygroups.
         */
        nl->maxnri      = homenr*4;
        nl->maxnrj      = 0;
        nl->maxlen      = 0;
        nl->nri         = -1;
        nl->nrj         = 0;
        nl->iinr        = NULL;
        nl->gid         = NULL;
        nl->shift       = NULL;
        nl->jindex      = NULL;
        reallocate_nblist(nl);
        nl->jindex[0] = 0;

        if (debug)
        {
            fprintf(debug, "Initiating neighbourlist (ielec=%d, ivdw=%d, type=%d) for %s interactions,\nwith %d SR, %d LR atoms.\n",
                    nl->ielec, nl->ivdw, nl->type, gmx_nblist_geometry_names[nl->igeometry], maxsr, maxlr);
        }
    }
}

void init_neighbor_list(FILE *log, t_forcerec *fr, int homenr)
{
    /* Make maxlr tunable! (does not seem to be a big difference though)
     * This parameter determines the number of i particles in a long range
     * neighbourlist. Too few means many function calls, too many means
     * cache trashing.
     */
    int        maxsr, maxsr_wat, maxlr, maxlr_wat;
    int        ielec, ielecf, ivdw, ielecmod, ielecmodf, ivdwmod, type;
    int        solvent;
    int        igeometry_def, igeometry_w, igeometry_ww;
    int        i;
    t_nblists *nbl;

    /* maxsr     = homenr-fr->nWatMol*3; */
    maxsr     = homenr;

    if (maxsr < 0)
    {
        gmx_fatal(FARGS, "%s, %d: Negative number of short range atoms.\n"
                  "Call your Gromacs dealer for assistance.", __FILE__, __LINE__);
    }
    /* This is just for initial allocation, so we do not reallocate
     * all the nlist arrays many times in a row.
     * The numbers seem very accurate, but they are uncritical.
     */
    maxsr_wat = min(fr->nWatMol, (homenr+2)/3);
    if (fr->bTwinRange)
    {
        maxlr     = 50;
        maxlr_wat = min(maxsr_wat, maxlr);
    }
    else
    {
        maxlr = maxlr_wat = 0;
    }

    /* Determine the values for ielec/ivdw. */
    ielec    = fr->nbkernel_elec_interaction;
    ivdw     = fr->nbkernel_vdw_interaction;
    ielecmod = fr->nbkernel_elec_modifier;
    ivdwmod  = fr->nbkernel_vdw_modifier;
    type     = GMX_NBLIST_INTERACTION_STANDARD;

    fr->ns.bCGlist = (getenv("GMX_NBLISTCG") != 0);
    if (!fr->ns.bCGlist)
    {
        igeometry_def = GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE;
    }
    else
    {
        igeometry_def = GMX_NBLIST_GEOMETRY_CG_CG;
        if (log != NULL)
        {
            fprintf(log, "\nUsing charge-group - charge-group neighbor lists and kernels\n\n");
        }
    }

    if (fr->solvent_opt == esolTIP4P)
    {
        igeometry_w  = GMX_NBLIST_GEOMETRY_WATER4_PARTICLE;
        igeometry_ww = GMX_NBLIST_GEOMETRY_WATER4_WATER4;
    }
    else
    {
        igeometry_w  = GMX_NBLIST_GEOMETRY_WATER3_PARTICLE;
        igeometry_ww = GMX_NBLIST_GEOMETRY_WATER3_WATER3;
    }

    for (i = 0; i < fr->nnblists; i++)
    {
        nbl = &(fr->nblists[i]);

        if ((fr->adress_type != eAdressOff) && (i >= fr->nnblists/2))
        {
            type = GMX_NBLIST_INTERACTION_ADRESS;
        }
        init_nblist(log, &nbl->nlist_sr[eNL_VDWQQ], &nbl->nlist_lr[eNL_VDWQQ],
                    maxsr, maxlr, ivdw, ivdwmod, ielec, ielecmod, igeometry_def, type);
        init_nblist(log, &nbl->nlist_sr[eNL_VDW], &nbl->nlist_lr[eNL_VDW],
                    maxsr, maxlr, ivdw, ivdwmod, GMX_NBKERNEL_ELEC_NONE, eintmodNONE, igeometry_def, type);
        init_nblist(log, &nbl->nlist_sr[eNL_QQ], &nbl->nlist_lr[eNL_QQ],
                    maxsr, maxlr, GMX_NBKERNEL_VDW_NONE, eintmodNONE, ielec, ielecmod, igeometry_def, type);
        init_nblist(log, &nbl->nlist_sr[eNL_VDWQQ_WATER], &nbl->nlist_lr[eNL_VDWQQ_WATER],
                    maxsr_wat, maxlr_wat, ivdw, ivdwmod, ielec, ielecmod, igeometry_w, type);
        init_nblist(log, &nbl->nlist_sr[eNL_QQ_WATER], &nbl->nlist_lr[eNL_QQ_WATER],
                    maxsr_wat, maxlr_wat, GMX_NBKERNEL_VDW_NONE, eintmodNONE, ielec, ielecmod, igeometry_w, type);
        init_nblist(log, &nbl->nlist_sr[eNL_VDWQQ_WATERWATER], &nbl->nlist_lr[eNL_VDWQQ_WATERWATER],
                    maxsr_wat, maxlr_wat, ivdw, ivdwmod, ielec, ielecmod, igeometry_ww, type);
        init_nblist(log, &nbl->nlist_sr[eNL_QQ_WATERWATER], &nbl->nlist_lr[eNL_QQ_WATERWATER],
                    maxsr_wat, maxlr_wat, GMX_NBKERNEL_VDW_NONE, eintmodNONE, ielec, ielecmod, igeometry_ww, type);

        /* Did we get the solvent loops so we can use optimized water kernels? */
        if (nbl->nlist_sr[eNL_VDWQQ_WATER].kernelptr_vf == NULL
            || nbl->nlist_sr[eNL_QQ_WATER].kernelptr_vf == NULL
#ifndef DISABLE_WATERWATER_NLIST
            || nbl->nlist_sr[eNL_VDWQQ_WATERWATER].kernelptr_vf == NULL
            || nbl->nlist_sr[eNL_QQ_WATERWATER].kernelptr_vf == NULL
#endif
            )
        {
            fr->solvent_opt = esolNO;
            fprintf(log, "Note: The available nonbonded kernels do not support water optimization - disabling.\n");
        }

        if (fr->efep != efepNO)
        {
            if ((fr->bEwald) && (fr->sc_alphacoul > 0)) /* need to handle long range differently if using softcore */
            {
                ielecf    = GMX_NBKERNEL_ELEC_EWALD;
                ielecmodf = eintmodNONE;
            }
            else
            {
                ielecf    = ielec;
                ielecmodf = ielecmod;
            }

            init_nblist(log, &nbl->nlist_sr[eNL_VDWQQ_FREE], &nbl->nlist_lr[eNL_VDWQQ_FREE],
                        maxsr, maxlr, ivdw, ivdwmod, ielecf, ielecmod, GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE, GMX_NBLIST_INTERACTION_FREE_ENERGY);
            init_nblist(log, &nbl->nlist_sr[eNL_VDW_FREE], &nbl->nlist_lr[eNL_VDW_FREE],
                        maxsr, maxlr, ivdw, ivdwmod, GMX_NBKERNEL_ELEC_NONE, eintmodNONE, GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE, GMX_NBLIST_INTERACTION_FREE_ENERGY);
            init_nblist(log, &nbl->nlist_sr[eNL_QQ_FREE], &nbl->nlist_lr[eNL_QQ_FREE],
                        maxsr, maxlr, GMX_NBKERNEL_VDW_NONE, eintmodNONE, ielecf, ielecmod, GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE, GMX_NBLIST_INTERACTION_FREE_ENERGY);
        }
    }
    /* QMMM MM list */
    if (fr->bQMMM && fr->qr->QMMMscheme != eQMMMschemeoniom)
    {
        init_nblist(log, &fr->QMMMlist, NULL,
                    maxsr, maxlr, 0, 0, ielec, ielecmod, GMX_NBLIST_GEOMETRY_PARTICLE_PARTICLE, GMX_NBLIST_INTERACTION_STANDARD);
    }

    if (log != NULL)
    {
        fprintf(log, "\n");
    }

    fr->ns.nblist_initialized = TRUE;
}

static void reset_nblist(t_nblist *nl)
{
    nl->nri       = -1;
    nl->nrj       = 0;
    nl->maxlen    = 0;
    if (nl->jindex)
    {
        nl->jindex[0] = 0;
    }
}

static void reset_neighbor_lists(t_forcerec *fr, gmx_bool bResetSR, gmx_bool bResetLR)
{
    int n, i;

    if (fr->bQMMM)
    {
        /* only reset the short-range nblist */
        reset_nblist(&(fr->QMMMlist));
    }

    for (n = 0; n < fr->nnblists; n++)
    {
        for (i = 0; i < eNL_NR; i++)
        {
            if (bResetSR)
            {
                reset_nblist( &(fr->nblists[n].nlist_sr[i]) );
            }
            if (bResetLR)
            {
                reset_nblist( &(fr->nblists[n].nlist_lr[i]) );
            }
        }
    }
}




static inline void new_i_nblist(t_nblist *nlist,
                                gmx_bool bLR, atom_id i_atom, int shift, int gid)
{
    int    i, k, nri, nshift;

    nri = nlist->nri;

    /* Check whether we have to increase the i counter */
    if ((nri == -1) ||
        (nlist->iinr[nri]  != i_atom) ||
        (nlist->shift[nri] != shift) ||
        (nlist->gid[nri]   != gid))
    {
        /* This is something else. Now see if any entries have
         * been added in the list of the previous atom.
         */
        if ((nri == -1) ||
            ((nlist->jindex[nri+1] > nlist->jindex[nri]) &&
             (nlist->gid[nri] != -1)))
        {
            /* If so increase the counter */
            nlist->nri++;
            nri++;
            if (nlist->nri >= nlist->maxnri)
            {
                nlist->maxnri += over_alloc_large(nlist->nri);
                reallocate_nblist(nlist);
            }
        }
        /* Set the number of neighbours and the atom number */
        nlist->jindex[nri+1] = nlist->jindex[nri];
        nlist->iinr[nri]     = i_atom;
        nlist->gid[nri]      = gid;
        nlist->shift[nri]    = shift;
    }
}

static inline void close_i_nblist(t_nblist *nlist)
{
    int nri = nlist->nri;
    int len;

    if (nri >= 0)
    {
        /* Add elements up to padding. Since we allocate memory in units
         * of the simd_padding width, we do not have to check for possible
         * list reallocation here.
         */
        while ((nlist->nrj % nlist->simd_padding_width) != 0)
        {
            /* Use -4 here, so we can write forces for 4 atoms before real data */
            nlist->jjnr[nlist->nrj++] = -4;
        }
        nlist->jindex[nri+1] = nlist->nrj;

        len = nlist->nrj -  nlist->jindex[nri];

        /* nlist length for water i molecules is treated statically
         * in the innerloops
         */
        if (len > nlist->maxlen)
        {
            nlist->maxlen = len;
        }
    }
}

static inline void close_nblist(t_nblist *nlist)
{
    /* Only close this nblist when it has been initialized.
     * Avoid the creation of i-lists with no j-particles.
     */
    if (nlist->nrj == 0)
    {
        /* Some assembly kernels do not support empty lists,
         * make sure here that we don't generate any empty lists.
         * With the current ns code this branch is taken in two cases:
         * No i-particles at all: nri=-1 here
         * There are i-particles, but no j-particles; nri=0 here
         */
        nlist->nri = 0;
    }
    else
    {
        /* Close list number nri by incrementing the count */
        nlist->nri++;
    }
}

static inline void close_neighbor_lists(t_forcerec *fr, gmx_bool bMakeQMMMnblist)
{
    int n, i;

    if (bMakeQMMMnblist)
    {
        close_nblist(&(fr->QMMMlist));
    }

    for (n = 0; n < fr->nnblists; n++)
    {
        for (i = 0; (i < eNL_NR); i++)
        {
            close_nblist(&(fr->nblists[n].nlist_sr[i]));
            close_nblist(&(fr->nblists[n].nlist_lr[i]));
        }
    }
}


static inline void add_j_to_nblist(t_nblist *nlist, atom_id j_atom, gmx_bool bLR)
{
    int nrj = nlist->nrj;

    if (nlist->nrj >= nlist->maxnrj)
    {
        nlist->maxnrj = round_up_to_simd_width(over_alloc_small(nlist->nrj + 1), nlist->simd_padding_width);

        if (gmx_debug_at)
        {
            fprintf(debug, "Increasing %s nblist (ielec=%d,ivdw=%d,type=%d,igeometry=%d) j size to %d\n",
                    bLR ? "LR" : "SR", nlist->ielec, nlist->ivdw, nlist->type, nlist->igeometry, nlist->maxnrj);
        }

        srenew(nlist->jjnr, nlist->maxnrj);
    }

    nlist->jjnr[nrj] = j_atom;
    nlist->nrj++;
}

static inline void add_j_to_nblist_cg(t_nblist *nlist,
                                      atom_id j_start, int j_end,
                                      t_excl *bexcl, gmx_bool i_is_j,
                                      gmx_bool bLR)
{
    int nrj = nlist->nrj;
    int j;

    if (nlist->nrj >= nlist->maxnrj)
    {
        nlist->maxnrj = over_alloc_small(nlist->nrj + 1);
        if (gmx_debug_at)
        {
            fprintf(debug, "Increasing %s nblist (ielec=%d,ivdw=%d,type=%d,igeometry=%d) j size to %d\n",
                    bLR ? "LR" : "SR", nlist->ielec, nlist->ivdw, nlist->type, nlist->igeometry, nlist->maxnrj);
        }

        srenew(nlist->jjnr, nlist->maxnrj);
        srenew(nlist->jjnr_end, nlist->maxnrj);
        srenew(nlist->excl, nlist->maxnrj*MAX_CGCGSIZE);
    }

    nlist->jjnr[nrj]     = j_start;
    nlist->jjnr_end[nrj] = j_end;

    if (j_end - j_start > MAX_CGCGSIZE)
    {
        gmx_fatal(FARGS, "The charge-group - charge-group neighborlist do not support charge groups larger than %d, found a charge group of size %d", MAX_CGCGSIZE, j_end-j_start);
    }

    /* Set the exclusions */
    for (j = j_start; j < j_end; j++)
    {
        nlist->excl[nrj*MAX_CGCGSIZE + j - j_start] = bexcl[j];
    }
    if (i_is_j)
    {
        /* Avoid double counting of intra-cg interactions */
        for (j = 1; j < j_end-j_start; j++)
        {
            nlist->excl[nrj*MAX_CGCGSIZE + j] |= (1<<j) - 1;
        }
    }

    nlist->nrj++;
}

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

static void
put_in_list_at(gmx_bool              bHaveVdW[],
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
               int                   solvent_opt)
{
    /* The a[] index has been removed,
     * to put it back in i_atom should be a[i0] and jj should be a[jj].
     */
    t_nblist  *   vdwc;
    t_nblist  *   vdw;
    t_nblist  *   coul;
    t_nblist  *   vdwc_free  = NULL;
    t_nblist  *   vdw_free   = NULL;
    t_nblist  *   coul_free  = NULL;
    t_nblist  *   vdwc_ww    = NULL;
    t_nblist  *   coul_ww    = NULL;

    int           i, j, jcg, igid, gid, nbl_ind, ind_ij;
    atom_id       jj, jj0, jj1, i_atom;
    int           i0, nicg, len;

    int          *cginfo;
    int          *type, *typeB;
    real         *charge, *chargeB;
    real          qi, qiB, qq, rlj;
    gmx_bool      bFreeEnergy, bFree, bFreeJ, bNotEx, *bPert;
    gmx_bool      bDoVdW_i, bDoCoul_i, bDoCoul_i_sol;
    int           iwater, jwater;
    t_nblist     *nlist;

    /* Copy some pointers */
    cginfo  = fr->cginfo;
    charge  = md->chargeA;
    chargeB = md->chargeB;
    type    = md->typeA;
    typeB   = md->typeB;
    bPert   = md->bPerturbed;

    /* Get atom range */
    i0     = index[icg];
    nicg   = index[icg+1]-i0;

    /* Get the i charge group info */
    igid   = GET_CGINFO_GID(cginfo[icg]);

    iwater = (solvent_opt != esolNO) ? GET_CGINFO_SOLOPT(cginfo[icg]) : esolNO;

    bFreeEnergy = FALSE;
    if (md->nPerturbed)
    {
        /* Check if any of the particles involved are perturbed.
         * If not we can do the cheaper normal put_in_list
         * and use more solvent optimization.
         */
        for (i = 0; i < nicg; i++)
        {
            bFreeEnergy |= bPert[i0+i];
        }
        /* Loop over the j charge groups */
        for (j = 0; (j < nj && !bFreeEnergy); j++)
        {
            jcg = jjcg[j];
            jj0 = index[jcg];
            jj1 = index[jcg+1];
            /* Finally loop over the atoms in the j-charge group */
            for (jj = jj0; jj < jj1; jj++)
            {
                bFreeEnergy |= bPert[jj];
            }
        }
    }

    /* Unpack pointers to neighbourlist structs */
    if (fr->nnblists == 1)
    {
        nbl_ind = 0;
    }
    else
    {
        nbl_ind = fr->gid2nblists[GID(igid, jgid, ngid)];
    }
    if (bLR)
    {
        nlist = fr->nblists[nbl_ind].nlist_lr;
    }
    else
    {
        nlist = fr->nblists[nbl_ind].nlist_sr;
    }

    if (iwater != esolNO)
    {
        vdwc = &nlist[eNL_VDWQQ_WATER];
        vdw  = &nlist[eNL_VDW];
        coul = &nlist[eNL_QQ_WATER];
#ifndef DISABLE_WATERWATER_NLIST
        vdwc_ww = &nlist[eNL_VDWQQ_WATERWATER];
        coul_ww = &nlist[eNL_QQ_WATERWATER];
#endif
    }
    else
    {
        vdwc = &nlist[eNL_VDWQQ];
        vdw  = &nlist[eNL_VDW];
        coul = &nlist[eNL_QQ];
    }

    if (!bFreeEnergy)
    {
        if (iwater != esolNO)
        {
            /* Loop over the atoms in the i charge group */
            i_atom  = i0;
            gid     = GID(igid, jgid, ngid);
            /* Create new i_atom for each energy group */
            if (bDoCoul && bDoVdW)
            {
                new_i_nblist(vdwc, bLR, i_atom, shift, gid);
#ifndef DISABLE_WATERWATER_NLIST
                new_i_nblist(vdwc_ww, bLR, i_atom, shift, gid);
#endif
            }
            if (bDoVdW)
            {
                new_i_nblist(vdw, bLR, i_atom, shift, gid);
            }
            if (bDoCoul)
            {
                new_i_nblist(coul, bLR, i_atom, shift, gid);
#ifndef DISABLE_WATERWATER_NLIST
                new_i_nblist(coul_ww, bLR, i_atom, shift, gid);
#endif
            }
            /* Loop over the j charge groups */
            for (j = 0; (j < nj); j++)
            {
                jcg = jjcg[j];

                if (jcg == icg)
                {
                    continue;
                }

                jj0    = index[jcg];
                jwater = GET_CGINFO_SOLOPT(cginfo[jcg]);

                if (iwater == esolSPC && jwater == esolSPC)
                {
                    /* Interaction between two SPC molecules */
                    if (!bDoCoul)
                    {
                        /* VdW only - only first atoms in each water interact */
                        add_j_to_nblist(vdw, jj0, bLR);
                    }
                    else
                    {
#ifdef DISABLE_WATERWATER_NLIST
                        /* Add entries for the three atoms - only do VdW if we need to */
                        if (!bDoVdW)
                        {
                            add_j_to_nblist(coul, jj0, bLR);
                        }
                        else
                        {
                            add_j_to_nblist(vdwc, jj0, bLR);
                        }
                        add_j_to_nblist(coul, jj0+1, bLR);
                        add_j_to_nblist(coul, jj0+2, bLR);
#else
                        /* One entry for the entire water-water interaction */
                        if (!bDoVdW)
                        {
                            add_j_to_nblist(coul_ww, jj0, bLR);
                        }
                        else
                        {
                            add_j_to_nblist(vdwc_ww, jj0, bLR);
                        }
#endif
                    }
                }
                else if (iwater == esolTIP4P && jwater == esolTIP4P)
                {
                    /* Interaction between two TIP4p molecules */
                    if (!bDoCoul)
                    {
                        /* VdW only - only first atoms in each water interact */
                        add_j_to_nblist(vdw, jj0, bLR);
                    }
                    else
                    {
#ifdef DISABLE_WATERWATER_NLIST
                        /* Add entries for the four atoms - only do VdW if we need to */
                        if (bDoVdW)
                        {
                            add_j_to_nblist(vdw, jj0, bLR);
                        }
                        add_j_to_nblist(coul, jj0+1, bLR);
                        add_j_to_nblist(coul, jj0+2, bLR);
                        add_j_to_nblist(coul, jj0+3, bLR);
#else
                        /* One entry for the entire water-water interaction */
                        if (!bDoVdW)
                        {
                            add_j_to_nblist(coul_ww, jj0, bLR);
                        }
                        else
                        {
                            add_j_to_nblist(vdwc_ww, jj0, bLR);
                        }
#endif
                    }
                }
                else
                {
                    /* j charge group is not water, but i is.
                     * Add entries to the water-other_atom lists; the geometry of the water
                     * molecule doesn't matter - that is taken care of in the nonbonded kernel,
                     * so we don't care if it is SPC or TIP4P...
                     */

                    jj1 = index[jcg+1];

                    if (!bDoVdW)
                    {
                        for (jj = jj0; (jj < jj1); jj++)
                        {
                            if (charge[jj] != 0)
                            {
                                add_j_to_nblist(coul, jj, bLR);
                            }
                        }
                    }
                    else if (!bDoCoul)
                    {
                        for (jj = jj0; (jj < jj1); jj++)
                        {
                            if (bHaveVdW[type[jj]])
                            {
                                add_j_to_nblist(vdw, jj, bLR);
                            }
                        }
                    }
                    else
                    {
                        /* _charge_ _groups_ interact with both coulomb and LJ */
                        /* Check which atoms we should add to the lists!       */
                        for (jj = jj0; (jj < jj1); jj++)
                        {
                            if (bHaveVdW[type[jj]])
                            {
                                if (charge[jj] != 0)
                                {
                                    add_j_to_nblist(vdwc, jj, bLR);
                                }
                                else
                                {
                                    add_j_to_nblist(vdw, jj, bLR);
                                }
                            }
                            else if (charge[jj] != 0)
                            {
                                add_j_to_nblist(coul, jj, bLR);
                            }
                        }
                    }
                }
            }
            close_i_nblist(vdw);
            close_i_nblist(coul);
            close_i_nblist(vdwc);
#ifndef DISABLE_WATERWATER_NLIST
            close_i_nblist(coul_ww);
            close_i_nblist(vdwc_ww);
#endif
        }
        else
        {
            /* no solvent as i charge group */
            /* Loop over the atoms in the i charge group */
            for (i = 0; i < nicg; i++)
            {
                i_atom  = i0+i;
                gid     = GID(igid, jgid, ngid);
                qi      = charge[i_atom];

                /* Create new i_atom for each energy group */
                if (bDoVdW && bDoCoul)
                {
                    new_i_nblist(vdwc, bLR, i_atom, shift, gid);
                }
                if (bDoVdW)
                {
                    new_i_nblist(vdw, bLR, i_atom, shift, gid);
                }
                if (bDoCoul)
                {
                    new_i_nblist(coul, bLR, i_atom, shift, gid);
                }
                bDoVdW_i  = (bDoVdW  && bHaveVdW[type[i_atom]]);
                bDoCoul_i = (bDoCoul && qi != 0);

                if (bDoVdW_i || bDoCoul_i)
                {
                    /* Loop over the j charge groups */
                    for (j = 0; (j < nj); j++)
                    {
                        jcg = jjcg[j];

                        /* Check for large charge groups */
                        if (jcg == icg)
                        {
                            jj0 = i0 + i + 1;
                        }
                        else
                        {
                            jj0 = index[jcg];
                        }

                        jj1 = index[jcg+1];
                        /* Finally loop over the atoms in the j-charge group */
                        for (jj = jj0; jj < jj1; jj++)
                        {
                            bNotEx = NOTEXCL(bExcl, i, jj);

                            if (bNotEx)
                            {
                                if (!bDoVdW_i)
                                {
                                    if (charge[jj] != 0)
                                    {
                                        add_j_to_nblist(coul, jj, bLR);
                                    }
                                }
                                else if (!bDoCoul_i)
                                {
                                    if (bHaveVdW[type[jj]])
                                    {
                                        add_j_to_nblist(vdw, jj, bLR);
                                    }
                                }
                                else
                                {
                                    if (bHaveVdW[type[jj]])
                                    {
                                        if (charge[jj] != 0)
                                        {
                                            add_j_to_nblist(vdwc, jj, bLR);
                                        }
                                        else
                                        {
                                            add_j_to_nblist(vdw, jj, bLR);
                                        }
                                    }
                                    else if (charge[jj] != 0)
                                    {
                                        add_j_to_nblist(coul, jj, bLR);
                                    }
                                }
                            }
                        }
                    }
                }
                close_i_nblist(vdw);
                close_i_nblist(coul);
                close_i_nblist(vdwc);
            }
        }
    }
    else
    {
        /* we are doing free energy */
        vdwc_free = &nlist[eNL_VDWQQ_FREE];
        vdw_free  = &nlist[eNL_VDW_FREE];
        coul_free = &nlist[eNL_QQ_FREE];
        /* Loop over the atoms in the i charge group */
        for (i = 0; i < nicg; i++)
        {
            i_atom  = i0+i;
            gid     = GID(igid, jgid, ngid);
            qi      = charge[i_atom];
            qiB     = chargeB[i_atom];

            /* Create new i_atom for each energy group */
            if (bDoVdW && bDoCoul)
            {
                new_i_nblist(vdwc, bLR, i_atom, shift, gid);
            }
            if (bDoVdW)
            {
                new_i_nblist(vdw, bLR, i_atom, shift, gid);
            }
            if (bDoCoul)
            {
                new_i_nblist(coul, bLR, i_atom, shift, gid);
            }

            new_i_nblist(vdw_free, bLR, i_atom, shift, gid);
            new_i_nblist(coul_free, bLR, i_atom, shift, gid);
            new_i_nblist(vdwc_free, bLR, i_atom, shift, gid);

            bDoVdW_i  = (bDoVdW  &&
                         (bHaveVdW[type[i_atom]] || bHaveVdW[typeB[i_atom]]));
            bDoCoul_i = (bDoCoul && (qi != 0 || qiB != 0));
            /* For TIP4P the first atom does not have a charge,
             * but the last three do. So we should still put an atom
             * without LJ but with charge in the water-atom neighborlist
             * for a TIP4p i charge group.
             * For SPC type water the first atom has LJ and charge,
             * so there is no such problem.
             */
            if (iwater == esolNO)
            {
                bDoCoul_i_sol = bDoCoul_i;
            }
            else
            {
                bDoCoul_i_sol = bDoCoul;
            }

            if (bDoVdW_i || bDoCoul_i_sol)
            {
                /* Loop over the j charge groups */
                for (j = 0; (j < nj); j++)
                {
                    jcg = jjcg[j];

                    /* Check for large charge groups */
                    if (jcg == icg)
                    {
                        jj0 = i0 + i + 1;
                    }
                    else
                    {
                        jj0 = index[jcg];
                    }

                    jj1 = index[jcg+1];
                    /* Finally loop over the atoms in the j-charge group */
                    bFree = bPert[i_atom];
                    for (jj = jj0; (jj < jj1); jj++)
                    {
                        bFreeJ = bFree || bPert[jj];
                        /* Complicated if, because the water H's should also
                         * see perturbed j-particles
                         */
                        if (iwater == esolNO || i == 0 || bFreeJ)
                        {
                            bNotEx = NOTEXCL(bExcl, i, jj);

                            if (bNotEx)
                            {
                                if (bFreeJ)
                                {
                                    if (!bDoVdW_i)
                                    {
                                        if (charge[jj] != 0 || chargeB[jj] != 0)
                                        {
                                            add_j_to_nblist(coul_free, jj, bLR);
                                        }
                                    }
                                    else if (!bDoCoul_i)
                                    {
                                        if (bHaveVdW[type[jj]] || bHaveVdW[typeB[jj]])
                                        {
                                            add_j_to_nblist(vdw_free, jj, bLR);
                                        }
                                    }
                                    else
                                    {
                                        if (bHaveVdW[type[jj]] || bHaveVdW[typeB[jj]])
                                        {
                                            if (charge[jj] != 0 || chargeB[jj] != 0)
                                            {
                                                add_j_to_nblist(vdwc_free, jj, bLR);
                                            }
                                            else
                                            {
                                                add_j_to_nblist(vdw_free, jj, bLR);
                                            }
                                        }
                                        else if (charge[jj] != 0 || chargeB[jj] != 0)
                                        {
                                            add_j_to_nblist(coul_free, jj, bLR);
                                        }
                                    }
                                }
                                else if (!bDoVdW_i)
                                {
                                    /* This is done whether or not bWater is set */
                                    if (charge[jj] != 0)
                                    {
                                        add_j_to_nblist(coul, jj, bLR);
                                    }
                                }
                                else if (!bDoCoul_i_sol)
                                {
                                    if (bHaveVdW[type[jj]])
                                    {
                                        add_j_to_nblist(vdw, jj, bLR);
                                    }
                                }
                                else
                                {
                                    if (bHaveVdW[type[jj]])
                                    {
                                        if (charge[jj] != 0)
                                        {
                                            add_j_to_nblist(vdwc, jj, bLR);
                                        }
                                        else
                                        {
                                            add_j_to_nblist(vdw, jj, bLR);
                                        }
                                    }
                                    else if (charge[jj] != 0)
                                    {
                                        add_j_to_nblist(coul, jj, bLR);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            close_i_nblist(vdw);
            close_i_nblist(coul);
            close_i_nblist(vdwc);
            close_i_nblist(vdw_free);
            close_i_nblist(coul_free);
            close_i_nblist(vdwc_free);
        }
    }
}

static void
put_in_list_adress(gmx_bool              bHaveVdW[],
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
                   int                   solvent_opt)
{
    /* The a[] index has been removed,
     * to put it back in i_atom should be a[i0] and jj should be a[jj].
     */
    t_nblist  *   vdwc;
    t_nblist  *   vdw;
    t_nblist  *   coul;
    t_nblist  *   vdwc_adress  = NULL;
    t_nblist  *   vdw_adress   = NULL;
    t_nblist  *   coul_adress  = NULL;
    t_nblist  *   vdwc_ww      = NULL;
    t_nblist  *   coul_ww      = NULL;

    int           i, j, jcg, igid, gid, nbl_ind, nbl_ind_adress;
    atom_id       jj, jj0, jj1, i_atom;
    int           i0, nicg, len;

    int          *cginfo;
    int          *type, *typeB;
    real         *charge, *chargeB;
    real         *wf;
    real          qi, qiB, qq, rlj;
    gmx_bool      bFreeEnergy, bFree, bFreeJ, bNotEx, *bPert;
    gmx_bool      bDoVdW_i, bDoCoul_i, bDoCoul_i_sol;
    gmx_bool      b_hybrid;
    gmx_bool      j_all_atom;
    int           iwater, jwater;
    t_nblist     *nlist, *nlist_adress;

    /* Copy some pointers */
    cginfo  = fr->cginfo;
    charge  = md->chargeA;
    chargeB = md->chargeB;
    type    = md->typeA;
    typeB   = md->typeB;
    bPert   = md->bPerturbed;
    wf      = md->wf;

    /* Get atom range */
    i0     = index[icg];
    nicg   = index[icg+1]-i0;

    /* Get the i charge group info */
    igid   = GET_CGINFO_GID(cginfo[icg]);

    iwater = (solvent_opt != esolNO) ? GET_CGINFO_SOLOPT(cginfo[icg]) : esolNO;

    if (md->nPerturbed)
    {
        gmx_fatal(FARGS, "AdResS does not support free energy pertubation\n");
    }

    /* Unpack pointers to neighbourlist structs */
    if (fr->nnblists == 2)
    {
        nbl_ind        = 0;
        nbl_ind_adress = 1;
    }
    else
    {
        nbl_ind        = fr->gid2nblists[GID(igid, jgid, ngid)];
        nbl_ind_adress = nbl_ind+fr->nnblists/2;
    }
    if (bLR)
    {
        nlist        = fr->nblists[nbl_ind].nlist_lr;
        nlist_adress = fr->nblists[nbl_ind_adress].nlist_lr;
    }
    else
    {
        nlist        = fr->nblists[nbl_ind].nlist_sr;
        nlist_adress = fr->nblists[nbl_ind_adress].nlist_sr;
    }


    vdwc = &nlist[eNL_VDWQQ];
    vdw  = &nlist[eNL_VDW];
    coul = &nlist[eNL_QQ];

    vdwc_adress = &nlist_adress[eNL_VDWQQ];
    vdw_adress  = &nlist_adress[eNL_VDW];
    coul_adress = &nlist_adress[eNL_QQ];

    /* We do not support solvent optimization with AdResS for now.
       For this we would need hybrid solvent-other kernels */

    /* no solvent as i charge group */
    /* Loop over the atoms in the i charge group */
    for (i = 0; i < nicg; i++)
    {
        i_atom  = i0+i;
        gid     = GID(igid, jgid, ngid);
        qi      = charge[i_atom];

        /* Create new i_atom for each energy group */
        if (bDoVdW && bDoCoul)
        {
            new_i_nblist(vdwc, bLR, i_atom, shift, gid);
            new_i_nblist(vdwc_adress, bLR, i_atom, shift, gid);

        }
        if (bDoVdW)
        {
            new_i_nblist(vdw, bLR, i_atom, shift, gid);
            new_i_nblist(vdw_adress, bLR, i_atom, shift, gid);

        }
        if (bDoCoul)
        {
            new_i_nblist(coul, bLR, i_atom, shift, gid);
            new_i_nblist(coul_adress, bLR, i_atom, shift, gid);
        }
        bDoVdW_i  = (bDoVdW  && bHaveVdW[type[i_atom]]);
        bDoCoul_i = (bDoCoul && qi != 0);

        if (bDoVdW_i || bDoCoul_i)
        {
            /* Loop over the j charge groups */
            for (j = 0; (j < nj); j++)
            {
                jcg = jjcg[j];

                /* Check for large charge groups */
                if (jcg == icg)
                {
                    jj0 = i0 + i + 1;
                }
                else
                {
                    jj0 = index[jcg];
                }

                jj1 = index[jcg+1];
                /* Finally loop over the atoms in the j-charge group */
                for (jj = jj0; jj < jj1; jj++)
                {
                    bNotEx = NOTEXCL(bExcl, i, jj);

                    b_hybrid = !((wf[i_atom] == 1 && wf[jj] == 1) || (wf[i_atom] == 0 && wf[jj] == 0));

                    if (bNotEx)
                    {
                        if (!bDoVdW_i)
                        {
                            if (charge[jj] != 0)
                            {
                                if (!b_hybrid)
                                {
                                    add_j_to_nblist(coul, jj, bLR);
                                }
                                else
                                {
                                    add_j_to_nblist(coul_adress, jj, bLR);
                                }
                            }
                        }
                        else if (!bDoCoul_i)
                        {
                            if (bHaveVdW[type[jj]])
                            {
                                if (!b_hybrid)
                                {
                                    add_j_to_nblist(vdw, jj, bLR);
                                }
                                else
                                {
                                    add_j_to_nblist(vdw_adress, jj, bLR);
                                }
                            }
                        }
                        else
                        {
                            if (bHaveVdW[type[jj]])
                            {
                                if (charge[jj] != 0)
                                {
                                    if (!b_hybrid)
                                    {
                                        add_j_to_nblist(vdwc, jj, bLR);
                                    }
                                    else
                                    {
                                        add_j_to_nblist(vdwc_adress, jj, bLR);
                                    }
                                }
                                else
                                {
                                    if (!b_hybrid)
                                    {
                                        add_j_to_nblist(vdw, jj, bLR);
                                    }
                                    else
                                    {
                                        add_j_to_nblist(vdw_adress, jj, bLR);
                                    }

                                }
                            }
                            else if (charge[jj] != 0)
                            {
                                if (!b_hybrid)
                                {
                                    add_j_to_nblist(coul, jj, bLR);
                                }
                                else
                                {
                                    add_j_to_nblist(coul_adress, jj, bLR);
                                }

                            }
                        }
                    }
                }
            }

            close_i_nblist(vdw);
            close_i_nblist(coul);
            close_i_nblist(vdwc);
            close_i_nblist(vdw_adress);
            close_i_nblist(coul_adress);
            close_i_nblist(vdwc_adress);
        }
    }
}

static void
put_in_list_qmmm(gmx_bool              bHaveVdW[],
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
                 int                   solvent_opt)
{
    t_nblist  *   coul;
    int           i, j, jcg, igid, gid;
    atom_id       jj, jj0, jj1, i_atom;
    int           i0, nicg;
    gmx_bool      bNotEx;

    /* Get atom range */
    i0     = index[icg];
    nicg   = index[icg+1]-i0;

    /* Get the i charge group info */
    igid   = GET_CGINFO_GID(fr->cginfo[icg]);

    coul = &fr->QMMMlist;

    /* Loop over atoms in the ith charge group */
    for (i = 0; i < nicg; i++)
    {
        i_atom = i0+i;
        gid    = GID(igid, jgid, ngid);
        /* Create new i_atom for each energy group */
        new_i_nblist(coul, bLR, i_atom, shift, gid);

        /* Loop over the j charge groups */
        for (j = 0; j < nj; j++)
        {
            jcg = jjcg[j];

            /* Charge groups cannot have QM and MM atoms simultaneously */
            if (jcg != icg)
            {
                jj0 = index[jcg];
                jj1 = index[jcg+1];
                /* Finally loop over the atoms in the j-charge group */
                for (jj = jj0; jj < jj1; jj++)
                {
                    bNotEx = NOTEXCL(bExcl, i, jj);
                    if (bNotEx)
                    {
                        add_j_to_nblist(coul, jj, bLR);
                    }
                }
            }
        }
        close_i_nblist(coul);
    }
}

static void
put_in_list_cg(gmx_bool              bHaveVdW[],
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
               int                   solvent_opt)
{
    int          cginfo;
    int          igid, gid, nbl_ind;
    t_nblist *   vdwc;
    int          j, jcg;

    cginfo = fr->cginfo[icg];

    igid = GET_CGINFO_GID(cginfo);
    gid  = GID(igid, jgid, ngid);

    /* Unpack pointers to neighbourlist structs */
    if (fr->nnblists == 1)
    {
        nbl_ind = 0;
    }
    else
    {
        nbl_ind = fr->gid2nblists[gid];
    }
    if (bLR)
    {
        vdwc = &fr->nblists[nbl_ind].nlist_lr[eNL_VDWQQ];
    }
    else
    {
        vdwc = &fr->nblists[nbl_ind].nlist_sr[eNL_VDWQQ];
    }

    /* Make a new neighbor list for charge group icg.
     * Currently simply one neighbor list is made with LJ and Coulomb.
     * If required, zero interactions could be removed here
     * or in the force loop.
     */
    new_i_nblist(vdwc, bLR, index[icg], shift, gid);
    vdwc->iinr_end[vdwc->nri] = index[icg+1];

    for (j = 0; (j < nj); j++)
    {
        jcg = jjcg[j];
        /* Skip the icg-icg pairs if all self interactions are excluded */
        if (!(jcg == icg && GET_CGINFO_EXCL_INTRA(cginfo)))
        {
            /* Here we add the j charge group jcg to the list,
             * exclusions are also added to the list.
             */
            add_j_to_nblist_cg(vdwc, index[jcg], index[jcg+1], bExcl, icg == jcg, bLR);
        }
    }

    close_i_nblist(vdwc);
}

static void setexcl(atom_id start, atom_id end, t_blocka *excl, gmx_bool b,
                    t_excl bexcl[])
{
    atom_id i, k;

    if (b)
    {
        for (i = start; i < end; i++)
        {
            for (k = excl->index[i]; k < excl->index[i+1]; k++)
            {
                SETEXCL(bexcl, i-start, excl->a[k]);
            }
        }
    }
    else
    {
        for (i = start; i < end; i++)
        {
            for (k = excl->index[i]; k < excl->index[i+1]; k++)
            {
                RMEXCL(bexcl, i-start, excl->a[k]);
            }
        }
    }
}

int calc_naaj(int icg, int cgtot)
{
    int naaj;

    if ((cgtot % 2) == 1)
    {
        /* Odd number of charge groups, easy */
        naaj = 1 + (cgtot/2);
    }
    else if ((cgtot % 4) == 0)
    {
        /* Multiple of four is hard */
        if (icg < cgtot/2)
        {
            if ((icg % 2) == 0)
            {
                naaj = 1+(cgtot/2);
            }
            else
            {
                naaj = cgtot/2;
            }
        }
        else
        {
            if ((icg % 2) == 1)
            {
                naaj = 1+(cgtot/2);
            }
            else
            {
                naaj = cgtot/2;
            }
        }
    }
    else
    {
        /* cgtot/2 = odd */
        if ((icg % 2) == 0)
        {
            naaj = 1+(cgtot/2);
        }
        else
        {
            naaj = cgtot/2;
        }
    }
#ifdef DEBUG
    fprintf(log, "naaj=%d\n", naaj);
#endif

    return naaj;
}

/************************************************
 *
 *  S I M P L E      C O R E     S T U F F
 *
 ************************************************/

static real calc_image_tric(rvec xi, rvec xj, matrix box,
                            rvec b_inv, int *shift)
{
    /* This code assumes that the cut-off is smaller than
     * a half times the smallest diagonal element of the box.
     */
    const real h25 = 2.5;
    real       dx, dy, dz;
    real       r2;
    int        tx, ty, tz;

    /* Compute diff vector */
    dz = xj[ZZ] - xi[ZZ];
    dy = xj[YY] - xi[YY];
    dx = xj[XX] - xi[XX];

    /* Perform NINT operation, using trunc operation, therefore
     * we first add 2.5 then subtract 2 again
     */
    tz  = dz*b_inv[ZZ] + h25;
    tz -= 2;
    dz -= tz*box[ZZ][ZZ];
    dy -= tz*box[ZZ][YY];
    dx -= tz*box[ZZ][XX];

    ty  = dy*b_inv[YY] + h25;
    ty -= 2;
    dy -= ty*box[YY][YY];
    dx -= ty*box[YY][XX];

    tx  = dx*b_inv[XX]+h25;
    tx -= 2;
    dx -= tx*box[XX][XX];

    /* Distance squared */
    r2 = (dx*dx) + (dy*dy) + (dz*dz);

    *shift = XYZ2IS(tx, ty, tz);

    return r2;
}

static real calc_image_rect(rvec xi, rvec xj, rvec box_size,
                            rvec b_inv, int *shift)
{
    const real h15 = 1.5;
    real       ddx, ddy, ddz;
    real       dx, dy, dz;
    real       r2;
    int        tx, ty, tz;

    /* Compute diff vector */
    dx = xj[XX] - xi[XX];
    dy = xj[YY] - xi[YY];
    dz = xj[ZZ] - xi[ZZ];

    /* Perform NINT operation, using trunc operation, therefore
     * we first add 1.5 then subtract 1 again
     */
    tx = dx*b_inv[XX] + h15;
    ty = dy*b_inv[YY] + h15;
    tz = dz*b_inv[ZZ] + h15;
    tx--;
    ty--;
    tz--;

    /* Correct diff vector for translation */
    ddx = tx*box_size[XX] - dx;
    ddy = ty*box_size[YY] - dy;
    ddz = tz*box_size[ZZ] - dz;

    /* Distance squared */
    r2 = (ddx*ddx) + (ddy*ddy) + (ddz*ddz);

    *shift = XYZ2IS(tx, ty, tz);

    return r2;
}

static void add_simple(t_ns_buf *nsbuf, int nrj, atom_id cg_j,
                       gmx_bool bHaveVdW[], int ngid, t_mdatoms *md,
                       int icg, int jgid, t_block *cgs, t_excl bexcl[],
                       int shift, t_forcerec *fr, put_in_list_t *put_in_list)
{
    if (nsbuf->nj + nrj > MAX_CG)
    {
        put_in_list(bHaveVdW, ngid, md, icg, jgid, nsbuf->ncg, nsbuf->jcg,
                    cgs->index, bexcl, shift, fr, FALSE, TRUE, TRUE, fr->solvent_opt);
        /* Reset buffer contents */
        nsbuf->ncg = nsbuf->nj = 0;
    }
    nsbuf->jcg[nsbuf->ncg++] = cg_j;
    nsbuf->nj               += nrj;
}

static void ns_inner_tric(rvec x[], int icg, int *i_egp_flags,
                          int njcg, atom_id jcg[],
                          matrix box, rvec b_inv, real rcut2,
                          t_block *cgs, t_ns_buf **ns_buf,
                          gmx_bool bHaveVdW[], int ngid, t_mdatoms *md,
                          t_excl bexcl[], t_forcerec *fr,
                          put_in_list_t *put_in_list)
{
    int       shift;
    int       j, nrj, jgid;
    int      *cginfo = fr->cginfo;
    atom_id   cg_j, *cgindex;
    t_ns_buf *nsbuf;

    cgindex = cgs->index;
    shift   = CENTRAL;
    for (j = 0; (j < njcg); j++)
    {
        cg_j   = jcg[j];
        nrj    = cgindex[cg_j+1]-cgindex[cg_j];
        if (calc_image_tric(x[icg], x[cg_j], box, b_inv, &shift) < rcut2)
        {
            jgid  = GET_CGINFO_GID(cginfo[cg_j]);
            if (!(i_egp_flags[jgid] & EGP_EXCL))
            {
                add_simple(&ns_buf[jgid][shift], nrj, cg_j,
                           bHaveVdW, ngid, md, icg, jgid, cgs, bexcl, shift, fr,
                           put_in_list);
            }
        }
    }
}

static void ns_inner_rect(rvec x[], int icg, int *i_egp_flags,
                          int njcg, atom_id jcg[],
                          gmx_bool bBox, rvec box_size, rvec b_inv, real rcut2,
                          t_block *cgs, t_ns_buf **ns_buf,
                          gmx_bool bHaveVdW[], int ngid, t_mdatoms *md,
                          t_excl bexcl[], t_forcerec *fr,
                          put_in_list_t *put_in_list)
{
    int       shift;
    int       j, nrj, jgid;
    int      *cginfo = fr->cginfo;
    atom_id   cg_j, *cgindex;
    t_ns_buf *nsbuf;

    cgindex = cgs->index;
    if (bBox)
    {
        shift = CENTRAL;
        for (j = 0; (j < njcg); j++)
        {
            cg_j   = jcg[j];
            nrj    = cgindex[cg_j+1]-cgindex[cg_j];
            if (calc_image_rect(x[icg], x[cg_j], box_size, b_inv, &shift) < rcut2)
            {
                jgid  = GET_CGINFO_GID(cginfo[cg_j]);
                if (!(i_egp_flags[jgid] & EGP_EXCL))
                {
                    add_simple(&ns_buf[jgid][shift], nrj, cg_j,
                               bHaveVdW, ngid, md, icg, jgid, cgs, bexcl, shift, fr,
                               put_in_list);
                }
            }
        }
    }
    else
    {
        for (j = 0; (j < njcg); j++)
        {
            cg_j   = jcg[j];
            nrj    = cgindex[cg_j+1]-cgindex[cg_j];
            if ((rcut2 == 0) || (distance2(x[icg], x[cg_j]) < rcut2))
            {
                jgid  = GET_CGINFO_GID(cginfo[cg_j]);
                if (!(i_egp_flags[jgid] & EGP_EXCL))
                {
                    add_simple(&ns_buf[jgid][CENTRAL], nrj, cg_j,
                               bHaveVdW, ngid, md, icg, jgid, cgs, bexcl, CENTRAL, fr,
                               put_in_list);
                }
            }
        }
    }
}

/* ns_simple_core needs to be adapted for QMMM still 2005 */

static int ns_simple_core(t_forcerec *fr,
                          gmx_localtop_t *top,
                          t_mdatoms *md,
                          matrix box, rvec box_size,
                          t_excl bexcl[], atom_id *aaj,
                          int ngid, t_ns_buf **ns_buf,
                          put_in_list_t *put_in_list, gmx_bool bHaveVdW[])
{
    int          naaj, k;
    real         rlist2;
    int          nsearch, icg, jcg, igid, i0, nri, nn;
    int         *cginfo;
    t_ns_buf    *nsbuf;
    /* atom_id  *i_atoms; */
    t_block     *cgs  = &(top->cgs);
    t_blocka    *excl = &(top->excls);
    rvec         b_inv;
    int          m;
    gmx_bool     bBox, bTriclinic;
    int         *i_egp_flags;

    rlist2 = sqr(fr->rlist);

    bBox = (fr->ePBC != epbcNONE);
    if (bBox)
    {
        for (m = 0; (m < DIM); m++)
        {
            b_inv[m] = divide(1.0, box_size[m]);
        }
        bTriclinic = TRICLINIC(box);
    }
    else
    {
        bTriclinic = FALSE;
    }

    cginfo = fr->cginfo;

    nsearch = 0;
    for (icg = fr->cg0; (icg < fr->hcg); icg++)
    {
        /*
           i0        = cgs->index[icg];
           nri       = cgs->index[icg+1]-i0;
           i_atoms   = &(cgs->a[i0]);
           i_eg_excl = fr->eg_excl + ngid*md->cENER[*i_atoms];
           setexcl(nri,i_atoms,excl,TRUE,bexcl);
         */
        igid        = GET_CGINFO_GID(cginfo[icg]);
        i_egp_flags = fr->egp_flags + ngid*igid;
        setexcl(cgs->index[icg], cgs->index[icg+1], excl, TRUE, bexcl);

        naaj = calc_naaj(icg, cgs->nr);
        if (bTriclinic)
        {
            ns_inner_tric(fr->cg_cm, icg, i_egp_flags, naaj, &(aaj[icg]),
                          box, b_inv, rlist2, cgs, ns_buf,
                          bHaveVdW, ngid, md, bexcl, fr, put_in_list);
        }
        else
        {
            ns_inner_rect(fr->cg_cm, icg, i_egp_flags, naaj, &(aaj[icg]),
                          bBox, box_size, b_inv, rlist2, cgs, ns_buf,
                          bHaveVdW, ngid, md, bexcl, fr, put_in_list);
        }
        nsearch += naaj;

        for (nn = 0; (nn < ngid); nn++)
        {
            for (k = 0; (k < SHIFTS); k++)
            {
                nsbuf = &(ns_buf[nn][k]);
                if (nsbuf->ncg > 0)
                {
                    put_in_list(bHaveVdW, ngid, md, icg, nn, nsbuf->ncg, nsbuf->jcg,
                                cgs->index, bexcl, k, fr, FALSE, TRUE, TRUE, fr->solvent_opt);
                    nsbuf->ncg = nsbuf->nj = 0;
                }
            }
        }
        /* setexcl(nri,i_atoms,excl,FALSE,bexcl); */
        setexcl(cgs->index[icg], cgs->index[icg+1], excl, FALSE, bexcl);
    }
    close_neighbor_lists(fr, FALSE);

    return nsearch;
}

/************************************************
 *
 *    N S 5     G R I D     S T U F F
 *
 ************************************************/

static inline void get_dx(int Nx, real gridx, real rc2, int xgi, real x,
                          int *dx0, int *dx1, real *dcx2)
{
    real dcx, tmp;
    int  xgi0, xgi1, i;

    if (xgi < 0)
    {
        *dx0 = 0;
        xgi0 = -1;
        *dx1 = -1;
        xgi1 = 0;
    }
    else if (xgi >= Nx)
    {
        *dx0 = Nx;
        xgi0 = Nx-1;
        *dx1 = Nx-1;
        xgi1 = Nx;
    }
    else
    {
        dcx2[xgi] = 0;
        *dx0      = xgi;
        xgi0      = xgi-1;
        *dx1      = xgi;
        xgi1      = xgi+1;
    }

    for (i = xgi0; i >= 0; i--)
    {
        dcx = (i+1)*gridx-x;
        tmp = dcx*dcx;
        if (tmp >= rc2)
        {
            break;
        }
        *dx0    = i;
        dcx2[i] = tmp;
    }
    for (i = xgi1; i < Nx; i++)
    {
        dcx = i*gridx-x;
        tmp = dcx*dcx;
        if (tmp >= rc2)
        {
            break;
        }
        *dx1    = i;
        dcx2[i] = tmp;
    }
}

static inline void get_dx_dd(int Nx, real gridx, real rc2, int xgi, real x,
                             int ncpddc, int shift_min, int shift_max,
                             int *g0, int *g1, real *dcx2)
{
    real dcx, tmp;
    int  g_min, g_max, shift_home;

    if (xgi < 0)
    {
        g_min = 0;
        g_max = Nx - 1;
        *g0   = 0;
        *g1   = -1;
    }
    else if (xgi >= Nx)
    {
        g_min = 0;
        g_max = Nx - 1;
        *g0   = Nx;
        *g1   = Nx - 1;
    }
    else
    {
        if (ncpddc == 0)
        {
            g_min = 0;
            g_max = Nx - 1;
        }
        else
        {
            if (xgi < ncpddc)
            {
                shift_home = 0;
            }
            else
            {
                shift_home = -1;
            }
            g_min = (shift_min == shift_home ? 0          : ncpddc);
            g_max = (shift_max == shift_home ? ncpddc - 1 : Nx - 1);
        }
        if (shift_min > 0)
        {
            *g0 = g_min;
            *g1 = g_min - 1;
        }
        else if (shift_max < 0)
        {
            *g0 = g_max + 1;
            *g1 = g_max;
        }
        else
        {
            *g0       = xgi;
            *g1       = xgi;
            dcx2[xgi] = 0;
        }
    }

    while (*g0 > g_min)
    {
        /* Check one grid cell down */
        dcx = ((*g0 - 1) + 1)*gridx - x;
        tmp = dcx*dcx;
        if (tmp >= rc2)
        {
            break;
        }
        (*g0)--;
        dcx2[*g0] = tmp;
    }

    while (*g1 < g_max)
    {
        /* Check one grid cell up */
        dcx = (*g1 + 1)*gridx - x;
        tmp = dcx*dcx;
        if (tmp >= rc2)
        {
            break;
        }
        (*g1)++;
        dcx2[*g1] = tmp;
    }
}


#define sqr(x) ((x)*(x))
#define calc_dx2(XI, YI, ZI, y) (sqr(XI-y[XX]) + sqr(YI-y[YY]) + sqr(ZI-y[ZZ]))
#define calc_cyl_dx2(XI, YI, y) (sqr(XI-y[XX]) + sqr(YI-y[YY]))
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
{
    *rs2 = sqr(fr->rlist);

    if (bDoLongRange && fr->bTwinRange)
    {
        /* The VdW and elec. LR cut-off's could be different,
         * so we can not simply set them to rlistlong.
         */
        if (EVDW_MIGHT_BE_ZERO_AT_CUTOFF(fr->vdwtype) &&
            fr->rvdw > fr->rlist)
        {
            *rvdw2  = sqr(fr->rlistlong);
        }
        else
        {
            *rvdw2  = sqr(fr->rvdw);
        }
        if (EEL_MIGHT_BE_ZERO_AT_CUTOFF(fr->eeltype) &&
            fr->rcoulomb > fr->rlist)
        {
            *rcoul2 = sqr(fr->rlistlong);
        }
        else
        {
            *rcoul2 = sqr(fr->rcoulomb);
        }
    }
    else
    {
        /* Workaround for a gcc -O3 or -ffast-math problem */
        *rvdw2  = *rs2;
        *rcoul2 = *rs2;
    }
    *rm2 = min(*rvdw2, *rcoul2);
    *rl2 = max(*rvdw2, *rcoul2);
}

static void init_nsgrid_lists(t_forcerec *fr, int ngid, gmx_ns_t *ns)
{
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

    for (j = 0; (j < ngid); j++)
    {
        snew(ns->nl_sr[j], MAX_CG);
        snew(ns->nl_lr_ljc[j], MAX_CG);
        snew(ns->nl_lr_one[j], MAX_CG);
    }
    if (debug)
    {
        fprintf(debug,
                "ns5_core: rs2 = %g, rm2 = %g, rl2 = %g (nm^2)\n",
                rs2, rm2, rl2);
    }
}

static int nsgrid_core(FILE *log, t_commrec *cr, t_forcerec *fr,
                       matrix box, rvec box_size, int ngid,
                       gmx_localtop_t *top,
                       t_grid *grid, rvec x[],
                       t_excl bexcl[], gmx_bool *bExcludeAlleg,
                       t_nrnb *nrnb, t_mdatoms *md,
                       real *lambda, real *dvdlambda,
                       gmx_grppairener_t *grppener,
                       put_in_list_t *put_in_list,
                       gmx_bool bHaveVdW[],
                       gmx_bool bDoLongRange, gmx_bool bMakeQMMMnblist)
{
    gmx_ns_t     *ns;
    atom_id     **nl_lr_ljc, **nl_lr_one, **nl_sr;
    int          *nlr_ljc, *nlr_one, *nsr;
    gmx_domdec_t *dd     = NULL;
    t_block      *cgs    = &(top->cgs);
    int          *cginfo = fr->cginfo;
    /* atom_id *i_atoms,*cgsindex=cgs->index; */
    ivec          sh0, sh1, shp;
    int           cell_x, cell_y, cell_z;
    int           d, tx, ty, tz, dx, dy, dz, cj;
#ifdef ALLOW_OFFDIAG_LT_HALFDIAG
    int           zsh_ty, zsh_tx, ysh_tx;
#endif
    int           dx0, dx1, dy0, dy1, dz0, dz1;
    int           Nx, Ny, Nz, shift = -1, j, nrj, nns, nn = -1;
    real          gridx, gridy, gridz, grid_x, grid_y, grid_z;
    real         *dcx2, *dcy2, *dcz2;
    int           zgi, ygi, xgi;
    int           cg0, cg1, icg = -1, cgsnr, i0, igid, nri, naaj, max_jcg;
    int           jcg0, jcg1, jjcg, cgj0, jgid;
    int          *grida, *gridnra, *gridind;
    gmx_bool      rvdw_lt_rcoul, rcoul_lt_rvdw;
    rvec          xi, *cgcm, grid_offset;
    real          r2, rs2, rvdw2, rcoul2, rm2, rl2, XI, YI, ZI, dcx, dcy, dcz, tmp1, tmp2;
    int          *i_egp_flags;
    gmx_bool      bTriclinicX, bTriclinicY;
    ivec          ncpddc;

    ns = &fr->ns;


    bTriclinicX = ((YY < grid->npbcdim &&
                    box[YY][XX] != 0) ||
                   (ZZ < grid->npbcdim &&
                    box[ZZ][XX] != 0));
    bTriclinicY =  (ZZ < grid->npbcdim &&
                    box[ZZ][YY] != 0);

    cgsnr    = cgs->nr;

    get_cutoff2(fr, bDoLongRange, &rvdw2, &rcoul2, &rs2, &rm2, &rl2);

    rvdw_lt_rcoul = (rvdw2 >= rcoul2);
    rcoul_lt_rvdw = (rcoul2 >= rvdw2);

    if (bMakeQMMMnblist)
    {
        rm2 = rl2;
        rs2 = rl2;
    }

    nl_sr     = ns->nl_sr;
    nsr       = ns->nsr;
    nl_lr_ljc = ns->nl_lr_ljc;
    nl_lr_one = ns->nl_lr_one;
    nlr_ljc   = ns->nlr_ljc;
    nlr_one   = ns->nlr_one;

    /* Unpack arrays */
    cgcm    = fr->cg_cm;
    Nx      = grid->n[XX];
    Ny      = grid->n[YY];
    Nz      = grid->n[ZZ];
    grida   = grid->a;
    gridind = grid->index;
    gridnra = grid->nra;
    nns     = 0;

    gridx      = grid->cell_size[XX];
    gridy      = grid->cell_size[YY];
    gridz      = grid->cell_size[ZZ];
    grid_x     = 1/gridx;
    grid_y     = 1/gridy;
    grid_z     = 1/gridz;
    copy_rvec(grid->cell_offset, grid_offset);
    copy_ivec(grid->ncpddc, ncpddc);
    dcx2       = grid->dcx2;
    dcy2       = grid->dcy2;
    dcz2       = grid->dcz2;

#ifdef ALLOW_OFFDIAG_LT_HALFDIAG
    zsh_ty = floor(-box[ZZ][YY]/box[YY][YY]+0.5);
    zsh_tx = floor(-box[ZZ][XX]/box[XX][XX]+0.5);
    ysh_tx = floor(-box[YY][XX]/box[XX][XX]+0.5);
    if (zsh_tx != 0 && ysh_tx != 0)
    {
        /* This could happen due to rounding, when both ratios are 0.5 */
        ysh_tx = 0;
    }
#endif

    debug_gmx();

    if (fr->n_tpi)
    {
        /* We only want a list for the test particle */
        cg0 = cgsnr - 1;
    }
    else
    {
        cg0 = grid->icg0;
    }
    cg1 = grid->icg1;

    /* Set the shift range */
    for (d = 0; d < DIM; d++)
    {
        sh0[d] = -1;
        sh1[d] = 1;
        /* Check if we need periodicity shifts.
         * Without PBC or with domain decomposition we don't need them.
         */
        if (d >= ePBC2npbcdim(fr->ePBC) )
        {
            shp[d] = 0;
        }
        else
        {
            if (d == XX &&
                box[XX][XX] - fabs(box[YY][XX]) - fabs(box[ZZ][XX]) < sqrt(rl2))
            {
                shp[d] = 2;
            }
            else
            {
                shp[d] = 1;
            }
        }
    }

    /* No need to perform any left-over force calculations anymore (as we used to do here)
     * since we now save the proper long-range lists for later evaluation.
     */

    debug_gmx();

    /* Close neighbourlists */
    close_neighbor_lists(fr, bMakeQMMMnblist);

    return nns;
}

void ns_realloc_natoms(gmx_ns_t *ns, int natoms)
{
    int i;

    if (natoms > ns->nra_alloc)
    {
        ns->nra_alloc = over_alloc_dd(natoms);
        srenew(ns->bexcl, ns->nra_alloc);
        for (i = 0; i < ns->nra_alloc; i++)
        {
            ns->bexcl[i] = 0;
        }
    }
}

void init_ns(FILE *fplog, const t_commrec *cr,
             gmx_ns_t *ns, t_forcerec *fr,
             const gmx_mtop_t *mtop,
             matrix box)
{
    int  mt, icg, nr_in_cg, maxcg, i, j, jcg, ngid, ncg;
    t_block *cgs;
    char *ptr;

    /* Compute largest charge groups size (# atoms) */
    nr_in_cg = 1;
    for (mt = 0; mt < mtop->nmoltype; mt++)
    {
        cgs = &mtop->moltype[mt].cgs;
        for (icg = 0; (icg < cgs->nr); icg++)
        {
            nr_in_cg = max(nr_in_cg, (int)(cgs->index[icg+1]-cgs->index[icg]));
        }
    }

    /* Verify whether largest charge group is <= max cg.
     * This is determined by the type of the local exclusion type
     * Exclusions are stored in bits. (If the type is not large
     * enough, enlarge it, unsigned char -> unsigned short -> unsigned long)
     */
    maxcg = sizeof(t_excl)*8;
    if (nr_in_cg > maxcg)
    {
        gmx_fatal(FARGS, "Max #atoms in a charge group: %d > %d\n",
                  nr_in_cg, maxcg);
    }

    ngid = mtop->groups.grps[egcENER].nr;
    snew(ns->bExcludeAlleg, ngid);
    for (i = 0; i < ngid; i++)
    {
        ns->bExcludeAlleg[i] = TRUE;
        for (j = 0; j < ngid; j++)
        {
            if (!(fr->egp_flags[i*ngid+j] & EGP_EXCL))
            {
                ns->bExcludeAlleg[i] = FALSE;
            }
        }
    }

    if (fr->bGrid)
    {
        /* Grid search */
        ns->grid = init_grid(fplog, fr);
        init_nsgrid_lists(fr, ngid, ns);
    }
    else
    {
        /* Simple search */
        snew(ns->ns_buf, ngid);
        for (i = 0; (i < ngid); i++)
        {
            snew(ns->ns_buf[i], SHIFTS);
        }
        ncg = ncg_mtop(mtop);
        snew(ns->simple_aaj, 2*ncg);
        for (jcg = 0; (jcg < ncg); jcg++)
        {
            ns->simple_aaj[jcg]     = jcg;
            ns->simple_aaj[jcg+ncg] = jcg;
        }
    }

    /* Create array that determines whether or not atoms have VdW */
    snew(ns->bHaveVdW, fr->ntype);
    for (i = 0; (i < fr->ntype); i++)
    {
        for (j = 0; (j < fr->ntype); j++)
        {
            ns->bHaveVdW[i] = (ns->bHaveVdW[i] ||
                               (fr->bBHAM ?
                                ((BHAMA(fr->nbfp, fr->ntype, i, j) != 0) ||
                                 (BHAMB(fr->nbfp, fr->ntype, i, j) != 0) ||
                                 (BHAMC(fr->nbfp, fr->ntype, i, j) != 0)) :
                                ((C6(fr->nbfp, fr->ntype, i, j) != 0) ||
                                 (C12(fr->nbfp, fr->ntype, i, j) != 0))));
        }
    }
    if (debug)
    {
        pr_bvec(debug, 0, "bHaveVdW", ns->bHaveVdW, fr->ntype, TRUE);
    }

    ns->nra_alloc = 0;
    ns->bexcl     = NULL;
        /* This could be reduced with particle decomposition */
        ns_realloc_natoms(ns, mtop->natoms);

    ns->nblist_initialized = FALSE;

    /* nbr list debug dump */
    {
        char *ptr = getenv("GMX_DUMP_NL");
        if (ptr)
        {
            ns->dump_nl = strtol(ptr, NULL, 10);
            if (fplog)
            {
                fprintf(fplog, "GMX_DUMP_NL = %d", ns->dump_nl);
            }
        }
        else
        {
            ns->dump_nl = 0;
        }
    }
}


int search_neighbours(FILE *log, t_forcerec *fr,
                      rvec x[], matrix box,
                      gmx_localtop_t *top,
                      gmx_groups_t *groups,
                      t_commrec *cr,
                      t_nrnb *nrnb, t_mdatoms *md,
                      real *lambda, real *dvdlambda,
                      gmx_grppairener_t *grppener,
                      gmx_bool bFillGrid,
                      gmx_bool bDoLongRangeNS,
                      gmx_bool bPadListsForKernels)
{
    t_block  *cgs = &(top->cgs);
    rvec     box_size, grid_x0, grid_x1;
    int      i, j, m, ngid;
    real     min_size, grid_dens;
    int      nsearch;
    gmx_bool     bGrid;
    char     *ptr;
    gmx_bool     *i_egp_flags;
    int      cg_start, cg_end, start, end;
    gmx_ns_t *ns;
    t_grid   *grid;
    gmx_domdec_zones_t *dd_zones;
    put_in_list_t *put_in_list;

    ns = &fr->ns;

    /* Set some local variables */
    bGrid = fr->bGrid;
    ngid  = groups->grps[egcENER].nr;

    for (m = 0; (m < DIM); m++)
    {
        box_size[m] = box[m][m];
    }

    if (fr->ePBC != epbcNONE)
    {
        if (sqr(fr->rlistlong) >= max_cutoff2(fr->ePBC, box))
        {
            gmx_fatal(FARGS, "One of the box vectors has become shorter than twice the cut-off length or box_yy-|box_zy| or box_zz has become smaller than the cut-off.");
        }
        if (!bGrid)
        {
            min_size = min(box_size[XX], min(box_size[YY], box_size[ZZ]));
            if (2*fr->rlistlong >= min_size)
            {
                gmx_fatal(FARGS, "One of the box diagonal elements has become smaller than twice the cut-off length.");
            }
        }
    }

    debug_gmx();

    /* Reset the neighbourlists */
    reset_neighbor_lists(fr, TRUE, TRUE);

    debug_gmx();

    if (fr->adress_type == eAdressOff)
    {
        if (!fr->ns.bCGlist)
        {
            put_in_list = put_in_list_at;
        }
        else
        {
            put_in_list = put_in_list_cg;
        }
    }
    else
    {
        put_in_list = put_in_list_adress;
    }

    /* Do the core! */
    if (bGrid)
    {
        grid    = ns->grid;
        nsearch = nsgrid_core(log, cr, fr, box, box_size, ngid, top,
                              grid, x, ns->bexcl, ns->bExcludeAlleg,
                              nrnb, md, lambda, dvdlambda, grppener,
                              put_in_list, ns->bHaveVdW,
                              bDoLongRangeNS, FALSE);

        /* neighbour searching withouth QMMM! QM atoms have zero charge in
         * the classical calculation. The charge-charge interaction
         * between QM and MM atoms is handled in the QMMM core calculation
         * (see QMMM.c). The VDW however, we'd like to compute classically
         * and the QM MM atom pairs have just been put in the
         * corresponding neighbourlists. in case of QMMM we still need to
         * fill a special QMMM neighbourlist that contains all neighbours
         * of the QM atoms. If bQMMM is true, this list will now be made:
         */
        if (fr->bQMMM && fr->qr->QMMMscheme != eQMMMschemeoniom)
        {
            nsearch += nsgrid_core(log, cr, fr, box, box_size, ngid, top,
                                   grid, x, ns->bexcl, ns->bExcludeAlleg,
                                   nrnb, md, lambda, dvdlambda, grppener,
                                   put_in_list_qmmm, ns->bHaveVdW,
                                   bDoLongRangeNS, TRUE);
        }
    }
    else
    {
        nsearch = ns_simple_core(fr, top, md, box, box_size,
                                 ns->bexcl, ns->simple_aaj,
                                 ngid, ns->ns_buf, put_in_list, ns->bHaveVdW);
    }
    debug_gmx();

#ifdef DEBUG
    pr_nsblock(log);
#endif

    inc_nrnb(nrnb, eNR_NS, nsearch);
    /* inc_nrnb(nrnb,eNR_LR,fr->nlr); */

    return nsearch;
}

int natoms_beyond_ns_buffer(t_inputrec *ir, t_forcerec *fr, t_block *cgs,
                            matrix scale_tot, rvec *x)
{
    int  cg0, cg1, cg, a0, a1, a, i, j;
    real rint, hbuf2, scale;
    rvec *cg_cm, cgsc;
    gmx_bool bIsotropic;
    int  nBeyond;

    nBeyond = 0;

    rint = max(ir->rcoulomb, ir->rvdw);
    if (ir->rlist < rint)
    {
        gmx_fatal(FARGS, "The neighbor search buffer has negative size: %f nm",
                  ir->rlist - rint);
    }
    cg_cm = fr->cg_cm;

    cg0 = fr->cg0;
    cg1 = fr->hcg;

    if (!EI_DYNAMICS(ir->eI) || !DYNAMIC_BOX(*ir))
    {
        hbuf2 = sqr(0.5*(ir->rlist - rint));
        for (cg = cg0; cg < cg1; cg++)
        {
            a0 = cgs->index[cg];
            a1 = cgs->index[cg+1];
            for (a = a0; a < a1; a++)
            {
                if (distance2(cg_cm[cg], x[a]) > hbuf2)
                {
                    nBeyond++;
                }
            }
        }
    }
    else
    {
        bIsotropic = TRUE;
        scale      = scale_tot[0][0];
        for (i = 1; i < DIM; i++)
        {
            /* With anisotropic scaling, the original spherical ns volumes become
             * ellipsoids. To avoid costly transformations we use the minimum
             * eigenvalue of the scaling matrix for determining the buffer size.
             * Since the lower half is 0, the eigenvalues are the diagonal elements.
             */
            scale = min(scale, scale_tot[i][i]);
            if (scale_tot[i][i] != scale_tot[i-1][i-1])
            {
                bIsotropic = FALSE;
            }
            for (j = 0; j < i; j++)
            {
                if (scale_tot[i][j] != 0)
                {
                    bIsotropic = FALSE;
                }
            }
        }
        hbuf2 = sqr(0.5*(scale*ir->rlist - rint));
        if (bIsotropic)
        {
            for (cg = cg0; cg < cg1; cg++)
            {
                svmul(scale, cg_cm[cg], cgsc);
                a0 = cgs->index[cg];
                a1 = cgs->index[cg+1];
                for (a = a0; a < a1; a++)
                {
                    if (distance2(cgsc, x[a]) > hbuf2)
                    {
                        nBeyond++;
                    }
                }
            }
        }
        else
        {
            /* Anistropic scaling */
            for (cg = cg0; cg < cg1; cg++)
            {
                /* Since scale_tot contains the transpose of the scaling matrix,
                 * we need to multiply with the transpose.
                 */
                tmvmul_ur0(scale_tot, cg_cm[cg], cgsc);
                a0 = cgs->index[cg];
                a1 = cgs->index[cg+1];
                for (a = a0; a < a1; a++)
                {
                    if (distance2(cgsc, x[a]) > hbuf2)
                    {
                        nBeyond++;
                    }
                }
            }
        }
    }

    return nBeyond;
}
