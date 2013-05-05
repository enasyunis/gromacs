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
#include "sysstuff.h"
#include "typedefs.h"
#include "vec.h"
#include "maths.h"
#include "pbc.h"
#include "smalloc.h"
#include "txtdump.h"
#include "gmx_fatal.h"
#include "names.h"
#include "gmx_omp_nthreads.h"

/* Skip 0 so we have more chance of detecting if we forgot to call set_pbc. */
enum {
    epbcdxRECTANGULAR = 1, epbcdxTRICLINIC,
    epbcdx2D_RECT,       epbcdx2D_TRIC,
    epbcdx1D_RECT,       epbcdx1D_TRIC,
    epbcdxSCREW_RECT,    epbcdxSCREW_TRIC,
    epbcdxNOPBC,         epbcdxUNSUPPORTED
};

/* Margin factor for error message and correction if the box is too skewed */
#define BOX_MARGIN         1.0010
#define BOX_MARGIN_CORRECT 1.0005


const char *check_box(int ePBC, matrix box)
{
    const char *ptr;

    if (ePBC == -1)
    {
        ePBC = guess_ePBC(box);
    }

    if (ePBC == epbcNONE)
    {
        return NULL;
    }

    if ((box[XX][YY] != 0) || (box[XX][ZZ] != 0) || (box[YY][ZZ] != 0))
    {
        ptr = "Only triclinic boxes with the first vector parallel to the x-axis and the second vector in the xy-plane are supported.";
    }
    else if (ePBC == epbcSCREW && (box[YY][XX] != 0 || box[ZZ][XX] != 0))
    {
        ptr = "The unit cell can not have off-diagonal x-components with screw pbc";
    }
    else if (fabs(box[YY][XX]) > BOX_MARGIN*0.5*box[XX][XX] ||
             (ePBC != epbcXY &&
              (fabs(box[ZZ][XX]) > BOX_MARGIN*0.5*box[XX][XX] ||
               fabs(box[ZZ][YY]) > BOX_MARGIN*0.5*box[YY][YY])))
    {
        ptr = "Triclinic box is too skewed.";
    }
    else
    {
        ptr = NULL;
    }

    return ptr;
}

real max_cutoff2(int ePBC, matrix box)
{
    real min_hv2, min_ss;

    /* Physical limitation of the cut-off
     * by half the length of the shortest box vector.
     */
    min_hv2 = min(0.25*norm2(box[XX]), 0.25*norm2(box[YY]));
    if (ePBC != epbcXY)
    {
        min_hv2 = min(min_hv2, 0.25*norm2(box[ZZ]));
    }

    /* Limitation to the smallest diagonal element due to optimizations:
     * checking only linear combinations of single box-vectors (2 in x)
     * in the grid search and pbc_dx is a lot faster
     * than checking all possible combinations.
     */
    if (ePBC == epbcXY)
    {
        min_ss = min(box[XX][XX], box[YY][YY]);
    }
    else
    {
        min_ss = min(box[XX][XX], min(box[YY][YY]-fabs(box[ZZ][YY]), box[ZZ][ZZ]));
    }

    return min(min_hv2, min_ss*min_ss);
}

/* this one is mostly harmless... */
static gmx_bool bWarnedGuess = FALSE;

int guess_ePBC(matrix box)
{
    int ePBC;

    if (box[XX][XX] > 0 && box[YY][YY] > 0 && box[ZZ][ZZ] > 0)
    {
        ePBC = epbcXYZ;
    }
    else if (box[XX][XX] > 0 && box[YY][YY] > 0 && box[ZZ][ZZ] == 0)
    {
        ePBC = epbcXY;
    }
    else if (box[XX][XX] == 0 && box[YY][YY] == 0 && box[ZZ][ZZ] == 0)
    {
        ePBC = epbcNONE;
    }
    else
    {
        if (!bWarnedGuess)
        {
            fprintf(stderr, "WARNING: Unsupported box diagonal %f %f %f, "
                    "will not use periodic boundary conditions\n\n",
                    box[XX][XX], box[YY][YY], box[ZZ][ZZ]);
            bWarnedGuess = TRUE;
        }
        ePBC = epbcNONE;
    }

    if (debug)
    {
        fprintf(debug, "Guessed pbc = %s from the box matrix\n", epbc_names[ePBC]);
    }

    return ePBC;
}

static int correct_box_elem(FILE *fplog, int step, tensor box, int v, int d)
{
    int shift, maxshift = 10;

    shift = 0;

    /* correct elem d of vector v with vector d */
    while (box[v][d] > BOX_MARGIN_CORRECT*0.5*box[d][d])
    {
        if (fplog)
        {
            fprintf(fplog, "Step %d: correcting invalid box:\n", step);
            pr_rvecs(fplog, 0, "old box", box, DIM);
        }
        rvec_dec(box[v], box[d]);
        shift--;
        if (fplog)
        {
            pr_rvecs(fplog, 0, "new box", box, DIM);
        }
        if (shift <= -maxshift)
        {
            gmx_fatal(FARGS,
                      "Box was shifted at least %d times. Please see log-file.",
                      maxshift);
        }
    }
    while (box[v][d] < -BOX_MARGIN_CORRECT*0.5*box[d][d])
    {
        if (fplog)
        {
            fprintf(fplog, "Step %d: correcting invalid box:\n", step);
            pr_rvecs(fplog, 0, "old box", box, DIM);
        }
        rvec_inc(box[v], box[d]);
        shift++;
        if (fplog)
        {
            pr_rvecs(fplog, 0, "new box", box, DIM);
        }
        if (shift >= maxshift)
        {
            gmx_fatal(FARGS,
                      "Box was shifted at least %d times. Please see log-file.",
                      maxshift);
        }
    }

    return shift;
}

gmx_bool correct_box(FILE *fplog, int step, tensor box, t_graph *graph)
{
    int      zy, zx, yx, i;
    gmx_bool bCorrected;

    /* check if the box still obeys the restrictions, if not, correct it */
    zy = correct_box_elem(fplog, step, box, ZZ, YY);
    zx = correct_box_elem(fplog, step, box, ZZ, XX);
    yx = correct_box_elem(fplog, step, box, YY, XX);

    bCorrected = (zy || zx || yx);

    if (bCorrected && graph)
    {
        /* correct the graph */
        for (i = graph->at_start; i < graph->at_end; i++)
        {
            graph->ishift[i][YY] -= graph->ishift[i][ZZ]*zy;
            graph->ishift[i][XX] -= graph->ishift[i][ZZ]*zx;
            graph->ishift[i][XX] -= graph->ishift[i][YY]*yx;
        }
    }

    return bCorrected;
}


static void low_set_pbc(t_pbc *pbc, int ePBC, ivec *dd_nc, matrix box)
{
    int         order[5] = {0, -1, 1, -2, 2};
    int         ii, jj, kk, i, j, k, d, dd, jc, kc, npbcdim, shift;
    ivec        bPBC;
    real        d2old, d2new, d2new_c;
    rvec        trial, pos;
    gmx_bool    bXY, bUse;
    const char *ptr;

    pbc->ndim_ePBC =3; 

    copy_mat(box, pbc->box);
    pbc->bLimitDistance = FALSE;
    pbc->max_cutoff2    = 0;
    pbc->dim            = -1;

    for (i = 0; (i < DIM); i++)
    {
        pbc->fbox_diag[i]  =  box[i][i];
        pbc->hbox_diag[i]  =  pbc->fbox_diag[i]*0.5;
        pbc->mhbox_diag[i] = -pbc->hbox_diag[i];
    }

    ptr = check_box(ePBC, box);
    if (ePBC == epbcNONE)
    {
        pbc->ePBCDX = epbcdxNOPBC;
    }
    else if (ptr)
    {
        fprintf(stderr,   "Warning: %s\n", ptr);
        pr_rvecs(stderr, 0, "         Box", box, DIM);
        fprintf(stderr,   "         Can not fix pbc.\n");
        pbc->ePBCDX          = epbcdxUNSUPPORTED;
        pbc->bLimitDistance  = TRUE;
        pbc->limit_distance2 = 0;
    }
    else
    {
        if (ePBC == epbcSCREW && dd_nc)
        {
            /* This combinated should never appear here */
            gmx_incons("low_set_pbc called with screw pbc and dd_nc != NULL");
        }

        npbcdim = 0;
        for (i = 0; i < DIM; i++)
        {
            if ((dd_nc && (*dd_nc)[i] > 1) || (ePBC == epbcXY && i == ZZ))
            {
                bPBC[i] = 0;
            }
            else
            {
                bPBC[i] = 1;
                npbcdim++;
            }
        }
        switch (npbcdim)
        {
            case 1:
                /* 1D pbc is not an mdp option and it is therefore only used
                 * with single shifts.
                 */
                pbc->ePBCDX = epbcdx1D_RECT;
                for (i = 0; i < DIM; i++)
                {
                    if (bPBC[i])
                    {
                        pbc->dim = i;
                    }
                }
                for (i = 0; i < pbc->dim; i++)
                {
                    if (pbc->box[pbc->dim][i] != 0)
                    {
                        pbc->ePBCDX = epbcdx1D_TRIC;
                    }
                }
                break;
            case 2:
                pbc->ePBCDX = epbcdx2D_RECT;
                for (i = 0; i < DIM; i++)
                {
                    if (!bPBC[i])
                    {
                        pbc->dim = i;
                    }
                }
                for (i = 0; i < DIM; i++)
                {
                    if (bPBC[i])
                    {
                        for (j = 0; j < i; j++)
                        {
                            if (pbc->box[i][j] != 0)
                            {
                                pbc->ePBCDX = epbcdx2D_TRIC;
                            }
                        }
                    }
                }
                break;
            case 3:
                if (ePBC != epbcSCREW)
                {
                    if (TRICLINIC(box))
                    {
                        pbc->ePBCDX = epbcdxTRICLINIC;
                    }
                    else
                    {
                        pbc->ePBCDX = epbcdxRECTANGULAR;
                    }
                }
                else
                {
                    pbc->ePBCDX = (box[ZZ][YY] == 0 ? epbcdxSCREW_RECT : epbcdxSCREW_TRIC);
                    if (pbc->ePBCDX == epbcdxSCREW_TRIC)
                    {
                        fprintf(stderr,
                                "Screw pbc is not yet implemented for triclinic boxes.\n"
                                "Can not fix pbc.\n");
                        pbc->ePBCDX = epbcdxUNSUPPORTED;
                    }
                }
                break;
            default:
                gmx_fatal(FARGS, "Incorrect number of pbc dimensions with DD: %d",
                          npbcdim);
        }
        pbc->max_cutoff2 = max_cutoff2(ePBC, box);

        if (pbc->ePBCDX == epbcdxTRICLINIC ||
            pbc->ePBCDX == epbcdx2D_TRIC ||
            pbc->ePBCDX == epbcdxSCREW_TRIC)
        {
            if (debug)
            {
                pr_rvecs(debug, 0, "Box", box, DIM);
                fprintf(debug, "max cutoff %.3f\n", sqrt(pbc->max_cutoff2));
            }
            pbc->ntric_vec = 0;
            /* We will only use single shifts, but we will check a few
             * more shifts to see if there is a limiting distance
             * above which we can not be sure of the correct distance.
             */
            for (kk = 0; kk < 5; kk++)
            {
                k = order[kk];
                if (!bPBC[ZZ] && k != 0)
                {
                    continue;
                }
                for (jj = 0; jj < 5; jj++)
                {
                    j = order[jj];
                    if (!bPBC[YY] && j != 0)
                    {
                        continue;
                    }
                    for (ii = 0; ii < 3; ii++)
                    {
                        i = order[ii];
                        if (!bPBC[XX] && i != 0)
                        {
                            continue;
                        }
                        /* A shift is only useful when it is trilinic */
                        if (j != 0 || k != 0)
                        {
                            d2old = 0;
                            d2new = 0;
                            for (d = 0; d < DIM; d++)
                            {
                                trial[d] = i*box[XX][d] + j*box[YY][d] + k*box[ZZ][d];
                                /* Choose the vector within the brick around 0,0,0 that
                                 * will become the shortest due to shift try.
                                 */
                                if (d == pbc->dim)
                                {
                                    trial[d] = 0;
                                    pos[d]   = 0;
                                }
                                else
                                {
                                    if (trial[d] < 0)
                                    {
                                        pos[d] = min( pbc->hbox_diag[d], -trial[d]);
                                    }
                                    else
                                    {
                                        pos[d] = max(-pbc->hbox_diag[d], -trial[d]);
                                    }
                                }
                                d2old += sqr(pos[d]);
                                d2new += sqr(pos[d] + trial[d]);
                            }
                            if (BOX_MARGIN*d2new < d2old)
                            {
                                if (j < -1 || j > 1 || k < -1 || k > 1)
                                {
                                    /* Check if there is a single shift vector
                                     * that decreases this distance even more.
                                     */
                                    jc = 0;
                                    kc = 0;
                                    if (j < -1 || j > 1)
                                    {
                                        jc = j/2;
                                    }
                                    if (k < -1 || k > 1)
                                    {
                                        kc = k/2;
                                    }
                                    d2new_c = 0;
                                    for (d = 0; d < DIM; d++)
                                    {
                                        d2new_c += sqr(pos[d] + trial[d]
                                                       - jc*box[YY][d] - kc*box[ZZ][d]);
                                    }
                                    if (d2new_c > BOX_MARGIN*d2new)
                                    {
                                        /* Reject this shift vector, as there is no a priori limit
                                         * to the number of shifts that decrease distances.
                                         */
                                        if (!pbc->bLimitDistance || d2new <  pbc->limit_distance2)
                                        {
                                            pbc->limit_distance2 = d2new;
                                        }
                                        pbc->bLimitDistance = TRUE;
                                    }
                                }
                                else
                                {
                                    /* Check if shifts with one box vector less do better */
                                    bUse = TRUE;
                                    for (dd = 0; dd < DIM; dd++)
                                    {
                                        shift = (dd == 0 ? i : (dd == 1 ? j : k));
                                        if (shift)
                                        {
                                            d2new_c = 0;
                                            for (d = 0; d < DIM; d++)
                                            {
                                                d2new_c += sqr(pos[d] + trial[d] - shift*box[dd][d]);
                                            }
                                            if (d2new_c <= BOX_MARGIN*d2new)
                                            {
                                                bUse = FALSE;
                                            }
                                        }
                                    }
                                    if (bUse)
                                    {
                                        /* Accept this shift vector. */
                                        if (pbc->ntric_vec >= MAX_NTRICVEC)
                                        {
                                            fprintf(stderr, "\nWARNING: Found more than %d triclinic correction vectors, ignoring some.\n"
                                                    "  There is probably something wrong with your box.\n", MAX_NTRICVEC);
                                            pr_rvecs(stderr, 0, "         Box", box, DIM);
                                        }
                                        else
                                        {
                                            copy_rvec(trial, pbc->tric_vec[pbc->ntric_vec]);
                                            pbc->tric_shift[pbc->ntric_vec][XX] = i;
                                            pbc->tric_shift[pbc->ntric_vec][YY] = j;
                                            pbc->tric_shift[pbc->ntric_vec][ZZ] = k;
                                            pbc->ntric_vec++;
                                        }
                                    }
                                }
                                if (debug)
                                {
                                    fprintf(debug, "  tricvec %2d = %2d %2d %2d  %5.2f %5.2f  %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f\n",
                                            pbc->ntric_vec, i, j, k,
                                            sqrt(d2old), sqrt(d2new),
                                            trial[XX], trial[YY], trial[ZZ],
                                            pos[XX], pos[YY], pos[ZZ]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void set_pbc(t_pbc *pbc, int ePBC, matrix box)
{
    if (ePBC == -1)
    {
        ePBC = guess_ePBC(box);
    }

    low_set_pbc(pbc, ePBC, NULL, box);
}

t_pbc *set_pbc_dd(t_pbc *pbc, int ePBC,
                  gmx_domdec_t *dd, gmx_bool bSingleDir, matrix box)
{
    ivec nc2;
    int  npbcdim, i;

    if (dd == NULL)
    {
        npbcdim = DIM;
    }
    else
    {
        if (ePBC == epbcSCREW && dd->nc[XX] > 1)
        {
            /* The rotation has been taken care of during coordinate communication */
            ePBC = epbcXYZ;
        }
        npbcdim = 0;
        for (i = 0; i < DIM; i++)
        {
            if (dd->nc[i] <= (bSingleDir ? 1 : 2))
            {
                nc2[i] = 1;
                if (!(ePBC == epbcXY && i == ZZ))
                {
                    npbcdim++;
                }
            }
            else
            {
                nc2[i] = dd->nc[i];
            }
        }
    }

    if (npbcdim > 0)
    {
        low_set_pbc(pbc, ePBC, npbcdim < DIM ? &nc2 : NULL, box);
    }

    return (npbcdim > 0 ? pbc : NULL);
}



void calc_shifts(matrix box, rvec shift_vec[])
{
    int k, l, m, d, n, test;

    n = 0;
    for (m = -D_BOX_Z; m <= D_BOX_Z; m++)
    {
        for (l = -D_BOX_Y; l <= D_BOX_Y; l++)
        {
            for (k = -D_BOX_X; k <= D_BOX_X; k++, n++)
            {
                for (d = 0; d < DIM; d++)
                {
                    shift_vec[n][d] = k*box[XX][d] + l*box[YY][d] + m*box[ZZ][d];
                }
            }
        }
    }
}


void put_atoms_in_box_omp(int ePBC, matrix box, int natoms, rvec x[])
{
    int t, nth;
    nth = gmx_omp_nthreads_get(emntDefault);

#pragma omp parallel for num_threads(nth) schedule(static)
    for (t = 0; t < nth; t++)
    {
        int offset, len;

        offset = (natoms*t    )/nth;
        len    = (natoms*(t + 1))/nth - offset;
        put_atoms_in_box(ePBC, box, len, x + offset);
    }
}

void put_atoms_in_box(int ePBC, matrix box, int natoms, rvec x[])
{
    int npbcdim, i, m, d;


    if (ePBC == epbcXY)
    {
        npbcdim = 2;
    }
    else
    {
        npbcdim = 3;
    }

    if (TRICLINIC(box))
    {
        for (i = 0; (i < natoms); i++)
        {
            for (m = npbcdim-1; m >= 0; m--)
            {
                while (x[i][m] < 0)
                {
                    for (d = 0; d <= m; d++)
                    {
                        x[i][d] += box[m][d];
                    }
                }
                while (x[i][m] >= box[m][m])
                {
                    for (d = 0; d <= m; d++)
                    {
                        x[i][d] -= box[m][d];
                    }
                }
            }
        }
    }
    else
    {
        for (i = 0; i < natoms; i++)
        {
            for (d = 0; d < npbcdim; d++)
            {
                while (x[i][d] < 0)
                {
                    x[i][d] += box[d][d];
                }
                while (x[i][d] >= box[d][d])
                {
                    x[i][d] -= box[d][d];
                }
            }
        }
    }
}

