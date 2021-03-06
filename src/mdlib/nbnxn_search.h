/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2012, The GROMACS development team,
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

#ifndef _nbnxn_search_h
#define _nsnxn_search_h
#include "visibility.h"
#include "typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif


/* Allocates and initializes a pair search data structure */
void nbnxn_init_search(nbnxn_search_t    * nbs_ptr,
                       ivec               *n_dd_cells,
                       int                 nthread_max);

/* Put the atoms on the pair search grid.
 * Only atoms a0 to a1 in x are put on the grid.
 * The atom_density is used to determine the grid size.
 * When atom_density=-1, the density is determined from a1-a0 and the corners.
 * With domain decomposition part of the n particles might have migrated,
 * but have not been removed yet. This count is given by nmoved.
 * When move[i] < 0 particle i has migrated and will not be put on the grid.
 * Without domain decomposition move will be NULL.
 */
void nbnxn_put_on_grid(nbnxn_search_t nbs,
                       int ePBC, matrix box,
                       int dd_zone,
                       rvec corner0, rvec corner1,
                       int a0, int a1,
                       real atom_density,
                       const int *atinfo,
                       rvec *x,
                       int nmoved, int *move,
                       int nb_kernel_type,
                       nbnxn_atomdata_t *nbat);




/* Initializes a set of pair lists stored in nbnxn_pairlist_set_t */
void nbnxn_init_pairlist_set(nbnxn_pairlist_set_t *nbl_list,
                             gmx_bool simple, gmx_bool combined,
                             nbnxn_alloc_t *alloc,
                             nbnxn_free_t  *free);

/* Make a apir-list with radius rlist, store it in nbl.
 * The parameter min_ci_balanced sets the minimum required
 * number or roughly equally sized ci blocks in nbl.
 * When set >0 ci lists will be chopped up when the estimate
 * for the number of equally sized lists is below min_ci_balanced.
 */
void nbnxn_make_pairlist(const nbnxn_search_t  nbs,
                         nbnxn_atomdata_t     *nbat,
                         const t_blocka       *excl,
                         real                  rlist,
                         int                   min_ci_balanced,
                         nbnxn_pairlist_set_t *nbl_list,
                         int                   iloc,
                         int                   nb_kernel_type);

#ifdef __cplusplus
}
#endif

#endif
