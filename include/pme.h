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

#ifndef _pme_h
#define _pme_h

#include <stdio.h>
#include "visibility.h"
#include "typedefs.h"
#include "gmxcomplex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef real *splinevec[DIM];


GMX_LIBMD_EXPORT
int gmx_pme_init(gmx_pme_t *pmedata, t_commrec *cr,
                 int nnodes_major, int nnodes_minor,
                 t_inputrec *ir, int homenr,
                 gmx_bool bFreeEnergy, gmx_bool bReproducible, int nthread);
/* Initialize the pme data structures resepectively.
 * Return value 0 indicates all well, non zero is an error code.
 */



#define GMX_PME_SPREAD_Q      (1<<0)
#define GMX_PME_SOLVE         (1<<1)
#define GMX_PME_CALC_F        (1<<2)
#define GMX_PME_CALC_ENER_VIR (1<<3)
/* This forces the grid to be backtransformed even without GMX_PME_CALC_F */
#define GMX_PME_CALC_POT      (1<<4)
#define GMX_PME_DO_ALL_F  (GMX_PME_SPREAD_Q | GMX_PME_SOLVE | GMX_PME_CALC_F)

int gmx_pme_do(gmx_pme_t pme,
               int start,       int homenr,
               rvec x[],        rvec f[],
               real chargeA[],  real chargeB[],
               matrix box,      t_commrec *cr,
               int  maxshift_x, int maxshift_y,
               matrix lrvir,    real ewaldcoeff,
               real *energy,    real lambda,
               real *dvdlambda, int flags);
/* Do a PME calculation for the long range electrostatics.
 * flags, defined above, determine which parts of the calculation are performed.
 * Return value 0 indicates all well, non zero is an error code.
 */

#ifdef __cplusplus
}
#endif

#endif
