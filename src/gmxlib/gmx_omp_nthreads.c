/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2010, The GROMACS development team,
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "gmx_fatal.h"
#include "typedefs.h"
#include "macros.h"
#include "gmx_omp.h"
#include "gmx_omp_nthreads.h"
#include "md_logging.h"

/*! Structure with the number of threads for each OpenMP multi-threaded
 *  algorithmic module in mdrun. */
typedef struct
{
    int      gnth;          /*! Global num. of threads per PP or PP+PME process/tMPI thread. */
    int      gnth_pme;      /*! Global num. of threads per PME only process/tMPI thread. */

    int      nth[emntNR];   /*! Number of threads for each module, indexed with module_nth_t */
    gmx_bool initialized;   /*! TRUE if the module as been initialized. */
} omp_module_nthreads_t;

/*! Names of environment variables to set the per module number of threads.
 *
 *  Indexed with the values of module_nth_t.
 * */
static const char *modth_env_var[emntNR] =
{
    "GMX_DEFAULT_NUM_THREADS should never be set",
    "GMX_DOMDEC_NUM_THREADS", "GMX_PAIRSEARCH_NUM_THREADS",
    "GMX_NONBONDED_NUM_THREADS", "GMX_BONDED_NUM_THREADS",
    "GMX_PME_NUM_THREADS", "GMX_UPDATE_NUM_THREADS",
    "GMX_VSITE_NUM_THREADS",
    "GMX_LINCS_NUM_THREADS", "GMX_SETTLE_NUM_THREADS"
};

/*! Names of the modules. */
static const char *mod_name[emntNR] =
{
    "default", "domain decomposition", "pair search", "non-bonded",
    "bonded", "PME", "update", "LINCS", "SETTLE"
};

/*! Number of threads for each algorithmic module.
 *
 *  File-scope global variable that gets set once in \init_module_nthreads
 *  and queried via gmx_omp_nthreads_get.
 *
 *  All fields are initialized to 0 which should result in errors if
 *  the init call is omitted.
 * */
static omp_module_nthreads_t modth = { 0, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0}, FALSE};


void gmx_omp_nthreads_init(FILE *fplog, t_commrec *cr,
                           int omp_nthreads_req,
                           int omp_nthreads_pme_req,
                           gmx_bool bThisNodePMEOnly,
                           gmx_bool bFullOmpSupport)
{
    modth.gnth = gmx_omp_get_max_threads();
    modth.gnth_pme = 0;

    /* now set the per-module values */
    modth.nth[emntDefault]    = modth.gnth;
    modth.nth[emntPairsearch] = modth.gnth;
    modth.nth[emntNonbonded]  = modth.gnth;
    modth.nth[emntBonded]     = modth.gnth;
    modth.nth[emntPME]        = modth.gnth;

    /* set the number of threads globally */
    gmx_omp_set_num_threads(modth.gnth);
    modth.initialized = TRUE;

    printf("Using %d OpenMP threads per tMPI thread\n",modth.gnth);
}

int gmx_omp_nthreads_get(int mod)
{
    if (mod < 0 || mod >= emntNR)
    {
        /* invalid module queried */
        return -1;
    }
    else
    { // calling enum module_nth 0,2,3,4, and (5 for pme)
        return modth.nth[mod];
    }
}
