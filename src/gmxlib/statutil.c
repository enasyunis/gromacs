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


#include <ctype.h>
#include <assert.h>
#include "copyrite.h"
#include "sysstuff.h"
#include "macros.h"
#include "string2.h"
#include "smalloc.h"
#include "pbc.h"
#include "statutil.h"
#include "names.h"
#include "vec.h"
#include "futil.h"
#include "wman.h"
#include "tpxio.h"
#include "gmx_fatal.h"
#include "network.h"
#include "vec.h"
#include "mtop_util.h"
#include "gmxfio.h"

#ifdef GMX_THREAD_MPI
#include "thread_mpi.h"
#endif

/* used for npri */
#ifdef __sgi
#include <sys/schedctl.h>
#include <sys/sysmp.h>
#endif

/* The source code in this file should be thread-safe.
      Please keep it that way. */

/******************************************************************
 *
 *             T R A J E C T O R Y   S T U F F
 *
 ******************************************************************/

/* inherently globally shared names: */
static const char *program_name = NULL;
static char       *cmd_line     = NULL;

#ifdef GMX_THREAD_MPI
/* For now, some things here are simply not re-entrant, so
   we have to actively lock them. */
static tMPI_Thread_mutex_t init_mutex = TMPI_THREAD_MUTEX_INITIALIZER;
#endif


/****************************************************************
 *
 *            E X P O R T E D   F U N C T I O N S
 *
 ****************************************************************/


/* progam names, etc. */

const char *ShortProgram(void)
{
    const char *pr, *ret;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&init_mutex);
#endif
    pr = ret = program_name;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&init_mutex);
#endif
    if ((pr = strrchr(ret, DIR_SEPARATOR)) != NULL)
    {
        ret = pr+1;
    }
    return ret;
}

const char *Program(void)
{
    const char *ret;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&init_mutex);
#endif
    ret = program_name;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&init_mutex);
#endif
    return ret;
}


