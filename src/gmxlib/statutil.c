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

const char *command_line(void)
{
    const char *ret;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&init_mutex);
#endif
    ret = cmd_line;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&init_mutex);
#endif
    return ret;
}

void set_program_name(const char *argvzero)
{
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&init_mutex);
#endif
    if (program_name == NULL)
    {
        /* if filename has file ending (e.g. .exe) then strip away */
        char* extpos = strrchr(argvzero, '.');
        if (extpos > strrchr(argvzero, DIR_SEPARATOR))
        {
            program_name = gmx_strndup(argvzero, extpos-argvzero);
        }
        else
        {
            program_name = gmx_strdup(argvzero);
        }
    }
    if (program_name == NULL)
    {
        program_name = "GROMACS";
    }
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&init_mutex);
#endif
}



/* utility functions */

gmx_bool bRmod_fd(double a, double b, double c, gmx_bool bDouble)
{
    int    iq;
    double tol;

    tol = 2*(bDouble ? GMX_DOUBLE_EPS : GMX_FLOAT_EPS);

    iq = (a - b + tol*a)/c;

    if (fabs(a - b - c*iq) <= tol*fabs(a))
    {
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}

int check_times2(real t, real t0, real tp, real tpp, gmx_bool bDouble)
{
    int  r;
    real margin;

#ifndef GMX_DOUBLE
    /* since t is float, we can not use double precision for bRmod */
    bDouble = FALSE;
#endif

    if (t-tp > 0 && tp-tpp > 0)
    {
        margin = 0.1*min(t-tp, tp-tpp);
    }
    else
    {
        margin = 0;
    }

    r = -1;
    if ((!bTimeSet(TBEGIN) || (t >= rTimeValue(TBEGIN)))  &&
        (!bTimeSet(TEND)   || (t <= rTimeValue(TEND))))
    {
        if (bTimeSet(TDELTA) && !bRmod_fd(t, t0, rTimeValue(TDELTA), bDouble))
        {
            r = -1;
        }
        else
        {
            r = 0;
        }
    }
    else if (bTimeSet(TEND) && (t >= rTimeValue(TEND)))
    {
        r = 1;
    }
    if (debug)
    {
        fprintf(debug, "t=%g, t0=%g, b=%g, e=%g, dt=%g: r=%d\n",
                t, t0, rTimeValue(TBEGIN), rTimeValue(TEND), rTimeValue(TDELTA), r);
    }
    return r;
}

int check_times(real t)
{
    return check_times2(t, t, t, t, FALSE);
}



/*************************************************************
 *
 *           P A R S I N G   S T U F F
 *
 *************************************************************/

static void usage(const char *type, const char *arg)
{
    assert(arg);
    gmx_fatal(FARGS, "Expected %s argument for option %s\n", type, arg);
}

int iscan(int argc, char *argv[], int *i)
{
    int var = 0;

    if (argc > (*i)+1)
    {
        if (!sscanf(argv[++(*i)], "%d", &var))
        {
            usage("an integer", argv[(*i)-1]);
        }
    }
    else
    {
        usage("an integer", argv[*i]);
    }

    return var;
}

gmx_large_int_t istepscan(int argc, char *argv[], int *i)
{
    gmx_large_int_t var = 0;

    if (argc > (*i)+1)
    {
        if (!sscanf(argv[++(*i)], gmx_large_int_pfmt, &var))
        {
            usage("an integer", argv[(*i)-1]);
        }
    }
    else
    {
        usage("an integer", argv[*i]);
    }

    return var;
}

double dscan(int argc, char *argv[], int *i)
{
    double var = 0;

    if (argc > (*i)+1)
    {
        if (!sscanf(argv[++(*i)], "%lf", &var))
        {
            usage("a real", argv[(*i)-1]);
        }
    }
    else
    {
        usage("a real", argv[*i]);
    }

    return var;
}

