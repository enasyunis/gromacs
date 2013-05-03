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


void set_command_line(int argc, char *argv[])
{
    int    i;
    size_t cmdlength;

#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&init_mutex);
#endif
    if (cmd_line == NULL)
    {
        cmdlength = strlen(argv[0]);
        for (i = 1; i < argc; i++)
        {
            cmdlength += strlen(argv[i]);
        }

        /* Fill the cmdline string */
        snew(cmd_line, cmdlength+argc+1);
        for (i = 0; i < argc; i++)
        {
            strcat(cmd_line, argv[i]);
            strcat(cmd_line, " ");
        }
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




static void set_default_time_unit(const char *time_list[], gmx_bool bCanTime)
{
    int         i      = 0, j;
    const char *select = NULL;

    if (bCanTime)
    {
        select = getenv("GMXTIMEUNIT");
        if (select != NULL)
        {
            i = 1;
            while (time_list[i] && strcmp(time_list[i], select) != 0)
            {
                i++;
            }
        }
    }
    if (!bCanTime || select == NULL ||
        time_list[i] == NULL || strcmp(time_list[i], select) != 0)
    {
        /* Set it to the default: ps */
        i = 1;
        while (time_list[i] && strcmp(time_list[i], "ps") != 0)
        {
            i++;
        }

    }
    time_list[0] = time_list[i];
}


static void set_default_xvg_format(const char *xvg_list[])
{
    int         i, j;
    const char *select, *tmp;

    select = getenv("GMX_VIEW_XVG");
    if (select == NULL)
    {
        /* The default is the first option */
        xvg_list[0] = xvg_list[1];
    }
    else
    {
        i = 1;
        while (xvg_list[i] && strcmp(xvg_list[i], select) != 0)
        {
            i++;
        }
        if (xvg_list[i] != NULL)
        {
            xvg_list[0] = xvg_list[i];
        }
        else
        {
            xvg_list[0] = xvg_list[exvgNONE];
        }
    }
}


/***** T O P O L O G Y   S T U F F ******/

t_topology *read_top(const char *fn, int *ePBC)
{
    int         epbc, natoms;
    t_topology *top;

    snew(top, 1);
    epbc = read_tpx_top(fn, NULL, NULL, &natoms, NULL, NULL, NULL, top);
    if (ePBC)
    {
        *ePBC = epbc;
    }

    return top;
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

char *sscan(int argc, char *argv[], int *i)
{
    if (argc > (*i)+1)
    {
        if ( (argv[(*i)+1][0] == '-') && (argc > (*i)+2) &&
             (argv[(*i)+2][0] != '-') )
        {
            fprintf(stderr, "Possible missing string argument for option %s\n\n",
                    argv[*i]);
        }
    }
    else
    {
        usage("a string", argv[*i]);
    }

    return argv[++(*i)];
}

int nenum(const char *const enumc[])
{
    int i;

    i = 1;
    /* we *can* compare pointers directly here! */
    while (enumc[i] && enumc[0] != enumc[i])
    {
        i++;
    }

    return i;
}

static void pdesc(char *desc)
{
    char *ptr, *nptr;

    ptr = desc;
    if ((int)strlen(ptr) < 70)
    {
        fprintf(stderr, "\t%s\n", ptr);
    }
    else
    {
        for (nptr = ptr+70; (nptr != ptr) && (!isspace(*nptr)); nptr--)
        {
            ;
        }
        if (nptr == ptr)
        {
            fprintf(stderr, "\t%s\n", ptr);
        }
        else
        {
            *nptr = '\0';
            nptr++;
            fprintf(stderr, "\t%s\n", ptr);
            pdesc(nptr);
        }
    }
}

static FILE *man_file(const output_env_t oenv, const char *mantp)
{
    FILE       *fp;
    char        buf[256];
    const char *pr = output_env_get_short_program_name(oenv);

    if (strcmp(mantp, "ascii") != 0)
    {
        sprintf(buf, "%s.%s", pr, mantp);
    }
    else
    {
        sprintf(buf, "%s.txt", pr);
    }
    fp = gmx_fio_fopen(buf, "w");

    return fp;
}

static int add_parg(int npargs, t_pargs *pa, t_pargs *pa_add)
{
    memcpy(&(pa[npargs]), pa_add, sizeof(*pa_add));

    return npargs+1;
}

static char *mk_desc(t_pargs *pa, const char *time_unit_str)
{
    char      *newdesc = NULL, *ndesc = NULL, *nptr = NULL;
    const char*ptr     = NULL;
    int        len, k;

    /* First compute length for description */
    len = strlen(pa->desc)+1;
    if ((ptr = strstr(pa->desc, "HIDDEN")) != NULL)
    {
        len += 4;
    }
    if (pa->type == etENUM)
    {
        len += 10;
        for (k = 1; (pa->u.c[k] != NULL); k++)
        {
            len += strlen(pa->u.c[k])+12;
        }
    }
    snew(newdesc, len);

    /* add label for hidden options */
    if (is_hidden(pa))
    {
        sprintf(newdesc, "[hidden] %s", ptr+6);
    }
    else
    {
        strcpy(newdesc, pa->desc);
    }

    /* change '%t' into time_unit */
#define TUNITLABEL "%t"
#define NTUNIT strlen(TUNITLABEL)
    if (pa->type == etTIME)
    {
        while ( (nptr = strstr(newdesc, TUNITLABEL)) != NULL)
        {
            nptr[0] = '\0';
            nptr   += NTUNIT;
            len    += strlen(time_unit_str)-NTUNIT;
            snew(ndesc, len);
            strcpy(ndesc, newdesc);
            strcat(ndesc, time_unit_str);
            strcat(ndesc, nptr);
            sfree(newdesc);
            newdesc = ndesc;
            ndesc   = NULL;
        }
    }
#undef TUNITLABEL
#undef NTUNIT

    /* Add extra comment for enumerateds */
    if (pa->type == etENUM)
    {
        strcat(newdesc, ": ");
        for (k = 1; (pa->u.c[k] != NULL); k++)
        {
            strcat(newdesc, "[TT]");
            strcat(newdesc, pa->u.c[k]);
            strcat(newdesc, "[tt]");
            /* Print a comma everywhere but at the last one */
            if (pa->u.c[k+1] != NULL)
            {
                if (pa->u.c[k+2] == NULL)
                {
                    strcat(newdesc, " or ");
                }
                else
                {
                    strcat(newdesc, ", ");
                }
            }
        }
    }
    return newdesc;
}


void parse_common_args(int *argc, char *argv[], unsigned long Flags,
                       int nfile, t_filenm fnm[], int npargs, t_pargs *pa,
                       int ndesc, const char **desc,
                       int nbugs, const char **bugs
                       )
{
  char *filename = getenv("FILENAME");
  printf("%s\n",filename);
  set_default_file_name(filename);
  parse_file_args(argc, argv, nfile, fnm, 0, 1);
  if (*argc > 1) gmx_cmd(argv[1]);

}
