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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef HAVE_DIRENT_H
/* POSIX */
#include <dirent.h>
#endif



#include "sysstuff.h"
#include "string2.h"
#include "futil.h"
#include "network.h"
#include "gmx_fatal.h"
#include "smalloc.h"


#ifdef GMX_THREAD_MPI
#include "thread_mpi.h"
#endif

/* Windows file stuff, only necessary for visual studio */
#ifdef _MSC_VER
#include "windows.h"
#endif

/* we keep a linked list of all files opened through pipes (i.e.
   compressed or .gzipped files. This way we can distinguish between them
   without having to change the semantics of reading from/writing to files)
 */
typedef struct t_pstack {
    FILE            *fp;
    struct t_pstack *prev;
} t_pstack;

static t_pstack    *pstack      = NULL;
static gmx_bool     bUnbuffered = FALSE;

#ifdef GMX_THREAD_MPI
/* this linked list is an intrinsically globally shared object, so we have
   to protect it with mutexes */
static tMPI_Thread_mutex_t pstack_mutex = TMPI_THREAD_MUTEX_INITIALIZER;
#endif

void no_buffers(void)
{
    bUnbuffered = TRUE;
}

void push_ps(FILE *fp)
{
    t_pstack *ps;

#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&pstack_mutex);
#endif

    snew(ps, 1);
    ps->fp   = fp;
    ps->prev = pstack;
    pstack   = ps;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&pstack_mutex);
#endif
}

#ifdef ffclose
#undef ffclose
#endif

#ifndef HAVE_PIPES
static FILE *popen(const char *nm, const char *mode)
{
    gmx_impl("Sorry no pipes...");

    return NULL;
}

static int pclose(FILE *fp)
{
    gmx_impl("Sorry no pipes...");

    return 0;
}
#endif

int ffclose(FILE *fp)
{
#ifdef SKIP_FFOPS
    return fclose(fp);
#else
    t_pstack *ps, *tmp;
    int       ret = 0;
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_lock(&pstack_mutex);
#endif

    ps = pstack;
    if (ps == NULL)
    {
        if (fp != NULL)
        {
            ret = fclose(fp);
        }
    }
    else if (ps->fp == fp)
    {
        if (fp != NULL)
        {
            ret = pclose(fp);
        }
        pstack = pstack->prev;
        sfree(ps);
    }
    else
    {
        while ((ps->prev != NULL) && (ps->prev->fp != fp))
        {
            ps = ps->prev;
        }
        if ((ps->prev != NULL) && ps->prev->fp == fp)
        {
            if (ps->prev->fp != NULL)
            {
                ret = pclose(ps->prev->fp);
            }
            tmp      = ps->prev;
            ps->prev = ps->prev->prev;
            sfree(tmp);
        }
        else
        {
            if (fp != NULL)
            {
                ret = fclose(fp);
            }
        }
    }
#ifdef GMX_THREAD_MPI
    tMPI_Thread_mutex_unlock(&pstack_mutex);
#endif
    return ret;
#endif
}


#ifdef rewind
#undef rewind
#endif


int gmx_fseek(FILE *stream, gmx_off_t offset, int whence)
{
#ifdef HAVE_FSEEKO
    return fseeko(stream, offset, whence);
#else
#ifdef HAVE__FSEEKI64
    return _fseeki64(stream, offset, whence);
#else
    return fseek(stream, offset, whence);
#endif
#endif
}


gmx_off_t gmx_ftell(FILE *stream)
{
#ifdef HAVE_FSEEKO
    return ftello(stream);
#else
#ifdef HAVE__FSEEKI64
    return _ftelli64(stream);
#else
    return ftell(stream);
#endif
#endif
}

static FILE *uncompress(const char *fn, const char *mode)
{
    FILE *fp;
    char  buf[256];

    sprintf(buf, "uncompress -c < %s", fn);
    fprintf(stderr, "Going to execute '%s'\n", buf);
    if ((fp = popen(buf, mode)) == NULL)
    {
        gmx_open(fn);
    }
    push_ps(fp);

    return fp;
}

static FILE *gunzip(const char *fn, const char *mode)
{
    FILE *fp;
    char  buf[256];

    sprintf(buf, "gunzip -c < %s", fn);
    fprintf(stderr, "Going to execute '%s'\n", buf);
    if ((fp = popen(buf, mode)) == NULL)
    {
        gmx_open(fn);
    }
    push_ps(fp);

    return fp;
}

gmx_bool gmx_fexist(const char *fname)
{
    FILE *test;

    if (fname == NULL)
    {
        return FALSE;
    }
    test = fopen(fname, "r");
    if (test == NULL)
    {
        /*Windows doesn't allow fopen of directory - so we need to check this seperately */
        return FALSE;
    }
    else
    {
        fclose(test);
        return TRUE;
    }
}

static char *backup_fn(const char *file, int count_max)
{
    /* Use a reasonably low value for countmax; we might
     * generate 4-5 files in each round, and we dont
     * want to hit directory limits of 1024 or 2048 files.
     */
#define COUNTMAX 99
    int          i, count = 1;
    char        *directory, *fn;
    char        *buf;

    if (count_max == -1)
    {
        count_max = COUNTMAX;
    }

    smalloc(buf, GMX_PATH_MAX);

    for (i = strlen(file)-1; ((i > 0) && (file[i] != DIR_SEPARATOR)); i--)
    {
        ;
    }
    /* Must check whether i > 0, i.e. whether there is a directory
     * in the file name. In that case we overwrite the / sign with
     * a '\0' to end the directory string .
     */
    if (i > 0)
    {
        directory    = gmx_strdup(file);
        directory[i] = '\0';
        fn           = gmx_strdup(file+i+1);
    }
    else
    {
        directory    = gmx_strdup(".");
        fn           = gmx_strdup(file);
    }
    do
    {
        sprintf(buf, "%s/#%s.%d#", directory, fn, count);
        count++;
    }
    while ((count <= count_max) && gmx_fexist(buf));

    /* Arbitrarily bail out */
    if (count > count_max)
    {
        gmx_fatal(FARGS, "Won't make more than %d backups of %s for you.\n"
                  "The env.var. GMX_MAXBACKUP controls this maximum, -1 disables backups.",
                  count_max, fn);
    }

    sfree(directory);
    sfree(fn);

    return buf;
}

gmx_bool make_backup(const char * name)
{
    char * env;
    int    count_max;
    char * backup;


    if (gmx_fexist(name))
    {
        env = getenv("GMX_MAXBACKUP");
        if (env != NULL)
        {
            count_max = 0;
            sscanf(env, "%d", &count_max);
            if (count_max == -1)
            {
                /* Do not make backups and possibly overwrite old files */
                return TRUE;
            }
        }
        else
        {
            /* Use the default maximum */
            count_max = -1;
        }
        backup = backup_fn(name, count_max);
        if (rename(name, backup) == 0)
        {
            fprintf(stderr, "\nBack Off! I just backed up %s to %s\n",
                    name, backup);
        }
        else
        {
            fprintf(stderr, "Sorry couldn't backup %s to %s\n", name, backup);
            return FALSE;
        }
        sfree(backup);
    }
    return TRUE;
}

FILE *ffopen(const char *file, const char *mode)
{
#ifdef SKIP_FFOPS
    return fopen(file, mode);
#else
    FILE    *ff = NULL;
    char     buf[256], *bf, *bufsize = 0, *ptr;
    gmx_bool bRead;
    int      bs;

    if (file == NULL)
    {
        return NULL;
    }

    if (mode[0] == 'w')
    {
        make_backup(file);
    }
    where();

    bRead = (mode[0] == 'r' && mode[1] != '+');
    strcpy(buf, file);
    if (!bRead || gmx_fexist(buf))
    {
        if ((ff = fopen(buf, mode)) == NULL)
        {
            gmx_file(buf);
        }
        where();
        /* Check whether we should be using buffering (default) or not
         * (for debugging)
         */
        if (bUnbuffered || ((bufsize = getenv("LOG_BUFS")) != NULL))
        {
            /* Check whether to use completely unbuffered */
            if (bUnbuffered)
            {
                bs = 0;
            }
            else
            {
                bs = strtol(bufsize, NULL, 10);
            }
            if (bs <= 0)
            {
                setbuf(ff, NULL);
            }
            else
            {
                snew(ptr, bs+8);
                if (setvbuf(ff, ptr, _IOFBF, bs) != 0)
                {
                    gmx_file("Buffering File");
                }
            }
        }
        where();
    }
    else
    {
        sprintf(buf, "%s.Z", file);
        if (gmx_fexist(buf))
        {
            ff = uncompress(buf, mode);
        }
        else
        {
            sprintf(buf, "%s.gz", file);
            if (gmx_fexist(buf))
            {
                ff = gunzip(buf, mode);
            }
            else
            {
                gmx_file(file);
            }
        }
    }
    return ff;
#endif
}

/* Our own implementation of dirent-like functionality to scan directories. */
struct gmx_directory
{
#ifdef HAVE_DIRENT_H
    DIR  *               dirent_handle;
#else
    int                  dummy;
#endif
};







