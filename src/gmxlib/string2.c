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
/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "visibility.h"

#ifdef GMX_CRAY_XT3
#undef HAVE_PWD_H
#endif

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <time.h>

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif


#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#include <time.h>
#include <assert.h>

#include "typedefs.h"
#include "smalloc.h"
#include "gmx_fatal.h"
#include "macros.h"
#include "string2.h"
#include "futil.h"



int gmx_strcasecmp(const char *str1, const char *str2)
{
    char ch1, ch2;

    do
    {
        ch1 = toupper(*(str1++));
        ch2 = toupper(*(str2++));
        if (ch1 != ch2)
        {
            return (ch1-ch2);
        }
    }
    while (ch1);
    return 0;
}


char *gmx_strdup(const char *src)
{
    char *dest;

    snew(dest, strlen(src)+1);
    strcpy(dest, src);

    return dest;
}

/* Magic hash init number for Dan J. Bernsteins algorithm.
 * Do NOT use any other value unless you really know what you are doing.
 */
const unsigned int
    gmx_string_hash_init = 5381;



char *gmx_strsep(char **stringp, const char *delim)
{
    char *ret;
    int   len = strlen(delim);
    int   i, j = 0;
    int   found = 0;

    if (!*stringp)
    {
        return NULL;
    }
    ret = *stringp;
    do
    {
        if ( (*stringp)[j] == '\0')
        {
            found    = 1;
            *stringp = NULL;
            break;
        }
        for (i = 0; i < len; i++)
        {
            if ( (*stringp)[j] == delim[i])
            {
                (*stringp)[j] = '\0';
                *stringp      = *stringp+j+1;
                found         = 1;
                break;
            }
        }
        j++;
    }
    while (!found);

    return ret;
}
