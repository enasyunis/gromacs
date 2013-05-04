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

