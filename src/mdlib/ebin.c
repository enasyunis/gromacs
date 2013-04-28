/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "typedefs.h"
#include "gmx_fatal.h"
#include "string2.h"
#include "ebin.h"
#include "main.h"
#include "maths.h"
#include "vec.h"
#include "physics.h"

t_ebin *mk_ebin(void)
{ // called  
    t_ebin *eb;

    snew(eb, 1);

    return eb;
}

int get_ebin_space(t_ebin *eb, int nener, const char *enm[], const char *unit)
{ // called  
    int         index;
    int         i, f;
    const char *u;

    index      = eb->nener;
    eb->nener += nener;
    srenew(eb->e, eb->nener);
    srenew(eb->e_sim, eb->nener);
    srenew(eb->enm, eb->nener);
    for (i = index; (i < eb->nener); i++) // (index,nener): (0,9), (9,18), (18,27), (27,28), (28,29), (29,30)
    {
        eb->e[i].e        = 0;
        eb->e[i].eav      = 0;
        eb->e[i].esum     = 0;
        eb->e_sim[i].e    = 0;
        eb->e_sim[i].eav  = 0;
        eb->e_sim[i].esum = 0;
        eb->enm[i].name   = strdup(enm[i-index]);
        if (unit != NULL) 
        {
            eb->enm[i].unit = strdup(unit);
        } 
	else
	{
	    eb->enm[i].unit = i==7?strdup(unit_temp_K):i==8?strdup(unit_pres_bar):strdup(unit_energy);
	}
    }
    return index;
}

void add_ebin(t_ebin *eb, int index, int nener, real ener[], gmx_bool bSum)
{ // called  
    int       i, m;
    double    e, sum, sigma, invmm, diff;
    t_energy *eg, *egs;


    eg = &(eb->e[index]);
    for (i = 0; (i < nener); i++) // three 9's followed by three 1's
    {
        eg[i].e      = ener[i];
    }
    egs = &(eb->e_sim[index]);

    m = eb->nsum;

    for (i = 0; (i < nener); i++)
    {
         eg[i].eav    = 0;
         eg[i].esum   = ener[i];
         egs[i].esum += ener[i];
    }
}

void ebin_increase_count(t_ebin *eb, gmx_bool bSum)
{// called  
    eb->nsteps++;
    eb->nsteps_sim++;
    eb->nsum++;
    eb->nsum_sim++;
}

void reset_ebin_sums(t_ebin *eb)
{// called  
    eb->nsteps = 0;
    eb->nsum   = 0;
    /* The actual sums are cleared when the next frame is stored */
}

void pr_ebin(FILE *fp, t_ebin *eb, int index, int nener, int nperline,
             int prmode, gmx_bool bPrHead)
{ // called 
    int  i, j, i0;
    real ee = 0;
    int  rc;
    char buf[30];

    rc = 0;
    nener = index + nener;
    for (i = index; (i < nener) && rc >= 0; ) // nener=9 called four times.
    {
        if (bPrHead) // true on the first two calls
        {
            i0 = i;
            for (j = 0; (j < nperline) && (i < nener) && rc >= 0; j++, i++)
            {
                if (strncmp(eb->enm[i].name, "Pres", 4) == 0)
                {
                    /* Print the pressure unit to avoid confusion */
                    sprintf(buf, "%s (%s)", eb->enm[i].name, unit_pres_bar);
                    rc = fprintf(fp, "%15s", buf);
                }
                else
                {
                    rc = fprintf(fp, "%15s", eb->enm[i].name);
                }
            }

            if (rc >= 0)
            {
                rc = fprintf(fp, "\n");
            }

            i = i0;
        }
        for (j = 0; (j < nperline) && (i < nener) && rc >= 0; j++, i++)
        {
            switch (prmode)
            {
                case eprNORMAL: ee = eb->e[i].e; break;
                case eprAVER:   ee = eb->e_sim[i].esum/eb->nsum_sim; break;
                default: gmx_fatal(FARGS, "Invalid print mode %d in pr_ebin",
                                   prmode);
            }

            rc = fprintf(fp, "   %12.5e", ee);
        }
        if (rc >= 0)
        {
            rc = fprintf(fp, "\n");
        }
    }
}

