#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include <assert.h>

#include "typedefs.h"
#include "smalloc.h"
#include "vec.h"
#include "macros.h"
#include "physics.h"
#include "names.h"
#include "gmx_fatal.h"
#include "txtdump.h"
#include "nrnb.h"
#include "gmx_random.h"
#include "mdrun.h"

#define NTROTTERPARTS 3


/*
 * This file implements temperature and pressure coupling algorithms:
 * For now only the Weak coupling and the modified weak coupling.
 *
 * Furthermore computation of pressure and temperature is done here
 *
 */

real calc_pres(int ePBC, int nwall, matrix box, tensor ekin, tensor vir,
               tensor pres)
{//called 
    int  n, m;
    real fac;
        /* Uitzoeken welke ekin hier van toepassing is, zie Evans & Morris - E.
         * Wrs. moet de druktensor gecorrigeerd worden voor de netto stroom in
         * het systeem...
         */

        fac = PRESFAC*2.0/det(box);
        for (n = 0; (n < DIM); n++)
        {
            for (m = 0; (m < DIM); m++)
            {
                pres[n][m] = (ekin[n][m] - vir[n][m])*fac;
            }
        }

    return trace(pres)/DIM;
}

real calc_temp(real ekin, real nrdf)
{// called
    if (nrdf > 0) // always
    {
        return (2.0*ekin)/(nrdf*BOLTZ);
    }
    else
    {
        return 0;
    }
}


extern void init_npt_masses(t_inputrec *ir, t_state *state, t_extmass *MassQ, gmx_bool bInit)
{// called
    int           ngtc;
    t_grpopts    *opts;

    opts    = &(ir->opts); /* just for ease of referencing */
    ngtc    = ir->opts.ngtc;
    snew(MassQ->Qinv, ngtc);
    MassQ->Qinv[0] = 1.0/(sqr(opts->tau_t[0]/M_2PI)*opts->ref_t[0]);
}

int **init_npt_vars(t_inputrec *ir, t_state *state, t_extmass *MassQ, gmx_bool bTrotter)
{// called
    int           i, j, ngtc;
    int         **trotter_seq;

    init_npt_masses(ir, state, MassQ, TRUE);

    /* first, initialize clear all the trotter calls */
    snew(trotter_seq, ettTSEQMAX);
    for (i = 0; i < ettTSEQMAX; i++) // 5
    {
        snew(trotter_seq[i], NTROTTERPARTS);
        for (j = 0; j < NTROTTERPARTS; j++) // 3
        {
            trotter_seq[i][j] = etrtNONE;
        }
        trotter_seq[i][0] = etrtSKIPALL;
    }
    /* no trotter calls, so we never use the values in the array.
     * We access them (so we need to define them, but ignore
     * then.*/

    return trotter_seq;
}


