#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <math.h>
#include "maths.h"
#include "typedefs.h"
#include "vec.h"
#include "coulomb.h"
#include "smalloc.h"
#include "physics.h"
#include "txtdump.h"
#include "futil.h"
#include "names.h"
#include "writeps.h"
#include "macros.h"

real calc_ewaldcoeff(real rc, real dtol)
{ // called 
    real x = 5, low, high;
    int  n, i = 0;


    do
    {
        i++;
        x *= 2;
    }
    while (gmx_erfc(x*rc) > dtol);

    n    = i+60; /* search tolerance is 2^-60 */
    low  = 0;
    high = x;
    for (i = 0; i < n; i++)
    {
        x = (low+high)/2;
        if (gmx_erfc(x*rc) > dtol)
        {
            low = x;
        }
        else
        {
            high = x;
        }
    }
    return x;
}

real ewald_charge_correction(t_commrec *cr, t_forcerec *fr, real lambda,
                             matrix box,
                             real *dvdlambda, tensor vir)

{ // called
    real vol, fac, qs2A, qs2B, vc, enercorr;
    int  d;

    /* Apply charge correction */
    vol = box[XX][XX]*box[YY][YY]*box[ZZ][ZZ];

    fac = M_PI*ONE_4PI_EPS0/(fr->epsilon_r*2.0*vol*vol*sqr(fr->ewaldcoeff));

    qs2A = fr->qsum[0]*fr->qsum[0];
    qs2B = fr->qsum[1]*fr->qsum[1];

    vc = (qs2A*(1 - lambda) + qs2B*lambda)*fac;

    enercorr = -vol*vc;

    *dvdlambda += -vol*(qs2B - qs2A)*fac;

    for (d = 0; d < DIM; d++)
    {
        vir[d][d] += vc;
    }


    return enercorr;
}
