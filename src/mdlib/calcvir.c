/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sysstuff.h"
#include "force.h"
#include "vec.h"
#include "mshift.h"
#include "macros.h"

static void upd_vir(rvec vir, real dvx, real dvy, real dvz)
{//called 
    vir[XX] -= 0.5*dvx;
    vir[YY] -= 0.5*dvy;
    vir[ZZ] -= 0.5*dvz;
}

void calc_vir(FILE *log, int nxf, rvec x[], rvec f[], tensor vir,
              gmx_bool bScrewPBC, matrix box)
{ // called
    int      i, isx;
    double   dvxx = 0, dvxy = 0, dvxz = 0, dvyx = 0, dvyy = 0, dvyz = 0, dvzx = 0, dvzy = 0, dvzz = 0;

    for (i = 0; (i < nxf); i++) // 45, 3000
    {
        dvxx += x[i][XX]*f[i][XX];
        dvxy += x[i][XX]*f[i][YY];
        dvxz += x[i][XX]*f[i][ZZ];

        dvyx += x[i][YY]*f[i][XX];
        dvyy += x[i][YY]*f[i][YY];
        dvyz += x[i][YY]*f[i][ZZ];

        dvzx += x[i][ZZ]*f[i][XX];
        dvzy += x[i][ZZ]*f[i][YY];
        dvzz += x[i][ZZ]*f[i][ZZ];

    }

    upd_vir(vir[XX], dvxx, dvxy, dvxz);
    upd_vir(vir[YY], dvyx, dvyy, dvyz);
    upd_vir(vir[ZZ], dvzx, dvzy, dvzz);
}


void f_calc_vir(FILE *log, int i0, int i1, rvec x[], rvec f[], tensor vir,
                t_graph *g, matrix box)
{ // called 
    calc_vir(log, i1-i0, x + i0, f + i0, vir, FALSE, box);
}
