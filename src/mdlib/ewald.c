#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <math.h>
#include "typedefs.h"
#include "vec.h"
#include "gmxcomplex.h"
#include "smalloc.h"
#include "futil.h"
#include "gmx_fatal.h"
#include "physics.h"
#include "coulomb.h"

#define TOL 2e-5

struct ewald_tab
{
    int        nx, ny, nz, kmax;
    cvec     **eir;
    t_complex *tab_xy, *tab_qxyz;
};



/* TODO: fix thread-safety */

/* the other routines are in complex.h */
static t_complex conjmul(t_complex a, t_complex b)
{ // called
    t_complex c;

    c.re = a.re*b.re + a.im*b.im;
    c.im = a.im*b.re - a.re*b.im;

    return c;
}




static void tabulate_eir(int natom, rvec x[], int kmax, cvec **eir, rvec lll)
{ // called by Ewald  // kmax=15
    int  i, j, m;

    if (kmax < 1)
    {
        printf("Go away! kmax = %d\n", kmax);
        exit(1);
    }

    for (i = 0; (i < natom); i++)
    {
        for (m = 0; (m < 3); m++)
        {
            eir[0][i][m].re = 1;
            eir[0][i][m].im = 0;
        }

        for (m = 0; (m < 3); m++)
        {
            eir[1][i][m].re = cos(x[i][m]*lll[m]);
            eir[1][i][m].im = sin(x[i][m]*lll[m]);
        }
        for (j = 2; (j < kmax); j++)
        {
            for (m = 0; (m < 3); m++)
            {
                eir[j][i][m] = cmul(eir[j-1][i][m], eir[1][i][m]);
            }
        }
    }
}

int init_ewald_tab(ewald_tab_t *et, const t_commrec *cr, const t_inputrec *ir,
                    FILE *fp)
{ // called by boht PME and Ewald
    int n;

    snew(*et, 1);
    if (fp)
    {   printf("\n** Ordinary reciprocal space ewald\n");
        fprintf(fp, "Will do ordinary reciprocal space Ewald sum.\n");
    }

    (*et)->nx       = ir->nkx+1;
    (*et)->ny       = ir->nky+1;
    (*et)->nz       = ir->nkz+1;
    (*et)->kmax     = max((*et)->nx, max((*et)->ny, (*et)->nz));
    printf("\n** kmax %d\n", (*et)->kmax);
    (*et)->eir      = NULL;
    (*et)->tab_xy   = NULL;
    (*et)->tab_qxyz = NULL;

    return (*et)->kmax;
}



real do_ewald(FILE *log,       gmx_bool bVerbose,
              t_inputrec *ir,
              rvec x[],        rvec f[],
              real chargeA[],  real chargeB[],
              rvec box,
              t_commrec *cr,   int natoms,
              matrix lrvir,    real ewaldcoeff,
              real lambda,     real *dvdlambda,
              ewald_tab_t et)
{ // called by Ewald
    real     factor     = -1.0/(4*ewaldcoeff*ewaldcoeff);
    real     scaleRecip = 4.0*M_PI/(box[XX]*box[YY]*box[ZZ])*ONE_4PI_EPS0/ir->epsilon_r; /* 1/(Vol*e0) */
    real    *charge, energy_AB[2], energy;
    rvec     lll;
    int      lowiy, lowiz, ix, iy, iz, n, q;
    real     tmp, cs, ss, ak, akv, mx, my, mz, m2, scale;
    gmx_bool bFreeEnergy;


    snew(et->eir, et->kmax);
    for (n = 0; n < et->kmax; n++)
    {
        snew(et->eir[n], natoms);
    }
    snew(et->tab_xy, natoms);
    snew(et->tab_qxyz, natoms);

    bFreeEnergy = (ir->efep != efepNO); // FALSE

    clear_mat(lrvir);

    calc_lll(box, lll);
    /* make tables for the structure factor parts */
    tabulate_eir(natoms, x, et->kmax, et->eir, lll);
    q=0;
    charge = chargeA;
    scale  = 1.0;
    lowiy        = 0;
    lowiz        = 1;
    energy_AB[q] = 0;
    for (ix = 0; ix < et->nx; ix++)
    {
        mx = ix*lll[XX];
        for (iy = lowiy; iy < et->ny; iy++)
        {
            my = iy*lll[YY];
            if (iy >= 0)
            {
                for (n = 0; n < natoms; n++)
                {
                    et->tab_xy[n] = cmul(et->eir[ix][n][XX], et->eir[iy][n][YY]);
                }
            }
            else
            {
                for (n = 0; n < natoms; n++)
                {
                    et->tab_xy[n] = conjmul(et->eir[ix][n][XX], et->eir[-iy][n][YY]);
                }
            }
            for (iz = lowiz; iz < et->nz; iz++)
            {
                mz  = iz*lll[ZZ];
                m2  = mx*mx+my*my+mz*mz;
                ak  = exp(m2*factor)/m2;
                akv = 2.0*ak*(1.0/m2-factor);
                if (iz >= 0)
                {
                    for (n = 0; n < natoms; n++)
                    {
                        et->tab_qxyz[n] = rcmul(charge[n], cmul(et->tab_xy[n],
                                                              et->eir[iz][n][ZZ]));
                    }
                }
                else
                {
                   for (n = 0; n < natoms; n++)
                   {
                       et->tab_qxyz[n] = rcmul(charge[n], conjmul(et->tab_xy[n],
                                                                et->eir[-iz][n][ZZ]));
                    }
                }

                cs = ss = 0;
                for (n = 0; n < natoms; n++)
                {
                     cs += et->tab_qxyz[n].re;
                     ss += et->tab_qxyz[n].im;
                 }
                 energy_AB[q]  += ak*(cs*cs+ss*ss);
                 tmp            = scale*akv*(cs*cs+ss*ss);
                 lrvir[XX][XX] -= tmp*mx*mx;
                 lrvir[XX][YY] -= tmp*mx*my;
                 lrvir[XX][ZZ] -= tmp*mx*mz;
                 lrvir[YY][YY] -= tmp*my*my;
                 lrvir[YY][ZZ] -= tmp*my*mz;
                 lrvir[ZZ][ZZ] -= tmp*mz*mz;
                 for (n = 0; n < natoms; n++)
                 {
                     tmp       = scale*ak*(cs*et->tab_qxyz[n].im-ss*et->tab_qxyz[n].re);
                     f[n][XX] += tmp*mx*2*scaleRecip;
                     f[n][YY] += tmp*my*2*scaleRecip;
                     f[n][ZZ] += tmp*mz*2*scaleRecip;
                 }
                 lowiz = 1-et->nz;
             }
             lowiy = 1-et->ny;
        }
    }
    energy = energy_AB[0];

    lrvir[XX][XX] = -0.5*scaleRecip*(lrvir[XX][XX]+energy);
    lrvir[XX][YY] = -0.5*scaleRecip*(lrvir[XX][YY]);
    lrvir[XX][ZZ] = -0.5*scaleRecip*(lrvir[XX][ZZ]);
    lrvir[YY][YY] = -0.5*scaleRecip*(lrvir[YY][YY]+energy);
    lrvir[YY][ZZ] = -0.5*scaleRecip*(lrvir[YY][ZZ]);
    lrvir[ZZ][ZZ] = -0.5*scaleRecip*(lrvir[ZZ][ZZ]+energy);

    lrvir[YY][XX] = lrvir[XX][YY];
    lrvir[ZZ][XX] = lrvir[XX][ZZ];
    lrvir[ZZ][YY] = lrvir[YY][ZZ];

    energy *= scaleRecip;

    return energy;
}
