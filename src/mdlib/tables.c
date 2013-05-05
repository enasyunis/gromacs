#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include "maths.h"
#include "typedefs.h"
#include "names.h"
#include "smalloc.h"
#include "gmx_fatal.h"
#include "futil.h"
#include "vec.h"
#include "physics.h"
#include "force.h"
#include "gmxfio.h"
#include "tables.h"

/* All the possible (implemented) table functions */
enum {
    etabLJ6,
    etabLJ12,
    etabLJ6Shift,
    etabLJ12Shift,
    etabShift,
    etabRF,
    etabRF_ZERO,
    etabCOUL,
    etabEwald,
    etabEwaldSwitch,
    etabEwaldUser,
    etabEwaldUserSwitch,
    etabLJ6Switch,
    etabLJ12Switch,
    etabCOULSwitch,
    etabLJ6Encad,
    etabLJ12Encad,
    etabCOULEncad,
    etabEXPMIN,
    etabUSER,
    etabNR
};

/** Evaluates to true if the table type contains user data. */
#define ETAB_USER(e)  ((e) == etabUSER || \
                       (e) == etabEwaldUser || (e) == etabEwaldUserSwitch)

typedef struct {
    const char *name;
    gmx_bool    bCoulomb;
} t_tab_props;

/* This structure holds name and a flag that tells whether
   this is a Coulomb type funtion */
static const t_tab_props tprops[etabNR] = {
    { "LJ6",  FALSE },
    { "LJ12", FALSE },
    { "LJ6Shift", FALSE },
    { "LJ12Shift", FALSE },
    { "Shift", TRUE },
    { "RF", TRUE },
    { "RF-zero", TRUE },
    { "COUL", TRUE },
    { "Ewald", TRUE },
    { "Ewald-Switch", TRUE },
    { "Ewald-User", TRUE },
    { "Ewald-User-Switch", TRUE },
    { "LJ6Switch", FALSE },
    { "LJ12Switch", FALSE },
    { "COULSwitch", TRUE },
    { "LJ6-Encad shift", FALSE },
    { "LJ12-Encad shift", FALSE },
    { "COUL-Encad shift",  TRUE },
    { "EXPMIN", FALSE },
    { "USER", FALSE }
};

/* Index in the table that says which function to use */
enum {
    etiCOUL, etiLJ6, etiLJ12, etiNR
};

typedef struct {
    int     nx, nx0;
    double  tabscale;
    double *x, *v, *f;
} t_tabledata;

#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define pow5(x) ((x)*(x)*(x)*(x)*(x))


static double v_ewald_lr(double beta, double r)
{ // called 
    if (r == 0) // both can be called
    { 
        return beta*2/sqrt(M_PI);
    }
    else
    {
        return gmx_erfd(beta*r)/r;
    }
}

void table_spline3_fill_ewald_lr(real *table_f,
                                 real *table_v,
                                 real *table_fdv0,
                                 int   ntab,
                                 real  dx,
                                 real  beta)
{// called 
    real     tab_max;
    int      i, i_inrange;
    double   dc, dc_new;
    double   v_r0, v_r1, v_inrange, vi, a0, a1, a2dx;
    double   x_r0;


    /* We need some margin to be able to divide table values by r
     * in the kernel and also to do the integration arithmetics
     * without going out of range. Furthemore, we divide by dx below.
     */
    tab_max = GMX_REAL_MAX*0.0001;

    /* This function produces a table with:
     * maximum energy error: V'''/(6*12*sqrt(3))*dx^3
     * maximum force error:  V'''/(6*4)*dx^2
     * The rms force error is the max error times 1/sqrt(5)=0.45.
     */

    i_inrange   = ntab;
    v_inrange   = 0;
    dc          = 0;
    for (i = ntab-1; i >= 0; i--) // ntab=3192
    {
        x_r0 = i*dx;

        v_r0 = v_ewald_lr(beta, x_r0);

        i_inrange = i;
        v_inrange = v_r0;

        vi = v_r0;

        table_v[i] = vi;

        if (i == 0)
        {
            continue;
        }

        /* Get the potential at table point i-1 */
        v_r1 = v_ewald_lr(beta, (i-1)*dx);


        /* Calculate the average second derivative times dx over interval i-1 to i.
         * Using the function values at the end points and in the middle.
         */
        a2dx = (v_r0 + v_r1 - 2*v_ewald_lr(beta, x_r0-0.5*dx))/(0.25*dx);
        /* Set the derivative of the spline to match the difference in potential
         * over the interval plus the average effect of the quadratic term.
         * This is the essential step for minimizing the error in the force.
         */
        dc = (v_r0 - v_r1)/dx + 0.5*a2dx;

        if (i == ntab - 1)
        {
            /* Fill the table with the force, minus the derivative of the spline */
            table_f[i] = -dc;
        }
        else
        {
            /* tab[i] will contain the average of the splines over the two intervals */
            table_f[i] += -0.5*dc;
        }

        /* Make spline s(x) = a0 + a1*(x - xr) + 0.5*a2*(x - xr)^2
         * matching the potential at the two end points
         * and the derivative dc at the end point xr.
         */
        a0   = v_r0;
        a1   = dc;
        a2dx = (a1*dx + v_r1 - a0)*2/dx;

        /* Set dc to the derivative at the next point */
        dc_new = a1 - a2dx;

        dc = dc_new;

        table_f[(i-1)] = -0.5*dc;
    }
    /* Currently the last value only contains half the force: double it */
    table_f[0] *= 2;
    /* Copy to FDV0 table too. Allocation occurs in forcerec.c,
     * init_ewald_f_table().
     */
    for (i = 0; i < ntab-1; i++)
    {
            table_fdv0[4*i]     = table_f[i];
            table_fdv0[4*i+1]   = table_f[i+1]-table_f[i];
            table_fdv0[4*i+2]   = table_v[i];
            table_fdv0[4*i+3]   = 0.0;
    }
    table_fdv0[4*(ntab-1)]    = table_f[(ntab-1)];
    table_fdv0[4*(ntab-1)+1]  = -table_f[(ntab-1)];
    table_fdv0[4*(ntab-1)+2]  = table_v[(ntab-1)];
    table_fdv0[4*(ntab-1)+3]  = 0.0;
}

/* The scale (1/spacing) for third order spline interpolation
 * of the Ewald mesh contribution which needs to be subtracted
 * from the non-bonded interactions.
 */
real ewald_spline3_table_scale(real ewaldcoeff, real rc)
{// called 
    double erf_x_d3 = 1.0522; /* max of (erf(x)/x)''' */
    double ftol, etol;
    double sc_f, sc_e;

    /* Force tolerance: single precision accuracy */
    ftol = GMX_FLOAT_EPS;
    sc_f = sqrt(erf_x_d3/(6*4*ftol*ewaldcoeff))*ewaldcoeff;

    /* Energy tolerance: 10x more accurate than the cut-off jump */
    etol = 0.1*gmx_erfc(ewaldcoeff*rc);
    etol = max(etol, GMX_REAL_EPS);
    sc_e = pow(erf_x_d3/(6*12*sqrt(3)*etol), 1.0/3.0)*ewaldcoeff;

    return max(sc_f, sc_e);
}


static void copy2table(int n, int offset, int stride,
                       double x[], double Vtab[], double Ftab[], real scalefactor,
                       real dest[])
{// called 
/* Use double prec. for the intermediary variables
 * and temporary x/vtab/vtab2 data to avoid unnecessary
 * loss of precision.
 */
    int    i, nn0;
    double F, G, H, h;

    h = 0;
    for (i = 0; (i < n); i++) // 4000
    {
        if (i < n-1)
        {
            h   = x[i+1] - x[i];
            F   = -Ftab[i]*h;
            G   =  3*(Vtab[i+1] - Vtab[i]) + (Ftab[i+1] + 2*Ftab[i])*h;
            H   = -2*(Vtab[i+1] - Vtab[i]) - (Ftab[i+1] +   Ftab[i])*h;
        }
        else
        {
            /* Fill the last entry with a linear potential,
             * this is mainly for rounding issues with angle and dihedral potentials.
             */
            F   = -Ftab[i]*h;
            G   = 0;
            H   = 0;
        }
        nn0         = offset + i*stride;
        dest[nn0]   = scalefactor*Vtab[i];
        dest[nn0+1] = scalefactor*F;
        dest[nn0+2] = scalefactor*G;
        dest[nn0+3] = scalefactor*H;
    }
}

static void init_table(FILE *fp, int n, int nx0,
                       double tabscale, t_tabledata *td, gmx_bool bAlloc)
{// called 
    int i;

    td->nx       = n;
    td->nx0      = nx0;
    td->tabscale = tabscale;
    snew(td->x, td->nx);
    snew(td->v, td->nx);
    snew(td->f, td->nx);
    for (i = 0; (i < td->nx); i++) // 4000
    {
        td->x[i] = i/tabscale;
    }
}


static void done_tabledata(t_tabledata *td)
{// called 
    int i;

    sfree(td->x);
    sfree(td->v);
    sfree(td->f);
}

static void fill_table(t_tabledata *td, int tp, const t_forcerec *fr)
{// called 
    /* Fill the table according to the formulas in the manual.
     * In principle, we only need the potential and the second
     * derivative, but then we would have to do lots of calculations
     * in the inner loop. By precalculating some terms (see manual)
     * we get better eventual performance, despite a larger table.
     *
     * Since some of these higher-order terms are very small,
     * we always use double precision to calculate them here, in order
     * to avoid unnecessary loss of precision.
     */
    int      i;
    double   reppow, p;
    double   r1, rc, r12, r13;
    double   r, r2, r6, rc6;
    double   expr, Vtab, Ftab;
    /* Parameters for David's function */
    double   A = 0, B = 0, C = 0, A_3 = 0, B_4 = 0;
    /* Parameters for the switching function */
    double   ksw, swi, swi1;
    /* Temporary parameters */
    double   ewc = fr->ewaldcoeff;

    reppow = fr->reppow;

    if (tprops[tp].bCoulomb) // both can be true
    {  
        r1 = fr->rcoulomb_switch;
        rc = fr->rcoulomb;
    }
    else
    { 
        r1 = fr->rvdw_switch;
        rc = fr->rvdw;
    }
    ksw  = 0.0;


    for (i = td->nx0; (i < td->nx); i++) // 4000
    {
        r     = td->x[i];
        r2    = r*r;
        r6    = 1.0/(r2*r2*r2);
        r12   = r6*r6;
        Vtab  = 0.0;
        Ftab  = 0.0;
        swi   = 1.0;
        swi1  = 0.0;

        rc6 = rc*rc*rc;
        rc6 = 1.0/(rc6*rc6);
        switch (tp)
        {
            case etabLJ6:
                /* Dispersion */
                Vtab = -r6;
                Ftab = 6.0*Vtab/r;
                break;
            case etabLJ12:
                /* Repulsion */
                Vtab  = r12;
                Ftab  = reppow*Vtab/r;
                break;
            case etabEwald:
            case etabEwaldSwitch:
                Vtab  = gmx_erfc(ewc*r)/r;
                Ftab  = gmx_erfc(ewc*r)/r2+exp(-(ewc*ewc*r2))*ewc*M_2_SQRTPI/r;
                break;
        }

        /* Convert to single precision when we store to mem */
        td->v[i]  = Vtab;
        td->f[i]  = Ftab;
    }

    /* Continue the table linearly from nx0 to 0.
     * These values are only required for energy minimization with overlap or TPI.
     */
    for (i = td->nx0-1; i >= 0; i--) // td->nx0 = 10
    {
        td->v[i] = td->v[i+1] + td->f[i+1]*(td->x[i+1] - td->x[i]);
        td->f[i] = td->f[i+1];
    }

}

static void set_table_type(int tabsel[], const t_forcerec *fr, gmx_bool b14only)
{// called 

    tabsel[etiCOUL] = etabEwald;
    tabsel[etiLJ6]  = etabLJ6;
    tabsel[etiLJ12] = etabLJ12;
}

t_forcetable make_tables(FILE *out, 
                         const t_forcerec *fr,
                         gmx_bool bVerbose, const char *fn,
                         real rtab, int flags)
{// called 
    const char     *fns[3]   = { "ctab.xvg", "dtab.xvg", "rtab.xvg" };
    const char     *fns14[3] = { "ctab14.xvg", "dtab14.xvg", "rtab14.xvg" };
    FILE           *fp;
    t_tabledata    *td;
    gmx_bool        b14only, bReadTab, bGenTab;
    real            x0, y0, yp;
    int             i, j, k, nx, nx0, tabsel[etiNR];
    real            scalefactor;

    t_forcetable    table;

    b14only = (flags & GMX_MAKETABLES_14ONLY);

    set_table_type(tabsel, fr, b14only);
    snew(td, etiNR);
    table.r         = rtab;
    table.scale     = 0;
    table.n         = 0;
    table.scale_exp = 0;
    nx0             = 10;
    nx              = 0;

    table.interaction   = GMX_TABLE_INTERACTION_ELEC_VDWREP_VDWDISP;
    table.format        = GMX_TABLE_FORMAT_CUBICSPLINE_YFGH;
    table.formatsize    = 4;
    table.ninteractions = 3;
    table.stride        = table.formatsize*table.ninteractions;

    /* Check whether we have to read or generate */
    bReadTab = FALSE;
    bGenTab  = TRUE;

    table.scale = 2000.0;
    nx = table.n = rtab*table.scale;

    /* Each table type (e.g. coul,lj6,lj12) requires four
     * numbers per nx+1 data points. For performance reasons we want
     * the table data to be aligned to 16-byte.
     */
    snew_aligned(table.data, 12*(nx+1)*sizeof(real), 32);

    for (k = 0; (k < etiNR); k++)
    {
        if (tabsel[k] != etabUSER)
        {
            init_table(out, nx, nx0,
                       (tabsel[k] == etabEXPMIN) ? table.scale_exp : table.scale,
                       &(td[k]), !bReadTab);
            fill_table(&(td[k]), tabsel[k], fr);
            if (out)
            {
                fprintf(out, "%s table with %d data points for %s%s.\n"
                        "Tabscale = %g points/nm\n",
                        ETAB_USER(tabsel[k]) ? "Modified" : "Generated",
                        td[k].nx, b14only ? "1-4 " : "", tprops[tabsel[k]].name,
                        td[k].tabscale);
            }
        }

        /* Set scalefactor for c6/c12 tables. This is because we save flops in the non-table kernels
         * by including the derivative constants (6.0 or 12.0) in the parameters, since
         * we no longer calculate force in most steps. This means the c6/c12 parameters
         * have been scaled up, so we need to scale down the table interactions too.
         * It comes here since we need to scale user tables too.
         */
        if (k == etiLJ6)
        { 
            scalefactor = 1.0/6.0;
        }
        else if (k == etiLJ12 && tabsel[k] != etabEXPMIN)
        {
            scalefactor = 1.0/12.0;
        }
        else
        {
            scalefactor = 1.0;
        }

        copy2table(table.n, k*4, 12, td[k].x, td[k].v, td[k].f, scalefactor, table.data);

        done_tabledata(&(td[k]));
    }
    sfree(td);

    return table;
}

