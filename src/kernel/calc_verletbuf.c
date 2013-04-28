#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "assert.h"

#include <sys/types.h>
#include <math.h>
#include "typedefs.h"
#include "physics.h"
#include "smalloc.h"
#include "gmx_fatal.h"
#include "macros.h"
#include "vec.h"
#include "coulomb.h"
#include "calc_verletbuf.h"
#include "../mdlib/nbnxn_consts.h"

/* Struct for unique atom type for calculating the energy drift.
 * The atom displacement depends on mass and constraints.
 * The energy jump for given distance depend on LJ type and q.
 */
typedef struct
{
    real     mass; /* mass */
    int      type; /* type (used for LJ parameters) */
    real     q;    /* charge */
    int      con;  /* constrained: 0, else 1, if 1, use #DOF=2 iso 3 */
    int      n;    /* total #atoms of this type in the system */
} verletbuf_atomtype_t;


void verletbuf_get_list_setup(gmx_bool                bGPU,
                              verletbuf_list_setup_t *list_setup)
{//called
    list_setup->cluster_size_i     = NBNXN_CPU_CLUSTER_I_SIZE;

    list_setup->cluster_size_j = NBNXN_CPU_CLUSTER_I_SIZE;
}

static void add_at(verletbuf_atomtype_t **att_p, int *natt_p,
                   real mass, int type, real q, int con, int nmol)
{//called
    verletbuf_atomtype_t *att;
    int                   natt, i;

    if (mass == 0)
    {
        /* Ignore massless particles */
        return;
    }

    att  = *att_p;
    natt = *natt_p;

    i = 0;
    while (i < natt &&
           !(mass == att[i].mass &&
             type == att[i].type &&
             q    == att[i].q &&
             con  == att[i].con))
    {
        i++;
    }

    if (i < natt)
    {
        att[i].n += nmol;
    }
    else
    {
        (*natt_p)++;
        srenew(*att_p, *natt_p);
        (*att_p)[i].mass = mass;
        (*att_p)[i].type = type;
        (*att_p)[i].q    = q;
        (*att_p)[i].con  = con;
        (*att_p)[i].n    = nmol;
    }
}

static void get_verlet_buffer_atomtypes(const gmx_mtop_t      *mtop,
                                        verletbuf_atomtype_t **att_p,
                                        int                   *natt_p,
                                        int                   *n_nonlin_vsite)
{//called
    verletbuf_atomtype_t *att;
    int                   natt;
    int                   mb, nmol, ft, i, j, a1, a2, a3, a;
    const t_atoms        *atoms;
    const t_ilist        *il;
    const t_atom         *at;
    const t_iparams      *ip;
    real                 *con_m, *vsite_m, cam[5];

    att  = NULL;
    natt = 0;
    mb=0;
        nmol = mtop->molblock[mb].nmol;

        atoms = &mtop->moltype[mtop->molblock[mb].type].atoms;

        /* Check for constraints, as they affect the kinetic energy */
        snew(con_m, atoms->nr);
        snew(vsite_m, atoms->nr);
        for (ft = F_CONSTR; ft <= F_CONSTRNC; ft++)
        {
            il = &mtop->moltype[mtop->molblock[mb].type].ilist[ft];
        }

        il = &mtop->moltype[mtop->molblock[mb].type].ilist[F_SETTLE];
        /* Check for virtual sites, determine mass from constructing atoms */
        for (ft = 0; ft < F_NRE; ft++)
        {
            if (IS_VSITE(ft))
            {
                il = &mtop->moltype[mtop->molblock[mb].type].ilist[ft];
            }
        }
        for (a = 0; a < atoms->nr; a++)
        {
            at = &atoms->atom[a];
            /* We consider an atom constrained, #DOF=2, when it is
             * connected with constraints to one or more atoms with
             * total mass larger than 1.5 that of the atom itself.
             */
            add_at(&att, &natt,
                   at->m, at->type, at->q, con_m[a] > 1.5*at->m, nmol);
        }

        sfree(vsite_m);
        sfree(con_m);

    *att_p  = att;
    *natt_p = natt;
}


static real ener_drift(const verletbuf_atomtype_t *att, int natt,
                       const gmx_ffparams_t *ffp,
                       real kT_fac,
                       real md_ljd, real md_ljr, real md_el, real dd_el,
                       real r_buffer,
                       real rlist, real boxvol)
{//called
    double drift_tot, pot1, pot2, pot;
    int    i, j;
    real   s2i, s2j, s2, s;
    int    ti, tj;
    real   md, dd;
    real   sc_fac, rsh;
    double c_exp, c_erfc;

    drift_tot = 0;

    /* Loop over the different atom type pairs */
    for (i = 0; i < natt; i++)
    {
        s2i = kT_fac/att[i].mass;
        ti  = att[i].type;

        for (j = i; j < natt; j++)
        {
            s2j = kT_fac/att[j].mass;
            tj  = att[j].type;

            /* Note that attractive and repulsive potentials for individual
             * pairs will partially cancel.
             */
            /* -dV/dr at the cut-off for LJ + Coulomb */
            md =
                md_ljd*ffp->iparams[ti*ffp->atnr+tj].lj.c6 +
                md_ljr*ffp->iparams[ti*ffp->atnr+tj].lj.c12 +
                md_el*att[i].q*att[j].q;

            /* d2V/dr2 at the cut-off for Coulomb, we neglect LJ */
            dd = dd_el*att[i].q*att[j].q;

            s2  = s2i + s2j;

            rsh    = r_buffer;
            sc_fac = 1.0;

            /* Exact contribution of an atom pair with Gaussian displacement
             * with sigma s to the energy drift for a potential with
             * derivative -md and second derivative dd at the cut-off.
             * The only catch is that for potentials that change sign
             * near the cut-off there could be an unlucky compensation
             * of positive and negative energy drift.
             * Such potentials are extremely rare though.
             *
             * Note that pot has unit energy*length, as the linear
             * atom density still needs to be put in.
             */
            c_exp  = exp(-rsh*rsh/(2*s2))/sqrt(2*M_PI);
            c_erfc = 0.5*gmx_erfc(rsh/(sqrt(2*s2)));
            s      = sqrt(s2);

            pot1 = sc_fac*
                md/2*((rsh*rsh + s2)*c_erfc - rsh*s*c_exp);
            pot2 = sc_fac*
                dd/6*(s*(rsh*rsh + 2*s2)*c_exp - rsh*(rsh*rsh + 3*s2)*c_erfc);
            pot = pot1 + pot2;


            /* Multiply by the number of atom pairs */
            if (j == i)
            {
                pot *= (double)att[i].n*(att[i].n - 1)/2;
            }
            else
            {
                pot *= (double)att[i].n*att[j].n;
            }
            /* We need the line density to get the energy drift of the system.
             * The effective average r^2 is close to (rlist+sigma)^2.
             */
            pot *= 4*M_PI*sqr(rlist + s)/boxvol;

            /* Add the unsigned drift to avoid cancellation of errors */
            drift_tot += fabs(pot);
        }
    }

    return drift_tot;
}

static real surface_frac(int cluster_size, real particle_distance, real rlist)
{//called
    real d, area_rel;

    if (rlist < 0.5*particle_distance)
    {
        /* We have non overlapping spheres */
        return 1.0;
    }

    /* Half the inter-particle distance relative to rlist */
    d = 0.5*particle_distance/rlist;

    /* Determine the area of the surface at distance rlist to the closest
     * particle, relative to surface of a sphere of radius rlist.
     * The formulas below assume close to cubic cells for the pair search grid,
     * which the pair search code tries to achieve.
     * Note that in practice particle distances will not be delta distributed,
     * but have some spread, often involving shorter distances,
     * as e.g. O-H bonds in a water molecule. Thus the estimates below will
     * usually be slightly too high and thus conservative.
     */
    switch (cluster_size)
    {
        case 1:
            /* One particle: trivial */
            area_rel = 1.0;
            break;
        case 2:
            /* Two particles: two spheres at fractional distance 2*a */
            area_rel = 1.0 + d;
            break;
        case 4:
            /* We assume a perfect, symmetric tetrahedron geometry.
             * The surface around a tetrahedron is too complex for a full
             * analytical solution, so we use a Taylor expansion.
             */
// TODO ENAS RIO Surface computatios 
            area_rel = (1.0 + 1/M_PI*(6*acos(1/sqrt(3))*d +
                                      sqrt(3)*d*d*(1.0 +
                                                   5.0/18.0*d*d +
                                                   7.0/45.0*d*d*d*d +
                                                   83.0/756.0*d*d*d*d*d*d)));
            break;
        default:
            gmx_incons("surface_frac called with unsupported cluster_size");
            area_rel = 1.0;
    }

    return area_rel/cluster_size;
}

void calc_verlet_buffer_size(const gmx_mtop_t *mtop, real boxvol,
                             const t_inputrec *ir, real drift_target,
                             const verletbuf_list_setup_t *list_setup,
                             int *n_nonlin_vsite,
                             real *rlist)
{ // called
    double                resolution;
    char                 *env;

    real                  particle_distance;
    real                  nb_clust_frac_pairs_not_in_list_at_cutoff;

    verletbuf_atomtype_t *att  = NULL;
    int                   natt = -1, i;
    double                reppow;
    real                  md_ljd, md_ljr, md_el, dd_el;
    real                  elfac;
    real                  kT_fac, mass_min;
    int                   ib0, ib1, ib;
    real                  rb, rl;
    real                  drift;

    /* Resolution of the buffer size */
    resolution = 0.001;

    /* Worst case assumption: HCP packing of particles gives largest distance */
    particle_distance = pow(boxvol*sqrt(2)/mtop->natoms, 1.0/3.0);

    get_verlet_buffer_atomtypes(mtop, &att, &natt, n_nonlin_vsite);
    assert(att != NULL && natt >= 0);

    reppow = mtop->ffparams.reppow;
    md_ljd = 0;
    md_ljr = 0;
    /* -dV/dr of -r^-6 and r^-repporw */
    md_ljd = -6*pow(ir->rvdw, -7.0);
    md_ljr = reppow*pow(ir->rvdw, -(reppow+1));
    /* The contribution of the second derivative is negligible */

    elfac = ONE_4PI_EPS0/ir->epsilon_r;

    /* Determine md=-dV/dr and dd=d^2V/dr^2 */
    md_el = 0;
    dd_el = 0;
    {
        real b, rc, br;

        b     = calc_ewaldcoeff(ir->rcoulomb, ir->ewald_rtol);
        rc    = ir->rcoulomb;
        br    = b*rc;
        md_el = elfac*(b*exp(-br*br)*M_2_SQRTPI/rc + gmx_erfc(br)/(rc*rc));
        dd_el = elfac/(rc*rc)*(2*b*(1 + br*br)*exp(-br*br)*M_2_SQRTPI + 2*gmx_erfc(br)/rc);
    }

    /* Determine the variance of the atomic displacement
     * over nstlist-1 steps: kT_fac
     * For inertial dynamics (not Brownian dynamics) the mass factor
     * is not included in kT_fac, it is added later.
     */
    kT_fac = BOLTZ*ir->opts.ref_t[0]*sqr((ir->nstlist-1)*ir->delta_t);

    mass_min = att[0].mass;
    for (i = 1; i < natt; i++)
    {
        mass_min = min(mass_min, att[i].mass);
    }


    /* Search using bisection */
    ib0 = -1;
    /* The drift will be neglible at 5 times the max sigma */
    ib1 = (int)(5*2*sqrt(kT_fac/mass_min)/resolution) + 1;
    while (ib1 - ib0 > 1)
    {
        ib = (ib0 + ib1)/2;
        rb = ib*resolution;
        rl = max(ir->rvdw, ir->rcoulomb) + rb;

        /* Calculate the average energy drift at the last step
         * of the nstlist steps at which the pair-list is used.
         */
        drift = ener_drift(att, natt, &mtop->ffparams,
                           kT_fac,
                           md_ljd, md_ljr, md_el, dd_el, rb,
                           rl, boxvol);

        /* Correct for the fact that we are using a Ni x Nj particle pair list
         * and not a 1 x 1 particle pair list. This reduces the drift.
         */
        /* We don't have a formula for 8 (yet), use 4 which is conservative */
        nb_clust_frac_pairs_not_in_list_at_cutoff =
        surface_frac(min(list_setup->cluster_size_i, 4),
                         particle_distance, rl)*
        surface_frac(min(list_setup->cluster_size_j, 4),
                         particle_distance, rl);
        drift *= nb_clust_frac_pairs_not_in_list_at_cutoff;

        /* Convert the drift to drift per unit time per atom */
        drift /= ir->nstlist*ir->delta_t*mtop->natoms;

        ib1 = ib;
    }

    sfree(att);

    *rlist = max(ir->rvdw, ir->rcoulomb) + ib1*resolution;
}
