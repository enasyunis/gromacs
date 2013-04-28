#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "typedefs.h"
#include "vec.h"
#include "smalloc.h"
#include "force.h"
#include "gmx_omp_nthreads.h"
#include "nbnxn_kernel_ref.h"
#include "../nbnxn_consts.h"
#include "nbnxn_kernel_common.h"

// UNROLLI and UNROLLJ = 4
#define UNROLLI    NBNXN_CPU_CLUSTER_I_SIZE
#define UNROLLJ    NBNXN_CPU_CLUSTER_I_SIZE

/* We could use nbat->xstride and nbat->fstride, but macros might be faster */
#define X_STRIDE   3
#define F_STRIDE   3
/* Local i-atom buffer strides */
#define XI_STRIDE  3
#define FI_STRIDE  3



void nbnxn_kernel_ref_tab_ener(const nbnxn_pairlist_t     *nbl,
                                const nbnxn_atomdata_t     *nbat,
                                const interaction_const_t  *ic,
                                rvec                       *shift_vec,
                                real                       *f,
                                real                       *fshift,
                                real                       *Vvdw, // zero in and out
                                real                       *Vc)
{
    const nbnxn_ci_t   *nbln;
    const nbnxn_cj_t   *l_cj;
    const real         *q;
    const real         *shiftvec;
    const real         *x;
    real                rcut2;
    real                facel;
    int                 n, ci, ci_sh;
    int                 ish, ishf;
    int                 cjind0, cjind1, cjind;
    int                 ip, jp;

    real                xi[UNROLLI*XI_STRIDE];
    real                fi[UNROLLI*FI_STRIDE];
    real                qi[UNROLLI];

    real       Vc_ci;

    real       halfsp;
    const real *tab_coul_F;
    const real *tab_coul_V;

    int ninner;



    halfsp = 0.5/ic->tabq_scale; // 3.134706e-04
    tab_coul_F    = ic->tabq_coul_F;
    tab_coul_V    = ic->tabq_coul_V;

    rcut2               = ic->rcoulomb*ic->rcoulomb;

    q                   = nbat->q;
    facel               = ic->epsfac;
    shiftvec            = shift_vec[0];
    x                   = nbat->x;

    l_cj = nbl->cj;

    ninner = 0;
    for (n = 0; n < nbl->nci; n++) // different numbers ranging from 116 to 252
    {
        int i, d;

        nbln = &nbl->ci[n];

        ish              = (nbln->shift & NBNXN_CI_SHIFT);
        /* x, f and fshift are assumed to be stored with stride 3 */
        ishf             = ish*DIM;
        cjind0           = nbln->cj_ind_start;
        cjind1           = nbln->cj_ind_end;
        /* Currently only works super-cells equal to sub-cells */
        ci               = nbln->ci;
        ci_sh            = (ish == CENTRAL ? ci : -1);

        /* We have 5 LJ/C combinations, but use only three inner loops,
         * as the other combinations are unlikely and/or not much faster:
         * inner half-LJ + C for half-LJ + C / no-LJ + C
         * inner LJ + C      for full-LJ + C
         * inner LJ          for full-LJ + no-C / half-LJ + no-C
         */
        Vc_ci   = 0;



        for (i = 0; i < UNROLLI; i++)//4
        {
            for (d = 0; d < DIM; d++)//3
            {
                xi[i*XI_STRIDE+d] = x[(ci*UNROLLI+i)*X_STRIDE+d] + shiftvec[ishf+d];
                fi[i*FI_STRIDE+d] = 0;
            }
        }

        real Vc_sub_self = 0.5*tab_coul_V[0];

        for (i = 0; i < UNROLLI; i++)
        {
           qi[i] = facel*q[ci*UNROLLI+i];

           if (l_cj[nbln->cj_ind_start].cj == ci_sh)
           {
               Vc[0]
                  -= qi[i]*q[ci*UNROLLI+i]*Vc_sub_self;
           }
        }

        cjind = cjind0;
        while (cjind < cjind1 && nbl->cj[cjind].excl != 0xffff)
        {
//#include "nbnxn_kernel_ref_inner.h"

/* When calculating RF or Ewald interactions we calculate the electrostatic
 * forces and energies on excluded atom pairs here in the non-bonded loops.
 */

{
    int cj;
    int i;

    cj = l_cj[cjind].cj;

    for (i = 0; i < UNROLLI; i++)//4
    {
        int j;

        for (j = 0; j < UNROLLJ; j++)//4
        {
            int  aj;
            real dx, dy, dz;
            real rsq, rinv;
            real rinvsq, rinvsix;
            real qq;
            real fcoul;
            real rs, frac;
            int  ri;
            real fexcl;
            real vcoul;
            real fscal;
            real fx, fy, fz;

            /* A multiply mask used to zero an interaction
             * when either the distance cutoff is exceeded, or
             * (if appropriate) the i and j indices are
             * unsuitable for this kind of inner loop. */
            real skipmask;
            /* A multiply mask used to zero an interaction
             * when that interaction should be excluded
             * (e.g. because of bonding). */
            int interact;

	    // WHEN atoms interact then interact=1 and skipamsk=1.0 otherwise both are zero.
            interact = ((l_cj[cjind].excl>>(i*UNROLLI + j)) & 1);
            skipmask = !(cj == ci_sh && j <= i);


            aj = cj*UNROLLJ + j;

            dx  = xi[i*XI_STRIDE+XX] - x[aj*X_STRIDE+XX];
            dy  = xi[i*XI_STRIDE+YY] - x[aj*X_STRIDE+YY];
            dz  = xi[i*XI_STRIDE+ZZ] - x[aj*X_STRIDE+ZZ];

            rsq = dx*dx + dy*dy + dz*dz;

            /* Prepare to enforce the cut-off. */
            skipmask = (rsq >= rcut2) ? 0 : skipmask;
            /* 9 flops for r^2 + cut-off check */

            /* Excluded atoms are allowed to be on top of each other.
             * To avoid overflow of rinv, rinvsq and rinvsix
             * we add a small number to rsq for excluded pairs only.
             */
            rsq += (1 - interact)*NBNXN_AVOID_SING_R2_INC;


            rinv = gmx_invsqrt(rsq);
            /* 5 flops for invsqrt */

            /* Partially enforce the cut-off (and perhaps
             * exclusions) to avoid possible overflow of
             * rinvsix when computing LJ, and/or overflowing
             * the Coulomb table during lookup. */
            rinv = rinv * skipmask;

            rinvsq  = rinv*rinv;


            /* Enforce the cut-off and perhaps exclusions. In
             * those cases, rinv is zero because of skipmask,
             * but fcoul and vcoul will later be non-zero (in
             * both RF and table cases) because of the
             * contributions that do not depend on rinv. These
             * contributions cannot be allowed to accumulate
                                                                             
             * to the force and potential, and the easiest way
             * to do this is to zero the charges in
             * advance. */
            qq = skipmask * qi[i] * q[aj];


            rs     = rsq*rinv*ic->tabq_scale;
            ri     = (int)rs;
            frac   = rs - ri;
            /* fexcl = (1-frac) * F_i + frac * F_(i+1) */
            fexcl  = (1 - frac)*tab_coul_F[ri] + frac*tab_coul_F[ri+1];
            fcoul  = interact*rinvsq - fexcl;
            /* 7 flops for float 1/r-table force */
            vcoul  = qq*(interact*(rinv - ic->sh_ewald)
                         -(tab_coul_V[ri]
                           -halfsp*frac*(tab_coul_F[ri] + fexcl)));
            fcoul *= qq*rinv;

            Vc_ci += vcoul;
            /* 1 flop for Coulomb energy addition */

            fx = fcoul*dx;
            fy = fcoul*dy;
            fz = fcoul*dz;

            /* Increment i-atom force */
            fi[i*FI_STRIDE+XX] += fx;
            fi[i*FI_STRIDE+YY] += fy;
            fi[i*FI_STRIDE+ZZ] += fz;
            /* Decrement j-atom force */
            f[aj*F_STRIDE+XX]  -= fx;
            f[aj*F_STRIDE+YY]  -= fy;
            f[aj*F_STRIDE+ZZ]  -= fz;
            /* 9 flops for force addition */
        }
    }
}





// END OF INNER

            cjind++;
        }
        ninner += cjind1 - cjind0;

        /* Add accumulated i-forces to the force array */
        for (i = 0; i < UNROLLI; i++) // 4
        {
            for (d = 0; d < DIM; d++)
            {
                f[(ci*UNROLLI+i)*F_STRIDE+d] += fi[i*FI_STRIDE+d];
            }
        }
        /* Add i forces to shifted force list */
        for (i = 0; i < UNROLLI; i++)
        {
           for (d = 0; d < DIM; d++)
           {
               fshift[ishf+d] += fi[i*FI_STRIDE+d];
           }
        }

        *Vc   += Vc_ci;
    }
}









void
nbnxn_kernel_ref(const nbnxn_pairlist_set_t *nbl_list,
                 const nbnxn_atomdata_t     *nbat,
                 const interaction_const_t  *ic,
                 rvec                       *shift_vec,
                 int                         force_flags,
                 int                         clearF,
                 real                       *fshift,
                 real                       *Vc,
                 real                       *Vvdw)
{
    int                nnbl;
    nbnxn_pairlist_t **nbl;
    int                nb;

    nnbl = nbl_list->nnbl;
    nbl  = nbl_list->nbl;


#pragma omp parallel for schedule(static) num_threads(gmx_omp_nthreads_get(emntNonbonded))
    for (nb = 0; nb < nnbl; nb++) // 12
    {
        nbnxn_atomdata_output_t *out;
        real                    *fshift_p;

        out = &nbat->out[nb];

        clear_f(nbat, nb, out->f);

        fshift_p = out->fshift;

        clear_fshift(fshift_p);

        /* No energy groups */
        out->Vvdw[0] = 0;
        out->Vc[0]   = 0;

        nbnxn_kernel_ref_tab_ener(nbl[nb], nbat,
                                ic,
                                shift_vec,
                                out->f,
                                fshift_p,
                                out->Vvdw,
                                out->Vc);
    }
    reduce_energies_over_lists(nbat, nnbl, Vvdw, Vc);
}

#undef X_STRIDE
#undef F_STRIDE
#undef XI_STRIDE
#undef FI_STRIDE

#undef UNROLLI
#undef UNROLLJ
