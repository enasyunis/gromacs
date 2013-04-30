#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include "physics.h"
#include "vec.h"
#include "maths.h"
#include "txtdump.h"
#include "bondf.h"
#include "smalloc.h"
#include "pbc.h"
#include "ns.h"
#include "macros.h"
#include "names.h"
#include "gmx_fatal.h"
#include "mshift.h"
#include "main.h"
#include "disre.h"
#include "orires.h"
#include "force.h"
#include "nonbonded.h"

#if !defined GMX_DOUBLE && defined GMX_X86_SSE2
#include "gmx_x86_simd_single.h"
#define SSE_PROPER_DIHEDRALS
#endif

/* Find a better place for this? */
const int cmap_coeff_matrix[] = {
    1, 0, -3,  2, 0, 0,  0,  0, -3,  0,  9, -6,  2,  0, -6,  4,
    0, 0,  0,  0, 0, 0,  0,  0,  3,  0, -9,  6, -2,  0,  6, -4,
    0, 0,  0,  0, 0, 0,  0,  0,  0,  0,  9, -6,  0,  0, -6,  4,
    0, 0,  3, -2, 0, 0,  0,  0,  0,  0, -9,  6,  0,  0,  6, -4,
    0, 0,  0,  0, 1, 0, -3,  2, -2,  0,  6, -4,  1,  0, -3,  2,
    0, 0,  0,  0, 0, 0,  0,  0, -1,  0,  3, -2,  1,  0, -3,  2,
    0, 0,  0,  0, 0, 0,  0,  0,  0,  0, -3,  2,  0,  0,  3, -2,
    0, 0,  0,  0, 0, 0,  3, -2,  0,  0, -6,  4,  0,  0,  3, -2,
    0, 1, -2,  1, 0, 0,  0,  0,  0, -3,  6, -3,  0,  2, -4,  2,
    0, 0,  0,  0, 0, 0,  0,  0,  0,  3, -6,  3,  0, -2,  4, -2,
    0, 0,  0,  0, 0, 0,  0,  0,  0,  0, -3,  3,  0,  0,  2, -2,
    0, 0, -1,  1, 0, 0,  0,  0,  0,  0,  3, -3,  0,  0, -2,  2,
    0, 0,  0,  0, 0, 1, -2,  1,  0, -2,  4, -2,  0,  1, -2,  1,
    0, 0,  0,  0, 0, 0,  0,  0,  0, -1,  2, -1,  0,  1, -2,  1,
    0, 0,  0,  0, 0, 0,  0,  0,  0,  0,  1, -1,  0,  0, -1,  1,
    0, 0,  0,  0, 0, 0, -1,  1,  0,  0,  2, -2,  0,  0, -1,  1
};



static unsigned
calc_bonded_reduction_mask(const t_idef *idef,
                           int shift,
                           int t, int nt)
{// called 
    unsigned mask;
    int      ftype, nb, nat1, nb0, nb1, i, a;

    mask = 0;

    for (ftype = 0; ftype < F_NRE; ftype++)
    {
        if (interaction_function[ftype].flags & IF_BOND &&
            !(ftype == F_CONNBONDS || ftype == F_POSRES) &&
            (ftype<F_GB12 || ftype>F_GB14))
        {
            nb = idef->il[ftype].nr;
            if (nb > 0)
            {
                nat1 = interaction_function[ftype].nratoms + 1;

                /* Divide this interaction equally over the threads.
                 * This is not stored: should match division in calc_bonds.
                 */
                nb0 = (((nb/nat1)* t   )/nt)*nat1;
                nb1 = (((nb/nat1)*(t+1))/nt)*nat1;

                for (i = nb0; i < nb1; i += nat1)
                {
                    for (a = 1; a < nat1; a++)
                    {
                        mask |= (1U << (idef->il[ftype].iatoms[i+a]>>shift));
                    }
                }
            }
        }
    }

    return mask;
}

void init_bonded_thread_force_reduction(t_forcerec   *fr,
                                        const t_idef *idef)
{ //called
#define MAX_BLOCK_BITS 32
    int t;
    int ctot, c, b;

    if (fr->nthreads <= 1)
    {
        fr->red_nblock = 0;

        return;
    }

    /* We divide the force array in a maximum of 32 blocks.
     * Minimum force block reduction size is 2^6=64.
     */
    fr->red_ashift = 6;
    while (fr->natoms_force > (int)(MAX_BLOCK_BITS*(1U<<fr->red_ashift)))
    {
        fr->red_ashift++;
    }
    if (debug)
    {
        fprintf(debug, "bonded force buffer block atom shift %d bits\n",
                fr->red_ashift);
    }

    /* Determine to which blocks each thread's bonded force calculation
     * contributes. Store this is a mask for each thread.
     */
#pragma omp parallel for num_threads(fr->nthreads) schedule(static)
    for (t = 1; t < fr->nthreads; t++)
    {
        fr->f_t[t].red_mask =
            calc_bonded_reduction_mask(idef, fr->red_ashift, t, fr->nthreads);
    }

    /* Determine the maximum number of blocks we need to reduce over */
    fr->red_nblock = 0;
    ctot           = 0;
    for (t = 0; t < fr->nthreads; t++)
    {
        c = 0;
        for (b = 0; b < MAX_BLOCK_BITS; b++)
        {
            if (fr->f_t[t].red_mask & (1U<<b))
            {
                fr->red_nblock = max(fr->red_nblock, b+1);
                c++;
            }
        }
        if (debug)
        {
            fprintf(debug, "thread %d flags %x count %d\n",
                    t, fr->f_t[t].red_mask, c);
        }
        ctot += c;
    }
    if (debug)
    {
        fprintf(debug, "Number of blocks to reduce: %d of size %d\n",
                fr->red_nblock, 1<<fr->red_ashift);
        fprintf(debug, "Reduction density %.2f density/#thread %.2f\n",
                ctot*(1<<fr->red_ashift)/(double)fr->natoms_force,
                ctot*(1<<fr->red_ashift)/(double)(fr->natoms_force*fr->nthreads));
    }
}

static void zero_thread_forces(f_thread_t *f_t, int n,
                               int nblock, int blocksize)
{ // called 
    int b, a0, a1, a, i, j;

    if (n > f_t->f_nalloc)
    {
        f_t->f_nalloc = over_alloc_large(n);
        srenew(f_t->f, f_t->f_nalloc);
    }

    if (f_t->red_mask != 0)
    {
        for (b = 0; b < nblock; b++)
        {
            if (f_t->red_mask && (1U<<b))
            {
                a0 = b*blocksize;
                a1 = min((b+1)*blocksize, n);
                for (a = a0; a < a1; a++)
                {
                    clear_rvec(f_t->f[a]);
                }
            }
        }
    }
    for (i = 0; i < SHIFTS; i++)
    {
        clear_rvec(f_t->fshift[i]);
    }
    for (i = 0; i < F_NRE; i++)
    {
        f_t->ener[i] = 0;
    }
    for (i = 0; i < egNR; i++)
    {
        for (j = 0; j < f_t->grpp.nener; j++)
        {
            f_t->grpp.ener[i][j] = 0;
        }
    }
    for (i = 0; i < efptNR; i++)
    {
        f_t->dvdl[i] = 0;
    }
}


static void reduce_thread_forces(int n, rvec *f, rvec *fshift,
                                 real *ener, gmx_grppairener_t *grpp, real *dvdl,
                                 int nthreads, f_thread_t *f_t,
                                 int nblock, int block_size,
                                 gmx_bool bCalcEnerVir,
                                 gmx_bool bDHDL)
{ //called 

    /* When necessary, reduce energy and virial using one thread only */
    if (bCalcEnerVir)
    {
        int t, i, j;

        for (i = 0; i < SHIFTS; i++)
        {
            for (t = 1; t < nthreads; t++)
            {
                rvec_inc(fshift[i], f_t[t].fshift[i]);
            }
        }
        for (i = 0; i < F_NRE; i++)
        {
            for (t = 1; t < nthreads; t++)
            {
                ener[i] += f_t[t].ener[i];
            }
        }
        for (i = 0; i < egNR; i++)
        {
            for (j = 0; j < f_t[1].grpp.nener; j++)
            {
                for (t = 1; t < nthreads; t++)
                {

                    grpp->ener[i][j] += f_t[t].grpp.ener[i][j];
                }
            }
        }
        if (bDHDL)
        {
            for (i = 0; i < efptNR; i++)
            {

                for (t = 1; t < nthreads; t++)
                {
                    dvdl[i] += f_t[t].dvdl[i];
                }
            }
        }
    }
}


void calc_bonds(FILE *fplog, const gmx_multisim_t *ms,
                const t_idef *idef,
                rvec x[], history_t *hist,
                rvec f[], t_forcerec *fr,
                const t_pbc *pbc, const t_graph *g,
                gmx_enerdata_t *enerd, t_nrnb *nrnb,
                real *lambda,
                const t_mdatoms *md,
                t_fcdata *fcd, int *global_atom_index,
                t_atomtypes *atype, gmx_genborn_t *born,
                int force_flags,
                gmx_bool bPrintSepPot, gmx_large_int_t step)
{//called 
    gmx_bool      bCalcEnerVir;
    int           i;
    real          v, dvdl[efptNR], dvdl_dum[efptNR]; /* The dummy array is to have a place to store the dhdl at other values
                                                        of lambda, which will be thrown away in the end*/
    const  t_pbc *pbc_null;
    char          buf[22];
    int           thread;

    bCalcEnerVir = (force_flags & (GMX_FORCE_VIRIAL | GMX_FORCE_ENERGY));

    for (i = 0; i < efptNR; i++)
    {
        dvdl[i] = 0.0;
    }
    if (fr->bMolPBC)
    {
        pbc_null = pbc;
    }
    else
    {
        pbc_null = NULL;
    }
    if (bPrintSepPot)
    {
        fprintf(fplog, "Step %s: bonded V and dVdl for this node\n",
                gmx_step_str(step, buf));
    }

#ifdef DEBUG
    if (g && debug)
    {
        p_graph(debug, "Bondage is fun", g);
    }
#endif

    /* Do pre force calculation stuff which might require communication */
    if (idef->il[F_ORIRES].nr)
    {
        enerd->term[F_ORIRESDEV] =
            calc_orires_dev(ms, idef->il[F_ORIRES].nr,
                            idef->il[F_ORIRES].iatoms,
                            idef->iparams, md, (const rvec*)x,
                            pbc_null, fcd, hist);
    }
    if (idef->il[F_DISRES].nr)
    {
        calc_disres_R_6(ms, idef->il[F_DISRES].nr,
                        idef->il[F_DISRES].iatoms,
                        idef->iparams, (const rvec*)x, pbc_null,
                        fcd, hist);
    }

#pragma omp parallel for num_threads(fr->nthreads) schedule(static)
    for (thread = 0; thread < fr->nthreads; thread++)
    {
        int                ftype, nbonds, ind, nat1;
        real              *epot, v;
        /* thread stuff */
        rvec              *ft, *fshift;
        real              *dvdlt;
        gmx_grppairener_t *grpp;
        int                nb0, nbn;

        if (thread == 0)
        {
            ft     = f;
            fshift = fr->fshift;
            epot   = enerd->term;
            grpp   = &enerd->grpp;
            dvdlt  = dvdl;
        }
        else
        {
            zero_thread_forces(&fr->f_t[thread], fr->natoms_force,
                               fr->red_nblock, 1<<fr->red_ashift);

            ft     = fr->f_t[thread].f;
            fshift = fr->f_t[thread].fshift;
            epot   = fr->f_t[thread].ener;
            grpp   = &fr->f_t[thread].grpp;
            dvdlt  = fr->f_t[thread].dvdl;
        }
    }
    if (fr->nthreads > 1)
    {
        reduce_thread_forces(fr->natoms_force, f, fr->fshift,
                             enerd->term, &enerd->grpp, dvdl,
                             fr->nthreads, fr->f_t,
                             fr->red_nblock, 1<<fr->red_ashift,
                             bCalcEnerVir,
                             force_flags & GMX_FORCE_DHDL);
    }
    if (force_flags & GMX_FORCE_DHDL)
    {
        for (i = 0; i < efptNR; i++)
        {
            enerd->dvdl_nonlin[i] += dvdl[i];
        }
    }

    /* Copy the sum of violations for the distance restraints from fcd */
    if (fcd)
    {
        enerd->term[F_DISRESVIOL] = fcd->disres.sumviol;

    }
}

real unimplemented(int nbonds,
                   const t_iatom forceatoms[], const t_iparams forceparams[],
                   const rvec x[], rvec f[], rvec fshift[],
                   const t_pbc *pbc, const t_graph *g,
                   real lambda, real *dvdlambda,
                   const t_mdatoms *md, t_fcdata *fcd,
                   int *global_atom_index)
{ // kept for compilation only
    gmx_impl("*** you are using a not implemented function");

    return 0.0; /* To make the compiler happy */
}


