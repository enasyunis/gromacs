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
#include "main.h"
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


static void zero_thread_forces(f_thread_t *f_t, int n,
                               int nblock, int blocksize)
{ // called 
    int b, a0, a1, a, i, j;

    f_t->f_nalloc = over_alloc_large(n);
    srenew(f_t->f, f_t->f_nalloc);

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
}


void calc_bonds(FILE *fplog, const gmx_multisim_t *ms,
                const t_idef *idef,
                rvec x[], history_t *hist,
                rvec f[], t_forcerec *fr,
                const t_pbc *pbc, const t_graph *g,
                gmx_enerdata_t *enerd, t_nrnb *nrnb,
                real *lambda,
                const t_mdatoms *md,
                int *global_atom_index,
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
    pbc_null = NULL;
    fprintf(fplog, "Step %s: bonded V and dVdl for this node\n",
                gmx_step_str(step, buf));



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
                               fr->red_nblock, 1<<7);

            ft     = fr->f_t[thread].f;
            fshift = fr->f_t[thread].fshift;
            epot   = fr->f_t[thread].ener;
            grpp   = &fr->f_t[thread].grpp;
            dvdlt  = fr->f_t[thread].dvdl;
        }
    }
    reduce_thread_forces(fr->natoms_force, f, fr->fshift,
                             enerd->term, &enerd->grpp, dvdl,
                             fr->nthreads, fr->f_t,
                             fr->red_nblock, 1<<7,
                             bCalcEnerVir,
                             force_flags & GMX_FORCE_DHDL);

    /* Copy the sum of violations for the distance restraints from fcd */
    enerd->term[F_DISRESVIOL] = 0.0;
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


