#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include "smalloc.h"
#include "macros.h"
#include "vec.h"
#include "nbnxn_consts.h"
#include "nbnxn_internal.h"
#include "nbnxn_search.h"
#include "nbnxn_atomdata.h"
#include "gmx_omp_nthreads.h"

/* Default nbnxn allocation routine, allocates NBNXN_MEM_ALIGN byte aligned */
void nbnxn_alloc_aligned(void **ptr, size_t nbytes)
{ // called  
    *ptr = save_malloc_aligned("ptr", __FILE__, __LINE__, nbytes, 1, NBNXN_MEM_ALIGN);
}


/* Reallocation wrapper function for nbnxn data structures */
void nbnxn_realloc_void(void **ptr,
                        int nbytes_copy, int nbytes_new,
                        nbnxn_alloc_t *ma,
                        nbnxn_free_t  *mf)
{ // called 
    void *ptr_new;

    ma(&ptr_new, nbytes_new);

    *ptr = ptr_new;
}

/* Reallocate the nbnxn_atomdata_t for a size of n atoms */
void nbnxn_atomdata_realloc(nbnxn_atomdata_t *nbat, int n)
{ // called 
    int t;

    nbnxn_realloc_void((void **)&nbat->type,
                       nbat->natoms*sizeof(*nbat->type),
                       n*sizeof(*nbat->type),
                       nbat->alloc, nbat->free);
    nbnxn_realloc_void((void **)&nbat->lj_comb,
                       nbat->natoms*2*sizeof(*nbat->lj_comb),
                       n*2*sizeof(*nbat->lj_comb),
                       nbat->alloc, nbat->free);
    nbnxn_realloc_void((void **)&nbat->q,
                           nbat->natoms*sizeof(*nbat->q),
                           n*sizeof(*nbat->q),
                           nbat->alloc, nbat->free);
    nbnxn_realloc_void((void **)&nbat->x,
                       nbat->natoms*nbat->xstride*sizeof(*nbat->x),
                       n*nbat->xstride*sizeof(*nbat->x),
                       nbat->alloc, nbat->free);
    for (t = 0; t < nbat->nout; t++) // 12
    {
        /* Allocate one element extra for possible signaling with CUDA */
        nbnxn_realloc_void((void **)&nbat->out[t].f,
                           nbat->natoms*nbat->fstride*sizeof(*nbat->out[t].f),
                           n*nbat->fstride*sizeof(*nbat->out[t].f),
                           nbat->alloc, nbat->free);
    }
    nbat->nalloc = n;
}

/* Initializes an nbnxn_atomdata_output_t data structure */
static void nbnxn_atomdata_output_init(nbnxn_atomdata_output_t *out,
                                       int nb_kernel_type,
                                       int nenergrp, int stride,
                                       nbnxn_alloc_t *ma)
{  // called 
    int cj_size;

    out->f = NULL;
    ma((void **)&out->fshift, SHIFTS*DIM*sizeof(*out->fshift));
    out->nV = nenergrp*nenergrp;
    ma((void **)&out->Vvdw, out->nV*sizeof(*out->Vvdw));
    ma((void **)&out->Vc, out->nV*sizeof(*out->Vc  ));

    out->nVS = 0;
}

static void copy_int_to_nbat_int(const int *a, int na, int na_round,
                                 const int *in, int fill, int *innb)
{  // called 
    int i, j;
    j = 0;
    for (i = 0; i < na; i++) // values ranging from 27 to 49
    {
        innb[j++] = in[a[i]];
    }
    /* Complete the partially filled last cell with fill */
    for (; i < na_round; i++) // starting from i=na, values ranging from 28 to 52
    {
        innb[j++] = fill;
    }
}


void copy_rvec_to_nbat_real(const int *a, int na, int na_round,
                            rvec *x, int nbatFormat, real *xnb, int a0,
                            int cx, int cy, int cz)
{  // called 
    int i, j;

/* We might need to place filler particles to fill up the cell to na_round.
 * The coefficients (LJ and q) for such particles are zero.
 * But we might still get NaN as 0*NaN when distances are too small.
 * We hope that -107 nm is far away enough from to zero
 * to avoid accidental short distances to particles shifted down for pbc.
 */
#define NBAT_FAR_AWAY 107
    j = a0*STRIDE_XYZ;
    for (i = 0; i < na; i++)
    {
         xnb[j++] = x[a[i]][XX];
         xnb[j++] = x[a[i]][YY];
         xnb[j++] = x[a[i]][ZZ];
    }
    /* Complete the partially filled last cell with copies of the last element.
     * This simplifies the bounding box calculation and avoid
     * numerical issues with atoms that are coincidentally close.
     */
    for (; i < na_round; i++)
    {
         xnb[j++] = -NBAT_FAR_AWAY*(1 + cx);
         xnb[j++] = -NBAT_FAR_AWAY*(1 + cy);
         xnb[j++] = -NBAT_FAR_AWAY*(1 + cz + i);
    }
}

/* Determines the combination rule (or none) to be used, stores it,
 * and sets the LJ parameters required with the rule.
 */
static void set_combination_rule_data(nbnxn_atomdata_t *nbat)
{ // called 
    int  nt, i;

    nt = nbat->ntype; // 2
    nbat->comb_rule = ljcrGEOM;

    for (i = 0; i < nt; i++) // 2
    {
         /* Copy the diagonal from the nbfp matrix */
         nbat->nbfp_comb[i*2  ] = sqrt(nbat->nbfp[(i*nt+i)*2  ]);
         nbat->nbfp_comb[i*2+1] = sqrt(nbat->nbfp[(i*nt+i)*2+1]);
    }
}

/* Initializes an nbnxn_atomdata_t data structure */
void nbnxn_atomdata_init(FILE *fp,
                         nbnxn_atomdata_t *nbat,
                         int nb_kernel_type,
                         int ntype, const real *nbfp,
                         int n_energygroups,
                         int nout,
                         nbnxn_alloc_t *alloc,
                         nbnxn_free_t  *free)
{  // called 
    int      i, j;

    nbat->alloc = nbnxn_alloc_aligned;
    nbat->free = free;

    nbat->ntype = ntype + 1;// nbat->ntype:=2 , ntype=1
    nbat->alloc((void **)&nbat->nbfp,
                nbat->ntype*nbat->ntype*2*sizeof(*nbat->nbfp));
    nbat->alloc((void **)&nbat->nbfp_comb, nbat->ntype*2*sizeof(*nbat->nbfp_comb));


    /* We prefer the geometic combination rule,
     * as that gives a slightly faster kernel than the LB rule.
     */
    nbat->comb_rule = ljcrGEOM;
    fprintf(fp, "Using %s Lennard-Jones combination rule\n\n",
           nbat->comb_rule == ljcrGEOM ? "geometric" : "Lorentz-Berthelot");

    set_combination_rule_data(nbat);

    nbat->natoms  = 0;
    nbat->type    = NULL;
    nbat->lj_comb = NULL;
    int pack_x;

    nbat->XFormat = nbatXYZ;
    nbat->FFormat = nbat->XFormat;
    nbat->q        = NULL;
    nbat->nenergrp = n_energygroups;
    nbat->neg_2log = 1; 
    nbat->energrp = NULL;
    nbat->alloc((void **)&nbat->shift_vec, SHIFTS*sizeof(*nbat->shift_vec));
    nbat->xstride = (nbat->XFormat == nbatXYZQ ? STRIDE_XYZQ : DIM);
    nbat->fstride = (nbat->FFormat == nbatXYZQ ? STRIDE_XYZQ : DIM);
    nbat->x       = NULL;


    /* Initialize the output data structures */
    nbat->nout    = nout;
    snew(nbat->out, nbat->nout);
    nbat->nalloc  = 0;
    for (i = 0; i < nbat->nout; i++) // 12 
    {
        nbnxn_atomdata_output_init(&nbat->out[i],
                                   nb_kernel_type,
                                   nbat->nenergrp, 1<<nbat->neg_2log,
                                   nbat->alloc);
    }
    nbat->buffer_flags.flag        = NULL;
    nbat->buffer_flags.flag_nalloc = 0;
}


/* Sets the atom type and LJ data in nbnxn_atomdata_t */
static void nbnxn_atomdata_set_atomtypes(nbnxn_atomdata_t    *nbat,
                                         int                  ngrid,
                                         const nbnxn_search_t nbs,
                                         const int           *type)
{ // called 
    int                 i, ncz, ash;
    const nbnxn_grid_t *grid;
    grid = &nbs->grid[0];
    /* Loop over all columns and copy and fill */
    for (i = 0; i < grid->ncx*grid->ncy; i++) // 81 (aka 9 in each direction)
    {
        ncz = grid->cxy_ind[i+1] - grid->cxy_ind[i];
        ash = (grid->cell0 + grid->cxy_ind[i])*grid->na_sc;

        copy_int_to_nbat_int(nbs->a+ash, grid->cxy_na[i], ncz*grid->na_sc,
                               type, nbat->ntype-1, nbat->type+ash);

    }
}

/* Sets the charges in nbnxn_atomdata_t *nbat */
static void nbnxn_atomdata_set_charges(nbnxn_atomdata_t    *nbat,
                                       int                  ngrid,
                                       const nbnxn_search_t nbs,
                                       const real          *charge)
{ // called 
    int                 g=0, cxy, ncz, ash, na, na_round, i, j;
    real               *q;
    const nbnxn_grid_t *grid;

        grid = &nbs->grid[g];

        /* Loop over all columns and copy and fill */
        for (cxy = 0; cxy < grid->ncx*grid->ncy; cxy++)
        {
            ash      = (grid->cell0 + grid->cxy_ind[cxy])*grid->na_sc;
            na       = grid->cxy_na[cxy];
            na_round = (grid->cxy_ind[cxy+1] - grid->cxy_ind[cxy])*grid->na_sc;

            q = nbat->q + ash;
            for (i = 0; i < na; i++)
            {
                *q = charge[nbs->a[ash+i]];
                 q++;
            }
            /* Complete the partially filled last cell with zeros */
            for (; i < na_round; i++)
            {
                *q = 0;
                 q++;
            }
        }
}

/* Sets all required atom parameter data in nbnxn_atomdata_t */
void nbnxn_atomdata_set(nbnxn_atomdata_t    *nbat,
                        int                  locality,
                        const nbnxn_search_t nbs,
                        const t_mdatoms     *mdatoms,
                        const int           *atinfo)
{ // called 
    int ngrid;

    ngrid = nbs->ngrid;

    nbnxn_atomdata_set_atomtypes(nbat, ngrid, nbs, mdatoms->typeA);

    nbnxn_atomdata_set_charges(nbat, ngrid, nbs, mdatoms->chargeA);

}

/* Copies the shift vector array to nbnxn_atomdata_t */
void nbnxn_atomdata_copy_shiftvec(gmx_bool          bDynamicBox,
                                  rvec             *shift_vec,
                                  nbnxn_atomdata_t *nbat)
{ // called 
    int i;
    nbat->bDynamicBox = bDynamicBox;
    for (i = 0; i < SHIFTS; i++) // 45
    {
        copy_rvec(shift_vec[i], nbat->shift_vec[i]);
    }
}

static void
nbnxn_atomdata_reduce_reals(real * gmx_restrict dest,
                            gmx_bool bDestSet,
                            real ** gmx_restrict src,
                            int nsrc,
                            int i0, int i1)
{ // called, everything here gets called at some poitn 
    int i, s;

    if (bDestSet)
    {
        /* The destination buffer contains data, add to it */
        for (i = i0; i < i1; i++) // i1 ranges from 0 to 9360
        {
            for (s = 0; s < nsrc; s++) // ranges from 4 to 10
            {
                dest[i] += src[s][i];
            }
        }
    }
    else
    {
        /* The destination buffer is unitialized, set it first */
        for (i = i0; i < i1; i++)
        {
            dest[i] = src[0][i];
            for (s = 1; s < nsrc; s++)
            {
                dest[i] += src[s][i];
            }
        }
    }
}


/* Add part of the force array(s) from nbnxn_atomdata_t to f */
static void
nbnxn_atomdata_add_nbat_f_to_f_part(const nbnxn_search_t nbs,
                                    const nbnxn_atomdata_t *nbat,
                                    nbnxn_atomdata_output_t *out,
                                    int nfa,
                                    int a0, int a1,
                                    rvec *f)
{ // called 
    int         a, i;
    const int  *cell;
    const real *fnb;

    cell = nbs->cell;
    /* Loop over all columns and copy and fill */
    fnb = out[0].f;

    for (a = a0; a < a1; a++) // 3000 divided equally on threads.
    {
       i = cell[a]*nbat->fstride;

       f[a][XX] += fnb[i];
       f[a][YY] += fnb[i+1];
       f[a][ZZ] += fnb[i+2];
    }
}

/* Add the force array(s) from nbnxn_atomdata_t to f */
void nbnxn_atomdata_add_nbat_f_to_f(const nbnxn_search_t    nbs,
                                    int                     locality,
                                    const nbnxn_atomdata_t *nbat,
                                    rvec                   *f)
{  // called 
    int a0 = 0, na = 0;
    int nth, th;

    nbs_cycle_start(&nbs->cc[enbsCCreducef]);

    a0 = 0;
    na = nbs->natoms_nonlocal;

    nth = gmx_omp_nthreads_get(emntNonbonded);

    if (nbat->nout > 1) // number of openmp threads.
    {
        /* Reduce the force thread output buffers into buffer 0, before adding
         * them to the, differently ordered, "real" force buffer.
         */
#pragma omp parallel for num_threads(nth) schedule(static)
        for (th = 0; th < nth; th++)
        {
            const nbnxn_buffer_flags_t *flags;
            int   b0, b1, b;
            int   i0, i1;
            int   nfptr;
            real *fptr[NBNXN_BUFFERFLAG_MAX_THREADS];
            int   out;

            flags = &nbat->buffer_flags;

            /* Calculate the cell-block range for our thread */
            b0 = (flags->nflag* th   )/nth;
            b1 = (flags->nflag*(th+1))/nth;
            for (b = b0; b < b1; b++)
            {
                i0 =  b   *NBNXN_BUFFERFLAG_SIZE*nbat->fstride;
                i1 = (b+1)*NBNXN_BUFFERFLAG_SIZE*nbat->fstride;

                nfptr = 0;
                for (out = 1; out < nbat->nout; out++)
                {
                    if (flags->flag[b] & (1U<<out))
                    {
                        fptr[nfptr++] = nbat->out[out].f;
                    } 
                }
                if (nfptr > 0)
                {
                    nbnxn_atomdata_reduce_reals
                        (nbat->out[0].f,
                        flags->flag[b] & (1U<<0),
                        fptr, nfptr,
                        i0, i1);
                }
            }
        }
    }

#pragma omp parallel for num_threads(nth) schedule(static)
    for (th = 0; th < nth; th++)
    {
        nbnxn_atomdata_add_nbat_f_to_f_part(nbs, nbat,
                                            nbat->out,
                                            1,
                                            a0+((th+0)*na)/nth,
                                            a0+((th+1)*na)/nth,
                                            f);
    }

    nbs_cycle_stop(&nbs->cc[enbsCCreducef]);
}

/* Adds the shift forces from nbnxn_atomdata_t to fshift */
void nbnxn_atomdata_add_nbat_fshift_to_fshift(const nbnxn_atomdata_t *nbat,
                                              rvec                   *fshift)
{ // called 
    const nbnxn_atomdata_output_t *out;
    int  th;
    int  s;
    rvec sum;

    out = nbat->out;
    for (s = 0; s < SHIFTS; s++) // 45
    {
        clear_rvec(sum);
        for (th = 0; th < nbat->nout; th++) // 12
        {
            sum[XX] += out[th].fshift[s*DIM+XX];
            sum[YY] += out[th].fshift[s*DIM+YY];
            sum[ZZ] += out[th].fshift[s*DIM+ZZ];
        }
        rvec_inc(fshift[s], sum);
    }
}
