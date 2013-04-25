#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include "sysstuff.h"
#include "smalloc.h"
#include "macros.h"
#include "maths.h"
#include "vec.h"
#include "pbc.h"
#include "nbnxn_consts.h"
#include "nbnxn_internal.h"
#include "nbnxn_atomdata.h"
#include "nbnxn_search.h"
#include "gmx_cyclecounter.h"
#include "gmxfio.h"
#include "gmx_omp_nthreads.h"
#include "nrnb.h"


/* Pair search box lower and upper corner in x,y,z.
 * Store this in 4 iso 3 reals, which is useful with SSE.
 * To avoid complicating the code we also use 4 without SSE.
 */
#define NNBSBB_C         4
#define NNBSBB_B         (2*NNBSBB_C)
/* Pair search box lower and upper bound in z only. */
#define NNBSBB_D         2
/* Pair search box lower and upper corner x,y,z indices */
#define BBL_X  0
#define BBL_Y  1
#define BBL_Z  2
#define BBU_X  4
#define BBU_Y  5
#define BBU_Z  6


/* We use SSE or AVX-128bit for bounding box calculations */


/* Include basic SSE2 stuff */
#include <emmintrin.h>


/* The width of SSE/AVX128 with single precision for bounding boxes with GPU.
 * Here AVX-256 turns out to be slightly slower than AVX-128.
 */
#define STRIDE_PBB        4
#define STRIDE_PBB_2LOG   2


#ifdef GMX_NBNXN_SIMD

/* The functions below are macros as they are performance sensitive */

/* 4x4 list, pack=4: no complex conversion required */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J4(ci)   (ci)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J4(ci)  ((ci)*STRIDE_P4)
#define X_IND_CJ_J4(cj)  ((cj)*STRIDE_P4)

/* 4x2 list, pack=4: j-cluster size is half the packing width */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J2(ci)  ((ci)<<1)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J2(ci)  ((ci)*STRIDE_P4)
#define X_IND_CJ_J2(cj)  (((cj)>>1)*STRIDE_P4 + ((cj) & 1)*(PACK_X4>>1))

/* 4x8 list, pack=8: i-cluster size is half the packing width */
/* i-cluster to j-cluster conversion */
#define CI_TO_CJ_J8(ci)  ((ci)>>1)
/* cluster index to coordinate array index conversion */
#define X_IND_CI_J8(ci)  (((ci)>>1)*STRIDE_P8 + ((ci) & 1)*(PACK_X8>>1))
#define X_IND_CJ_J8(cj)  ((cj)*STRIDE_P8)

/* The j-cluster size is matched to the SIMD width */
#if GMX_NBNXN_SIMD_BITWIDTH == 128
#ifdef GMX_DOUBLE
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J2(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J2(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J2(cj)
#else
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J4(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J4(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J4(cj)
#endif
#else
#if GMX_NBNXN_SIMD_BITWIDTH == 256
#ifdef GMX_DOUBLE
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J4(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J4(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J4(cj)
#else
#define CI_TO_CJ_SIMD_4XN(ci)  CI_TO_CJ_J8(ci)
#define X_IND_CI_SIMD_4XN(ci)  X_IND_CI_J8(ci)
#define X_IND_CJ_SIMD_4XN(cj)  X_IND_CJ_J8(cj)
/* Half SIMD with j-cluster size */
#define CI_TO_CJ_SIMD_2XNN(ci) CI_TO_CJ_J4(ci)
#define X_IND_CI_SIMD_2XNN(ci) X_IND_CI_J4(ci)
#define X_IND_CJ_SIMD_2XNN(cj) X_IND_CJ_J4(cj)
#endif
#else
#error "unsupported GMX_NBNXN_SIMD_WIDTH"
#endif
#endif

#endif /* GMX_NBNXN_SIMD */


/* Interaction masks for 4xN atom interactions.
 * Bit i*CJ_SIZE + j tells if atom i and j interact.
 */
/* All interaction mask is the same for all kernels */
#define NBNXN_INT_MASK_ALL        0xffffffff
/* 4x4 kernel diagonal mask */
#define NBNXN_INT_MASK_DIAG       0x08ce
/* 4x2 kernel diagonal masks */
#define NBNXN_INT_MASK_DIAG_J2_0  0x0002
#define NBNXN_INT_MASK_DIAG_J2_1  0x002F
/* 4x8 kernel diagonal masks */
#define NBNXN_INT_MASK_DIAG_J8_0  0xf0f8fcfe
#define NBNXN_INT_MASK_DIAG_J8_1  0x0080c0e0


/* Store bounding boxes corners as quadruplets: xxxxyyyyzzzz */
#define NBNXN_BBXXXX
/* Size of bounding box corners quadruplet */
#define NNBSBB_XXXX      (NNBSBB_D*DIM*STRIDE_PBB)

/* We shift the i-particles backward for PBC.
 * This leads to more conditionals than shifting forward.
 * We do this to get more balanced pair lists.
 */
#define NBNXN_SHIFT_BACKWARD


/* This define is a lazy way to avoid interdependence of the grid
 * and searching data structures.
 */
#define NBNXN_NA_SC_MAX (GPU_NSUBCELL*NBNXN_GPU_CLUSTER_SIZE)


static void nbs_cycle_clear(nbnxn_cycle_t *cc)
{// called
    int i;

    for (i = 0; i < enbsCCnr; i++)
    {
        cc[i].count = 0;
        cc[i].c     = 0;
    }
}

static void nbnxn_grid_init(nbnxn_grid_t * grid)
{ // called
    grid->cxy_na      = NULL;
    grid->cxy_ind     = NULL;
    grid->cxy_nalloc  = 0;
    grid->bb          = NULL;
    grid->bbj         = NULL;
    grid->nc_nalloc   = 0;
}

static int get_2log(int n)
{ // called
    int log2;

    log2 = 0;
    while ((1<<log2) < n)
    {
        log2++;
    }
    if ((1<<log2) != n)
    {
        gmx_fatal(FARGS, "nbnxn na_c (%d) is not a power of 2", n);
    }

    return log2;
}

static int nbnxn_kernel_to_ci_size(int nb_kernel_type)
{ // called
    switch (nb_kernel_type)
    {
        case nbnxnk4x4_PlainC:
        case nbnxnk4xN_SIMD_4xN:
        case nbnxnk4xN_SIMD_2xNN:
            return NBNXN_CPU_CLUSTER_I_SIZE;
        case nbnxnk8x8x8_CUDA:
        case nbnxnk8x8x8_PlainC:
            /* The cluster size for super/sub lists is only set here.
             * Any value should work for the pair-search and atomdata code.
             * The kernels, of course, might require a particular value.
             */
            return NBNXN_GPU_CLUSTER_SIZE;
        default:
            gmx_incons("unknown kernel type");
    }

    return 0;
}

int nbnxn_kernel_to_cj_size(int nb_kernel_type)
{ // called
    int nbnxn_simd_width = 0;
    int cj_size          = 0;

#ifdef GMX_NBNXN_SIMD
    nbnxn_simd_width = GMX_NBNXN_SIMD_BITWIDTH/(sizeof(real)*8);
#endif

    switch (nb_kernel_type)
    {
        case nbnxnk4x4_PlainC:
            cj_size = NBNXN_CPU_CLUSTER_I_SIZE;
            break;
        case nbnxnk4xN_SIMD_4xN:
            cj_size = nbnxn_simd_width;
            break;
        case nbnxnk4xN_SIMD_2xNN:
            cj_size = nbnxn_simd_width/2;
            break;
        case nbnxnk8x8x8_CUDA:
        case nbnxnk8x8x8_PlainC:
            cj_size = nbnxn_kernel_to_ci_size(nb_kernel_type);
            break;
        default:
            gmx_incons("unknown kernel type");
    }

    return cj_size;
}


gmx_bool nbnxn_kernel_pairlist_simple(int nb_kernel_type)
{ // called
    if (nb_kernel_type == nbnxnkNotSet)
    {
        gmx_fatal(FARGS, "Non-bonded kernel type not set for Verlet-style pair-list.");
    }

    switch (nb_kernel_type)
    {
        case nbnxnk8x8x8_CUDA:
        case nbnxnk8x8x8_PlainC:
            return FALSE;

        case nbnxnk4x4_PlainC:
        case nbnxnk4xN_SIMD_4xN:
        case nbnxnk4xN_SIMD_2xNN:
            return TRUE;

        default:
            gmx_incons("Invalid nonbonded kernel type passed!");
            return FALSE;
    }
}

void nbnxn_init_search(nbnxn_search_t    * nbs_ptr,
                       ivec               *n_dd_cells,
                       gmx_domdec_zones_t *zones,
                       int                 nthread_max)
{ // called
    nbnxn_search_t nbs;
    int            d, g, t;

    snew(nbs, 1);
    *nbs_ptr = nbs;

    nbs->DomDec = (n_dd_cells != NULL);

    clear_ivec(nbs->dd_dim);
    nbs->ngrid = 1;
    if (nbs->DomDec)
    {
        nbs->zones = zones;

        for (d = 0; d < DIM; d++)
        {
            if ((*n_dd_cells)[d] > 1)
            {
                nbs->dd_dim[d] = 1;
                /* Each grid matches a DD zone */
                nbs->ngrid *= 2;
            }
        }
    }

    snew(nbs->grid, nbs->ngrid);
    for (g = 0; g < nbs->ngrid; g++)
    {
        nbnxn_grid_init(&nbs->grid[g]);
    }
    nbs->cell        = NULL;
    nbs->cell_nalloc = 0;
    nbs->a           = NULL;
    nbs->a_nalloc    = 0;

    nbs->nthread_max = nthread_max;

    /* Initialize the work data structures for each thread */
    snew(nbs->work, nbs->nthread_max);
    for (t = 0; t < nbs->nthread_max; t++)
    {
        nbs->work[t].cxy_na           = NULL;
        nbs->work[t].cxy_na_nalloc    = 0;
        nbs->work[t].sort_work        = NULL;
        nbs->work[t].sort_work_nalloc = 0;
    }

    /* Initialize detailed nbsearch cycle counting */
    nbs->print_cycles = (getenv("GMX_NBNXN_CYCLE") != 0);
    nbs->search_count = 0;
    nbs_cycle_clear(nbs->cc);
    for (t = 0; t < nbs->nthread_max; t++)
    {
        nbs_cycle_clear(nbs->work[t].cc);
    }
}

static real grid_atom_density(int n, rvec corner0, rvec corner1)
{ // called
    rvec size;

    rvec_sub(corner1, corner0, size);

    return n/(size[XX]*size[YY]*size[ZZ]);
}

static int set_grid_size_xy(const nbnxn_search_t nbs,
                            nbnxn_grid_t *grid,
                            int dd_zone,
                            int n, rvec corner0, rvec corner1,
                            real atom_density,
                            int XFormat)
{ // called
    rvec size;
    int  na_c;
    real adens, tlen, tlen_x, tlen_y, nc_max;
    int  t;

    rvec_sub(corner1, corner0, size);

    if (n > grid->na_sc)
    {
        /* target cell length */
        if (grid->bSimple)
        {
            /* To minimize the zero interactions, we should make
             * the largest of the i/j cell cubic.
             */
            na_c = max(grid->na_c, grid->na_cj);

            /* Approximately cubic cells */
            tlen   = pow(na_c/atom_density, 1.0/3.0);
            tlen_x = tlen;
            tlen_y = tlen;
        }
        else
        {
            /* Approximately cubic sub cells */
            tlen   = pow(grid->na_c/atom_density, 1.0/3.0);
            tlen_x = tlen*GPU_NSUBCELL_X;
            tlen_y = tlen*GPU_NSUBCELL_Y;
        }
        /* We round ncx and ncy down, because we get less cell pairs
         * in the nbsist when the fixed cell dimensions (x,y) are
         * larger than the variable one (z) than the other way around.
         */
        grid->ncx = max(1, (int)(size[XX]/tlen_x));
        grid->ncy = max(1, (int)(size[YY]/tlen_y));
    }
    else
    {
        grid->ncx = 1;
        grid->ncy = 1;
    }

    grid->sx     = size[XX]/grid->ncx;
    grid->sy     = size[YY]/grid->ncy;
    grid->inv_sx = 1/grid->sx;
    grid->inv_sy = 1/grid->sy;

    if (dd_zone > 0)
    {
        /* This is a non-home zone, add an extra row of cells
         * for particles communicated for bonded interactions.
         * These can be beyond the cut-off. It doesn't matter where
         * they end up on the grid, but for performance it's better
         * if they don't end up in cells that can be within cut-off range.
         */
        grid->ncx++;
        grid->ncy++;
    }

    /* We need one additional cell entry for particles moved by DD */
    if (grid->ncx*grid->ncy+1 > grid->cxy_nalloc)
    {
        grid->cxy_nalloc = over_alloc_large(grid->ncx*grid->ncy+1);
        srenew(grid->cxy_na, grid->cxy_nalloc);
        srenew(grid->cxy_ind, grid->cxy_nalloc+1);
    }
    for (t = 0; t < nbs->nthread_max; t++)
    {
        if (grid->ncx*grid->ncy+1 > nbs->work[t].cxy_na_nalloc)
        {
            nbs->work[t].cxy_na_nalloc = over_alloc_large(grid->ncx*grid->ncy+1);
            srenew(nbs->work[t].cxy_na, nbs->work[t].cxy_na_nalloc);
        }
    }

    /* Worst case scenario of 1 atom in each last cell */
    if (grid->na_cj <= grid->na_c)
    {
        nc_max = n/grid->na_sc + grid->ncx*grid->ncy;
    }
    else
    {
        nc_max = n/grid->na_sc + grid->ncx*grid->ncy*grid->na_cj/grid->na_c;
    }

    if (nc_max > grid->nc_nalloc)
    {
        int bb_nalloc;

        grid->nc_nalloc = over_alloc_large(nc_max);
        srenew(grid->nsubc, grid->nc_nalloc);
        srenew(grid->bbcz, grid->nc_nalloc*NNBSBB_D);
        bb_nalloc = grid->nc_nalloc*GPU_NSUBCELL*NNBSBB_B;
        sfree_aligned(grid->bb);
        /* This snew also zeros the contents, this avoid possible
         * floating exceptions in SSE with the unused bb elements.
         */
        snew_aligned(grid->bb, bb_nalloc, 16);

        if (grid->bSimple)
        {
            if (grid->na_cj == grid->na_c)
            {
                grid->bbj = grid->bb;
            }
            else
            {
                sfree_aligned(grid->bbj);
                snew_aligned(grid->bbj, bb_nalloc*grid->na_c/grid->na_cj, 16);
            }
        }

        srenew(grid->flags, grid->nc_nalloc);
    }

    copy_rvec(corner0, grid->c0);
    copy_rvec(corner1, grid->c1);

    return nc_max;
}

/* We need to sort paricles in grid columns on z-coordinate.
 * As particle are very often distributed homogeneously, we a sorting
 * algorithm similar to pigeonhole sort. We multiply the z-coordinate
 * by a factor, cast to an int and try to store in that hole. If the hole
 * is full, we move this or another particle. A second pass is needed to make
 * contiguous elements. SORT_GRID_OVERSIZE is the ratio of holes to particles.
 * 4 is the optimal value for homogeneous particle distribution and allows
 * for an O(#particles) sort up till distributions were all particles are
 * concentrated in 1/4 of the space. No NlogN fallback is implemented,
 * as it can be expensive to detect imhomogeneous particle distributions.
 * SGSF is the maximum ratio of holes used, in the worst case all particles
 * end up in the last hole and we need #particles extra holes at the end.
 */
#define SORT_GRID_OVERSIZE 4
#define SGSF (SORT_GRID_OVERSIZE + 1)

/* Sort particle index a on coordinates x along dim.
 * Backwards tells if we want decreasing iso increasing coordinates.
 * h0 is the minimum of the coordinate range.
 * invh is the inverse hole spacing.
 * nsort, the theortical hole limit, is only used for debugging.
 * sort is the sorting work array.
 */
static void sort_atoms(int dim, gmx_bool Backwards,
                       int *a, int n, rvec *x,
                       real h0, real invh, int nsort, int *sort)
{ // called
    int i, c;
    int zi, zim, zi_min, zi_max;
    int cp, tmp;

    if (n <= 1)
    {
        /* Nothing to do */
        return;
    }

    /* Determine the index range used, so we can limit it for the second pass */
    zi_min = INT_MAX;
    zi_max = -1;

    /* Sort the particles using a simple index sort */
    for (i = 0; i < n; i++)
    {
        /* The cast takes care of float-point rounding effects below zero.
         * This code assumes particles are less than 1/SORT_GRID_OVERSIZE
         * times the box height out of the box.
         */
        zi = (int)((x[a[i]][dim] - h0)*invh);

#ifdef DEBUG_NBNXN_GRIDDING
        if (zi < 0 || zi >= nsort)
        {
            gmx_fatal(FARGS, "(int)((x[%d][%c]=%f - %f)*%f) = %d, not in 0 - %d\n",
                      a[i], 'x'+dim, x[a[i]][dim], h0, invh, zi, nsort);
        }
#endif

        /* Ideally this particle should go in sort cell zi,
         * but that might already be in use,
         * in that case find the first empty cell higher up
         */
        if (sort[zi] < 0)
        {
            sort[zi] = a[i];
            zi_min   = min(zi_min, zi);
            zi_max   = max(zi_max, zi);
        }
        else
        {
            /* We have multiple atoms in the same sorting slot.
             * Sort on real z for minimal bounding box size.
             * There is an extra check for identical z to ensure
             * well-defined output order, independent of input order
             * to ensure binary reproducibility after restarts.
             */
            while (sort[zi] >= 0 && ( x[a[i]][dim] >  x[sort[zi]][dim] ||
                                      (x[a[i]][dim] == x[sort[zi]][dim] &&
                                       a[i] > sort[zi])))
            {
                zi++;
            }

            if (sort[zi] >= 0)
            {
                /* Shift all elements by one slot until we find an empty slot */
                cp  = sort[zi];
                zim = zi + 1;
                while (sort[zim] >= 0)
                {
                    tmp       = sort[zim];
                    sort[zim] = cp;
                    cp        = tmp;
                    zim++;
                }
                sort[zim] = cp;
                zi_max    = max(zi_max, zim);
            }
            sort[zi] = a[i];
            zi_max   = max(zi_max, zi);
        }
    }

    c = 0;
    if (!Backwards)
    {
        for (zi = 0; zi < nsort; zi++)
        {
            if (sort[zi] >= 0)
            {
                a[c++]   = sort[zi];
                sort[zi] = -1;
            }
        }
    }
    else
    {
        for (zi = zi_max; zi >= zi_min; zi--)
        {
            if (sort[zi] >= 0)
            {
                a[c++]   = sort[zi];
                sort[zi] = -1;
            }
        }
    }
    if (c < n)
    {
        gmx_incons("Lost particles while sorting");
    }
}

#ifdef GMX_DOUBLE
#define R2F_D(x) ((float)((x) >= 0 ? ((1-GMX_FLOAT_EPS)*(x)) : ((1+GMX_FLOAT_EPS)*(x))))
#define R2F_U(x) ((float)((x) >= 0 ? ((1+GMX_FLOAT_EPS)*(x)) : ((1-GMX_FLOAT_EPS)*(x))))
#else
#define R2F_D(x) (x)
#define R2F_U(x) (x)
#endif

/* Coordinate order x,y,z, bb order xyz0 */
static void calc_bounding_box(int na, int stride, const real *x, float *bb)
{ // called
    int  i, j;
    real xl, xh, yl, yh, zl, zh;

    i  = 0;
    xl = x[i+XX];
    xh = x[i+XX];
    yl = x[i+YY];
    yh = x[i+YY];
    zl = x[i+ZZ];
    zh = x[i+ZZ];
    i += stride;
    for (j = 1; j < na; j++)
    {
        xl = min(xl, x[i+XX]);
        xh = max(xh, x[i+XX]);
        yl = min(yl, x[i+YY]);
        yh = max(yh, x[i+YY]);
        zl = min(zl, x[i+ZZ]);
        zh = max(zh, x[i+ZZ]);
        i += stride;
    }
    /* Note: possible double to float conversion here */
    bb[BBL_X] = R2F_D(xl);
    bb[BBL_Y] = R2F_D(yl);
    bb[BBL_Z] = R2F_D(zl);
    bb[BBU_X] = R2F_U(xh);
    bb[BBU_Y] = R2F_U(yh);
    bb[BBU_Z] = R2F_U(zh);
}

/* Potentially sorts atoms on LJ coefficients !=0 and ==0.
 * Also sets interaction flags.
 */
void sort_on_lj(nbnxn_atomdata_t *nbat, int na_c,
                int a0, int a1, const int *atinfo,
                int *order,
                int *flags)
{ // called
    int      subc, s, a, n1, n2, a_lj_max, i, j;
    int      sort1[NBNXN_NA_SC_MAX/GPU_NSUBCELL];
    int      sort2[NBNXN_NA_SC_MAX/GPU_NSUBCELL];
    gmx_bool haveQ;

    *flags = 0;

    subc = 0;
    for (s = a0; s < a1; s += na_c)
    {
        /* Make lists for this (sub-)cell on atoms with and without LJ */
        n1       = 0;
        n2       = 0;
        haveQ    = FALSE;
        a_lj_max = -1;
        for (a = s; a < min(s+na_c, a1); a++)
        {
            haveQ = haveQ || GET_CGINFO_HAS_Q(atinfo[order[a]]);

            if (GET_CGINFO_HAS_VDW(atinfo[order[a]]))
            {
                sort1[n1++] = order[a];
                a_lj_max    = a;
            }
            else
            {
                sort2[n2++] = order[a];
            }
        }

        /* If we don't have atom with LJ, there's nothing to sort */
        if (n1 > 0)
        {
            *flags |= NBNXN_CI_DO_LJ(subc);

            if (2*n1 <= na_c)
            {
                /* Only sort when strictly necessary. Ordering particles
                 * Ordering particles can lead to less accurate summation
                 * due to rounding, both for LJ and Coulomb interactions.
                 */
                if (2*(a_lj_max - s) >= na_c)
                {
                    for (i = 0; i < n1; i++)
                    {
                        order[a0+i] = sort1[i];
                    }
                    for (j = 0; j < n2; j++)
                    {
                        order[a0+n1+j] = sort2[j];
                    }
                }

                *flags |= NBNXN_CI_HALF_LJ(subc);
            }
        }
        if (haveQ)
        {
            *flags |= NBNXN_CI_DO_COUL(subc);
        }
        subc++;
    }
}

/* Fill a pair search cell with atoms.
 * Potentially sorts atoms and sets the interaction flags.
 */
void fill_cell(const nbnxn_search_t nbs,
               nbnxn_grid_t *grid,
               nbnxn_atomdata_t *nbat,
               int a0, int a1,
               const int *atinfo,
               rvec *x,
               int sx, int sy, int sz,
               float *bb_work)
{ // called
    int     na, a;
    size_t  offset;
    float  *bb_ptr;

    na = a1 - a0;

    if (grid->bSimple)
    {
        sort_on_lj(nbat, grid->na_c, a0, a1, atinfo, nbs->a,
                   grid->flags+(a0>>grid->na_c_2log)-grid->cell0);
    }

    /* Now we have sorted the atoms, set the cell indices */
    for (a = a0; a < a1; a++)
    {
        nbs->cell[nbs->a[a]] = a;
    }

    copy_rvec_to_nbat_real(nbs->a+a0, a1-a0, grid->na_c, x,
                           nbat->XFormat, nbat->x, a0,
                           sx, sy, sz);

        /* Store the bounding boxes as xyz.xyz. */
        bb_ptr = grid->bb+((a0-grid->cell0*grid->na_sc)>>grid->na_c_2log)*NNBSBB_B;

        calc_bounding_box(na, nbat->xstride, nbat->x+a0*nbat->xstride,
                          bb_ptr);

}

/* Spatially sort the atoms within one grid column */
static void sort_columns_simple(const nbnxn_search_t nbs,
                                int dd_zone,
                                nbnxn_grid_t *grid,
                                int a0, int a1,
                                const int *atinfo,
                                rvec *x,
                                nbnxn_atomdata_t *nbat,
                                int cxy_start, int cxy_end,
                                int *sort_work)
{ // called
    int  cxy;
    int  cx, cy, cz, ncz, cfilled, c;
    int  na, ash, ind, a;
    int  na_c, ash_c;


    /* Sort the atoms within each x,y column in 3 dimensions */
    for (cxy = cxy_start; cxy < cxy_end; cxy++)
    {
        cx = cxy/grid->ncy;
        cy = cxy - cx*grid->ncy;

        na  = grid->cxy_na[cxy];
        ncz = grid->cxy_ind[cxy+1] - grid->cxy_ind[cxy];
        ash = (grid->cell0 + grid->cxy_ind[cxy])*grid->na_sc;

        /* Sort the atoms within each x,y column on z coordinate */
        sort_atoms(ZZ, FALSE,
                   nbs->a+ash, na, x,
                   grid->c0[ZZ],
                   ncz*grid->na_sc*SORT_GRID_OVERSIZE/nbs->box[ZZ][ZZ],
                   ncz*grid->na_sc*SGSF, sort_work);

        /* Fill the ncz cells in this column */
        cfilled = grid->cxy_ind[cxy];
        for (cz = 0; cz < ncz; cz++)
        {
            c  = grid->cxy_ind[cxy] + cz;

            ash_c = ash + cz*grid->na_sc;
            na_c  = min(grid->na_sc, na-(ash_c-ash));

            fill_cell(nbs, grid, nbat,
                      ash_c, ash_c+na_c, atinfo, x,
                      grid->na_sc*cx + (dd_zone >> 2),
                      grid->na_sc*cy + (dd_zone & 3),
                      grid->na_sc*cz,
                      NULL);

            /* This copy to bbcz is not really necessary.
             * But it allows to use the same grid search code
             * for the simple and supersub cell setups.
             */
            if (na_c > 0)
            {
                cfilled = c;
            }
            grid->bbcz[c*NNBSBB_D  ] = grid->bb[cfilled*NNBSBB_B+2];
            grid->bbcz[c*NNBSBB_D+1] = grid->bb[cfilled*NNBSBB_B+6];
        }

        /* Set the unused atom indices to -1 */
        for (ind = na; ind < ncz*grid->na_sc; ind++)
        {
            nbs->a[ash+ind] = -1;
        }
    }
}


/* Determine in which grid column atoms should go */
static void calc_column_indices(nbnxn_grid_t *grid,
                                int a0, int a1,
                                rvec *x,
                                int dd_zone, const int *move,
                                int thread, int nthread,
                                int *cell,
                                int *cxy_na)
{ //called
    int  n0, n1, i;
    int  cx, cy;

    /* We add one extra cell for particles which moved during DD */
    for (i = 0; i < grid->ncx*grid->ncy+1; i++)
    {
        cxy_na[i] = 0;
    }

    n0 = a0 + (int)((thread+0)*(a1 - a0))/nthread;
    n1 = a0 + (int)((thread+1)*(a1 - a0))/nthread;
    if (dd_zone == 0)
    {
        /* Home zone */
        for (i = n0; i < n1; i++)
        {
            if (move == NULL || move[i] >= 0)
            {
                /* We need to be careful with rounding,
                 * particles might be a few bits outside the local zone.
                 * The int cast takes care of the lower bound,
                 * we will explicitly take care of the upper bound.
                 */
                cx = (int)((x[i][XX] - grid->c0[XX])*grid->inv_sx);
                cy = (int)((x[i][YY] - grid->c0[YY])*grid->inv_sy);

#ifdef DEBUG_NBNXN_GRIDDING
                if (cx < 0 || cx >= grid->ncx ||
                    cy < 0 || cy >= grid->ncy)
                {
                    gmx_fatal(FARGS,
                              "grid cell cx %d cy %d out of range (max %d %d)\n"
                              "atom %f %f %f, grid->c0 %f %f",
                              cx, cy, grid->ncx, grid->ncy,
                              x[i][XX], x[i][YY], x[i][ZZ], grid->c0[XX], grid->c0[YY]);
                }
#endif
                /* Take care of potential rouding issues */
                cx = min(cx, grid->ncx - 1);
                cy = min(cy, grid->ncy - 1);

                /* For the moment cell will contain only the, grid local,
                 * x and y indices, not z.
                 */
                cell[i] = cx*grid->ncy + cy;
            }
            else
            {
                /* Put this moved particle after the end of the grid,
                 * so we can process it later without using conditionals.
                 */
                cell[i] = grid->ncx*grid->ncy;
            }

            cxy_na[cell[i]]++;
        }
    }
    else
    {
        /* Non-home zone */
        for (i = n0; i < n1; i++)
        {
            cx = (int)((x[i][XX] - grid->c0[XX])*grid->inv_sx);
            cy = (int)((x[i][YY] - grid->c0[YY])*grid->inv_sy);

            /* For non-home zones there could be particles outside
             * the non-bonded cut-off range, which have been communicated
             * for bonded interactions only. For the result it doesn't
             * matter where these end up on the grid. For performance
             * we put them in an extra row at the border.
             */
            cx = max(cx, 0);
            cx = min(cx, grid->ncx - 1);
            cy = max(cy, 0);
            cy = min(cy, grid->ncy - 1);

            /* For the moment cell will contain only the, grid local,
             * x and y indices, not z.
             */
            cell[i] = cx*grid->ncy + cy;

            cxy_na[cell[i]]++;
        }
    }
}

/* Determine in which grid cells the atoms should go */
static void calc_cell_indices(const nbnxn_search_t nbs,
                              int dd_zone,
                              nbnxn_grid_t *grid,
                              int a0, int a1,
                              const int *atinfo,
                              rvec *x,
                              const int *move,
                              nbnxn_atomdata_t *nbat)
{//called
    int   n0, n1, i;
    int   cx, cy, cxy, ncz_max, ncz;
    int   nthread, thread;
    int  *cxy_na, cxy_na_i;

    nthread = gmx_omp_nthreads_get(emntPairsearch);

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (thread = 0; thread < nthread; thread++)
    {
        calc_column_indices(grid, a0, a1, x, dd_zone, move, thread, nthread,
                            nbs->cell, nbs->work[thread].cxy_na);
    }

    /* Make the cell index as a function of x and y */
    ncz_max          = 0;
    ncz              = 0;
    grid->cxy_ind[0] = 0;
    for (i = 0; i < grid->ncx*grid->ncy+1; i++)
    {
        /* We set ncz_max at the beginning of the loop iso at the end
         * to skip i=grid->ncx*grid->ncy which are moved particles
         * that do not need to be ordered on the grid.
         */
        if (ncz > ncz_max)
        {
            ncz_max = ncz;
        }
        cxy_na_i = nbs->work[0].cxy_na[i];
        for (thread = 1; thread < nthread; thread++)
        {
            cxy_na_i += nbs->work[thread].cxy_na[i];
        }
        ncz = (cxy_na_i + grid->na_sc - 1)/grid->na_sc;
        if (nbat->XFormat == nbatX8)
        {
            /* Make the number of cell a multiple of 2 */
            ncz = (ncz + 1) & ~1;
        }
        grid->cxy_ind[i+1] = grid->cxy_ind[i] + ncz;
        /* Clear cxy_na, so we can reuse the array below */
        grid->cxy_na[i] = 0;
    }
    grid->nc = grid->cxy_ind[grid->ncx*grid->ncy] - grid->cxy_ind[0];

    nbat->natoms = (grid->cell0 + grid->nc)*grid->na_sc;

    /* Make sure the work array for sorting is large enough */
    if (ncz_max*grid->na_sc*SGSF > nbs->work[0].sort_work_nalloc)
    {
        for (thread = 0; thread < nbs->nthread_max; thread++)
        {
            nbs->work[thread].sort_work_nalloc =
                over_alloc_large(ncz_max*grid->na_sc*SGSF);
            srenew(nbs->work[thread].sort_work,
                   nbs->work[thread].sort_work_nalloc);
            /* When not in use, all elements should be -1 */
            for (i = 0; i < nbs->work[thread].sort_work_nalloc; i++)
            {
                nbs->work[thread].sort_work[i] = -1;
            }
        }
    }

    /* Now we know the dimensions we can fill the grid.
     * This is the first, unsorted fill. We sort the columns after this.
     */
    for (i = a0; i < a1; i++)
    {
        /* At this point nbs->cell contains the local grid x,y indices */
        cxy = nbs->cell[i];
        nbs->a[(grid->cell0 + grid->cxy_ind[cxy])*grid->na_sc + grid->cxy_na[cxy]++] = i;
    }

    if (dd_zone == 0)
    {
        /* Set the cell indices for the moved particles */
        n0 = grid->nc*grid->na_sc;
        n1 = grid->nc*grid->na_sc+grid->cxy_na[grid->ncx*grid->ncy];
        if (dd_zone == 0)
        {
            for (i = n0; i < n1; i++)
            {
                nbs->cell[nbs->a[i]] = i;
            }
        }
    }

    /* Sort the super-cell columns along z into the sub-cells. */
#pragma omp parallel for num_threads(nbs->nthread_max) schedule(static)
    for (thread = 0; thread < nbs->nthread_max; thread++)
    {
            sort_columns_simple(nbs, dd_zone, grid, a0, a1, atinfo, x, nbat,
                                ((thread+0)*grid->ncx*grid->ncy)/nthread,
                                ((thread+1)*grid->ncx*grid->ncy)/nthread,
                                nbs->work[thread].sort_work);
    }



}

static void init_buffer_flags(nbnxn_buffer_flags_t *flags,
                              int                   natoms)
{//called
    int b;

    flags->nflag = (natoms + NBNXN_BUFFERFLAG_SIZE - 1)/NBNXN_BUFFERFLAG_SIZE;
    if (flags->nflag > flags->flag_nalloc)
    {
        flags->flag_nalloc = over_alloc_large(flags->nflag);
        srenew(flags->flag, flags->flag_nalloc);
    }
    for (b = 0; b < flags->nflag; b++)
    {
        flags->flag[b] = 0;
    }
}

/* Sets up a grid and puts the atoms on the grid.
 * This function only operates on one domain of the domain decompostion.
 * Note that without domain decomposition there is only one domain.
 */
void nbnxn_put_on_grid(nbnxn_search_t nbs,
                       int ePBC, matrix box,
                       int dd_zone,
                       rvec corner0, rvec corner1,
                       int a0, int a1,
                       real atom_density,
                       const int *atinfo,
                       rvec *x,
                       int nmoved, int *move,
                       int nb_kernel_type,
                       nbnxn_atomdata_t *nbat)
{//called
    nbnxn_grid_t *grid;
    int           n;
    int           nc_max_grid, nc_max;

    grid = &nbs->grid[dd_zone];

    nbs_cycle_start(&nbs->cc[enbsCCgrid]);

    grid->bSimple = nbnxn_kernel_pairlist_simple(nb_kernel_type);

    grid->na_c      = nbnxn_kernel_to_ci_size(nb_kernel_type);
    grid->na_cj     = nbnxn_kernel_to_cj_size(nb_kernel_type);
    grid->na_sc     = (grid->bSimple ? 1 : GPU_NSUBCELL)*grid->na_c;
    grid->na_c_2log = get_2log(grid->na_c);

    nbat->na_c = grid->na_c;

    if (dd_zone == 0)
    {
        grid->cell0 = 0;
    }
    else
    {
        grid->cell0 =
            (nbs->grid[dd_zone-1].cell0 + nbs->grid[dd_zone-1].nc)*
            nbs->grid[dd_zone-1].na_sc/grid->na_sc;
    }

    n = a1 - a0;

    if (dd_zone == 0)
    {
        nbs->ePBC = ePBC;
        copy_mat(box, nbs->box);

        if (atom_density >= 0)
        {
            grid->atom_density = atom_density;
        }
        else
        {
            grid->atom_density = grid_atom_density(n-nmoved, corner0, corner1);
        }

        grid->cell0 = 0;

        nbs->natoms_local    = a1 - nmoved;
        /* We assume that nbnxn_put_on_grid is called first
         * for the local atoms (dd_zone=0).
         */
        nbs->natoms_nonlocal = a1 - nmoved;
    }
    else
    {
        nbs->natoms_nonlocal = max(nbs->natoms_nonlocal, a1);
    }

    nc_max_grid = set_grid_size_xy(nbs, grid,
                                   dd_zone, n-nmoved, corner0, corner1,
                                   nbs->grid[0].atom_density,
                                   nbat->XFormat);

    nc_max = grid->cell0 + nc_max_grid;

    if (a1 > nbs->cell_nalloc)
    {
        nbs->cell_nalloc = over_alloc_large(a1);
        srenew(nbs->cell, nbs->cell_nalloc);
    }

    /* To avoid conditionals we store the moved particles at the end of a,
     * make sure we have enough space.
     */
    if (nc_max*grid->na_sc + nmoved > nbs->a_nalloc)
    {
        nbs->a_nalloc = over_alloc_large(nc_max*grid->na_sc + nmoved);
        srenew(nbs->a, nbs->a_nalloc);
    }

    /* We need padding up to a multiple of the buffer flag size: simply add */
    if (nc_max*grid->na_sc + NBNXN_BUFFERFLAG_SIZE > nbat->nalloc)
    {
        nbnxn_atomdata_realloc(nbat, nc_max*grid->na_sc+NBNXN_BUFFERFLAG_SIZE);
    }

    calc_cell_indices(nbs, dd_zone, grid, a0, a1, atinfo, x, move, nbat);

    if (dd_zone == 0)
    {
        nbat->natoms_local = nbat->natoms;
    }

    nbs_cycle_stop(&nbs->cc[enbsCCgrid]);
}


/* Determines the cell range along one dimension that
 * the bounding box b0 - b1 sees.
 */
static void get_cell_range(real b0, real b1,
                           int nc, real c0, real s, real invs,
                           real d2, real r2, int *cf, int *cl)
{ // called
    *cf = max((int)((b0 - c0)*invs), 0);

    while (*cf > 0 && d2 + sqr((b0 - c0) - (*cf-1+1)*s) < r2)
    {
        (*cf)--;
    }

    *cl = min((int)((b1 - c0)*invs), nc-1);
    while (*cl < nc-1 && d2 + sqr((*cl+1)*s - (b1 - c0)) < r2)
    {
        (*cl)++;
    }
}

/* Plain C code calculating the distance^2 between two bounding boxes */
static float subc_bb_dist2(int si, const float *bb_i_ci,
                           int csj, const float *bb_j_all)
{ // called
    const float *bb_i, *bb_j;
    float        d2;
    float        dl, dh, dm, dm0;

    bb_i = bb_i_ci  +  si*NNBSBB_B;
    bb_j = bb_j_all + csj*NNBSBB_B;

    d2 = 0;

    dl  = bb_i[BBL_X] - bb_j[BBU_X];
    dh  = bb_j[BBL_X] - bb_i[BBU_X];
    dm  = max(dl, dh);
    dm0 = max(dm, 0);
    d2 += dm0*dm0;

    dl  = bb_i[BBL_Y] - bb_j[BBU_Y];
    dh  = bb_j[BBL_Y] - bb_i[BBU_Y];
    dm  = max(dl, dh);
    dm0 = max(dm, 0);
    d2 += dm0*dm0;

    dl  = bb_i[BBL_Z] - bb_j[BBU_Z];
    dh  = bb_j[BBL_Z] - bb_i[BBU_Z];
    dm  = max(dl, dh);
    dm0 = max(dm, 0);
    d2 += dm0*dm0;

    return d2;
}



/* Ensures there is enough space for ncell extra j-cells in the list */
static void check_subcell_list_space_simple(nbnxn_pairlist_t *nbl,
                                            int               ncell)
{// called
    int cj_max;

    cj_max = nbl->ncj + ncell;

    if (cj_max > nbl->cj_nalloc)
    {
        nbl->cj_nalloc = over_alloc_small(cj_max);
        nbnxn_realloc_void((void **)&nbl->cj,
                           nbl->ncj*sizeof(*nbl->cj),
                           nbl->cj_nalloc*sizeof(*nbl->cj),
                           nbl->alloc, nbl->free);
    }
}


/* Initializes a single nbnxn_pairlist_t data structure */
static void nbnxn_init_pairlist(nbnxn_pairlist_t *nbl,
                                gmx_bool          bSimple,
                                nbnxn_alloc_t    *alloc,
                                nbnxn_free_t     *free)
{ // called
    if (alloc == NULL)
    {
        nbl->alloc = nbnxn_alloc_aligned;
    }
    else
    {
        nbl->alloc = alloc;
    }
    if (free == NULL)
    {
        nbl->free = nbnxn_free_aligned;
    }
    else
    {
        nbl->free = free;
    }

    nbl->bSimple     = bSimple;
    nbl->na_sc       = 0;
    nbl->na_ci       = 0;
    nbl->na_cj       = 0;
    nbl->nci         = 0;
    nbl->ci          = NULL;
    nbl->ci_nalloc   = 0;
    nbl->ncj         = 0;
    nbl->cj          = NULL;
    nbl->cj_nalloc   = 0;
    nbl->ncj4        = 0;
    /* We need one element extra in sj, so alloc initially with 1 */
    nbl->cj4_nalloc  = 0;
    nbl->cj4         = NULL;
    nbl->nci_tot     = 0;


    snew(nbl->work, 1);
#ifdef NBNXN_BBXXXX
    snew_aligned(nbl->work->bb_ci, GPU_NSUBCELL/STRIDE_PBB*NNBSBB_XXXX, NBNXN_MEM_ALIGN);
#else
    snew_aligned(nbl->work->bb_ci, GPU_NSUBCELL*NNBSBB_B, NBNXN_MEM_ALIGN);
#endif
    snew_aligned(nbl->work->x_ci, NBNXN_NA_SC_MAX*DIM, NBNXN_MEM_ALIGN);
#ifdef GMX_NBNXN_SIMD
    snew_aligned(nbl->work->x_ci_simd_4xn, 1, NBNXN_MEM_ALIGN);
    snew_aligned(nbl->work->x_ci_simd_2xnn, 1, NBNXN_MEM_ALIGN);
#endif
    snew_aligned(nbl->work->d2, GPU_NSUBCELL, NBNXN_MEM_ALIGN);
}

void nbnxn_init_pairlist_set(nbnxn_pairlist_set_t *nbl_list,
                             gmx_bool bSimple, gmx_bool bCombined,
                             nbnxn_alloc_t *alloc,
                             nbnxn_free_t  *free)
{ // called
    int i;

    nbl_list->bSimple   = bSimple;
    nbl_list->bCombined = bCombined;

    nbl_list->nnbl = gmx_omp_nthreads_get(emntNonbonded);

    if (!nbl_list->bCombined &&
        nbl_list->nnbl > NBNXN_BUFFERFLAG_MAX_THREADS)
    {
        gmx_fatal(FARGS, "%d OpenMP threads were requested. Since the non-bonded force buffer reduction is prohibitively slow with more than %d threads, we do not allow this. Use %d or less OpenMP threads.",
                  nbl_list->nnbl, NBNXN_BUFFERFLAG_MAX_THREADS, NBNXN_BUFFERFLAG_MAX_THREADS);
    }

    snew(nbl_list->nbl, nbl_list->nnbl);
    /* Execute in order to avoid memory interleaving between threads */
#pragma omp parallel for num_threads(nbl_list->nnbl) schedule(static)
    for (i = 0; i < nbl_list->nnbl; i++)
    {
        /* Allocate the nblist data structure locally on each thread
         * to optimize memory access for NUMA architectures.
         */
        snew(nbl_list->nbl[i], 1);

        /* Only list 0 is used on the GPU, use normal allocation for i>0 */
        if (i == 0)
        {
            nbnxn_init_pairlist(nbl_list->nbl[i], nbl_list->bSimple, alloc, free);
        }
        else
        {
            nbnxn_init_pairlist(nbl_list->nbl[i], nbl_list->bSimple, NULL, NULL);
        }
    }
}
/* Returns a diagonal or off-diagonal interaction mask for plain C lists */
static unsigned int get_imask(gmx_bool rdiag, int ci, int cj)
{ // called
    return (rdiag && ci == cj ? NBNXN_INT_MASK_DIAG : NBNXN_INT_MASK_ALL);
}


/* Plain C code for making a pair list of cell ci vs cell cjf-cjl.
 * Checks bounding box distances and possibly atom pair distances.
 */
static void make_cluster_list_simple(const nbnxn_grid_t *gridj,
                                     nbnxn_pairlist_t *nbl,
                                     int ci, int cjf, int cjl,
                                     gmx_bool remove_sub_diag,
                                     const real *x_j,
                                     real rl2, float rbb2,
                                     int *ndistc)
{ // called
    const nbnxn_list_work_t *work;

    const float             *bb_ci;
    const real              *x_ci;

    gmx_bool                 InRange;
    real                     d2;
    int                      cjf_gl, cjl_gl, cj;

    work = nbl->work;

    bb_ci = nbl->work->bb_ci;
    x_ci  = nbl->work->x_ci;

    InRange = FALSE;
    while (!InRange && cjf <= cjl)
    {
        d2       = subc_bb_dist2(0, bb_ci, cjf, gridj->bb);
        *ndistc += 2;

        /* Check if the distance is within the distance where
         * we use only the bounding box distance rbb,
         * or within the cut-off and there is at least one atom pair
         * within the cut-off.
         */
        if (d2 < rbb2)
        {
            InRange = TRUE;
        }
        else if (d2 < rl2)
        {
            int i, j;

            cjf_gl = gridj->cell0 + cjf;
            for (i = 0; i < NBNXN_CPU_CLUSTER_I_SIZE && !InRange; i++)
            {
                for (j = 0; j < NBNXN_CPU_CLUSTER_I_SIZE; j++)
                {
                    InRange = InRange ||
                        (sqr(x_ci[i*STRIDE_XYZ+XX] - x_j[(cjf_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+XX]) +
                         sqr(x_ci[i*STRIDE_XYZ+YY] - x_j[(cjf_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+YY]) +
                         sqr(x_ci[i*STRIDE_XYZ+ZZ] - x_j[(cjf_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+ZZ]) < rl2);
                }
            }
            *ndistc += NBNXN_CPU_CLUSTER_I_SIZE*NBNXN_CPU_CLUSTER_I_SIZE;
        }
        if (!InRange)
        {
            cjf++;
        }
    }
    if (!InRange)
    {
        return;
    }

    InRange = FALSE;
    while (!InRange && cjl > cjf)
    {
        d2       = subc_bb_dist2(0, bb_ci, cjl, gridj->bb);
        *ndistc += 2;

        /* Check if the distance is within the distance where
         * we use only the bounding box distance rbb,
         * or within the cut-off and there is at least one atom pair
         * within the cut-off.
         */
        if (d2 < rbb2)
        {
            InRange = TRUE;
        }
        else if (d2 < rl2)
        {
            int i, j;

            cjl_gl = gridj->cell0 + cjl;
            for (i = 0; i < NBNXN_CPU_CLUSTER_I_SIZE && !InRange; i++)
            {
                for (j = 0; j < NBNXN_CPU_CLUSTER_I_SIZE; j++)
                {
                    InRange = InRange ||
                        (sqr(x_ci[i*STRIDE_XYZ+XX] - x_j[(cjl_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+XX]) +
                         sqr(x_ci[i*STRIDE_XYZ+YY] - x_j[(cjl_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+YY]) +
                         sqr(x_ci[i*STRIDE_XYZ+ZZ] - x_j[(cjl_gl*NBNXN_CPU_CLUSTER_I_SIZE+j)*STRIDE_XYZ+ZZ]) < rl2);
                }
            }
            *ndistc += NBNXN_CPU_CLUSTER_I_SIZE*NBNXN_CPU_CLUSTER_I_SIZE;
        }
        if (!InRange)
        {
            cjl--;
        }
    }

    if (cjf <= cjl)
    {
        for (cj = cjf; cj <= cjl; cj++)
        {
            /* Store cj and the interaction mask */
            nbl->cj[nbl->ncj].cj   = gridj->cell0 + cj;
            nbl->cj[nbl->ncj].excl = get_imask(remove_sub_diag, ci, cj);
            nbl->ncj++;
        }
        /* Increase the closing index in i super-cell list */
        nbl->ci[nbl->nci].cj_ind_end = nbl->ncj;
    }
}

#ifdef GMX_NBNXN_SIMD_4XN
#include "nbnxn_search_simd_4xn.h"
#endif
#ifdef GMX_NBNXN_SIMD_2XNN
#include "nbnxn_search_simd_2xnn.h"
#endif


/* Set all atom-pair exclusions from the topology stored in excl
 * as masks in the pair-list for simple list i-entry nbl_ci
 */
static void set_ci_top_excls(const nbnxn_search_t nbs,
                             nbnxn_pairlist_t    *nbl,
                             gmx_bool             diagRemoved,
                             int                  na_ci_2log,
                             int                  na_cj_2log,
                             const nbnxn_ci_t    *nbl_ci,
                             const t_blocka      *excl)
{ //called
    const int    *cell;
    int           ci;
    int           cj_ind_first, cj_ind_last;
    int           cj_first, cj_last;
    int           ndirect;
    int           i, ai, aj, si, eind, ge, se;
    int           found, cj_ind_0, cj_ind_1, cj_ind_m;
    int           cj_m;
    gmx_bool      Found_si;
    int           si_ind;
    nbnxn_excl_t *nbl_excl;
    int           inner_i, inner_e;

    cell = nbs->cell;

    if (nbl_ci->cj_ind_end == nbl_ci->cj_ind_start)
    {
        /* Empty list */
        return;
    }

    ci = nbl_ci->ci;

    cj_ind_first = nbl_ci->cj_ind_start;
    cj_ind_last  = nbl->ncj - 1;

    cj_first = nbl->cj[cj_ind_first].cj;
    cj_last  = nbl->cj[cj_ind_last].cj;

    /* Determine how many contiguous j-cells we have starting
     * from the first i-cell. This number can be used to directly
     * calculate j-cell indices for excluded atoms.
     */
    ndirect = 0;
    if (na_ci_2log == na_cj_2log)
    {
        while (cj_ind_first + ndirect <= cj_ind_last &&
               nbl->cj[cj_ind_first+ndirect].cj == ci + ndirect)
        {
            ndirect++;
        }
    }

    /* Loop over the atoms in the i super-cell */
    for (i = 0; i < nbl->na_sc; i++)
    {
        ai = nbs->a[ci*nbl->na_sc+i];
        if (ai >= 0)
        {
            si  = (i>>na_ci_2log);

            /* Loop over the topology-based exclusions for this i-atom */
            for (eind = excl->index[ai]; eind < excl->index[ai+1]; eind++)
            {
                aj = excl->a[eind];

                if (aj == ai)
                {
                    /* The self exclusion are already set, save some time */
                    continue;
                }

                ge = cell[aj];

                /* Without shifts we only calculate interactions j>i
                 * for one-way pair-lists.
                 */
                if (diagRemoved && ge <= ci*nbl->na_sc + i)
                {
                    continue;
                }

                se = (ge >> na_cj_2log);

                /* Could the cluster se be in our list? */
                if (se >= cj_first && se <= cj_last)
                {
                    if (se < cj_first + ndirect)
                    {
                        /* We can calculate cj_ind directly from se */
                        found = cj_ind_first + se - cj_first;
                    }
                    else
                    {
                        /* Search for se using bisection */
                        found    = -1;
                        cj_ind_0 = cj_ind_first + ndirect;
                        cj_ind_1 = cj_ind_last + 1;
                        while (found == -1 && cj_ind_0 < cj_ind_1)
                        {
                            cj_ind_m = (cj_ind_0 + cj_ind_1)>>1;

                            cj_m = nbl->cj[cj_ind_m].cj;

                            if (se == cj_m)
                            {
                                found = cj_ind_m;
                            }
                            else if (se < cj_m)
                            {
                                cj_ind_1 = cj_ind_m;
                            }
                            else
                            {
                                cj_ind_0 = cj_ind_m + 1;
                            }
                        }
                    }

                    if (found >= 0)
                    {
                        inner_i = i  - (si << na_ci_2log);
                        inner_e = ge - (se << na_cj_2log);

                        nbl->cj[found].excl &= ~(1U<<((inner_i<<na_cj_2log) + inner_e));
                    }
                }
            }
        }
    }
}


/* Reallocate the simple ci list for at least n entries */
static void nb_realloc_ci(nbnxn_pairlist_t *nbl, int n)
{ // called
    nbl->ci_nalloc = over_alloc_small(n);
    nbnxn_realloc_void((void **)&nbl->ci,
                       nbl->nci*sizeof(*nbl->ci),
                       nbl->ci_nalloc*sizeof(*nbl->ci),
                       nbl->alloc, nbl->free);
}


/* Make a new ci entry at index nbl->nci */
static void new_ci_entry(nbnxn_pairlist_t *nbl, int ci, int shift, int flags,
                         nbnxn_list_work_t *work)
{ // called
    if (nbl->nci + 1 > nbl->ci_nalloc)
    {
        nb_realloc_ci(nbl, nbl->nci+1);
    }
    nbl->ci[nbl->nci].ci            = ci;
    nbl->ci[nbl->nci].shift         = shift;
    /* Store the interaction flags along with the shift */
    nbl->ci[nbl->nci].shift        |= flags;
    nbl->ci[nbl->nci].cj_ind_start  = nbl->ncj;
    nbl->ci[nbl->nci].cj_ind_end    = nbl->ncj;
}


/* Sort the simple j-list cj on exclusions.
 * Entries with exclusions will all be sorted to the beginning of the list.
 */
static void sort_cj_excl(nbnxn_cj_t *cj, int ncj,
                         nbnxn_list_work_t *work)
{ // called
    int jnew, j;

    if (ncj > work->cj_nalloc)
    {
        work->cj_nalloc = over_alloc_large(ncj);
        srenew(work->cj, work->cj_nalloc);
    }

    /* Make a list of the j-cells involving exclusions */
    jnew = 0;
    for (j = 0; j < ncj; j++)
    {
        if (cj[j].excl != NBNXN_INT_MASK_ALL)
        {
            work->cj[jnew++] = cj[j];
        }
    }
    /* Check if there are exclusions at all or not just the first entry */
    if (!((jnew == 0) ||
          (jnew == 1 && cj[0].excl != NBNXN_INT_MASK_ALL)))
    {
        for (j = 0; j < ncj; j++)
        {
            if (cj[j].excl == NBNXN_INT_MASK_ALL)
            {
                work->cj[jnew++] = cj[j];
            }
        }
        for (j = 0; j < ncj; j++)
        {
            cj[j] = work->cj[j];
        }
    }
}

/* Close this simple list i entry */
static void close_ci_entry_simple(nbnxn_pairlist_t *nbl)
{ // called
    int jlen;

    /* All content of the new ci entry have already been filled correctly,
     * we only need to increase the count here (for non empty lists).
     */
    jlen = nbl->ci[nbl->nci].cj_ind_end - nbl->ci[nbl->nci].cj_ind_start;
    if (jlen > 0)
    {
        sort_cj_excl(nbl->cj+nbl->ci[nbl->nci].cj_ind_start, jlen, nbl->work);

        /* The counts below are used for non-bonded pair/flop counts
         * and should therefore match the available kernel setups.
         */
        if (!(nbl->ci[nbl->nci].shift & NBNXN_CI_DO_COUL(0)))
        {
            nbl->work->ncj_noq += jlen;
        }
        else if ((nbl->ci[nbl->nci].shift & NBNXN_CI_HALF_LJ(0)) ||
                 !(nbl->ci[nbl->nci].shift & NBNXN_CI_DO_LJ(0)))
        {
            nbl->work->ncj_hlj += jlen;
        }

        nbl->nci++;
    }
}


/* Clears an nbnxn_pairlist_t data structure */
static void clear_pairlist(nbnxn_pairlist_t *nbl)
{ //called
    nbl->nci           = 0;
    nbl->nsci          = 0;
    nbl->ncj           = 0;
    nbl->ncj4          = 0;
    nbl->nci_tot       = 0;
    nbl->nexcl         = 1;

    nbl->work->ncj_noq = 0;
    nbl->work->ncj_hlj = 0;
}

/* Sets a simple list i-cell bounding box, including PBC shift */
static void set_icell_bb_simple(const float *bb, int ci,
                                real shx, real shy, real shz,
                                float *bb_ci)
{// called
    int ia;

    ia           = ci*NNBSBB_B;
    bb_ci[BBL_X] = bb[ia+BBL_X] + shx;
    bb_ci[BBL_Y] = bb[ia+BBL_Y] + shy;
    bb_ci[BBL_Z] = bb[ia+BBL_Z] + shz;
    bb_ci[BBU_X] = bb[ia+BBU_X] + shx;
    bb_ci[BBU_Y] = bb[ia+BBU_Y] + shy;
    bb_ci[BBU_Z] = bb[ia+BBU_Z] + shz;
}


/* Copies PBC shifted i-cell atom coordinates x,y,z to working array */
static void icell_set_x_simple(int ci,
                               real shx, real shy, real shz,
                               int na_c,
                               int stride, const real *x,
                               nbnxn_list_work_t *work)
{//called
    int  ia, i;

    ia = ci*NBNXN_CPU_CLUSTER_I_SIZE;

    for (i = 0; i < NBNXN_CPU_CLUSTER_I_SIZE; i++)
    {
        work->x_ci[i*STRIDE_XYZ+XX] = x[(ia+i)*stride+XX] + shx;
        work->x_ci[i*STRIDE_XYZ+YY] = x[(ia+i)*stride+YY] + shy;
        work->x_ci[i*STRIDE_XYZ+ZZ] = x[(ia+i)*stride+ZZ] + shz;
    }
}

/* Returns the next ci to be processes by our thread */
static gmx_bool next_ci(const nbnxn_grid_t *grid,
                        int conv,
                        int nth, int ci_block,
                        int *ci_x, int *ci_y,
                        int *ci_b, int *ci)
{ // called1
    (*ci_b)++;
    (*ci)++;

    if (*ci_b == ci_block)
    {
        /* Jump to the next block assigned to this task */
        *ci   += (nth - 1)*ci_block;
        *ci_b  = 0;
    }

    if (*ci >= grid->nc*conv)
    {
        return FALSE;
    }

    while (*ci >= grid->cxy_ind[*ci_x*grid->ncy + *ci_y + 1]*conv)
    {
        *ci_y += 1;
        if (*ci_y == grid->ncy)
        {
            *ci_x += 1;
            *ci_y  = 0;
        }
    }

    return TRUE;
}

/* Returns the distance^2 for which we put cell pairs in the list
 * without checking atom pair distances. This is usually < rlist^2.
 */
static float boundingbox_only_distance2(const nbnxn_grid_t *gridi,
                                        const nbnxn_grid_t *gridj,
                                        real                rlist,
                                        gmx_bool            simple)
{ // called
    /* If the distance between two sub-cell bounding boxes is less
     * than this distance, do not check the distance between
     * all particle pairs in the sub-cell, since then it is likely
     * that the box pair has atom pairs within the cut-off.
     * We use the nblist cut-off minus 0.5 times the average x/y diagonal
     * spacing of the sub-cells. Around 40% of the checked pairs are pruned.
     * Using more than 0.5 gains at most 0.5%.
     * If forces are calculated more than twice, the performance gain
     * in the force calculation outweighs the cost of checking.
     * Note that with subcell lists, the atom-pair distance check
     * is only performed when only 1 out of 8 sub-cells in within range,
     * this is because the GPU is much faster than the cpu.
     */
    real bbx, bby;
    real rbb2;

    bbx = 0.5*(gridi->sx + gridj->sx);
    bby = 0.5*(gridi->sy + gridj->sy);
    if (!simple)
    {
        bbx /= GPU_NSUBCELL_X;
        bby /= GPU_NSUBCELL_Y;
    }

    rbb2 = sqr(max(0, rlist - 0.5*sqrt(bbx*bbx + bby*bby)));

#ifndef GMX_DOUBLE
    return rbb2;
#else
    return (float)((1+GMX_FLOAT_EPS)*rbb2);
#endif
}

static int get_ci_block_size(const nbnxn_grid_t *gridi,
                             gmx_bool bDomDec, int nth)
{// called
    const int ci_block_enum      = 5;
    const int ci_block_denom     = 11;
    const int ci_block_min_atoms = 16;
    int ci_block;

    /* Here we decide how to distribute the blocks over the threads.
     * We use prime numbers to try to avoid that the grid size becomes
     * a multiple of the number of threads, which would lead to some
     * threads getting "inner" pairs and others getting boundary pairs,
     * which in turns will lead to load imbalance between threads.
     * Set the block size as 5/11/ntask times the average number of cells
     * in a y,z slab. This should ensure a quite uniform distribution
     * of the grid parts of the different thread along all three grid
     * zone boundaries with 3D domain decomposition. At the same time
     * the blocks will not become too small.
     */
    ci_block = (gridi->nc*ci_block_enum)/(ci_block_denom*gridi->ncx*nth);

    /* Ensure the blocks are not too small: avoids cache invalidation */
    if (ci_block*gridi->na_sc < ci_block_min_atoms)
    {
        ci_block = (ci_block_min_atoms + gridi->na_sc - 1)/gridi->na_sc;
    }

    /* Without domain decomposition
     * or with less than 3 blocks per task, divide in nth blocks.
     */
    if (!bDomDec || ci_block*3*nth > gridi->nc)
    {
        ci_block = (gridi->nc + nth - 1)/nth;
    }

    return ci_block;
}

/* Generates the part of pair-list nbl assigned to our thread */
static void nbnxn_make_pairlist_part(const nbnxn_search_t nbs,
                                     const nbnxn_grid_t *gridi,
                                     const nbnxn_grid_t *gridj,
                                     nbnxn_search_work_t *work,
                                     const nbnxn_atomdata_t *nbat,
                                     const t_blocka *excl,
                                     real rlist,
                                     int nb_kernel_type,
                                     int ci_block,
                                     gmx_bool bFBufferFlag,
                                     int nsubpair_max,
                                     gmx_bool progBal,
                                     int min_ci_balanced,
                                     int th, int nth,
                                     nbnxn_pairlist_t *nbl)
{// called
    int  na_cj_2log;
    matrix box;
    real rl2;
    float rbb2;
    int  d;
    int  ci_b, ci, ci_x, ci_y, ci_xy, cj;
    ivec shp;
    int  tx, ty, tz;
    int  shift;
    gmx_bool bMakeList;
    real shx, shy, shz;
    int  conv_i, cell0_i;
    const float *bb_i, *bbcz_i, *bbcz_j;
    const int *flags_i;
    real bx0, bx1, by0, by1, bz0, bz1;
    real bz1_frac;
    real d2cx, d2z, d2z_cx, d2z_cy, d2zx, d2zxy, d2xy;
    int  cxf, cxl, cyf, cyf_x, cyl;
    int  cx, cy;
    int  c0, c1, cs, cf, cl;
    int  ndistc;
    int  ncpcheck;
    int  gridi_flag_shift = 0, gridj_flag_shift = 0;
    unsigned *gridj_flag  = NULL;
    int  ncj_old_i, ncj_old_j;

    nbs_cycle_start(&work->cc[enbsCCsearch]);

    if (! gridj->bSimple)
    {
        gmx_incons("Grid incompatible with pair-list");
    }

    nbl->na_sc = gridj->na_sc;
    nbl->na_ci = gridj->na_c;
    nbl->na_cj = nbnxn_kernel_to_cj_size(nb_kernel_type);
    na_cj_2log = get_2log(nbl->na_cj);

    nbl->rlist  = rlist;

    if (bFBufferFlag)
    {
        /* Determine conversion of clusters to flag blocks */
        gridi_flag_shift = 0;
        while ((nbl->na_ci<<gridi_flag_shift) < NBNXN_BUFFERFLAG_SIZE)
        {
            gridi_flag_shift++;
        }
        gridj_flag_shift = 0;
        while ((nbl->na_cj<<gridj_flag_shift) < NBNXN_BUFFERFLAG_SIZE)
        {
            gridj_flag_shift++;
        }

        gridj_flag = work->buffer_flags.flag;
    }

    copy_mat(nbs->box, box);

    rl2 = nbl->rlist*nbl->rlist;

    rbb2 = boundingbox_only_distance2(gridi, gridj, nbl->rlist, nbl->bSimple);


    /* Set the shift range */
    for (d = 0; d < DIM; d++)
    {
        /* Check if we need periodicity shifts.
         * Without PBC or with domain decomposition we don't need them.
         */
        if (d >= ePBC2npbcdim(nbs->ePBC) || nbs->dd_dim[d])
        {
            shp[d] = 0;
        }
        else
        {
            if (d == XX &&
                box[XX][XX] - fabs(box[YY][XX]) - fabs(box[ZZ][XX]) < sqrt(rl2))
            {
                shp[d] = 2;
            }
            else
            {
                shp[d] = 1;
            }
        }
    }

    if (!gridi->bSimple)
    {
        conv_i  = gridi->na_sc/gridj->na_sc;
        bb_i    = gridi->bb_simple;
        bbcz_i  = gridi->bbcz_simple;
        flags_i = gridi->flags_simple;
    }
    else
    {
        conv_i  = 1;
        bb_i    = gridi->bb;
        bbcz_i  = gridi->bbcz;
        flags_i = gridi->flags;
    }
    cell0_i = gridi->cell0*conv_i;

    bbcz_j = gridj->bbcz;

    if (conv_i != 1)
    {
        /* Blocks of the conversion factor - 1 give a large repeat count
         * combined with a small block size. This should result in good
         * load balancing for both small and large domains.
         */
        ci_block = conv_i - 1;
    }

    ndistc   = 0;
    ncpcheck = 0;

    /* Initially ci_b and ci to 1 before where we want them to start,
     * as they will both be incremented in next_ci.
     */
    ci_b = -1;
    ci   = th*ci_block - 1;
    ci_x = 0;
    ci_y = 0;
    while (next_ci(gridi, conv_i, nth, ci_block, &ci_x, &ci_y, &ci_b, &ci))
    {
        if (flags_i[ci] == 0)
        {
            continue;
        }

        ncj_old_i = nbl->ncj;

        d2cx = 0;
        if (gridj != gridi && shp[XX] == 0)
        {
                bx1 = bb_i[ci*NNBSBB_B+NNBSBB_C+XX];
            if (bx1 < gridj->c0[XX])
            {
                d2cx = sqr(gridj->c0[XX] - bx1);

                if (d2cx >= rl2)
                {
                    continue;
                }
            }
        }

        ci_xy = ci_x*gridi->ncy + ci_y;

        /* Loop over shift vectors in three dimensions */
        for (tz = -shp[ZZ]; tz <= shp[ZZ]; tz++)
        {
            shz = tz*box[ZZ][ZZ];

            bz0 = bbcz_i[ci*NNBSBB_D  ] + shz;
            bz1 = bbcz_i[ci*NNBSBB_D+1] + shz;

            if (tz == 0)
            {
                d2z = 0;
            }
            else if (tz < 0)
            {
                d2z = sqr(bz1);
            }
            else
            {
                d2z = sqr(bz0 - box[ZZ][ZZ]);
            }

            d2z_cx = d2z + d2cx;

            if (d2z_cx >= rl2)
            {
                continue;
            }

            bz1_frac =
                bz1/((real)(gridi->cxy_ind[ci_xy+1] - gridi->cxy_ind[ci_xy]));
            if (bz1_frac < 0)
            {
                bz1_frac = 0;
            }
            /* The check with bz1_frac close to or larger than 1 comes later */

            for (ty = -shp[YY]; ty <= shp[YY]; ty++)
            {
                shy = ty*box[YY][YY] + tz*box[ZZ][YY];

                    by0 = bb_i[ci*NNBSBB_B         +YY] + shy;
                    by1 = bb_i[ci*NNBSBB_B+NNBSBB_C+YY] + shy;

                get_cell_range(by0, by1,
                               gridj->ncy, gridj->c0[YY], gridj->sy, gridj->inv_sy,
                               d2z_cx, rl2,
                               &cyf, &cyl);

                if (cyf > cyl)
                {
                    continue;
                }

                d2z_cy = d2z;
                if (by1 < gridj->c0[YY])
                {
                    d2z_cy += sqr(gridj->c0[YY] - by1);
                }
                else if (by0 > gridj->c1[YY])
                {
                    d2z_cy += sqr(by0 - gridj->c1[YY]);
                }

                for (tx = -shp[XX]; tx <= shp[XX]; tx++)
                {
                    shift = XYZ2IS(tx, ty, tz);

#ifdef NBNXN_SHIFT_BACKWARD
                    if (gridi == gridj && shift > CENTRAL)
                    {
                        continue;
                    }
#endif

                    shx = tx*box[XX][XX] + ty*box[YY][XX] + tz*box[ZZ][XX];

                        bx0 = bb_i[ci*NNBSBB_B         +XX] + shx;
                        bx1 = bb_i[ci*NNBSBB_B+NNBSBB_C+XX] + shx;

                    get_cell_range(bx0, bx1,
                                   gridj->ncx, gridj->c0[XX], gridj->sx, gridj->inv_sx,
                                   d2z_cy, rl2,
                                   &cxf, &cxl);

                    if (cxf > cxl)
                    {
                        continue;
                    }
                    new_ci_entry(nbl, cell0_i+ci, shift, flags_i[ci],
                                     nbl->work);

#ifndef NBNXN_SHIFT_BACKWARD
                    if (cxf < ci_x)
#else
                    if (shift == CENTRAL && gridi == gridj &&
                        cxf < ci_x)
#endif
                    {
                        /* Leave the pairs with i > j.
                         * x is the major index, so skip half of it.
                         */
                        cxf = ci_x;
                    }

                    set_icell_bb_simple(bb_i, ci, shx, shy, shz,
                                            nbl->work->bb_ci);

                    nbs->icell_set_x(cell0_i+ci, shx, shy, shz,
                                     gridi->na_c, nbat->xstride, nbat->x,
                                     nbl->work);

                    for (cx = cxf; cx <= cxl; cx++)
                    {
                        d2zx = d2z;
                        if (gridj->c0[XX] + cx*gridj->sx > bx1)
                        {
                            d2zx += sqr(gridj->c0[XX] + cx*gridj->sx - bx1);
                        }
                        else if (gridj->c0[XX] + (cx+1)*gridj->sx < bx0)
                        {
                            d2zx += sqr(gridj->c0[XX] + (cx+1)*gridj->sx - bx0);
                        }

#ifndef NBNXN_SHIFT_BACKWARD
                        if (gridi == gridj &&
                            cx == 0 && cyf < ci_y)
#else
                        if (gridi == gridj &&
                            cx == 0 && shift == CENTRAL && cyf < ci_y)
#endif
                        {
                            /* Leave the pairs with i > j.
                             * Skip half of y when i and j have the same x.
                             */
                            cyf_x = ci_y;
                        }
                        else
                        {
                            cyf_x = cyf;
                        }

                        for (cy = cyf_x; cy <= cyl; cy++)
                        {
                            c0 = gridj->cxy_ind[cx*gridj->ncy+cy];
                            c1 = gridj->cxy_ind[cx*gridj->ncy+cy+1];
#ifdef NBNXN_SHIFT_BACKWARD
                            if (gridi == gridj &&
                                shift == CENTRAL && c0 < ci)
                            {
                                c0 = ci;
                            }
#endif

                            d2zxy = d2zx;
                            if (gridj->c0[YY] + cy*gridj->sy > by1)
                            {
                                d2zxy += sqr(gridj->c0[YY] + cy*gridj->sy - by1);
                            }
                            else if (gridj->c0[YY] + (cy+1)*gridj->sy < by0)
                            {
                                d2zxy += sqr(gridj->c0[YY] + (cy+1)*gridj->sy - by0);
                            }
                            if (c1 > c0 && d2zxy < rl2)
                            {
                                cs = c0 + (int)(bz1_frac*(c1 - c0));
                                if (cs >= c1)
                                {
                                    cs = c1 - 1;
                                }

                                d2xy = d2zxy - d2z;

                                /* Find the lowest cell that can possibly
                                 * be within range.
                                 */
                                cf = cs;
                                while (cf > c0 &&
                                       (bbcz_j[cf*NNBSBB_D+1] >= bz0 ||
                                        d2xy + sqr(bbcz_j[cf*NNBSBB_D+1] - bz0) < rl2))
                                {
                                    cf--;
                                }

                                /* Find the highest cell that can possibly
                                 * be within range.
                                 */
                                cl = cs;
                                while (cl < c1-1 &&
                                       (bbcz_j[cl*NNBSBB_D] <= bz1 ||
                                        d2xy + sqr(bbcz_j[cl*NNBSBB_D] - bz1) < rl2))
                                {
                                    cl++;
                                }

#ifdef NBNXN_REFCODE
                                {
                                    /* Simple reference code, for debugging,
                                     * overrides the more complex code above.
                                     */
                                    int k;
                                    cf = c1;
                                    cl = -1;
                                    for (k = c0; k < c1; k++)
                                    {
                                        if (box_dist2(bx0, bx1, by0, by1, bz0, bz1,
                                                      bb+k*NNBSBB_B) < rl2 &&
                                            k < cf)
                                        {
                                            cf = k;
                                        }
                                        if (box_dist2(bx0, bx1, by0, by1, bz0, bz1,
                                                      bb+k*NNBSBB_B) < rl2 &&
                                            k > cl)
                                        {
                                            cl = k;
                                        }
                                    }
                                }
#endif

                                if (gridi == gridj)
                                {
                                    /* We want each atom/cell pair only once,
                                     * only use cj >= ci.
                                     */
#ifndef NBNXN_SHIFT_BACKWARD
                                    cf = max(cf, ci);
#else
                                    if (shift == CENTRAL)
                                    {
                                        cf = max(cf, ci);
                                    }
#endif
                                }

                                if (cf <= cl)
                                {
                                    /* For f buffer flags with simple lists */
                                    ncj_old_j = nbl->ncj;

                                            check_subcell_list_space_simple(nbl, cl-cf+1);

                                            make_cluster_list_simple(gridj,
                                                                     nbl, ci, cf, cl,
                                                                     (gridi == gridj && shift == CENTRAL),
                                                                     nbat->x,
                                                                     rl2, rbb2,
                                                                     &ndistc);
                                    ncpcheck += cl - cf + 1;

                                    if (bFBufferFlag && nbl->ncj > ncj_old_j)
                                    {
                                        int cbf, cbl, cb;

                                        cbf = nbl->cj[ncj_old_j].cj >> gridj_flag_shift;
                                        cbl = nbl->cj[nbl->ncj-1].cj >> gridj_flag_shift;
                                        for (cb = cbf; cb <= cbl; cb++)
                                        {
                                            gridj_flag[cb] = 1U<<th;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    /* Set the exclusions for this ci list */
                    set_ci_top_excls(nbs,
                                         nbl,
                                         shift == CENTRAL && gridi == gridj,
                                         gridj->na_c_2log,
                                         na_cj_2log,
                                         &(nbl->ci[nbl->nci]),
                                         excl);

                    /* Close this ci list */
                    close_ci_entry_simple(nbl);
                }
            }
        }

        if (bFBufferFlag && nbl->ncj > ncj_old_i)
        {
            work->buffer_flags.flag[(gridi->cell0+ci)>>gridi_flag_shift] = 1U<<th;
        }
    }

    work->ndistc = ndistc;

    nbs_cycle_stop(&work->cc[enbsCCsearch]);

}

static void reduce_buffer_flags(const nbnxn_search_t        nbs,
                                int                         nsrc,
                                const nbnxn_buffer_flags_t *dest)
{//called
    int s, b;
    const unsigned *flag;

    for (s = 0; s < nsrc; s++)
    {
        flag = nbs->work[s].buffer_flags.flag;

        for (b = 0; b < dest->nflag; b++)
        {
            dest->flag[b] |= flag[b];
        }
    }
}

void nbnxn_make_pairlist(const nbnxn_search_t  nbs,
                         nbnxn_atomdata_t     *nbat,
                         const t_blocka       *excl,
                         real                  rlist,
                         int                   min_ci_balanced,
                         nbnxn_pairlist_set_t *nbl_list,
                         int                   iloc,
                         int                   nb_kernel_type,
                         t_nrnb               *nrnb)
{ //called
    nbnxn_grid_t *gridi, *gridj;
    gmx_bool bGPUCPU;
    int nzi, zi, zj0, zj1, zj;
    int nsubpair_max;
    int th;
    int nnbl;
    nbnxn_pairlist_t **nbl;
    int ci_block;
    gmx_bool CombineNBLists;
    int np_tot, np_noq, np_hlj, nap;

    /* Check if we are running hybrid GPU + CPU nbnxn mode */
    bGPUCPU = (!nbs->grid[0].bSimple && nbl_list->bSimple);

    nnbl            = nbl_list->nnbl;
    nbl             = nbl_list->nbl;
    CombineNBLists  = nbl_list->bCombined;


    nbat->bUseBufferFlags = (nbat->nout > 1);
    /* We should re-init the flags before making the first list */
    if (nbat->bUseBufferFlags && (LOCAL_I(iloc) || bGPUCPU))
    {
        init_buffer_flags(&nbat->buffer_flags, nbat->natoms);
    }

        switch (nb_kernel_type)
        {
#ifdef GMX_NBNXN_SIMD_4XN
            case nbnxnk4xN_SIMD_4xN:
                nbs->icell_set_x = icell_set_x_simd_4xn;
                break;
#endif
#ifdef GMX_NBNXN_SIMD_2XNN
            case nbnxnk4xN_SIMD_2xNN:
                nbs->icell_set_x = icell_set_x_simd_2xnn;
                break;
#endif
            default:
                nbs->icell_set_x = icell_set_x_simple;
                break;
        }

    if (LOCAL_I(iloc))
    {
        /* Only zone (grid) 0 vs 0 */
        nzi = 1;
        zj0 = 0;
        zj1 = 1;
    }
    else
    {
        nzi = nbs->zones->nizone;
    }

        nsubpair_max = 0;

    /* Clear all pair-lists */
    for (th = 0; th < nnbl; th++)
    {
        clear_pairlist(nbl[th]);
    }

    for (zi = 0; zi < nzi; zi++)
    {
        gridi = &nbs->grid[zi];

        if (NONLOCAL_I(iloc))
        {
            zj0 = nbs->zones->izone[zi].j0;
            zj1 = nbs->zones->izone[zi].j1;
            if (zi == 0)
            {
                zj0++;
            }
        }
        for (zj = zj0; zj < zj1; zj++)
        {
            gridj = &nbs->grid[zj];


            nbs_cycle_start(&nbs->cc[enbsCCsearch]);

            if (nbl[0]->bSimple && !gridi->bSimple)
            {
                /* Hybrid list, determine blocking later */
                ci_block = 0;
            }
            else
            {
                ci_block = get_ci_block_size(gridi, nbs->DomDec, nnbl);
            }

#pragma omp parallel for num_threads(nnbl) schedule(static)
            for (th = 0; th < nnbl; th++)
            {
                /* Re-init the thread-local work flag data before making
                 * the first list (not an elegant conditional).
                 */
                if (nbat->bUseBufferFlags && ((zi == 0 && zj == 0) ||
                                              (bGPUCPU && zi == 0 && zj == 1)))
                {
                    init_buffer_flags(&nbs->work[th].buffer_flags, nbat->natoms);
                }

                if (CombineNBLists && th > 0)
                {
                    clear_pairlist(nbl[th]);
                }

                /* Divide the i super cell equally over the nblists */
                nbnxn_make_pairlist_part(nbs, gridi, gridj,
                                         &nbs->work[th], nbat, excl,
                                         rlist,
                                         nb_kernel_type,
                                         ci_block,
                                         nbat->bUseBufferFlags,
                                         nsubpair_max,
                                         (LOCAL_I(iloc) || nbs->zones->n <= 2),
                                         min_ci_balanced,
                                         th, nnbl,
                                         nbl[th]);
            }
            nbs_cycle_stop(&nbs->cc[enbsCCsearch]);

            np_tot = 0;
            np_noq = 0;
            np_hlj = 0;
            for (th = 0; th < nnbl; th++)
            {
                inc_nrnb(nrnb, eNR_NBNXN_DIST2, nbs->work[th].ndistc);

                if (nbl_list->bSimple)
                {
                    np_tot += nbl[th]->ncj;
                    np_noq += nbl[th]->work->ncj_noq;
                    np_hlj += nbl[th]->work->ncj_hlj;
                }
                else
                {
                    /* This count ignores potential subsequent pair pruning */
                    np_tot += nbl[th]->nci_tot;
                }
            }
            nap                   = nbl[0]->na_ci*nbl[0]->na_cj;
            nbl_list->natpair_ljq = (np_tot - np_noq)*nap - np_hlj*nap/2;
            nbl_list->natpair_lj  = np_noq*nap;
            nbl_list->natpair_q   = np_hlj*nap/2;

        }
    }

    if (nbat->bUseBufferFlags)
    {
        reduce_buffer_flags(nbs, nnbl, &nbat->buffer_flags);
    }

    /*
       print_supersub_nsp("nsubpair",nbl[0],iloc);
     */

    /* Special performance logging stuff (env.var. GMX_NBNXN_CYCLE) */
    if (LOCAL_I(iloc))
    {
        nbs->search_count++;
    }

}
