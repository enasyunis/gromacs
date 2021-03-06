/* IMPORTANT FOR DEVELOPERS:
 *
 * Triclinic pme stuff isn't entirely trivial, and we've experienced
 * some bugs during development (many of them due to me). To avoid
 * this in the future, please check the following things if you make
 * changes in this file:
 *
 * 1. You should obtain identical (at least to the PME precision)
 *    energies, forces, and virial for
 *    a rectangular box and a triclinic one where the z (or y) axis is
 *    tilted a whole box side. For instance you could use these boxes:
 *
 *    rectangular       triclinic
 *     2  0  0           2  0  0
 *     0  2  0           0  2  0
 *     0  0  6           2  2  6
 *
 * 2. You should check the energy conservation in a triclinic box.
 *
 * It might seem an overkill, but better safe than sorry.
 * /Erik 001109
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_LIB_MPI
#include <mpi.h>
#endif
#ifdef GMX_THREAD_MPI
#include "tmpi.h"
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "typedefs.h"
#include "txtdump.h"
#include "vec.h"
#include "gmxcomplex.h"
#include "smalloc.h"
#include "futil.h"
#include "coulomb.h"
#include "gmx_fatal.h"
#include "pme.h"
#include "physics.h"
#include "gmx_parallel_3dfft.h"
#include "gmx_omp.h"


#define DFT_TOL 1e-7
/* #define PRT_FORCE */
/* conditions for on the fly time-measurement */
/* #define TAKETIME (step > 1 && timesteps < 10) */
#define TAKETIME FALSE
#define mpi_type MPI_DOUBLE

/* GMX_CACHE_SEP should be a multiple of 16 to preserve alignment */
#define GMX_CACHE_SEP 64

/* We only define a maximum to be able to use local arrays without allocation.
 * An order larger than 12 should never be needed, even for test cases.
 * If needed it can be changed here.
 */
#define PME_ORDER_MAX 12

/* Internal datastructures */
typedef struct {
    int send_index0;
    int send_nindex;
    int recv_index0;
    int recv_nindex;
    int recv_size;   /* Receive buffer width, used with OpenMP */
} pme_grid_comm_t;

typedef struct {
    MPI_Comm         mpi_comm;
    int              nnodes, nodeid;
    int             *s2g0;
    int             *s2g1;
    int              noverlap_nodes;
    int             *send_id, *recv_id;
    int              send_size; /* Send buffer width, used with OpenMP */
    pme_grid_comm_t *comm_data;
    real            *sendbuf;
    real            *recvbuf;
} pme_overlap_t;

typedef struct {
    int *n;      /* Cumulative counts of the number of particles per thread */
    int  nalloc; /* Allocation size of i */
    int *i;      /* Particle indices ordered on thread index (n) */
} thread_plist_t;

typedef struct {
    int      *thread_one;
    int       n;
    int      *ind;
    splinevec theta;
    real     *ptr_theta_z;
    splinevec dtheta;
    real     *ptr_dtheta_z;
} splinedata_t;

typedef struct {
    int      dimind;        /* The index of the dimension, 0=x, 1=y */
    int      nslab;
    int      nodeid;
    MPI_Comm mpi_comm;

    int     *node_dest;     /* The nodes to send x and q to with DD */
    int     *node_src;      /* The nodes to receive x and q from with DD */
    int     *buf_index;     /* Index for commnode into the buffers */

    int      maxshift;

    int      npd;
    int      pd_nalloc;
    int     *pd;
    int     *count;         /* The number of atoms to send to each node */
    int    **count_thread;
    int     *rcount;        /* The number of atoms to receive */

    int      n;
    int      nalloc;
    rvec    *x;
    real    *q;
    rvec    *f;
    gmx_bool bSpread;       /* These coordinates are used for spreading */
    int      pme_order;
    ivec    *idx;
    rvec    *fractx;            /* Fractional coordinate relative to the
                                 * lower cell boundary
                                 */
    int             nthread;
    int            *thread_idx; /* Which thread should spread which charge */
    thread_plist_t *thread_plist;
    splinedata_t   *spline;
} pme_atomcomm_t;

#define FLBS  3
#define FLBSZ 4

typedef struct {
    ivec  ci;     /* The spatial location of this grid         */
    ivec  n;      /* The used size of *grid, including order-1 */
    ivec  offset; /* The grid offset from the full node grid   */
    int   order;  /* PME spreading order                       */
    ivec  s;      /* The allocated size of *grid, s >= n       */
    real *grid;   /* The grid local thread, size n             */
} pmegrid_t;

typedef struct {
    pmegrid_t  grid;         /* The full node grid (non thread-local)            */
    int        nthread;      /* The number of threads operating on this grid     */
    ivec       nc;           /* The local spatial decomposition over the threads */
    pmegrid_t *grid_th;      /* Array of grids for each thread                   */
    real      *grid_all;     /* Allocated array for the grids in *grid_th        */
    int      **g2t;          /* The grid to thread index                         */
    ivec       nthread_comm; /* The number of threads to communicate with        */
} pmegrids_t;



typedef struct {
    /* work data for solve_pme */
    int      nalloc;
    real *   mhx;
    real *   mhy;
    real *   mhz;
    real *   m2;
    real *   denom;
    real *   tmp1_alloc;
    real *   tmp1;
    real *   eterm;
    real *   m2inv;

    real     energy;
    matrix   vir;
} pme_work_t;

typedef struct gmx_pme {
    int           ndecompdim; /* The number of decomposition dimensions */
    int           nodeid;     /* Our nodeid in mpi->mpi_comm */
    int           nodeid_major;
    int           nodeid_minor;
    int           nnodes;    /* The number of nodes doing PME */
    int           nnodes_major;
    int           nnodes_minor;

    MPI_Comm      mpi_comm;
    MPI_Comm      mpi_comm_d[2]; /* Indexed on dimension, 0=x, 1=y */
    MPI_Datatype  rvec_mpi;      /* the pme vector's MPI type */

    int        nthread;       /* The number of threads doing PME */

    gmx_bool   bPPnode;       /* Node also does particle-particle forces */
    gmx_bool   bFEP;          /* Compute Free energy contribution */
    int        nkx, nky, nkz; /* Grid dimensions */
    gmx_bool   bP3M;          /* Do P3M: optimize the influence function */
    int        pme_order;
    real       epsilon_r;

    pmegrids_t pmegridA;  /* Grids on which we do spreading/interpolation, includes overlap */
    pmegrids_t pmegridB;
    /* The PME charge spreading grid sizes/strides, includes pme_order-1 */
    int        pmegrid_nx, pmegrid_ny, pmegrid_nz;
    /* pmegrid_nz might be larger than strictly necessary to ensure
     * memory alignment, pmegrid_nz_base gives the real base size.
     */
    int     pmegrid_nz_base;
    /* The local PME grid starting indices */
    int     pmegrid_start_ix, pmegrid_start_iy, pmegrid_start_iz;


    real                 *fftgridA; /* Grids for FFT. With 1D FFT decomposition this can be a pointer */
    real                 *fftgridB; /* inside the interpolation grid, but separate for 2D PME decomp. */
    int                   fftgrid_nx, fftgrid_ny, fftgrid_nz;

    t_complex            *cfftgridA;  /* Grids for complex FFT data */
    t_complex            *cfftgridB;
    int                   cfftgrid_nx, cfftgrid_ny, cfftgrid_nz;

    gmx_parallel_3dfft_t  pfft_setupA;
    gmx_parallel_3dfft_t  pfft_setupB;

    int                  *nnx, *nny, *nnz;
    real                 *fshx, *fshy, *fshz;

    pme_atomcomm_t        atc[2]; /* Indexed on decomposition index */
    matrix                recipbox;
    splinevec             bsp_mod;

    pme_overlap_t         overlap[2]; /* Indexed on dimension, 0=x, 1=y */

    pme_atomcomm_t        atc_energy; /* Only for gmx_pme_calc_energy */

    rvec                 *bufv;       /* Communication buffer */
    real                 *bufr;       /* Communication buffer */
    int                   buf_nalloc; /* The communication buffer size */

    /* thread local work data for solve_pme */
    pme_work_t *work;

    /* Work data for PME_redist */
    gmx_bool redist_init;
    int *    scounts;
    int *    rcounts;
    int *    sdispls;
    int *    rdispls;
    int *    sidx;
    int *    idxa;
    real *   redist_buf;
    int      redist_buf_nalloc;

    /* Work data for sum_qgrid */
    real *   sum_qgrid_tmp;
    real *   sum_qgrid_dd_tmp;
} t_gmx_pme;


static void calc_interpolation_idx(gmx_pme_t pme, pme_atomcomm_t *atc,
                                   int start, int end, int thread)
{ // called
    int             i;
    int            *idxptr, tix, tiy, tiz;
    real           *xptr, *fptr, tx, ty, tz;
    real            rxx, ryx, ryy, rzx, rzy, rzz;
    int             nx, ny, nz;
    int             start_ix, start_iy, start_iz;
    int            *g2tx, *g2ty, *g2tz;
    int            *thread_idx = NULL;
    thread_plist_t *tpl        = NULL;
    int            *tpl_n      = NULL;
    int             thread_i;

    nx  = pme->nkx;
    ny  = pme->nky;
    nz  = pme->nkz;

    start_ix = pme->pmegrid_start_ix;
    start_iy = pme->pmegrid_start_iy;
    start_iz = pme->pmegrid_start_iz;

    rxx = pme->recipbox[XX][XX];
    ryx = pme->recipbox[YY][XX];
    ryy = pme->recipbox[YY][YY];
    rzx = pme->recipbox[ZZ][XX];
    rzy = pme->recipbox[ZZ][YY];
    rzz = pme->recipbox[ZZ][ZZ];

    g2tx = pme->pmegridA.g2t[XX];
    g2ty = pme->pmegridA.g2t[YY];
    g2tz = pme->pmegridA.g2t[ZZ];

        thread_idx = atc->thread_idx;

        tpl   = &atc->thread_plist[thread];
        tpl_n = tpl->n;
        for (i = 0; i < atc->nthread; i++) // 12
        {
            tpl_n[i] = 0;
        }
    for (i = start; i < end; i++) // 3000 divided amongst 12 threads
    {
        xptr   = atc->x[i];
        idxptr = atc->idx[i];
        fptr   = atc->fractx[i];

        /* Fractional coordinates along box vectors, add 2.0 to make 100% sure we are positive for triclinic boxes */
        tx = nx * ( xptr[XX] * rxx + xptr[YY] * ryx + xptr[ZZ] * rzx + 2.0 );
        ty = ny * (                  xptr[YY] * ryy + xptr[ZZ] * rzy + 2.0 );
        tz = nz * (                                   xptr[ZZ] * rzz + 2.0 );

        tix = (int)(tx);
        tiy = (int)(ty);
        tiz = (int)(tz);

        /* Because decomposition only occurs in x and y,
         * we never have a fraction correction in z.
         */
        fptr[XX] = tx - tix + pme->fshx[tix];
        fptr[YY] = ty - tiy + pme->fshy[tiy];
        fptr[ZZ] = tz - tiz;

        idxptr[XX] = pme->nnx[tix];
        idxptr[YY] = pme->nny[tiy];
        idxptr[ZZ] = pme->nnz[tiz];

            thread_i      = g2tx[idxptr[XX]] + g2ty[idxptr[YY]] + g2tz[idxptr[ZZ]];
            thread_idx[i] = thread_i;
            tpl_n[thread_i]++;
    }

        /* Make a list of particle indices sorted on thread */

        /* Get the cumulative count */
        for (i = 1; i < atc->nthread; i++)
        {
            tpl_n[i] += tpl_n[i-1];
        }
        /* The current implementation distributes particles equally
         * over the threads, so we could actually allocate for that
         * in pme_realloc_atomcomm_things.
         */
        tpl->nalloc = over_alloc_large(tpl_n[atc->nthread-1]);
        srenew(tpl->i, tpl->nalloc);
        /* Set tpl_n to the cumulative start */
        for (i = atc->nthread-1; i >= 1; i--)
        {
            tpl_n[i] = tpl_n[i-1];
        }
        tpl_n[0] = 0;

        /* Fill our thread local array with indices sorted on thread */
        for (i = start; i < end; i++)
        {
            tpl->i[tpl_n[atc->thread_idx[i]]++] = i;
        }
        /* Now tpl_n contains the cummulative count again */
}

static void make_thread_local_ind(pme_atomcomm_t *atc,
                                  int thread, splinedata_t *spline)
{ // called
    int             n, t, i, start, end;
    thread_plist_t *tpl;

    /* Combine the indices made by each thread into one index */
    n     = 0;
    start = 0;
    for (t = 0; t < atc->nthread; t++)
    {
        tpl = &atc->thread_plist[t];
        /* Copy our part (start - end) from the list of thread t */
        if (thread > 0) // not master thread id.
        {
            start = tpl->n[thread-1];
        }
        end = tpl->n[thread];
        for (i = start; i < end; i++)
        {
            spline->ind[n++] = tpl->i[i];
        }
    }

    spline->n = n;
}



static void realloc_splinevec(splinevec th, real **ptr_z, int nalloc)
{ // called
    const int padding = 4;
    int       i;

    srenew(th[XX], nalloc);
    srenew(th[YY], nalloc);
    /* In z we add padding, this is only required for the aligned SSE code */
    srenew(*ptr_z, nalloc+2*padding);
    th[ZZ] = *ptr_z + padding;
    for (i = 0; i < padding; i++)
    {
        (*ptr_z)[               i] = 0;
        (*ptr_z)[padding+nalloc+i] = 0;
    }
}

static void pme_realloc_splinedata(splinedata_t *spline, pme_atomcomm_t *atc)
{ // called
    int i, d;

    srenew(spline->ind, atc->nalloc);
    /* Initialize the index to identity so it works without threads */
    for (i = 0; i < atc->nalloc; i++) // 3000
    {
        spline->ind[i] = i;
    }

    realloc_splinevec(spline->theta, &spline->ptr_theta_z,
                      atc->pme_order*atc->nalloc);
    realloc_splinevec(spline->dtheta, &spline->ptr_dtheta_z,
                      atc->pme_order*atc->nalloc);
}

static void pme_realloc_atomcomm_things(pme_atomcomm_t *atc)
{ // called
    int nalloc_old, i, j, nalloc_tpl;

    /* We have to avoid a NULL pointer for atc->x to avoid
     * possible fatal errors in MPI routines.
     */
        nalloc_old  = atc->nalloc;
        atc->nalloc = over_alloc_dd(max(atc->n, 1));
        srenew(atc->fractx, atc->nalloc);
        srenew(atc->idx, atc->nalloc);

        srenew(atc->thread_idx, atc->nalloc);

        for (i = 0; i < atc->nthread; i++)
        {
            pme_realloc_splinedata(&atc->spline[i], atc);
        }
}


static int
copy_fftgrid_to_pmegrid(gmx_pme_t pme, const real *fftgrid, real *pmegrid,
                        int nthread, int thread)
{ // called
    ivec          local_fft_ndata, local_fft_offset, local_fft_size;
    ivec          local_pme_size;
    int           ixy0, ixy1, ixy, ix, iy, iz;
    int           pmeidx, fftidx;
    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_real_limits(pme->pfft_setupA,
                                   local_fft_ndata,
                                   local_fft_offset,
                                   local_fft_size);

    local_pme_size[0] = pme->pmegrid_nx;
    local_pme_size[1] = pme->pmegrid_ny;
    local_pme_size[2] = pme->pmegrid_nz;

    /* The fftgrid is always 'justified' to the lower-left corner of the PME grid,
       the offset is identical, and the PME grid always has more data (due to overlap)
     */
    ixy0 = ((thread  )*local_fft_ndata[XX]*local_fft_ndata[YY])/nthread;
    ixy1 = ((thread+1)*local_fft_ndata[XX]*local_fft_ndata[YY])/nthread;

    for (ixy = ixy0; ixy < ixy1; ixy++)
    {
        ix = ixy/local_fft_ndata[YY];
        iy = ixy - ix*local_fft_ndata[YY];

        pmeidx = (ix*local_pme_size[YY] + iy)*local_pme_size[ZZ];
        fftidx = (ix*local_fft_size[YY] + iy)*local_fft_size[ZZ];
        for (iz = 0; iz < local_fft_ndata[ZZ]; iz++)
        {
            pmegrid[pmeidx+iz] = fftgrid[fftidx+iz];
        }
    }

    return 0;
}

static void
unwrap_periodic_pmegrid(gmx_pme_t pme, real *pmegrid)
{ // called
    int     nx, ny, nz, pnx, pny, pnz, ny_x, overlap, ix;

    nx = pme->nkx;
    ny = pme->nky;
    nz = pme->nkz;

    pnx = pme->pmegrid_nx;
    pny = pme->pmegrid_ny;
    pnz = pme->pmegrid_nz;

    overlap = pme->pme_order - 1;
    ny_x = ny;

        for (ix = 0; ix < overlap; ix++) // 7
        {
            int iy, iz;

            for (iy = 0; iy < ny_x; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[((nx+ix)*pny+iy)*pnz+iz] =
                        pmegrid[(ix*pny+iy)*pnz+iz];
                }
            }
        }

#pragma omp parallel for num_threads(pme->nthread) schedule(static)
        for (ix = 0; ix < pme->pmegrid_nx; ix++)
        {
            int iy, iz;

            for (iy = 0; iy < overlap; iy++)
            {
                for (iz = 0; iz < nz; iz++)
                {
                    pmegrid[(ix*pny+ny+iy)*pnz+iz] =
                        pmegrid[(ix*pny+iy)*pnz+iz];
                }
            }
        }

    /* Copy periodic overlap in z */
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
    for (ix = 0; ix < pme->pmegrid_nx; ix++)
    {
        int iy, iz;

        for (iy = 0; iy < pme->pmegrid_ny; iy++)
        {
            for (iz = 0; iz < overlap; iz++)
            {
                pmegrid[(ix*pny+iy)*pnz+nz+iz] =
                    pmegrid[(ix*pny+iy)*pnz+iz];
            }
        }
    }
}

/* This has to be a macro to enable full compiler optimization with xlC (and probably others too) */
#define DO_BSPLINE(order)                            \
    for (ithx = 0; (ithx < order); ithx++)                    \
    {                                                    \
        index_x = (i0+ithx)*pny*pnz;                     \
        valx    = qn*thx[ithx];                          \
                                                     \
        for (ithy = 0; (ithy < order); ithy++)                \
        {                                                \
            valxy    = valx*thy[ithy];                   \
            index_xy = index_x+(j0+ithy)*pnz;            \
                                                     \
            for (ithz = 0; (ithz < order); ithz++)            \
            {                                            \
                index_xyz        = index_xy+(k0+ithz);   \
                grid[index_xyz] += valxy*thz[ithz];      \
            }                                            \
        }                                                \
    }


static void spread_q_bsplines_thread(pmegrid_t *pmegrid,
                                     pme_atomcomm_t *atc, splinedata_t *spline
                                     )
{ // called

    /* spread charges from home atoms to local grid */
    real          *grid;
    pme_overlap_t *ol;
    int            b, i, nn, n, ithx, ithy, ithz, i0, j0, k0;
    int       *    idxptr;
    int            order, norder, index_x, index_xy, index_xyz;
    real           valx, valxy, qn;
    real          *thx, *thy, *thz;
    int            localsize, bndsize;
    int            pnx, pny, pnz, ndatatot;
    int            offx, offy, offz;

    pnx = pmegrid->s[XX];
    pny = pmegrid->s[YY];
    pnz = pmegrid->s[ZZ];

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];

    ndatatot = pnx*pny*pnz;
    grid     = pmegrid->grid;
    for (i = 0; i < ndatatot; i++)
    {
        grid[i] = 0;
    }

    order = pmegrid->order;

    for (nn = 0; nn < spline->n; nn++)
    {
        n  = spline->ind[nn];
        qn = atc->q[n];

        if (qn != 0)
        {
            idxptr = atc->idx[n];
            norder = nn*order;

            i0   = idxptr[XX] - offx;
            j0   = idxptr[YY] - offy;
            k0   = idxptr[ZZ] - offz;

            thx = spline->theta[XX] + norder;
            thy = spline->theta[YY] + norder;
            thz = spline->theta[ZZ] + norder;

	    DO_BSPLINE(order);

        }
    }
}

static void pmegrid_init(pmegrid_t *grid,
                         int cx, int cy, int cz,
                         int x0, int y0, int z0,
                         int x1, int y1, int z1,
                         gmx_bool set_alignment,
                         int pme_order,
                         real *ptr)
{ //called
    int nz, gridsize;

    grid->ci[XX]     = cx;
    grid->ci[YY]     = cy;
    grid->ci[ZZ]     = cz;
    grid->offset[XX] = x0;
    grid->offset[YY] = y0;
    grid->offset[ZZ] = z0;
    grid->n[XX]      = x1 - x0 + pme_order - 1;
    grid->n[YY]      = y1 - y0 + pme_order - 1;
    grid->n[ZZ]      = z1 - z0 + pme_order - 1;
    copy_ivec(grid->n, grid->s);

    nz = grid->s[ZZ];
    if (set_alignment)
    {
        grid->s[ZZ] = nz;
    }
    else if (nz != grid->s[ZZ])
    {
        gmx_incons("pmegrid_init call with an unaligned z size");
    }

    grid->order = pme_order;
    if (ptr == NULL)
    {
        gridsize = grid->s[XX]*grid->s[YY]*grid->s[ZZ];
        snew_aligned(grid->grid, gridsize, 16);
    }
    else
    {
        grid->grid = ptr;
    }
}

static int div_round_up(int enumerator, int denominator)
{ // called
    return (enumerator + denominator - 1)/denominator;
}

static void make_subgrid_division(const ivec n, int ovl, int nthread,
                                  ivec nsub)
{ // called
    int gsize_opt, gsize;
    int nsx, nsy, nsz;
    char *env;

    gsize_opt = -1;
    for (nsx = 1; nsx <= nthread; nsx++)
    {
        if (nthread % nsx == 0)
        {
            for (nsy = 1; nsy <= nthread; nsy++)
            {
                if (nsx*nsy <= nthread && nthread % (nsx*nsy) == 0)
                {
                    nsz = nthread/(nsx*nsy);

                    /* Determine the number of grid points per thread */
                    gsize =
                        (div_round_up(n[XX], nsx) + ovl)*
                        (div_round_up(n[YY], nsy) + ovl)*
                        (div_round_up(n[ZZ], nsz) + ovl);

                    /* Minimize the number of grids points per thread
                     * and, secondarily, the number of cuts in minor dimensions.
                     */
                    if (gsize_opt == -1 ||
                        gsize < gsize_opt ||
                        (gsize == gsize_opt &&
                         (nsz < nsub[ZZ] || (nsz == nsub[ZZ] && nsy < nsub[YY]))))
                    {
                        nsub[XX]  = nsx;
                        nsub[YY]  = nsy;
                        nsub[ZZ]  = nsz;
                        gsize_opt = gsize;
                    }
                }
            }
        }
    }

    env = getenv("GMX_PME_THREAD_DIVISION");
    if (env != NULL)
    {
        sscanf(env, "%d %d %d", &nsub[XX], &nsub[YY], &nsub[ZZ]);
    }

    if (nsub[XX]*nsub[YY]*nsub[ZZ] != nthread)
    {
        gmx_fatal(FARGS, "PME grid thread division (%d x %d x %d) does not match the total number of threads (%d)", nsub[XX], nsub[YY], nsub[ZZ], nthread);
    }
}

static void pmegrids_init(pmegrids_t *grids,
                          int nx, int ny, int nz, int nz_base,
                          int pme_order,
                          int nthread,
                          int overlap_x,
                          int overlap_y)
{ // called
    ivec n, n_base, g0, g1;
    int t, x, y, z, d, i, tfac;
    int max_comm_lines = -1;

    n[XX] = nx - (pme_order - 1);
    n[YY] = ny - (pme_order - 1);
    n[ZZ] = nz - (pme_order - 1);

    copy_ivec(n, n_base);
    n_base[ZZ] = nz_base;

    pmegrid_init(&grids->grid, 0, 0, 0, 0, 0, 0, n[XX], n[YY], n[ZZ], FALSE, pme_order,
                 NULL);

    grids->nthread = nthread;

    make_subgrid_division(n_base, pme_order-1, grids->nthread, grids->nc);

        ivec nst;
        int gridsize;

        for (d = 0; d < DIM; d++)
        {
            nst[d] = div_round_up(n[d], grids->nc[d]) + pme_order - 1;
        }


        snew(grids->grid_th, grids->nthread);
        t        = 0;
        gridsize = nst[XX]*nst[YY]*nst[ZZ];
        snew_aligned(grids->grid_all,
                     grids->nthread*gridsize+(grids->nthread+1)*GMX_CACHE_SEP,
                     16);

        for (x = 0; x < grids->nc[XX]; x++)
        {
            for (y = 0; y < grids->nc[YY]; y++)
            {
                for (z = 0; z < grids->nc[ZZ]; z++)
                {
                    pmegrid_init(&grids->grid_th[t],
                                 x, y, z,
                                 (n[XX]*(x  ))/grids->nc[XX],
                                 (n[YY]*(y  ))/grids->nc[YY],
                                 (n[ZZ]*(z  ))/grids->nc[ZZ],
                                 (n[XX]*(x+1))/grids->nc[XX],
                                 (n[YY]*(y+1))/grids->nc[YY],
                                 (n[ZZ]*(z+1))/grids->nc[ZZ],
                                 TRUE,
                                 pme_order,
                                 grids->grid_all+GMX_CACHE_SEP+t*(gridsize+GMX_CACHE_SEP));
                    t++;
                }
            }
        }
    

    snew(grids->g2t, DIM);
    tfac = 1;
    for (d = DIM-1; d >= 0; d--)
    {
        snew(grids->g2t[d], n[d]);
        t = 0;
        for (i = 0; i < n[d]; i++)
        {
            /* The second check should match the parameters
             * of the pmegrid_init call above.
             */
            while (t + 1 < grids->nc[d] && i >= (n[d]*(t+1))/grids->nc[d])
            {
                t++;
            }
            grids->g2t[d][i] = t*tfac;
        }

        tfac *= grids->nc[d];

        switch (d)
        {
            case XX: max_comm_lines = overlap_x;     break;
            case YY: max_comm_lines = overlap_y;     break;
            case ZZ: max_comm_lines = pme_order - 1; break;
        }
        grids->nthread_comm[d] = 0;
        while ((n[d]*grids->nthread_comm[d])/grids->nc[d] < max_comm_lines &&
               grids->nthread_comm[d] < grids->nc[d])
        {
            grids->nthread_comm[d]++;
        }
        /* It should be possible to make grids->nthread_comm[d]==grids->nc[d]
         * work, but this is not a problematic restriction.
         */
        if (grids->nc[d] > 1 && grids->nthread_comm[d] > grids->nc[d])
        {
            gmx_fatal(FARGS, "Too many threads for PME (%d) compared to the number of grid lines, reduce the number of threads doing PME", grids->nthread);
        }
    }
}

static void realloc_work(pme_work_t *work, int nkx)
{ // called
    if (nkx > work->nalloc)
    {
        work->nalloc = nkx;
        srenew(work->mhx, work->nalloc);
        srenew(work->mhy, work->nalloc);
        srenew(work->mhz, work->nalloc);
        srenew(work->m2, work->nalloc);
        /* Allocate an aligned pointer for SSE operations, including 3 extra
         * elements at the end since SSE operates on 4 elements at a time.
         */
        sfree_aligned(work->denom);
        sfree_aligned(work->tmp1);
        sfree_aligned(work->eterm);
        snew_aligned(work->denom, work->nalloc+3, 16);
        snew_aligned(work->tmp1, work->nalloc+3, 16);
        snew_aligned(work->eterm, work->nalloc+3, 16);
        srenew(work->m2inv, work->nalloc);
    }
}


inline static void calc_exponentials(int start, int end, real f, real *d, real *r, real *e)
{ // called
    int kx;
    for (kx = start; kx < end; kx++)
    {
        d[kx] = 1.0/d[kx];
    }
    for (kx = start; kx < end; kx++)
    {
        r[kx] = exp(r[kx]);
    }
    for (kx = start; kx < end; kx++)
    {
        e[kx] = f*r[kx]*d[kx];
    }
}


static int solve_pme_yzx(gmx_pme_t pme, t_complex *grid,
                         real ewaldcoeff, real vol,
                         gmx_bool bEnerVir,
                         int nthread, int thread)
{ // called
    /* do recip sum over local cells in grid */
    /* y major, z middle, x minor or continuous */
    t_complex *p0;
    int     kx, ky, kz, maxkx, maxky, maxkz;
    int     nx, ny, nz, iyz0, iyz1, iyz, iy, iz, kxstart, kxend;
    real    mx, my, mz;
    real    factor = M_PI*M_PI/(ewaldcoeff*ewaldcoeff);
    real    ets2, struct2, vfactor, ets2vf;
    real    d1, d2, energy = 0;
    real    by, bz;
    real    virxx = 0, virxy = 0, virxz = 0, viryy = 0, viryz = 0, virzz = 0;
    real    rxx, ryx, ryy, rzx, rzy, rzz;
    pme_work_t *work;
    real    *mhx, *mhy, *mhz, *m2, *denom, *tmp1, *eterm, *m2inv;
    real    mhxk, mhyk, mhzk, m2k;
    real    corner_fac;
    ivec    complex_order;
    ivec    local_ndata, local_offset, local_size;
    real    elfac;

    elfac = ONE_4PI_EPS0/pme->epsilon_r;

    nx = pme->nkx;
    ny = pme->nky;
    nz = pme->nkz;

    /* Dimensions should be identical for A/B grid, so we just use A here */
    gmx_parallel_3dfft_complex_limits(pme->pfft_setupA,
                                      complex_order,
                                      local_ndata,
                                      local_offset,
                                      local_size);

    rxx = pme->recipbox[XX][XX];
    ryx = pme->recipbox[YY][XX];
    ryy = pme->recipbox[YY][YY];
    rzx = pme->recipbox[ZZ][XX];
    rzy = pme->recipbox[ZZ][YY];
    rzz = pme->recipbox[ZZ][ZZ];

    maxkx = (nx+1)/2;
    maxky = (ny+1)/2;
    maxkz = nz/2+1;

    work  = &pme->work[thread];
    mhx   = work->mhx;
    mhy   = work->mhy;
    mhz   = work->mhz;
    m2    = work->m2;
    denom = work->denom;
    tmp1  = work->tmp1;
    eterm = work->eterm;
    m2inv = work->m2inv;

    iyz0 = local_ndata[YY]*local_ndata[ZZ]* thread   /nthread;
    iyz1 = local_ndata[YY]*local_ndata[ZZ]*(thread+1)/nthread;

    for (iyz = iyz0; iyz < iyz1; iyz++)
    {
        iy = iyz/local_ndata[ZZ];
        iz = iyz - iy*local_ndata[ZZ];

        ky = iy + local_offset[YY];

        if (ky < maxky)
        {
            my = ky;
        }
        else
        {
            my = (ky - ny);
        }

        by = M_PI*vol*pme->bsp_mod[YY][ky];

        kz = iz + local_offset[ZZ];

        mz = kz;

        bz = pme->bsp_mod[ZZ][kz];

        /* 0.5 correction for corner points */
        corner_fac = 1;
        if (kz == 0 || kz == (nz+1)/2)
        {
            corner_fac = 0.5;
        }

        p0 = grid + iy*local_size[ZZ]*local_size[XX] + iz*local_size[XX];

        /* We should skip the k-space point (0,0,0) */
        if (local_offset[XX] > 0 || ky > 0 || kz > 0)
        {
            kxstart = local_offset[XX];
        }
        else
        {
            kxstart = local_offset[XX] + 1;
            p0++;
        }
        kxend = local_offset[XX] + local_ndata[XX];

        if (bEnerVir)
        {
            /* More expensive inner loop, especially because of the storage
             * of the mh elements in array's.
             * Because x is the minor grid index, all mh elements
             * depend on kx for triclinic unit cells.
             */

            /* Two explicit loops to avoid a conditional inside the loop */
            for (kx = kxstart; kx < maxkx; kx++)
            {
                mx = kx;

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                mhx[kx]   = mhxk;
                mhy[kx]   = mhyk;
                mhz[kx]   = mhzk;
                m2[kx]    = m2k;
                denom[kx] = m2k*bz*by*pme->bsp_mod[XX][kx];
                tmp1[kx]  = -factor*m2k;
            }

            for (kx = maxkx; kx < kxend; kx++)
            {
                mx = (kx - nx);

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                mhx[kx]   = mhxk;
                mhy[kx]   = mhyk;
                mhz[kx]   = mhzk;
                m2[kx]    = m2k;
                denom[kx] = m2k*bz*by*pme->bsp_mod[XX][kx];
                tmp1[kx]  = -factor*m2k;
            }

            for (kx = kxstart; kx < kxend; kx++)
            {
                m2inv[kx] = 1.0/m2[kx];
            }

            calc_exponentials(kxstart, kxend, elfac, denom, tmp1, eterm);

            for (kx = kxstart; kx < kxend; kx++, p0++)
            {
                d1      = p0->re;
                d2      = p0->im;

                p0->re  = d1*eterm[kx];
                p0->im  = d2*eterm[kx];

                struct2 = 2.0*(d1*d1+d2*d2);

                tmp1[kx] = eterm[kx]*struct2;
            }

            for (kx = kxstart; kx < kxend; kx++)
            {
                ets2     = corner_fac*tmp1[kx];
                vfactor  = (factor*m2[kx] + 1.0)*2.0*m2inv[kx];
                energy  += ets2;

                ets2vf   = ets2*vfactor;
                virxx   += ets2vf*mhx[kx]*mhx[kx] - ets2;
                virxy   += ets2vf*mhx[kx]*mhy[kx];
                virxz   += ets2vf*mhx[kx]*mhz[kx];
                viryy   += ets2vf*mhy[kx]*mhy[kx] - ets2;
                viryz   += ets2vf*mhy[kx]*mhz[kx];
                virzz   += ets2vf*mhz[kx]*mhz[kx] - ets2;
            }
        }
        else
        {
            /* We don't need to calculate the energy and the virial.
             * In this case the triclinic overhead is small.
             */

            /* Two explicit loops to avoid a conditional inside the loop */

            for (kx = kxstart; kx < maxkx; kx++)
            {
                mx = kx;

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                denom[kx] = m2k*bz*by*pme->bsp_mod[XX][kx];
                tmp1[kx]  = -factor*m2k;
            }

            for (kx = maxkx; kx < kxend; kx++)
            {
                mx = (kx - nx);

                mhxk      = mx * rxx;
                mhyk      = mx * ryx + my * ryy;
                mhzk      = mx * rzx + my * rzy + mz * rzz;
                m2k       = mhxk*mhxk + mhyk*mhyk + mhzk*mhzk;
                denom[kx] = m2k*bz*by*pme->bsp_mod[XX][kx];
                tmp1[kx]  = -factor*m2k;
            }

            calc_exponentials(kxstart, kxend, elfac, denom, tmp1, eterm);

            for (kx = kxstart; kx < kxend; kx++, p0++)
            {
                d1      = p0->re;
                d2      = p0->im;

                p0->re  = d1*eterm[kx];
                p0->im  = d2*eterm[kx];
            }
        }
    }

    if (bEnerVir)
    {
        /* Update virial with local values.
         * The virial is symmetric by definition.
         * this virial seems ok for isotropic scaling, but I'm
         * experiencing problems on semiisotropic membranes.
         * IS THAT COMMENT STILL VALID??? (DvdS, 2001/02/07).
         */
        work->vir[XX][XX] = 0.25*virxx;
        work->vir[YY][YY] = 0.25*viryy;
        work->vir[ZZ][ZZ] = 0.25*virzz;
        work->vir[XX][YY] = work->vir[YY][XX] = 0.25*virxy;
        work->vir[XX][ZZ] = work->vir[ZZ][XX] = 0.25*virxz;
        work->vir[YY][ZZ] = work->vir[ZZ][YY] = 0.25*viryz;

        /* This energy should be corrected for a charged system */
        work->energy = 0.5*energy;
    }

    /* Return the loop count */
    return local_ndata[YY]*local_ndata[XX];
}

static void get_pme_ener_vir(const gmx_pme_t pme, int nthread,
                             real *mesh_energy, matrix vir)
{ // called
    /* This function sums output over threads
     * and should therefore only be called after thread synchronization.
     */
    int thread;

    *mesh_energy = pme->work[0].energy;
    copy_mat(pme->work[0].vir, vir);

    for (thread = 1; thread < nthread; thread++)
    {
        *mesh_energy += pme->work[thread].energy;
        m_add(vir, pme->work[thread].vir, vir);
    }
}

#define DO_FSPLINE(order)                      \
    for (ithx = 0; (ithx < order); ithx++)              \
    {                                              \
        index_x = (i0+ithx)*pny*pnz;               \
        tx      = thx[ithx];                       \
        dx      = dthx[ithx];                      \
                                               \
        for (ithy = 0; (ithy < order); ithy++)          \
        {                                          \
            index_xy = index_x+(j0+ithy)*pnz;      \
            ty       = thy[ithy];                  \
            dy       = dthy[ithy];                 \
            fxy1     = fz1 = 0;                    \
                                               \
            for (ithz = 0; (ithz < order); ithz++)      \
            {                                      \
                gval  = grid[index_xy+(k0+ithz)];  \
                fxy1 += thz[ithz]*gval;            \
                fz1  += dthz[ithz]*gval;           \
            }                                      \
            fx += dx*ty*fxy1;                      \
            fy += tx*dy*fxy1;                      \
            fz += tx*ty*fz1;                       \
        }                                          \
    }


static void gather_f_bsplines(gmx_pme_t pme, real *grid,
                              gmx_bool bClearF, pme_atomcomm_t *atc,
                              splinedata_t *spline,
                              real scale)
{ // called
    /* sum forces for local particles */
    int     nn, n, ithx, ithy, ithz, i0, j0, k0;
    int     index_x, index_xy;
    int     nx, ny, nz, pnx, pny, pnz;
    int *   idxptr;
    real    tx, ty, dx, dy, qn;
    real    fx, fy, fz, gval;
    real    fxy1, fz1;
    real    *thx, *thy, *thz, *dthx, *dthy, *dthz;
    int     norder;
    real    rxx, ryx, ryy, rzx, rzy, rzz;
    int     order;



    order = pme->pme_order;
    thx   = spline->theta[XX];
    thy   = spline->theta[YY];
    thz   = spline->theta[ZZ];
    dthx  = spline->dtheta[XX];
    dthy  = spline->dtheta[YY];
    dthz  = spline->dtheta[ZZ];
    nx    = pme->nkx;
    ny    = pme->nky;
    nz    = pme->nkz;
    pnx   = pme->pmegrid_nx;
    pny   = pme->pmegrid_ny;
    pnz   = pme->pmegrid_nz;

    rxx   = pme->recipbox[XX][XX];
    ryx   = pme->recipbox[YY][XX];
    ryy   = pme->recipbox[YY][YY];
    rzx   = pme->recipbox[ZZ][XX];
    rzy   = pme->recipbox[ZZ][YY];
    rzz   = pme->recipbox[ZZ][ZZ];

    for (nn = 0; nn < spline->n; nn++)
    {
        n  = spline->ind[nn];
        qn = scale*atc->q[n];

        if (bClearF)
        {
            atc->f[n][XX] = 0;
            atc->f[n][YY] = 0;
            atc->f[n][ZZ] = 0;
        }
        if (qn != 0)
        {
            fx     = 0;
            fy     = 0;
            fz     = 0;
            idxptr = atc->idx[n];
            norder = nn*order;

            i0   = idxptr[XX];
            j0   = idxptr[YY];
            k0   = idxptr[ZZ];

            /* Pointer arithmetic alert, next six statements */
            thx  = spline->theta[XX] + norder;
            thy  = spline->theta[YY] + norder;
            thz  = spline->theta[ZZ] + norder;
            dthx = spline->dtheta[XX] + norder;
            dthy = spline->dtheta[YY] + norder;
            dthz = spline->dtheta[ZZ] + norder;
	    DO_FSPLINE(order);

            atc->f[n][XX] += -qn*( fx*nx*rxx );
            atc->f[n][YY] += -qn*( fx*nx*ryx + fy*ny*ryy );
            atc->f[n][ZZ] += -qn*( fx*nx*rzx + fy*ny*rzy + fz*nz*rzz );
        }
    }
    /* Since the energy and not forces are interpolated
     * the net force might not be exactly zero.
     * This can be solved by also interpolating F, but
     * that comes at a cost.
     * A better hack is to remove the net force every
     * step, but that must be done at a higher level
     * since this routine doesn't see all atoms if running
     * in parallel. Don't know how important it is?  EL 990726
     */
}



/* Macro to force loop unrolling by fixing order.
 * This gives a significant performance gain.
 */
#define CALC_SPLINE(order)                     \
    {                                              \
        int j, k, l;                                 \
        real dr, div;                               \
        real data[PME_ORDER_MAX];                  \
        real ddata[PME_ORDER_MAX];                 \
                                               \
        for (j = 0; (j < DIM); j++)                     \
        {                                          \
            dr  = xptr[j];                         \
                                               \
            /* dr is relative offset from lower cell limit */ \
            data[order-1] = 0;                     \
            data[1]       = dr;                          \
            data[0]       = 1 - dr;                      \
                                               \
            for (k = 3; (k < order); k++)               \
            {                                      \
                div       = 1.0/(k - 1.0);               \
                data[k-1] = div*dr*data[k-2];      \
                for (l = 1; (l < (k-1)); l++)           \
                {                                  \
                    data[k-l-1] = div*((dr+l)*data[k-l-2]+(k-l-dr)* \
                                       data[k-l-1]);                \
                }                                  \
                data[0] = div*(1-dr)*data[0];      \
            }                                      \
            /* differentiate */                    \
            ddata[0] = -data[0];                   \
            for (k = 1; (k < order); k++)               \
            {                                      \
                ddata[k] = data[k-1] - data[k];    \
            }                                      \
                                               \
            div           = 1.0/(order - 1);                 \
            data[order-1] = div*dr*data[order-2];  \
            for (l = 1; (l < (order-1)); l++)           \
            {                                      \
                data[order-l-1] = div*((dr+l)*data[order-l-2]+    \
                                       (order-l-dr)*data[order-l-1]); \
            }                                      \
            data[0] = div*(1 - dr)*data[0];        \
                                               \
            for (k = 0; k < order; k++)                 \
            {                                      \
                theta[j][i*order+k]  = data[k];    \
                dtheta[j][i*order+k] = ddata[k];   \
            }                                      \
        }                                          \
    }

void make_bsplines(splinevec theta, splinevec dtheta, int order,
                   rvec fractx[], int nr, int ind[], real charge[],
                   gmx_bool bFreeEnergy)
{ // called
    /* construct splines for local atoms */
    int  i, ii;
    real *xptr;

    for (i = 0; i < nr; i++)
    {
        /* With free energy we do not use the charge check.
         * In most cases this will be more efficient than calling make_bsplines
         * twice, since usually more than half the particles have charges.
         */
        ii = ind[i];
        if (bFreeEnergy || charge[ii] != 0.0)
        {
            xptr = fractx[ii];
            switch (order)
            {
                case 4:  CALC_SPLINE(4);     break;
                case 5:  CALC_SPLINE(5);     break;
                default: CALC_SPLINE(order); break;
            }
        }
    }
}


void make_dft_mod(real *mod, real *data, int ndata)
{ //called
    int i, j;
    real sc, ss, arg;

    for (i = 0; i < ndata; i++)
    {
        sc = ss = 0;
        for (j = 0; j < ndata; j++)
        {
            arg = (2.0*M_PI*i*j)/ndata;
            sc += data[j]*cos(arg);
            ss += data[j]*sin(arg);
        }
        mod[i] = sc*sc+ss*ss;
    }
    for (i = 0; i < ndata; i++)
    {
        if (mod[i] < 1e-7)
        {
            mod[i] = (mod[i-1]+mod[i+1])*0.5;
        }
    }
}


static void make_bspline_moduli(splinevec bsp_mod,
                                int nx, int ny, int nz, int order)
{ // called
    int nmax = max(nx, max(ny, nz));
    real *data, *ddata, *bsp_data;
    int i, k, l;
    real div;

    snew(data, order);
    snew(ddata, order);
    snew(bsp_data, nmax);

    data[order-1] = 0;
    data[1]       = 0;
    data[0]       = 1;

    for (k = 3; k < order; k++)
    {
        div       = 1.0/(k-1.0);
        data[k-1] = 0;
        for (l = 1; l < (k-1); l++)
        {
            data[k-l-1] = div*(l*data[k-l-2]+(k-l)*data[k-l-1]);
        }
        data[0] = div*data[0];
    }
    /* differentiate */
    ddata[0] = -data[0];
    for (k = 1; k < order; k++)
    {
        ddata[k] = data[k-1]-data[k];
    }
    div           = 1.0/(order-1);
    data[order-1] = 0;
    for (l = 1; l < (order-1); l++)
    {
        data[order-l-1] = div*(l*data[order-l-2]+(order-l)*data[order-l-1]);
    }
    data[0] = div*data[0];

    for (i = 0; i < nmax; i++)
    {
        bsp_data[i] = 0;
    }
    for (i = 1; i <= order; i++)
    {
        bsp_data[i] = data[i-1];
    }

    make_dft_mod(bsp_mod[XX], bsp_data, nx);
    make_dft_mod(bsp_mod[YY], bsp_data, ny);
    make_dft_mod(bsp_mod[ZZ], bsp_data, nz);

    sfree(data);
    sfree(ddata);
    sfree(bsp_data);
}

static void init_atomcomm(gmx_pme_t pme, pme_atomcomm_t *atc, t_commrec *cr,
                          int dimind, gmx_bool bSpread)
{ // called 
    int nk, k, s, thread;

    atc->dimind    = dimind;
    atc->nslab     = 1;
    atc->nodeid    = 0;
    atc->pd_nalloc = 0;

    atc->bSpread   = bSpread; // TRUE
    atc->pme_order = pme->pme_order;


    atc->nthread = pme->nthread;
        snew(atc->thread_plist, atc->nthread);
    snew(atc->spline, atc->nthread);
    for (thread = 0; thread < atc->nthread; thread++)
    {
            snew(atc->thread_plist[thread].n, atc->nthread+2*GMX_CACHE_SEP);
            atc->thread_plist[thread].n += GMX_CACHE_SEP;
        snew(atc->spline[thread].thread_one, pme->nthread);
        atc->spline[thread].thread_one[thread] = 1;
    }
}

static void
init_overlap_comm(pme_overlap_t *  ol,
                  int              norder,
                  MPI_Comm         comm,
                  int              nnodes,
                  int              nodeid,
                  int              ndata,
                  int              commplainsize)
{ // called 
    int lbnd, rbnd, maxlr, b, i;
    int exten;
    int nn, nk;
    pme_grid_comm_t *pgc;
    gmx_bool bCont;
    int fft_start, fft_end, send_index1, recv_index1;
    MPI_Status stat;

    ol->mpi_comm = comm;

    ol->nnodes = nnodes;
    ol->nodeid = nodeid;

    /* Linear translation of the PME grid won't affect reciprocal space
     * calculations, so to optimize we only interpolate "upwards",
     * which also means we only have to consider overlap in one direction.
     * I.e., particles on this node might also be spread to grid indices
     * that belong to higher nodes (modulo nnodes)
     */

    snew(ol->s2g0, ol->nnodes+1);
    snew(ol->s2g1, ol->nnodes);
    for (i = 0; i < nnodes; i++)
    {
        /* s2g0 the local interpolation grid start.
         * s2g1 the local interpolation grid end.
         * Because grid overlap communication only goes forward,
         * the grid the slabs for fft's should be rounded down.
         */
        ol->s2g0[i] = ( i   *ndata + 0       )/nnodes;
        ol->s2g1[i] = ((i+1)*ndata + nnodes-1)/nnodes + norder - 1;

    }
    ol->s2g0[nnodes] = ndata;

    /* Determine with how many nodes we need to communicate the grid overlap */
    b = 0;
    b++;
    bCont = FALSE;
    if ((i+b <  nnodes && ol->s2g1[0] > ol->s2g0[b]) ||
        (i+b >= nnodes && ol->s2g1[0] > ol->s2g0[b-nnodes] + ndata))
    {
        bCont = TRUE;
    }
    ol->noverlap_nodes = b - 1;

    snew(ol->send_id, ol->noverlap_nodes);
    snew(ol->recv_id, ol->noverlap_nodes);
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        ol->send_id[b] = (ol->nodeid + (b + 1)) % ol->nnodes;
        ol->recv_id[b] = (ol->nodeid - (b + 1) + ol->nnodes) % ol->nnodes;
    }
    snew(ol->comm_data, ol->noverlap_nodes);

    ol->send_size = 0;
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        pgc = &ol->comm_data[b];
        /* Send */
        fft_start        = ol->s2g0[ol->send_id[b]];
        fft_end          = ol->s2g0[ol->send_id[b]+1];
        if (ol->send_id[b] < nodeid)
        {
            fft_start += ndata;
            fft_end   += ndata;
        }
        send_index1       = ol->s2g1[nodeid];
        send_index1       = min(send_index1, fft_end);
        pgc->send_index0  = fft_start;
        pgc->send_nindex  = max(0, send_index1 - pgc->send_index0);
        ol->send_size    += pgc->send_nindex;

        /* We always start receiving to the first index of our slab */
        fft_start        = ol->s2g0[ol->nodeid];
        fft_end          = ol->s2g0[ol->nodeid+1];
        recv_index1      = ol->s2g1[ol->recv_id[b]];
        if (ol->recv_id[b] > nodeid)
        {
            recv_index1 -= ndata;
        }
        recv_index1      = min(recv_index1, fft_end);
        pgc->recv_index0 = fft_start;
        pgc->recv_nindex = max(0, recv_index1 - pgc->recv_index0);
    }

    /* Communicate the buffer sizes to receive */
    for (b = 0; b < ol->noverlap_nodes; b++)
    {
        MPI_Sendrecv(&ol->send_size, 1, MPI_INT, ol->send_id[b], b,
                     &ol->comm_data[b].recv_size, 1, MPI_INT, ol->recv_id[b], b,
                     ol->mpi_comm, &stat);
    }

    /* For non-divisible grid we need pme_order iso pme_order-1 */
    snew(ol->sendbuf, norder*commplainsize);
    snew(ol->recvbuf, norder*commplainsize);
}

static void
make_gridindex5_to_localindex(int n, int local_start, int local_range,
                              int **global_to_local,
                              real **fraction_shift)
{
    int i;
    int * gtl;
    real * fsh;

    snew(gtl, 5*n);
    snew(fsh, 5*n);
    for (i = 0; (i < 5*n); i++)
    {
        /* Determine the global to local grid index */
        gtl[i] = (i - local_start + n) % n;
        /* For coordinates that fall within the local grid the fraction
         * is correct, we don't need to shift it.
         */
        fsh[i] = 0;
        if (local_range < n)
        {
            /* Due to rounding issues i could be 1 beyond the lower or
             * upper boundary of the local grid. Correct the index for this.
             * If we shift the index, we need to shift the fraction by
             * the same amount in the other direction to not affect
             * the weights.
             * Note that due to this shifting the weights at the end of
             * the spline might change, but that will only involve values
             * between zero and values close to the precision of a real,
             * which is anyhow the accuracy of the whole mesh calculation.
             */
            /* With local_range=0 we should not change i=local_start */
            if (i % n != local_start)
            {
                if (gtl[i] == n-1)
                {
                    gtl[i] = 0;
                    fsh[i] = -1;
                }
                else if (gtl[i] == local_range)
                {
                    gtl[i] = local_range - 1;
                    fsh[i] = 1;
                }
            }
        }
    }

    *global_to_local = gtl;
    *fraction_shift  = fsh;
}

int gmx_pme_init(gmx_pme_t *         pmedata,
                 t_commrec *         cr,
                 int                 nnodes_major, // 1
                 int                 nnodes_minor, // 1
                 t_inputrec *        ir,
                 int                 homenr, // 3000
                 gmx_bool            bFreeEnergy, // 0
                 gmx_bool            bReproducible, // 0
                 int                 nthread) //12
{ // called
    gmx_pme_t pme = NULL;

    pme_atomcomm_t *atc;
    ivec ndata;

    snew(pme, 1);

    pme->redist_init         = FALSE;
    pme->sum_qgrid_tmp       = NULL;
    pme->sum_qgrid_dd_tmp    = NULL;
    pme->buf_nalloc          = 0;
    pme->redist_buf_nalloc   = 0;

    pme->nnodes              = 1;
    pme->bPPnode             = TRUE;

    pme->nnodes_major        = nnodes_major;
    pme->nnodes_minor        = nnodes_minor;

    pme->mpi_comm = MPI_COMM_NULL;

    pme->mpi_comm_d[0] = MPI_COMM_NULL;
    pme->mpi_comm_d[1] = MPI_COMM_NULL;
    pme->ndecompdim   = 0;
    pme->nodeid_major = 0;
    pme->nodeid_minor = 0;
    pme->mpi_comm_d[0] = pme->mpi_comm_d[1] = MPI_COMM_NULL;

    pme->nthread = nthread;


    pme->bFEP        = FALSE;
    pme->nkx         = ir->nkx;
    pme->nky         = ir->nky;
    pme->nkz         = ir->nkz;
    pme->bP3M        = (ir->coulombtype == eelP3M_AD || getenv("GMX_PME_P3M") != NULL);
    pme->pme_order   = ir->pme_order;
    pme->epsilon_r   = ir->epsilon_r;


    /* For non-divisible grid we need pme_order iso pme_order-1 */
    /* In sum_qgrid_dd x overlap is copied in place: take padding into account.
     * y is always copied through a buffer: we don't need padding in z,
     * but we do need the overlap in x because of the communication order.
     */
    init_overlap_comm(&pme->overlap[0], pme->pme_order,
                      pme->mpi_comm_d[0],
                      pme->nnodes_major, pme->nodeid_major,
                      pme->nkx,
                      (div_round_up(pme->nky, pme->nnodes_minor)+pme->pme_order)*(pme->nkz+pme->pme_order-1));

    /* Along overlap dim 1 we can send in multiple pulses in sum_fftgrid_dd.
     * We do this with an offset buffer of equal size, so we need to allocate
     * extra for the offset. That's what the (+1)*pme->nkz is for.
     */
    init_overlap_comm(&pme->overlap[1], pme->pme_order,
                      pme->mpi_comm_d[1],
                      pme->nnodes_minor, pme->nodeid_minor,
                      pme->nky,
                      (div_round_up(pme->nkx, pme->nnodes_major)+pme->pme_order+1)*pme->nkz);


    snew(pme->bsp_mod[XX], pme->nkx);
    snew(pme->bsp_mod[YY], pme->nky);
    snew(pme->bsp_mod[ZZ], pme->nkz);

    /* The required size of the interpolation grid, including overlap.
     * The allocated size (pmegrid_n?) might be slightly larger.
     */
    pme->pmegrid_nx = pme->overlap[0].s2g1[pme->nodeid_major] -
        pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_ny = pme->overlap[1].s2g1[pme->nodeid_minor] -
        pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_nz_base = pme->nkz;
    pme->pmegrid_nz      = pme->pmegrid_nz_base + pme->pme_order - 1;

    pme->pmegrid_start_ix = pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_start_iy = pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_start_iz = 0;

    make_gridindex5_to_localindex(pme->nkx,
                                  pme->pmegrid_start_ix,
                                  pme->pmegrid_nx - (pme->pme_order-1),
                                  &pme->nnx, &pme->fshx);
    make_gridindex5_to_localindex(pme->nky,
                                  pme->pmegrid_start_iy,
                                  pme->pmegrid_ny - (pme->pme_order-1),
                                  &pme->nny, &pme->fshy);
    make_gridindex5_to_localindex(pme->nkz,
                                  pme->pmegrid_start_iz,
                                  pme->pmegrid_nz_base,
                                  &pme->nnz, &pme->fshz);

    pmegrids_init(&pme->pmegridA,
                  pme->pmegrid_nx, pme->pmegrid_ny, pme->pmegrid_nz,
                  pme->pmegrid_nz_base,
                  pme->pme_order,
                  pme->nthread,
                  pme->overlap[0].s2g1[pme->nodeid_major]-pme->overlap[0].s2g0[pme->nodeid_major+1],
                  pme->overlap[1].s2g1[pme->nodeid_minor]-pme->overlap[1].s2g0[pme->nodeid_minor+1]);


    ndata[0] = pme->nkx;
    ndata[1] = pme->nky;
    ndata[2] = pme->nkz;

    /* This routine will allocate the grid data to fit the FFTs */
    gmx_parallel_3dfft_init(&pme->pfft_setupA, ndata,
                            &pme->fftgridA, &pme->cfftgridA,
                            pme->mpi_comm_d,
                            pme->overlap[0].s2g0, pme->overlap[1].s2g0,
                            bReproducible, pme->nthread);

    pme->pmegridB.grid.grid = NULL;
    pme->fftgridB           = NULL;
    pme->cfftgridB          = NULL;

    /* Use plain SPME B-spline interpolation */
    make_bspline_moduli(pme->bsp_mod, pme->nkx, pme->nky, pme->nkz, pme->pme_order);

    /* Use atc[0] for spreading */
    init_atomcomm(pme, &pme->atc[0], cr, nnodes_major > 1 ? 0 : 1, TRUE);

    pme->atc[0].n = homenr;
    pme_realloc_atomcomm_things(&pme->atc[0]);

    {
        int thread;

        /* Use fft5d, order after FFT is y major, z, x minor */

        snew(pme->work, pme->nthread);
        for (thread = 0; thread < pme->nthread; thread++)
        {
            realloc_work(&pme->work[thread], pme->nkx);
        }
    }

    *pmedata = pme;

    return 0;
}



static void copy_local_grid(gmx_pme_t pme,
                            pmegrids_t *pmegrids, int thread, real *fftgrid)
{ // called 
    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    int  fft_my, fft_mz;
    int  nsx, nsy, nsz;
    ivec nf;
    int  offx, offy, offz, x, y, z, i0, i0t;
    int  d;
    pmegrid_t *pmegrid;
    real *grid_th;

    gmx_parallel_3dfft_real_limits(pme->pfft_setupA,
                                   local_fft_ndata,
                                   local_fft_offset,
                                   local_fft_size);
    fft_my = local_fft_size[YY];
    fft_mz = local_fft_size[ZZ];

    pmegrid = &pmegrids->grid_th[thread];

    nsx = pmegrid->s[XX];
    nsy = pmegrid->s[YY];
    nsz = pmegrid->s[ZZ];

    for (d = 0; d < DIM; d++)
    {
        nf[d] = min(pmegrid->n[d] - (pmegrid->order - 1),
                    local_fft_ndata[d] - pmegrid->offset[d]);
    }

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];

    /* Directly copy the non-overlapping parts of the local grids.
     * This also initializes the full grid.
     */
    grid_th = pmegrid->grid;
    for (x = 0; x < nf[XX]; x++)
    {
        for (y = 0; y < nf[YY]; y++)
        {
            i0  = ((offx + x)*fft_my + (offy + y))*fft_mz + offz;
            i0t = (x*nsy + y)*nsz;
            for (z = 0; z < nf[ZZ]; z++)
            {
                fftgrid[i0+z] = grid_th[i0t+z];
            }
        }
    }
}

static void
reduce_threadgrid_overlap(gmx_pme_t pme,
                          const pmegrids_t *pmegrids, int thread,
                          real *fftgrid, real *commbuf_x, real *commbuf_y)
{ // called 
    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    int  fft_nx, fft_ny, fft_nz;
    int  fft_my, fft_mz;
    int  buf_my = -1;
    int  nsx, nsy, nsz;
    ivec ne;
    int  offx, offy, offz, x, y, z, i0, i0t;
    int  sx, sy, sz, fx, fy, fz, tx1, ty1, tz1, ox, oy, oz;
    gmx_bool bClearBufX, bClearBufY, bClearBufXY, bClearBuf;
    gmx_bool bCommX, bCommY;
    int  d;
    int  thread_f;
    const pmegrid_t *pmegrid, *pmegrid_g, *pmegrid_f;
    const real *grid_th;
    real *commbuf = NULL;

    gmx_parallel_3dfft_real_limits(pme->pfft_setupA,
                                   local_fft_ndata,
                                   local_fft_offset,
                                   local_fft_size);
    fft_nx = local_fft_ndata[XX];
    fft_ny = local_fft_ndata[YY];
    fft_nz = local_fft_ndata[ZZ];

    fft_my = local_fft_size[YY];
    fft_mz = local_fft_size[ZZ];

    /* This routine is called when all thread have finished spreading.
     * Here each thread sums grid contributions calculated by other threads
     * to the thread local grid volume.
     * To minimize the number of grid copying operations,
     * this routines sums immediately from the pmegrid to the fftgrid.
     */

    /* Determine which part of the full node grid we should operate on,
     * this is our thread local part of the full grid.
     */
    pmegrid = &pmegrids->grid_th[thread];

    for (d = 0; d < DIM; d++)
    {
        ne[d] = min(pmegrid->offset[d]+pmegrid->n[d]-(pmegrid->order-1),
                    local_fft_ndata[d]);
    }

    offx = pmegrid->offset[XX];
    offy = pmegrid->offset[YY];
    offz = pmegrid->offset[ZZ];


    bClearBufX  = TRUE;
    bClearBufY  = TRUE;
    bClearBufXY = TRUE;

    /* Now loop over all the thread data blocks that contribute
     * to the grid region we (our thread) are operating on.
     */
    /* Note that ffy_nx/y is equal to the number of grid points
     * between the first point of our node grid and the one of the next node.
     */
    for (sx = 0; sx >= -pmegrids->nthread_comm[XX]; sx--)
    {
        fx     = pmegrid->ci[XX] + sx;
        ox     = 0;
        bCommX = FALSE;
        if (fx < 0)
        {
            fx    += pmegrids->nc[XX];
            ox    -= fft_nx;
            bCommX = (pme->nnodes_major > 1);
        }
        pmegrid_g = &pmegrids->grid_th[fx*pmegrids->nc[YY]*pmegrids->nc[ZZ]];
        ox       += pmegrid_g->offset[XX];
        if (!bCommX)
        {
            tx1 = min(ox + pmegrid_g->n[XX], ne[XX]);
        }
        else
        {
            tx1 = min(ox + pmegrid_g->n[XX], pme->pme_order);
        }

        for (sy = 0; sy >= -pmegrids->nthread_comm[YY]; sy--)
        {
            fy     = pmegrid->ci[YY] + sy;
            oy     = 0;
            bCommY = FALSE;
            if (fy < 0)
            {
                fy    += pmegrids->nc[YY];
                oy    -= fft_ny;
                bCommY = FALSE;
            }
            pmegrid_g = &pmegrids->grid_th[fy*pmegrids->nc[ZZ]];
            oy       += pmegrid_g->offset[YY];
            if (!bCommY)
            {
                ty1 = min(oy + pmegrid_g->n[YY], ne[YY]);
            }
            else
            {
                ty1 = min(oy + pmegrid_g->n[YY], pme->pme_order);
            }

            for (sz = 0; sz >= -pmegrids->nthread_comm[ZZ]; sz--)
            {
                fz = pmegrid->ci[ZZ] + sz;
                oz = 0;
                if (fz < 0)
                {
                    fz += pmegrids->nc[ZZ];
                    oz -= fft_nz;
                }
                pmegrid_g = &pmegrids->grid_th[fz];
                oz       += pmegrid_g->offset[ZZ];
                tz1       = min(oz + pmegrid_g->n[ZZ], ne[ZZ]);

                if (sx == 0 && sy == 0 && sz == 0)
                {
                    /* We have already added our local contribution
                     * before calling this routine, so skip it here.
                     */
                    continue;
                }

                thread_f = (fx*pmegrids->nc[YY] + fy)*pmegrids->nc[ZZ] + fz;

                pmegrid_f = &pmegrids->grid_th[thread_f];

                grid_th = pmegrid_f->grid;

                nsx = pmegrid_f->s[XX];
                nsy = pmegrid_f->s[YY];
                nsz = pmegrid_f->s[ZZ];


                if (!(bCommX || bCommY))
                {
                    /* Copy from the thread local grid to the node grid */
                    for (x = offx; x < tx1; x++)
                    {
                        for (y = offy; y < ty1; y++)
                        {
                            i0  = (x*fft_my + y)*fft_mz;
                            i0t = ((x - ox)*nsy + (y - oy))*nsz - oz;
                            for (z = offz; z < tz1; z++)
                            {
                                fftgrid[i0+z] += grid_th[i0t+z];
                            }
                        }
                    }
                }
                else
                {
                    /* The order of this conditional decides
                     * where the corner volume gets stored with x+y decomp.
                     */
                    if (bCommY)
                    {
                        commbuf = commbuf_y;
                        buf_my  = ty1 - offy;
                        if (bCommX)
                        {
                            /* We index commbuf modulo the local grid size */
                            commbuf += buf_my*fft_nx*fft_nz;

                            bClearBuf   = bClearBufXY;
                            bClearBufXY = FALSE;
                        }
                        else
                        {
                            bClearBuf  = bClearBufY;
                            bClearBufY = FALSE;
                        }
                    }
                    else
                    {
                        commbuf    = commbuf_x;
                        buf_my     = fft_ny;
                        bClearBuf  = bClearBufX;
                        bClearBufX = FALSE;
                    }

                    /* Copy to the communication buffer */
                    for (x = offx; x < tx1; x++)
                    {
                        for (y = offy; y < ty1; y++)
                        {
                            i0  = (x*buf_my + y)*fft_nz;
                            i0t = ((x - ox)*nsy + (y - oy))*nsz - oz;

                            if (bClearBuf)
                            {
                                /* First access of commbuf, initialize it */
                                for (z = offz; z < tz1; z++)
                                {
                                    commbuf[i0+z]  = grid_th[i0t+z];
                                }
                            }
                            else
                            {
                                for (z = offz; z < tz1; z++)
                                {
                                    commbuf[i0+z] += grid_th[i0t+z];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


static void sum_fftgrid_dd(gmx_pme_t pme, real *fftgrid)
{
    ivec local_fft_ndata, local_fft_offset, local_fft_size;
    pme_overlap_t *overlap;
    int  send_index0, send_nindex;
    int  recv_nindex;
    MPI_Status stat;
    int  send_size_y, recv_size_y;
    int  ipulse, send_id, recv_id, datasize, gridsize, size_yx;
    real *sendptr, *recvptr;
    int  x, y, z, indg, indb;

    /* Note that this routine is only used for forward communication.
     * Since the force gathering, unlike the charge spreading,
     * can be trivially parallelized over the particles,
     * the backwards process is much simpler and can use the "old"
     * communication setup.
     */

    gmx_parallel_3dfft_real_limits(pme->pfft_setupA,
                                   local_fft_ndata,
                                   local_fft_offset,
                                   local_fft_size);

}


static void spread_on_grid(gmx_pme_t pme,
                           pme_atomcomm_t *atc, pmegrids_t *grids,
                           gmx_bool bCalcSplines, gmx_bool bSpread,
                           real *fftgrid)
{ // called 
    int nthread, thread;

    nthread = pme->nthread;
    assert(nthread > 0);

    if (bCalcSplines)
    {
#pragma omp parallel for num_threads(nthread) schedule(static)
        for (thread = 0; thread < nthread; thread++)
        {
            int start, end;

            start = atc->n* thread   /nthread;
            end   = atc->n*(thread+1)/nthread;

            /* Compute fftgrid index for all atoms,
             * with help of some extra variables.
             */
            calc_interpolation_idx(pme, atc, start, end, thread);
        }
    }

#pragma omp parallel for num_threads(nthread) schedule(static)
    for (thread = 0; thread < nthread; thread++)
    {
        splinedata_t *spline;
        pmegrid_t *grid;

        /* make local bsplines  */
        spline = &atc->spline[thread];

        make_thread_local_ind(atc, thread, spline);

        grid = &grids->grid_th[thread];
        make_bsplines(spline->theta, spline->dtheta, pme->pme_order,
                          atc->fractx, spline->n, spline->ind, atc->q, pme->bFEP);


        /* put local atoms on grid. */
        spread_q_bsplines_thread(grid, atc, spline);

        copy_local_grid(pme, grids, thread, fftgrid);
    }

#pragma omp parallel for num_threads(grids->nthread) schedule(static)
        for (thread = 0; thread < grids->nthread; thread++) // 12
        {
            reduce_threadgrid_overlap(pme, grids, thread,
                                      fftgrid,
                                      pme->overlap[0].sendbuf,
                                      pme->overlap[1].sendbuf);
        }


}

int gmx_pme_do(gmx_pme_t pme,
               int start,       int homenr,
               rvec x[],        rvec f[],
               real *chargeA,   real *chargeB,
               matrix box, t_commrec *cr,
               int  maxshift_x, int maxshift_y,
               matrix vir,      real ewaldcoeff,
               real *energy,    real lambda,
               real *dvdlambda, int flags)
{ // called
    int     q, d, i, j, ntot, npme;
    int     nx, ny, nz;
    int     n_d, local_ny;
    pme_atomcomm_t *atc = NULL;
    pmegrids_t *pmegrid = NULL;
    real    *grid       = NULL;
    real    *ptr;
    rvec    *x_d, *f_d;
    real    *charge = NULL, *q_d;
    real    energy_AB[2];
    matrix  vir_AB[2];
    gmx_bool bClearF;
    gmx_parallel_3dfft_t pfft_setup;
    real *  fftgrid;
    t_complex * cfftgrid;
    int     thread;
    const gmx_bool bCalcEnerVir = flags & GMX_PME_CALC_ENER_VIR;
    const gmx_bool bCalcF       = flags & GMX_PME_CALC_F;


    /* This could be necessary for TPI */
    pme->atc[0].n = homenr;

    for (q = 0; q < (pme->bFEP ? 2 : 1); q++)
    {
        if (q == 0)
        {
            pmegrid    = &pme->pmegridA;
            fftgrid    = pme->fftgridA;
            cfftgrid   = pme->cfftgridA;
            pfft_setup = pme->pfft_setupA;
            charge     = chargeA+start;
        }
        else
        {
            pmegrid    = &pme->pmegridB;
            fftgrid    = pme->fftgridB;
            cfftgrid   = pme->cfftgridB;
            pfft_setup = pme->pfft_setupB;
            charge     = chargeB+start;
        }
        grid = pmegrid->grid.grid;
        /* Unpack structure */
        where();

        m_inv_ur0(box, pme->recipbox);

            atc = &pme->atc[0];
            if (DOMAINDECOMP(cr))
            {
                atc->n = homenr;
                pme_realloc_atomcomm_things(atc);
            }
            atc->x = x;
            atc->q = charge;
            atc->f = f;

        if (flags & GMX_PME_SPREAD_Q)
        {

            /* Spread the charges on a grid */

            /* Spread the charges on a grid */
            spread_on_grid(pme, &pme->atc[0], pmegrid, q == 0, TRUE, fftgrid);
        }

        /* Here we start a large thread parallel region */
#pragma omp parallel num_threads(pme->nthread) private(thread)
        {
            thread = gmx_omp_get_thread_num();
            if (flags & GMX_PME_SOLVE)
            {
                int loop_count;

                gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_REAL_TO_COMPLEX,
                                           fftgrid, cfftgrid, thread);
                where();

                /* solve in k-space for our local cells */
                loop_count =
                    solve_pme_yzx(pme, cfftgrid, ewaldcoeff,
                                  box[XX][XX]*box[YY][YY]*box[ZZ][ZZ],
                                  bCalcEnerVir,
                                  pme->nthread, thread);
            }

            if (bCalcF)
            {
                /* do 3d-invfft */
                gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_COMPLEX_TO_REAL,
                                           cfftgrid, fftgrid, thread);
                if (thread == 0)
                {

                    where();

                    if (pme->nodeid == 0)
                    {
                        ntot  = pme->nkx*pme->nky*pme->nkz;
                        npme  = ntot*log((real)ntot)/log(2.0);
                    }

                }

                copy_fftgrid_to_pmegrid(pme, fftgrid, grid, pme->nthread, thread);
            }
        }
        /* End of thread parallel section.
         * With MPI we have to synchronize here before gmx_sum_qgrid_dd.
         */

        if (bCalcF)
        {
            /* distribute local grid to all nodes */
            where();

            unwrap_periodic_pmegrid(pme, grid);


            where();

            /* If we are running without parallelization,
             * atc->f is the actual force array, not a buffer,
             * therefore we should not clear it.
             */
            bClearF = (q == 0 && PAR(cr));
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
            for (thread = 0; thread < pme->nthread; thread++)
            {
                gather_f_bsplines(pme, grid, bClearF, atc,
                                  &atc->spline[thread],
                                  pme->bFEP ? (q == 0 ? 1.0-lambda : lambda) : 1.0);
            }

            where();


        }

        if (bCalcEnerVir)
        {
            /* This should only be called on the master thread
             * and after the threads have synchronized.
             */
            get_pme_ener_vir(pme, pme->nthread, &energy_AB[q], vir_AB[q]);
        }
    } /* of q-loop */
    where();

    if (bCalcEnerVir)
    {
        if (!pme->bFEP)
        {
            *energy = energy_AB[0];
            m_add(vir, vir_AB[0], vir);
        }
        else
        {
            *energy     = (1.0-lambda)*energy_AB[0] + lambda*energy_AB[1];
            *dvdlambda += energy_AB[1] - energy_AB[0];
            for (i = 0; i < DIM; i++)
            {
                for (j = 0; j < DIM; j++)
                {
                    vir[i][j] += (1.0-lambda)*vir_AB[0][i][j] +
                        lambda*vir_AB[1][i][j];
                }
            }
        }
    }
    else
    {
        *energy = 0;
    }


    return 0;
}
