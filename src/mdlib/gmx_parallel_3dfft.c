#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#ifdef GMX_LIB_MPI
#include <mpi.h>
#endif
#ifdef GMX_THREAD_MPI
#include "tmpi.h"
#endif

#include "smalloc.h"
#include "gmx_parallel_3dfft.h"
#include "gmx_fft.h"
#include "gmxcomplex.h"
#include "gmx_fatal.h"

#include "fft5d.h"

struct gmx_parallel_3dfft  {
    fft5d_plan p1, p2;
};

int
gmx_parallel_3dfft_init   (gmx_parallel_3dfft_t     *    pfft_setup,
                           ivec                          ndata,
                           real     **                   real_data,
                           t_complex     **              complex_data,
                           MPI_Comm                      comm[2],
                           int     *                     slab2index_major,
                           int     *                     slab2index_minor,
                           gmx_bool                      bReproducible,
                           int                           nthreads)
{ // called 
    int        rN      = ndata[2], M = ndata[1], K = ndata[0];
    int        flags   = FFT5D_REALCOMPLEX | FFT5D_ORDER_YZ; /* FFT5D_DEBUG */
    MPI_Comm   rcomm[] = {comm[1], comm[0]};
    int        Nb, Mb, Kb;                                   /* dimension for backtransform (in starting order) */
    t_complex *buf1, *buf2;                                  /*intermediate buffers - used internally.*/

    snew(*pfft_setup, 1);

    Nb = K; Mb = rN; Kb = M;  /* currently always true because ORDER_YZ always set */

    (*pfft_setup)->p1 = fft5d_plan_3d(rN, M, K, rcomm, flags, (t_complex**)real_data, complex_data, &buf1, &buf2, nthreads);

    (*pfft_setup)->p2 = fft5d_plan_3d(Nb, Mb, Kb, rcomm,
                                      (flags|FFT5D_BACKWARD|FFT5D_NOMALLOC)^FFT5D_ORDER_YZ, complex_data, (t_complex**)real_data, &buf1, &buf2, nthreads);

    return (*pfft_setup)->p1 != 0 && (*pfft_setup)->p2 != 0;
}


static int
fft5d_limits(fft5d_plan                p,
             ivec                      local_ndata,
             ivec                      local_offset,
             ivec                      local_size)
{// called  
    int N1, M0, K0, K1, *coor;
    fft5d_local_size(p, &N1, &M0, &K0, &K1, &coor);  /* M0=MG/P[0], K1=KG/P[1], NG,MG,KG global sizes */

    local_offset[2] = 0;
    local_offset[1] = p->oM[0];  /*=p->coor[0]*p->MG/p->P[0]; */
    local_offset[0] = p->oK[0];  /*=p->coor[1]*p->KG/p->P[1]; */

    local_ndata[2] = p->rC[0];
    local_ndata[1] = p->pM[0];
    local_ndata[0] = p->pK[0];


    if ((!(p->flags&FFT5D_BACKWARD)) && (p->flags&FFT5D_REALCOMPLEX)) // System runs in True, Then runs in false.
    {
        local_size[2] = p->C[0]*2;
    }
    else
    {
        local_size[2] = p->C[0];
    }
    local_size[1] = p->pM[0];
    local_size[0] = p->pK[0];
    return 0;
}

int
gmx_parallel_3dfft_real_limits(gmx_parallel_3dfft_t      pfft_setup,
                               ivec                      local_ndata,
                               ivec                      local_offset,
                               ivec                      local_size)
{// called  
    return fft5d_limits(pfft_setup->p1, local_ndata, local_offset, local_size);
}

static void reorder_ivec_yzx(ivec v)
{ // called  
    real tmp;

    tmp   = v[0];
    v[XX] = v[2];
    v[ZZ] = v[1];
    v[YY] = tmp;
}

int
gmx_parallel_3dfft_complex_limits(gmx_parallel_3dfft_t      pfft_setup,
                                  ivec                      complex_order,
                                  ivec                      local_ndata,
                                  ivec                      local_offset,
                                  ivec                      local_size)
{ // called  
    int ret;

    /* For now everything is in-order, but prepare to save communication by avoiding transposes */
    complex_order[0] = 0;
    complex_order[1] = 1;
    complex_order[2] = 2;

    ret = fft5d_limits(pfft_setup->p2, local_ndata, local_offset, local_size);

    reorder_ivec_yzx(local_ndata);
    reorder_ivec_yzx(local_offset);
    reorder_ivec_yzx(local_size);

    return ret;
}


int
gmx_parallel_3dfft_execute(gmx_parallel_3dfft_t    pfft_setup,
                           enum gmx_fft_direction  dir,
                           void *                  in_data,
                           void *                  out_data,
                           int                     thread)
{// called  


    if (dir == GMX_FFT_FORWARD || dir == GMX_FFT_REAL_TO_COMPLEX) // TRUE then FALSE
    {
        fft5d_execute(pfft_setup->p1, thread);
    }
    else
    {
        fft5d_execute(pfft_setup->p2, thread);
    }
    return 0;
}

