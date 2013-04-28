// CALLED BY PME ONLY
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GMX_PARALLEL_ENV_INITIALIZED 1

#include "tmpi.h"

/* TODO: Do we still need this? Are we still planning ot use fftw + OpenMP? */
#define FFT5D_THREADS
/* requires fftw compiled with openmp */
/* #define FFT5D_FFTW_THREADS (now set by cmake) */

#include "fft5d.h"
#include <float.h>
#include <math.h>
#include <assert.h>
#include "smalloc.h"
#include "gmx_fatal.h"

/* none of the fftw3 calls, except execute(), are thread-safe, so
   we need to serialize them with this mutex. */
static tMPI_Thread_mutex_t big_fftw_mutex = TMPI_THREAD_MUTEX_INITIALIZER;

#define FFTW_LOCK tMPI_Thread_mutex_lock(&big_fftw_mutex)
#define FFTW_UNLOCK tMPI_Thread_mutex_unlock(&big_fftw_mutex)

static double fft5d_fmax(double a, double b)
{
    return (a > b) ? a : b;
}
static int vmax(int* a, int s)
{// called 
    int i, max = 0;
    for (i = 0; i < s; i++)
    {
        if (a[i] > max)
        {
            max = a[i];
        }
    }
    return max;
}


/* NxMxK the size of the data
 * comm communicator to use for fft5d
 * P0 number of processor in 1st axes (can be null for automatic)
 * lin is allocated by fft5d because size of array is only known after planning phase
 * rlout2 is only used as intermediate buffer - only returned after allocation to reuse for back transform - should not be used by caller
 */
fft5d_plan fft5d_plan_3d(int NG, int MG, int KG, MPI_Comm comm[2], int flags, t_complex** rlin, t_complex** rlout, t_complex** rlout2, t_complex** rlout3, int nthreads)
{// called 

    int        P[2], bMaster, prank[2], i, t;
    int        rNG, rMG, rKG;
    int       *N0 = 0, *N1 = 0, *M0 = 0, *M1 = 0, *K0 = 0, *K1 = 0, *oN0 = 0, *oN1 = 0, *oM0 = 0, *oM1 = 0, *oK0 = 0, *oK1 = 0;
    int        N[3], M[3], K[3], pN[3], pM[3], pK[3], oM[3], oK[3], *iNin[3] = {0}, *oNin[3] = {0}, *iNout[3] = {0}, *oNout[3] = {0};
    int        C[3], rC[3], nP[2];
    int        lsize;
    t_complex *lin = 0, *lout = 0, *lout2 = 0, *lout3 = 0;
    fft5d_plan plan;
    int        s;

    P[0]     = 1;
    prank[0] = 0;
    P[1]     = 1;
    prank[1] = 0;

    bMaster = (prank[0] == 0 && prank[1] == 0);


    rNG = NG; rMG = MG; rKG = KG;
    if (!(flags&FFT5D_BACKWARD)) // first call
    {
        NG = NG/2+1;
    }
    else // second call
    {
        MG = MG/2+1;
    }


    /*for transpose we need to know the size for each processor not only our own size*/

    N0  = (int*)malloc(P[0]*sizeof(int)); N1 = (int*)malloc(P[1]*sizeof(int));
    M0  = (int*)malloc(P[0]*sizeof(int)); M1 = (int*)malloc(P[1]*sizeof(int));
    K0  = (int*)malloc(P[0]*sizeof(int)); K1 = (int*)malloc(P[1]*sizeof(int));
    oN0 = (int*)malloc(P[0]*sizeof(int)); oN1 = (int*)malloc(P[1]*sizeof(int));
    oM0 = (int*)malloc(P[0]*sizeof(int)); oM1 = (int*)malloc(P[1]*sizeof(int));
    oK0 = (int*)malloc(P[0]*sizeof(int)); oK1 = (int*)malloc(P[1]*sizeof(int));

    for (i = 0; i < P[0]; i++)
    {
        #define EVENDIST
        #ifndef EVENDIST
        oN0[i] = i*ceil((double)NG/P[0]);
        oM0[i] = i*ceil((double)MG/P[0]);
        oK0[i] = i*ceil((double)KG/P[0]);
        #else
        oN0[i] = (NG*i)/P[0];
        oM0[i] = (MG*i)/P[0];
        oK0[i] = (KG*i)/P[0];
        #endif
    }
    for (i = 0; i < P[1]; i++)
    {
        #ifndef EVENDIST
        oN1[i] = i*ceil((double)NG/P[1]);
        oM1[i] = i*ceil((double)MG/P[1]);
        oK1[i] = i*ceil((double)KG/P[1]);
        #else
        oN1[i] = (NG*i)/P[1];
        oM1[i] = (MG*i)/P[1];
        oK1[i] = (KG*i)/P[1];
        #endif
    }
    for (i = 0; i < P[0]-1; i++)
    {
        N0[i] = oN0[i+1]-oN0[i];
        M0[i] = oM0[i+1]-oM0[i];
        K0[i] = oK0[i+1]-oK0[i];
    }
    N0[P[0]-1] = NG-oN0[P[0]-1];
    M0[P[0]-1] = MG-oM0[P[0]-1];
    K0[P[0]-1] = KG-oK0[P[0]-1];
    for (i = 0; i < P[1]-1; i++)
    {
        N1[i] = oN1[i+1]-oN1[i];
        M1[i] = oM1[i+1]-oM1[i];
        K1[i] = oK1[i+1]-oK1[i];
    }
    N1[P[1]-1] = NG-oN1[P[1]-1];
    M1[P[1]-1] = MG-oM1[P[1]-1];
    K1[P[1]-1] = KG-oK1[P[1]-1];

    /* for step 1-3 the local N,M,K sizes of the transposed system
       C: contiguous dimension, and nP: number of processor in subcommunicator
       for that step */


    pM[0] = M0[prank[0]];
    oM[0] = oM0[prank[0]];
    pK[0] = K1[prank[1]];
    oK[0] = oK1[prank[1]];
    C[0]  = NG;
    rC[0] = rNG;
    if (!(flags&FFT5D_ORDER_YZ)) // second call 
    {
        N[0]     = vmax(N1, P[1]);
        M[0]     = M0[prank[0]];
        K[0]     = vmax(K1, P[1]);
        pN[0]    = N1[prank[1]];
        iNout[0] = N1;
        oNout[0] = oN1;
        nP[0]    = P[1];
        C[1]     = KG;
        rC[1]    = rKG;
        N[1]     = vmax(K0, P[0]);
        pN[1]    = K0[prank[0]];
        iNin[1]  = K1;
        oNin[1]  = oK1;
        iNout[1] = K0;
        oNout[1] = oK0;
        M[1]     = vmax(M0, P[0]);
        pM[1]    = M0[prank[0]];
        oM[1]    = oM0[prank[0]];
        K[1]     = N1[prank[1]];
        pK[1]    = N1[prank[1]];
        oK[1]    = oN1[prank[1]];
        nP[1]    = P[0];
        C[2]     = MG;
        rC[2]    = rMG;
        iNin[2]  = M0;
        oNin[2]  = oM0;
        M[2]     = vmax(K0, P[0]);
        pM[2]    = K0[prank[0]];
        oM[2]    = oK0[prank[0]];
        K[2]     = vmax(N1, P[1]);
        pK[2]    = N1[prank[1]];
        oK[2]    = oN1[prank[1]];
        free(N0); free(oN0); /*these are not used for this order*/
        free(M1); free(oM1); /*the rest is freed in destroy*/
    }
    else // first call
    {
        N[0]     = vmax(N0, P[0]);
        M[0]     = vmax(M0, P[0]);
        K[0]     = K1[prank[1]];
        pN[0]    = N0[prank[0]];
        iNout[0] = N0;
        oNout[0] = oN0;
        nP[0]    = P[0];
        C[1]     = MG;
        rC[1]    = rMG;
        N[1]     = vmax(M1, P[1]);
        pN[1]    = M1[prank[1]];
        iNin[1]  = M0;
        oNin[1]  = oM0;
        iNout[1] = M1;
        oNout[1] = oM1;
        M[1]     = N0[prank[0]];
        pM[1]    = N0[prank[0]];
        oM[1]    = oN0[prank[0]];
        K[1]     = vmax(K1, P[1]);
        pK[1]    = K1[prank[1]];
        oK[1]    = oK1[prank[1]];
        nP[1]    = P[1];
        C[2]     = KG;
        rC[2]    = rKG;
        iNin[2]  = K1;
        oNin[2]  = oK1;
        M[2]     = vmax(N0, P[0]);
        pM[2]    = N0[prank[0]];
        oM[2]    = oN0[prank[0]];
        K[2]     = vmax(M1, P[1]);
        pK[2]    = M1[prank[1]];
        oK[2]    = oM1[prank[1]];
        free(N1); free(oN1); /*these are not used for this order*/
        free(K0); free(oK0); /*the rest is freed in destroy*/
    }
    N[2] = pN[2] = -1;       /*not used*/

    /*
       Difference between x-y-z regarding 2d decomposition is whether they are
       distributed along axis 1, 2 or both
     */

    /* int lsize = fmax(N[0]*M[0]*K[0]*nP[0],N[1]*M[1]*K[1]*nP[1]); */
    lsize = fft5d_fmax(N[0]*M[0]*K[0]*nP[0], fft5d_fmax(N[1]*M[1]*K[1]*nP[1], C[2]*M[2]*K[2]));
    /* int lsize = fmax(C[0]*M[0]*K[0],fmax(C[1]*M[1]*K[1],C[2]*M[2]*K[2])); */
    if (!(flags&FFT5D_NOMALLOC)) // first call
    {
        snew_aligned(lin, lsize, 32);
        snew_aligned(lout, lsize, 32);
        if (nthreads > 1)
        {
            /* We need extra transpose buffers to avoid OpenMP barriers */
            snew_aligned(lout2, lsize, 32);
            snew_aligned(lout3, lsize, 32);
        }
        else
        {
            /* We can reuse the buffers to avoid cache misses */
            lout2 = lin;
            lout3 = lout;
        }
    }
    else // second call
    {
        lin  = *rlin;
        lout = *rlout;
        if (nthreads > 1)
        {
            lout2 = *rlout2;
            lout3 = *rlout3;
        }
        else
        {
            lout2 = lin;
            lout3 = lout;
        }
    }

    plan = (fft5d_plan)calloc(1, sizeof(struct fft5d_plan_t));




    for (s = 0; s < 3; s++)
    {
        plan->p1d[s] = (gmx_fft_t*)malloc(sizeof(gmx_fft_t)*nthreads);

        /* Make sure that the init routines are only called by one thread at a time and in order
           (later is only important to not confuse valgrind)
         */
#pragma omp parallel for num_threads(nthreads) schedule(static) ordered
        for (t = 0; t < nthreads; t++)
        {
#pragma omp ordered
            {
                int tsize = ((t+1)*pM[s]*pK[s]/nthreads)-(t*pM[s]*pK[s]/nthreads);

                if ((flags&FFT5D_REALCOMPLEX) && ((!(flags&FFT5D_BACKWARD) && s == 0) || ((flags&FFT5D_BACKWARD) && s == 2)))
                {
                    gmx_fft_init_many_1d_real( &plan->p1d[s][t], rC[s], tsize, (flags&FFT5D_NOMEASURE) ? GMX_FFT_FLAG_CONSERVATIVE : 0 );
                }
                else
                {
                    gmx_fft_init_many_1d     ( &plan->p1d[s][t],  C[s], tsize, (flags&FFT5D_NOMEASURE) ? GMX_FFT_FLAG_CONSERVATIVE : 0 );
                }
            }
        }
    }

    if ((flags&FFT5D_ORDER_YZ))   /*plan->cart is in the order of transposes */
    {
        plan->cart[0] = comm[0]; plan->cart[1] = comm[1];
    }
    else
    {
        plan->cart[1] = comm[0]; plan->cart[0] = comm[1];
    }

    plan->lin   = lin;
    plan->lout  = lout;
    plan->lout2 = lout2;
    plan->lout3 = lout3;

    plan->NG = NG; plan->MG = MG; plan->KG = KG;

    for (s = 0; s < 3; s++)
    {
        plan->N[s]    = N[s]; plan->M[s] = M[s]; plan->K[s] = K[s]; plan->pN[s] = pN[s]; plan->pM[s] = pM[s]; plan->pK[s] = pK[s];
        plan->oM[s]   = oM[s]; plan->oK[s] = oK[s];
        plan->C[s]    = C[s]; plan->rC[s] = rC[s];
        plan->iNin[s] = iNin[s]; plan->oNin[s] = oNin[s]; plan->iNout[s] = iNout[s]; plan->oNout[s] = oNout[s];
    }
    for (s = 0; s < 2; s++)
    {
        plan->P[s] = nP[s]; plan->coor[s] = prank[s];
    }

    plan->flags    = flags;
    plan->nthreads = nthreads;
    *rlin          = lin;
    *rlout         = lout;
    *rlout2        = lout2;
    *rlout3        = lout3;
    return plan;
}


enum order {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX
};



/*make axis contiguous again (after AllToAll) and also do local transpose*/
/*transpose mayor and major dimension
   variables see above
   the major, middle, minor order is only correct for x,y,z (N,M,K) for the input
   N,M,K local dimensions
   KG global size*/
static void joinAxesTrans13(t_complex* lout, const t_complex* lin,
                            int maxN, int maxM, int maxK, int pN, int pM, int pK,
                            int P, int KG, int* K, int* oK, int starty, int startx, int endy, int endx)
{// called 
    int i, x, y, z;
    int out_i, in_i, out_x, in_x, out_z, in_z;
    int s_y, e_y;

    for (x = startx; x < endx+1; x++) /*1.j*/
    {
        if (x == startx)
        {
            s_y = starty;
        }
        else
        {
            s_y = 0;
        }
        if (x == endx)
        {
            e_y = endy;
        }
        else
        {
            e_y = pM;
        }

        out_x  = x*KG*pM;
        in_x   = x;

        for (i = 0; i < P; i++) /*index cube along long axis*/
        {
            out_i  = out_x  + oK[i];
            in_i   = in_x + i*maxM*maxN*maxK;
            for (z = 0; z < K[i]; z++) /*3.l*/
            {
                out_z  = out_i  + z;
                in_z   = in_i + z*maxM*maxN;
                for (y = s_y; y < e_y; y++)              /*2.k*/
                {
                    lout[out_z+y*KG] = lin[in_z+y*maxN]; /*out=x*KG*pM+oK[i]+z+y*KG*/
                }
            }
        }
    }
}

/*make axis contiguous again (after AllToAll) and also do local transpose
   tranpose mayor and middle dimension
   variables see above
   the minor, middle, major order is only correct for x,y,z (N,M,K) for the input
   N,M,K local size
   MG, global size*/
static void joinAxesTrans12(t_complex* lout, const t_complex* lin, int maxN, int maxM, int maxK, int pN, int pM, int pK,
                            int P, int MG, int* M, int* oM, int startx, int startz, int endx, int endz)
{//called 
    int i, z, y, x;
    int out_i, in_i, out_z, in_z, out_x, in_x;
    int s_x, e_x;

    for (z = startz; z < endz+1; z++)
    {
        if (z == startz)
        {
            s_x = startx;
        }
        else
        {
            s_x = 0;
        }
        if (z == endz)
        {
            e_x = endx;
        }
        else
        {
            e_x = pN;
        }
        out_z  = z*MG*pN;
        in_z   = z*maxM*maxN;

        for (i = 0; i < P; i++) /*index cube along long axis*/
        {
            out_i  = out_z  + oM[i];
            in_i   = in_z + i*maxM*maxN*maxK;
            for (x = s_x; x < e_x; x++)
            {
                out_x  = out_i  + x*MG;
                in_x   = in_i + x;
                for (y = 0; y < M[i]; y++)
                {
                    lout[out_x+y] = lin[in_x+y*maxN]; /*out=z*MG*pN+oM[i]+x*MG+y*/
                }
            }
        }
    }
}

void fft5d_execute(fft5d_plan plan, int thread, fft5d_time times)
{// called 
    t_complex  *lin   = plan->lin;
    t_complex  *lout  = plan->lout;
    t_complex  *lout2 = plan->lout2;
    t_complex  *lout3 = plan->lout3;
    t_complex  *fftout, *joinin;

    gmx_fft_t **p1d = plan->p1d;
    MPI_Comm *cart = plan->cart;

    double time_fft = 0, time_local = 0, time_mpi[2] = {0}, time = 0;
    int   *N        = plan->N, *M = plan->M, *K = plan->K, *pN = plan->pN, *pM = plan->pM, *pK = plan->pK,
    *C              = plan->C, *P = plan->P, **iNin = plan->iNin, **oNin = plan->oNin, **iNout = plan->iNout, **oNout = plan->oNout;
    int    s        = 0, tstart, tend;


    s = 0;

    /*lin: x,y,z*/

    for (s = 0; s < 2; s++)  /*loop over first two FFT steps (corner rotations)*/
    {

        /* ---------- START FFT ------------ */

        if (s == 0)
        {
            fftout = lout3;
        }
        else
        {
            fftout = lout2;
        }

        tstart = (thread*pM[s]*pK[s]/plan->nthreads)*C[s];
        if ((plan->flags&FFT5D_REALCOMPLEX) && !(plan->flags&FFT5D_BACKWARD) && s == 0)
        {
            gmx_fft_many_1d_real(p1d[s][thread], (plan->flags&FFT5D_BACKWARD) ? GMX_FFT_COMPLEX_TO_REAL : GMX_FFT_REAL_TO_COMPLEX, lin+tstart, fftout+tstart);
        }
        else
        {
            gmx_fft_many_1d(     p1d[s][thread], (plan->flags&FFT5D_BACKWARD) ? GMX_FFT_BACKWARD : GMX_FFT_FORWARD,               lin+tstart, fftout+tstart);

        }

        /* ---------- END FFT ------------ */

#pragma omp barrier  /*both needed for parallel and non-parallel dimension (either have to wait on data from AlltoAll or from last FFT*/


        /* ---------- START JOIN ------------ */

        joinin = fftout;
        /*bring back in matrix form
           thus make  new 1. axes contiguos
           also local transpose 1 and 2/3
           runs on thread used for following FFT (thus needing a barrier before but not afterwards)
         */
        if ((s == 0 && !(plan->flags&FFT5D_ORDER_YZ)) || (s == 1 && (plan->flags&FFT5D_ORDER_YZ)))
        {
            if (pM[s] > 0)
            {
                tstart = ( thread   *pM[s]*pN[s]/plan->nthreads);
                tend   = ((thread+1)*pM[s]*pN[s]/plan->nthreads);
                joinAxesTrans13(lin, joinin, N[s], pM[s], K[s], pN[s], pM[s], pK[s], P[s], C[s+1], iNin[s+1], oNin[s+1], tstart%pM[s], tstart/pM[s], tend%pM[s], tend/pM[s]);
            }
        }
        else
        {
            if (pN[s] > 0)
            {
                tstart = ( thread   *pK[s]*pN[s]/plan->nthreads);
                tend   = ((thread+1)*pK[s]*pN[s]/plan->nthreads);
                joinAxesTrans12(lin, joinin, N[s], M[s], pK[s], pN[s], pM[s], pK[s], P[s], C[s+1], iNin[s+1], oNin[s+1], tstart%pN[s], tstart/pN[s], tend%pN[s], tend/pN[s]);
            }
        }

        /* ---------- END JOIN ------------ */

    }  /* for(s=0;s<2;s++) */
    /*  ----------- FFT ----------- */
    tstart = (thread*pM[s]*pK[s]/plan->nthreads)*C[s];
    if ((plan->flags&FFT5D_REALCOMPLEX) && (plan->flags&FFT5D_BACKWARD)) // second call
    {
        gmx_fft_many_1d_real(p1d[s][thread], (plan->flags&FFT5D_BACKWARD) ? GMX_FFT_COMPLEX_TO_REAL : GMX_FFT_REAL_TO_COMPLEX, lin+tstart, lout+tstart);
    }
    else // first call
    {
        gmx_fft_many_1d(     p1d[s][thread], (plan->flags&FFT5D_BACKWARD) ? GMX_FFT_BACKWARD : GMX_FFT_FORWARD,               lin+tstart, lout+tstart);
    }
    /* ------------ END FFT ---------*/


}


/*Is this better than direct access of plan? enough data?
   here 0,1 reference divided by which processor grid dimension (not FFT step!)*/
void fft5d_local_size(fft5d_plan plan, int* N1, int* M0, int* K0, int* K1, int** coor)
{// called 
    *N1 = plan->N[0];
    *M0 = plan->M[0];
    *K1 = plan->K[0];
    *K0 = plan->N[1];

    *coor = plan->coor;
}


