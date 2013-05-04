/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "sysstuff.h"
#include "typedefs.h"
#include "smalloc.h"
#include "invblock.h"
#include "macros.h"
#include "main.h"
#include "ns.h"
#include "partdec.h"
#include "splitter.h"
#include "mtop_util.h"
#include "mvdata.h"
#include "vec.h"

typedef struct gmx_partdec_constraint
{
    int                  left_range_receive;
    int                  right_range_receive;
    int                  left_range_send;
    int                  right_range_send;
    int                  nconstraints;
    int *                nlocalatoms;
    rvec *               sendbuf;
    rvec *               recvbuf;
}
gmx_partdec_constraint_t;


typedef struct gmx_partdec {
    int   neighbor[2];                         /* The nodeids of left and right neighb */
    int  *cgindex;                             /* The charge group boundaries,         */
                                               /* size nnodes+1,                       */
                                               /* only allocated with particle decomp. */
    int  *index;                               /* The home particle boundaries,        */
                                               /* size nnodes+1,                       */
                                               /* only allocated with particle decomp. */
    int  shift, bshift;                        /* Coordinates are shifted left for     */
                                               /* 'shift' systolic pulses, and right   */
                                               /* for 'bshift' pulses. Forces are      */
                                               /* shifted right for 'shift' pulses     */
                                               /* and left for 'bshift' pulses         */
                                               /* This way is not necessary to shift   */
                                               /* the coordinates over the entire ring */
    rvec                          *vbuf;       /* Buffer for summing the forces        */
#ifdef GMX_MPI
    MPI_Request                    mpi_req_rx; /* MPI reqs for async transfers */
    MPI_Request                    mpi_req_tx;
#endif
    gmx_partdec_constraint_t *     constraints;
} gmx_partdec_t;


t_state *partdec_init_local_state(t_commrec *cr, t_state *state_global)
{ // called
    int      i;
    t_state *state_local;

    snew(state_local, 1);
    /* Copy all the contents */
    *state_local = *state_global;
    snew(state_local->lambda, efptNR);

    /* local storage for lambda */
    for (i = 0; i < efptNR; i++) // 7
    {
        state_local->lambda[i] = state_global->lambda[i];
    }

    return state_local;
}
