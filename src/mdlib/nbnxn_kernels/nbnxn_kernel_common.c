#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "nbnxn_kernel_common.h"


static void
clear_f_flagged(const nbnxn_atomdata_t *nbat, int output_index, real *f)
{// called 
    const nbnxn_buffer_flags_t *flags;
    unsigned                    our_flag;
    int g, b, a0, a1, i;

    flags = &nbat->buffer_flags;

    our_flag = (1U << output_index);
    for (b = 0; b < flags->nflag; b++) // 390
    {
        if (flags->flag[b] & our_flag)
        { 
            a0 = b*NBNXN_BUFFERFLAG_SIZE;
            a1 = a0 + NBNXN_BUFFERFLAG_SIZE;
            for (i = a0*nbat->fstride; i < a1*nbat->fstride; i++)
            {
                f[i] = 0;
            }
        }
    }
}

void
clear_f(const nbnxn_atomdata_t *nbat, int output_index, real *f)
{ //called
   clear_f_flagged(nbat, output_index, f);
}

void
clear_fshift(real *fshift)
{// called 
    int i;
    for (i = 0; i < SHIFTS*DIM; i++) // 135
    {
        fshift[i] = 0;
    }
}

void
reduce_energies_over_lists(const nbnxn_atomdata_t     *nbat,
                           int                         nlist,
                           real                       *Vvdw,
                           real                       *Vc)
{// called 
    int nb;
    int ind;
    for (nb = 0; nb < nlist; nb++) // 12
    {
       /* Reduce the diagonal terms */
       ind        = 0;
       Vvdw[ind] += nbat->out[nb].Vvdw[ind];
       Vc[ind]   += nbat->out[nb].Vc[ind];

    }
}
