
#ifndef _nbnxn_kernel_common_h
#define _nbnxn_kernel_common_h

#include "typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Clear the force buffer f. Either the whole buffer or only the parts
 * used by the current thread when nbat->bUseBufferFlags is set.
 * In the latter case output_index is the task/thread list/buffer index.
 */
void
clear_f(const nbnxn_atomdata_t *nbat, int output_index, real *f);

/* Clear the shift forces */
void
clear_fshift(real *fshift);

/* Reduce the collected energy terms over the pair-lists/threads */
void
reduce_energies_over_lists(const nbnxn_atomdata_t     *nbat,
                           int                         nlist,
                           real                       *Vvdw,
                           real                       *Vc);

#ifdef __cplusplus
}
#endif

#endif
