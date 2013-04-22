/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdlib.h>

#include "sysstuff.h"
#include "typedefs.h"
#include "types/commrec.h"
#include "macros.h"
#include "smalloc.h"
#include "nsgrid.h"
#include "gmx_fatal.h"
#include "vec.h"
#include "network.h"
#include "domdec.h"
#include "partdec.h"
#include "pbc.h"
#include <stdio.h>
#include "futil.h"
#include "pdbio.h"

/***********************************
 *         Grid Routines
 ***********************************/

const char *range_warn =
    "Explanation: During neighborsearching, we assign each particle to a grid\n"
    "based on its coordinates. If your system contains collisions or parameter\n"
    "errors that give particles very high velocities you might end up with some\n"
    "coordinates being +-Infinity or NaN (not-a-number). Obviously, we cannot\n"
    "put these on a grid, so this is usually where we detect those errors.\n"
    "Make sure your system is properly energy-minimized and that the potential\n"
    "energy seems reasonable before trying again.";


t_grid *init_grid(FILE *fplog, t_forcerec *fr)
{
    int     d, m;
    char   *ptr;
    t_grid *grid;

    snew(grid, 1);

    grid->npbcdim = ePBC2npbcdim(fr->ePBC);

    grid->nboundeddim = grid->npbcdim;

    /* The ideal number of cg's per ns grid cell seems to be 10 */
    grid->ncg_ideal = 10;

    return grid;
}


