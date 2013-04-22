#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <stdio.h>
#include "typedefs.h"
#include "sysstuff.h"
#include "gmx_fatal.h"
#include "network.h"
#include "txtdump.h"
#include "names.h"
#include "physics.h"
#include "vec.h"
#include "maths.h"
#include "mvdata.h"
#include "main.h"
#include "force.h"
#include "vcm.h"
#include "smalloc.h"
#include "futil.h"
#include "network.h"
#include "rbin.h"
#include "tgroup.h"
#include "xtcio.h"
#include "gmxfio.h"
#include "trnio.h"
#include "statutil.h"
#include "domdec.h"
#include "partdec.h"
#include "constr.h"
#include "checkpoint.h"
#include "xvgr.h"
#include "md_support.h"
#include "mdrun.h"
#include "sim_util.h"

typedef struct gmx_global_stat
{
    t_bin *rb;
    int   *itc0;
    int   *itc1;
} t_gmx_global_stat;

gmx_global_stat_t global_stat_init(t_inputrec *ir)
{ //called
    gmx_global_stat_t gs;

    snew(gs, 1);

    gs->rb = mk_bin();
    snew(gs->itc0, ir->opts.ngtc);
    snew(gs->itc1, ir->opts.ngtc);

    return gs;
}

int do_per_step(gmx_large_int_t step, gmx_large_int_t nstep)
{ //called
    if (nstep != 0) // 100, 100, 0, 500, 500, 500, 10, 100, 100
    {
        return ((step % nstep) == 0);
    }
    else
    {
        return 0;
    }
}

gmx_mdoutf_t *init_mdoutf(int nfile, const t_filenm fnm[], int mdrun_flags,
                          const t_commrec *cr, const t_inputrec *ir,
                          const output_env_t oenv)
{//called
    gmx_mdoutf_t *of;
    char          filemode[3];

    snew(of, 1);

    of->fp_trn   = NULL;
    of->fp_ene   = NULL;
    of->fp_xtc   = NULL;
    of->fp_dhdl  = NULL;
    of->fp_field = NULL;

    of->eIntegrator     = ir->eI;
    of->bExpanded       = ir->bExpanded;
    of->elamstats       = ir->expandedvals->elamstats;
    of->simulation_part = ir->simulation_part;

    of->bKeepAndNumCPT = (mdrun_flags & MD_KEEPANDNUMCPT);

    sprintf(filemode, "w+");

    of->fp_trn = open_trn(ftp2fn(efTRN, nfile, fnm), filemode);

    of->fp_ene = open_enx(ftp2fn(efEDR, nfile, fnm), filemode);
    of->fn_cpt = opt2fn("-cpo", nfile, fnm);

    return of;
}

void done_mdoutf(gmx_mdoutf_t *of)
{//called
   close_enx(of->fp_ene);
   close_trn(of->fp_trn);

    sfree(of);
}

void write_traj(FILE *fplog, t_commrec *cr,
                gmx_mdoutf_t *of,
                int mdof_flags,
                gmx_mtop_t *top_global,
                gmx_large_int_t step, double t,
                t_state *state_local, t_state *state_global,
                rvec *f_local, rvec *f_global,
                int *n_xtc, rvec **x_xtc)
{//called
    fwrite_trn(of->fp_trn, step, t, state_local->lambda[efptFEP],
                       state_local->box, top_global->natoms,
                       NULL,
                       NULL,
                       f_global);
    gmx_fio_check_file_position(of->fp_trn);
}
