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
{//called
    if (nstep != 0)
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
    gmx_bool      bAppendFiles;

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

    if (MASTER(cr))
    {
        bAppendFiles = (mdrun_flags & MD_APPENDFILES);

        of->bKeepAndNumCPT = (mdrun_flags & MD_KEEPANDNUMCPT);

        sprintf(filemode, bAppendFiles ? "a+" : "w+");

        if ((EI_DYNAMICS(ir->eI) || EI_ENERGY_MINIMIZATION(ir->eI))
            &&
            !(EI_DYNAMICS(ir->eI) &&
              ir->nstxout == 0 &&
              ir->nstvout == 0 &&
              ir->nstfout == 0)
            )
        {
            of->fp_trn = open_trn(ftp2fn(efTRN, nfile, fnm), filemode);
        }
        if (EI_DYNAMICS(ir->eI) &&
            ir->nstxtcout > 0)
        {
            of->fp_xtc   = open_xtc(ftp2fn(efXTC, nfile, fnm), filemode);
            of->xtc_prec = ir->xtcprec;
        }
        if (EI_DYNAMICS(ir->eI) || EI_ENERGY_MINIMIZATION(ir->eI))
        {
            of->fp_ene = open_enx(ftp2fn(efEDR, nfile, fnm), filemode);
        }
        of->fn_cpt = opt2fn("-cpo", nfile, fnm);

        if ((ir->efep != efepNO || ir->bSimTemp) && ir->fepvals->nstdhdl > 0 &&
            (ir->fepvals->separate_dhdl_file == esepdhdlfileYES ) &&
            EI_DYNAMICS(ir->eI))
        {
            if (bAppendFiles)
            {
                of->fp_dhdl = gmx_fio_fopen(opt2fn("-dhdl", nfile, fnm), filemode);
            }
            else
            {
                of->fp_dhdl = open_dhdl(opt2fn("-dhdl", nfile, fnm), ir, oenv);
            }
        }

        if (opt2bSet("-field", nfile, fnm) &&
            (ir->ex[XX].n || ir->ex[YY].n || ir->ex[ZZ].n))
        {
            if (bAppendFiles)
            {
                of->fp_dhdl = gmx_fio_fopen(opt2fn("-field", nfile, fnm),
                                            filemode);
            }
            else
            {
                of->fp_field = xvgropen(opt2fn("-field", nfile, fnm),
                                        "Applied electric field", "Time (ps)",
                                        "E (V/nm)", oenv);
            }
        }
    }

    return of;
}

void done_mdoutf(gmx_mdoutf_t *of)
{//called
    if (of->fp_ene != NULL)
    {
        close_enx(of->fp_ene);
    }
    if (of->fp_xtc)
    {
        close_xtc(of->fp_xtc);
    }
    if (of->fp_trn)
    {
        close_trn(of->fp_trn);
    }
    if (of->fp_dhdl != NULL)
    {
        gmx_fio_fclose(of->fp_dhdl);
    }
    if (of->fp_field != NULL)
    {
        gmx_fio_fclose(of->fp_field);
    }

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
    int           i, j;
    gmx_groups_t *groups;
    rvec         *xxtc;
    rvec         *local_v;
    rvec         *global_v;


    /* MRS -- defining these variables is to manage the difference
     * between half step and full step velocities, but there must be a better way . . . */

    local_v  = state_local->v;
    global_v = state_global->v;

    {
        if (mdof_flags & MDOF_CPT)
        {
            /* All pointers in state_local are equal to state_global,
             * but we need to copy the non-pointer entries.
             */
            state_global->lambda = state_local->lambda;
            state_global->veta   = state_local->veta;
            state_global->vol0   = state_local->vol0;
            copy_mat(state_local->box, state_global->box);
            copy_mat(state_local->boxv, state_global->boxv);
            copy_mat(state_local->svir_prev, state_global->svir_prev);
            copy_mat(state_local->fvir_prev, state_global->fvir_prev);
            copy_mat(state_local->pres_prev, state_global->pres_prev);
        }
        if (cr->nnodes > 1)
        {
            /* Particle decomposition, collect the data on the master node */
            if (mdof_flags & MDOF_CPT)
            {
                if (state_global->nrngi > 1)
                {
                    if (state_local->flags & (1<<estLD_RNG))
                    {
#ifdef GMX_MPI
                        MPI_Gather(state_local->ld_rng,
                                   state_local->nrng*sizeof(state_local->ld_rng[0]), MPI_BYTE,
                                   state_global->ld_rng,
                                   state_local->nrng*sizeof(state_local->ld_rng[0]), MPI_BYTE,
                                   MASTERRANK(cr), cr->mpi_comm_mygroup);
#endif
                    }
                    if (state_local->flags & (1<<estLD_RNGI))
                    {
#ifdef GMX_MPI
                        MPI_Gather(state_local->ld_rngi,
                                   sizeof(state_local->ld_rngi[0]), MPI_BYTE,
                                   state_global->ld_rngi,
                                   sizeof(state_local->ld_rngi[0]), MPI_BYTE,
                                   MASTERRANK(cr), cr->mpi_comm_mygroup);
#endif
                    }
                }
            }
        }
    }

    if (MASTER(cr))
    {
        if (mdof_flags & MDOF_CPT)
        {
            write_checkpoint(of->fn_cpt, of->bKeepAndNumCPT,
                             fplog, cr, of->eIntegrator, of->simulation_part,
                             of->bExpanded, of->elamstats, step, t, state_global);
        }

        if (mdof_flags & (MDOF_X | MDOF_V | MDOF_F))
        {
            fwrite_trn(of->fp_trn, step, t, state_local->lambda[efptFEP],
                       state_local->box, top_global->natoms,
                       (mdof_flags & MDOF_X) ? state_global->x : NULL,
                       (mdof_flags & MDOF_V) ? global_v : NULL,
                       (mdof_flags & MDOF_F) ? f_global : NULL);
            if (gmx_fio_flush(of->fp_trn) != 0)
            {
                gmx_file("Cannot write trajectory; maybe you are out of disk space?");
            }
            gmx_fio_check_file_position(of->fp_trn);
        }
        if (mdof_flags & MDOF_XTC)
        {
            groups = &top_global->groups;
            if (*n_xtc == -1)
            {
                *n_xtc = 0;
                for (i = 0; (i < top_global->natoms); i++)
                {
                    if (ggrpnr(groups, egcXTC, i) == 0)
                    {
                        (*n_xtc)++;
                    }
                }
                if (*n_xtc != top_global->natoms)
                {
                    snew(*x_xtc, *n_xtc);
                }
            }
            if (*n_xtc == top_global->natoms)
            {
                xxtc = state_global->x;
            }
            else
            {
                xxtc = *x_xtc;
                j    = 0;
                for (i = 0; (i < top_global->natoms); i++)
                {
                    if (ggrpnr(groups, egcXTC, i) == 0)
                    {
                        copy_rvec(state_global->x[i], xxtc[j++]);
                    }
                }
            }
            if (write_xtc(of->fp_xtc, *n_xtc, step, t,
                          state_local->box, xxtc, of->xtc_prec) == 0)
            {
                gmx_fatal(FARGS, "XTC error - maybe you are out of disk space?");
            }
            gmx_fio_check_file_position(of->fp_xtc);
        }
    }
}
