/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_CRAY_XT3
#include <catamount/dclock.h>
#endif


#include <stdio.h>
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif
#include <math.h>
#include "visibility.h"
#include "typedefs.h"
#include "string2.h"
#include "gmxfio.h"
#include "smalloc.h"
#include "names.h"
#include "confio.h"
#include "mvdata.h"
#include "txtdump.h"
#include "pbc.h"
#include "chargegroup.h"
#include "vec.h"
#include <time.h>
#include "nrnb.h"
#include "mshift.h"
#include "mdrun.h"
#include "sim_util.h"
#include "update.h"
#include "physics.h"
#include "main.h"
#include "mdatoms.h"
#include "force.h"
#include "bondf.h"
#include "pme.h"
#include "disre.h"
#include "orires.h"
#include "network.h"
#include "calcmu.h"
#include "constr.h"
#include "xvgr.h"
#include "trnio.h"
#include "xtcio.h"
#include "copyrite.h"
#include "pull_rotation.h"
#include "gmx_random.h"
#include "mpelogging.h"
#include "domdec.h"
#include "partdec.h"
#include "gmx_wallcycle.h"
#include "genborn.h"
#include "nbnxn_atomdata.h"
#include "nbnxn_search.h"
#include "nbnxn_kernels/nbnxn_kernel_ref.h"
#include "nbnxn_kernels/nbnxn_kernel_gpu_ref.h"

#include "tmpi.h"
#include "types/simple.h"
#include "typedefs.h"
#include "qmmm.h"

/* Portable version of ctime_r implemented in src/gmxlib/string2.c, but we do not want it declared in public installed headers */
GMX_LIBGMX_EXPORT
char *
gmx_ctime_r(const time_t *clock, char *buf, int n);


double
gmx_gettime()
{
    struct timeval t;
    double         seconds;

    gettimeofday(&t, NULL);

    seconds = (double) t.tv_sec + 1e-6*(double)t.tv_usec;

    return seconds;
}


#define difftime(end, start) ((double)(end)-(double)(start))

void print_time(FILE *out, gmx_runtime_t *runtime, gmx_large_int_t step,
                t_inputrec *ir, t_commrec *cr)
{
    time_t finish;
    char   timebuf[STRLEN];
    double dt;
    char   buf[48];

    fprintf(out, "\r");
    fprintf(out, "step %s", gmx_step_str(step, buf));
    if ((step >= ir->nstlist))
    {
        runtime->last          = gmx_gettime();
        dt                     = difftime(runtime->last, runtime->real);
        runtime->time_per_step = dt/(step - ir->init_step + 1);

        dt = (ir->nsteps + ir->init_step - step)*runtime->time_per_step;

        if (ir->nsteps >= 0)
        {
            if (dt >= 300)
            {
                finish = (time_t) (runtime->last + dt);
                gmx_ctime_r(&finish, timebuf, STRLEN);
                sprintf(buf, "%s", timebuf);
                buf[strlen(buf)-1] = '\0';
                fprintf(out, ", will finish %s", buf);
            }
            else
            {
                fprintf(out, ", remaining runtime: %5d s          ", (int)dt);
            }
        }
        else
        {
            fprintf(out, " performance: %.1f ns/day    ",
                    ir->delta_t/1000*24*60*60/runtime->time_per_step);
        }
    }

    fflush(out);
}

#ifdef NO_CLOCK
#define clock() -1
#endif

static double set_proctime(gmx_runtime_t *runtime)
{ // called
    double diff;
    clock_t prev;

    prev          = runtime->proc;
    runtime->proc = clock();

    diff = (double)(runtime->proc - prev)/(double)CLOCKS_PER_SEC;

    return diff;
}

void runtime_start(gmx_runtime_t *runtime)
{
    runtime->real          = gmx_gettime();
    runtime->proc          = 0;
    set_proctime(runtime);
    runtime->realtime      = 0;
    runtime->proctime      = 0;
    runtime->last          = 0;
    runtime->time_per_step = 0;
}

void runtime_end(gmx_runtime_t *runtime)
{
    double now;

    now = gmx_gettime();

    runtime->proctime += set_proctime(runtime);
    runtime->realtime  = now - runtime->real;
    runtime->real      = now;
}

void runtime_upd_proc(gmx_runtime_t *runtime)
{
    runtime->proctime += set_proctime(runtime);
}

void print_date_and_time(FILE *fplog, int nodeid, const char *title,
                         const gmx_runtime_t *runtime)
{ // called
    int    i;
    char   timebuf[STRLEN];
    char   time_string[STRLEN];
    time_t tmptime;
        tmptime = (time_t) runtime->real;
        gmx_ctime_r(&tmptime, timebuf, STRLEN);
        for (i = 0; timebuf[i] >= ' '; i++)
        {
            time_string[i] = timebuf[i];
        }
        time_string[i] = '\0';

        fprintf(fplog, "%s on node %d %s\n", title, nodeid, time_string);
}

static void sum_forces(int start, int end, rvec f[], rvec flr[])
{ // called
    int i;
    for (i = start; (i < end); i++)
    {
        rvec_inc(f[i], flr[i]);
    }
}

static void calc_virial(FILE *fplog, int start, int homenr, rvec x[], rvec f[],
                        tensor vir_part, t_graph *graph, matrix box,
                        t_nrnb *nrnb, const t_forcerec *fr, int ePBC)
{ // CALLED
    int    i, j;
    tensor virtest;

    /* The short-range virial from surrounding boxes */
    clear_mat(vir_part);
    calc_vir(fplog, SHIFTS, fr->shift_vec, fr->fshift, vir_part, ePBC == epbcSCREW, box);
    inc_nrnb(nrnb, eNR_VIRIAL, SHIFTS);

    /* Calculate partial virial, for local atoms only, based on short range.
     * Total virial is computed in global_stat, called from do_md
     */
    f_calc_vir(fplog, start, start+homenr, x, f, vir_part, graph, box);
    inc_nrnb(nrnb, eNR_VIRIAL, homenr);

    /* Add position restraint contribution */
    for (i = 0; i < DIM; i++)
    {
        vir_part[i][i] += fr->vir_diag_posres[i];
    }

    /* Add wall contribution */
    for (i = 0; i < DIM; i++)
    {
        vir_part[i][ZZ] += fr->vir_wall_z[i];
    }

}


static void post_process_forces(FILE *fplog,
                                t_commrec *cr,
                                gmx_large_int_t step,
                                t_nrnb *nrnb, gmx_wallcycle_t wcycle,
                                gmx_localtop_t *top,
                                matrix box, rvec x[],
                                rvec f[],
                                tensor vir_force,
                                t_mdatoms *mdatoms,
                                t_graph *graph,
                                t_forcerec *fr, gmx_vsite_t *vsite,
                                int flags)
{ // being called
   /* Now add the forces, this is local */
   sum_forces(mdatoms->start, mdatoms->start+mdatoms->homenr,
                         f, fr->f_novirsum);
   /* Add the mesh contribution to the virial */
   m_add(vir_force, fr->vir_el_recip, vir_force);

}

static void do_nb_verlet(t_forcerec *fr,
                         interaction_const_t *ic,
                         gmx_enerdata_t *enerd,
                         int flags, int ilocality,
                         int clearF,
                         t_nrnb *nrnb,
                         gmx_wallcycle_t wcycle)
{
    int                        nnbl, kernel_type, enr_nbnxn_kernel_ljc, enr_nbnxn_kernel_lj;
    char                      *env;
    nonbonded_verlet_group_t  *nbvg;

    nbvg = &fr->nbv->grp[ilocality];
    wallcycle_sub_start(wcycle, ewcsNONBONDED);
    nbnxn_kernel_ref(&nbvg->nbl_lists,
                             nbvg->nbat, ic,
                             fr->shift_vec,
                             flags,
                             clearF,
                             fr->fshift[0],
                             enerd->grpp.ener[egCOULSR],
                             fr->bBHAM ?
                             enerd->grpp.ener[egBHAMSR] :
                             enerd->grpp.ener[egLJSR]);
    wallcycle_sub_stop(wcycle, ewcsNONBONDED);
    enr_nbnxn_kernel_ljc = eNR_NBNXN_LJ_TAB +1;
    enr_nbnxn_kernel_lj = eNR_NBNXN_LJ +1;
    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc,
             nbvg->nbl_lists.natpair_ljq);
    inc_nrnb(nrnb, enr_nbnxn_kernel_lj,
             nbvg->nbl_lists.natpair_lj);
    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc-eNR_NBNXN_LJ_RF+eNR_NBNXN_RF,
             nbvg->nbl_lists.natpair_q);
}

void do_force_cutsVERLET(FILE *fplog, t_commrec *cr,
                         t_inputrec *inputrec,
                         gmx_large_int_t step, t_nrnb *nrnb, gmx_wallcycle_t wcycle,
                         gmx_localtop_t *top,
                         gmx_mtop_t *mtop,
                         gmx_groups_t *groups,
                         matrix box, rvec x[], history_t *hist,
                         rvec f[],
                         tensor vir_force,
                         t_mdatoms *mdatoms,
                         gmx_enerdata_t *enerd, t_fcdata *fcd,
                         real *lambda, t_graph *graph,
                         t_forcerec *fr, interaction_const_t *ic,
                         gmx_vsite_t *vsite, rvec mu_tot,
                         double t, FILE *field, gmx_edsam_t ed,
                         gmx_bool bBornRadii,
                         int flags)
{
    int                 cg0, cg1, i, j;
    int                 start, homenr;
    int                 nb_kernel_type;
    double              mu[2*DIM];
    gmx_bool            bSepDVDL, bBS;
    gmx_bool            bDiffKernels = FALSE;
    matrix              boxs;
    rvec                vzero, box_diag;
    real                e, v, dvdl;
    float               cycles_pme, cycles_force;
    nonbonded_verlet_t *nbv;

    cycles_force   = 0;
    nbv            = fr->nbv;
    nb_kernel_type = fr->nbv->grp[0].kernel_type;

    start  = mdatoms->start;
    homenr = mdatoms->homenr;

    bSepDVDL = (fr->bSepDVDL && do_per_step(step, inputrec->nstlog));

    clear_mat(vir_force);

    cg0 = 0;
    cg1 = top->cgs.nr;

    put_atoms_in_box_omp(fr->ePBC, box, homenr, x);
    inc_nrnb(nrnb, eNR_SHIFTX, homenr);

    nbnxn_atomdata_copy_shiftvec(flags & GMX_FORCE_DYNAMICBOX,
                                 fr->shift_vec, nbv->grp[0].nbat);


        clear_rvec(vzero);
        box_diag[XX] = box[XX][XX];
        box_diag[YY] = box[YY][YY];
        box_diag[ZZ] = box[ZZ][ZZ];

        wallcycle_start(wcycle, ewcNS);
            wallcycle_sub_start(wcycle, ewcsNBS_GRID_LOCAL);
            nbnxn_put_on_grid(nbv->nbs, fr->ePBC, box,
                              0, vzero, box_diag,
                              0, mdatoms->homenr, -1, fr->cginfo, x,
                              0, NULL,
                              nbv->grp[eintLocal].kernel_type,
                              nbv->grp[eintLocal].nbat);
            wallcycle_sub_stop(wcycle, ewcsNBS_GRID_LOCAL);

       nbnxn_atomdata_set(nbv->grp[eintLocal].nbat, eatAll,
                               nbv->nbs, mdatoms, fr->cginfo);
        wallcycle_stop(wcycle, ewcNS);


    /* do local pair search */
        wallcycle_start_nocount(wcycle, ewcNS);
        wallcycle_sub_start(wcycle, ewcsNBS_SEARCH_LOCAL);
        nbnxn_make_pairlist(nbv->nbs, nbv->grp[eintLocal].nbat,
                            &top->excls,
                            ic->rlist,
                            nbv->min_ci_balanced,
                            &nbv->grp[eintLocal].nbl_lists,
                            eintLocal,
                            nbv->grp[eintLocal].kernel_type,
                            nrnb);
        wallcycle_sub_stop(wcycle, ewcsNBS_SEARCH_LOCAL);

        wallcycle_stop(wcycle, ewcNS);


        copy_rvec(fr->mu_tot[0], mu_tot);

    /* Reset energies */
    reset_enerdata(&(inputrec->opts), fr, 1, enerd, MASTER(cr));
    clear_rvecs(SHIFTS, fr->fshift);


    /* Start the force cycle counter.
     * This counter is stopped in do_forcelow_level.
     * No parallel communication should occur while this counter is running,
     * since that will interfere with the dynamic load balancing.
     */
    wallcycle_start(wcycle, ewcFORCE);
        /* Reset forces for which the virial is calculated separately:
         * PME/Ewald forces if necessary */
         fr->f_novirsum = fr->f_novirsum_alloc;
         GMX_BARRIER(cr->mpi_comm_mygroup);
         clear_rvecs(homenr, fr->f_novirsum+start);
         GMX_BARRIER(cr->mpi_comm_mygroup);

        /* Clear the short- and long-range forces */
        clear_rvecs(fr->natoms_force_constr, f);

        clear_rvec(fr->vir_diag_posres);

        GMX_BARRIER(cr->mpi_comm_mygroup);



    /* Compute the bonded and non-bonded energies and optionally forces */
    do_force_lowlevel(fplog, step, fr, inputrec, &(top->idef),
                      cr, nrnb, wcycle, mdatoms, &(inputrec->opts),
                      x, hist, f, f, enerd, fcd, mtop, top, 
                      &(top->atomtypes), bBornRadii, box,
                      inputrec->fepvals, lambda, graph, &(top->excls), fr->mu_tot,
                      flags, &cycles_pme);


    /* Maybe we should move this into do_force_lowlevel */
    do_nb_verlet(fr, ic, enerd, flags, eintLocal, enbvClearFYes,
                     nrnb, wcycle);



        /* Add all the non-bonded force to the normal force array.
         * This can be split into a local a non-local part when overlapping
         * communication with calculation with domain decomposition.
         */
        cycles_force += wallcycle_stop(wcycle, ewcFORCE);
        wallcycle_start(wcycle, ewcNB_XF_BUF_OPS);
        wallcycle_sub_start(wcycle, ewcsNB_F_BUF_OPS);
        nbnxn_atomdata_add_nbat_f_to_f(nbv->nbs, eatAll, nbv->grp[eintLocal].nbat, f);
        wallcycle_sub_stop(wcycle, ewcsNB_F_BUF_OPS);
        cycles_force += wallcycle_stop(wcycle, ewcNB_XF_BUF_OPS);
        wallcycle_start_nocount(wcycle, ewcFORCE);

        /* if there are multiple fshift output buffers reduce them */
        nbnxn_atomdata_add_nbat_fshift_to_fshift(nbv->grp[eintLocal].nbat,
                                                     fr->fshift);

    cycles_force += wallcycle_stop(wcycle, ewcFORCE);
    GMX_BARRIER(cr->mpi_comm_mygroup);



    /* Calculation of the virial must be done after vsites! */
    calc_virial(fplog, mdatoms->start, mdatoms->homenr, x, f,
                        vir_force, graph, box, nrnb, fr, inputrec->ePBC);
    post_process_forces(fplog, cr, step, nrnb, wcycle,
                            top, box, x, f, vir_force, mdatoms, graph, fr, vsite,
                            flags);

    /* Sum the potential energy terms from group contributions */
    sum_epot(&(inputrec->opts), &(enerd->grpp), enerd->term);
}


void do_force(FILE *fplog, t_commrec *cr,
              t_inputrec *inputrec,
              gmx_large_int_t step, t_nrnb *nrnb, gmx_wallcycle_t wcycle,
              gmx_localtop_t *top,
              gmx_mtop_t *mtop,
              gmx_groups_t *groups,
              matrix box, rvec x[], history_t *hist,
              rvec f[],
              tensor vir_force,
              t_mdatoms *mdatoms,
              gmx_enerdata_t *enerd, t_fcdata *fcd,
              real *lambda, t_graph *graph,
              t_forcerec *fr,
              gmx_vsite_t *vsite, rvec mu_tot,
              double t, FILE *field, gmx_edsam_t ed,
              gmx_bool bBornRadii,
              int flags)
{
    do_force_cutsVERLET(fplog, cr, inputrec,
                                step, nrnb, wcycle,
                                top, mtop,
                                groups,
                                box, x, hist,
                                f, vir_force,
                                mdatoms,
                                enerd, fcd,
                                lambda, graph,
                                fr, fr->ic,
                                vsite, mu_tot,
                                t, field, ed,
                                bBornRadii,
                                flags);
}


void calc_dispcorr(FILE *fplog, t_inputrec *ir, t_forcerec *fr,
                   gmx_large_int_t step, int natoms,
                   matrix box, real lambda, tensor pres, tensor virial,
                   real *prescorr, real *enercorr, real *dvdlcorr)
{
    gmx_bool bCorrAll, bCorrPres;
    real     dvdlambda, invvol, dens, ninter, avcsix, avctwelve, enerdiff, svir = 0, spres = 0;
    int      m;

    *prescorr = 0;
    *enercorr = 0;
    *dvdlcorr = 0;

    clear_mat(virial);
    clear_mat(pres);

}


static void low_do_pbc_mtop(FILE *fplog, int ePBC, matrix box,
                            gmx_mtop_t *mtop, rvec x[],
                            gmx_bool bFirst)
{
    t_graph        *graph;
    int             mb, as, mol;
    gmx_molblock_t *molb;

    if (bFirst && fplog)
    {
        fprintf(fplog, "Removing pbc first time\n");
    }

    snew(graph, 1);
    as = 0;
    for (mb = 0; mb < mtop->nmolblock; mb++)
    {
        molb = &mtop->molblock[mb];
            /* Pass NULL iso fplog to avoid graph prints for each molecule type */
            mk_graph_ilist(NULL, mtop->moltype[molb->type].ilist,
                           0, molb->natoms_mol, FALSE, FALSE, graph);

            for (mol = 0; mol < molb->nmol; mol++)
            {
                mk_mshift(fplog, graph, ePBC, box, x+as);

                shift_self(graph, box, x+as);
                /* The molecule is whole now.
                 * We don't need the second mk_mshift call as in do_pbc_first,
                 * since we no longer need this graph.
                 */

                as += molb->natoms_mol;
            }
            done_graph(graph);
    }
    sfree(graph);
}

void do_pbc_first_mtop(FILE *fplog, int ePBC, matrix box,
                       gmx_mtop_t *mtop, rvec x[])
{
    low_do_pbc_mtop(fplog, ePBC, box, mtop, x, TRUE);
}

void do_pbc_mtop(FILE *fplog, int ePBC, matrix box,
                 gmx_mtop_t *mtop, rvec x[])
{
    low_do_pbc_mtop(fplog, ePBC, box, mtop, x, FALSE);
}

void finish_run(FILE *fplog, t_commrec *cr, const char *confout,
                t_inputrec *inputrec,
                t_nrnb nrnb[], gmx_wallcycle_t wcycle,
                gmx_runtime_t *runtime,
                wallclock_gpu_t *gputimes,
                int omp_nth_pp,
                gmx_bool bWriteStat)
{
    int     i, j;
    t_nrnb *nrnb_tot = NULL;
    real    delta_t;
    double  nbfs, mflop;

    wallcycle_sum(cr, wcycle);

    nrnb_tot = nrnb;


    print_flop(fplog, nrnb_tot, &nbfs, &mflop);
    {
        wallcycle_print(fplog, cr->nnodes, cr->npmenodes, runtime->realtime,
                        wcycle, gputimes);

        if (EI_DYNAMICS(inputrec->eI))
        {
            delta_t = inputrec->delta_t;
        }
        else
        {
            delta_t = 0;
        }

        if (fplog)
        {
            print_perf(fplog, runtime->proctime, runtime->realtime,
                       cr->nnodes-cr->npmenodes,
                       runtime->nsteps_done, delta_t, nbfs, mflop,
                       omp_nth_pp);
        }
        if (bWriteStat)
        {
            print_perf(stderr, runtime->proctime, runtime->realtime,
                       cr->nnodes-cr->npmenodes,
                       runtime->nsteps_done, delta_t, nbfs, mflop,
                       omp_nth_pp);
        }
    }
}

extern void initialize_lambdas(FILE *fplog, t_inputrec *ir, int *fep_state, real *lambda, double *lam0)
{
    /* this function works, but could probably use a logic rewrite to keep all the different
       types of efep straight. */

    int       i;
    t_lambda *fep = ir->fepvals;

    for (i = 0; i < efptNR; i++)
    {
        lambda[i] = 0.0;
        if (lam0)
        {
            lam0[i] = 0.0;
        }
    }
}


void init_md(FILE *fplog,
             t_commrec *cr, t_inputrec *ir, const output_env_t oenv,
             double *t, double *t0,
             real *lambda, int *fep_state, double *lam0,
             t_nrnb *nrnb, gmx_mtop_t *mtop,
             gmx_update_t *upd,
             int nfile, const t_filenm fnm[],
             gmx_mdoutf_t **outf, t_mdebin **mdebin,
             tensor force_vir, tensor shake_vir, rvec mu_tot,
             gmx_bool *bSimAnn, t_vcm **vcm, t_state *state, unsigned long Flags)
{
    int  i, j, n;
    real tmpt, mod;

    /* Initial values */
    *t = *t0       = ir->init_t;

    *bSimAnn = FALSE; // KEEP

    /* Initialize lambda variables */
    initialize_lambdas(fplog, ir, fep_state, lambda, lam0);

    *upd = init_update(fplog, ir);


    *vcm = init_vcm(fplog, &mtop->groups, ir);


    init_nrnb(nrnb);

    *outf = init_mdoutf(nfile, fnm, Flags, cr, ir, oenv);
    *mdebin = init_mdebin((*outf)->fp_ene,
                              mtop, ir, (*outf)->fp_dhdl);

    /* Initiate variables */
    clear_mat(force_vir);
    clear_mat(shake_vir);
    clear_rvec(mu_tot);

    debug_gmx();
}
