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
//#include "nbnxn_kernels/nbnxn_kernel_gpu_ref.h"

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
    enr_nbnxn_kernel_ljc = eNR_NBNXN_LJ_TAB +1;
    enr_nbnxn_kernel_lj = eNR_NBNXN_LJ +1;
    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc,
             nbvg->nbl_lists.natpair_ljq);
    inc_nrnb(nrnb, enr_nbnxn_kernel_lj,
             nbvg->nbl_lists.natpair_lj);
    inc_nrnb(nrnb, enr_nbnxn_kernel_ljc-eNR_NBNXN_LJ_RF+eNR_NBNXN_RF,
             nbvg->nbl_lists.natpair_q);
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
    int                 start, homenr;
    int                 nb_kernel_type;
    double              mu[2*DIM];
    rvec                vzero, box_diag;
    real                e, v;
    float               cycles_pme;
    nonbonded_verlet_t *nbv;

    nbv            = fr->nbv;
    nb_kernel_type = fr->nbv->grp[0].kernel_type;

    start  = mdatoms->start;
    homenr = mdatoms->homenr;


    put_atoms_in_box_omp(fr->ePBC, box, homenr, x);


    inc_nrnb(nrnb, eNR_SHIFTX, homenr);

    nbnxn_atomdata_copy_shiftvec(flags & GMX_FORCE_DYNAMICBOX,
                                 fr->shift_vec, nbv->grp[0].nbat);


    clear_rvec(vzero);
    box_diag[XX] = box[XX][XX];
    box_diag[YY] = box[YY][YY];
    box_diag[ZZ] = box[ZZ][ZZ];

    nbnxn_put_on_grid(nbv->nbs, fr->ePBC, box,
                     0, vzero, box_diag,
                     0, mdatoms->homenr, -1, fr->cginfo, x,
                     0, NULL,
                     nbv->grp[eintLocal].kernel_type,
                     nbv->grp[eintLocal].nbat);

    nbnxn_atomdata_set(nbv->grp[eintLocal].nbat, eatAll,
                               nbv->nbs, mdatoms, fr->cginfo);


    /* do local pair search */
    nbnxn_make_pairlist(nbv->nbs, nbv->grp[eintLocal].nbat,
                       &top->excls,
                       fr->ic->rlist,
                       nbv->min_ci_balanced,
                       &nbv->grp[eintLocal].nbl_lists,
                       eintLocal,
                       nbv->grp[eintLocal].kernel_type,
                       nrnb);


     copy_rvec(fr->mu_tot[0], mu_tot);

    /* Reset energies */
    reset_enerdata(&(inputrec->opts), fr, 1, enerd, MASTER(cr));
    clear_rvecs(SHIFTS, fr->fshift);


    /* Start the force cycle counter.
     * This counter is stopped in do_forcelow_level.
     * No parallel communication should occur while this counter is running,
     * since that will interfere with the dynamic load balancing.
     */
    /* Reset forces for which the virial is calculated separately:
     * PME/Ewald forces if necessary */
    fr->f_novirsum = fr->f_novirsum_alloc;
    GMX_BARRIER(cr->mpi_comm_mygroup);
    clear_rvecs(homenr, fr->f_novirsum+start);
    GMX_BARRIER(cr->mpi_comm_mygroup);

    /* Clear the short- and long-range forces */
    clear_rvecs(fr->natoms_force_constr, f);

   // clear_rvec(fr->vir_diag_posres);

    GMX_BARRIER(cr->mpi_comm_mygroup);



    /* Compute the bonded and non-bonded energies and optionally forces */
    do_force_lowlevel(fplog, step, fr, inputrec, &(top->idef),
                      cr, nrnb, wcycle, mdatoms, &(inputrec->opts),
                      x, hist, f, f, enerd, fcd, mtop, top,
                      &(top->atomtypes), bBornRadii, box,
                      inputrec->fepvals, lambda, graph, &(top->excls), fr->mu_tot,
                      flags, &cycles_pme);


    /* Maybe we should move this into do_force_lowlevel */
    do_nb_verlet(fr, fr->ic, enerd, flags, eintLocal, enbvClearFYes,
                     nrnb, wcycle);



   /* Add all the non-bonded force to the normal force array.
    * This can be split into a local a non-local part when overlapping
    * communication with calculation with domain decomposition.
    */
    nbnxn_atomdata_add_nbat_f_to_f(nbv->nbs, eatAll, nbv->grp[eintLocal].nbat, f);

    /* if there are multiple fshift output buffers reduce them */
    nbnxn_atomdata_add_nbat_fshift_to_fshift(nbv->grp[eintLocal].nbat,
                                                     fr->fshift);

    GMX_BARRIER(cr->mpi_comm_mygroup);


    sum_forces(mdatoms->start, mdatoms->start+mdatoms->homenr,
                         f, fr->f_novirsum);

    /* Sum the potential energy terms from group contributions */
    sum_epot(&(inputrec->opts), &(enerd->grpp), enerd->term);
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
