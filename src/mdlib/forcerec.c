#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include "assert.h"
#include "sysstuff.h"
#include "typedefs.h"
#include "vec.h"
#include "maths.h"
#include "macros.h"
#include "smalloc.h"
#include "macros.h"
#include "gmx_fatal.h"
#include "gmx_fatal_collective.h"
#include "physics.h"
#include "force.h"
#include "tables.h"
#include "nonbonded.h"
#include "invblock.h"
#include "names.h"
#include "network.h"
#include "pbc.h"
#include "ns.h"
#include "mshift.h"
#include "txtdump.h"
#include "coulomb.h"
#include "md_support.h"
#include "md_logging.h"
#include "qmmm.h"
#include "copyrite.h"
#include "mtop_util.h"
#include "nbnxn_search.h"
#include "nbnxn_atomdata.h"
#include "nbnxn_consts.h"
#include "statutil.h"
#include "gmx_omp_nthreads.h"

#ifdef _MSC_VER
/* MSVC definition for __cpuid() */
#include <intrin.h>
#endif


t_forcerec *mk_forcerec(void) // called
{
    t_forcerec *fr;

    snew(fr, 1);

    return fr;
}


static real *mk_nbfp(const gmx_ffparams_t *idef, gmx_bool bBHAM)
{// called
    real *nbfp;
    int   i, j, k, atnr;
    i = j = k = 0;
    atnr = idef->atnr;
    snew(nbfp, 2*atnr*atnr);
    /* nbfp now includes the 6.0/12.0 derivative prefactors */
    C6(nbfp, atnr, i, j)   = idef->iparams[k].lj.c6*6.0;
    C12(nbfp, atnr, i, j)  = idef->iparams[k].lj.c12*12.0;

    return nbfp;
}

/* This routine sets fr->solvent_opt to the most common solvent in the
 * system, e.g. esolSPC or esolTIP4P. It will also mark each charge group in
 * the fr->solvent_type array with the correct type (or esolNO).
 *
 * Charge groups that fulfill the conditions but are not identical to the
 * most common one will be marked as esolNO in the solvent_type array.
 *
 * TIP3p is identical to SPC for these purposes, so we call it
 * SPC in the arrays (Apologies to Bill Jorgensen ;-)
 *
 * NOTE: QM particle should not
 * become an optimized solvent. Not even if there is only one charge
 * group in the Qm
 */

typedef struct
{
    int    model;
    int    count;
    int    vdwtype[4];
    real   charge[4];
} solvent_parameters_t;

static void
check_solvent(FILE  *                fp,
              const gmx_mtop_t  *    mtop,
              t_forcerec  *          fr,
              cginfo_mb_t           *cginfo_mb)
{//called
    fr->nWatMol = 0;
    SET_CGINFO_SOLOPT(cginfo_mb[0].cginfo[0], esolNO);
    SET_CGINFO_SOLOPT(cginfo_mb[0].cginfo[1], esolNO);
    SET_CGINFO_SOLOPT(cginfo_mb[0].cginfo[2], esolNO);

    fr->solvent_opt = esolNO;
}

enum {
    acNONE = 0, acCONSTRAINT, acSETTLE
};

static cginfo_mb_t *init_cginfo_mb(FILE *fplog, const gmx_mtop_t *mtop,
                                   t_forcerec *fr, gmx_bool bNoSolvOpt,
                                   gmx_bool *bExcl_IntraCGAll_InterCGNone)
{//called
    const t_block        *cgs;
    const gmx_moltype_t  *molt;
    const gmx_molblock_t *molb;
    cginfo_mb_t          *cginfo_mb;
    int                  *cginfo;
    int                   mb, m, ncg_tot, cg, a0, a1, gid, ai, j, aj;

    ncg_tot = ncg_mtop(mtop);
    snew(cginfo_mb, mtop->nmolblock);


    *bExcl_IntraCGAll_InterCGNone = TRUE;

        molb = &mtop->molblock[0];
        molt = &mtop->moltype[molb->type];
        cgs  = &molt->cgs;

        /* Check if the cginfo is identical for all molecules in this block.
         * If so, we only need an array of the size of one molecule.
         * Otherwise we make an array of #mol times #cgs per molecule.
         */


        cginfo_mb[0].cg_start = 0;
        cginfo_mb[0].cg_end   = molb->nmol*cgs->nr;
        cginfo_mb[0].cg_mod   = cgs->nr;
        snew(cginfo_mb[0].cginfo, cginfo_mb[0].cg_mod);
        cginfo = cginfo_mb[0].cginfo;

        /* Set constraints flags for constrained atoms */

            for (cg = 0; cg < cgs->nr; cg++) // 3
            {
                a0 = cgs->index[cg];
                a1 = cgs->index[cg+1];

                /* Store the energy group in cginfo */
                gid = ggrpnr(&mtop->groups, egcENER, a0);
                SET_CGINFO_GID(cginfo[cg], gid);

                SET_CGINFO_EXCL_INTRA(cginfo[cg]);
                
                SET_CGINFO_HAS_Q(cginfo[cg]);
                /* Store the charge group size */
                SET_CGINFO_NATOMS(cginfo[cg], 1);

            }


    check_solvent(fplog, mtop, fr, cginfo_mb);

    for (cg = 0; cg < cginfo_mb[0].cg_mod; cg++) // 3
    {
          SET_CGINFO_SOLOPT(cginfo_mb[0].cginfo[cg], esolNO);
    }

    return cginfo_mb;
}

static int *cginfo_expand(int nmb, cginfo_mb_t *cgi_mb)
{//called
    int  ncg, cg;
    int *cginfo;

    ncg = cgi_mb[nmb-1].cg_end;
    snew(cginfo, ncg);
    for (cg = 0; cg < ncg; cg++) // 3000
    {
        cginfo[cg] =
            cgi_mb[0].cginfo[(cg - cgi_mb[0].cg_start) % cgi_mb[0].cg_mod];
    }

    return cginfo;
}

static void set_chargesum(FILE *log, t_forcerec *fr, const gmx_mtop_t *mtop)
{//called
    double         qsum, q2sum, q;
    int            nmol, i;
    const t_atoms *atoms;
// TODO ENAS RIO charges being added here
    qsum  = 0;
    q2sum = 0;
    nmol  = mtop->molblock[0].nmol;
    atoms = &mtop->moltype[mtop->molblock[0].type].atoms;
        for (i = 0; i < atoms->nr; i++)
        {
            q      = atoms->atom[i].q;
            qsum  += nmol*q;
            q2sum += nmol*q*q;
        }
    fr->qsum[0]  = qsum;
    fr->q2sum[0] = q2sum;
    fr->qsum[1]  = fr->qsum[0];
    fr->q2sum[1] = fr->q2sum[0];
    fprintf(log, "System total charge: %.3f\n", fr->qsum[0]);
}

static void make_nbf_tables(FILE *fp, 
                            t_forcerec *fr, real rtab,
                            const t_commrec *cr,
                            const char *tabfn, char *eg1, char *eg2,
                            t_nblists *nbl)
{ //called
    char buf[STRLEN];
    int  i, j;

    sprintf(buf, "%s", tabfn);
    nbl->table_elec_vdw = make_tables(fp, fr, MASTER(cr), buf, rtab, 0);
    /* Copy the contents of the table to separate coulomb and LJ tables too,
     * to improve cache performance.
     */
    /* For performance reasons we want
     * the table data to be aligned to 16-byte. The pointers could be freed
     * but currently aren't.
     */
    nbl->table_elec.interaction   = GMX_TABLE_INTERACTION_ELEC;
    nbl->table_elec.format        = nbl->table_elec_vdw.format;
    nbl->table_elec.r             = nbl->table_elec_vdw.r;
    nbl->table_elec.n             = nbl->table_elec_vdw.n;
    nbl->table_elec.scale         = nbl->table_elec_vdw.scale;
    nbl->table_elec.scale_exp     = nbl->table_elec_vdw.scale_exp;
    nbl->table_elec.formatsize    = nbl->table_elec_vdw.formatsize;
    nbl->table_elec.ninteractions = 1;
    nbl->table_elec.stride        = nbl->table_elec.formatsize * nbl->table_elec.ninteractions;
    snew_aligned(nbl->table_elec.data, nbl->table_elec.stride*(nbl->table_elec.n+1), 32);

    nbl->table_vdw.interaction   = GMX_TABLE_INTERACTION_VDWREP_VDWDISP;
    nbl->table_vdw.format        = nbl->table_elec_vdw.format;
    nbl->table_vdw.r             = nbl->table_elec_vdw.r;
    nbl->table_vdw.n             = nbl->table_elec_vdw.n;
    nbl->table_vdw.scale         = nbl->table_elec_vdw.scale;
    nbl->table_vdw.scale_exp     = nbl->table_elec_vdw.scale_exp;
    nbl->table_vdw.formatsize    = nbl->table_elec_vdw.formatsize;
    nbl->table_vdw.ninteractions = 2;
    nbl->table_vdw.stride        = nbl->table_vdw.formatsize * nbl->table_vdw.ninteractions;
    snew_aligned(nbl->table_vdw.data, nbl->table_vdw.stride*(nbl->table_vdw.n+1), 32);

    for (i = 0; i <= nbl->table_elec_vdw.n; i++) // 4000
    {
        for (j = 0; j < 4; j++)
        {
            nbl->table_elec.data[4*i+j] = nbl->table_elec_vdw.data[12*i+j];
        }
        for (j = 0; j < 8; j++)
        {
            nbl->table_vdw.data[8*i+j] = nbl->table_elec_vdw.data[12*i+4+j];
        }
    }
}

void forcerec_set_ranges(t_forcerec *fr,
                         int ncg_home, int ncg_force,
                         int natoms_force,
                         int natoms_force_constr, int natoms_f_novirsum)
{//called
    fr->cg0 = 0;
    fr->hcg = ncg_home;

    /* fr->ncg_force is unused in the standard code,
     * but it can be useful for modified code dealing with charge groups.
     */
    fr->ncg_force           = ncg_force;
    fr->natoms_force        = natoms_force;
    fr->natoms_force_constr = natoms_force_constr;
    fr->nalloc_force = over_alloc_dd(fr->natoms_force_constr);
    fr->f_novirsum_n = natoms_f_novirsum;
    fr->f_novirsum_nalloc = over_alloc_dd(fr->f_novirsum_n);
    srenew(fr->f_novirsum_alloc, fr->f_novirsum_nalloc);
}

real calc_ewaldcoeff(real rc, real dtol)
{ // called 
    real x = 5, low, high;
    int  n, i = 0;


    do
    {
        i++;
        x *= 2;
    }
    while (gmx_erfc(x*rc) > dtol);

    n    = i+60; /* search tolerance is 2^-60 */
    low  = 0;
    high = x;
    for (i = 0; i < n; i++)
    {
        x = (low+high)/2;
        if (gmx_erfc(x*rc) > dtol)
        {
            low = x;
        }
        else
        {
            high = x;
        }
    }
    return x;
}


static void init_forcerec_f_threads(t_forcerec *fr, int nenergrp)
{ // called
    int t, i;

    /* These thread local data structures are used for bondeds only */
    fr->nthreads = gmx_omp_nthreads_get(emntBonded);

    if (fr->nthreads > 1) // OpenMP threads used
    {
        snew(fr->f_t, fr->nthreads);
        /* Thread 0 uses the global force and energy arrays */
        for (t = 1; t < fr->nthreads; t++)
        {
            fr->f_t[t].f        = NULL;
            fr->f_t[t].f_nalloc = 0;
            snew(fr->f_t[t].fshift, SHIFTS);
            fr->f_t[t].grpp.nener = nenergrp*nenergrp;
            for (i = 0; i < egNR; i++)
            {
                snew(fr->f_t[t].grpp.ener[i], fr->f_t[t].grpp.nener);
            }
        }
    }
}



static void pick_nbnxn_resources(FILE                *fp,
                                 const t_commrec     *cr,
                                 gmx_bool             bDoNonbonded,
                                 gmx_bool            *bUseGPU,
                                 gmx_bool            *bEmulateGPU)
{//called
    gmx_bool bEmulateGPUEnvVarSet;
    char     gpu_err_str[STRLEN];

    *bUseGPU = FALSE;

    bEmulateGPUEnvVarSet = (getenv("GMX_EMULATE_GPU") != NULL);

    /* Run GPU emulation mode if GMX_EMULATE_GPU is defined. Because
     * GPUs (currently) only handle non-bonded calculations, we will
     * automatically switch to emulation if non-bonded calculations are
     * turned off via GMX_NO_NONBONDED - this is the simple and elegant
     * way to turn off GPU initialization, data movement, and cleanup.
     *
     * GPU emulation can be useful to assess the performance one can expect by
     * adding GPU(s) to the machine. The conditional below allows this even
     * if mdrun is compiled without GPU acceleration support.
     * Note that you should freezing the system as otherwise it will explode.
     */
    *bEmulateGPU = FALSE; 

}

gmx_bool uses_simple_tables(int                 cutoff_scheme,
                            nonbonded_verlet_t *nbv,
                            int                 group)
{//called
    gmx_bool bUsesSimpleTables = TRUE;

    assert(NULL != nbv && NULL != nbv->grp);
    int grp_index         = (group < 0) ? 0 : (nbv->ngrp - 1);
    bUsesSimpleTables = TRUE; 
    return bUsesSimpleTables;
}

static void init_ewald_f_table(interaction_const_t *ic,
                               gmx_bool             bUsesSimpleTables,
                               real                 rtab)
{//called
    real maxr;
    /* With a spacing of 0.0005 we are at the force summation accuracy
     * for the SSE kernels for "normal" atomistic simulations.
     */
    ic->tabq_scale = ewald_spline3_table_scale(ic->ewaldcoeff,
                                                   ic->rcoulomb);

    maxr           = (rtab > ic->rcoulomb) ? rtab : ic->rcoulomb;
    ic->tabq_size  = (int)(maxr*ic->tabq_scale) + 2;

    sfree_aligned(ic->tabq_coul_FDV0);
    sfree_aligned(ic->tabq_coul_F);
    sfree_aligned(ic->tabq_coul_V);

    /* Create the original table data in FDV0 */
    snew_aligned(ic->tabq_coul_FDV0, ic->tabq_size*4, 32);
    snew_aligned(ic->tabq_coul_F, ic->tabq_size, 32);
    snew_aligned(ic->tabq_coul_V, ic->tabq_size, 32);
    table_spline3_fill_ewald_lr(ic->tabq_coul_F, ic->tabq_coul_V, ic->tabq_coul_FDV0,
                                ic->tabq_size, 1/ic->tabq_scale, ic->ewaldcoeff);
}

void init_interaction_const(FILE                 *fp,
                            interaction_const_t **interaction_const,
                            const t_forcerec     *fr,
                            real                  rtab)
{//called
    interaction_const_t *ic;
    gmx_bool             bUsesSimpleTables = TRUE;

    snew(ic, 1);

    /* Just allocate something so we can free it */
    snew_aligned(ic->tabq_coul_FDV0, 16, 32);
    snew_aligned(ic->tabq_coul_F, 16, 32);
    snew_aligned(ic->tabq_coul_V, 16, 32);

    ic->rlist       = fr->rlist;
    ic->rlistlong   = fr->rlistlong;

    /* Lennard-Jones */
    ic->rvdw        = fr->rvdw;
    ic->sh_invrc6 = pow(ic->rvdw, -6.0);

    /* Electrostatics */
    ic->eeltype     = fr->eeltype;
    ic->rcoulomb    = fr->rcoulomb;
    ic->epsilon_r   = fr->epsilon_r;
    ic->epsfac      = fr->epsfac;

    /* Ewald */
    ic->ewaldcoeff  = fr->ewaldcoeff;

    ic->sh_ewald = gmx_erfc(ic->ewaldcoeff*ic->rcoulomb);
printf("\n** ic->sh_ewald = %e\n", ic->sh_ewald);
    /* Reaction-field */

        /* For plain cut-off we might use the reaction-field kernels */
        ic->epsilon_rf = ic->epsilon_r;
        ic->k_rf       = 0;
        ic->c_rf   = 1/ic->rcoulomb;

        fprintf(fp, "Potential shift: LJ r^-12: %.3f r^-6 %.3f",
                sqr(ic->sh_invrc6), ic->sh_invrc6);
        if (EEL_PME(ic->eeltype))// 0 EWALD, 1 PME
        {
            fprintf(fp, ", Ewald %.3e", ic->sh_ewald);
        }
        fprintf(fp, "\n");

    *interaction_const = ic;


    bUsesSimpleTables = uses_simple_tables(fr->cutoff_scheme, fr->nbv, -1);

    init_ewald_f_table(ic, bUsesSimpleTables, rtab);

    fprintf(fp, "Initialized non-bonded Ewald correction tables, spacing: %.2e size: %d\n\n",
                    1/ic->tabq_scale, ic->tabq_size);
}

static void init_nb_verlet(FILE                *fp,
                           nonbonded_verlet_t **nb_verlet,
                           const t_inputrec    *ir,
                           const t_forcerec    *fr,
                           const t_commrec     *cr,
                           const char          *nbpu_opt)
{//called
    nonbonded_verlet_t *nbv;
    int                 i;
    char               *env;
    gmx_bool            bEmulateGPU, bHybridGPURun = FALSE;


    snew(nbv, 1);

    pick_nbnxn_resources(fp, cr, 
                         fr->bNonbonded,
                         &nbv->bUseGPU,
                         &bEmulateGPU);

    nbv->nbs = NULL;
    nbv->ngrp = 1;
    nbv->grp[0].nbl_lists.nnbl = 0;
    nbv->grp[0].nbat           = NULL;
    nbv->grp[0].kernel_type    = nbnxnkNotSet;

    nbv->grp[0].ewald_excl  = ewaldexclTable;
    nbv->grp[0].kernel_type = nbnxnk4x4_PlainC;


    nbv->min_ci_balanced = 0;

    *nb_verlet = nbv;

    nbnxn_init_search(&nbv->nbs,
                      NULL,
                      gmx_omp_nthreads_get(emntNonbonded));


    nbnxn_init_pairlist_set(&nbv->grp[0].nbl_lists,
                                TRUE,
                                FALSE,
                                NULL, NULL);

    snew(nbv->grp[0].nbat, 1);
    nbnxn_atomdata_init(fp,
                           nbv->grp[0].nbat,
                           nbv->grp[0].kernel_type,
                           fr->ntype, fr->nbfp,
                           ir->opts.ngener,
                           gmx_omp_nthreads_get(emntNonbonded),
                           NULL, NULL);
}

void init_forcerec(FILE              *fp,
                   t_forcerec        *fr,
                   const t_inputrec  *ir,
                   const gmx_mtop_t  *mtop,
                   const t_commrec   *cr,
                   matrix             box,
                   gmx_bool           bMolEpot,
                   const char        *tabfn,
                   const char        *tabafn,
                   const char        *tabpfn,
                   const char        *tabbfn,
                   const char        *nbpu_opt,
                   gmx_bool           bNoSolvOpt,
                   real               print_force)
{//called
    int            i, j, m, natoms, ngrp, egi, egj;
    real           rtab;
    char          *env;
    double         dbl;
    rvec           box_size;
    const t_block *cgs;
    t_nblists     *nbl;
    int           *nm_ind, egp_flags;

    /* By default we turn acceleration on, but it might be turned off further down... */
    fr->use_cpu_acceleration = TRUE;

    fr->bDomDec = FALSE; 

    natoms = mtop->natoms;
    fr->n_tpi = 0;

    /* Copy AdResS parameters */
    fr->adress_type           = eAdressOff;
    fr->adress_do_hybridpairs = FALSE;

    /* Copy the user determined parameters */
    fr->userint1  = ir->userint1;
    fr->userint2  = ir->userint2;
    fr->userint3  = ir->userint3;
    fr->userint4  = ir->userint4;
    fr->userreal1 = ir->userreal1;
    fr->userreal2 = ir->userreal2;
    fr->userreal3 = ir->userreal3;
    fr->userreal4 = ir->userreal4;

    /* Shell stuff */
    fr->fc_stepsize = ir->fc_stepsize;

    /* Free energy */
    fr->efep        = ir->efep;
    fr->sc_alphavdw = ir->fepvals->sc_alpha;
    fr->sc_alphacoul  = 0;
    fr->sc_sigma6_min = 0; /* only needed when bScCoul is on */
    fr->sc_power      = ir->fepvals->sc_power;
    fr->sc_r_power    = ir->fepvals->sc_r_power;
    fr->sc_sigma6_def = pow(ir->fepvals->sc_sigma, 6);

    env = getenv("GMX_SCSIGMA_MIN");

    fr->bNonbonded = TRUE;
    fr->use_cpu_acceleration = FALSE;
    fprintf(fp,
                    "\nFound environment variable GMX_DISABLE_CPU_ACCELERATION.\n"
                    "Disabling all CPU architecture-specific (e.g. SSE2/SSE4/AVX) routines.\n\n");

    fr->bBHAM = (mtop->ffparams.functype[0] == F_BHAM);

    /* Check if we can/should do all-vs-all kernels */
    fr->bAllvsAll       = FALSE;
    fr->AllvsAll_work   = NULL;
    fr->AllvsAll_workgb = NULL;


    /* Neighbour searching stuff */
    fr->cutoff_scheme = ir->cutoff_scheme;
    fr->bGrid         = (ir->ns_type == ensGRID);
    fr->ePBC          = ir->ePBC;

    /* Determine if we will do PBC for distances in bonded interactions */

    fr->bMolPBC = TRUE;
    fr->bGB = (ir->implicit_solvent == eisGBSA);

    fr->rc_scaling = ir->refcoord_scaling;
    copy_rvec(ir->posres_com, fr->posres_com);
    copy_rvec(ir->posres_comB, fr->posres_comB);
    fr->rlist      = ir->rlist;
    fr->rlistlong  = ir->rlistlong;
    fr->eeltype    = ir->coulombtype;
    fr->vdwtype    = ir->vdwtype;

    fr->coulomb_modifier = ir->coulomb_modifier;
    fr->vdw_modifier     = ir->vdw_modifier;

    /* Electrostatics: Translate from interaction-setting-in-mdp-file to kernel interaction format */
    fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_EWALD;

    /* Vdw: Translate from mdp settings to kernel format */
    fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_LENNARDJONES;

    /* These start out identical to ir, but might be altered if we e.g. tabulate the interaction in the kernel */
    fr->nbkernel_elec_modifier    = fr->coulomb_modifier;
    fr->nbkernel_vdw_modifier     = fr->vdw_modifier;

    fr->bTwinRange = fr->rlistlong > fr->rlist;
    fr->bEwald     = (EEL_PME(fr->eeltype) || fr->eeltype == eelEWALD);

    fr->reppow     = mtop->ffparams.reppow;
    fr->bvdwtab  = FALSE;
    fr->bcoultab = FALSE;

    /* Tables are used for direct ewald sum */
        if (EEL_PME(ir->coulombtype)) // 0 EWALD, 1 PME
        {
            fprintf(fp, "Will do PME sum in reciprocal space.\n");

        }
        fr->ewaldcoeff = calc_ewaldcoeff(ir->rcoulomb, ir->ewald_rtol);
	printf("\n** ewaldcoeff: %e\n", fr->ewaldcoeff); 
        init_ewald_tab(&(fr->ewald_table), cr, ir, fp);
        fprintf(fp, "Using a Gaussian width (1/beta) of %g nm for Ewald\n",
                    1/fr->ewaldcoeff);
        /*
	../data/ThsndWP

         	 ** beta :: ewaldcoeff: 3.458911e+00

         	 ** kmax 29

	../data/ThsndWE

         	 ** beta :: ewaldcoeff: 3.458911e+00

         	 ** kmax 15

	/ Set the charge scaling /
	fr->epsfac = ONE_4PI_EPS0/fr->epsilon_r; // 138.802623674
         */



    /* Electrostatics */
    fr->epsilon_r       = ir->epsilon_r;
    fr->epsilon_rf      = ir->epsilon_rf;
    fr->fudgeQQ         = mtop->ffparams.fudgeQQ;
    fr->rcoulomb_switch = ir->rcoulomb_switch;
    fr->rcoulomb        = ir->rcoulomb;

    /* Parameters for generalized RF */
    fr->zsquare = 0.0;
    fr->temp    = 0.0;


    fr->bF_NoVirSum = TRUE; /*(EEL_FULL(fr->eeltype) ||
                       gmx_mtop_ftype_count(mtop, F_POSRES) > 0 ||
                       IR_ELEC_FIELD(*ir) ||
                       (fr->adress_icor != eAdressICOff)
                       );*/
    snew(fr->shift_vec, SHIFTS);
    snew(fr->fshift, SHIFTS);
    fr->ntype = mtop->ffparams.atnr;
    fr->nbfp  = mk_nbfp(&mtop->ffparams, fr->bBHAM);

    /* Copy the energy group exclusions */
    fr->egp_flags = ir->opts.egp_flags;

    /* Van der Waals stuff */
    fr->rvdw        = ir->rvdw;
    fr->rvdw_switch = ir->rvdw_switch;


    fprintf(fp, "Cut-off's:   NS: %g   Coulomb: %g   %s: %g\n",
                fr->rlist, fr->rcoulomb, fr->bBHAM ? "BHAM" : "LJ", fr->rvdw);

    fr->eDispCorr = ir->eDispCorr;


    fr->gb_epsilon_solvent = ir->gb_epsilon_solvent;

    /* Copy the GBSA data (radius, volume and surftens for each
     * atomtype) from the topology atomtype section to forcerec.
     */
    snew(fr->atype_radius, fr->ntype);
    snew(fr->atype_vol, fr->ntype);
    snew(fr->atype_surftens, fr->ntype);
    snew(fr->atype_gb_radius, fr->ntype);
    snew(fr->atype_S_hct, fr->ntype);
    fr->atype_radius[0] = mtop->atomtypes.radius[0];
    fr->atype_vol[0] = mtop->atomtypes.vol[0];
    fr->atype_surftens[0] = mtop->atomtypes.surftens[0];
    fr->atype_gb_radius[0] = mtop->atomtypes.gb_radius[0];
    fr->atype_S_hct[0] = mtop->atomtypes.S_hct[0];

    /* Set the charge scaling */
    fr->epsfac = ONE_4PI_EPS0/fr->epsilon_r;

    set_chargesum(fp, fr, mtop);

    /* if we are using LR electrostatics, and they are tabulated,
     * the tables will contain modified coulomb interactions.
     * Since we want to use the non-shifted ones for 1-4
     * coulombic interactions, we must have an extra set of tables.
     */

    /* Construct tables.
     * A little unnecessary to make both vdw and coul tables sometimes,
     * but what the heck... */

    egp_flags = ir->opts.egp_flags[GID(0, 0, ir->opts.ngener)];
    fr->nnblists = 1;

    snew(fr->nblists, fr->nnblists);

    /* This code automatically gives table length tabext without cut-off's,
     * in that case grompp should already have checked that we do not need
     * normal tables and we only generate tables for 1-4 interactions.
     */
    rtab = ir->rlistlong + ir->tabext;

    /* make tables for ordinary interactions */
    make_nbf_tables(fp, fr, rtab, cr, tabfn, NULL, NULL, &fr->nblists[0]);
    fr->tab14 = fr->nblists[0].table_elec_vdw;
    m = 1;



    /* When using particle decomposition, the effect of the second argument,
     * which sets fr->hcg, is corrected later in do_md and init_em.
     */
    forcerec_set_ranges(fr, ncg_mtop(mtop), ncg_mtop(mtop),
                            mtop->natoms, mtop->natoms, mtop->natoms);



    /* coarse load balancing vars */
    fr->t_fnbf    = 0.;
    fr->t_wait    = 0.;
    fr->timesteps = 0;

    /* Initialize neighbor search */
    init_ns(fp, cr, &fr->ns, fr, mtop, box);

    /* Initialize the thread working data for bonded interactions */
    init_forcerec_f_threads(fr, mtop->groups.grps[egcENER].nr);

    snew(fr->excl_load, fr->nthreads+1);

    init_nb_verlet(fp, &fr->nbv, ir, fr, cr, nbpu_opt);

    /* fr->ic is used both by verlet and group kernels (to some extent) now */
    init_interaction_const(fp, &fr->ic, fr, rtab);
}

#define pr_real(fp, r) fprintf(fp, "%s: %e\n",#r, r)
#define pr_int(fp, i)  fprintf((fp), "%s: %d\n",#i, i)
#define pr_bool(fp, b) fprintf((fp), "%s: %s\n",#b, bool_names[b])

void forcerec_set_excl_load(t_forcerec *fr,
                            const gmx_localtop_t *top, const t_commrec *cr)
{//called
    int        t;

    fr->excl_load[0] = 0;
    for (t = 1; t <= fr->nthreads; t++) // =OpenMP thread count
    {
        fr->excl_load[t] = 0;
    }
}
