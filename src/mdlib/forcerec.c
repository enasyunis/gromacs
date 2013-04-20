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
#include "domdec.h"
#include "partdec.h"
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

#include "types/nbnxn_cuda_types_ext.h"
#include "gpu_utils.h"
#include "nbnxn_cuda_data_mgmt.h"
#include "pmalloc_cuda.h"

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
    int  ncg, mb, cg;
    int *cginfo;

    ncg = cgi_mb[nmb-1].cg_end;
    snew(cginfo, ncg);
    mb = 0;
    for (cg = 0; cg < ncg; cg++)
    {
        while (cg >= cgi_mb[mb].cg_end)
        {
            mb++;
        }
        cginfo[cg] =
            cgi_mb[mb].cginfo[(cg - cgi_mb[mb].cg_start) % cgi_mb[mb].cg_mod];
    }

    return cginfo;
}

static void set_chargesum(FILE *log, t_forcerec *fr, const gmx_mtop_t *mtop)
{//called
    double         qsum, q2sum, q;
    int            mb, nmol, i;
    const t_atoms *atoms;

    qsum  = 0;
    q2sum = 0;
    for (mb = 0; mb < mtop->nmolblock; mb++)
    {
        nmol  = mtop->molblock[mb].nmol;
        atoms = &mtop->moltype[mtop->molblock[mb].type].atoms;
        for (i = 0; i < atoms->nr; i++)
        {
            q      = atoms->atom[i].q;
            qsum  += nmol*q;
            q2sum += nmol*q*q;
        }
    }
    fr->qsum[0]  = qsum;
    fr->q2sum[0] = q2sum;
    if (fr->efep != efepNO)
    {
        qsum  = 0;
        q2sum = 0;
        for (mb = 0; mb < mtop->nmolblock; mb++)
        {
            nmol  = mtop->molblock[mb].nmol;
            atoms = &mtop->moltype[mtop->molblock[mb].type].atoms;
            for (i = 0; i < atoms->nr; i++)
            {
                q      = atoms->atom[i].qB;
                qsum  += nmol*q;
                q2sum += nmol*q*q;
            }
            fr->qsum[1]  = qsum;
            fr->q2sum[1] = q2sum;
        }
    }
    else
    {
        fr->qsum[1]  = fr->qsum[0];
        fr->q2sum[1] = fr->q2sum[0];
    }
    if (log)
    {
        if (fr->efep == efepNO)
        {
            fprintf(log, "System total charge: %.3f\n", fr->qsum[0]);
        }
        else
        {
            fprintf(log, "System total charge, top. A: %.3f top. B: %.3f\n",
                    fr->qsum[0], fr->qsum[1]);
        }
    }
}

void update_forcerec(FILE *log, t_forcerec *fr, matrix box)
{//called
    if (fr->eeltype == eelGRF)
    {
        calc_rffac(NULL, fr->eeltype, fr->epsilon_r, fr->epsilon_rf,
                   fr->rcoulomb, fr->temp, fr->zsquare, box,
                   &fr->kappa, &fr->k_rf, &fr->c_rf);
    }
}


static void make_nbf_tables(FILE *fp, const output_env_t oenv,
                            t_forcerec *fr, real rtab,
                            const t_commrec *cr,
                            const char *tabfn, char *eg1, char *eg2,
                            t_nblists *nbl)
{ //called
    char buf[STRLEN];
    int  i, j;

    if (tabfn == NULL)
    {
        if (debug)
        {
            fprintf(debug, "No table file name passed, can not read table, can not do non-bonded interactions\n");
        }
        return;
    }

    sprintf(buf, "%s", tabfn);
    if (eg1 && eg2)
    {
        /* Append the two energy group names */
        sprintf(buf + strlen(tabfn) - strlen(ftp2ext(efXVG)) - 1, "_%s_%s.%s",
                eg1, eg2, ftp2ext(efXVG));
    }
    nbl->table_elec_vdw = make_tables(fp, oenv, fr, MASTER(cr), buf, rtab, 0);
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

    for (i = 0; i <= nbl->table_elec_vdw.n; i++)
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

static void count_tables(int ftype1, int ftype2, const gmx_mtop_t *mtop,
                         int *ncount, int **count)
{//called
    const gmx_moltype_t *molt;
    const t_ilist       *il;
    int                  mt, ftype, stride, i, j, tabnr;

    for (mt = 0; mt < mtop->nmoltype; mt++)
    {
        molt = &mtop->moltype[mt];
        for (ftype = 0; ftype < F_NRE; ftype++)
        {
            if (ftype == ftype1 || ftype == ftype2)
            {
                il     = &molt->ilist[ftype];
                stride = 1 + NRAL(ftype);
                for (i = 0; i < il->nr; i += stride)
                {
                    tabnr = mtop->ffparams.iparams[il->iatoms[i]].tab.table;
                    if (tabnr < 0)
                    {
                        gmx_fatal(FARGS, "A bonded table number is smaller than 0: %d\n", tabnr);
                    }
                    if (tabnr >= *ncount)
                    {
                        srenew(*count, tabnr+1);
                        for (j = *ncount; j < tabnr+1; j++)
                        {
                            (*count)[j] = 0;
                        }
                        *ncount = tabnr+1;
                    }
                    (*count)[tabnr]++;
                }
            }
        }
    }
}

static bondedtable_t *make_bonded_tables(FILE *fplog,
                                         int ftype1, int ftype2,
                                         const gmx_mtop_t *mtop,
                                         const char *basefn, const char *tabext)
{ //called
    int            i, ncount, *count;
    char           tabfn[STRLEN];
    bondedtable_t *tab;

    tab = NULL;

    ncount = 0;
    count  = NULL;
    count_tables(ftype1, ftype2, mtop, &ncount, &count);

    if (ncount > 0)
    {
        snew(tab, ncount);
        for (i = 0; i < ncount; i++)
        {
            if (count[i] > 0)
            {
                sprintf(tabfn, "%s", basefn);
                sprintf(tabfn + strlen(basefn) - strlen(ftp2ext(efXVG)) - 1, "_%s%d.%s",
                        tabext, i, ftp2ext(efXVG));
                tab[i] = make_bonded_table(fplog, tabfn, NRAL(ftype1)-2);
            }
        }
        sfree(count);
    }

    return tab;
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

    if (fr->natoms_force_constr > fr->nalloc_force)
    {
        fr->nalloc_force = over_alloc_dd(fr->natoms_force_constr);

        if (fr->bTwinRange)
        {
            srenew(fr->f_twin, fr->nalloc_force);
        }
    }

    if (fr->bF_NoVirSum)
    {
        fr->f_novirsum_n = natoms_f_novirsum;
        if (fr->f_novirsum_n > fr->f_novirsum_nalloc)
        {
            fr->f_novirsum_nalloc = over_alloc_dd(fr->f_novirsum_n);
            srenew(fr->f_novirsum_alloc, fr->f_novirsum_nalloc);
        }
    }
    else
    {
        fr->f_novirsum_n = 0;
    }
}

static real cutoff_inf(real cutoff)
{ //called 
    if (cutoff == 0)
    {
        cutoff = GMX_CUTOFF_INF;
    }

    return cutoff;
}


gmx_bool can_use_allvsall(const t_inputrec *ir, const gmx_mtop_t *mtop,
                          gmx_bool bPrintNote, t_commrec *cr, FILE *fp)
{ //called 
    gmx_bool bAllvsAll;

    bAllvsAll =
        (
            ir->rlist == 0            &&
            ir->rcoulomb == 0         &&
            ir->rvdw == 0             &&
            ir->ePBC == epbcNONE      &&
            ir->vdwtype == evdwCUT    &&
            ir->coulombtype == eelCUT &&
            ir->efep == efepNO        &&
            (ir->implicit_solvent == eisNO ||
             (ir->implicit_solvent == eisGBSA && (ir->gb_algorithm == egbSTILL ||
                                                  ir->gb_algorithm == egbHCT   ||
                                                  ir->gb_algorithm == egbOBC))) &&
            getenv("GMX_NO_ALLVSALL") == NULL
        );

    if (bAllvsAll && ir->opts.ngener > 1)
    {
        const char *note = "NOTE: Can not use all-vs-all force loops, because there are multiple energy monitor groups; you might get significantly higher performance when using only a single energy monitor group.\n";

        if (bPrintNote)
        {
            if (MASTER(cr))
            {
                fprintf(stderr, "\n%s\n", note);
            }
            if (fp != NULL)
            {
                fprintf(fp, "\n%s\n", note);
            }
        }
        bAllvsAll = FALSE;
    }

    if (bAllvsAll && fp && MASTER(cr))
    {
        fprintf(fp, "\nUsing accelerated all-vs-all kernels.\n\n");
    }

    return bAllvsAll;
}


static void init_forcerec_f_threads(t_forcerec *fr, int nenergrp)
{ // called
    int t, i;

    /* These thread local data structures are used for bondeds only */
    fr->nthreads = gmx_omp_nthreads_get(emntBonded);

    if (fr->nthreads > 1)
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


static void pick_nbnxn_kernel(FILE                *fp,
                              const t_commrec     *cr,
                              const gmx_hw_info_t *hwinfo,
                              gmx_bool             use_cpu_acceleration,
                              gmx_bool             bUseGPU,
                              gmx_bool             bEmulateGPU,
                              const t_inputrec    *ir,
                              int                 *kernel_type,
                              int                 *ewald_excl,
                              gmx_bool             bDoNonbonded)
{ //called 
    assert(kernel_type);

    *kernel_type = nbnxnkNotSet;
    *ewald_excl  = ewaldexclTable;


    *kernel_type = nbnxnk4x4_PlainC;

}

static void pick_nbnxn_resources(FILE                *fp,
                                 const t_commrec     *cr,
                                 const gmx_hw_info_t *hwinfo,
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
    *bEmulateGPU = (bEmulateGPUEnvVarSet ||
                    (!bDoNonbonded && hwinfo->bCanUseGPU));

    /* Enable GPU mode when GPUs are available or no GPU emulation is requested.
     */
    if (hwinfo->bCanUseGPU && !(*bEmulateGPU))
    {
        /* Each PP node will use the intra-node id-th device from the
         * list of detected/selected GPUs. */
        if (!init_gpu(cr->rank_pp_intranode, gpu_err_str, &hwinfo->gpu_info))
        {
            /* At this point the init should never fail as we made sure that
             * we have all the GPUs we need. If it still does, we'll bail. */
            gmx_fatal(FARGS, "On node %d failed to initialize GPU #%d: %s",
                      cr->nodeid,
                      get_gpu_device_id(&hwinfo->gpu_info, cr->rank_pp_intranode),
                      gpu_err_str);
        }

        /* Here we actually turn on hardware GPU acceleration */
        *bUseGPU = TRUE;
    }
}

gmx_bool uses_simple_tables(int                 cutoff_scheme,
                            nonbonded_verlet_t *nbv,
                            int                 group)
{//called
    gmx_bool bUsesSimpleTables = TRUE;

    assert(NULL != nbv && NULL != nbv->grp);
    int grp_index         = (group < 0) ? 0 : (nbv->ngrp - 1);
    bUsesSimpleTables = nbnxn_kernel_pairlist_simple(nbv->grp[grp_index].kernel_type);
    return bUsesSimpleTables;
}

static void init_ewald_f_table(interaction_const_t *ic,
                               gmx_bool             bUsesSimpleTables,
                               real                 rtab)
{//called
    real maxr;

    if (bUsesSimpleTables)
    {
        /* With a spacing of 0.0005 we are at the force summation accuracy
         * for the SSE kernels for "normal" atomistic simulations.
         */
        ic->tabq_scale = ewald_spline3_table_scale(ic->ewaldcoeff,
                                                   ic->rcoulomb);

        maxr           = (rtab > ic->rcoulomb) ? rtab : ic->rcoulomb;
        ic->tabq_size  = (int)(maxr*ic->tabq_scale) + 2;
    }
    else
    {
        ic->tabq_size = GPU_EWALD_COULOMB_FORCE_TABLE_SIZE;
        /* Subtract 2 iso 1 to avoid access out of range due to rounding */
        ic->tabq_scale = (ic->tabq_size - 2)/ic->rcoulomb;
    }

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

void init_interaction_const_tables(FILE                *fp,
                                   interaction_const_t *ic,
                                   gmx_bool             bUsesSimpleTables,
                                   real                 rtab)
{//called
    real spacing;

    if (ic->eeltype == eelEWALD || EEL_PME(ic->eeltype))
    {
        init_ewald_f_table(ic, bUsesSimpleTables, rtab);

        if (fp != NULL)
        {
            fprintf(fp, "Initialized non-bonded Ewald correction tables, spacing: %.2e size: %d\n\n",
                    1/ic->tabq_scale, ic->tabq_size);
        }
    }
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
    if (fr->vdw_modifier == eintmodPOTSHIFT)
    {
        ic->sh_invrc6 = pow(ic->rvdw, -6.0);
    }
    else
    {
        ic->sh_invrc6 = 0;
    }

    /* Electrostatics */
    ic->eeltype     = fr->eeltype;
    ic->rcoulomb    = fr->rcoulomb;
    ic->epsilon_r   = fr->epsilon_r;
    ic->epsfac      = fr->epsfac;

    /* Ewald */
    ic->ewaldcoeff  = fr->ewaldcoeff;
    if (fr->coulomb_modifier == eintmodPOTSHIFT)
    {
        ic->sh_ewald = gmx_erfc(ic->ewaldcoeff*ic->rcoulomb);
    }
    else
    {
        ic->sh_ewald = 0;
    }

    /* Reaction-field */
    if (EEL_RF(ic->eeltype))
    {
        ic->epsilon_rf = fr->epsilon_rf;
        ic->k_rf       = fr->k_rf;
        ic->c_rf       = fr->c_rf;
    }
    else
    {
        /* For plain cut-off we might use the reaction-field kernels */
        ic->epsilon_rf = ic->epsilon_r;
        ic->k_rf       = 0;
        if (fr->coulomb_modifier == eintmodPOTSHIFT)
        {
            ic->c_rf   = 1/ic->rcoulomb;
        }
        else
        {
            ic->c_rf   = 0;
        }
    }

    if (fp != NULL)
    {
        fprintf(fp, "Potential shift: LJ r^-12: %.3f r^-6 %.3f",
                sqr(ic->sh_invrc6), ic->sh_invrc6);
        if (ic->eeltype == eelCUT)
        {
            fprintf(fp, ", Coulomb %.3f", ic->c_rf);
        }
        else if (EEL_PME(ic->eeltype))
        {
            fprintf(fp, ", Ewald %.3e", ic->sh_ewald);
        }
        fprintf(fp, "\n");
    }

    *interaction_const = ic;

    if (fr->nbv != NULL && fr->nbv->bUseGPU)
    {
        nbnxn_cuda_init_const(fr->nbv->cu_nbv, ic, fr->nbv);
    }

    bUsesSimpleTables = uses_simple_tables(fr->cutoff_scheme, fr->nbv, -1);
    init_interaction_const_tables(fp, ic, bUsesSimpleTables, rtab);
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

    nbnxn_alloc_t      *nb_alloc;
    nbnxn_free_t       *nb_free;

    snew(nbv, 1);

    pick_nbnxn_resources(fp, cr, fr->hwinfo,
                         fr->bNonbonded,
                         &nbv->bUseGPU,
                         &bEmulateGPU);

    nbv->nbs = NULL;

    nbv->ngrp = (DOMAINDECOMP(cr) ? 2 : 1);
    for (i = 0; i < nbv->ngrp; i++)
    {
        nbv->grp[i].nbl_lists.nnbl = 0;
        nbv->grp[i].nbat           = NULL;
        nbv->grp[i].kernel_type    = nbnxnkNotSet;

        if (i == 0) /* local */
        {
            pick_nbnxn_kernel(fp, cr, fr->hwinfo, fr->use_cpu_acceleration,
                              nbv->bUseGPU, bEmulateGPU,
                              ir,
                              &nbv->grp[i].kernel_type,
                              &nbv->grp[i].ewald_excl,
                              fr->bNonbonded);
        }
        else /* non-local */
        {
            if (nbpu_opt != NULL && strcmp(nbpu_opt, "gpu_cpu") == 0)
            {
                /* Use GPU for local, select a CPU kernel for non-local */
                pick_nbnxn_kernel(fp, cr, fr->hwinfo, fr->use_cpu_acceleration,
                                  FALSE, FALSE,
                                  ir,
                                  &nbv->grp[i].kernel_type,
                                  &nbv->grp[i].ewald_excl,
                                  fr->bNonbonded);

                bHybridGPURun = TRUE;
            }
            else
            {
                /* Use the same kernel for local and non-local interactions */
                nbv->grp[i].kernel_type = nbv->grp[0].kernel_type;
                nbv->grp[i].ewald_excl  = nbv->grp[0].ewald_excl;
            }
        }
    }

    if (nbv->bUseGPU)
    {
        /* init the NxN GPU data; the last argument tells whether we'll have
         * both local and non-local NB calculation on GPU */
        nbnxn_cuda_init(fp, &nbv->cu_nbv,
                        &fr->hwinfo->gpu_info, cr->rank_pp_intranode,
                        (nbv->ngrp > 1) && !bHybridGPURun);

        if ((env = getenv("GMX_NB_MIN_CI")) != NULL)
        {
            char *end;

            nbv->min_ci_balanced = strtol(env, &end, 10);
            if (!end || (*end != 0) || nbv->min_ci_balanced <= 0)
            {
                gmx_fatal(FARGS, "Invalid value passed in GMX_NB_MIN_CI=%s, positive integer required", env);
            }

            if (debug)
            {
                fprintf(debug, "Neighbor-list balancing parameter: %d (passed as env. var.)\n",
                        nbv->min_ci_balanced);
            }
        }
        else
        {
            nbv->min_ci_balanced = nbnxn_cuda_min_ci_balanced(nbv->cu_nbv);
            if (debug)
            {
                fprintf(debug, "Neighbor-list balancing parameter: %d (auto-adjusted to the number of GPU multi-processors)\n",
                        nbv->min_ci_balanced);
            }
        }
    }
    else
    {
        nbv->min_ci_balanced = 0;
    }

    *nb_verlet = nbv;

    nbnxn_init_search(&nbv->nbs,
                      DOMAINDECOMP(cr) ? &cr->dd->nc : NULL,
                      DOMAINDECOMP(cr) ? domdec_zones(cr->dd) : NULL,
                      gmx_omp_nthreads_get(emntNonbonded));

    for (i = 0; i < nbv->ngrp; i++)
    {
        if (nbv->grp[0].kernel_type == nbnxnk8x8x8_CUDA)
        {
            nb_alloc = &pmalloc;
            nb_free  = &pfree;
        }
        else
        {
            nb_alloc = NULL;
            nb_free  = NULL;
        }

        nbnxn_init_pairlist_set(&nbv->grp[i].nbl_lists,
                                nbnxn_kernel_pairlist_simple(nbv->grp[i].kernel_type),
                                /* 8x8x8 "non-simple" lists are ATM always combined */
                                !nbnxn_kernel_pairlist_simple(nbv->grp[i].kernel_type),
                                nb_alloc, nb_free);

        if (i == 0 ||
            nbv->grp[0].kernel_type != nbv->grp[i].kernel_type)
        {
            snew(nbv->grp[i].nbat, 1);
            nbnxn_atomdata_init(fp,
                                nbv->grp[i].nbat,
                                nbv->grp[i].kernel_type,
                                fr->ntype, fr->nbfp,
                                ir->opts.ngener,
                                nbnxn_kernel_pairlist_simple(nbv->grp[i].kernel_type) ? gmx_omp_nthreads_get(emntNonbonded) : 1,
                                nb_alloc, nb_free);
        }
        else
        {
            nbv->grp[i].nbat = nbv->grp[0].nbat;
        }
    }
}

void init_forcerec(FILE              *fp,
                   const output_env_t oenv,
                   t_forcerec        *fr,
                   t_fcdata          *fcd,
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
    int            i, j, m, natoms, ngrp, negp_pp, negptable, egi, egj;
    real           rtab;
    char          *env;
    double         dbl;
    rvec           box_size;
    const t_block *cgs;
    gmx_bool       bGenericKernelOnly;
    gmx_bool       bTab, bSep14tab, bNormalnblists;
    t_nblists     *nbl;
    int           *nm_ind, egp_flags;

    /* By default we turn acceleration on, but it might be turned off further down... */
    fr->use_cpu_acceleration = TRUE;

    fr->bDomDec = DOMAINDECOMP(cr);

    natoms = mtop->natoms;

    if (check_box(ir->ePBC, box))
    {
        gmx_fatal(FARGS, check_box(ir->ePBC, box));
    }

    /* Test particle insertion ? */
    if (EI_TPI(ir->eI))
    {
        /* Set to the size of the molecule to be inserted (the last one) */
        /* Because of old style topologies, we have to use the last cg
         * instead of the last molecule type.
         */
        cgs       = &mtop->moltype[mtop->molblock[mtop->nmolblock-1].type].cgs;
        fr->n_tpi = cgs->index[cgs->nr] - cgs->index[cgs->nr-1];
        if (fr->n_tpi != mtop->mols.index[mtop->mols.nr] - mtop->mols.index[mtop->mols.nr-1])
        {
            gmx_fatal(FARGS, "The molecule to insert can not consist of multiple charge groups.\nMake it a single charge group.");
        }
    }
    else
    {
        fr->n_tpi = 0;
    }

    /* Copy AdResS parameters */
    if (ir->bAdress)
    {
        fr->adress_type           = ir->adress->type;
        fr->adress_const_wf       = ir->adress->const_wf;
        fr->adress_ex_width       = ir->adress->ex_width;
        fr->adress_hy_width       = ir->adress->hy_width;
        fr->adress_icor           = ir->adress->icor;
        fr->adress_site           = ir->adress->site;
        fr->adress_ex_forcecap    = ir->adress->ex_forcecap;
        fr->adress_do_hybridpairs = ir->adress->do_hybridpairs;


        snew(fr->adress_group_explicit, ir->adress->n_energy_grps);
        for (i = 0; i < ir->adress->n_energy_grps; i++)
        {
            fr->adress_group_explicit[i] = ir->adress->group_explicit[i];
        }

        fr->n_adress_tf_grps = ir->adress->n_tf_grps;
        snew(fr->adress_tf_table_index, fr->n_adress_tf_grps);
        for (i = 0; i < fr->n_adress_tf_grps; i++)
        {
            fr->adress_tf_table_index[i] = ir->adress->tf_table_index[i];
        }
        copy_rvec(ir->adress->refs, fr->adress_refs);
    }
    else
    {
        fr->adress_type           = eAdressOff;
        fr->adress_do_hybridpairs = FALSE;
    }

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
    if (ir->fepvals->bScCoul)
    {
        fr->sc_alphacoul  = ir->fepvals->sc_alpha;
        fr->sc_sigma6_min = pow(ir->fepvals->sc_sigma_min, 6);
    }
    else
    {
        fr->sc_alphacoul  = 0;
        fr->sc_sigma6_min = 0; /* only needed when bScCoul is on */
    }
    fr->sc_power      = ir->fepvals->sc_power;
    fr->sc_r_power    = ir->fepvals->sc_r_power;
    fr->sc_sigma6_def = pow(ir->fepvals->sc_sigma, 6);

    env = getenv("GMX_SCSIGMA_MIN");
    if (env != NULL)
    {
        dbl = 0;
        sscanf(env, "%lf", &dbl);
        fr->sc_sigma6_min = pow(dbl, 6);
        if (fp)
        {
            fprintf(fp, "Setting the minimum soft core sigma to %g nm\n", dbl);
        }
    }

    fr->bNonbonded = TRUE;
    if (getenv("GMX_NO_NONBONDED") != NULL)
    {
        /* turn off non-bonded calculations */
        fr->bNonbonded = FALSE;
        md_print_warn(cr, fp,
                      "Found environment variable GMX_NO_NONBONDED.\n"
                      "Disabling nonbonded calculations.\n");
    }

    bGenericKernelOnly = FALSE;

    /* We now check in the NS code whether a particular combination of interactions
     * can be used with water optimization, and disable it if that is not the case.
     */

    if (getenv("GMX_NB_GENERIC") != NULL)
    {
        if (fp != NULL)
        {
            fprintf(fp,
                    "Found environment variable GMX_NB_GENERIC.\n"
                    "Disabling all interaction-specific nonbonded kernels, will only\n"
                    "use the slow generic ones in src/gmxlib/nonbonded/nb_generic.c\n\n");
        }
        bGenericKernelOnly = TRUE;
    }

    if (bGenericKernelOnly == TRUE)
    {
        bNoSolvOpt         = TRUE;
    }
// ENAS DISABLED ACCELRAtiON
        fr->use_cpu_acceleration = FALSE;
        if (fp != NULL)
        {
            fprintf(fp,
                    "\nFound environment variable GMX_DISABLE_CPU_ACCELERATION.\n"
                    "Disabling all CPU architecture-specific (e.g. SSE2/SSE4/AVX) routines.\n\n");
        }

    fr->bBHAM = (mtop->ffparams.functype[0] == F_BHAM);

    /* Check if we can/should do all-vs-all kernels */
    fr->bAllvsAll       = can_use_allvsall(ir, mtop, FALSE, NULL, NULL);
    fr->AllvsAll_work   = NULL;
    fr->AllvsAll_workgb = NULL;


    /* Neighbour searching stuff */
    fr->cutoff_scheme = ir->cutoff_scheme;
    fr->bGrid         = (ir->ns_type == ensGRID);
    fr->ePBC          = ir->ePBC;

    /* Determine if we will do PBC for distances in bonded interactions */
    if (fr->ePBC == epbcNONE)
    {
        fr->bMolPBC = FALSE;
    }
    else
    {
        if (!DOMAINDECOMP(cr))
        {
            /* The group cut-off scheme and SHAKE assume charge groups
             * are whole, but not using molpbc is faster in most cases.
             */
            if (fr->cutoff_scheme == ecutsGROUP ||
                (ir->eConstrAlg == econtSHAKE &&
                 (gmx_mtop_ftype_count(mtop, F_CONSTR) > 0 ||
                  gmx_mtop_ftype_count(mtop, F_CONSTRNC) > 0)))
            {
                fr->bMolPBC = ir->bPeriodicMols;
            }
            else
            {
                fr->bMolPBC = TRUE;
                if (getenv("GMX_USE_GRAPH") != NULL)
                {
                    fr->bMolPBC = FALSE;
                    if (fp)
                    {
                        fprintf(fp, "\nGMX_MOLPBC is set, using the graph for bonded interactions\n\n");
                    }
                }
            }
        }
        else
        {
            fr->bMolPBC = dd_bonded_molpbc(cr->dd, fr->ePBC);
        }
    }
    fr->bGB = (ir->implicit_solvent == eisGBSA);

    fr->rc_scaling = ir->refcoord_scaling;
    copy_rvec(ir->posres_com, fr->posres_com);
    copy_rvec(ir->posres_comB, fr->posres_comB);
    fr->rlist      = cutoff_inf(ir->rlist);
    fr->rlistlong  = cutoff_inf(ir->rlistlong);
    fr->eeltype    = ir->coulombtype;
    fr->vdwtype    = ir->vdwtype;

    fr->coulomb_modifier = ir->coulomb_modifier;
    fr->vdw_modifier     = ir->vdw_modifier;

    /* Electrostatics: Translate from interaction-setting-in-mdp-file to kernel interaction format */
    switch (fr->eeltype)
    {
        case eelCUT:
            fr->nbkernel_elec_interaction = (fr->bGB) ? GMX_NBKERNEL_ELEC_GENERALIZEDBORN : GMX_NBKERNEL_ELEC_COULOMB;
            break;

        case eelRF:
        case eelGRF:
        case eelRF_NEC:
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_REACTIONFIELD;
            break;

        case eelRF_ZERO:
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_REACTIONFIELD;
            fr->coulomb_modifier          = eintmodEXACTCUTOFF;
            break;

        case eelSWITCH:
        case eelSHIFT:
        case eelUSER:
        case eelENCADSHIFT:
        case eelPMESWITCH:
        case eelPMEUSER:
        case eelPMEUSERSWITCH:
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_CUBICSPLINETABLE;
            break;

        case eelPME:
        case eelEWALD:
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_EWALD;
            break;

        default:
            gmx_fatal(FARGS, "Unsupported electrostatic interaction: %s", eel_names[fr->eeltype]);
            break;
    }

    /* Vdw: Translate from mdp settings to kernel format */
    switch (fr->vdwtype)
    {
        case evdwCUT:
            if (fr->bBHAM)
            {
                fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_BUCKINGHAM;
            }
            else
            {
                fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_LENNARDJONES;
            }
            break;

        case evdwSWITCH:
        case evdwSHIFT:
        case evdwUSER:
        case evdwENCADSHIFT:
            fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_CUBICSPLINETABLE;
            break;

        default:
            gmx_fatal(FARGS, "Unsupported vdw interaction: %s", evdw_names[fr->vdwtype]);
            break;
    }

    /* These start out identical to ir, but might be altered if we e.g. tabulate the interaction in the kernel */
    fr->nbkernel_elec_modifier    = fr->coulomb_modifier;
    fr->nbkernel_vdw_modifier     = fr->vdw_modifier;

    fr->bTwinRange = fr->rlistlong > fr->rlist;
    fr->bEwald     = (EEL_PME(fr->eeltype) || fr->eeltype == eelEWALD);

    fr->reppow     = mtop->ffparams.reppow;

    if (ir->cutoff_scheme == ecutsGROUP)
    {
        fr->bvdwtab    = (fr->vdwtype != evdwCUT ||
                          !gmx_within_tol(fr->reppow, 12.0, 10*GMX_DOUBLE_EPS));
        /* We have special kernels for standard Ewald and PME, but the pme-switch ones are tabulated above */
        fr->bcoultab   = !(fr->eeltype == eelCUT ||
                           fr->eeltype == eelEWALD ||
                           fr->eeltype == eelPME ||
                           fr->eeltype == eelRF ||
                           fr->eeltype == eelRF_ZERO);

        /* If the user absolutely wants different switch/shift settings for coul/vdw, it is likely
         * going to be faster to tabulate the interaction than calling the generic kernel.
         */
        if (fr->nbkernel_elec_modifier == eintmodPOTSWITCH && fr->nbkernel_vdw_modifier == eintmodPOTSWITCH)
        {
            if ((fr->rcoulomb_switch != fr->rvdw_switch) || (fr->rcoulomb != fr->rvdw))
            {
                fr->bcoultab = TRUE;
            }
        }
        else if ((fr->nbkernel_elec_modifier == eintmodPOTSHIFT && fr->nbkernel_vdw_modifier == eintmodPOTSHIFT) ||
                 ((fr->nbkernel_elec_interaction == GMX_NBKERNEL_ELEC_REACTIONFIELD &&
                   fr->nbkernel_elec_modifier == eintmodEXACTCUTOFF &&
                   (fr->nbkernel_vdw_modifier == eintmodPOTSWITCH || fr->nbkernel_vdw_modifier == eintmodPOTSHIFT))))
        {
            if (fr->rcoulomb != fr->rvdw)
            {
                fr->bcoultab = TRUE;
            }
        }

        if (getenv("GMX_REQUIRE_TABLES"))
        {
            fr->bvdwtab  = TRUE;
            fr->bcoultab = TRUE;
        }

        if (fp)
        {
            fprintf(fp, "Table routines are used for coulomb: %s\n", bool_names[fr->bcoultab]);
            fprintf(fp, "Table routines are used for vdw:     %s\n", bool_names[fr->bvdwtab ]);
        }

        if (fr->bvdwtab == TRUE)
        {
            fr->nbkernel_vdw_interaction = GMX_NBKERNEL_VDW_CUBICSPLINETABLE;
            fr->nbkernel_vdw_modifier    = eintmodNONE;
        }
        if (fr->bcoultab == TRUE)
        {
            fr->nbkernel_elec_interaction = GMX_NBKERNEL_ELEC_CUBICSPLINETABLE;
            fr->nbkernel_elec_modifier    = eintmodNONE;
        }
    }

    if (ir->cutoff_scheme == ecutsVERLET)
    {
        if (!gmx_within_tol(fr->reppow, 12.0, 10*GMX_DOUBLE_EPS))
        {
            gmx_fatal(FARGS, "Cut-off scheme %S only supports LJ repulsion power 12", ecutscheme_names[ir->cutoff_scheme]);
        }
        fr->bvdwtab  = FALSE;
        fr->bcoultab = FALSE;
    }

    /* Tables are used for direct ewald sum */
    if (fr->bEwald)
    {
        if (EEL_PME(ir->coulombtype))
        {
            if (fp)
            {
                fprintf(fp, "Will do PME sum in reciprocal space.\n");
            }
            if (ir->coulombtype == eelP3M_AD)
            {
                please_cite(fp, "Hockney1988");
                please_cite(fp, "Ballenegger2012");
            }
            else
            {
                please_cite(fp, "Essmann95a");
            }

            if (ir->ewald_geometry == eewg3DC)
            {
                if (fp)
                {
                    fprintf(fp, "Using the Ewald3DC correction for systems with a slab geometry.\n");
                }
                please_cite(fp, "In-Chul99a");
            }
        }
        fr->ewaldcoeff = calc_ewaldcoeff(ir->rcoulomb, ir->ewald_rtol);
        init_ewald_tab(&(fr->ewald_table), cr, ir, fp);
        if (fp)
        {
            fprintf(fp, "Using a Gaussian width (1/beta) of %g nm for Ewald\n",
                    1/fr->ewaldcoeff);
        }
    }

    /* Electrostatics */
    fr->epsilon_r       = ir->epsilon_r;
    fr->epsilon_rf      = ir->epsilon_rf;
    fr->fudgeQQ         = mtop->ffparams.fudgeQQ;
    fr->rcoulomb_switch = ir->rcoulomb_switch;
    fr->rcoulomb        = cutoff_inf(ir->rcoulomb);

    /* Parameters for generalized RF */
    fr->zsquare = 0.0;
    fr->temp    = 0.0;

    if (fr->eeltype == eelGRF)
    {
        init_generalized_rf(fp, mtop, ir, fr);
    }
    else if (fr->eeltype == eelSHIFT)
    {
        for (m = 0; (m < DIM); m++)
        {
            box_size[m] = box[m][m];
        }

        if ((fr->eeltype == eelSHIFT && fr->rcoulomb > fr->rcoulomb_switch))
        {
            set_shift_consts(fp, fr->rcoulomb_switch, fr->rcoulomb, box_size, fr);
        }
    }

    fr->bF_NoVirSum = (EEL_FULL(fr->eeltype) ||
                       gmx_mtop_ftype_count(mtop, F_POSRES) > 0 ||
                       IR_ELEC_FIELD(*ir) ||
                       (fr->adress_icor != eAdressICOff)
                       );

    if (fr->cutoff_scheme == ecutsGROUP &&
        ncg_mtop(mtop) > fr->cg_nalloc && !DOMAINDECOMP(cr))
    {
        /* Count the total number of charge groups */
        fr->cg_nalloc = ncg_mtop(mtop);
        srenew(fr->cg_cm, fr->cg_nalloc);
    }
    if (fr->shift_vec == NULL)
    {
        snew(fr->shift_vec, SHIFTS);
    }

    if (fr->fshift == NULL)
    {
        snew(fr->fshift, SHIFTS);
    }

    if (fr->nbfp == NULL)
    {
        fr->ntype = mtop->ffparams.atnr;
        fr->nbfp  = mk_nbfp(&mtop->ffparams, fr->bBHAM);
    }

    /* Copy the energy group exclusions */
    fr->egp_flags = ir->opts.egp_flags;

    /* Van der Waals stuff */
    fr->rvdw        = cutoff_inf(ir->rvdw);
    fr->rvdw_switch = ir->rvdw_switch;
    if ((fr->vdwtype != evdwCUT) && (fr->vdwtype != evdwUSER) && !fr->bBHAM)
    {
        if (fr->rvdw_switch >= fr->rvdw)
        {
            gmx_fatal(FARGS, "rvdw_switch (%f) must be < rvdw (%f)",
                      fr->rvdw_switch, fr->rvdw);
        }
        if (fp)
        {
            fprintf(fp, "Using %s Lennard-Jones, switch between %g and %g nm\n",
                    (fr->eeltype == eelSWITCH) ? "switched" : "shifted",
                    fr->rvdw_switch, fr->rvdw);
        }
    }

    if (fr->bBHAM && (fr->vdwtype == evdwSHIFT || fr->vdwtype == evdwSWITCH))
    {
        gmx_fatal(FARGS, "Switch/shift interaction not supported with Buckingham");
    }

    if (fp)
    {
        fprintf(fp, "Cut-off's:   NS: %g   Coulomb: %g   %s: %g\n",
                fr->rlist, fr->rcoulomb, fr->bBHAM ? "BHAM" : "LJ", fr->rvdw);
    }

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

    if (mtop->atomtypes.nr > 0)
    {
        for (i = 0; i < fr->ntype; i++)
        {
            fr->atype_radius[i] = mtop->atomtypes.radius[i];
        }
        for (i = 0; i < fr->ntype; i++)
        {
            fr->atype_vol[i] = mtop->atomtypes.vol[i];
        }
        for (i = 0; i < fr->ntype; i++)
        {
            fr->atype_surftens[i] = mtop->atomtypes.surftens[i];
        }
        for (i = 0; i < fr->ntype; i++)
        {
            fr->atype_gb_radius[i] = mtop->atomtypes.gb_radius[i];
        }
        for (i = 0; i < fr->ntype; i++)
        {
            fr->atype_S_hct[i] = mtop->atomtypes.S_hct[i];
        }
    }

    /* Generate the GB table if needed */
    if (fr->bGB)
    {
#ifdef GMX_DOUBLE
        fr->gbtabscale = 2000;
#else
        fr->gbtabscale = 500;
#endif

        fr->gbtabr = 100;
        fr->gbtab  = make_gb_table(fp, oenv, fr, tabpfn, fr->gbtabscale);

        init_gb(&fr->born, cr, fr, ir, mtop, ir->rgbradii, ir->gb_algorithm);

        /* Copy local gb data (for dd, this is done in dd_partition_system) */
        if (!DOMAINDECOMP(cr))
        {
            make_local_gb(cr, fr->born, ir->gb_algorithm);
        }
    }

    /* Set the charge scaling */
    if (fr->epsilon_r != 0)
    {
        fr->epsfac = ONE_4PI_EPS0/fr->epsilon_r;
    }
    else
    {
        /* eps = 0 is infinite dieletric: no coulomb interactions */
        fr->epsfac = 0;
    }

    /* Reaction field constants */
    if (EEL_RF(fr->eeltype))
    {
        calc_rffac(fp, fr->eeltype, fr->epsilon_r, fr->epsilon_rf,
                   fr->rcoulomb, fr->temp, fr->zsquare, box,
                   &fr->kappa, &fr->k_rf, &fr->c_rf);
    }

    set_chargesum(fp, fr, mtop);

    /* if we are using LR electrostatics, and they are tabulated,
     * the tables will contain modified coulomb interactions.
     * Since we want to use the non-shifted ones for 1-4
     * coulombic interactions, we must have an extra set of tables.
     */

    /* Construct tables.
     * A little unnecessary to make both vdw and coul tables sometimes,
     * but what the heck... */

    bTab = fr->bcoultab || fr->bvdwtab || fr->bEwald;

    bSep14tab = ((!bTab || fr->eeltype != eelCUT || fr->vdwtype != evdwCUT ||
                  fr->bBHAM || fr->bEwald) &&
                 (gmx_mtop_ftype_count(mtop, F_LJ14) > 0 ||
                  gmx_mtop_ftype_count(mtop, F_LJC14_Q) > 0 ||
                  gmx_mtop_ftype_count(mtop, F_LJC_PAIRS_NB) > 0));

    negp_pp   = ir->opts.ngener - ir->nwall;
    negptable = 0;
    if (!bTab)
    {
        bNormalnblists = TRUE;
        fr->nnblists   = 1;
    }
    else
    {
        bNormalnblists = (ir->eDispCorr != edispcNO);
        for (egi = 0; egi < negp_pp; egi++)
        {
            for (egj = egi; egj < negp_pp; egj++)
            {
                egp_flags = ir->opts.egp_flags[GID(egi, egj, ir->opts.ngener)];
                if (!(egp_flags & EGP_EXCL))
                {
                    if (egp_flags & EGP_TABLE)
                    {
                        negptable++;
                    }
                    else
                    {
                        bNormalnblists = TRUE;
                    }
                }
            }
        }
        if (bNormalnblists)
        {
            fr->nnblists = negptable + 1;
        }
        else
        {
            fr->nnblists = negptable;
        }
        if (fr->nnblists > 1)
        {
            snew(fr->gid2nblists, ir->opts.ngener*ir->opts.ngener);
        }
    }

    if (ir->adress)
    {
        fr->nnblists *= 2;
    }

    snew(fr->nblists, fr->nnblists);

    /* This code automatically gives table length tabext without cut-off's,
     * in that case grompp should already have checked that we do not need
     * normal tables and we only generate tables for 1-4 interactions.
     */
    rtab = ir->rlistlong + ir->tabext;

    if (bTab)
    {
        /* make tables for ordinary interactions */
        if (bNormalnblists)
        {
            make_nbf_tables(fp, oenv, fr, rtab, cr, tabfn, NULL, NULL, &fr->nblists[0]);
            if (ir->adress)
            {
                make_nbf_tables(fp, oenv, fr, rtab, cr, tabfn, NULL, NULL, &fr->nblists[fr->nnblists/2]);
            }
            if (!bSep14tab)
            {
                fr->tab14 = fr->nblists[0].table_elec_vdw;
            }
            m = 1;
        }
        else
        {
            m = 0;
        }
        if (negptable > 0)
        {
            /* Read the special tables for certain energy group pairs */
            nm_ind = mtop->groups.grps[egcENER].nm_ind;
            for (egi = 0; egi < negp_pp; egi++)
            {
                for (egj = egi; egj < negp_pp; egj++)
                {
                    egp_flags = ir->opts.egp_flags[GID(egi, egj, ir->opts.ngener)];
                    if ((egp_flags & EGP_TABLE) && !(egp_flags & EGP_EXCL))
                    {
                        nbl = &(fr->nblists[m]);
                        if (fr->nnblists > 1)
                        {
                            fr->gid2nblists[GID(egi, egj, ir->opts.ngener)] = m;
                        }
                        /* Read the table file with the two energy groups names appended */
                        make_nbf_tables(fp, oenv, fr, rtab, cr, tabfn,
                                        *mtop->groups.grpname[nm_ind[egi]],
                                        *mtop->groups.grpname[nm_ind[egj]],
                                        &fr->nblists[m]);
                        if (ir->adress)
                        {
                            make_nbf_tables(fp, oenv, fr, rtab, cr, tabfn,
                                            *mtop->groups.grpname[nm_ind[egi]],
                                            *mtop->groups.grpname[nm_ind[egj]],
                                            &fr->nblists[fr->nnblists/2+m]);
                        }
                        m++;
                    }
                    else if (fr->nnblists > 1)
                    {
                        fr->gid2nblists[GID(egi, egj, ir->opts.ngener)] = 0;
                    }
                }
            }
        }
    }
    if (bSep14tab)
    {
        /* generate extra tables with plain Coulomb for 1-4 interactions only */
        fr->tab14 = make_tables(fp, oenv, fr, MASTER(cr), tabpfn, rtab,
                                GMX_MAKETABLES_14ONLY);
    }

    /* Read AdResS Thermo Force table if needed */
    if (fr->adress_icor == eAdressICThermoForce)
    {
        /* old todo replace */

            /* load the default table */
            snew(fr->atf_tabs, 1);
            fr->atf_tabs[DEFAULT_TF_TABLE] = make_atf_table(fp, oenv, fr, tabafn, box);
    }

    /* Wall stuff */
    fr->nwall = ir->nwall;
    if (ir->nwall && ir->wall_type == ewtTABLE)
    {
        make_wall_tables(fp, oenv, ir, tabfn, &mtop->groups, fr);
    }

    if (fcd && tabbfn)
    {
        fcd->bondtab  = make_bonded_tables(fp,
                                           F_TABBONDS, F_TABBONDSNC,
                                           mtop, tabbfn, "b");
        fcd->angletab = make_bonded_tables(fp,
                                           F_TABANGLES, -1,
                                           mtop, tabbfn, "a");
        fcd->dihtab   = make_bonded_tables(fp,
                                           F_TABDIHS, -1,
                                           mtop, tabbfn, "d");
    }
    else
    {
        if (debug)
        {
            fprintf(debug, "No fcdata or table file name passed, can not read table, can not do bonded interactions\n");
        }
    }

    /* QM/MM initialization if requested
     */
    if (ir->bQMMM)
    {
        fprintf(stderr, "QM/MM calculation requested.\n");
    }

    fr->bQMMM      = ir->bQMMM;
    fr->qr         = mk_QMMMrec();

    /* Set all the static charge group info */
    fr->cginfo_mb = init_cginfo_mb(fp, mtop, fr, bNoSolvOpt,
                                   &fr->bExcl_IntraCGAll_InterCGNone);
    if (DOMAINDECOMP(cr))
    {
        fr->cginfo = NULL;
    }
    else
    {
        fr->cginfo = cginfo_expand(mtop->nmolblock, fr->cginfo_mb);
    }

    if (!DOMAINDECOMP(cr))
    {
        /* When using particle decomposition, the effect of the second argument,
         * which sets fr->hcg, is corrected later in do_md and init_em.
         */
        forcerec_set_ranges(fr, ncg_mtop(mtop), ncg_mtop(mtop),
                            mtop->natoms, mtop->natoms, mtop->natoms);
    }

    fr->print_force = print_force;


    /* coarse load balancing vars */
    fr->t_fnbf    = 0.;
    fr->t_wait    = 0.;
    fr->timesteps = 0;

    /* Initialize neighbor search */
    init_ns(fp, cr, &fr->ns, fr, mtop, box);

    if (cr->duty & DUTY_PP)
    {
        gmx_nonbonded_setup(fp, fr, bGenericKernelOnly);
        /*
           if (ir->bAdress)
            {
                gmx_setup_adress_kernels(fp,bGenericKernelOnly);
            }
         */
    }

    /* Initialize the thread working data for bonded interactions */
    init_forcerec_f_threads(fr, mtop->groups.grps[egcENER].nr);

    snew(fr->excl_load, fr->nthreads+1);

    if (fr->cutoff_scheme == ecutsVERLET)
    {
        if (ir->rcoulomb != ir->rvdw)
        {
            gmx_fatal(FARGS, "With Verlet lists rcoulomb and rvdw should be identical");
        }

        init_nb_verlet(fp, &fr->nbv, ir, fr, cr, nbpu_opt);
    }

    /* fr->ic is used both by verlet and group kernels (to some extent) now */
    init_interaction_const(fp, &fr->ic, fr, rtab);
    if (ir->eDispCorr != edispcNO)
    {
        //calc_enervirdiff(fp, ir->eDispCorr, fr);
    }
}

#define pr_real(fp, r) fprintf(fp, "%s: %e\n",#r, r)
#define pr_int(fp, i)  fprintf((fp), "%s: %d\n",#i, i)
#define pr_bool(fp, b) fprintf((fp), "%s: %s\n",#b, bool_names[b])

void forcerec_set_excl_load(t_forcerec *fr,
                            const gmx_localtop_t *top, const t_commrec *cr)
{//called
    const int *ind, *a;
    int        t, i, j, ntot, n, ntarget;

    if (cr != NULL && PARTDECOMP(cr))
    {
        /* No OpenMP with particle decomposition */
        pd_at_range(cr,
                    &fr->excl_load[0],
                    &fr->excl_load[1]);

        return;
    }

    ind = top->excls.index;
    a   = top->excls.a;

    ntot = 0;
    for (i = 0; i < top->excls.nr; i++)
    {
        for (j = ind[i]; j < ind[i+1]; j++)
        {
            if (a[j] > i)
            {
                ntot++;
            }
        }
    }

    fr->excl_load[0] = 0;
    n                = 0;
    i                = 0;
    for (t = 1; t <= fr->nthreads; t++)
    {
        ntarget = (ntot*t)/fr->nthreads;
        while (i < top->excls.nr && n < ntarget)
        {
            for (j = ind[i]; j < ind[i+1]; j++)
            {
                if (a[j] > i)
                {
                    n++;
                }
            }
            i++;
        }
        fr->excl_load[t] = i;
    }
}
