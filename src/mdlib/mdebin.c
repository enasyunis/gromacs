#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <float.h>
#include "typedefs.h"
#include "string2.h"
#include "mdebin.h"
#include "smalloc.h"
#include "physics.h"
#include "enxio.h"
#include "vec.h"
#include "disre.h"
#include "main.h"
#include "network.h"
#include "names.h"
#include "orires.h"
#include "constr.h"
#include "mtop_util.h"
#include "xvgr.h"
#include "gmxfio.h"
#include "mdrun.h"
#include "mdebin_bar.h"


static const char *conrmsd_nm[] = { "Constr. rmsd", "Constr.2 rmsd" };

static const char *boxs_nm[] = { "Box-X", "Box-Y", "Box-Z" };

static const char *tricl_boxs_nm[] = {
    "Box-XX", "Box-YY", "Box-ZZ",
    "Box-YX", "Box-ZX", "Box-ZY"
};

static const char *vol_nm[] = { "Volume" };

static const char *dens_nm[] = {"Density" };

static const char *pv_nm[] = {"pV" };

static const char *enthalpy_nm[] = {"Enthalpy" };

static const char *boxvel_nm[] = {
    "Box-Vel-XX", "Box-Vel-YY", "Box-Vel-ZZ",
    "Box-Vel-YX", "Box-Vel-ZX", "Box-Vel-ZY"
};

#define NBOXS asize(boxs_nm)
#define NTRICLBOXS asize(tricl_boxs_nm)

t_mdebin *init_mdebin(ener_file_t       fp_ene,
                      const gmx_mtop_t *mtop,
                      const t_inputrec *ir,
                      FILE             *fp_dhdl)
{
    const char         *ener_nm[F_NRE];
    static const char  *vir_nm[] = {
        "Vir-XX", "Vir-XY", "Vir-XZ",
        "Vir-YX", "Vir-YY", "Vir-YZ",
        "Vir-ZX", "Vir-ZY", "Vir-ZZ"
    };
    static const char  *sv_nm[] = {
        "ShakeVir-XX", "ShakeVir-XY", "ShakeVir-XZ",
        "ShakeVir-YX", "ShakeVir-YY", "ShakeVir-YZ",
        "ShakeVir-ZX", "ShakeVir-ZY", "ShakeVir-ZZ"
    };
    static const char  *fv_nm[] = {
        "ForceVir-XX", "ForceVir-XY", "ForceVir-XZ",
        "ForceVir-YX", "ForceVir-YY", "ForceVir-YZ",
        "ForceVir-ZX", "ForceVir-ZY", "ForceVir-ZZ"
    };
    static const char  *pres_nm[] = {
        "Pres-XX", "Pres-XY", "Pres-XZ",
        "Pres-YX", "Pres-YY", "Pres-YZ",
        "Pres-ZX", "Pres-ZY", "Pres-ZZ"
    };
    static const char  *surft_nm[] = {
        "#Surf*SurfTen"
    };
    static const char  *mu_nm[] = {
        "Mu-X", "Mu-Y", "Mu-Z"
    };
    static const char  *vcos_nm[] = {
        "2CosZ*Vel-X"
    };
    static const char  *visc_nm[] = {
        "1/Viscosity"
    };
    static const char  *baro_nm[] = {
        "Barostat"
    };

    char              **grpnms;
    const gmx_groups_t *groups;
    char              **gnm;
    char                buf[256];
    const char         *bufi;
    t_mdebin           *md;
    int                 i, j, ni, nj, n, nh, k, kk, ncon, nset;

    snew(md, 1);

    md->bVir   = TRUE;
    md->bPress = TRUE;
    md->bSurft = TRUE;
    md->bMu    = TRUE;

    md->delta_t = ir->delta_t;

    groups = &mtop->groups;


    ncon           = gmx_mtop_ftype_count(mtop, F_CONSTR);
    nset           = gmx_mtop_ftype_count(mtop, F_SETTLE);
    md->bConstr    = (ncon > 0 || nset > 0);
    md->bConstrVir = FALSE;
    md->nCrmsd = 0;
    md->f_nre = 0;

    /* Energy monitoring */
    for (i = 0; i < egNR; i++) // 9
    {
        md->bEInd[i] = FALSE;
    }
    for (i = 0; i < F_NRE; i++) // 86
    {
        md->bEner[i] = FALSE; //default setting

        if ((i == F_COUL_RECIP) || (i == F_COUL_SR) || 
            (i == F_EPOT) || (i == F_PRES)  || 
            (i == F_ETOT) || (i == F_EKIN) || 
            (i == F_TEMP) || (i == F_ECONSERVED))
        {
            md->bEner[i] = TRUE;
            ener_nm[md->f_nre] = interaction_function[i].longname;
            md->f_nre++;
        }
    } 

    md->epc            = ir->epc;
    md->bDiagPres      = !TRICLINIC(ir->ref_p);
    md->ref_p          = (ir->ref_p[XX][XX]+ir->ref_p[YY][YY]+ir->ref_p[ZZ][ZZ])/DIM;
    md->bTricl         = TRICLINIC(ir->compress) || TRICLINIC(ir->deform);
    md->bDynBox        = DYNAMIC_BOX(*ir);
    md->etc            = ir->etc;
    md->bNHC_trotter   = IR_NVT_TROTTER(ir);
    md->bPrintNHChains = ir->bPrintNHChains;
    md->bMTTK          = (IR_NPT_TROTTER(ir) || IR_NPH_TROTTER(ir));
    md->bMu            = NEED_MUTOT(*ir);

    md->ebin  = mk_ebin();
    /* Pass NULL for unit to let get_ebin_space determine the units
     * for interaction_function[i].longname
     */
    md->ie    = get_ebin_space(md->ebin, md->f_nre, ener_nm, NULL);
    md->ivir   = get_ebin_space(md->ebin, asize(vir_nm), vir_nm, unit_energy);
    md->ipres  = get_ebin_space(md->ebin, asize(pres_nm), pres_nm, unit_pres_bar);
    md->isurft = get_ebin_space(md->ebin, asize(surft_nm), surft_nm,
                                    unit_surft_bar);
    md->bEInd[egCOULSR] = TRUE;
    md->bEInd[egLJSR  ] = TRUE;

    md->nEc = 2;
    n = groups->grps[egcENER].nr;
    /* for adress simulations, most energy terms are not meaningfull, and thus disabled*/
    /*standard simulation*/
    md->nEg = n;
    md->nE  = (n*(n+1))/2;
    snew(md->igrp, md->nE);

    md->nTC  = groups->grps[egcTC].nr;
    md->nNHC = ir->opts.nhchainlength; /* shorthand for number of NH chains */
    md->nTCP = 0;
    md->mde_n  = md->nTC;
    md->mdeb_n = 0;

    snew(md->tmp_r, md->mde_n);
    snew(md->tmp_v, md->mde_n);
    snew(md->grpnms, md->mde_n);
    grpnms = md->grpnms;

    ni = groups->grps[egcTC].nm_ind[0];
    sprintf(buf, "T-%s", *(groups->grpname[ni]));
    grpnms[0] = strdup(buf);
    md->itemp = get_ebin_space(md->ebin, md->nTC, (const char **)grpnms,
                               unit_temp_K);

    ni = groups->grps[egcTC].nm_ind[0];
    sprintf(buf, "Lamb-%s", *(groups->grpname[ni]));
    grpnms[0] = strdup(buf);
    md->itc = get_ebin_space(md->ebin, md->mde_n, (const char **)grpnms, "");

    sfree(grpnms);


    md->nU = groups->grps[egcACC].nr;

    do_enxnms(fp_ene, &md->ebin->nener, &md->ebin->enm);

    md->print_grpnms = NULL;

    /* check whether we're going to write dh histograms */
    md->dhc = NULL;
    md->fp_dhdl = fp_dhdl;
    snew(md->dE, ir->fepvals->n_lambda);
    return md;
}


static void copy_energy(t_mdebin *md, real e[], real ecpy[])
{
    int i, j;

    for (i = j = 0; (i < F_NRE); i++)
    {
        if (md->bEner[i])
        {
            ecpy[j++] = e[i];
        }
    }
}

void upd_mdebin(t_mdebin       *md,
                gmx_bool        bDoDHDL,
                gmx_bool        bSum,
                double          time,
                real            tmass,
                gmx_enerdata_t *enerd,
                t_state        *state,
                t_lambda       *fep,
                t_expanded     *expand,
                matrix          box,
                tensor          svir,
                tensor          fvir,
                tensor          vir,
                tensor          pres,
                gmx_ekindata_t *ekind,
                rvec            mu_tot,
                gmx_constr_t    constr)
{
    int    i, j, k, kk, m, n, gid;
    real   crmsd[2], tmp6[6];
    real   bs[NTRICLBOXS], vol, dens, pv, enthalpy;
    real   eee[egNR];
    real   ecopy[F_NRE];
    double store_dhdl[efptNR];
    real   store_energy = 0;
    real   tmp;

    /* Do NOT use the box in the state variable, but the separate box provided
     * as an argument. This is because we sometimes need to write the box from
     * the last timestep to match the trajectory frames.
     */
    copy_energy(md, enerd->term, ecopy);
    add_ebin(md->ebin, md->ie, md->f_nre, ecopy, bSum);
    add_ebin(md->ebin, md->ivir, 9, vir[0], bSum);
    add_ebin(md->ebin, md->ipres, 9, pres[0], bSum);
    tmp = (pres[ZZ][ZZ]-(pres[XX][XX]+pres[YY][YY])*0.5)*box[ZZ][ZZ];
    add_ebin(md->ebin, md->isurft, 1, &tmp, bSum);

    md->tmp_r[0] = ekind->tcstat[0].T;
    add_ebin(md->ebin, md->itemp, md->nTC, md->tmp_r, bSum);

    md->tmp_r[0] = ekind->tcstat[0].lambda;
    add_ebin(md->ebin, md->itc, md->nTC, md->tmp_r, bSum);

    ebin_increase_count(md->ebin, bSum);

}



static void npr(FILE *log, int n, char c)
{ 
    for (; (n > 0); n--)
    {
        fprintf(log, "%c", c);
    }
}

static void pprint(FILE *log, const char *s, t_mdebin *md)
{ 
    char CHAR = '#';
    int  slen;
    char buf1[22], buf2[22];

    slen = strlen(s);
    fprintf(log, "\t<======  ");
    npr(log, slen, CHAR);
    fprintf(log, "  ==>\n");
    fprintf(log, "\t<====  %s  ====>\n", s);
    fprintf(log, "\t<==  ");
    npr(log, slen, CHAR);
    fprintf(log, "  ======>\n\n");

    fprintf(log, "\tStatistics over %s steps using %s frames\n",
            gmx_step_str(md->ebin->nsteps_sim, buf1),
            gmx_step_str(md->ebin->nsum_sim, buf2));
    fprintf(log, "\n");
}

void print_ebin_header(FILE *log, gmx_large_int_t steps, double time, real lambda)
{ 
    char buf[22];

    fprintf(log, "   %12s   %12s   %12s\n"
            "   %12s   %12.5f   %12.5f\n\n",
            "Step", "Time", "Lambda", gmx_step_str(steps, buf), time, lambda);
}

void print_ebin(ener_file_t fp_ene, gmx_bool bEne, gmx_bool bDR, gmx_bool bOR,
                FILE *log,
                gmx_large_int_t step, double time,
                int mode, gmx_bool bCompact,
                t_mdebin *md, t_fcdata *fcd,
                gmx_groups_t *groups, t_grpopts *opts)
{
    /*static char **grpnms=NULL;*/
    char         buf[246];
    int          i, j, n, ni, nj, ndr, nor, b;
    int          ndisre = 0;
    real        *disre_rm3tav, *disre_rt;

    /* these are for the old-style blocks (1 subblock, only reals), because
       there can be only one per ID for these */
    int          nr[enxNR];
    int          id[enxNR];
    real        *block[enxNR];

    /* temporary arrays for the lambda values to write out */
    double      enxlambda_data[2];

    t_enxframe  fr;

    switch (mode)
    {
        case eprNORMAL:
            init_enxframe(&fr);
            fr.t            = time;
            fr.step         = step;
            fr.nsteps       = md->ebin->nsteps;
            fr.dt           = md->delta_t;
            fr.nsum         = md->ebin->nsum;
            fr.nre          = (bEne) ? md->ebin->nener : 0;
            fr.ener         = md->ebin->e;
            ndisre          = bDR ? fcd->disres.npair : 0;
            disre_rm3tav    = fcd->disres.rm3tav;
            disre_rt        = fcd->disres.rt;
            /* Optional additional old-style (real-only) blocks. */
            for (i = 0; i < enxNR; i++)
            {
                nr[i] = 0;
            }

            /* the old-style blocks go first */
            fr.nblock = 0;
            add_blocks_enxframe(&fr, fr.nblock);
            for (b = 0; b < fr.nblock; b++)
            {
                 add_subblocks_enxblock(&(fr.block[b]), 1);
                 fr.block[b].id        = id[b];
                 fr.block[b].sub[0].nr = nr[b];
                 fr.block[b].sub[0].type = xdr_datatype_double;
                 fr.block[b].sub[0].dval = block[b];
            }

            /* do the actual I/O */
            do_enx(fp_ene, &fr);
            gmx_fio_check_file_position(enx_file_pointer(fp_ene));
            /* We have stored the sums, so reset the sum history */
            reset_ebin_sums(md->ebin);
            free_enxframe(&fr);
            break;
        case eprAVER:
            pprint(log, "A V E R A G E S", md);
            break;
    }

    if (mode == eprNORMAL && fcd->orires.nr > 0)
    fprintf(log, "   Energies (%s)\n", unit_energy);
    pr_ebin(log, md->ebin, md->ie, md->f_nre+md->nCrmsd, 5, mode, TRUE);
    fprintf(log, "\n");

    if (!bCompact)
    {
       fprintf(log, "   Total Virial (%s)\n", unit_energy);
       pr_ebin(log, md->ebin, md->ivir, 9, 3, mode, FALSE);
       fprintf(log, "\n");
       fprintf(log, "   Pressure (%s)\n", unit_pres_bar);
       pr_ebin(log, md->ebin, md->ipres, 9, 3, mode, FALSE);
       fprintf(log, "\n");

    }
}

void update_energyhistory(energyhistory_t * enerhist, t_mdebin * mdebin)
{
    int i;

    enerhist->nsteps     = mdebin->ebin->nsteps;
    enerhist->nsum       = mdebin->ebin->nsum;
    enerhist->nsteps_sim = mdebin->ebin->nsteps_sim;
    enerhist->nsum_sim   = mdebin->ebin->nsum_sim;
    enerhist->nener      = mdebin->ebin->nener;
}

