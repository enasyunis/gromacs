#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include <string.h>
#include <assert.h>
#include "sysstuff.h"
#include "vec.h"
#include "typedefs.h"
#include "macros.h"
#include "smalloc.h"
#include "macros.h"
#include "physics.h"
#include "force.h"
#include "names.h"
#include "pbc.h"
#include "ns.h"
#include "bondf.h"
#include "txtdump.h"
#include "coulomb.h"
#include "pme.h"
#include "mdrun.h"
#include "gmx_omp_nthreads.h"

/* ENAS ADDED START */
extern void FMMcalccoulomb_ij(int ni, double* xi, double* qi, double* fi,
  int nj, double* xj, double* qj, double rscale, int tblno, double size, 
  int periodicflag, int ksize, double alpha);

//extern void FMMcalcvdw_ij(int ni, double* xi, int* atypei, double* fi,
//  int nj, double* xj, int* atypej, int nat, double* gscale, double* rscale,
//  int tblno, double size, int periodicflag);

int fmmsteptaken=0;
/* ENAS ADDED END */


void do_force_lowlevel(FILE       *fplog,   gmx_large_int_t step,
                       t_forcerec *fr,      t_inputrec *ir,
                       t_idef     *idef,    t_commrec  *cr,
                       t_mdatoms  *md,
                       t_grpopts  *opts,
                       rvec       x[],      history_t  *hist,
                       rvec       f[],
                       rvec       f_longrange[],
                       gmx_enerdata_t *enerd,
                       gmx_mtop_t     *mtop,
                       gmx_localtop_t *top,
                       t_atomtypes *atype,
                       matrix     box,
                       t_lambda   *fepvals,
                       real       *lambda,
                       t_blocka   *excl,
                       int        flags)
{// called 
    int         i, j, status;
    int         donb_flags;
    gmx_bool    bDoEpot, bSB;
    int         pme_flags;
    matrix      boxs;
    rvec        box_size;
    real        Vlr;
    t_pbc       pbc;
    real        dvdgb;
    char        buf[22];
    double      clam_i, vlam_i;
    real        dvdl_dum[efptNR], dvdl, dvdl_nb[efptNR], lam_i[efptNR];
    real        dvdlsum;

    double  t0 = 0.0, t1, t2, t3; /* time measurement for coarse load balancing */

#define PRINT_SEPDVDL(s, v, dvdlambda) fprintf(fplog, sepdvdlformat, s, v, dvdlambda); 

    set_pbc(&pbc, fr->ePBC, box);

    /* reset free energy components */
    for (i = 0; i < efptNR; i++)// 7
    {
        dvdl_nb[i]  = 0;
        dvdl_dum[i] = 0;
    }

    /* Reset box */
    for (i = 0; (i < DIM); i++) // 3 need xx, yy, zz 
    {
        box_size[i] = box[i][i];
    }

    fprintf(fplog, "Step %s: non-bonded V and dVdl for node %d:\n",
                gmx_step_str(step, buf), cr->nodeid);

    /* Call the short range functions all in one go. */


    where();
    /* We only do non-bonded calculation with group scheme here, the verlet
     * calls are done from do_force_cutsVERLET(). */

    enerd->dvdl_lin[efptVDW] += dvdl_nb[efptVDW];
    enerd->dvdl_lin[efptCOUL] += dvdl_nb[efptCOUL];




    /* Since all atoms are in the rectangular or triclinic unit-cell,
     * only single box vector shifts (2 in x) are required.
     */
    set_pbc_dd(&pbc, fr->ePBC, cr->dd, TRUE, box);
        calc_bonds(fplog, cr->ms,
                   idef, x, hist, f, fr, &pbc, NULL, enerd, NULL, lambda, md, 
                    NULL, atype, NULL,
                   flags,
                   fr->bSepDVDL, step);


    where();

        bSB = FALSE;

        clear_mat(fr->vir_el_recip);



        status = 0;
        Vlr    = 0;
        dvdl   = 0;
real         *xEY = fr->nbv->grp[0].nbat->x;
real         *qEY = fr->nbv->grp[0].nbat->q;
real        alpha = fr->ewaldcoeff;
int         ksize = fr->ksize;
	    if (FALSE) // ENAS, RIO, TODO make this the place for FMM 
            {
                        /* ENAS ADDED START */
                        if (fmmsteptaken == 0) {
                        printf("\nBEFORE FMM\n");
                        int N = md->homenr;
                        double *xi; snew(xi, 3*N); // double *xi     = new double [3*N];
                        double *qi; snew(qi, N);   // double *qi     = new double [N];
                        double *fi; snew(fi, 3*N); // double *fi     = new double [3*N];
                        double *pi; snew(pi, 3*N); // double *pi     = new double [3*N];
                        int eyi=0, eyG=0;
                        for (;eyi<N;++eyi){
                                while (xEY[eyG] < 0)
                                {
                                        eyG++;
                                } 

                                qi[eyi]=qEY[eyG/3];
                                xi[eyi*3+0]=xEY[eyG++];
                                xi[eyi*3+1]=xEY[eyG++];
                                xi[eyi*3+2]=xEY[eyG++];

                                fi[eyi*3+0]=fi[eyi*3+1]=fi[eyi*3+2]=0.0;
                                pi[eyi*3+0]=pi[eyi*3+1]=pi[eyi*3+2]=0.0;
                        }
                        printf("About to call FMM\n");
                        FMMcalccoulomb_ij(N, xi, qi, fi, N, xi, qi, 0.0, 
                                          0, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 1 \n");
                        FMMcalccoulomb_ij(N, xi, qi, pi, N, xi, qi, 0.0, 
                                          1, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 2 \n");
                        double eyP=0.0;
                        for(eyi=0;eyi<N;++eyi){
                                eyP+=pi[eyi*3+0];
                        }
                        printf("FMM Total Coulomb Potential: %.9e\n", eyP*138.935485);
                        sfree(xi);
                        sfree(qi);
                        sfree(fi);
                        sfree(pi);
                        printf("\nAFTER FMM\n");
                        } fmmsteptaken=1;
                        /* ENAS ADDED END */ 
            }
            else if (fr->eeltype == eelPME)
            {
                    pme_flags = GMX_PME_SPREAD_Q | GMX_PME_SOLVE;
                    pme_flags |= GMX_PME_CALC_F;
                    pme_flags |= GMX_PME_CALC_ENER_VIR;
                    status = gmx_pme_do(fr->pmedata,
                                            md->start, md->homenr - fr->n_tpi,
                                            x, fr->f_novirsum,
                                            md->chargeA, md->chargeB,
                                            box, cr,
                                            0,
                                            0,
                                            fr->vir_el_recip, fr->ewaldcoeff,
                                            &Vlr, lambda[efptCOUL], &dvdl,
                                            pme_flags);
                    // PME mesh: Vlr  3.11728e+02  dvdl  0.00000e+0
                    PRINT_SEPDVDL("PME mesh", Vlr, dvdl);
                        /* ENAS ADDED START */
                        if (fmmsteptaken == 0) {
                        printf("\nBEFORE FMM\n");

                        int N = md->homenr;
                        double *xi; snew(xi, 3*N); // double *xi     = new double [3*N];
                        double *qi; snew(qi, N);   // double *qi     = new double [N];
                        double *fi; snew(fi, 3*N); // double *fi     = new double [3*N];
                        double *pi; snew(pi, 3*N); // double *pi     = new double [3*N];
                        int eyi=0, eyG=0;
                        for (;eyi<N;++eyi){
				while (xEY[eyG] < 0) 
                                {
					eyG++;
				}

                                qi[eyi]=qEY[eyG/3];
                                xi[eyi*3+0]=xEY[eyG++];
                                xi[eyi*3+1]=xEY[eyG++];
                                xi[eyi*3+2]=xEY[eyG++];


                                fi[eyi*3+0]=fi[eyi*3+1]=fi[eyi*3+2]=0.0;
                                pi[eyi*3+0]=pi[eyi*3+1]=pi[eyi*3+2]=0.0;
                        } 
                        printf("About to call FMM\n");
                        FMMcalccoulomb_ij(N, xi, qi, fi, N, xi, qi, 0.0,                                           0, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 1 \n");
                        FMMcalccoulomb_ij(N, xi, qi, pi, N, xi, qi, 0.0,                                           1, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 2 \n");
                        double eyP=0.0;
                        for(eyi=0;eyi<N;++eyi){
                                eyP+=pi[eyi*3+0];
                        }
                        printf("FMM Total Coulomb Potential: %.9e\n", eyP*138.935485);
                        sfree(xi);
                        sfree(qi);
                        sfree(fi);
                        sfree(pi);
                        printf("\nAFTER FMM\n");
                        } fmmsteptaken=1;
                        /* ENAS ADDED END */
            }
            else if (fr->eeltype == eelEWALD) 
            {
                Vlr = do_ewald(fplog, FALSE, ir, x, fr->f_novirsum,
                               md->chargeA, md->chargeB,
                               box_size, cr, md->homenr,
                               fr->vir_el_recip, fr->ewaldcoeff,
                               lambda[efptCOUL], &dvdl, fr->ewald_table);
                PRINT_SEPDVDL("Ewald long-range", Vlr, dvdl);
                        /* ENAS ADDED START */
                        if (fmmsteptaken == 0) {
                        printf("\nBEFORE FMM\n");

                        int N = md->homenr;
                        double *xi; snew(xi, 3*N); // double *xi     = new double [3*N];
                        double *qi; snew(qi, N);   // double *qi     = new double [N];
                        double *fi; snew(fi, 3*N); // double *fi     = new double [3*N];
                        double *pi; snew(pi, 3*N); // double *pi     = new double [3*N];


                        int eyi=0, eyG=0;
                        for (;eyi<N;++eyi){
                                while (xEY[eyG] < 0)                      
                                {
                                        eyG++;
                                }

                                qi[eyi]=qEY[eyG/3];
                                xi[eyi*3+0]=xEY[eyG++];
                                xi[eyi*3+1]=xEY[eyG++];
                                xi[eyi*3+2]=xEY[eyG++];


                                fi[eyi*3+0]=fi[eyi*3+1]=fi[eyi*3+2]=0.0;
                                pi[eyi*3+0]=pi[eyi*3+1]=pi[eyi*3+2]=0.0;
                        } 


                        printf("About to call FMM\n");
                        FMMcalccoulomb_ij(N, xi, qi, fi, N, xi, qi, 0.0,                                           0, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 1 \n");
                        FMMcalccoulomb_ij(N, xi, qi, pi, N, xi, qi, 0.0,                                           1, (bSB?boxs[0][0]:box[0][0]), 1,
                                          ksize, alpha);
                        printf("Done calling FMM 2 \n");
                        double eyP=0.0;
                        for(eyi=0;eyi<N;++eyi){
                                eyP+=pi[eyi*3+0];
                        }
                        printf("FMM Total Coulomb Potential: %.9e\n", eyP*138.935485);
                        sfree(xi);
                        sfree(qi);
                        sfree(fi);
                        sfree(pi);
                        printf("\nAFTER FMM\n");
                        } fmmsteptaken=1;
                        /* ENAS ADDED END */
            }
        /* Note that with separate PME nodes we get the real energies later */
        enerd->dvdl_lin[efptCOUL] += dvdl;
        enerd->term[F_COUL_RECIP]  = Vlr;
    where();

}

void init_enerdata(int ngener, int n_lambda, gmx_enerdata_t *enerd)
{//called
    int i, n2;
    for (i = 0; i < F_NRE; i++) // 86
    {
        enerd->term[i]         = 0;
        enerd->foreign_term[i] = 0;
    }

    for (i = 0; i < efptNR; i++) // 7
    {
        enerd->dvdl_lin[i]     = 0;
        enerd->dvdl_nonlin[i]  = 0;
    }

    n2 = ngener*ngener;
    enerd->grpp.nener         = n2;
    enerd->foreign_grpp.nener = n2;
    for (i = 0; (i < egNR); i++) // 9
    {
        snew(enerd->grpp.ener[i], n2);
        snew(enerd->foreign_grpp.ener[i], n2);
    }
    enerd->n_lambda = 0;
}


void reset_foreign_enerdata(gmx_enerdata_t *enerd)
{// called 
    int  i, j;

    /* First reset all foreign energy components.  Foreign energies always called on
       neighbor search steps */
    for (i = 0; (i < egNR); i++) // 9
    {
            enerd->foreign_grpp.ener[i][0] = 0.0;
    }
    /* potential energy components */
    enerd->foreign_term[0] = 0.0;
    enerd->foreign_term[1] = 0.0;
}

void reset_enerdata(t_grpopts *opts,
                    t_forcerec *fr, gmx_bool bNS,
                    gmx_enerdata_t *enerd,
                    gmx_bool bMaster)
{// called                                 
    int      i;

    /* First reset all energy components, except for the long range terms
     * on the master at non neighbor search steps, since the long range
     * terms have already been summed at the last neighbor search step.
     */
    for (i = 0; (i < egNR); i++) // 9
    {
        enerd->grpp.ener[i][0] = 0.0;
    }
    for (i = 0; i < efptNR; i++) // 7
    {
        enerd->dvdl_lin[i]    = 0.0;
        enerd->dvdl_nonlin[i] = 0.0;
    }

    /* Normal potential energy components */
    enerd->term[0] = 0.0;
    enerd->term[1] = 0.0;
    /* Initialize the dVdlambda term with the long range contribution */
    /* Initialize the dvdl term with the long range contribution */
    enerd->term[F_DVDL]            = 0.0;
    enerd->term[F_DVDL_COUL]       = 0.0;
    enerd->term[F_DVDL_VDW]        = 0.0;
    enerd->term[F_DVDL_BONDED]     = 0.0;
    enerd->term[F_DVDL_RESTRAINT]  = 0.0;
    enerd->term[F_DKDL]            = 0.0;
    /* reset foreign energy data - separate function since we also call it elsewhere */
    reset_foreign_enerdata(enerd);
}
