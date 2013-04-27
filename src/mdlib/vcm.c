/* This file is completely threadsafe - keep it that way! */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "macros.h"
#include "vcm.h"
#include "vec.h"
#include "smalloc.h"
#include "names.h"
#include "txtdump.h"
#include "network.h"
#include "pbc.h"

t_vcm *init_vcm(FILE *fp, gmx_groups_t *groups, t_inputrec *ir)
{ // called 
    t_vcm *vcm;
    int    g;

    snew(vcm, 1);
    vcm->mode = ir->comm_mode;
    vcm->ndim = ndof_com(ir); // 3

	// vcm->nr = 1 :)

        vcm->nr = groups->grps[egcVCM].nr;
        snew(vcm->group_p, vcm->nr+1);
        snew(vcm->group_v, vcm->nr+1);
        snew(vcm->group_mass, vcm->nr+1);
        snew(vcm->group_name, vcm->nr);
        snew(vcm->group_ndf, vcm->nr);

	g=0;
        vcm->group_ndf[g] = ir->opts.nrdf[g];

        /* Copy pointer to group names and print it. */
        fprintf(fp, "Center of mass motion removal mode is %s\n",
                   ECOM(vcm->mode));
        fprintf(fp, "We have the following groups for center of"
                   " mass motion removal:\n");
        vcm->group_name[g] = *groups->grpname[groups->grps[egcVCM].nm_ind[g]];
        fprintf(fp, "%3d:  %s\n", g, vcm->group_name[g]);

    return vcm;
}


/* Center of mass code for groups */
void calc_vcm_grp(FILE *fp, int start, int homenr, t_mdatoms *md,
                  rvec x[], rvec v[], t_vcm *vcm)
{ // called 
    int    i, g, m;
    real   m0, xx, xy, xz, yy, yz, zz;
    rvec   j0;
        /* Also clear a possible rest group */
        for (g = 0; (g < vcm->nr+1); g++) // 2 iterations
        {
            /* Reset linear momentum */
            vcm->group_mass[g] = 0;
            clear_rvec(vcm->group_p[g]);

        }

        g = 0;
        for (i = start; (i < start+homenr); i++) // start=0, homenr=3000
        {
            m0 = md->massT[i];

            /* Calculate linear momentum */
            vcm->group_mass[g]  += m0;
            for (m = 0; (m < DIM); m++) // 3
            {
                vcm->group_p[g][m] += m0*v[i][m];
            }

        }
}

void do_stopcm_grp(FILE *fp, int start, int homenr, unsigned short *group_id,
                   rvec x[], rvec v[], t_vcm *vcm)
{ // called 
    int  i;
    /* Subtract linear momentum */
    for (i = start; (i < start+homenr); i++) // start=0, homenr=3000
    {
        rvec_dec(v[i], vcm->group_v[0]);
    }
}


void check_cm_grp(FILE *fp, t_vcm *vcm, t_inputrec *ir, real Temp_Max)
{ //called 
    int    m, g=0;
    real   ekcm, ekrot, tm, tm_1, Temp_cm;
    rvec   jcm;
    tensor Icm, Tcm;

    /* First analyse the total results */
    tm = vcm->group_mass[g];
    tm_1 = 1.0/tm;
    svmul(tm_1, vcm->group_p[g], vcm->group_v[g]);
    /* Else it's zero anyway! */
    ekcm    = 0;
    for (m = 0; m < vcm->ndim; m++) // 3
    {
         ekcm += sqr(vcm->group_v[g][m]);
    }
    ekcm   *= 0.5*vcm->group_mass[g];
    Temp_cm = 2*ekcm/vcm->group_ndf[g];
}
