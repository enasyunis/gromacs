#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "gmx_wallcycle.h"
#include "gmx_cyclecounter.h"
#include "smalloc.h"
#include "gmx_fatal.h"
#include "md_logging.h"
#include "string2.h"

#include "tmpi.h"


typedef struct
{
    int          n;
    gmx_cycles_t c;
    gmx_cycles_t start;
    gmx_cycles_t last;
} wallcc_t;

typedef struct gmx_wallcycle
{
    wallcc_t        *wcc;
    /* variables for testing/debugging */
    gmx_bool         wc_barrier;
    wallcc_t        *wcc_all;
    int              wc_depth;
    int               ewc_prev;
    gmx_cycles_t      cycle_prev;
    gmx_large_int_t   reset_counters;
    MPI_Comm          mpi_comm_mygroup;
    int               nthreads_pp;
    int               nthreads_pme;
    double           *cycles_sum;
} gmx_wallcycle_t_t;

/* Each name should not exceed 19 characters */
static const char *wcn[ewcNR] =
{
    "Run", "Step", "PP during PME", "Domain decomp.", "DD comm. load",
    "DD comm. bounds", "Vsite constr.", "Send X to PME", "Neighbor search", "Launch GPU ops.",
    "Comm. coord.", "Born radii", "Force", "Wait + Comm. F", "PME mesh",
    "PME redist. X/F", "PME spread/gather", "PME 3D-FFT", "PME 3D-FFT Comm.", "PME solve",
    "PME wait for PP", "Wait + Recv. PME F", "Wait GPU nonlocal", "Wait GPU local", "NB X/F buffer ops.",
    "Vsite spread", "Write traj.", "Update", "Constraints", "Comm. energies",
    "Enforced rotation", "Add rot. forces", "Test"
};

static const char *wcsn[ewcsNR] =
{
    "DD redist.", "DD NS grid + sort", "DD setup comm.",
    "DD make top.", "DD make constr.", "DD top. other",
    "NS grid local", "NS grid non-loc.", "NS search local", "NS search non-loc.",
    "Bonded F", "Nonbonded F", "Ewald F correction",
    "NB X buffer ops.", "NB F buffer ops."
};


gmx_wallcycle_t wallcycle_init(FILE *fplog, int resetstep, t_commrec *cr,
                               int nthreads_pp, int nthreads_pme)
{ // called
    gmx_wallcycle_t wc;

    snew(wc, 1);

    wc->wc_barrier          = FALSE;
    wc->wcc_all             = NULL;
    wc->wc_depth            = 0;
    wc->ewc_prev            = -1;
    wc->reset_counters      = resetstep;
    wc->nthreads_pp         = nthreads_pp;
    wc->nthreads_pme        = nthreads_pme;
    wc->cycles_sum          = NULL;


    snew(wc->wcc, ewcNR);

    return wc;
}


void wallcycle_start(gmx_wallcycle_t wc, int ewc)
{// called
    gmx_cycles_t cycle;
    cycle              = gmx_cycles_read();
    wc->wcc[ewc].start = cycle;
}

void wallcycle_start_nocount(gmx_wallcycle_t wc, int ewc)
{ // called
    wallcycle_start(wc, ewc);
    wc->wcc[ewc].n--;
}

double wallcycle_stop(gmx_wallcycle_t wc, int ewc)
{ // called
    gmx_cycles_t cycle, last;
    cycle           = gmx_cycles_read();
    last            = cycle - wc->wcc[ewc].start;
    wc->wcc[ewc].c += last;
    wc->wcc[ewc].n++;

    return last;
}

static gmx_bool is_pme_counter(int ewc)
{ // called
    return (ewc >= ewcPMEMESH && ewc <= ewcPMEWAITCOMM);
}

static gmx_bool is_pme_subcounter(int ewc)
{// called
    return (ewc >= ewcPME_REDISTXF && ewc < ewcPMEWAITCOMM);
}

void wallcycle_sum(t_commrec *cr, gmx_wallcycle_t wc)
{ // called
    wallcc_t *wcc;
    double   *cycles;
    double    cycles_n[ewcNR+ewcsNR], buf[ewcNR+ewcsNR], *cyc_all, *buf_all;
    int       i, j;
    int       nsum;


    snew(wc->cycles_sum, ewcNR+ewcsNR);
    cycles = wc->cycles_sum;

    wcc = wc->wcc;

    for (i = 0; i < ewcNR; i++)
    {
        if (is_pme_counter(i) || (i == ewcRUN && cr->duty == DUTY_PME)) // PME 1, EWALD 0
        {
            wcc[i].c *= wc->nthreads_pme;

        }
        else
        {
            wcc[i].c *= wc->nthreads_pp;

        }
    }


    /* All nodes do PME (or no PME at all) */
    if (wcc[ewcPMEMESH].n > 0) // EWALD 0 PME > 0 
    {
       wcc[ewcFORCE].c -= wcc[ewcPMEMESH].c;
    }
    /* Store the cycles in a double buffer for summing */
    for (i = 0; i < ewcNR; i++)
    {
        cycles_n[i] = (double)wcc[i].n;
        cycles[i]   = (double)wcc[i].c;
    }
    nsum = ewcNR;

}

static void print_cycles(FILE *fplog, double c2t, const char *name,
                         int nnodes_tot, int nnodes, int nthreads,
                         int n, double c, double tot)
{ // called - done
    char   num[11];
    char   thstr[6];
    double wallt;

    if (c > 0)
    { 
        if (n > 0)
        {
            snprintf(num, sizeof(num), "%10d", n);
            snprintf(thstr, sizeof(thstr), "%4d", nthreads);
        }
        else
        {
            sprintf(num, "          ");
            sprintf(thstr, "    ");
        }
        wallt = c*c2t*nnodes_tot/(double)nnodes;
        fprintf(fplog, " %-19s %4d %4s %10s  %10.3f %12.3f   %5.1f\n",
                name, nnodes, thstr, num, wallt, c*1e-9, 100*c/tot);
    }
}

void wallcycle_print(FILE *fplog, int nnodes, int npme, double realtime,
                     gmx_wallcycle_t wc, wallclock_gpu_t *gpu_t)
{ // called -- done
    double     *cycles;
    double      c2t, tot, tot_gpu, tot_cpu_overlap, gpu_cpu_ratio, sum, tot_k;
    int         i, j, npp, nth_pp, nth_pme;
    char        buf[STRLEN];
    const char *hline = "-----------------------------------------------------------------------------";


    nth_pp  = wc->nthreads_pp;
    nth_pme = wc->nthreads_pme;

    cycles = wc->cycles_sum;

    npp  = nnodes;
    npme = nnodes;
    tot = cycles[ewcRUN];

    /* Conversion factor from cycles to seconds */
    c2t = realtime/tot;

    fprintf(fplog, "\n     R E A L   C Y C L E   A N D   T I M E   A C C O U N T I N G\n\n");

    fprintf(fplog, " Computing:         Nodes   Th.     Count  Wall t (s)     G-Cycles       %c\n", '%');
    fprintf(fplog, "%s\n", hline);
    sum = 0;
    for (i = ewcPPDURINGPME+1; i < ewcNR; i++)
    {
        if (!is_pme_subcounter(i))
        {
            print_cycles(fplog, c2t, wcn[i], nnodes,
                         is_pme_counter(i) ? npme : npp,
                         is_pme_counter(i) ? nth_pme : nth_pp,
                         wc->wcc[i].n, cycles[i], tot);
            sum += cycles[i];
        } 
    }
    print_cycles(fplog, c2t, "Rest", npp, npp, -1, 0, tot-sum, tot);
    fprintf(fplog, "%s\n", hline);
    print_cycles(fplog, c2t, "Total", nnodes, nnodes, -1, 0, tot, tot);
    fprintf(fplog, "%s\n", hline);

    if (wc->wcc[ewcPMEMESH].n > 0)
    {
        fprintf(fplog, "%s\n", hline);
        for (i = ewcPPDURINGPME+1; i < ewcNR; i++)
        {
            if (is_pme_subcounter(i))
            {
                print_cycles(fplog, c2t, wcn[i], nnodes,
                             is_pme_counter(i) ? npme : npp,
                             is_pme_counter(i) ? nth_pme : nth_pp,
                             wc->wcc[i].n, cycles[i], tot);
            }
        }
        fprintf(fplog, "%s\n", hline);
    }

    if (wc->wcc[ewcNB_XF_BUF_OPS].n > 0 &&
        (cycles[ewcDOMDEC] > tot*0.1 ||
         cycles[ewcNS] > tot*0.1))
    {
            md_print_warn(NULL, fplog,
                          "NOTE: %d %% of the run time was spent in pair search,\n"
                          "      you might want to increase nstlist (this has no effect on accuracy)\n",
                          (int)(100*cycles[ewcNS]/tot+0.5));
    }

}


