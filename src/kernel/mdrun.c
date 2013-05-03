#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "typedefs.h"
#include "macros.h"
#include "copyrite.h"
#include "main.h"
#include "statutil.h"
#include "smalloc.h"
#include "futil.h"
#include "smalloc.h"
#include "edsam.h"
#include "mdrun.h"
#include "xmdrun.h"
#include "checkpoint.h"
#include "thread_mpi.h"

/* afm stuf */
#include "pull.h"

int cmain(int argc, char *argv[])
{
    const char   *desc[] = {""
    };
    t_commrec    *cr;
    t_filenm      fnm[] = {
        { efTPX, NULL,      NULL,       ffREAD },
        { efTRN, "-o",      NULL,       ffWRITE },
        { efXTC, "-x",      NULL,       ffOPTWR },
        { efCPT, "-cpi",    NULL,       ffOPTRD },
        { efCPT, "-cpo",    NULL,       ffOPTWR },
        { efSTO, "-c",      "confout",  ffWRITE },
        { efEDR, "-e",      "ener",     ffWRITE },
        { efLOG, "-g",      "md",       ffWRITE },
        { efXVG, "-dhdl",   "dhdl",     ffOPTWR },
        { efXVG, "-field",  "field",    ffOPTWR },
        { efXVG, "-table",  "table",    ffOPTRD },
        { efXVG, "-tabletf", "tabletf",    ffOPTRD },
        { efXVG, "-tablep", "tablep",   ffOPTRD },
        { efXVG, "-tableb", "table",    ffOPTRD },
        { efTRX, "-rerun",  "rerun",    ffOPTRD },
        { efXVG, "-tpi",    "tpi",      ffOPTWR },
        { efXVG, "-tpid",   "tpidist",  ffOPTWR },
        { efEDI, "-ei",     "sam",      ffOPTRD },
        { efXVG, "-eo",     "edsam",    ffOPTWR },
        { efGCT, "-j",      "wham",     ffOPTRD },
        { efGCT, "-jo",     "bam",      ffOPTWR },
        { efXVG, "-ffout",  "gct",      ffOPTWR },
        { efXVG, "-devout", "deviatie", ffOPTWR },
        { efXVG, "-runav",  "runaver",  ffOPTWR },
        { efXVG, "-px",     "pullx",    ffOPTWR },
        { efXVG, "-pf",     "pullf",    ffOPTWR },
        { efXVG, "-ro",     "rotation", ffOPTWR },
        { efLOG, "-ra",     "rotangles", ffOPTWR },
        { efLOG, "-rs",     "rotslabs", ffOPTWR },
        { efLOG, "-rt",     "rottorque", ffOPTWR },
        { efMTX, "-mtx",    "nm",       ffOPTWR },
        { efNDX, "-dn",     "dipole",   ffOPTWR },
        { efRND, "-multidir", NULL,      ffOPTRDMULT},
        { efDAT, "-membed", "membed",   ffOPTRD },
        { efTOP, "-mp",     "membed",   ffOPTRD },
        { efNDX, "-mn",     "membed",   ffOPTRD }
    };
#define NFILE asize(fnm)

    /* Command line options ! */
    gmx_bool      bCart         = FALSE;
    gmx_bool      bPPPME        = FALSE;
    gmx_bool      bPartDec      = FALSE;
    gmx_bool      bDDBondCheck  = TRUE;
    gmx_bool      bDDBondComm   = TRUE;
    gmx_bool      bTunePME      = TRUE;
    gmx_bool      bTestVerlet   = FALSE;
    gmx_bool      bVerbose      = TRUE;
    gmx_bool      bCompact      = TRUE;
    gmx_bool      bSepPot       = TRUE;
    gmx_bool      bRerunVSite   = FALSE;
    gmx_bool      bIonize       = FALSE;
    gmx_bool      bConfout      = TRUE;
    gmx_bool      bReproducible = FALSE;

    int           npme          = 0;
    int           nmultisim     = 0;
    int           repl_ex_nst   = 0;
    int           repl_ex_seed  = -1;
    int           repl_ex_nex   = 0;
    int           nstepout      = 100;
    int           resetstep     = -1;
    int           nsteps        = -2; /* the value -2 means that the mdp option will be used */

    rvec          realddxyz          = {0, 0, 0};
    const char   *ddno_opt[ddnoNR+1] =
    { NULL, "interleave", "pp_pme", "cartesian", NULL };
    const char   *dddlb_opt[] =
    { NULL, "auto", "no", "yes", NULL };
    const char   *thread_aff_opt[threadaffNR+1] =
    { NULL, "auto", "on", "off", NULL };
    const char   *nbpu_opt[] =
    { NULL, "auto", "cpu", "gpu", "gpu_cpu", NULL };
    real          rdd                   = 0.0, rconstr = 0.0, dlb_scale = 0.8, pforce = -1;
    char         *ddcsx                 = NULL, *ddcsy = NULL, *ddcsz = NULL;
    real          cpt_period            = 15.0, max_hours = -1;
    gmx_bool      bAppendFiles          = FALSE;
    gmx_bool      bKeepAndNumCPT        = FALSE;
    gmx_bool      bResetCountersHalfWay = FALSE;
    output_env_t  oenv                  = NULL;
    const char   *deviceOptions         = "";
    char         *deffnm                = "";

    gmx_hw_opt_t  hw_opt = {0, 0, 0, 0, threadaffSEL, 0, 0, NULL};

    gmx_edsam_t   ed;
    unsigned long Flags, PCA_Flags;
    ivec          ddxyz;
    int           dd_node_order;
    FILE         *fplog;
    int           rc;


    cr = init_par(&argc, &argv);

    PCA_Flags = ((1<<10) | (MASTER(cr) ? 0 : (1<<12)));
    parse_common_args(&argc, argv, PCA_Flags, NFILE, fnm, 0, NULL, 
                      asize(desc), desc, 0, NULL, &oenv);

    dd_node_order = nenum(ddno_opt);
    cr->npmenodes = npme;

    hw_opt.thread_affinity = nenum(thread_aff_opt);
    Flags =  (bSepPot       ? MD_SEPPOT       : 0); // 128
    Flags = Flags | (bDDBondCheck  ? MD_DDBONDCHECK  : 0); // 1024
    Flags = Flags | (bDDBondComm   ? MD_DDBONDCOMM   : 0); // 2048
    Flags = Flags | (bTunePME      ? MD_TUNEPME      : 0); // 1048576;
    Flags = Flags | (bConfout      ? MD_CONFOUT      : 0); // 4096


    gmx_log_open(ftp2fn(efLOG, NFILE, fnm), cr,
                 !bSepPot, Flags & MD_APPENDFILES, &fplog);

    ddxyz[XX] = (int)(realddxyz[XX] + 0.5);
    ddxyz[YY] = (int)(realddxyz[YY] + 0.5);
    ddxyz[ZZ] = (int)(realddxyz[ZZ] + 0.5);

    rc = mdrunner(&hw_opt, fplog, cr, NFILE, fnm, oenv, bVerbose, bCompact,
                  ddxyz, dd_node_order, rdd, rconstr,
                  dddlb_opt[0], dlb_scale, ddcsx, ddcsy, ddcsz,
                  "cpu",
                  nsteps, nstepout, resetstep,
                  nmultisim, repl_ex_nst, repl_ex_nex, repl_ex_seed,
                  pforce, cpt_period, max_hours, deviceOptions, Flags);

    gmx_finalize_par();

    gmx_log_close(fplog);

    return rc;
}
