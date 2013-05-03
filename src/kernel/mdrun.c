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



    rvec          realddxyz          = {0, 0, 0};
    char         *ddcsx                 = NULL, *ddcsy = NULL, *ddcsz = NULL;
    output_env_t  oenv                  = NULL;
    const char   *deviceOptions         = "";

    gmx_hw_opt_t  hw_opt = {0, 0, 0, 0, threadaffSEL, 0, 0, NULL};

    unsigned long Flags;
    ivec          ddxyz;
    FILE         *fplog;
    int           rc;


    cr = init_par(&argc, &argv);

    parse_common_args(&argc, argv, ((1<<10) |  (1<<12)), NFILE, fnm, 0, NULL, 
                      asize(desc), desc, 0, NULL, &oenv);

    cr->npmenodes = 0;

    hw_opt.thread_affinity = 4; //nenum(thread_aff_opt);
    Flags =  ( MD_SEPPOT ); // 128
    Flags = Flags | ( MD_DDBONDCHECK  ); // 1024
    Flags = Flags | ( MD_DDBONDCOMM   ); // 2048
    Flags = Flags | ( MD_TUNEPME      ); // 1048576;
    Flags = Flags | ( MD_CONFOUT  ); // 4096


    gmx_log_open(ftp2fn(efLOG, NFILE, fnm), cr,
                 FALSE, Flags & MD_APPENDFILES, &fplog);

    ddxyz[XX] = (int)(realddxyz[XX] + 0.5);
    ddxyz[YY] = (int)(realddxyz[YY] + 0.5);
    ddxyz[ZZ] = (int)(realddxyz[ZZ] + 0.5);

    rc = mdrunner(&hw_opt, fplog, cr, NFILE, fnm, oenv, 
                  ddxyz,  
                  ddcsx, ddcsy, ddcsz,
                  deviceOptions, Flags);

    gmx_finalize_par();

    gmx_log_close(fplog);

    return rc;
}
