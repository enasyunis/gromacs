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
#ifdef GMX_THREAD_MPI
#include "thread_mpi.h"
#endif

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
    gmx_bool      bVerbose      = FALSE;
    gmx_bool      bCompact      = TRUE;
    gmx_bool      bSepPot       = FALSE;
    gmx_bool      bRerunVSite   = FALSE;
    gmx_bool      bIonize       = FALSE;
    gmx_bool      bConfout      = TRUE;
    gmx_bool      bReproducible = FALSE;

    int           npme          = 0;
    int           nmultisim     = 0;
    int           nstglobalcomm = -1;
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
    gmx_bool      bAppendFiles          = TRUE;
    gmx_bool      bKeepAndNumCPT        = FALSE;
    gmx_bool      bResetCountersHalfWay = FALSE;
    output_env_t  oenv                  = NULL;
    const char   *deviceOptions         = "";

    gmx_hw_opt_t  hw_opt = {0, 0, 0, 0, threadaffSEL, 0, 0, NULL};

    t_pargs       pa[] = {

        { "-ddcheck", FALSE, etBOOL, {&bDDBondCheck},
          "Check for all bonded interactions with DD" },
        { "-ddbondcomm", FALSE, etBOOL, {&bDDBondComm},
          "HIDDENUse special bonded atom communication when [TT]-rdd[tt] > cut-off" },
        { "-rdd",     FALSE, etREAL, {&rdd},
          "The maximum distance for bonded interactions with DD (nm), 0 is determine from initial coordinates" },
        { "-rcon",    FALSE, etREAL, {&rconstr},
          "Maximum distance for P-LINCS (nm), 0 is estimate" },
        { "-dlb",     FALSE, etENUM, {dddlb_opt},
          "Dynamic load balancing (with DD)" },
        { "-dds",     FALSE, etREAL, {&dlb_scale},
          "Minimum allowed dlb scaling of the DD cell size" },
        { "-ddcsx",   FALSE, etSTR, {&ddcsx},
          "HIDDENThe DD cell sizes in x" },
        { "-ddcsy",   FALSE, etSTR, {&ddcsy},
          "HIDDENThe DD cell sizes in y" },
        { "-ddcsz",   FALSE, etSTR, {&ddcsz},
          "HIDDENThe DD cell sizes in z" },
        { "-gcom",    FALSE, etINT, {&nstglobalcomm},
          "Global communication frequency" },
        { "-nb",      FALSE, etENUM, {&nbpu_opt},
          "Calculate non-bonded interactions on" },
        { "-tunepme", FALSE, etBOOL, {&bTunePME},
          "Optimize PME load between PP/PME nodes or GPU/CPU" },
        { "-testverlet", FALSE, etBOOL, {&bTestVerlet},
          "Test the Verlet non-bonded scheme" },
        { "-v",       FALSE, etBOOL, {&bVerbose},
          "Be loud and noisy" },
        { "-compact", FALSE, etBOOL, {&bCompact},
          "Write a compact log file" },
        { "-seppot",  FALSE, etBOOL, {&bSepPot},
          "Write separate V and dVdl terms for each interaction type and node to the log file(s)" },
        { "-pforce",  FALSE, etREAL, {&pforce},
          "Print all forces larger than this (kJ/mol nm)" },
        { "-reprod",  FALSE, etBOOL, {&bReproducible},
          "Try to avoid optimizations that affect binary reproducibility" },
        { "-cpt",     FALSE, etREAL, {&cpt_period},
          "Checkpoint interval (minutes)" },
        { "-cpnum",   FALSE, etBOOL, {&bKeepAndNumCPT},
          "Keep and number checkpoint files" },
	{ "-append",  FALSE, etBOOL, {&bAppendFiles},
          "Append to previous output files when continuing from checkpoint instead of adding the simulation part number to all file names" },
        { "-nsteps",  FALSE, etINT, {&nsteps},
          "Run this number of steps, overrides .mdp file option" },
        { "-maxh",   FALSE, etREAL, {&max_hours},
          "Terminate after 0.99 times this time (hours)" },
        { "-multi",   FALSE, etINT, {&nmultisim},
          "Do multiple simulations in parallel" },
        { "-replex",  FALSE, etINT, {&repl_ex_nst},
          "Attempt replica exchange periodically with this period (steps)" },
        { "-nex",  FALSE, etINT, {&repl_ex_nex},
          "Number of random exchanges to carry out each exchange interval (N^3 is one suggestion).  -nex zero or not specified gives neighbor replica exchange." },
        { "-reseed",  FALSE, etINT, {&repl_ex_seed},
          "Seed for replica exchange, -1 is generate a seed" },
        { "-rerunvsite", FALSE, etBOOL, {&bRerunVSite},
          "HIDDENRecalculate virtual site coordinates with [TT]-rerun[tt]" },
        { "-ionize",  FALSE, etBOOL, {&bIonize},
          "Do a simulation including the effect of an X-Ray bombardment on your system" },
        { "-confout", FALSE, etBOOL, {&bConfout},
          "HIDDENWrite the last configuration with [TT]-c[tt] and force checkpointing at the last step" },
        { "-stepout", FALSE, etINT, {&nstepout},
          "HIDDENFrequency of writing the remaining runtime" },
        { "-resetstep", FALSE, etINT, {&resetstep},
          "HIDDENReset cycle counters after these many time steps" },
        { "-resethway", FALSE, etBOOL, {&bResetCountersHalfWay},
          "HIDDENReset the cycle counters after half the number of steps or halfway [TT]-maxh[tt]" }
    };
    gmx_edsam_t   ed;
    unsigned long Flags, PCA_Flags;
    ivec          ddxyz;
    int           dd_node_order;
    gmx_bool      bAddPart;
    FILE         *fplog, *fpmulti;
    int           sim_part, sim_part_fn;
    const char   *part_suffix = ".part";
    char          suffix[STRLEN];
    int           rc;
    char        **multidir = NULL;


    cr = init_par(&argc, &argv);

    PCA_Flags = (PCA_CAN_SET_DEFFNM | (MASTER(cr) ? 0 : PCA_QUIET));

    parse_common_args(&argc, argv, PCA_Flags, NFILE, fnm, asize(pa), pa,
                      asize(desc), desc, 0, NULL, &oenv);


    dd_node_order = nenum(ddno_opt);
    cr->npmenodes = npme;

    hw_opt.thread_affinity = nenum(thread_aff_opt);

    bAddPart = TRUE;

    sim_part    = 1;
    sim_part_fn = sim_part;

    bAppendFiles = FALSE;
    sim_part_fn = sim_part;

    if (bAddPart)
    {
        /* Rename all output files (except checkpoint files) */
        /* create new part name first (zero-filled) */
        sprintf(suffix, "%s%04d", part_suffix, sim_part_fn);

        add_suffix_to_output_names(fnm, NFILE, suffix);
        if (MULTIMASTER(cr))
        {
            fprintf(stdout, "Checkpoint file is from part %d, new output files will be suffixed '%s'.\n", sim_part-1, suffix);
        }
    }

    Flags = opt2bSet("-rerun", NFILE, fnm) ? MD_RERUN : 0;
    Flags = Flags | (bSepPot       ? MD_SEPPOT       : 0);
    Flags = Flags | (bIonize       ? MD_IONIZE       : 0);
    Flags = Flags | (bPartDec      ? MD_PARTDEC      : 0);
    Flags = Flags | (bDDBondCheck  ? MD_DDBONDCHECK  : 0);
    Flags = Flags | (bDDBondComm   ? MD_DDBONDCOMM   : 0);
    Flags = Flags | (bTunePME      ? MD_TUNEPME      : 0);
    Flags = Flags | (bTestVerlet   ? MD_TESTVERLET   : 0);
    Flags = Flags | (bConfout      ? MD_CONFOUT      : 0);
    Flags = Flags | (bRerunVSite   ? MD_RERUN_VSITE  : 0);
    Flags = Flags | (bReproducible ? MD_REPRODUCIBLE : 0);
    Flags = Flags | (bAppendFiles  ? MD_APPENDFILES  : 0);
    Flags = Flags | (opt2parg_bSet("-append", asize(pa), pa) ? MD_APPENDFILESSET : 0);
    Flags = Flags | (bKeepAndNumCPT ? MD_KEEPANDNUMCPT : 0);
    Flags = Flags | (sim_part > 1    ? MD_STARTFROMCPT : 0);
    Flags = Flags | (bResetCountersHalfWay ? MD_RESETCOUNTERSHALFWAY : 0);


    /* We postpone opening the log file if we are appending, so we can
       first truncate the old log file and append to the correct position
       there instead.  */
    if ((MASTER(cr) || bSepPot) && !bAppendFiles)
    {
        gmx_log_open(ftp2fn(efLOG, NFILE, fnm), cr,
                     !bSepPot, Flags & MD_APPENDFILES, &fplog);
        CopyRight(fplog, argv[0]);
        please_cite(fplog, "Hess2008b");
        please_cite(fplog, "Spoel2005a");
        please_cite(fplog, "Lindahl2001a");
        please_cite(fplog, "Berendsen95a");
    }
    else if (!MASTER(cr) && bSepPot)
    {
        gmx_log_open(ftp2fn(efLOG, NFILE, fnm), cr, !bSepPot, Flags, &fplog);
    }
    else
    {
        fplog = NULL;
    }

    ddxyz[XX] = (int)(realddxyz[XX] + 0.5);
    ddxyz[YY] = (int)(realddxyz[YY] + 0.5);
    ddxyz[ZZ] = (int)(realddxyz[ZZ] + 0.5);

    rc = mdrunner(&hw_opt, fplog, cr, NFILE, fnm, oenv, bVerbose, bCompact,
                  nstglobalcomm, ddxyz, dd_node_order, rdd, rconstr,
                  dddlb_opt[0], dlb_scale, ddcsx, ddcsy, ddcsz,
                  nbpu_opt[0],
                  nsteps, nstepout, resetstep,
                  nmultisim, repl_ex_nst, repl_ex_nex, repl_ex_seed,
                  pforce, cpt_period, max_hours, deviceOptions, Flags);

    gmx_finalize_par();

    if (MULTIMASTER(cr))
    {
        thanx(stderr);
    }

    /* Log file has to be closed in mdrunner if we are appending to it
       (fplog not set here) */
    if (MASTER(cr) && !bAppendFiles)
    {
        gmx_log_close(fplog);
    }

    return rc;
}
