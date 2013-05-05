/*
 * This file is part of the GROMACS molecular simulation package.
 *
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

#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "types/enums.h"
#include "types/hw_info.h"
#include "types/commrec.h"
#include "gmx_fatal.h"
#include "gmx_fatal_collective.h"
#include "smalloc.h"
#include "gpu_utils.h"
#include "statutil.h"
#include "gmx_detect_hardware.h"
#include "main.h"
#include "md_logging.h"

#if ((defined(WIN32) || defined( _WIN32 ) || defined(WIN64) || defined( _WIN64 )) && !(defined (__CYGWIN__) || defined (__CYGWIN32__)))
#include "windows.h"
#endif

/* Although we can't have more than 10 GPU different ID-s passed by the user as
 * the id-s are assumed to be represented by single digits, as multiple
 * processes can share a GPU, we can end up with more than 10 IDs.
 * To account for potential extreme cases we'll set the limit to a pretty
 * ridiculous number. */
static unsigned int max_gpu_ids_user = 64;

static const char * invalid_gpuid_hint =
    "A delimiter-free sequence of valid numeric IDs of available GPUs is expected.";

/* FW decl. */
void limit_num_gpus_used(gmx_hw_info_t *hwinfo, int count);

static void sprint_gpus(char *sbuf, const gmx_gpu_info_t *gpu_info, gmx_bool bPrintAll)
{
    int      i, ndev;
    char     stmp[STRLEN];

    ndev = gpu_info->ncuda_dev;

    sbuf[0] = '\0';
    for (i = 0; i < ndev; i++)
    {
        get_gpu_device_info_string(stmp, gpu_info, i);
        strcat(sbuf, "  ");
        strcat(sbuf, stmp);
        if (i < ndev - 1)
        {
            strcat(sbuf, "\n");
        }
    }
}

static void print_gpu_detection_stats(FILE                 *fplog,
                                      const gmx_gpu_info_t *gpu_info,
                                      const t_commrec      *cr)
{
    char onhost[266], stmp[STRLEN];
    int  ngpu;

    ngpu = gpu_info->ncuda_dev;

#if defined GMX_MPI && !defined GMX_THREAD_MPI
    /* We only print the detection on one, of possibly multiple, nodes */
    strncpy(onhost, " on host ", 10);
    gmx_gethostname(onhost+9, 256);
#else
    /* We detect all relevant GPUs */
    strncpy(onhost, "", 1);
#endif

    if (ngpu > 0)
    {
        sprint_gpus(stmp, gpu_info, TRUE);
        md_print_warn(cr, fplog, "%d GPU%s detected%s:\n%s\n",
                      ngpu, (ngpu > 1) ? "s" : "", onhost, stmp);
    }
    else
    {
        md_print_warn(cr, fplog, "No GPUs detected%s\n", onhost);
    }
}


/* Parse a "plain" GPU ID string which contains a sequence of digits corresponding
 * to GPU IDs; the order will indicate the process/tMPI thread - GPU assignment. */
static void parse_gpu_id_plain_string(const char *idstr, int *nid, int *idlist)
{
    int    i;
    size_t len_idstr;

    len_idstr = strlen(idstr);

    if (len_idstr > max_gpu_ids_user)
    {
        gmx_fatal(FARGS, "%d GPU IDs provided, but only at most %d are supported",
                  len_idstr, max_gpu_ids_user);
    }

    *nid = len_idstr;

    for (i = 0; i < *nid; i++)
    {
        if (idstr[i] < '0' || idstr[i] > '9')
        {
            gmx_fatal(FARGS, "Invalid character in GPU ID string: '%c'\n%s\n",
                      idstr[i], invalid_gpuid_hint);
        }
        idlist[i] = idstr[i] - '0';
    }
}



/* Return the number of hardware threads supported by the current CPU.
 * We assume that this is equal with the number of CPUs reported to be
 * online by the OS at the time of the call.
 */
static int get_nthreads_hw_avail(FILE *fplog, const t_commrec *cr)
{
    int ret = 0;

#if ((defined(WIN32) || defined( _WIN32 ) || defined(WIN64) || defined( _WIN64 )) && !(defined (__CYGWIN__) || defined (__CYGWIN32__)))
    /* Windows */
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    ret = sysinfo.dwNumberOfProcessors;
#elif defined HAVE_SYSCONF
    /* We are probably on Unix.
     * Now check if we have the argument to use before executing the call
     */
#if defined(_SC_NPROCESSORS_ONLN)
    ret = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_SC_NPROC_ONLN)
    ret = sysconf(_SC_NPROC_ONLN);
#elif defined(_SC_NPROCESSORS_CONF)
    ret = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(_SC_NPROC_CONF)
    ret = sysconf(_SC_NPROC_CONF);
#endif /* End of check for sysconf argument values */

#else
    /* Neither windows nor Unix. No fscking idea how many CPUs we have! */
    ret = -1;
#endif

    if (debug)
    {
        fprintf(debug, "Detected %d processors, will use this as the number "
                "of supported hardware threads.\n", ret);
    }

#ifdef GMX_OMPENMP
    if (ret != gmx_omp_get_num_procs())
    {
        md_print_warn(cr, fplog,
                      "Number of CPUs detected (%d) does not match the number reported by OpenMP (%d).\n"
                      "Consider setting the launch configuration manually!",
                      ret, gmx_omp_get_num_procs());
    }
#endif

    return ret;
}

void gmx_detect_hardware(FILE *fplog, gmx_hw_info_t *hwinfo,
                         const t_commrec *cr,
                         gmx_bool bForceUseGPU, gmx_bool bTryUseGPU,
                         const char *gpu_id)
{
    int              i;
    const char      *env;
    char             sbuf[STRLEN], stmp[STRLEN];
    gmx_hw_info_t   *hw;
    gmx_gpu_info_t   gpuinfo_auto, gpuinfo_user;
    gmx_bool         bGPUBin;

    assert(hwinfo);

    /* detect CPUID info; no fuss, we don't detect system-wide
     * -- sloppy, but that's it for now */
    if (gmx_cpuid_init(&hwinfo->cpuid_info) != 0)
    {
        gmx_fatal_collective(FARGS, cr, NULL, "CPUID detection failed!");
    }

    /* detect number of hardware threads */
    hwinfo->nthreads_hw_avail = get_nthreads_hw_avail(fplog, cr);

    /* detect GPUs */
    hwinfo->gpu_info.ncuda_dev_use  = 0;
    hwinfo->gpu_info.cuda_dev_use   = NULL;
    hwinfo->gpu_info.ncuda_dev      = 0;
    hwinfo->gpu_info.cuda_dev       = NULL;

#ifdef GMX_GPU
    bGPUBin      = TRUE;
#else
    bGPUBin      = FALSE;
#endif

    /* Bail if binary is not compiled with GPU acceleration, but this is either
     * explicitly (-nb gpu) or implicitly (gpu ID passed) requested. */
    if (bForceUseGPU && !bGPUBin)
    {
        gmx_fatal(FARGS, "GPU acceleration requested, but %s was compiled without GPU support!", ShortProgram());
    }
    if (gpu_id != NULL && !bGPUBin)
    {
        gmx_fatal(FARGS, "GPU ID string set, but %s was compiled without GPU support!", ShortProgram());
    }

    /* run the detection if the binary was compiled with GPU support */
    if (bGPUBin && getenv("GMX_DISABLE_GPU_DETECTION") == NULL)
    {
        char detection_error[STRLEN];

        if (detect_cuda_gpus(&hwinfo->gpu_info, detection_error) != 0)
        {
            if (detection_error != NULL && detection_error[0] != '\0')
            {
                sprintf(sbuf, ":\n      %s\n", detection_error);
            }
            else
            {
                sprintf(sbuf, ".");
            }
            md_print_warn(cr, fplog,
                          "NOTE: Error occurred during GPU detection%s"
                          "      Can not use GPU acceleration, will fall back to CPU kernels.\n",
                          sbuf);
        }
    }

    if (bForceUseGPU || bTryUseGPU)
    {
        env = getenv("GMX_GPU_ID");
        if (env != NULL && gpu_id != NULL)
        {
            gmx_fatal(FARGS, "GMX_GPU_ID and -gpu_id can not be used at the same time");
        }
        if (env == NULL)
        {
            env = gpu_id;
        }

        /* parse GPU IDs if the user passed any */
        if (env != NULL)
        {
            int *gpuid, *checkres;
            int  nid, res;

            snew(gpuid, max_gpu_ids_user);
            snew(checkres, max_gpu_ids_user);

            parse_gpu_id_plain_string(env, &nid, gpuid);

            if (nid == 0)
            {
                gmx_fatal(FARGS, "Empty GPU ID string encountered.\n%s\n", invalid_gpuid_hint);
            }

            res = check_select_cuda_gpus(checkres, &hwinfo->gpu_info, gpuid, nid);

            if (!res)
            {
                print_gpu_detection_stats(fplog, &hwinfo->gpu_info, cr);

                sprintf(sbuf, "Some of the requested GPUs do not exist, behave strangely, or are not compatible:\n");
                for (i = 0; i < nid; i++)
                {
                    if (checkres[i] != egpuCompatible)
                    {
                        sprintf(stmp, "    GPU #%d: %s\n",
                                gpuid[i], gpu_detect_res_str[checkres[i]]);
                        strcat(sbuf, stmp);
                    }
                }
                gmx_fatal(FARGS, "%s", sbuf);
            }

            hwinfo->gpu_info.bUserSet = TRUE;

            sfree(gpuid);
            sfree(checkres);
        }
        else
        {
            pick_compatible_gpus(&hwinfo->gpu_info);
            hwinfo->gpu_info.bUserSet = FALSE;
        }

        /* decide whether we can use GPU */
        hwinfo->bCanUseGPU = (hwinfo->gpu_info.ncuda_dev_use > 0);
        if (!hwinfo->bCanUseGPU && bForceUseGPU)
        {
            gmx_fatal(FARGS, "GPU acceleration requested, but no compatible GPUs were detected.");
        }
    }
}

