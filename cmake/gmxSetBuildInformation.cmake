#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012,2013, by the GROMACS development team, led by
# David van der Spoel, Berk Hess, Erik Lindahl, and including many
# others, as listed in the AUTHORS file in the top-level source
# directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.
#

# - Check the username performing the build, as well as date and time
#
# gmx_set_build_information()
#
# The macro variables will be set to the user/host/cpu used for configuration,
# or anonymous/unknown if it cannot be detected (windows)
#
# BUILD_TIME
# BUILD_USER
# BUILD_HOST
# BUILD_CPU_VENDOR
# BUILD_CPU_BRAND
# BUILD_CPU_FAMILY
# BUILD_CPU_MODEL
# BUILD_CPU_STEPPING
# BUILD_CPU_FEATURES
#

# we rely on inline asm support for GNU!
include(gmxTestInlineASM)

macro(gmx_set_build_information)
    IF(NOT DEFINED BUILD_USER)

    gmx_test_inline_asm_gcc_x86(GMX_X86_GCC_INLINE_ASM)

    if(GMX_X86_GCC_INLINE_ASM)
        set(GCC_INLINE_ASM_DEFINE "-DGMX_X86_GCC_INLINE_ASM")
    else(GMX_X86_GCC_INLINE_ASM)
        set(GCC_INLINE_ASM_DEFINE "")
    endif(GMX_X86_GCC_INLINE_ASM)

    message(STATUS "Setting build user/date/host/cpu information")
    if(CMAKE_HOST_UNIX)
        execute_process( COMMAND date     OUTPUT_VARIABLE TMP_TIME    OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process( COMMAND whoami   OUTPUT_VARIABLE TMP_USER       OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process( COMMAND hostname OUTPUT_VARIABLE TMP_HOSTNAME   OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(BUILD_USER    "@TMP_USER@\@@TMP_HOSTNAME@ [CMAKE]" CACHE INTERNAL "Build user")
        set(BUILD_TIME    "@TMP_TIME@" CACHE INTERNAL "Build date & time")
        execute_process( COMMAND uname -srm OUTPUT_VARIABLE TMP_HOST OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(BUILD_HOST    "@TMP_HOST@" CACHE INTERNAL "Build host & architecture")
        message(STATUS "Setting build user & time - OK")
    else(CMAKE_HOST_UNIX)
        set(BUILD_USER    "Anonymous@unknown [CMAKE]" CACHE INTERNAL "Build user")
        set(BUILD_TIME    "Unknown date" CACHE INTERNAL "Build date & time")
        set(BUILD_HOST    "@CMAKE_HOST_SYSTEM@ @CMAKE_HOST_SYSTEM_PROCESSOR@" CACHE INTERNAL "Build host & architecture")
        message(STATUS "Setting build user & time - not on Unix, using anonymous")
    endif(CMAKE_HOST_UNIX)

        
        set(BUILD_CPU_VENDOR   "Unknown, cross-compiled"   CACHE INTERNAL "Build CPU vendor")
        set(BUILD_CPU_BRAND    "Unknown, cross-compiled"    CACHE INTERNAL "Build CPU brand")
        set(BUILD_CPU_FAMILY   "0"   CACHE INTERNAL "Build CPU family")
        set(BUILD_CPU_MODEL    "0"    CACHE INTERNAL "Build CPU model")
        set(BUILD_CPU_STEPPING "0" CACHE INTERNAL "Build CPU stepping")
        set(BUILD_CPU_FEATURES "" CACHE INTERNAL "Build CPU features")


    ENDIF(NOT DEFINED BUILD_USER)
endmacro(gmx_set_build_information)

