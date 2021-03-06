include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# add target that generates version.c every time a make is run
# only do this if we generate the version
if(USE_VERSION_H)
    add_custom_target(gmx_version ALL
            COMMAND ${CMAKE_COMMAND} 
                -D GIT_EXECUTABLE="${GIT_EXECUTABLE}"
                -D GIT_VERSION="${GIT_VERSION}"
                -D PROJECT_VERSION="${PROJECT_VERSION}"
                -D PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
                -D VERSION_C_CMAKEIN="${CMAKE_SOURCE_DIR}/src/gmxlib/version.c.cmakein"
                -D VERSION_C_OUT="${CMAKE_CURRENT_BINARY_DIR}/version.c"
                -P ${CMAKE_SOURCE_DIR}/cmake/gmxGenerateVersionInfo.cmake 
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/src/gmxlib 
            DEPENDS ${CMAKE_SOURCE_DIR}/src/gmxlib/version.c.cmakein
            COMMENT "Generating version information")
endif()

# The nonbonded directory contains subdirectories that are only
# conditionally built, so we cannot use a GLOB_RECURSE here.
file(GLOB GMXLIB_SOURCES *.c)

if(NOT GMX_USE_PLUGINS)
  list(REMOVE_ITEM GMXLIB_SOURCES vmdio.c vmddlopen.c)
endif()

# This source file is generated
file(GLOB VERSION_SOURCE version.c)

if(VERSION_SOURCE)
	list(REMOVE_ITEM GMXLIB_SOURCES ${VERSION_SOURCE})
endif(VERSION_SOURCE)

# add version.c to the list of sources and tell cmake that it is generated
if(USE_VERSION_H)
LIST(APPEND GMXLIB_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/version.c) # auto-generated
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/version.c 
                PROPERTIES GENERATED true)
endif()

if(NOT GMX_EXTERNAL_BLAS)
  file(GLOB BLAS_SOURCES gmx_blas/*.c)
endif(NOT GMX_EXTERNAL_BLAS)

if(NOT GMX_EXTERNAL_LAPACK)
  file(GLOB LAPACK_SOURCES gmx_lapack/*.c)
endif(NOT GMX_EXTERNAL_LAPACK)


# NONBONDED_SOURCES is imported from the nonbonded subdirectory.
add_library(gmx ${GMXLIB_SOURCES} ${BLAS_SOURCES} ${LAPACK_SOURCES} ${THREAD_MPI_SRC} ${NONBONDED_SOURCES})
target_link_libraries(gmx ${GMX_GPU_LIBRARIES} ${GMX_EXTRA_LIBRARIES} ${FFT_LIBRARIES} ${THREAD_LIB} ${OpenMP_SHARED_LINKER_FLAGS})
if(USE_VERSION_H)
	add_dependencies(gmx gmx_version) 
endif()
set_target_properties(gmx PROPERTIES OUTPUT_NAME "gmx${GMX_LIBS_SUFFIX}" SOVERSION ${SOVERSION} INSTALL_NAME_DIR "${LIB_INSTALL_DIR}"
    COMPILE_FLAGS "${OpenMP_C_FLAGS}" COMPILE_DEFINITIONS "TMPI_EXPORTS")

install(TARGETS gmx DESTINATION ${LIB_INSTALL_DIR} COMPONENT libraries)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgmx.pc.cmakein ${CMAKE_CURRENT_BINARY_DIR}/libgmx.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgmx.pc
        DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
        RENAME "libgmx${GMX_LIBS_SUFFIX}.pc"
        COMPONENT development)

