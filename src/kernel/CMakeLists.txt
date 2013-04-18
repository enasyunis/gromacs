
set(GMXPREPROCESS_SOURCES 
    calc_verletbuf.c
    compute_io.c    
    )

set(MDRUN_SOURCES 
    runner.c
    md.c
    mdrun.c )

add_library(gmxpreprocess ${GMXPREPROCESS_SOURCES})
target_link_libraries(gmxpreprocess md)
set_target_properties(gmxpreprocess PROPERTIES OUTPUT_NAME "gmxpreprocess${GMX_LIBS_SUFFIX}" SOVERSION ${SOVERSION} INSTALL_NAME_DIR "${LIB_INSTALL_DIR}"
    COMPILE_FLAGS "${OpenMP_C_FLAGS}")


if(GMX_GPU)
    include_directories(${CMAKE_SOURCE_DIR}/src/gmxlib/gpu_utils)
endif()

if(GMX_OPENMM)
    # Even though the OpenMM build has "moved to contrib", many things
    # have be be done from within the scope of the CMakeLists.txt that
    # builds its mdrun, and that is here
    include(../contrib/BuildMdrunOpenMM)
endif(GMX_OPENMM)

if(GMX_GPU OR GMX_FORCE_CXX)
    set_source_files_properties(main.c PROPERTIES LANGUAGE CXX)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set_source_files_properties(main.c PROPERTIES COMPILE_FLAGS "-x c++")
    endif()
endif()


list(APPEND GMX_EXTRA_LIBRARIES gmxpreprocess md ${OpenMP_LINKER_FLAGS})



foreach(PROGRAM ${GMX_KERNEL_PROGRAMS})
    add_executable(${PROGRAM} ${PROGRAM}.c main.c)
    target_link_libraries(${PROGRAM} ${GMX_EXTRA_LIBRARIES})
    set_target_properties(${PROGRAM} PROPERTIES OUTPUT_NAME "${PROGRAM}${GMX_BINARY_SUFFIX}")
endforeach()

add_executable(mdrun ${MDRUN_SOURCES} main.c)
target_link_libraries(mdrun ${GMX_EXTRA_LIBRARIES})
set_target_properties(mdrun PROPERTIES OUTPUT_NAME "mdrun${GMX_BINARY_SUFFIX}" COMPILE_FLAGS "${OpenMP_C_FLAGS}")

# Construct component groups for installation; note that a component may
# belong to only one group
foreach(PROGRAM ${GMX_KERNEL_PROGRAMS})
    set(CPACK_COMPONENT_${PROGRAM}_GROUP tools)
endforeach()
set(CPACK_COMPONENT_MDRUN_GROUP mdrun)

foreach(PROGRAM ${GMX_KERNEL_PROGRAMS} mdrun)
    # Manage CPack component dependencies
    set(CPACK_COMPONENT_${PROGRAM}_DEPENDS libraries libraries-gmxpreprocess)

    # Create custom install-xxxx target
    if (BUILD_SHARED_LIBS)
    # If shared libraries are used, we need to install the libraries in
    # addition to the mdrun binary.
       add_custom_target(install-${PROGRAM}
           COMMAND ${CMAKE_COMMAND} -DCOMPONENT=libraries
                   -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
           COMMAND ${CMAKE_COMMAND} -DCOMPONENT=libraries-gmxpreprocess
                   -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
           COMMAND ${CMAKE_COMMAND} -DCOMPONENT=${PROGRAM}
                   -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
           COMMENT "Installing ${PROGRAM}")
    else()
       add_custom_target(install-${PROGRAM}
           COMMAND ${CMAKE_COMMAND} -DCOMPONENT=${PROGRAM}
                   -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
           COMMENT "Installing ${PROGRAM}")
    endif()
    add_dependencies(install-${PROGRAM} ${PROGRAM})

    # Finally, trigger installation
    install(
      TARGETS ${PROGRAM}
      COMPONENT ${PROGRAM}
      DESTINATION ${BIN_INSTALL_DIR}
      )
endforeach()

install(TARGETS gmxpreprocess DESTINATION ${LIB_INSTALL_DIR} COMPONENT libraries-gmxpreprocess)

if (INSTALL_CUDART_LIB) #can be set manual by user
    if (GMX_GPU)
        foreach(CUDA_LIB ${CUDA_LIBRARIES})
            string(REGEX MATCH "cudart" IS_CUDART ${CUDA_LIB})
            if(IS_CUDART) #libcuda should not be installed
                #install also name-links (linker uses those)
                file(GLOB CUDA_LIBS ${CUDA_LIB}*)
                install(FILES ${CUDA_LIBS} DESTINATION
                    ${LIB_INSTALL_DIR} COMPONENT libraries)
            endif()
        endforeach()
    else()
        message(WARNING "INSTALL_CUDART_LIB only makes sense with GMX_GPU")
    endif()
endif ()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgmxpreprocess.pc.cmakein ${CMAKE_CURRENT_BINARY_DIR}/libgmxpreprocess.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgmxpreprocess.pc
        DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
        RENAME "libgmxpreprocess${GMX_LIBS_SUFFIX}.pc"
        COMPONENT development)
