

set(MDRUN_SOURCES 
    runner.c
    md.c
    mdrun.c )


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


list(APPEND GMX_EXTRA_LIBRARIES md ${OpenMP_LINKER_FLAGS})



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
    set(CPACK_COMPONENT_${PROGRAM}_DEPENDS libraries )

    # Create custom install-xxxx target
    if (BUILD_SHARED_LIBS)
    # If shared libraries are used, we need to install the libraries in
    # addition to the mdrun binary.
       add_custom_target(install-${PROGRAM}
           COMMAND ${CMAKE_COMMAND} -DCOMPONENT=libraries
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

