

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

if(GMX_FORCE_CXX)
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

set(CPACK_COMPONENT_MDRUN_GROUP mdrun)

foreach(PROGRAM ${GMX_KERNEL_PROGRAMS} mdrun)
    # Manage CPack component dependencies
    set(CPACK_COMPONENT_${PROGRAM}_DEPENDS libraries )

    # Finally, trigger installation
    install(
      TARGETS ${PROGRAM}
      COMPONENT ${PROGRAM}
      DESTINATION ${BIN_INSTALL_DIR}
      )
endforeach()



