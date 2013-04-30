configure_file(config.h.cmakein config.h)
configure_file(buildinfo.h.cmakein buildinfo.h)


add_subdirectory(gmxlib)
add_subdirectory(mdlib)
add_subdirectory(kernel)

