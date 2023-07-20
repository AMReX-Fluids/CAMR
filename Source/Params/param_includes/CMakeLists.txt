target_sources(${CAMR_exe_name}
   PRIVATE
     CAMR_defaults.H
     CAMR_params.H
     CAMR_queries.H
   )

target_include_directories(${CAMR_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
