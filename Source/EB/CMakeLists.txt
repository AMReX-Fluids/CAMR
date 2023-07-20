target_include_directories(CAMR PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_sources(CAMR
   PRIVATE
   embedded_boundaries.cpp
   eb_box.cpp
   eb_cylinder.cpp
   eb_regular.cpp
   eb_sphere.cpp
   writeEBsurface.cpp
   eb_if.H
   )
