### LLNL units ###
set(UNITS_INSTALL OFF CACHE INTERNAL "")

set(UNITS_BUILD_OBJECT_LIBRARY OFF CACHE INTERNAL "")
set(UNITS_BUILD_STATIC_LIBRARY OFF CACHE INTERNAL "")
set(UNITS_BUILD_SHARED_LIBRARY ON CACHE INTERNAL "")
set(UNITS_BUILD_CONVERTER_APP OFF CACHE INTERNAL "")
set(UNITS_BUILD_WEBSERVER OFF CACHE INTERNAL "")
set(UNITS_CLANG_TIDY_OPTIONS "" CACHE INTERNAL "")
set(UNITS_BUILD_FUZZ_TARGETS OFF CACHE INTERNAL "")

add_subdirectory(
     "${PROJECT_SOURCE_DIR}/../external/units" "${PROJECT_BINARY_DIR}/../external/units"
)

if(Kokkos_ENABLE_CUDA)
  set_target_properties(units PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
endif()

### Dyablo units ###

add_library(dyablo_units "")

target_sources(dyablo_units
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Units.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CodeUnits.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitParser.cpp
)

target_link_libraries( dyablo_units PUBLIC Kokkos::kokkos units::units )

target_include_directories(dyablo_units
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/units
  ${CMAKE_SOURCE_DIR}/src
  )

add_library(dyablo::units ALIAS dyablo_units)
