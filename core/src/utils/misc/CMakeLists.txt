#######################
### MISC ##############
#######################
add_library(misc INTERFACE)

target_include_directories(misc
  INTERFACE
  ${CMAKE_SOURCE_DIR}/src/utils/misc
  )

add_library(dyablo::misc ALIAS misc)
