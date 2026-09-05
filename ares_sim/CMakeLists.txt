cmake_minimum_required(VERSION 3.16)
project(ares_sim)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor QUIET)
find_package(gz_plugin_vendor QUIET)

# The differential plugin needs Gazebo headers. Everything else in this
# package is data, so the worlds still install on a machine without them and
# the geometry tests still run.
if(gz_sim_vendor_FOUND)
  find_package(gz-sim8 REQUIRED)
  find_package(gz-plugin2 REQUIRED COMPONENTS register)

  add_library(ares_differential SHARED src/rocker_differential.cc)
  target_include_directories(ares_differential PRIVATE include)
  target_link_libraries(ares_differential
    PRIVATE gz-sim8::gz-sim8 gz-plugin2::register)

  install(TARGETS ares_differential
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
else()
  message(STATUS "Gazebo not found, skipping the rocker differential plugin")
endif()

install(
  DIRECTORY worlds textures launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(worlds test/test_worlds.py)
endif()

ament_package()
