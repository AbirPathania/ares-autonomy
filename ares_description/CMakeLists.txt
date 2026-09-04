cmake_minimum_required(VERSION 3.10)
project(ares_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes config launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # The geometry gate. Pure Python, no ROS, so it also runs on a laptop.
  ament_add_pytest_test(urdf_geometry test/test_urdf_geometry.py)
endif()

ament_package()
