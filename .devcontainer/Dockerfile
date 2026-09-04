# .devcontainer/Dockerfile
#
# Build spec section 06.2. ROS 2 Jazzy Jalisco, which is LTS to May 2029 and
# pairs with Gazebo Harmonic. Not Humble: Humble has eight months of support
# left. Isaac ROS targets Humble, and that tension is real and accepted, see
# section 06.1. Tier 3 is a data generation and validation environment, not
# where the flight software lives, so the bridge between them is recorded
# bags and trained weights, not shared source.
#
# osrf/ros images are multi-architecture, so this runs arm64 natively on
# Apple Silicon with no emulation and no penalty on the physics.
FROM osrf/ros:jazzy-desktop-full

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
      ros-jazzy-rtabmap-ros ros-jazzy-image-pipeline ros-jazzy-vision-opencv \
      python3-pip python3-opencv git-lfs \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages \
      numpy scipy pint sympy pytest hypothesis pyyaml \
      torch torchvision ultralytics open3d
