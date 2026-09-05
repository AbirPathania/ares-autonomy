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

# Two corrections to the pip layer as section 06.2 prints it, both found by
# running it rather than reading it.
#
# 1. ultralytics pulls psutil, and pip cannot uninstall the copy apt already
#    installed because a Debian packaged module carries no RECORD file. The
#    single line in the spec therefore fails the whole image build with
#    "Cannot uninstall psutil 5.9.8". Reinstalling psutil under pip first
#    puts it under pip's management and the rest proceeds.
#
# 2. torch defaults to the CUDA wheels, which drag in several gigabytes of
#    GPU runtime. Section 06.4 is explicit that there is no GPU here and
#    that anything CUDA is out, so the CPU index is not a downgrade, it is
#    the correct build for this tier. It also keeps the image inside the
#    32 GB a Codespace gets.
RUN pip install --break-system-packages --ignore-installed psutil

RUN pip install --break-system-packages \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision

RUN pip install --break-system-packages \
      numpy scipy pint sympy pytest hypothesis pyyaml ruff \
      ultralytics open3d
