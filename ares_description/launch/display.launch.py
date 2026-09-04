"""Show the rover in RViz. Phase 2 gate.

    ros2 launch ares_description display.launch.py

With joint_state_publisher_gui on, drag the rocker and bogie sliders and
watch the linkage. The differential is a Gazebo constraint, not a URDF one,
so in RViz alone the two rockers move independently. That is expected here.
Phase 3 checks the constraint in the simulator, where it is enforced.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PACKAGE = "ares_description"


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare(PACKAGE), "config", "ares.rviz"]
    )
    urdf = PathJoinSubstitution(
        [FindPackageShare(PACKAGE), "urdf", "ares_g.urdf.xacro"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Show the joint slider GUI so the linkage can be driven by hand.",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {"robot_description": Command(["xacro ", urdf])},
                    # The URDF is fully resolved at generation time, so this
                    # is a pass through. It stays here so the file loads the
                    # same way it will once macros are added.
                ],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                condition=IfCondition(gui),
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                condition=UnlessCondition(gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
            ),
        ]
    )
