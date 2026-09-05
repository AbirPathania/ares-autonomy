"""Bring up a world, the rover, the bridge, TF and RViz.

    ros2 launch ares_bringup sim.launch.py world:=trench_field

Headless by default. Build spec section 06.2: run the server headless and view
everything through RViz, which is cheap to render. The Gazebo GUI on software
rendered cores is not worth the cycles, and on the four core machine this
project actually gets it is the difference between a simulation that runs and
one that does not.

The bridge list is deliberately explicit. Bridging everything is how a
ground truth depth topic ends up subscribed by accident, and section 10's one
honesty rule is that /sim/ground_truth/depth exists for scoring and for
nothing else. It is bridged, because scoring needs it, and it is named so that
a perception node subscribing to it is visible in a launch file diff.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

WORLDS = ("flat", "step_field", "trench_field", "snow_scree")

# Every topic that crosses between Gazebo and ROS, named one at a time.
BRIDGE = [
    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
    "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
    "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
    "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
    # The two eyes of each unit. These are what the perception path may use.
    "/mast_cam/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
    "/mast_cam_right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
    "/mast_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    "/deck_cam/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
    "/deck_cam_right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image",
    "/deck_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
    # Scoring only. Never subscribe to this from the control path.
    "/sim/ground_truth/depth@sensor_msgs/msg/Image[gz.msgs.Image",
]


def generate_launch_description():
    world = LaunchConfiguration("world")
    headless = LaunchConfiguration("headless")
    rviz = LaunchConfiguration("rviz")
    model = LaunchConfiguration("model")

    world_file = PathJoinSubstitution(
        [
            FindPackageShare("ares_sim"),
            "worlds",
            PythonExpression(["'", world, "' + '.sdf'"]),
        ]
    )
    urdf = PathJoinSubstitution(
        [
            FindPackageShare("ares_description"),
            "urdf",
            PythonExpression(["'", model, "' + '.urdf.xacro'"]),
        ]
    )
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("ares_description"), "config", "ares.rviz"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="flat",
                description=f"One of {', '.join(WORLDS)}. Gate order is section 10.",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="Physics only. Watch through RViz, not the Gazebo GUI.",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "model",
                default_value="ares_g",
                description=(
                    "ares_g is the mast at 0.95 m. ares_g_mast_050 is the same "
                    "vehicle at 0.50 m, which section 02 says must FAIL the "
                    "trench gate. Phase 5 runs both. If the low one passes, the "
                    "simulation is wrong, not the analysis."
                ),
            ),
            # ogre, not ogre2, and the server alone unless asked otherwise.
            ExecuteProcess(
                cmd=[
                    "gz",
                    "sim",
                    "-s",
                    "-r",
                    "--render-engine",
                    "ogre",
                    world_file,
                ],
                output="screen",
                condition=IfCondition(headless),
            ),
            ExecuteProcess(
                cmd=["gz", "sim", "-r", "--render-engine", "ogre", world_file],
                output="screen",
                condition=UnlessCondition(headless),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[
                    {
                        "robot_description": Command(["xacro ", urdf]),
                        "use_sim_time": True,
                    }
                ],
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-topic",
                    "/robot_description",
                    "-name",
                    "ares_g",
                    "-z",
                    "0.2",
                ],
                output="screen",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=BRIDGE,
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            GroupAction(
                [
                    Node(
                        package="rviz2",
                        executable="rviz2",
                        arguments=["-d", rviz_config],
                        parameters=[{"use_sim_time": True}],
                    )
                ],
                condition=IfCondition(rviz),
            ),
        ]
    )
