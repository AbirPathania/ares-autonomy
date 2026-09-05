"""The perception path. Two eyes in, traversability and a speed limit out.

    ros2 launch ares_bringup perception.launch.py

Subscribes to the two rendered RGB topics per unit and to nothing else. There
is no depth camera in this graph. Build spec section 10:

    Simulate two RGB cameras and run the real stereo matcher on the rendered
    images. Never subscribe to a ground truth depth topic in the perception
    path.

The mast unit carries the speed limit, because section 02 derives it from
mast height. The deck unit fills the mast's blind zone between 0.17 and 0.81 m
and confirms obstacles close in during a stop and inspect. It is a coverage
scheme, not a redundancy scheme, so the two are not interchangeable and the
launch file does not treat them as if they were.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ares_math.constants import SENSORS


def generate_launch_description():
    use_deck = LaunchConfiguration("deck_unit")
    semantics = LaunchConfiguration("semantics")

    return LaunchDescription(
        [
            DeclareLaunchArgument("deck_unit", default_value="true"),
            DeclareLaunchArgument(
                "semantics",
                default_value="false",
                description=(
                    "Segmentation and detection. Off until weights exist, "
                    "because a head with no weights is a node that logs an "
                    "error every frame and hides the ones that matter."
                ),
            ),
            Node(
                package="ares_perception",
                executable="stereo_node",
                name="mast_stereo",
                parameters=[
                    {
                        "unit": "mast_cam",
                        "baseline": SENSORS.mast_baseline,
                        "use_sim_time": True,
                    }
                ],
                output="screen",
            ),
            Node(
                package="ares_perception",
                executable="stereo_node",
                name="deck_stereo",
                parameters=[
                    {
                        "unit": "deck_cam",
                        "baseline": SENSORS.deck_baseline,
                        "use_sim_time": True,
                    }
                ],
                condition=IfCondition(use_deck),
                output="screen",
            ),
            Node(
                package="ares_perception",
                executable="traversability_node",
                parameters=[{"unit": "mast_cam", "use_sim_time": True}],
                output="screen",
            ),
            # Nothing else publishes /cmd_vel. Planners and operators publish
            # /cmd_vel_raw and this node clamps it, so the speed limit is
            # binding rather than advisory.
            Node(
                package="ares_perception",
                executable="speed_governor",
                parameters=[{"use_sim_time": True}],
                output="screen",
            ),
            Node(
                package="ares_perception",
                executable="segmentation_node",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(semantics),
                output="screen",
            ),
            Node(
                package="ares_perception",
                executable="detection_node",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(semantics),
                output="screen",
            ),
        ]
    )
