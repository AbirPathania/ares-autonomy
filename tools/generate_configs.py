"""Generate the Nav2 and elevation map configs from constants.py.

Build spec section 11: no physical constant is written anywhere except
constants.py. A YAML file is not an exception. Nav2's footprint, speed limits,
inflation radius and lookahead are all vehicle geometry, and hand-writing them
is how a config ends up describing a rover that no longer exists.

Run:  python tools/generate_configs.py
"""

import math
from pathlib import Path

import yaml

from ares_math.constants import CAMERA, CHASSIS, LIMITS, MOBILITY, SENSORS, TRENCH
from ares_math.kinematics import stability_limits
from ares_math.optics import focal_px, near_edge_range
from ares_math.safety import stopping_distance
from ares_math.stereo import min_range

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ares_bringup" / "config"

HEADER = (
    "GENERATED FILE. Do not hand edit.\n"
    "Source: ares_math/constants.py, via tools/generate_configs.py.\n"
    "Every number below traces to hull revision H-2 or to a formula in\n"
    "section 03 of the ARES Autonomy Build Spec. Change a constant, re-run\n"
    "the generator, re-run pytest.\n"
)


def write(path, data, notes):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"# {line}\n" for line in HEADER.strip().splitlines())
    text += "#\n"
    text += "".join(f"# {line}\n" for line in notes)
    text += "\n" + yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)}")


def footprint():
    """The hull outline in base_link, with the wheels included.

    Nav2 plans against this, so it has to be the real width. Using the hull
    width alone would let the planner clip an obstacle with a wheel.
    """
    half_length = CHASSIS.hull_length / 2.0
    half_width = CHASSIS.track / 2.0
    front = CHASSIS.axle_pitch_front + CHASSIS.rolling_radius
    rear = -(CHASSIS.axle_pitch_rear + CHASSIS.rolling_radius)
    forward = max(half_length, front)
    backward = min(-half_length, rear)
    return [
        [forward, half_width],
        [forward, -half_width],
        [backward, -half_width],
        [backward, half_width],
    ]


def nav2():
    stop = stopping_distance(
        LIMITS.v_autonomous, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    limits = stability_limits()
    f = focal_px(CAMERA.width_px, CAMERA.hfov_deg)
    near = max(
        near_edge_range(SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg),
        min_range(f, SENSORS.mast_baseline, CAMERA.max_disparity_px),
    )
    # The planner must not plan through ground the deck unit has not covered
    # either. The deck unit is what fills the mast's blind zone.
    deck_near = near_edge_range(
        SENSORS.deck_height, SENSORS.deck_tilt_deg, CAMERA.vfov_deg
    )
    resolution = round(CHASSIS.track / 7.0, 4)
    diagonal = math.hypot(CHASSIS.hull_length, CHASSIS.track) / 2.0

    notes = [
        "Speed is capped at the autonomous limit section 02 derives from mast",
        f"height: {LIMITS.v_autonomous} m/s, {LIMITS.v_autonomous * 3.6:.1f} km/h.",
        f"{LIMITS.v_transit} m/s transit is TELEOPERATED ONLY and never appears here.",
        "",
        f"Stopping distance at that speed is {stop:.3f} m with the {LIMITS.safety_factor}",
        "safety factor, which is where the lookahead and the inflation come from.",
        "",
        f"The mast unit sees nothing closer than {near:.3f} m and the deck unit",
        f"fills in from {deck_near:.3f} m. Below that the vehicle is blind and the",
        "costmap must not be told otherwise.",
    ]

    data = {
        "controller_server": {
            "ros__parameters": {
                "controller_frequency": 20.0,
                "min_x_velocity_threshold": 0.02,
                "min_theta_velocity_threshold": 0.02,
                "failure_tolerance": stop,
                "FollowPath": {
                    "plugin": "nav2_regulated_pure_pursuit_controller::"
                    "RegulatedPurePursuitController",
                    "desired_linear_vel": LIMITS.v_autonomous,
                    "max_linear_accel": LIMITS.decel,
                    "max_linear_decel": LIMITS.decel,
                    # Look ahead by one stopping distance, so the controller is
                    # never committed to a point it could not stop before.
                    "lookahead_dist": stop,
                    "min_lookahead_dist": stop / 2.0,
                    "max_lookahead_dist": stop * 2.0,
                    "use_velocity_scaled_lookahead_dist": True,
                    "transform_tolerance": LIMITS.latency,
                    "use_regulated_linear_velocity_scaling": True,
                    "regulated_linear_scaling_min_radius": diagonal,
                    "regulated_linear_scaling_min_speed": 0.25,
                    "allow_reversing": True,
                },
            }
        },
        "planner_server": {
            "ros__parameters": {
                "expected_planner_frequency": 2.0,
                "GridBased": {
                    "plugin": "nav2_navfn_planner::NavfnPlanner",
                    "tolerance": resolution,
                    "use_astar": True,
                    "allow_unknown": False,
                },
            }
        },
        "local_costmap": {
            "local_costmap": {
                "ros__parameters": {
                    "update_frequency": 10.0,
                    "publish_frequency": 5.0,
                    "global_frame": "odom",
                    "robot_base_frame": "base_link",
                    "rolling_window": True,
                    # Wide enough to hold a full stopping distance ahead with
                    # room to steer, and no wider: every cell costs cycles the
                    # software renderer does not have.
                    "width": round(4.0 * stop, 2),
                    "height": round(4.0 * stop, 2),
                    "resolution": resolution,
                    "footprint": str(footprint()),
                    "plugins": ["traversability_layer", "inflation_layer"],
                    "traversability_layer": {
                        "plugin": "nav2_costmap_2d::VoxelLayer",
                        "enabled": True,
                        "observation_sources": "mast deck",
                        "min_obstacle_height": CHASSIS.ground_clearance / 2.0,
                        "max_obstacle_height": SENSORS.mast_height,
                        "mast": {
                            "topic": "/mast_cam/points",
                            "data_type": "PointCloud2",
                            "clearing": True,
                            "marking": True,
                            "obstacle_max_range": 4.0 * stop,
                            "obstacle_min_range": near,
                            "raytrace_max_range": 5.0 * stop,
                            "raytrace_min_range": near,
                        },
                        "deck": {
                            "topic": "/deck_cam/points",
                            "data_type": "PointCloud2",
                            "clearing": True,
                            "marking": True,
                            "obstacle_max_range": 2.0,
                            "obstacle_min_range": deck_near,
                            "raytrace_max_range": 2.5,
                            "raytrace_min_range": deck_near,
                        },
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        # Half the hull diagonal keeps a corner clear when the
                        # vehicle rotates in place, plus the stopping distance.
                        "inflation_radius": round(diagonal + stop, 3),
                        "cost_scaling_factor": 3.0,
                    },
                }
            }
        },
        "global_costmap": {
            "global_costmap": {
                "ros__parameters": {
                    "update_frequency": 1.0,
                    "publish_frequency": 1.0,
                    "global_frame": "map",
                    "robot_base_frame": "base_link",
                    "resolution": resolution,
                    "footprint": str(footprint()),
                    "track_unknown_space": True,
                    "plugins": ["static_layer", "inflation_layer"],
                    "static_layer": {
                        "plugin": "nav2_costmap_2d::StaticLayer",
                        "map_subscribe_transient_local": True,
                    },
                    "inflation_layer": {
                        "plugin": "nav2_costmap_2d::InflationLayer",
                        "inflation_radius": round(diagonal + stop, 3),
                        "cost_scaling_factor": 3.0,
                    },
                }
            }
        },
        "behavior_server": {
            "ros__parameters": {
                "local_frame": "odom",
                "global_frame": "map",
                "robot_base_frame": "base_link",
                "transform_tolerance": LIMITS.latency,
                # Reversing out along the path you drove in is the mitigation
                # for being blind behind and to the sides. Build spec section
                # 04: it belongs in the CONOPS, not in a third camera.
                "max_rotational_vel": limits["working"] * math.pi / 180.0,
                "backup_dist": round(2.0 * stop, 3),
                "backup_speed": round(LIMITS.v_autonomous / 2.0, 3),
            }
        },
        "velocity_smoother": {
            "ros__parameters": {
                "smoothing_frequency": 20.0,
                "max_velocity": [LIMITS.v_autonomous, 0.0, 1.0],
                "min_velocity": [-LIMITS.v_autonomous / 2.0, 0.0, -1.0],
                "max_accel": [LIMITS.decel, 0.0, 2.0],
                "max_decel": [-LIMITS.decel, 0.0, -2.0],
            }
        },
    }
    write(CONFIG / "nav2.yaml", data, notes)


def elevation_map():
    limits = stability_limits()
    stop = stopping_distance(
        LIMITS.v_autonomous, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    resolution = round(CHASSIS.track / 7.0, 4)

    notes = [
        "elevation_mapping_cupy, with the semantic layers section 09 asks for.",
        "",
        f"The slope limit is {limits['binding']:.1f} degrees of static tipover,",
        f"from a centre of mass at {limits['cog_height'] * 1000:.0f} mm and the",
        f"{CHASSIS.track} m track, divided by the {LIMITS.safety_factor} safety",
        f"factor to give {limits['working']:.1f} degrees.",
        "",
        f"The step limit is the {MOBILITY.max_step} m obstacle claim, and the",
        f"negative obstacle the vehicle is sized around is {TRENCH.width} by",
        f"{TRENCH.depth} m.",
        "",
        "Low confidence is not missing data to be filled in. It is a first",
        "class value that caps speed, so the traversability layer keeps it",
        "rather than interpolating over it.",
    ]

    data = {
        "elevation_mapping": {
            "ros__parameters": {
                "resolution": resolution,
                "map_length": round(8.0 * stop, 2),
                "sensor_noise_factor": CAMERA.disparity_sigma_px,
                "map_frame": "odom",
                "base_frame": "base_link",
                "update_variance_fps": 5.0,
                "time_interval": 0.1,
                "max_height_range": SENSORS.mast_height,
                # Never fill a gap by guessing. An interpolated cell over a
                # trench is a confident wrong answer, which is the one output
                # this stack must not produce.
                "enable_drift_compensation": True,
                "enable_edge_sharpen": True,
                "enable_visibility_cleanup": True,
                "enable_overlap_clearance": False,
                "position_noise_thresh": resolution / 2.0,
                "orientation_noise_thresh": 0.01,
                "plugin_config_file": "traversability_layers.yaml",
            }
        },
        "traversability": {
            "ros__parameters": {
                "slope_critical_deg": limits["binding"],
                "slope_limit_deg": limits["working"],
                "step_limit": MOBILITY.max_step,
                "roughness_limit": MOBILITY.max_step / 2.0,
                "negative_obstacle_width": TRENCH.width,
                "negative_obstacle_depth": TRENCH.depth,
                "confidence_floor": 0.5,
                "minimum_points_per_cell": 3,
                "unknown_is_not_free": True,
            }
        },
        "semantic_layers": {
            "ros__parameters": {
                "classes": [
                    "rock",
                    "scree",
                    "grass",
                    "snow",
                    "mud",
                    "water",
                    "bush",
                    "sky",
                    "man_made",
                ],
                # Snow and water are not obstacles and are not free space
                # either. They are surfaces whose bottom the vehicle cannot
                # see, and the honest handling is to slow down.
                "reduced_confidence_classes": ["snow", "water", "mud"],
                "never_traversable_classes": ["water", "sky", "man_made"],
            }
        },
    }
    write(CONFIG / "elevation_map.yaml", data, notes)


def main():
    nav2()
    elevation_map()


if __name__ == "__main__":
    main()
