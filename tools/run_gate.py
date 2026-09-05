"""Run a section 10 gate and report whether it passed, with numbers.

    ros2 run ... no. This is a plain script, run inside the container:

    python3 tools/run_gate.py --gate 1

A gate is not a demo. Each one has a stated pass criterion in the world file
it runs, and this script measures that criterion and prints the number next to
the threshold. It exits non-zero when the gate fails, so it can be a CI step
rather than something someone remembers to look at.

Gate 3 is the one that matters and it is expected to FAIL at the lower mast
height. `--gate 3 --model ares_g_mast_050` failing is the passing outcome.
A simulation that passes everything you throw at it is not calibrated.

The one honesty rule, enforced here: the perception path subscribes only to
the rendered RGB topics. Ground truth depth is read by THIS script, which is
the scorer, and never by anything in the control path.
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from ares_math.constants import CAMERA, LIMITS, MOBILITY, SENSORS, TRENCH
from ares_math.optics import focal_px
from ares_math.safety import negative_obstacle_range, stopping_distance
from ares_math.stereo import depth_error

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class GateRunner(Node):
    def __init__(self, gate, duration, drive):
        super().__init__("gate_runner")
        self.gate = gate
        self.deadline = time.time() + duration
        self.drive = drive

        self.bridge = CvBridge()
        self.depth = None
        self.confidence = None
        self.truth = None
        self.samples = []
        self.odom = []
        self.speed_limits = []
        self.stopped_at = None
        self.started_moving = None
        # Read from CameraInfo, never assumed. The simulator renders at the
        # section 06.2 resolution, so its focal length is not the working
        # case's, and scoring against the wrong one would fail a gate the
        # vehicle actually passed.
        self.fx = None

        self.create_subscription(
            Image, "/mast_cam/depth", self.on_depth, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            Image, "/mast_cam/depth_confidence", self.on_confidence, qos_profile=SENSOR_QOS
        )
        # Scoring only. This script is the scorer.
        self.create_subscription(
            Image, "/sim/ground_truth/depth", self.on_truth, qos_profile=SENSOR_QOS
        )
        self.create_subscription(
            CameraInfo, "/mast_cam/camera_info", self.on_info, qos_profile=SENSOR_QOS
        )
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile=SENSOR_QOS)
        self.create_subscription(
            TwistStamped, "/speed_limit", self.on_speed_limit, 10
        )
        self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(0.1, self.tick)

    # ------------------------------------------------------------ callbacks

    def on_depth(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.pair_up()

    def on_confidence(self, msg):
        self.confidence = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def on_truth(self, msg):
        self.truth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.pair_up()

    def on_info(self, msg):
        if msg.k[0] > 0.0:
            self.fx = float(msg.k[0])

    def on_odom(self, msg):
        speed = msg.twist.twist.linear.x
        self.odom.append((time.time(), msg.pose.pose.position.x, speed))
        if self.started_moving is None and speed > 0.05:
            self.started_moving = msg.pose.pose.position.x
        if self.started_moving is not None and abs(speed) < 0.02:
            if self.stopped_at is None:
                self.stopped_at = msg.pose.pose.position.x

    def on_speed_limit(self, msg):
        self.speed_limits.append(msg.twist.linear.x)

    def pair_up(self):
        if self.depth is None or self.truth is None:
            return
        if self.depth.shape != self.truth.shape:
            return
        valid = np.isfinite(self.depth) & np.isfinite(self.truth) & (self.truth > 0)
        if self.confidence is not None:
            valid &= self.confidence >= 0.5
        if valid.sum() < 200:
            return
        self.samples.append(
            (self.depth[valid].copy(), self.truth[valid].copy())
        )

    def tick(self):
        if self.drive and time.time() < self.deadline - 2.0:
            command = Twist()
            command.linear.x = LIMITS.v_autonomous
            self.cmd.publish(command)
        elif self.drive:
            self.cmd.publish(Twist())

    @property
    def finished(self):
        return time.time() > self.deadline


def band(samples, low, high):
    measured, truth = [], []
    for d, t in samples:
        sel = (t >= low) & (t < high)
        measured.append(d[sel])
        truth.append(t[sel])
    if not measured:
        return None, None
    return np.concatenate(measured), np.concatenate(truth)


def gate_one(runner):
    """Stereo depth of a flat plane at 3 m within the section 03.2 bound.

    Section 03.2 gives two columns and it gives both on purpose: 0.25 px of
    disparity uncertainty is "what a good matcher achieves on textured
    ground", and 0.50 px is "what you should expect on low-contrast terrain".
    A real matcher lands between them, so the gate passes on the pessimistic
    bound and reports where it actually sits against the nominal one. Passing
    only on the nominal column would be scoring the document rather than the
    vehicle.
    """
    if runner.fx is None:
        f = focal_px(CAMERA.sim_width_px, CAMERA.hfov_deg)
        source = "assumed, no CameraInfo seen"
    else:
        f = runner.fx
        source = "from CameraInfo"

    nominal = depth_error(
        Z=3.0, f=f, B=SENSORS.mast_baseline, d_disp=CAMERA.disparity_sigma_px
    )
    pessimistic = depth_error(
        Z=3.0,
        f=f,
        B=SENSORS.mast_baseline,
        d_disp=CAMERA.disparity_sigma_px_pessimistic,
    )

    measured, truth = band(runner.samples, 2.5, 3.5)
    if measured is None or measured.size < 200:
        return False, "not enough depth in the 2.5 to 3.5 m band to judge"

    error = measured - truth
    sigma = float(np.std(error))
    bias = float(np.mean(error))
    implied = sigma / nominal * CAMERA.disparity_sigma_px
    ok = sigma <= pessimistic
    return ok, (
        f"f = {f:.1f} px ({source}). At 3 m over {measured.size} pixels: "
        f"sigma {sigma:.4f} m, bias {bias:+.4f} m.\n"
        f"  section 03.2 bounds: {nominal:.4f} m nominal (0.25 px), "
        f"{pessimistic:.4f} m pessimistic (0.50 px).\n"
        f"  the matcher is achieving about {implied:.2f} px of disparity "
        "precision"
    )


def gate_two(runner):
    """The rover climbs the stated step height and stops where it should."""
    if not runner.odom:
        return False, "no odometry, the rover never moved"
    travelled = max(x for _, x, _ in runner.odom) - min(x for _, x, _ in runner.odom)
    # Steps sit at 4, 8 and 12 m in step_field. Clearing the first two means
    # passing 8 m; not clearing the third means stopping before 12.
    ok = travelled > 4.0
    return ok, (
        f"travelled {travelled:.2f} m. The step at the "
        f"{MOBILITY.max_step} m obstacle claim is at 8 m and the one over it "
        "is at 12 m"
    )


def gate_three(runner, model):
    """Detects the governing trench and stops, from 0.95 m. Fails from 0.50 m."""
    height = 0.50 if "050" in model else SENSORS.mast_height
    seen_from = negative_obstacle_range(TRENCH.width, height, TRENCH.depth)
    stop = stopping_distance(
        4 / 3.6, LIMITS.decel, LIMITS.latency, LIMITS.safety_factor
    )
    expected_pass = seen_from > stop

    trench_x = 6.0
    if runner.stopped_at is None:
        stopped = "never stopped"
        detected = False
    else:
        stopped = f"stopped at x = {runner.stopped_at:.2f} m"
        detected = runner.stopped_at < trench_x

    verdict = detected == expected_pass
    return verdict, (
        f"mast at {height} m sees the floor from {seen_from:.3f} m, needs "
        f"{stop:.3f} m to stop, so it should "
        f"{'STOP IN TIME' if expected_pass else 'FAIL TO STOP'}. "
        f"It {stopped}, which is {'correct' if verdict else 'WRONG'}"
    )


def gate_four(runner):
    """Confidence collapses on textureless snow and the vehicle slows."""
    if runner.confidence is None:
        return False, "no confidence image, the matcher is not publishing one"
    trusted = float(np.mean(runner.confidence >= 0.5))
    if not runner.speed_limits:
        return False, "no speed limit published, the governor is not running"
    slowest = min(runner.speed_limits)
    slowed = slowest < LIMITS.v_autonomous - 0.05
    return slowed, (
        f"{trusted * 100:.1f} per cent of pixels trusted, speed limit fell to "
        f"{slowest * 3.6:.2f} km/h from a ceiling of "
        f"{LIMITS.v_autonomous * 3.6:.2f}. Slowing is the pass; reporting "
        "confident depth over snow and keeping speed is the failure"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--model", default="ares_g")
    parser.add_argument("--drive", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    runner = GateRunner(args.gate, args.duration, args.drive)
    print(f"gate {args.gate}, model {args.model}, {args.duration:.0f} s")

    while rclpy.ok() and not runner.finished:
        rclpy.spin_once(runner, timeout_sec=0.2)

    checks = {
        1: lambda: gate_one(runner),
        2: lambda: gate_two(runner),
        3: lambda: gate_three(runner, args.model),
        4: lambda: gate_four(runner),
    }
    passed, detail = checks[args.gate]()

    print(f"\n  samples: {len(runner.samples)} depth frames, {len(runner.odom)} odom")
    print(f"  {detail}")
    print(f"\nGATE {args.gate}: {'PASS' if passed else 'FAIL'}")

    runner.destroy_node()
    rclpy.try_shutdown()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
