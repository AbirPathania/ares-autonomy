"""The phase 0 pipeline, verified on synthetic geometry.

RELLIS-3D is not in this repository, so these run against a scene whose
geometry is known exactly. That is enough to establish that the estimator,
the matcher and the scoring are correct. It is not enough to establish the
result, which needs real off-road data and is the point of the phase.
"""

import numpy as np
import pytest

from ares_math.constants import MOBILITY, TRENCH
from ares_math.kinematics import stability_limits
from ares_perception.matcher import StereoMatcher, depth_to_points
from ares_perception.traversability import (
    OBSTACLE,
    TRAVERSABLE,
    UNKNOWN,
    GridSpec,
    estimate,
    rasterise,
)
from eval.rellis_ablation import synthetic as syn
from eval.rellis_ablation.metrics import score_depth, score_traversability

# Small enough to keep the suite quick, large enough for stereo to be real.
SMALL = 240, 136


@pytest.fixture(scope="module")
def intrinsics():
    return syn.render_intrinsics(width=SMALL[0], height=SMALL[1])


@pytest.fixture(scope="module")
def flat(intrinsics):
    scene = syn.SCENES["flat_textured"]
    pose = syn.camera_pose()
    left, right, truth, points = syn.stereo_pair(scene, intrinsics, pose)
    return scene, pose, left, right, truth, points


# --------------------------------------------------------------------------
# The renderer produces a real stereo problem, not a picture of one.
# --------------------------------------------------------------------------


def test_the_two_eyes_are_not_the_same_image(flat):
    _, _, left, right, _, _ = flat
    assert left.shape == right.shape
    assert not np.array_equal(left, right), "a stereo pair with zero disparity"


def test_the_rendered_geometry_matches_the_height_field(flat):
    scene, pose, _, _, _, points = flat
    finite = np.isfinite(points).all(axis=2)
    sampled = points[finite]
    ground = scene.height(sampled[:, 0], sampled[:, 1])
    assert np.abs(sampled[:, 2] - ground) == pytest.approx(
        np.zeros(len(sampled)), abs=1e-3
    )


def test_the_camera_sits_where_the_urdf_says_it_does(flat):
    _, pose, _, _, _, _ = flat
    _, translation = pose
    from ares_math.constants import SENSORS

    assert translation[2] == pytest.approx(SENSORS.mast_height, abs=1e-12)


def test_the_near_edge_of_the_render_matches_the_blind_zone(flat):
    """The closest ground the camera sees is the section 03.4 number."""
    from ares_math.constants import CAMERA, SENSORS
    from ares_math.optics import near_edge_range

    _, _, _, _, _, points = flat
    finite = np.isfinite(points).all(axis=2)
    nearest = float(np.min(points[finite][:, 0]))
    expected = near_edge_range(
        SENSORS.mast_height, SENSORS.mast_tilt_deg, CAMERA.vfov_deg
    )
    # The render is 240 by 136, so its vertical field of view is narrower than
    # the 59 degrees section 03.4 assumes. The blind zone is therefore larger,
    # never smaller, and that inequality is the real assertion.
    assert nearest >= expected - 0.05


# --------------------------------------------------------------------------
# The matcher. Real depth, and error that tracks the theory.
# --------------------------------------------------------------------------


def test_the_matcher_recovers_depth_on_textured_ground(flat, intrinsics):
    _, _, left, right, truth, _ = flat
    depth, confidence, _ = StereoMatcher(intrinsics)(left, right)
    valid = np.isfinite(depth) & np.isfinite(truth) & (confidence > 0.3)
    assert valid.sum() > 500, "the matcher found almost nothing on textured ground"
    error = np.abs(depth[valid] - truth[valid])
    assert np.median(error) < 0.10


def test_measured_depth_error_tracks_the_section_03_curve(flat, intrinsics):
    """Phase 4's gate, brought forward. If this fails, no amount of downstream
    tuning will help, because the geometry or the calibration is wrong."""
    _, _, left, right, truth, _ = flat
    depth, confidence, _ = StereoMatcher(intrinsics)(left, right)
    bands = score_depth(depth, truth, intrinsics, confidence=confidence)
    assert len(bands) >= 2, "not enough range coverage to check the curve"
    for band in bands:
        assert band.ratio < 3.0, (
            f"{band.low} to {band.high} m: measured sigma {band.sigma:.3f} is "
            f"{band.ratio:.1f} times the predicted {band.predicted_sigma:.3f}"
        )
    # Error must grow with range. That is the shape of the curve, and it is
    # the property that governs baseline selection.
    sigmas = [b.sigma for b in bands]
    assert sigmas[-1] > sigmas[0]


def test_confidence_collapses_on_textureless_ground(intrinsics):
    """The single most important behaviour in the whole stack.

    Fresh snow has no texture for passive stereo to match. The mitigation is
    not a better matcher, it is a matcher that says when it does not know.
    """
    scene = syn.SCENES["snow_scree"]
    pose = syn.camera_pose()
    left, right, _, points = syn.stereo_pair(scene, intrinsics, pose)
    _, confidence, _ = StereoMatcher(intrinsics)(left, right)

    world_x = points[..., 0]
    scree = np.isfinite(world_x) & (world_x < 6.0)
    snow = np.isfinite(world_x) & (world_x >= 6.0)
    assert scree.sum() > 100 and snow.sum() > 100

    trusted_on_scree = float(np.mean(confidence[scree] > 0.5))
    trusted_on_snow = float(np.mean(confidence[snow] > 0.5))
    assert trusted_on_scree > 4 * trusted_on_snow, (
        f"confidence did not separate texture from its absence: "
        f"{trusted_on_scree:.3f} against {trusted_on_snow:.3f}"
    )
    assert trusted_on_snow < 0.25, "textureless snow was trusted"


# --------------------------------------------------------------------------
# The estimator, and its three states.
# --------------------------------------------------------------------------


def test_an_unobserved_cell_is_unknown_and_never_free_space():
    """The failure mode that drives a rover into a snowdrift at full speed."""
    spec = GridSpec(x_min=0.0, x_max=4.0, y_min=-2.0, y_max=2.0)
    points = np.array([[1.0, 0.0, 0.0]])
    state, _, _ = estimate(points, spec=spec)
    assert np.all(state[state != TRAVERSABLE] != TRAVERSABLE)
    assert np.mean(state == UNKNOWN) > 0.9
    assert not np.any(state[3:, :] == TRAVERSABLE)


def test_low_confidence_points_do_not_become_free_space():
    spec = GridSpec(x_min=0.0, x_max=2.0, y_min=-1.0, y_max=1.0)
    rng = np.random.default_rng(0)
    n = 40000
    points = np.column_stack(
        [rng.uniform(0.05, 1.95, n), rng.uniform(-0.95, 0.95, n), np.zeros(n)]
    )
    trusted, _, _ = estimate(points, spec=spec, confidence=np.ones(n))
    doubted, _, _ = estimate(points, spec=spec, confidence=np.full(n, 0.1))
    assert np.mean(trusted == TRAVERSABLE) > 0.8
    assert np.mean(doubted == TRAVERSABLE) == 0.0
    assert np.mean(doubted == UNKNOWN) == 1.0


def test_flat_ground_is_traversable():
    spec = GridSpec(x_min=0.0, x_max=4.0, y_min=-2.0, y_max=2.0)
    rng = np.random.default_rng(1)
    points = np.column_stack(
        [rng.uniform(0, 4, 40000), rng.uniform(-2, 2, 40000), np.zeros(40000)]
    )
    state, _, _ = estimate(points, spec=spec)
    assert np.mean(state == TRAVERSABLE) > 0.9


def test_a_step_over_the_obstacle_claim_is_an_obstacle():
    spec = GridSpec(x_min=0.0, x_max=4.0, y_min=-2.0, y_max=2.0)
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 4, 60000)
    y = rng.uniform(-2, 2, 60000)
    z = np.where(x > 2.0, MOBILITY.max_step * 3.0, 0.0)
    state, _, _ = estimate(np.column_stack([x, y, z]), spec=spec)
    ix = int(2.0 / spec.cell)
    assert np.any(state[ix - 1 : ix + 2, :] == OBSTACLE), "the wall was not seen"


def test_a_step_under_the_obstacle_claim_is_traversable():
    """A rocker-bogie climbs 245 mm. Calling that an obstacle wastes the linkage."""
    spec = GridSpec(x_min=0.0, x_max=4.0, y_min=-2.0, y_max=2.0)
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 4, 60000)
    y = rng.uniform(-2, 2, 60000)
    z = np.where(x > 2.0, MOBILITY.max_step * 0.5, 0.0)
    state, _, _ = estimate(np.column_stack([x, y, z]), spec=spec)
    assert np.mean(state == OBSTACLE) < 0.10


def test_a_slope_past_the_tipover_limit_is_an_obstacle():
    limits = stability_limits()
    spec = GridSpec(x_min=0.0, x_max=6.0, y_min=-2.0, y_max=2.0)
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 6, 80000)
    y = rng.uniform(-2, 2, 80000)
    steep = np.tan(np.radians(limits["working"] * 2.0))
    z = np.where(x > 3.0, (x - 3.0) * steep, 0.0)
    state, _, _ = estimate(np.column_stack([x, y, z]), spec=spec)
    far = int(4.0 / spec.cell)
    assert np.mean(state[far:, :] == OBSTACLE) > 0.5


def test_the_estimator_reads_its_limits_from_ares_math():
    """No traversability threshold is a literal in this package."""
    import ast
    import pathlib

    from ares_perception import traversability

    source = pathlib.Path(traversability.__file__).read_text()
    tree = ast.parse(source)
    # GridSpec's defaults are the extent of the map, not a claim about the
    # vehicle, so they are exempt. Everything else must come from ares_math.
    grid_lines = set()
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "GridSpec":
            grid_lines = set(range(cls.lineno, cls.end_lineno + 1))
    allowed = {0.0, 1.0, 2.0, 0.5, 7.0}
    offenders = [
        f"line {node.lineno}: {node.value}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value not in allowed
        and node.lineno not in grid_lines
    ]
    assert not offenders, f"physical thresholds hard coded: {offenders}"


# --------------------------------------------------------------------------
# Scoring.
# --------------------------------------------------------------------------


def test_false_free_space_counts_undecided_obstacles_against_the_estimator():
    truth = np.array([[False, False], [True, True]])
    state = np.array([[TRAVERSABLE, UNKNOWN], [TRAVERSABLE, TRAVERSABLE]], dtype=np.uint8)
    score = score_traversability(state, truth)
    # One of two obstacle cells was called traversable, the other unknown.
    assert score.false_free_space_rate == pytest.approx(0.5)
    # Recall only counts the decided one, so it reads worse, and both are
    # reported because they answer different questions.
    assert score.obstacle_recall == pytest.approx(0.0)


def test_declining_to_decide_is_not_scored_as_an_error():
    truth = np.array([[True, True]])
    state = np.array([[TRAVERSABLE, UNKNOWN]], dtype=np.uint8)
    score = score_traversability(state, truth)
    assert score.accuracy == pytest.approx(1.0)
    assert score.unknown_rate == pytest.approx(0.5)


def test_a_shape_mismatch_is_loud():
    with pytest.raises(ValueError, match="shape mismatch"):
        score_traversability(np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 3), bool))


# --------------------------------------------------------------------------
# The ablation itself. Same estimator, both sources.
# --------------------------------------------------------------------------


def test_both_sources_go_through_the_same_estimator(flat, intrinsics):
    """The property that makes this an ablation rather than a comparison."""
    scene, pose, left, right, _, _ = flat
    spec = GridSpec(x_min=1.0, x_max=8.0, y_min=-3.0, y_max=3.0)

    cloud = syn.lidar_sample(scene, pose, rings=32, azimuths=256)
    lidar_state, _, lidar_diag = estimate(cloud, spec=spec)

    depth, confidence, _ = StereoMatcher(intrinsics)(left, right)
    points, point_confidence = depth_to_points(
        depth, intrinsics, confidence=confidence, camera_to_body=pose
    )
    stereo_state, _, stereo_diag = estimate(
        points, spec=spec, confidence=point_confidence
    )

    assert lidar_state.shape == stereo_state.shape
    assert lidar_diag["slope_limit_deg"] == stereo_diag["slope_limit_deg"]
    assert lidar_diag["step_limit_m"] == stereo_diag["step_limit_m"]
    truth = syn.truth_grid(scene, spec)
    for state in (lidar_state, stereo_state):
        assert score_traversability(state, truth).accuracy > 0.8


def test_the_optical_transform_must_be_supplied(intrinsics):
    """Guessing it is the bug that costs everyone two days, so it raises."""
    depth = np.full((8, 8), 3.0)
    with pytest.raises(ValueError, match="camera_to_body is required"):
        depth_to_points(depth, intrinsics, camera_to_body=None)


def test_the_governing_trench_is_the_one_from_section_02():
    scene = syn.SCENES["trench_field"]
    step = 0.001
    xs = np.arange(2.0, 5.0, step)
    ys = np.zeros_like(xs)
    z = scene.height(xs, ys)
    below = xs[z < 0]
    # Sampled at 1 mm with strict edge inequalities, so the measured extent
    # is within one sample of the true width.
    assert below.max() - below.min() == pytest.approx(TRENCH.width, abs=2 * step)
    assert z.min() == pytest.approx(-TRENCH.depth, abs=1e-12)


def test_rasterise_marks_thinly_sampled_cells_unobserved():
    spec = GridSpec(x_min=0.0, x_max=1.0, y_min=-0.5, y_max=0.5)
    surface = rasterise(np.array([[0.5, 0.0, 0.0], [0.51, 0.0, 0.0]]), spec=spec)
    assert surface.count.sum() == 2
    assert not surface.observed.any(), "two returns is not a measured cell"
