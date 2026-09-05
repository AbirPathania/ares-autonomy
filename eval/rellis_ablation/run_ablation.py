"""Run the ablation and write the report. Build spec phase 0.

    python -m eval.rellis_ablation.run_ablation --synthetic
    python -m eval.rellis_ablation.run_ablation --dataset /path/to/Rellis-3D

The synthetic path runs today, on any machine, with no dataset. It verifies
the pipeline and produces the confidence result on textureless ground that no
public dataset contains.

The dataset path is the one that produces the number to put in front of an
investor's engineer, and it needs RELLIS-3D. See README.md in this directory.

Both paths run the identical estimator on both depth sources. That is the
whole design: the only thing that changes between the two runs is where the
points came from.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ares_math.constants import REVISION
from ares_math.kinematics import stability_limits
from ares_perception.matcher import StereoMatcher, depth_to_points
from ares_perception.traversability import UNKNOWN, GridSpec, estimate

from . import synthetic as syn
from .metrics import Ablation, score_by_range, score_depth, score_traversability

REPORT_DIR = Path(__file__).resolve().parent / "report"


def run_synthetic_scene(name, confidence_floor=0.5, spec=None):
    """One scene, both sources, same estimator."""
    scene = syn.SCENES[name]
    spec = spec or GridSpec(x_min=0.5, x_max=12.0, y_min=-4.0, y_max=4.0)
    intrinsics = syn.render_intrinsics()
    pose = syn.camera_pose()

    left, right, truth_depth, _ = syn.stereo_pair(scene, intrinsics, pose)

    # Source A, LiDAR. Sparse, accurate, and already in the body frame.
    cloud = syn.lidar_sample(scene, pose)
    lidar_state, lidar_surface, _ = estimate(cloud, spec=spec)

    # Source B, stereo. Dense, cheap, and wrong in a specific way.
    matcher = StereoMatcher(intrinsics)
    depth, confidence, _ = matcher(left, right)
    points, point_confidence = depth_to_points(
        depth, intrinsics, confidence=confidence, camera_to_body=pose
    )
    stereo_state, stereo_surface, _ = estimate(
        points, spec=spec, confidence=point_confidence,
        confidence_floor=confidence_floor,
    )

    truth = syn.truth_grid(scene, spec)

    result = Ablation(
        scene=name,
        lidar=score_traversability(lidar_state, truth).as_dict(),
        stereo=score_traversability(stereo_state, truth).as_dict(),
        lidar_by_range=score_by_range(lidar_state, truth, spec),
        stereo_by_range=score_by_range(stereo_state, truth, spec),
        depth_bands=score_depth(depth, truth_depth, intrinsics, confidence=confidence),
        confidence={
            "mean": float(np.mean(confidence)),
            "fraction_trusted": float(np.mean(confidence >= confidence_floor)),
            "cells_unknown_stereo": float(np.mean(stereo_state == UNKNOWN)),
            "cells_unknown_lidar": float(np.mean(lidar_state == UNKNOWN)),
        },
    )
    return result, {
        "state_lidar": lidar_state,
        "state_stereo": stereo_state,
        "truth": truth,
        "left": left,
        "depth": depth,
        "confidence": confidence,
        "surfaces": (lidar_surface, stereo_surface),
    }


def run_synthetic(scenes=None, confidence_floor=0.5):
    names = scenes or list(syn.SCENES)
    results = []
    for name in names:
        started = time.time()
        result, _ = run_synthetic_scene(name, confidence_floor=confidence_floor)
        print(
            f"  {name:<16} accuracy lost {result.accuracy_lost:+.3f}   "
            f"false free space added {result.false_free_space_added:+.4f}   "
            f"({time.time() - started:.1f}s)"
        )
        results.append(result)
    return results


def run_dataset(root, limit=None, confidence_floor=0.5):
    """The real ablation, on RELLIS-3D."""
    from .rellis import Rellis3D

    dataset = Rellis3D(root)
    spec = GridSpec(x_min=0.5, x_max=20.0, y_min=-8.0, y_max=8.0)
    matcher = StereoMatcher(dataset.intrinsics)

    per_class = {}
    frames = dataset.frames()[:limit] if limit else dataset.frames()
    for i, frame in enumerate(frames):
        sample = dataset.load(frame)
        lidar_state, _, _ = estimate(sample.lidar_points, spec=spec)
        depth, confidence, _ = matcher(sample.left, sample.right)
        points, point_confidence = depth_to_points(
            depth,
            dataset.intrinsics,
            confidence=confidence,
            camera_to_body=dataset.camera_to_body,
        )
        stereo_state, _, _ = estimate(
            points, spec=spec, confidence=point_confidence,
            confidence_floor=confidence_floor,
        )
        truth, classes = dataset.truth_grid(sample, spec)

        for label in np.unique(classes):
            if label < 0:
                continue
            name = dataset.class_name(label)
            mask = classes == label
            bucket = per_class.setdefault(name, {"lidar": [], "stereo": []})
            bucket["lidar"].append(score_traversability(lidar_state, truth, mask))
            bucket["stereo"].append(score_traversability(stereo_state, truth, mask))

        if i % 25 == 0:
            print(f"  frame {i}/{len(frames)}")

    return _aggregate(per_class)


def _aggregate(per_class):
    from .metrics import Confusion

    out = []
    for name, buckets in sorted(per_class.items()):
        totals = {}
        for source in ("lidar", "stereo"):
            merged = Confusion()
            for confusion in buckets[source]:
                for field_name in vars(merged):
                    setattr(
                        merged,
                        field_name,
                        getattr(merged, field_name) + getattr(confusion, field_name),
                    )
            totals[source] = merged.as_dict()
        out.append(Ablation(scene=name, lidar=totals["lidar"], stereo=totals["stereo"]))
    return out


def _n(value, places=3):
    """Format a metric, or say plainly that the scene could not produce one."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{places}f}"


def _plain(value, places=3):
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def write_report(results, path=None, source="synthetic"):
    path = Path(path) if path else REPORT_DIR / f"ablation_{source}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    limits = stability_limits()

    lines = [
        f"# Phase 0, the LiDAR ablation, {source}",
        "",
        "What accuracy the traversability estimate loses when the LiDAR is",
        "removed and stereo depth is put in its place. Identical estimator,",
        "identical frames, identical vehicle limits. Only the depth source",
        "changes.",
        "",
        f"Hull revision {REVISION}. Slope limit "
        f"{limits['working']:.1f} degrees, derived from a centre of mass at "
        f"{limits['cog_height'] * 1000:.0f} mm and the track, divided by the "
        "same safety factor the stopping distance uses.",
        "",
        "## Headline",
        "",
        "| Scene | LiDAR accuracy | Stereo accuracy | Accuracy lost | "
        "False free space, LiDAR | stereo | added |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r.scene}` | {_plain(r.lidar['accuracy'])} | "
            f"{_plain(r.stereo['accuracy'])} | **{_n(r.accuracy_lost)}** | "
            f"{_plain(r.lidar['false_free_space_rate'], 4)} | "
            f"{_plain(r.stereo['false_free_space_rate'], 4)} | "
            f"**{_n(r.false_free_space_added, 4)}** |"
        )

    lines += [
        "",
        "Accuracy is computed over cells the estimator was willing to decide.",
        "False free space is a true obstacle called traversable, counted against",
        "every true obstacle cell including the undecided ones, because declining",
        "to decide is safe and being wrong is not. It is reported separately and",
        "never averaged into accuracy, which would hide it behind a large easy",
        "background.",
        "",
        "`n/a` means the scene contains no obstacle cells, so the metric has no",
        "denominator. On such a scene accuracy reduces to recall on traversable",
        "ground, and a stereo run that scores above the LiDAR run has not beaten",
        "it at anything that matters.",
        "",
    ]

    for r in results:
        if not (r.depth_bands or r.confidence or r.stereo_by_range):
            continue
        lines += [f"## `{r.scene}`", ""]
        if r.confidence:
            lines += [
                f"Mean stereo confidence {r.confidence['mean']:.3f}. "
                f"{r.confidence['fraction_trusted'] * 100:.1f} per cent of pixels "
                "cleared the trust threshold. "
                f"{r.confidence['cells_unknown_stereo'] * 100:.1f} per cent of grid "
                "cells came back unknown from stereo, against "
                f"{r.confidence['cells_unknown_lidar'] * 100:.1f} from LiDAR.",
                "",
            ]
        if r.depth_bands:
            lines += [
                "Measured stereo depth error against the section 03.2 prediction:",
                "",
                "| Range | Samples | Bias | Measured sigma | Predicted | Ratio |",
                "|---|---|---|---|---|---|",
            ]
            for b in r.depth_bands:
                lines.append(
                    f"| {b.low:.0f} to {b.high:.0f} m | {b.count} | {b.bias:+.4f} m | "
                    f"{b.sigma:.4f} m | {b.predicted_sigma:.4f} m | {b.ratio:.2f} |"
                )
            lines += [
                "",
                "A ratio near or below one means the matcher is achieving the "
                "subpixel disparity precision the tables assume. Above one means "
                "it is not, and the range at which the depth is trustworthy is "
                "shorter than section 03.2 claims.",
                "",
            ]
        if r.stereo_by_range:
            lines += [
                "Traversability by range, stereo against LiDAR:",
                "",
                "| Range | LiDAR accuracy | Stereo accuracy | Stereo unknown |",
                "|---|---|---|---|",
            ]
            for band, stereo in r.stereo_by_range.items():
                lidar = r.lidar_by_range.get(band, {})
                lines.append(
                    f"| {band} | {_plain(lidar.get('accuracy'))} | "
                    f"{_plain(stereo['accuracy'])} | "
                    f"{_plain(stereo['unknown_rate'])} |"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps([r.as_dict() for r in results], indent=2, default=float)
    )
    return path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true", help="run the test fixtures")
    group.add_argument("--dataset", type=Path, help="path to a RELLIS-3D root")
    parser.add_argument("--scene", action="append", help="restrict to named scenes")
    parser.add_argument("--limit", type=int, help="frames to use from the dataset")
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.5,
        help="below this a stereo point is not evidence",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        print("phase 0 ablation, synthetic fixtures")
        results = run_synthetic(args.scene, confidence_floor=args.confidence_floor)
        source = "synthetic"
    else:
        print(f"phase 0 ablation, RELLIS-3D at {args.dataset}")
        results = run_dataset(
            args.dataset, limit=args.limit, confidence_floor=args.confidence_floor
        )
        source = "rellis3d"

    report, data = write_report(results, source=source)
    print(f"\nwrote {report}")
    print(f"wrote {data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
