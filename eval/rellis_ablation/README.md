# Phase 0, the LiDAR ablation

**Build tier 0 first. It answers the actual question, it needs nothing you do
not have, and it produces a number you can put in front of an investor's
engineer.**

Run the perception stack on RELLIS-3D, which carries synchronised stereo and
LiDAR from the same platform. Measure exactly what dropping the LiDAR costs,
on real off-road data, with no simulator in the loop.

## The design, in one sentence

Both depth sources go through the identical traversability estimator, on the
identical frames, with the identical vehicle limits, so the difference between
them is the sensor and nothing else.

```
LiDAR points   ─┐
                ├─►  rasterise  ─►  slope, step, roughness  ─►  three states
stereo points  ─┘                        (ares_math)
```

If the estimator differed between the two runs, this would be a comparison of
two systems and it would prove nothing. `test_both_sources_go_through_the_same_estimator`
holds that property.

## The three states, and why the third one is the point

    TRAVERSABLE   measured, and within the vehicle's limits
    OBSTACLE      measured, and outside them
    UNKNOWN       not measured well enough to say

A stack that folds UNKNOWN into TRAVERSABLE will drive into a snowdrift at full
speed, and it will do it in the field rather than in the sim. So UNKNOWN is a
first class output, it is scored separately, and downstream it caps speed
through `ares_math.safety.max_speed_for_range` exactly as an obstacle does.

## The metric that matters

Not accuracy. **False free space**: a cell that is really an obstacle, called
traversable. Accuracy averages that away against a large easy background. It is
reported on its own, it is never averaged, and it counts undecided obstacle
cells in its denominator rather than its numerator, because declining to decide
is safe and being wrong is not.

## Running it today, with no dataset

```bash
python -m eval.rellis_ablation.run_ablation --synthetic
```

Renders stereo pairs by ray casting a height field, with texture that is a
function of world position rather than pixel position, so the same ground point
looks the same from both eyes and a matcher has something true to find. Four
scenes, matching the four Gazebo worlds in section 10: flat, a step field, the
governing 0.5 by 0.3 m trench, and textureless snow over scree.

This verifies the pipeline. It is not the result.

### What the synthetic run already shows

Measured stereo depth error tracks the section 03.2 curve on textured ground,
at 0.7 to 1.0 times the predicted sigma across 1 to 10 m. The matcher is
achieving the subpixel disparity precision the tables assume.

On textureless snow the same matcher runs at 1.6 to 2.4 times the predicted
sigma, and confidence collapses: 60 per cent of pixels clear the trust
threshold on textured scree against 5 per cent on snow, with median depth error
21 times worse. The grid goes 96 per cent UNKNOWN in the band where the snow
starts, rather than filling with confident wrong depth.

That is the behaviour the whole camera-only architecture depends on, and it is
the one thing no public dataset can demonstrate, which is why section 10 of the
build spec asks for the snow world at all.

There is also a systematic positive depth bias that grows with range, from
+7 mm at 1 to 2 m to +73 mm at 7 to 10 m. That is the classic fronto-parallel
bias of block matching on a strongly slanted surface, and the ground plane seen
from a camera tilted 20 degrees down is exactly that. It is a property of the
classical matcher, it is the floor a learned matcher has to beat, and it is
recorded here rather than tuned away.

## Running the real thing

RELLIS-3D is not in this repository. It is tens of gigabytes and it is
distributed by Texas A&M under their own terms.

1. Go to the dataset page and accept the terms:
   <https://github.com/unmannedlab/RELLIS-3D>
2. Download the parts this ablation needs, per sequence:
   the camera images, the labelled images, the Ouster point clouds in KITTI bin
   format, the SemanticKITTI point labels, and the calibration.
3. Unpack them so each sequence is a directory under one root.
4. Run:

```bash
python -m eval.rellis_ablation.run_ablation --dataset ~/data/Rellis-3D --limit 200
```

### Two things the loader will stop on, deliberately

**The directory layout.** `LAYOUT` at the top of `rellis.py` is written from
the published structure and has not been run against a real download. If it
differs, the loader raises with the tree it actually found and names the one
place to correct. It does not guess.

**The stereo baseline and the camera to body extrinsic.** Both must come from
the dataset's own calibration and be passed in. `_read_calibration` raises
until that is done. Guessing a baseline silently scales every depth in the
study by a constant, and guessing an optical transform is, in the build spec's
own words, the bug that costs everyone two days.

These are not gaps to be filled by whatever value makes the code run. They are
measurements, and they have to come from the dataset.

## What comes out

`report/ablation_rellis3d.md` and the matching `.json`:

- accuracy and false free space, LiDAR against stereo, **per terrain class**
- the same, banded by range, because depth error grows with the square of
  range and a single number hides exactly the effect being measured
- measured stereo depth error against the section 03.2 prediction
- how much of the grid stereo declined to decide, and where

The per class breakdown is the deliverable. "We lose four points of accuracy
overall" is a weak claim. "We lose nothing on grass and dirt, two points on
rubble, and we go blind on standing water, which is why the CONOPS routes
around it" is an engineering position.
