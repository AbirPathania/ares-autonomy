# ares-autonomy

Camera-only autonomy for ARES-G, and the simulator that validates it.

Built to the **ARES Autonomy Build Spec v1**, 4 September 2026. Geometry taken
from frozen hull revision H-2. This repository is the implementation of that
document, and the document is the source of truth for every number in it.

Patroclus Systems. Confidential.

## The finding this repository exists to hold

Whether you can see into a hole is a question about where your eye is, not
what kind of eye it is.

```
NEGATIVE OBSTACLE DETECTION RANGE, SENSOR AGNOSTIC
R_max = w * h / d
```

No sensor type appears in that equation. A LiDAR mounted at the same height
has exactly the same blind geometry, so dropping it costs nothing on negative
obstacles. Raising the sensor is the only thing that helps, which makes mast
height a perception parameter and not a packaging convenience.

Three decisions follow, and all three are decisions rather than observations:

1. **The sensor mast is not optional and it has a number.** Solving
   `h = d * D_stop / w` gives 0.86 m at 4 km/h. The mast puts the primary
   camera at 0.95 m deployed, which clears 4 km/h with margin. The test suite
   pins that at 4.26 km/h.
2. **8 km/h cannot be autonomous.** Not with cameras, not with LiDAR, not at
   any mast height that fits a 25 kg two-person-carry vehicle. It needs 2.82 m.
   Transit speed is a teleoperated or known-clear-ground mode.
3. **The real risk of camera-only is texture, not geometry.** Fresh snow has
   no texture for passive stereo to match. That is the failure mode the
   simulation exists to characterise, and it is the reason every depth pixel
   in this stack carries a confidence value.

## Zero false detections is not the goal, and claiming it would be the bug

The camera-only bet fails in exactly one way: textureless terrain returns bad
depth that *looks* like good depth. The mitigation is not a better matcher. It
is a matcher that says when it does not know.

Low confidence over a region is a first-class signal that means slow down or
stop, exactly like an obstacle. A stack that treats unknown as free space will
drive into a snowdrift at full speed, and it will do it in the field, not in
the sim. Everything here is built so that degradation is measured and acted
on, not hidden.

## What is built

| Phase | Build | Where | Gate | State |
|---|---|---|---|---|
| 0 | RELLIS-3D ablation. Traversability with LiDAR, then stereo only, same platform, same data. | Mac, MPS | A measured number: what accuracy you lose without LiDAR, by terrain class. | not started |
| 1 | `ares_math` and its full test suite. No ROS yet. | Anywhere | Every table in section 03 reproduced to 1e-9. Symbolic check passes. | **passing** |
| 2 | Devcontainer, URDF generated from constants, TF tree, RViz. | Mac or Codespace | Rover displays correctly, differential enforced, no TF warnings, optical frame test passes. | not started |
| 3 | Gazebo worlds 1 and 2. Drive control and mobility. | Both | Gates 1 and 2 from section 10. | not started |
| 4 | Stereo pipeline on rendered RGB. Real matcher, confidence output. | Both | Measured depth error tracks the section 03.2 curve across the range. | not started |
| 5 | Segmentation, detection, elevation map, Nav2. | Both | Gate 3. Trench detected and stopped for at 0.95 m, missed at 0.50 m. | not started |
| 6 | BlenderProc snow and scree generation, fine-tune, gate 4. | Mac | Confidence collapses on textureless snow and the vehicle slows, rather than reporting confident wrong depth. | not started |
| 7 | Isaac Sim closed loop, if still warranted. | Rented RTX | Decide after phase 6 whether it adds anything phases 0 to 6 did not. | not started |

Phases run in order. Each gate exists because the next phase is unverifiable
without it.

## Phase 1, what is in it

```
ares_math/                  pure Python, no ROS, no side effects
  constants.py              ONLY place H-2 numbers live
  units.py                  the one pint registry, and the unit convention
  optics.py                 f_px, projection, unprojection, blind zone
  stereo.py                 depth, error, baseline selection, sync budget
  safety.py                 stopping distance, negative obstacle range
  kinematics.py             rocker-bogie pose solver, differential
  tests/
    golden_values.yaml      the tables in section 03, frozen
    test_optics.py
    test_stereo.py
    test_safety.py
    test_kinematics.py
    test_constants.py
    test_symbolic.py
tools/
  generate_golden.py        rebuilds golden_values.yaml from the document
  derive_symbolic.py        sympy derivations of every closed form
```

`ares_math` has no ROS dependency so it can be tested in milliseconds, on any
machine, without a container, without a simulator, and without a robot. Every
number in section 03 of the build spec is a test. That is what "one hundred
percent accurate" looks like in practice: not a claim, a test suite that fails
loudly the moment an equation is mistyped.

### The five mechanisms that make the math provably right

| # | Mechanism | Catches |
|---|---|---|
| 1 | **Golden values.** Every table in section 03 is in `golden_values.yaml`, asserted at 1e-9 relative tolerance. | A mistyped formula, a lost factor of two, a degree-radian slip |
| 2 | **Units carried, not assumed.** Public functions declare their units with `pint`. Millimetres and metres cannot be confused, and an angle where a length belongs raises. | The single most common and most expensive class of error in this project |
| 3 | **Two independent implementations.** Closed form and numerical, asserted equal. Depth error analytically and by finite difference. Rocker pose by solver and in closed form. | A derivation that is wrong in both the code and your head |
| 4 | **Property tests** via `hypothesis`. Project-then-unproject is identity. Rigid body distances are invariant under articulation. Depth error is strictly increasing in range. | Edge cases nobody thought to write a test for |
| 5 | **Symbolic derivation checked in.** `tools/derive_symbolic.py` derives `R_max = w*h/d`, the depth error relation, the stopping distance, the sync terms and the differential from first principles, and asserts they match the hand-written closed forms. | Silent divergence between the document and the code |

The golden file is deliberately **not** generated from `ares_math`. It is
generated by `tools/generate_golden.py`, which writes the formulas out inline
from the document and asserts its own results round to the values printed in
the PDF. So three things must agree before a test can pass: the printed table,
the generator, and the package.

### One erratum found in the source document

Section 03.4, blind zone table, mast row, 40 degree column. The document prints
0.35 m. `h / tan(tilt + VFOV/2)` with h = 0.95 and VFOV = 59 gives 0.35519,
which rounds to 0.36 at two places. Every other cell in that table rounds
correctly, so this is a single typo and not a different formula. It changes
nothing, because the mast is specified at 20 degrees of tilt, not 40.

It is recorded in `tools/generate_golden.py` under `ERRATA` and in the
`_errata` key of the golden file rather than quietly accommodated, because
silent divergence between the document and the code is exactly what section 11
exists to prevent.

## Stereo synchronisation

Two eyes only fuse if they look at the same instant. If the right camera
exposes `dt` after the left while the vehicle is moving, the pair no longer
shares a rigid baseline and the matcher solves a geometry that never existed.
Both error terms are derived symbolically and implemented in `stereo.py`:

```
translation, on the optical axis     bias   = v * dt
rotation, on the optical axis        d_disp = f * tan(omega * dt)
sync budget, the exact inverse       dt_max = atan(budget / f) / omega
```

Rotation dominates. It enters depth multiplied by `Z^2 / (f*B)` while the
translation term does not scale with range at all. At the working case, one
millisecond of skew during a 0.5 rad/s turn costs 62 mm of depth at 5 m, while
the same millisecond of forward motion at 4 km/h costs 1.1 mm.

Holding a tenth of a pixel through a 1 rad/s turn allows 156 microseconds of
skew. A shared hardware trigger clears that by an order of magnitude. Software
timestamp matching does not, reliably, which is why the requirement is written
as a test rather than a preference.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./ares_math
pip install pytest hypothesis sympy ruff
pytest -q
```

Rebuild the frozen tables from the document, and re-run the derivations:

```bash
python tools/generate_golden.py
python tools/derive_symbolic.py
```

## The rules that keep this true six months from now

- No physical constant is written anywhere except `constants.py`.
- No equation is implemented anywhere except `ares_math`. ROS nodes import.
  They never calculate.
- A magic number in a ROS node is a review rejection, no exceptions, because
  that is precisely how a deck ends up claiming 17 kg while the engineering
  says something else entirely.
- The URDF is generated from `constants.py`, never hand edited.
- When the hull revises, it is one edit in `constants.py` and a test run, not
  a search through the tree.

## Open items this repository cannot close

**The mast.** Section 02 produces a hard requirement for a sensor mast at
roughly 0.95 m deployed. H-2 assumes a retracted mast in its 455 mm stowed
height but does not budget the mast itself. That needs feeding back into the
hull along with the LiDAR deletion and the two camera units, and the
configuration ladder re-run against the 25 kg ceiling.

**The mass ladder.** The tracked plus drone bay configuration is over by
1.10 kg. Deleting the Livox Mid-360 is worth roughly 265 g of that. The mast
and cameras spend some of it back. Until it is re-run, the full ground-air
configuration is unresolved rather than solved. `test_constants.py` holds that
statement as a passing test so it cannot quietly become an assumption.

**Behind and to the sides.** The rover is blind there. Reversing out along the
path you drove in, using the map you built on the way, is the mitigation. That
belongs in the CONOPS, not in a third camera and a lost mass budget.
