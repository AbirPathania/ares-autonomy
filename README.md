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
| 0 | RELLIS-3D ablation. Traversability with LiDAR, then stereo only, same platform, same data. | Mac, MPS | A measured number: what accuracy you lose without LiDAR, by terrain class. | **pipeline complete and verified on synthetic geometry.** Needs the dataset for the real number |
| 1 | `ares_math` and its full test suite. No ROS yet. | Anywhere | Every table in section 03 reproduced to 1e-9. Symbolic check passes. | **passing** |
| 2 | Devcontainer, URDF generated from constants, TF tree, RViz. | Mac or Codespace | Rover displays correctly, differential enforced, no TF warnings, optical frame test passes. | **passing.** Rover spawns, TF is correct, all four cameras render |
| 3 | Gazebo worlds 1 and 2. Drive control and mobility. | Both | Gates 1 and 2 from section 10. | **gate 1 passing.** Gate 2 written, not yet run |
| 4 | Stereo pipeline on rendered RGB. Real matcher, confidence output. | Both | Measured depth error tracks the section 03.2 curve across the range. | **passing in simulation.** See below |
| 5 | Segmentation, detection, elevation map, Nav2. | Both | Gate 3. Trench detected and stopped for at 0.95 m, missed at 0.50 m. | in progress. Perception path runs end to end; gate 3 not yet passing |
| 6 | BlenderProc snow and scree generation, fine-tune, gate 4. | Mac | Confidence collapses on textureless snow and the vehicle slows, rather than reporting confident wrong depth. | demonstrated on synthetic data in phase 0. Gate 4 world built, not yet run |
| 7 | Isaac Sim closed loop, if still warranted. | Rented RTX | Decide after phase 6 whether it adds anything phases 0 to 6 did not. | not started |

Phases run in order. Each gate exists because the next phase is unverifiable
without it.

## Gate 1, measured

Run in Gazebo Harmonic, headless, four software rendered cameras, with the
real SGBM matcher on the rendered images and ground truth depth used only for
scoring:

```
f = 320.0 px (from CameraInfo). At 3 m over 512,890 pixels:
  sigma 0.0363 m, bias +0.0139 m
  section 03.2 bounds: 0.0352 m nominal (0.25 px), 0.0703 m pessimistic (0.50 px)
  the matcher is achieving about 0.26 px of disparity precision

GATE 1: PASS
```

Section 03.2 assumes a good matcher achieves 0.25 px of subpixel disparity
precision on textured ground. It achieves 0.26. The table is not optimistic.

Two numbers worth carrying forward from the same run:

**Pipeline latency is 83 to 93 ms, worst case 143 ms.** Section 03.3 budgets
300 ms for the whole sense-to-brake chain, and this is the matcher alone. At
4 km/h it is already spending about 100 mm of stopping distance.

**The simulated stereo pair is not synchronised.** The synchroniser refuses
around half of all pairs, with a worst observed skew of 600 ms. That is not a
fault in the synchroniser, it is the synchroniser working: Gazebo renders the
two eyes on separate passes and their timestamps drift. On the real vehicle
this is the shared hardware trigger requirement, and the budget is 39
microseconds at one radian per second. In the simulator it means the effective
stereo rate is half the camera rate, which is worth knowing before anyone
reads a frame rate off a demo.

## Phase 2, what is in it

The URDF is **generated**, never written. `tools/generate_urdf.py` reads
`constants.py` and emits `ares_description/urdf/ares_g.urdf.xacro`: 22 links,
21 joints, every origin traced back to a hull number. Change a constant, re-run
the generator, re-run pytest. CI fails if the committed URDF is not what the
constants currently produce.

`ares_description/test/test_urdf_geometry.py` parses the URDF as plain XML and
composes the transform chain with the same rotation convention the solver uses.
That is what lets the phase 2 geometry gate run on a laptop in a second, with
no ROS, no container and no simulator. 23 tests, including:

- the mast camera optical frame sits at 0.95 m in the default pose, which is
  the gate the build spec names
- the optical frame is Z forward, X right, Y down, and the mast optical axis
  points forward and exactly 20 degrees down in `base_link`
- the right eye is actually on the right, and both eyes of a pair are at the
  same height and the correct baseline apart
- the front wheel hangs off the rocker and the middle and rear off the bogie,
  which is the topology that makes it a rocker-bogie
- four RGB cameras are simulated, not one depth camera, and ground truth depth
  is on `/sim/ground_truth/depth` where it cannot reach the control path
- the drive plugin carries the 4 km/h autonomous cap
- the URDF is a tree, every link has mass or is a frame, and the masses sum to
  the configuration budget

What is *not* verified yet: the RViz render and the live TF tree. Both need the
container, and Docker is not installed on this machine. The differential is
declared two ways in the Gazebo block, as a gearbox joint and as a plugin, and
which one the physics engine actually honours is a phase 3 question that only
the simulator can answer.

### One more rounding found

Section 08 prints the optical frame rotation as `-1.5707963268`, a ten digit
rounding of `-pi/2`. Harmless as a printed value, but it leaves a 3.2 nanoradian
residual on every camera transform and turns an exact axis test into an
approximate one. `constants.py` stores `-pi/2` exactly and keeps the printed
form alongside it, with a test asserting the two agree to the precision the
document prints at.

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
ares_description/            ROS 2 package, hull revision H-2
  urdf/ares_g.urdf.xacro    GENERATED from constants.py, never hand edited
  launch/display.launch.py  the phase 2 RViz gate
  config/ares.rviz
  test/test_urdf_geometry.py
.devcontainer/              one definition, Mac and Codespaces
tools/
  generate_golden.py        rebuilds golden_values.yaml from the document
  check_golden.py           asserts the committed golden file still matches
  derive_symbolic.py        sympy derivations of every closed form
  generate_urdf.py          emits the URDF from constants.py
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

## What running it found

Everything below was found by executing the thing, not by reading it. Each one
is recorded rather than quietly worked around, and each has a commit.

**In the build spec**

- Section 03.4, blind zone table, mast row, 40 degree column prints 0.35 where
  the formula gives 0.35519, which rounds to 0.36. One cell out of twenty.
- Section 08 prints the optical rotation as `-1.5707963268`, a ten digit
  rounding of `-pi/2` that leaves a 3.2 nanoradian residual on every camera
  transform.
- Section 06.2's Dockerfile does not build. `ultralytics` pulls `psutil`,
  which pip cannot uninstall over the apt copy, and the image fails. Codespaces
  then silently substitutes a recovery container with no ROS in it, which looks
  exactly like a different problem.
- Section 06.2's devcontainer has no SSH server, so a Codespace built from it
  cannot be driven from a terminal, which is what section 06.4 wants it for.
- `torch` defaults to CUDA wheels, which section 06.4 says cannot be used here.

**In this implementation, found only by running it**

- The SDFormat gearbox joint was emitted with URDF attribute syntax. Valid XML,
  silently invalid SDF, and it took the whole model parse down so the rover
  never spawned.
- `StereoNode.handle` shadowed a property `rclpy` uses inside `Node.__init__`.
  Both stereo nodes died at startup while the simulator ran happily.
- `ares_perception` had no `setup.cfg`, so its console scripts installed to
  `bin/` while `ros2 run` looked in `lib/ares_perception`. Every node was
  invisible to launch, with an error that reads like a failed build.
- `robot_description` was not wrapped in `ParameterValue`, so launch tried to
  parse the URDF as YAML and threw before starting anything.
- The point cloud was published in the camera optical frame, correctly
  labelled, and consumed as if it were `base_link`. This is the section 08 bug,
  in my own code. It does not present as a frame error. It presents as the
  rover reporting an obstacle two centimetres ahead and creeping at 0.18 km/h.
- **Nothing applied the speed limit.** The traversability node published
  `/speed_limit`, no node subscribed, and the rover drove 59 m straight through
  the governing trench at full speed while a topic beside it read 0.18 km/h. A
  limit nothing applies is a log line, not a safety mechanism.
- The disparity search did not scale with image width, so at the simulator's
  640 by 360 it searched twice as far as the geometry needs, resolved nothing
  closer for it, and spent the memory anyway.
- `trusted_range` walked outward from the datum, which is the middle axle, so
  the first half metre of "unknown ground ahead" was the rover's own front
  wheels and the vehicle stopped permanently.

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
python tools/check_golden.py
python tools/derive_symbolic.py
python tools/generate_urdf.py
```

Bring up the container, on a Mac with Docker Desktop or in a Codespace, and
show the rover:

```bash
ros2 launch ares_description display.launch.py
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
