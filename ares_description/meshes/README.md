# meshes

Empty on purpose.

The URDF uses primitive shapes: a box for the hull, cylinders for the wheels
and for the rocker and bogie arms. That is enough for the physics, for the
frames, and for reading the articulation in RViz.

Visual meshes come from the CAD when there is CAD to take them from. Until
then a mesh here would be a drawing of a rover rather than a model of one,
and it would be the only geometry in the repository that did not trace back
to `ares_math/constants.py`.
