# HCAD — known issues & planned fixes

## Exact-radius reference snapping (concentric-boss seam)

**Symptom.** Draw a circle that snaps to a body's rounded edge, extrude it as a boss
concentric with that feature, and turn on **Seamless** (mesh kernel). The walls fuse,
but a faint broken "dashed" lip can remain at the join.

**Root cause.** The circle snaps to the body edge's **tessellation-approximated** radius,
not the source feature's exact radius. The body is triangulated at ~0.03 mm tolerance, so
the detected reference-circle radius (from `fit_circle` on the projected edge chain) is
only accurate to roughly ±0.01–0.03 mm. The new boss therefore ends up a hair off the
original cylinder, leaving a genuine sub-0.03 mm radial **step** at the join. Seamless
fuses coincident *walls*, but it can't erase a real (if tiny) radius difference — so the
step renders as a near-90° edge (the dashes).

**Proper fix.** Make the circle-snap reference the body feature's **exact** radius instead
of the tessellated approximation:

- Detect circular edges from the truck **B-rep** (query the arc's exact radius/centre from
  the kernel) rather than from `part.edges` (the tessellation), *or*
- When a sketch circle snaps to a body arc, record a real **concentric + equal-radius
  reference** relation to that arc (topological reference), so regenerate always rebuilds
  the boss at exactly the source radius.

Either makes concentric/stacked same-radius bosses come out exactly equal, so the join is
truly seamless with no lip.

**Workaround today.** Dimension both circles to the identical value (select each, type the
same radius/diameter). With exactly equal radii the join fuses cleanly under Seamless.

**Related code.** Reference-circle detection: the in-plane edge → `fit_circle` loop in
`sketch_interaction` (builds `reference_circles`); radius snapping: `snap_radius`. The
mesh fusion / edge classification: `regenerate_mesh` (boss overlap) and
`mesh_tessellation` (`feature_edges_opts`, 50° threshold) in `hworks-app`/`hworks-geometry`.
