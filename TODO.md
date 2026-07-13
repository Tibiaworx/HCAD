# HCAD — known issues & planned fixes

## Exact-radius reference snapping (concentric-boss seam) — largely fixed

**Symptom (historical).** Draw a circle that snaps to a body's rounded edge, extrude it
as a boss concentric with that feature, and turn on **Seamless** (mesh kernel). The walls
fuse, but a faint broken "dashed" lip could remain at the join.

**Root cause.** The circle snapped to the body edge's **tessellation-approximated**
radius (±0.01–0.03 mm from the triangulated mesh), not the source feature's exact radius,
leaving a real sub-0.03 mm radial step at the join. A second contributor: the face-sketch
origin was computed as an *unweighted* mean of triangle centroids, biasing a cylinder
cap's (0,0) off the true axis (~1% of radius), so stacked concentric features also
drifted positionally.

**Fixed (2026-07):**
- `exact_plane_circles` reads the **exact** centre/radius of every circular body edge in
  the sketch plane from the *timeline's source sketch entities* (circles, arcs, slot end
  caps, both sweep ends incl. Direction 2), and `refine_circle` snaps each
  tessellation-fitted reference circle to the matching exact one. Matching requires
  **centre AND radius** agreement, so concentric features (a boss with a bore) can't hand
  a hole reference the boss's radius.
- Face-sketch origins use the **area-weighted** face centroid, so a cylinder cap's (0,0)
  sits exactly on its axis regardless of triangulation.
- Extruded/cut circle profiles carry exact-arc annotations into the kernel (true
  cylindrical faces for the last solid feature; see `gate_arcs`), so exported STEP bores
  are exact cylinders.

**Remaining edges.**
- Circular edges created by **revolve/loft** features aren't in the exact-circle registry
  (only extrude/cut sketch entities are); their references still use the tessellation fit.
- The registry projects through each feature's **stored** plane; if an upstream edit
  slides a face along its normal, regen reprojects the geometry but the registry's cap
  offsets use the stored distances — a moved cap can fall outside the 0.02 mm plane test
  and lose refinement (detection still works, just unrefined).
- truck 0.6 booleans panic on NURBS-faced *base* solids, so exact-arc geometry is gated
  to the last solid feature; widen `gate_arcs` when the kernel (or an OCC backend) can
  take it.

## Over-defined diagnostics — done

`DofReport::conflicting` lists the constraints participating in a conflict: at the
least-squares solution Jᵀr = 0, so the leftover residual lands entirely on the
mutually-inconsistent rows, and mapping significant residual rows back to their
constraint identifies them (no extra solves). Redundant-but-consistent over-constraints
have zero residual and are correctly not flagged. Conflicting **dimensions** draw red in
the viewport and the status line counts them.

**Remaining:** geometric relations (horizontal/parallel/etc.) that conflict have no
viewport glyph, so they contribute to the count but don't individually show red — only
dimensions do. A relations list in the panel with per-row red marking would close that.

## Mesh-surgery fillet: corner patch isn't a true sphere

**Symptom.** Filleting a box's top-face perimeter (`saved files/fillererror.hcad`,
`fillererror2.hcad`): the straight edges round correctly, but the corners where two
fillets meet look wrong — a flat/pinched patch instead of a smooth continuation of the
round.

**Root cause.** `corner_centre()` in `bevel.rs` — a correct 3-face inscribed-sphere
solver — exists but is **never called**. `run_surgery`'s corner-patch step (§3) instead
fans triangles to the boundary loop's flat *vector average*, which sits inside the true
rounded volume rather than on its surface (a flat average of points spread around a
sphere is not itself on that sphere) — hence the pinch.

**Why it's still open (2026-07).** Two attempted fixes both caused real regressions
(watertightness holes in previously-working cases — whole-cube fillet, an L-prism's
concave edge, cut-after-bevel):
1. Matching the boundary onto the analytically-correct corner sphere: the boundary
   loop's non-apex points don't actually lie on one shared sphere in general — how many
   "mitred" (off-sphere) points a vertex's boundary contains depends on how many
   *selected* edges converge there (0 for an all-edges-sharp vertex — never reached; 1 for
   a corner where exactly 2 selected edges meet a 3rd unselected face; more when 3+ edges
   are all selected, e.g. a fully-rounded cube). A validation tuned for the 1-outlier case
   broke on the 3-selected-edge case.
2. Pushing the flat apex outward along the corner's averaged face normal, plus a
   subdivided (2-band) dome: also broke the 3-selected-edges case — some generated
   points collapsed onto each other (duplicate positions → degenerate/non-manifold
   triangles at the weld step).

Both attempts are reverted; `run_surgery`'s corner patch is back to the original flat
fan (safe, watertight, just visually pinched at the corner).

**Path to a real fix.** Handle vertex valence explicitly rather than one generic
formula: classify each corner-patch vertex by how many of its incident edges are
selected (exactly 2 vs 3+) and build the boundary/dome differently per case, instead of
assuming a single "N boundary points, at most one mitred" shape. Test against *all* of
`bevel_cube_is_watertight_and_unsharp`, `bevel_l_prism_with_concave_edge_is_watertight`,
`cut_after_bevel_reaches_full_depth`, AND the two saved top-rim-fillet files before
calling it done — any fix must keep every one of those watertight.

**Workaround today.** None needed structurally (the flat-fan corner is watertight and
correct in extent, just not perfectly round) — cosmetic only.
