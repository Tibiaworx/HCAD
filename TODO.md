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
