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

## Over-defined diagnostics

The sketch status line reports "Over defined — conflicting relations" (from the residual
of the solved system), but doesn't say **which** constraints conflict. Find a minimal
inconsistent subset (e.g. greedily drop rows and re-check residual) and highlight those
dimensions/relations red in the viewport and panel.
