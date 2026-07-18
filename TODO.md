# HCAD — known issues & planned fixes

## STL import (editable + reference scan) — v1 shipped 2026-07

Insert → **Import STL…** brings a mesh in as an editable body (`ImportMesh`): it unions into
the timeline like a boss, and downstream cuts/fillets/booleans apply to it (mesh kernel is
forced on, like fillets). Insert → **Reference Mesh (STL)…** brings it in as a translucent
teal ghost (`RefMesh`) that never joins the solid — for building parts against a 3D scan.
Sketching on any plane through a reference scan shows the scan's **section outline** in teal
and merges it into the snap pool (endpoints/midpoints/edge-pick), for tracing cross-sections.
Both embed the STL deflate+base64 in the `.hcad` (self-contained); decode is memoized.

**v2 (2026-07):**
- **Mesh PropertyManager** (double-click / context-menu on a [mesh]/[scan] row; auto-opens on
  import): scale with ×25.4 / ×1000 unit presets, XYZ rotation (about origin) + XYZ offset,
  and opacity for scans. Placement is baked into the cached mesh (`import_mesh_cached`);
  ghosts respawn via a placement fingerprint on `RefMeshEnt`.
- **Section fitting** (`fit_section_shapes`): raw triangle chords are welded, chained, and
  fitted into circles / arcs / decimated polylines (Kasa LSQ + greedy arc/line runs, 0.05 mm
  tol). Fitted circles feed `reference_circles`, so radius/centre snapping on a scanned
  bore is exact; the snap pool sees decimated segments, not thousands of chords.
- **Trace Section** button (sketch tab, shown when a scan crosses the plane): converts the
  fitted shapes into real sketch entities (circles/arcs/lines) in one click, undoable.

**v3 (2026-07):**
- **3-point align** (mesh PM): click three points on the mesh (its own placed surface, teal
  markers sized by pick order) → their plane maps onto Front/Top/Right, first point
  optionally → origin. `align_placement` composes onto the existing placement; verified
  end-to-end by test (picked plane lands exactly on the datum, rigid, volume-preserving).
- **Auto mesh repair** on every editable import (`repair_mesh`): weld (bbox-relative tol),
  drop degenerate/duplicate faces, ear-clip-fill boundary loops ≤ 64 edges; conservative —
  larger openings are reported (`open_edges_left`), not papered over. Toasts report what
  was done; still-open meshes warn toward Reference import.
- **Measure on scans**: the measure tool falls back to reference-scan surfaces (nearest
  scan hit) when the click misses the body — size features off the scan before modelling.
- **Section fit tolerance** slider (scan PM, 0.01–1 mm log scale, per scan): coarse for
  noisy scans → fewer, smoother fitted shapes; each scan fits with its own tolerance.

**Responsiveness (2026-07):**
- Mesh-kernel rebuilds (Seamless / fillets / imports) run on a **background thread**
  (`RegenJob` + `finish_regen_job`): the old body stays visible and interactive, a status-bar
  spinner shows "Rebuilding…", and a doc change mid-rebuild queues exactly one fresh run.
- Mesh-PM drags are **debounced** (300 ms) before triggering a rebuild; scan-ghost respawns
  settle 250 ms after the last placement change, so scrubbing scale/rot/offset stays smooth.
- The exact (truck) path still rebuilds synchronously — it's fast for its feature set, but
  could move onto the same job if it ever stalls.

**Remaining edges.**
- Scan sections come only from `RefMesh`; sectioning the solid body itself (imported or not)
  through a sketch plane could reuse the same helper (`mesh_plane_section`).
- Measure prefers the body: a scan surface in FRONT of body geometry can't be measure-picked
  (body hit wins). Fine until parts closely wrap the scan.
- Repair fills planar-ish loops ≤ 64 edges; big scan holes stay open (reported honestly).
- No decimated display copy for huge scans; no external-file linking (blob always embedded).
- Fit primitives to scan REGIONS (click a scanned cylinder → axis datum) still future work.

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

## Mesh-surgery fillet: corner patch isn't a true sphere — 2-edge case fixed

**Symptom.** Filleting a box's top-face perimeter (`saved files/fillererror.hcad`,
`fillererror2.hcad`): the straight edges round correctly, but the corners where two
fillets meet look wrong — a flat/pinched patch instead of a smooth continuation of the
round.

**Root cause.** `run_surgery`'s corner-patch step fanned triangles to the boundary
loop's flat *vector average* — a synthetic point pulled inside the true rounded volume
(an average of points spread around a curve sits inside what the curve bounds) — hence
the pinch.

**Fixed (2026-07) for the 2-edge case** — researched Blender's `bmesh_bevel.cc`
(`source/blender/bmesh/tools/`) for how it handles this. Key finding: a vertex where
exactly **two** edges are beveled (any number of sharp/unselected edges may run between
them) is Blender's `M_NONE` ("weld") case — it needs **no interior corner mesh at all**;
the two edges' own profiles already share a common point (the far face's corner) and
that's sufficient. Ported that specifically: when a corner's boundary walk contains
exactly two rounded (multi-point) arcs, fan from the real, already-computed point they
share (both arcs' `cpt()` value for that shared face is identical by construction)
instead of a synthetic average. No new vertex position is ever introduced for this case
— proven by `weld_corner_fans_from_a_real_arc_point_not_a_synthetic_average`, which
independently recomputes the two edges' allowed arc points and asserts every
corner-area mesh vertex matches one exactly.

**Still open: 3+ edges selected at one vertex** (e.g. a fully-rounded cube corner, or
any corner where 3+ fillets converge). Two earlier attempts at a general fix (matching
the boundary onto an analytically-correct corner sphere; pushing the flat apex outward
plus a subdivided dome) both caused watertightness regressions there — a vertex's
boundary can contain a variable number of "mitred" (off-sphere) points depending on how
many edges are selected, and neither attempt modeled that correctly. Both were reverted;
this case still falls through to the old flat-fan (safe, watertight, just visually
pinched). Blender's own general answer (`adj_vmesh`, for the true N≥3 case) is a
Catmull-Clark-style recursive subdivision from a coarse control mesh, snapping the
boundary onto the true profile curve at each level, with an empirically-tuned
"fullness" constant positioning the initial interior pole — a substantially bigger port
than the 2-edge case; Blender also has an *exact* analytic shortcut specifically for 3
orthogonal edges (`tri_corner_adj_vmesh`, snapping onto a unit sphere octant via
`corner_centre()`, which already exists in our code and is unused) that would be the
natural next target before attempting the fully general case.

Test against *all* of `bevel_cube_is_watertight_and_unsharp`,
`bevel_l_prism_with_concave_edge_is_watertight`, `cut_after_bevel_reaches_full_depth`,
AND the two saved top-rim-fillet files before calling any N≥3 fix done.

## Mesh-surgery fillet: single-edge fillets always failed — fixed

**Symptom.** Filleting just ONE edge of a box (`saved files/fillererror3.hcad`) declined
the surgery path and silently fell back to CSG, which produced a self-intersecting
"bowtie" mesh slicing through the whole part. Confirmed (via a clean worktree at commit
`30ba3ad`) this predates this session entirely — **every** single-edge fillet, on any
box, at any size, has always hit this.

**Root cause.** A face only insets near a *selected* edge on its own boundary; a face
untouched by any selected edge is left exactly as it was. So at a vertex where some
other, possibly distant, selected edge moved a *different* face's own corner, a sharp
(unselected) edge leading toward the untouched face ends up with two different endpoint
positions depending which of its two faces you ask — a real crack. The corner patch
never fixes this: it only runs at the vertex that moved, and never propagates that
movement along the sharp edge to whichever vertex — possibly far away — doesn't.

**Fixed (2026-07).** For every sharp edge, check whether its two bordering faces' own
corner positions agree at each endpoint; when they don't, stitch the two faces' corner
positions together as a trivial one-segment "edge strip" (the same triangle pattern
already used for rounded edges, just without an arc). When one end already agrees (the
ordinary case for virtually every untouched edge), that half degenerates to a zero-area
triangle the mesh builder already filters out; when both ends agree, nothing is emitted.
No new vertex position is ever introduced. Verified: all 12 single-edge fillets on a test
box now succeed (was 0/12); `fillererror3.hcad`'s exact geometry builds watertight and
manifold both directly and through the real `regenerate_mesh` pipeline.
