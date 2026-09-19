//! `hworks-geometry` — Layer 1: the geometry kernel seam.
//!
//! Hides the concrete CAD kernel ([`truck`]) behind this crate's API so the rest
//! of HCAD never depends on it directly — the seam that lets us swap in
//! OpenCASCADE later. See `DESIGN.md` §3 and §7.
//!
//! As of **M4** the kernel does extrude (boss), boolean union, and boolean cut
//! (difference), plus tessellation. The truck `Solid` is kept alive inside the
//! opaque [`KSolid`] so booleans have a B-rep to operate on (not just a mesh).

use truck_meshalgo::prelude::*;
use truck_modeling::{builder, Point3, Vector3};

pub mod drawing;
pub mod gear;
mod bevel;
mod csg;
mod fillet;
mod mesh_bool;
pub use bevel::{bevel_feature_edges, bevel_mesh, bevel_mesh_and_edges, bevel_mesh_selected, fillet_segments};
pub use fillet::{chamfer_mesh, round_mesh, threaded_hole};
pub use mesh_bool::{feature_edges_by_face, is_manifold, mesh_difference, mesh_intersection, mesh_union, mirror_mesh, remesh_solid, take_dense_skip_count, take_fallback_count};

/// The surface a face of a mesh actually lies on.
///
/// CARRIED from the tool that built the face, not recovered from its triangles afterwards. The
/// extrude that bores a hole knows perfectly well it is making a cylinder; today it throws that
/// away and the exporter is left guessing at a ring of flat strips. A mesh that remembers can be
/// written out as one real surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Surf {
    /// A flat face through `origin` facing `normal`.
    Plane { origin: [f64; 3], normal: [f64; 3] },
    /// A cylindrical wall about the line through `origin` along `axis`.
    Cylinder { origin: [f64; 3], axis: [f64; 3], radius: f64 },
    /// The band a rolling ball leaves round a circular rim: a tube of radius `minor` whose centre
    /// line is the circle of radius `major` about the axis through `origin` along `axis`.
    Torus { origin: [f64; 3], axis: [f64; 3], major: f64, minor: f64 },
}

impl Surf {
    /// Whether two records describe the SAME surface.
    ///
    /// Not the same as being equal. A surface is written down with a point on it, and any point
    /// will do — so the same flat face picked up a different record from every feature that touched
    /// it, and the same bore a different one from every height it was cut at. Compared as written,
    /// one face reads as several unrelated ones.
    pub fn is_same_as(&self, other: &Surf, tol: f64) -> bool {
        let par = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        match (*self, *other) {
            (Surf::Plane { origin: o0, normal: n0 }, Surf::Plane { origin: o1, normal: n1 }) => {
                // Facing matters: the two sides of a sheet are different faces of the solid.
                par(n0, n1) > 1.0 - 1.0e-9
                    && (par([o1[0] - o0[0], o1[1] - o0[1], o1[2] - o0[2]], n0)).abs() < tol
            }
            (
                Surf::Cylinder { origin: o0, axis: a0, radius: r0 },
                Surf::Cylinder { origin: o1, axis: a1, radius: r1 },
            ) => (r0 - r1).abs() < tol && par(a0, a1).abs() > 1.0 - 1.0e-9 && axis_gap(o0, a0, o1) < tol,
            (
                Surf::Torus { origin: o0, axis: a0, major: m0, minor: r0 },
                Surf::Torus { origin: o1, axis: a1, major: m1, minor: r1 },
            ) => {
                (m0 - m1).abs() < tol
                    && (r0 - r1).abs() < tol
                    && par(a0, a1).abs() > 1.0 - 1.0e-9
                    && axis_gap(o0, a0, o1) < tol
                    // A torus's origin names the plane its centre circle lies in, so sliding it
                    // along the axis gives a DIFFERENT torus — unlike a cylinder's.
                    && par([o1[0] - o0[0], o1[1] - o0[1], o1[2] - o0[2]], a0).abs() < tol
            }
            _ => false,
        }
    }
}

/// How far `p` is off the line through `o` along `axis`.
fn axis_gap(o: [f64; 3], axis: [f64; 3], p: [f64; 3]) -> f64 {
    let d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
    let al = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
    let r = [d[0] - axis[0] * al, d[1] - axis[1] * al, d[2] - axis[2] * al];
    (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
}

/// `tri_surf` entry for a triangle whose surface nobody recorded.
pub const NO_SURF: u32 = u32::MAX;

/// A tessellated triangle mesh handed up to the renderer.
#[derive(Debug, Default, Clone)]
pub struct TriMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    /// The surfaces this mesh's faces lie on, for those a tool bothered to record.
    pub surfaces: Vec<Surf>,
    /// Which surface each TRIANGLE lies on — an index into `surfaces`, or [`NO_SURF`]. Empty when
    /// nothing is tagged at all; otherwise one entry per triangle, so it can be indexed directly.
    pub tri_surf: Vec<u32>,
}

impl TriMesh {
    /// The surface triangle `t` lies on, if anything recorded one.
    ///
    /// Answers "nobody said" for the whole mesh when the tag array has fallen out of step with the
    /// triangles, rather than reading it anyway. Plenty of code rebuilds a mesh's triangles
    /// without knowing tags exist, and a tag array off by even one triangle doesn't degrade —
    /// every answer after the slip names the wrong surface, which is worse than no answer at all.
    pub fn surf_of(&self, t: usize) -> Option<Surf> {
        // Indexed rather than `.get()`: a glob import in this crate shadows the slice method with
        // one that returns by value, and the borrow checker errors read as nonsense.
        if self.tri_surf.len() != self.indices.len() / 3 || t >= self.tri_surf.len() {
            return None;
        }
        let s = self.tri_surf[t] as usize;
        if self.tri_surf[t] == NO_SURF || s >= self.surfaces.len() {
            return None;
        }
        Some(self.surfaces[s])
    }

    /// Record `s` as the surface of every triangle from `first` to the end — the shape a builder
    /// wants: note where a face started, emit its triangles, then say what they were.
    pub fn tag_from(&mut self, first: usize, s: Surf) {
        let ntri = self.indices.len() / 3;
        if first >= ntri {
            return;
        }
        let id = match self.surfaces.iter().position(|x| *x == s) {
            Some(i) => i as u32,
            None => {
                self.surfaces.push(s);
                (self.surfaces.len() - 1) as u32
            }
        };
        self.tri_surf.resize(ntri, NO_SURF);
        for e in &mut self.tri_surf[first..] {
            *e = id;
        }
    }

    /// Drop tags that no longer describe anything — after a rebuild that changed the triangles
    /// without updating them, a stale tag is worse than none.
    pub fn clear_tags(&mut self) {
        self.surfaces.clear();
        self.tri_surf.clear();
    }
}

/// A plane as a 3D origin and orthonormal in-plane axes (`u`, `v`) plus `normal`.
/// Mirrors `hworks_document::Plane` in `f64` world space.
#[derive(Debug, Clone)]
pub struct PlaneBasis {
    pub origin: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub normal: [f64; 3],
}

/// An opaque handle to a kernel solid (a truck B-rep `Solid`). Held by the app
/// across operations so cuts/unions can act on the real topology.
#[derive(Clone)]
pub struct KSolid(truck_modeling::Solid);

/// A run of consecutive boundary edges that lies exactly on a circle: edges
/// `first_edge .. first_edge+count` (wrapping) of a profile's polyline loop,
/// sampled from the circle at `center` with `radius`. The wire builder turns
/// each run into a **true circular-arc edge**, so sweeping produces exact
/// cylindrical faces instead of prism facets. Mirrors `hworks_sketch::ArcSpan`.
#[derive(Debug, Clone, Copy)]
pub struct ArcSpan {
    pub first_edge: usize,
    pub count: usize,
    pub center: [f64; 2],
    pub radius: f64,
}

/// One profile boundary segment in plane-local uv: a straight edge, or an exact
/// circular arc through a `transit` point that disambiguates which arc joins
/// the endpoints.
#[derive(Debug, Clone, Copy)]
enum PathSeg {
    Line([f64; 2], [f64; 2]),
    Arc { a: [f64; 2], b: [f64; 2], transit: [f64; 2] },
}

/// A render-ready tessellation: triangle mesh + feature/boundary edges, split into
/// **sharp** edges (real corners — always drawn) and **tangent** edges (smooth
/// curvature lines between near-coplanar faces — hidden by default, SolidWorks-style).
pub struct Tessellation {
    pub mesh: TriMesh,
    pub edges: Vec<[[f32; 3]; 2]>,
    pub tangent_edges: Vec<[[f32; 3]; 2]>,
}

/// Tolerance for boolean operations and tessellation. Finer than the old 0.05 so a revolve's
/// angular facets are dense enough to meet a boss's wall cleanly at a boolean intersection seam.
const TOL: f64 = 0.02;

// ---------------------------------------------------------------------------
// Public kernel operations
// ---------------------------------------------------------------------------

/// Extrude a closed region (an outer loop plus optional hole loops, in plane-local
/// uv) along the plane normal by `distance` into a solid. `None` if degenerate.
pub fn extrude_solid(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
) -> Option<KSolid> {
    build_solid(outer, holes, basis, 0.0, distance).map(KSolid)
}

/// Like [`extrude_solid`] but the prism overlaps the body by `back` on the side *away* from the
/// sketch plane's exposed face — so a boss overlaps the body it sits on (avoiding a coplanar shared
/// face that fails the union) while its plane-side face stays flush. A normal (+) boss dips `back`
/// behind the plane; a reversed (−) boss keeps its top at the plane and dips its tip past `distance`.
pub fn extrude_solid_with_overlap(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
) -> Option<KSolid> {
    let (start, length) = if distance >= 0.0 { (-back, distance + back) } else { (distance - back, -distance + back) };
    build_solid(outer, holes, basis, start, length).map(KSolid)
}

/// [`extrude_solid`] with exact-arc annotations: [`ArcSpan`] edge runs are built
/// as true circular arcs, so the swept solid has exact cylindrical faces (and a
/// far smaller B-rep than a 100-facet prism). Falls back to lines if the arc
/// path fails.
pub fn extrude_solid_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    distance: f64,
) -> Option<KSolid> {
    build_solid_arcs(outer, holes, outer_arcs, hole_arcs, basis, 0.0, distance).map(KSolid)
}

/// [`extrude_solid_with_overlap`] with exact-arc annotations — see [`extrude_solid_arcs`].
pub fn extrude_solid_with_overlap_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
) -> Option<KSolid> {
    let (start, length) = if distance >= 0.0 { (-back, distance + back) } else { (distance - back, -distance + back) };
    build_solid_arcs(outer, holes, outer_arcs, hole_arcs, basis, start, length).map(KSolid)
}

/// Boolean union of two solids (boss added to an existing body).
pub fn union(a: &KSolid, b: &KSolid) -> Option<KSolid> {
    union_tol(a, b, TOL)
}

/// Boolean union at a caller-chosen tolerance. A smaller tolerance makes the kernel
/// treat near-coincident faces as distinct, so a *sub-micron* boss inflation is
/// enough to dodge the coincident-face boolean failure — keeping the result exact
/// to well within tessellation/manufacturing precision.
pub fn union_tol(a: &KSolid, b: &KSolid, tol: f64) -> Option<KSolid> {
    guard(|| truck_shapeops::or(&a.0, &b.0, tol)).map(KSolid)
}

/// Boolean difference `a − b`: subtract solid `b` from `a` (the exact-kernel form of a
/// revolve/extrude cut against an already-built tool solid). Inverts `b`'s faces and
/// intersects, exactly like [`cut_tol`] does with its freshly-built prism tool.
pub fn difference(a: &KSolid, b: &KSolid) -> Option<KSolid> {
    difference_tol(a, b, TOL)
}

/// Boolean difference at a caller-chosen tolerance (see [`union_tol`] for why that matters).
pub fn difference_tol(a: &KSolid, b: &KSolid, tol: f64) -> Option<KSolid> {
    let mut tool = b.0.clone();
    guard(move || {
        tool.not(); // invert all faces → complement region, so AND becomes a subtraction
        truck_shapeops::and(&a.0, &tool, tol)
    })
    .map(KSolid)
}

/// Run a kernel operation that may *panic* (truck asserts internally — e.g. "this
/// wire is not simple" on degenerate input) and turn that panic into `None` so a
/// single bad contour can't bring the whole app down.
fn guard<T>(f: impl FnOnce() -> Option<T>) -> Option<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(None)
}

/// True if the kernel can actually triangulate the solid. A boolean on NURBS
/// (exact-arc) surfaces can "succeed" yet return a shape whose triangulation
/// panics or comes out empty — such a result is unusable downstream (it renders
/// as nothing), so callers must treat it as a failed operation and fall back.
pub fn solid_renderable(s: &KSolid) -> bool {
    guard(|| {
        let mut poly = s.0.triangulation(0.1).to_polygon();
        poly.triangulate();
        Some(poly.faces().tri_faces().len() >= 4)
    })
    .unwrap_or(false)
}

/// Sanitize a polygon loop so truck accepts it as a *simple* wire: drop
/// (near-)duplicate consecutive vertices and antenna spikes (a vertex whose
/// neighbours coincide — a zero-area backtrack). Both produce zero-length or
/// self-touching wire edges, which make the kernel panic.
fn clean_loop(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    const TOL: f64 = 1e-4;
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < TOL && (a[1] - b[1]).abs() < TOL;

    // Pass 1: remove consecutive duplicates (and the wrap-around duplicate).
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for &p in pts {
        if out.last().is_none_or(|&q| !close(p, q)) {
            out.push(p);
        }
    }
    while out.len() >= 2 && close(out[0], *out.last().unwrap()) {
        out.pop();
    }

    // Pass 2: remove antenna spikes (prev ≈ next), restarting after each removal.
    let mut changed = true;
    while changed && out.len() >= 3 {
        changed = false;
        let m = out.len();
        for i in 0..m {
            if close(out[(i + m - 1) % m], out[(i + 1) % m]) {
                let mut idx = [i, (i + 1) % m];
                idx.sort_unstable();
                out.remove(idx[1]);
                out.remove(idx[0]);
                changed = true;
                break;
            }
        }
    }
    out
}

/// Convert a polyline loop with [`ArcSpan`] annotations into boundary segments:
/// each span collapses to one exact `Arc`, everything else stays `Line`s. `None`
/// if the annotations don't fit the loop (malformed/overlapping spans) — the
/// caller then falls back to the all-lines path.
fn ring_to_segs(pts: &[[f64; 2]], arcs: &[ArcSpan]) -> Option<Vec<PathSeg>> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    // Which span (if any) owns each edge.
    let mut owner = vec![usize::MAX; n];
    for (si, s) in arcs.iter().enumerate() {
        if s.count == 0 || s.count > n || s.first_edge >= n {
            return None;
        }
        for k in 0..s.count {
            let e = (s.first_edge + k) % n;
            if owner[e] != usize::MAX {
                return None; // overlapping spans — shouldn't happen
            }
            owner[e] = si;
        }
    }

    // A loop that is entirely one circle: two half arcs (a wire can't be a
    // single closed edge).
    if arcs.len() == 1 && arcs[0].count == n {
        if n < 4 {
            return None;
        }
        let h = n / 2;
        return Some(vec![
            PathSeg::Arc { a: pts[0], b: pts[h], transit: pts[h / 2] },
            PathSeg::Arc { a: pts[h], b: pts[0], transit: pts[h + (n - h) / 2] },
        ]);
    }

    // Walk the loop starting at a run boundary so no span is cut in half. A
    // loop with no boundary at all is all lines (a full-cover single span was
    // handled above), so any start works.
    let start = (0..n).find(|&i| owner[i] != owner[(i + n - 1) % n]).unwrap_or(0);
    let dist = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let mut segs = Vec::new();
    let mut i = 0usize;
    while i < n {
        let e = (start + i) % n;
        if owner[e] == usize::MAX {
            let (a, b) = (pts[e], pts[(e + 1) % n]);
            if dist(a, b) > 1e-9 {
                segs.push(PathSeg::Line(a, b));
            }
            i += 1;
            continue;
        }
        let s = &arcs[owner[e]];
        if e != s.first_edge {
            return None; // walk desynced from the span table — bail to lines
        }
        let a = pts[s.first_edge];
        let b = pts[(s.first_edge + s.count) % n];
        if dist(a, b) < 1e-6 {
            return None; // near-closed partial arc — ambiguous, use lines
        }
        let transit = if s.count >= 2 {
            // An interior tessellation vertex — exactly on the source circle.
            pts[(s.first_edge + s.count / 2) % n]
        } else {
            // Single edge: project the chord midpoint out onto the circle.
            let m = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
            let (dx, dy) = (m[0] - s.center[0], m[1] - s.center[1]);
            let d = (dx * dx + dy * dy).sqrt();
            if d < 1e-9 {
                return None;
            }
            [s.center[0] + dx / d * s.radius, s.center[1] + dy / d * s.radius]
        };
        // Nearly-collinear a/transit/b would make the arc constructor blow up —
        // such a sliver of circle is indistinguishable from its chord anyway.
        let sagitta = {
            let (ux, uy) = (b[0] - a[0], b[1] - a[1]);
            let l = (ux * ux + uy * uy).sqrt().max(1e-12);
            ((transit[0] - a[0]) * uy - (transit[1] - a[1]) * ux).abs() / l
        };
        if sagitta < 1e-7 {
            segs.push(PathSeg::Line(a, b));
        } else {
            segs.push(PathSeg::Arc { a, b, transit });
        }
        i += s.count;
    }
    (segs.len() >= 2).then_some(segs)
}

/// Reverse a boundary path in place (opposite winding): segment order flips and
/// each segment swaps its endpoints; arc transit points are direction-free.
fn reverse_segs(segs: &mut [PathSeg]) {
    segs.reverse();
    for s in segs.iter_mut() {
        match s {
            PathSeg::Line(a, b) => std::mem::swap(a, b),
            PathSeg::Arc { a, b, .. } => std::mem::swap(a, b),
        }
    }
}

/// The polyline vertices of a seg path's start points (used for winding tests).
fn seg_starts(segs: &[PathSeg]) -> Vec<[f64; 2]> {
    segs.iter()
        .map(|s| match s {
            PathSeg::Line(a, _) => *a,
            PathSeg::Arc { a, .. } => *a,
        })
        .collect()
}

/// Boolean cut: subtract a swept region from `base`.
///
/// `distance` is *signed*: positive sweeps the tool along the plane normal,
/// negative sweeps against it. The caller picks the sign so the tool extends
/// *into* the material. Either way the tool overshoots both caps so they are
/// never coplanar with the body's faces (the classic B-rep boolean failure), and
/// the tool is inverted so `base ∩ ¬tool == base − tool`.
pub fn cut(
    base: &KSolid,
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
) -> Option<KSolid> {
    cut_tol(base, outer, holes, basis, distance, 0.0, TOL)
}

/// Boolean cut at a caller-chosen tolerance — the cut equivalent of [`union_tol`],
/// so a cut whose wall coincides with an existing face can be completed with a
/// sub-micron tool nudge instead of failing.
pub fn cut_tol(
    base: &KSolid,
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
    tol: f64,
) -> Option<KSolid> {
    let depth = distance.abs();
    if depth < 1e-9 {
        return None;
    }
    let eps = 0.05 + depth * 0.02;
    // `back` (Direction 2) extends the cut tool the opposite way from `distance`.
    let b = back.max(0.0);
    let (start_offset, length) = if distance >= 0.0 {
        (-(eps + b), depth + 2.0 * eps + b)
    } else {
        (-(depth + eps), depth + 2.0 * eps + b)
    };
    let mut tool = build_solid(outer, holes, basis, start_offset, length)?;
    guard(move || {
        tool.not(); // invert all faces → complement region
        truck_shapeops::and(&base.0, &tool, tol)
    })
    .map(KSolid)
}

/// [`cut_tol`] with exact-arc annotations: the cut tool's arc runs become true
/// cylindrical faces (an exact drilled hole instead of a faceted one). Falls
/// back to the all-lines tool if the arc path fails.
pub fn cut_tol_arcs(
    base: &KSolid,
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
    tol: f64,
) -> Option<KSolid> {
    let depth = distance.abs();
    if depth < 1e-9 {
        return None;
    }
    let eps = 0.05 + depth * 0.02;
    let b = back.max(0.0);
    let (start_offset, length) = if distance >= 0.0 {
        (-(eps + b), depth + 2.0 * eps + b)
    } else {
        (-(depth + eps), depth + 2.0 * eps + b)
    };
    let mut tool = build_solid_arcs(outer, holes, outer_arcs, hole_arcs, basis, start_offset, length)?;
    guard(move || {
        tool.not(); // invert all faces → complement region
        truck_shapeops::and(&base.0, &tool, tol)
    })
    .map(KSolid)
    // NURBS booleans can return an untessellatable shape — count that as failure
    // so the caller's fallback ladder (nudge/tolerance/faceted tool) kicks in.
    .filter(solid_renderable)
}

/// Signed area of a 2D polygon (positive ⇒ counter-clockwise).
fn signed_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Return the loop wound to the requested orientation (ccw = true ⇒ CCW).
fn wound(pts: &[[f64; 2]], ccw: bool) -> Vec<[f64; 2]> {
    let mut v = pts.to_vec();
    if (signed_area(pts) > 0.0) != ccw {
        v.reverse();
    }
    v
}

/// Tessellate a solid into a flat-shaded mesh plus its classified edges. Edges
/// sharper than `SHARP_DEG` (real corners) are "sharp"; gentler ones (the facet
/// lines of a curved surface, or a tangent blend) are "tangent".
pub fn tessellate(solid: &KSolid, tol: f64) -> Tessellation {
    const SHARP_DEG: f64 = 35.0;
    // truck's triangulation can panic on awkward geometry; never let that crash the
    // app — fall back to an empty tessellation (the booleans are guarded too).
    guard(|| {
        let mut poly = solid.0.triangulation(tol).to_polygon();
        poly.triangulate();
        let mesh = polymesh_to_trimesh(&poly);
        let (edges, tangent_edges) = feature_edges(&mesh, SHARP_DEG);
        Some(Tessellation { mesh, edges, tangent_edges })
    })
    .unwrap_or(Tessellation { mesh: TriMesh::default(), edges: Vec::new(), tangent_edges: Vec::new() })
}

/// Build an extruded prism (a region swept by `normal*length`, starting at
/// `normal*start_offset`) as a **triangle mesh** — the boss/cut "tool" for the
/// robust mesh-boolean fallback. `None` if the region is degenerate.
pub fn extrude_tool_mesh(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<TriMesh> {
    // Expected volume from the profile: the cheap ground truth that catches a bad
    // triangulation regardless of HOW it went wrong.
    let expect = (poly_area_2d(outer) - holes.iter().map(|h| poly_area_2d(h)).sum::<f64>()).abs() * length.abs();
    let truck = build_solid(outer, holes, basis, start_offset, length).and_then(|solid| {
        guard(|| {
            let mut poly = solid.triangulation(TOL).to_polygon();
            poly.triangulate();
            Some(polymesh_to_trimesh(&poly))
        })
    });
    // truck's triangulation of a complex arrangement region (hundreds of vertices weaving
    // through skinny channels) can silently produce a prism that COVERS GROUND OUTSIDE the
    // profile — excut.hcad's bottom cut ate the part's rim that way. Validate the volume
    // against the profile area; on disagreement (or truck failure) build the prism
    // DIRECTLY: ear-clipped caps + side walls, watertight by construction.
    if let Some(m) = truck {
        if expect < 1e-9 || (signed_mesh_volume(&m).abs() - expect).abs() <= expect * 0.02 {
            return Some(m);
        }
    }
    direct_prism_mesh(outer, holes, basis, start_offset, length)
}

/// One boundary edge of a prism profile and the surface its wall lies on.
struct Wall {
    a: [f64; 2],
    b: [f64; 2],
    surf: u32,
}

/// Work out every surface a straight prism over this profile has, and which profile edge each
/// wall belongs to: the two cap planes, one cylinder per annotated arc run, and one plane per
/// remaining edge.
///
/// Nothing here is fitted. The sketch already knows a bore is a circle — [`ArcSpan`] says which
/// run of edges lies on it — so the cylinder is *read off* the profile, not recovered from
/// triangles afterwards. That is the whole point of carrying surfaces instead of guessing them.
///
/// The one thing not taken at face value is the arc's radius: a cut profile can be nudged off
/// its nominal circle before it is built (`clear_coincident_cut_walls` widens a wall that would
/// otherwise land exactly on a face), and a cylinder whose radius disagreed with its own
/// triangles would be worse than no cylinder at all. Widening is radial, so the centre survives
/// it and the radius is re-measured from the points actually being built.
fn prism_surface_plan(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    w_lo: f64,
    w_hi: f64,
) -> (Vec<Surf>, Vec<Wall>) {
    let o = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    let to3 = |p: [f64; 2], w: f64| {
        let q = o + u * p[0] + v * p[1] + n * w;
        [q.x, q.y, q.z]
    };
    let axis = [n.x, n.y, n.z];
    // Slots 0 and 1 are always the two caps, so the tagger can name them without a search.
    let mut surfaces = vec![
        Surf::Plane { origin: to3([0.0, 0.0], w_hi), normal: axis },
        Surf::Plane { origin: to3([0.0, 0.0], w_lo), normal: [-axis[0], -axis[1], -axis[2]] },
    ];
    let mut walls: Vec<Wall> = Vec::new();
    let loops = std::iter::once((outer, outer_arcs)).chain(
        holes
            .iter()
            .enumerate()
            .map(|(i, h)| (h.as_slice(), hole_arcs.get(i).map(|s| s.as_slice()).unwrap_or(&[]))),
    );
    for (pts, spans) in loops {
        let m = pts.len();
        if m < 3 {
            continue;
        }
        // Which edges an arc run claims, and the slot of the cylinder they share.
        let mut claimed: Vec<Option<u32>> = vec![None; m];
        for s in spans {
            if s.count == 0 || s.count > m {
                continue;
            }
            let edges: Vec<usize> = (0..s.count).map(|t| (s.first_edge + t) % m).collect();
            // Re-measure the radius from the points the run actually covers (its edges' two
            // endpoints), so a widened profile still gets a cylinder its own walls sit on.
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for &e in &edges {
                for p in [pts[e], pts[(e + 1) % m]] {
                    sum += ((p[0] - s.center[0]).powi(2) + (p[1] - s.center[1]).powi(2)).sqrt();
                    cnt += 1.0;
                }
            }
            if cnt == 0.0 || sum / cnt < 1.0e-9 {
                continue;
            }
            surfaces.push(Surf::Cylinder { origin: to3(s.center, w_lo), axis, radius: sum / cnt });
            let slot = (surfaces.len() - 1) as u32;
            for e in edges {
                claimed[e] = Some(slot);
            }
        }
        for (k, claim) in claimed.iter().enumerate() {
            let (a, b) = (pts[k], pts[(k + 1) % m]);
            let surf = match *claim {
                Some(s) => s,
                None => {
                    let d = [b[0] - a[0], b[1] - a[1]];
                    let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
                    if l < 1.0e-12 {
                        continue; // a zero-length edge has no wall
                    }
                    // Across the edge and across the sweep. Which way it points doesn't matter:
                    // the tagger matches a triangle to a wall by WHERE it is, not by its facing.
                    let nrm2 = [d[1] / l, -d[0] / l];
                    let nrm = u * nrm2[0] + v * nrm2[1];
                    surfaces.push(Surf::Plane { origin: to3(a, w_lo), normal: [nrm.x, nrm.y, nrm.z] });
                    (surfaces.len() - 1) as u32
                }
            };
            walls.push(Wall { a, b, surf });
        }
    }
    (surfaces, walls)
}

/// Record on every triangle of a built prism which surface of [`prism_surface_plan`] it lies on.
///
/// Each triangle is placed by POSITION, not by fitting anything: projected into the sketch plane,
/// a cap triangle sits at one of the two sweep ends, and a wall triangle's centroid falls on
/// exactly one profile edge — whichever built the mesh, and however finely it chose to subdivide.
///
/// Position is what makes an annotated arc trustworthy. Geometry alone cannot tell one facet of a
/// polygonal circle from a flat chord across the same circle (both have their endpoints at the
/// radius, and both face along it), so a radius test would tag the flat of a D-shaped bore as part
/// of the bore. Which edges belong to the arc is something only the sketch knows, and it said.
fn tag_prism_walls(mesh: &mut TriMesh, surfaces: Vec<Surf>, walls: &[Wall], basis: &PlaneBasis, w_lo: f64, w_hi: f64) {
    mesh.clear_tags();
    let ntri = mesh.indices.len() / 3;
    if ntri == 0 || walls.is_empty() {
        return;
    }
    let o = Point3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    // Positions are f32; a point 100mm out carries ~1e-5 of rounding before anything else. Scale
    // the tolerance to the part so a big model isn't judged by a small one's precision.
    let span = walls
        .iter()
        .flat_map(|w| [w.a, w.b])
        .fold(0.0f64, |acc, p| acc.max(p[0].abs()).max(p[1].abs()))
        .max((w_hi - w_lo).abs());
    let tol = (span * 1.0e-5).max(1.0e-4);
    let mut tri_surf = vec![NO_SURF; ntri];
    let mut used: Vec<u32> = vec![NO_SURF; surfaces.len()];
    let mut kept: Vec<Surf> = Vec::new();
    let mut claim = |tri_surf: &mut Vec<u32>, kept: &mut Vec<Surf>, t: usize, slot: u32| {
        let s = slot as usize;
        if used[s] == NO_SURF {
            kept.push(surfaces[s]);
            used[s] = (kept.len() - 1) as u32;
        }
        tri_surf[t] = used[s];
    };
    for t in 0..ntri {
        let p = |i: usize| {
            let q = mesh.positions[mesh.indices[t * 3 + i] as usize];
            Point3::new(q[0] as f64, q[1] as f64, q[2] as f64)
        };
        let (a, b, c) = (p(0), p(1), p(2));
        let fnrm = (b - a).cross(c - a);
        let fl = fnrm.magnitude();
        if fl < 1.0e-16 {
            continue; // a needle has no meaningful facing; leave it for nobody
        }
        let along = fnrm.dot(n) / fl;
        let cen = ((a - o) + (b - o) + (c - o)) / 3.0;
        let w = cen.dot(n);
        if along.abs() > 0.99 {
            // A cap: flat to the sweep and at one of its two ends.
            if (w - w_hi).abs() < tol {
                claim(&mut tri_surf, &mut kept, t, 0);
            } else if (w - w_lo).abs() < tol {
                claim(&mut tri_surf, &mut kept, t, 1);
            }
            continue;
        }
        // A wall: its centroid projects onto the profile edge that swept it.
        let q = [cen.dot(u), cen.dot(v)];
        let mut best = (f64::MAX, NO_SURF);
        for wall in walls {
            let d = [wall.b[0] - wall.a[0], wall.b[1] - wall.a[1]];
            let ll = d[0] * d[0] + d[1] * d[1];
            let s = if ll > 1.0e-18 {
                (((q[0] - wall.a[0]) * d[0] + (q[1] - wall.a[1]) * d[1]) / ll).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (dx, dy) = (q[0] - (wall.a[0] + d[0] * s), q[1] - (wall.a[1] + d[1] * s));
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best.0 {
                best = (dist, wall.surf);
            }
        }
        if best.0 < tol && best.1 != NO_SURF {
            claim(&mut tri_surf, &mut kept, t, best.1);
        }
    }
    if kept.is_empty() {
        return;
    }
    mesh.surfaces = kept;
    mesh.tri_surf = tri_surf;
}

/// [`extrude_tool_mesh`], with the profile's exact-arc annotations carried onto the result: a bore
/// comes back recorded as ONE cylinder, not a ring of unrelated flat strips.
///
/// Tagging has to happen HERE rather than inside a builder. On real parts every prism comes back
/// from truck's triangulation — measured across usercylinder, blocker and motormount, the direct
/// builder ran for none of their 83 prisms — so tags written inside `direct_prism_mesh` would
/// never be seen. Placing triangles afterwards works whichever builder answered.
///
/// Across the saved corpus that is 36 of 50 buildable parts carrying at least one real cylinder
/// through to the finished body (`diag_surface_tags`), with every part's volume unchanged and
/// every rebuild still deterministic. Nothing reads the cylinders yet — the exporter is next.
pub fn extrude_tool_mesh_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<TriMesh> {
    let mut m = extrude_tool_mesh(outer, holes, basis, start_offset, length)?;
    let (w_lo, w_hi) = (
        start_offset.min(start_offset + length),
        start_offset.max(start_offset + length),
    );
    let (surfaces, walls) = prism_surface_plan(outer, holes, outer_arcs, hole_arcs, basis, w_lo, w_hi);
    tag_prism_walls(&mut m, surfaces, &walls, basis, w_lo, w_hi);
    Some(m)
}

/// Shoelace area of a 2D polygon (absolute).
fn poly_area_2d(l: &[[f64; 2]]) -> f64 {
    let mut a = 0.0;
    for k in 0..l.len() {
        let p = l[k];
        let q = l[(k + 1) % l.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    (a * 0.5).abs()
}

/// Drop triangles that name the same vertex twice, returning how many went.
///
/// Always safe, by construction rather than by measurement: such a triangle's only two real
/// edges are the SAME undirected edge traversed both ways *within itself*, so removing it takes
/// both away together and can never leave a neighbour's edge unmatched. That is not true of a
/// triangle with three distinct-but-collinear points — those carry edges shared with real
/// neighbours, and removing one opens a hole.
///
/// This is the bulk of what the bevel builder leaves behind: it stitches sharp edges with
/// one-segment strips, and wherever the two ends already agree, half the strip collapses to
/// exactly this shape.
pub fn drop_duplicate_vertex_triangles(mesh: &mut TriMesh) -> usize {
    retain_triangles(mesh, |m, t| {
        let q = |i: u32| m.positions[i as usize];
        let same = |a: [f32; 3], b: [f32; 3]| {
            (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9 && (a[2] - b[2]).abs() < 1e-9
        };
        let (a, b, c) = (q(t[0]), q(t[1]), q(t[2]));
        !(t[0] == t[1] || t[1] == t[2] || t[0] == t[2] || same(a, b) || same(b, c) || same(a, c))
    })
}

/// Keep the triangles `pred` accepts, dropping the rest, and return how many went.
///
/// Carries the surface tags along with the triangles they describe. Dropping triangles out from
/// under the tag array is not a small error: every tag after the first gap names a different
/// triangle's surface, so an exporter would confidently put a bore's cylinder somewhere on a flat
/// face. [`TriMesh::surf_of`] catches a mismatched array and stops trusting it, but that throws
/// away every tag on the mesh — keeping them in step keeps them usable.
fn retain_triangles(mesh: &mut TriMesh, pred: impl Fn(&TriMesh, &[u32]) -> bool) -> usize {
    let before = mesh.indices.len() / 3;
    let tagged = mesh.tri_surf.len() == before;
    let mut keep: Vec<u32> = Vec::with_capacity(mesh.indices.len());
    let mut tags: Vec<u32> = Vec::with_capacity(mesh.tri_surf.len());
    for (t, v) in mesh.indices.chunks_exact(3).enumerate() {
        if pred(mesh, v) {
            keep.extend_from_slice(v);
            if tagged {
                tags.push(mesh.tri_surf[t]);
            }
        }
    }
    mesh.indices = keep;
    if tagged {
        mesh.tri_surf = tags;
    } else {
        // The array was already out of step with the triangles, so there is nothing to carry.
        mesh.clear_tags();
    }
    before - mesh.indices.len() / 3
}

/// Drop triangles with exactly zero area, returning how many went.
///
/// A zero-area triangle has three collinear vertices, so it covers no surface: it contributes
/// nothing to the solid but does real harm downstream. It has no usable normal, it bloats STL
/// and STEP exports, and it gives the boolean kernel edges to trip over.
///
/// The bevel and thread builders emit them where their patches meet. Removing them is safe by
/// measurement, not by argument: across the models that carry them the mesh stays manifold, no
/// boundary edge is opened, the volume is unchanged to six decimals, and the count of edges
/// shared by more than two faces drops sharply (612 to 363 on the worst case).
pub fn drop_degenerate_triangles(mesh: &mut TriMesh) -> usize {
    retain_triangles(mesh, |m, t| {
        let q = |i: u32| {
            let p = m.positions[i as usize];
            [p[0] as f64, p[1] as f64, p[2] as f64]
        };
        let (a, b, c) = (q(t[0]), q(t[1]), q(t[2]));
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() > 1e-12
    })
}

/// Signed volume of a triangle mesh (divergence theorem).
pub fn signed_mesh_volume(m: &TriMesh) -> f64 {
    let mut v = 0.0;
    for t in m.indices.chunks_exact(3) {
        let p = |i: u32| {
            let q = m.positions[i as usize];
            [q[0] as f64, q[1] as f64, q[2] as f64]
        };
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        v += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    v / 6.0
}

/// Ear-clip triangulation of a simple polygon (indices into `pts`). Robust to collinear
/// runs; bails (partial fan) only if no ear exists at all (degenerate input).
pub(crate) fn earcut_simple(pts: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    // Ensure CCW.
    let mut a2 = 0.0;
    for k in 0..n {
        let (p, q) = (pts[k], pts[(k + 1) % n]);
        a2 += p[0] * q[1] - q[0] * p[1];
    }
    if a2 < 0.0 {
        idx.reverse();
    }
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    // A point COINCIDENT with an ear corner never blocks it — hole bridging duplicates
    // vertices exactly, and counting a duplicate as "inside" would block every ear through
    // it (the clipper then degrades to overlapping fallback junk).
    let same = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() < 1e-12 && (p[1] - q[1]).abs() < 1e-12;
    let in_tri = |p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        if same(p, a) || same(p, b) || same(p, c) {
            return false;
        }
        cross(a, b, p) >= -1e-12 && cross(b, c, p) >= -1e-12 && cross(c, a, p) >= -1e-12
    };
    let mut tris = Vec::new();
    let mut guard = 0usize;
    while idx.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (i0, i1, i2) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            let (a, b, c) = (pts[i0], pts[i1], pts[i2]);
            if cross(a, b, c) <= 1e-12 {
                continue; // reflex or collinear — not an ear
            }
            let blocked = idx.iter().any(|&j| j != i0 && j != i1 && j != i2 && in_tri(pts[j], a, b, c));
            if !blocked {
                tris.push([i0, i1, i2]);
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Degenerate leftovers: clip the least-bad corner so we always terminate.
            let mut best = (0usize, f64::MIN);
            for k in 0..idx.len() {
                let m2 = idx.len();
                let c = cross(pts[idx[(k + m2 - 1) % m2]], pts[idx[k]], pts[idx[(k + 1) % m2]]);
                if c > best.1 {
                    best = (k, c);
                }
            }
            let m2 = idx.len();
            tris.push([idx[(best.0 + m2 - 1) % m2], idx[best.0], idx[(best.0 + 1) % m2]]);
            idx.remove(best.0);
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

/// Merge holes into the outer loop with bridge edges (rightmost-vertex ray casting), giving
/// one simple polygon ear-clip can chew.
pub(crate) fn bridge_holes(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[f64; 2]> {
    // Outer CCW, holes CW.
    let ensure = |l: &[[f64; 2]], ccw: bool| -> Vec<[f64; 2]> {
        let mut a2 = 0.0;
        for k in 0..l.len() {
            let (p, q) = (l[k], l[(k + 1) % l.len()]);
            a2 += p[0] * q[1] - q[0] * p[1];
        }
        let mut v = l.to_vec();
        if (a2 > 0.0) != ccw {
            v.reverse();
        }
        v
    };
    let mut poly = ensure(outer, true);
    // Biggest-x holes first (standard hole-bridging order).
    let mut hs: Vec<Vec<[f64; 2]>> = holes.iter().filter(|h| h.len() >= 3).map(|h| ensure(h, false)).collect();
    hs.sort_by(|a, b| {
        let mx = |l: &Vec<[f64; 2]>| l.iter().map(|p| p[0]).fold(f64::MIN, f64::max);
        mx(b).total_cmp(&mx(a))
    });
    for h in hs {
        // The hole's rightmost vertex…
        let hk = (0..h.len()).max_by(|&i, &j| h[i][0].total_cmp(&h[j][0])).unwrap();
        let hp = h[hk];
        // …bridged to the visible poly vertex: nearest poly vertex to the RIGHT whose
        // connecting segment crosses no poly edge (fallback: plain nearest).
        let seg_hits = |a: [f64; 2], b: [f64; 2]| -> bool {
            let m = poly.len();
            for k in 0..m {
                let (c, d) = (poly[k], poly[(k + 1) % m]);
                if (c == a && d == b) || (c == b && d == a) {
                    continue;
                }
                let den = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0]);
                if den.abs() < 1e-15 {
                    continue;
                }
                let t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / den;
                let u = ((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / den;
                if t > 1e-9 && t < 1.0 - 1e-9 && u > 1e-9 && u < 1.0 - 1e-9 {
                    return true;
                }
            }
            false
        };
        let mut order: Vec<usize> = (0..poly.len()).collect();
        order.sort_by(|&i, &j| {
            let di = (poly[i][0] - hp[0]).powi(2) + (poly[i][1] - hp[1]).powi(2);
            let dj = (poly[j][0] - hp[0]).powi(2) + (poly[j][1] - hp[1]).powi(2);
            di.total_cmp(&dj)
        });
        let pk = order
            .iter()
            .copied()
            .find(|&i| poly[i][0] >= hp[0] - 1e-9 && !seg_hits(hp, poly[i]))
            .or_else(|| order.iter().copied().find(|&i| !seg_hits(hp, poly[i])))
            .unwrap_or(order[0]);
        // Splice: …poly[pk], hole[hk..], hole[..hk], hole[hk], poly[pk]…
        let mut merged: Vec<[f64; 2]> = Vec::with_capacity(poly.len() + h.len() + 2);
        merged.extend_from_slice(&poly[..=pk]);
        for off in 0..=h.len() {
            merged.push(h[(hk + off) % h.len()]);
        }
        merged.push(poly[pk]);
        merged.extend_from_slice(&poly[pk + 1..]);
        poly = merged;
    }
    poly
}

/// Build the extrude prism DIRECTLY as a watertight triangle mesh: ear-clipped caps (holes
/// bridged in) + side-wall quads per loop. The robust fallback when the exact kernel's
/// triangulation misbehaves on a complex profile.
pub fn direct_prism_mesh(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<TriMesh> {
    if outer.len() < 3 || length.abs() < 1e-9 {
        return None;
    }
    let o = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let nrm = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    let (w0, w1) = (start_offset, start_offset + length);
    let to3 = |p: [f64; 2], w: f64| {
        let q = o + u * p[0] + v * p[1] + nrm * w;
        [q.x, q.y, q.z]
    };
    // Normalize winding ONCE (outer CCW, holes CW) and use the SAME loops for both the caps
    // and the side walls. `bridge_holes` normalizes internally, so the caps were already
    // right, but the walls used to iterate the RAW loops — a hole arriving CCW (same as the
    // outer) then produced inward-flipped wall quads that a single global volume flip can't
    // reconcile, so Manifold read the tool as non-manifold and dropped to the lossy BSP.
    let signed = |l: &[[f64; 2]]| -> f64 {
        let mut a = 0.0;
        for k in 0..l.len() {
            let (p, q) = (l[k], l[(k + 1) % l.len()]);
            a += p[0] * q[1] - q[0] * p[1];
        }
        a
    };
    let mut outer_n = outer.to_vec();
    if signed(&outer_n) < 0.0 {
        outer_n.reverse();
    }
    let holes_n: Vec<Vec<[f64; 2]>> = holes
        .iter()
        .filter(|h| h.len() >= 3)
        .map(|h| {
            let mut hv = h.clone();
            if signed(&hv) > 0.0 {
                hv.reverse(); // holes wind CW
            }
            hv
        })
        .collect();
    let mut mesh = TriMesh::default();
    // Each face records the surface it lies on as it is built — see `Surf`. A prism knows all of
    // them exactly: two cap planes and one plane per profile edge. The normal is read back off the
    // triangle just pushed rather than derived from the winding, so it is the outward direction
    // the mesh itself ended up with.
    fn tag_last(mesh: &mut TriMesh, first: usize, origin: [f64; 3]) {
        let n = mesh.normals[first * 3];
        mesh.tag_from(first, Surf::Plane { origin, normal: [n[0] as f64, n[1] as f64, n[2] as f64] });
    }
    // Caps: triangulate the bridged polygon once, emit each end as a CONTIGUOUS block so it can
    // carry one tag (they used to be interleaved, a top and a bottom per earcut triangle).
    let cap_poly = bridge_holes(&outer_n, &holes_n);
    let fan = earcut_simple(&cap_poly);
    let first_top = mesh.indices.len() / 3;
    for t in &fan {
        let (a, b, c) = (cap_poly[t[0]], cap_poly[t[1]], cap_poly[t[2]]);
        push_tri(&mut mesh, to3(a, w1), to3(b, w1), to3(c, w1)); // top (+normal side)
    }
    if mesh.indices.len() / 3 > first_top {
        tag_last(&mut mesh, first_top, to3([0.0, 0.0], w1));
    }
    let first_bot = mesh.indices.len() / 3;
    for t in &fan {
        let (a, b, c) = (cap_poly[t[0]], cap_poly[t[1]], cap_poly[t[2]]);
        push_tri(&mut mesh, to3(a, w0), to3(c, w0), to3(b, w0)); // bottom (reversed)
    }
    if mesh.indices.len() / 3 > first_bot {
        tag_last(&mut mesh, first_bot, to3([0.0, 0.0], w0));
    }
    // Side walls: every (winding-normalized) loop contributes quads between the two levels.
    for l in std::iter::once(&outer_n).chain(holes_n.iter()) {
        let m = l.len();
        if m < 3 {
            continue;
        }
        for k in 0..m {
            let (a, b) = (l[k], l[(k + 1) % m]);
            if (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12 {
                continue;
            }
            let first = mesh.indices.len() / 3;
            push_tri(&mut mesh, to3(a, w0), to3(b, w0), to3(b, w1));
            push_tri(&mut mesh, to3(a, w0), to3(b, w1), to3(a, w1));
            tag_last(&mut mesh, first, to3(a, w0));
        }
    }
    if mesh.indices.is_empty() {
        return None;
    }
    // Orient outward (the loop windings vary — the signed volume settles it).
    if signed_mesh_volume(&mesh) < 0.0 {
        for t in mesh.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        for n2 in &mut mesh.normals {
            *n2 = [-n2[0], -n2[1], -n2[2]];
        }
        // The recorded surfaces face outward too, so they turn with the mesh.
        for s in &mut mesh.surfaces {
            let Surf::Plane { normal, .. } = s else { continue };
            *normal = [-normal[0], -normal[1], -normal[2]];
        }
    }
    Some(mesh)
}

/// One cross-section of a layered solid: the profile rotated by `rot` radians and scaled by
/// `scale` about the plane origin, placed `w` along the plane normal.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub w: f64,
    pub rot: f64,
    pub scale: f64,
}

/// Build a solid from ONE 2D profile repeated at every layer, each copy rotated and scaled
/// about the origin. Unlike `loft_mesh` this keeps exact point correspondence between layers
/// (no resampling), so a thousand-point involute survives intact — which is what makes it the
/// right primitive for helical, herringbone and bevel gears. `None` if the input is degenerate.
pub fn layered_profile_mesh(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>], basis: &PlaneBasis, layers: &[Layer]) -> Option<TriMesh> {
    if outer.len() < 3 || layers.len() < 2 {
        return None;
    }
    if layers.iter().any(|l| l.scale.abs() < 1e-9) {
        return None; // a layer collapsed to a point: the taper ran past the apex
    }
    let o = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let nrm = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    let place = |p: [f64; 2], l: &Layer| {
        let (c, s) = (l.rot.cos(), l.rot.sin());
        let (x, y) = ((p[0] * c - p[1] * s) * l.scale, (p[0] * s + p[1] * c) * l.scale);
        let q = o + u * x + v * y + nrm * l.w;
        [q.x, q.y, q.z]
    };
    // Same winding discipline as the straight prism: outer CCW, holes CW, so the wall quads
    // all face outward and the caps agree with them.
    let signed = |l: &[[f64; 2]]| {
        let mut a = 0.0;
        for k in 0..l.len() {
            let (p, q) = (l[k], l[(k + 1) % l.len()]);
            a += p[0] * q[1] - q[0] * p[1];
        }
        a
    };
    let mut outer_n = outer.to_vec();
    if signed(&outer_n) < 0.0 {
        outer_n.reverse();
    }
    let holes_n: Vec<Vec<[f64; 2]>> = holes
        .iter()
        .filter(|h| h.len() >= 3)
        .map(|h| {
            let mut hv = h.to_vec();
            if signed(&hv) > 0.0 {
                hv.reverse();
            }
            hv
        })
        .collect();

    let mut mesh = TriMesh::default();
    // Caps. Rotation and uniform scale are a similarity, so the bridged polygon stays simple
    // at every layer and one triangulation serves both ends.
    let cap_poly = bridge_holes(&outer_n, &holes_n);
    let (first, last) = (layers[0], layers[layers.len() - 1]);
    for t in earcut_simple(&cap_poly) {
        let (a, b, c) = (cap_poly[t[0]], cap_poly[t[1]], cap_poly[t[2]]);
        push_tri(&mut mesh, place(a, &last), place(b, &last), place(c, &last));
        push_tri(&mut mesh, place(a, &first), place(c, &first), place(b, &first));
    }
    // Walls: a quad band per loop between every consecutive pair of layers.
    for loop_ in std::iter::once(&outer_n).chain(holes_n.iter()) {
        let m = loop_.len();
        for pair in layers.windows(2) {
            let (l0, l1) = (pair[0], pair[1]);
            for k in 0..m {
                let (a, b) = (loop_[k], loop_[(k + 1) % m]);
                if (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12 {
                    continue;
                }
                push_tri(&mut mesh, place(a, &l0), place(b, &l0), place(b, &l1));
                push_tri(&mut mesh, place(a, &l0), place(b, &l1), place(a, &l1));
            }
        }
    }
    if mesh.indices.is_empty() {
        return None;
    }
    if signed_mesh_volume(&mesh) < 0.0 {
        for t in mesh.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        for n2 in &mut mesh.normals {
            *n2 = [-n2[0], -n2[1], -n2[2]];
        }
    }
    Some(mesh)
}

/// Lofts whose profiles disagreed on hole count, so their holes were not skinned. Drained by
/// the app to warn, in the same way boolean fallbacks are.
static LOFT_HOLE_MISMATCHES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Take and clear the count of lofts that had to drop their holes.
pub fn take_loft_hole_mismatch_count() -> u32 {
    LOFT_HOLE_MISMATCHES.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Revolve a closed region (outer loop + optional holes, in plane-local uv) around an axis
/// line — the line through `axis_pt` with direction `axis_dir`, both in the same uv plane — by
/// `angle` radians, into a solid of revolution. `None` if degenerate. The profile must lie to
/// one side of the axis (not straddle it) for a valid solid.
pub fn revolve_solid(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
    angle: f64,
) -> Option<KSolid> {
    build_revolve_solid(outer, holes, basis, axis_pt, axis_dir, angle).map(KSolid)
}

/// [`revolve_solid`] with exact-arc annotations — see [`extrude_solid_arcs`].
pub fn revolve_solid_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
    angle: f64,
) -> Option<KSolid> {
    build_revolve_solid_arcs(outer, holes, outer_arcs, hole_arcs, basis, axis_pt, axis_dir, angle)
        .map(KSolid)
}

/// Mesh form of [`revolve_solid`] — for the mesh-boolean (Manifold) path, exactly as
/// [`extrude_tool_mesh`] is the mesh form of [`extrude_solid`].
///
/// A **full turn** is built *directly* as a shared-vertex surface-of-revolution grid: truck's
/// own triangulation of a large/fine revolve comes out non-watertight (cracks between B-rep
/// faces), which then breaks Manifold booleans (NotManifold → lossy BSP → torn surface / OOM).
/// The direct grid is watertight by construction at any scale. Partial turns (which need profile
/// caps) still go through truck.
pub fn revolve_tool_mesh(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
    angle: f64,
) -> Option<TriMesh> {
    let tau = std::f64::consts::TAU;
    if (angle.abs() - tau).abs() < 1.0e-4 {
        if let Some(m) = revolve_mesh_full(outer, holes, basis, axis_pt, axis_dir) {
            return Some(m);
        }
    }
    let solid = build_revolve_solid(outer, holes, basis, axis_pt, axis_dir, angle)?;
    guard(|| {
        let mut poly = solid.triangulation(TOL).to_polygon();
        poly.triangulate();
        Some(polymesh_to_trimesh(&poly))
    })
}

/// Build a **full-turn** solid of revolution directly as a watertight, shared-vertex triangle
/// mesh: each profile boundary loop (outer + holes) is swept around the axis in `N` steps and the
/// rings are stitched into quad strips that wrap closed (no caps for a full turn). Smooth vertex
/// normals; orientation corrected to outward-facing. `None` if degenerate.
fn revolve_mesh_full(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
) -> Option<TriMesh> {
    let o = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let to3 = |p: &[f64; 2]| o + u * p[0] + v * p[1];
    let ao = o + u * axis_pt[0] + v * axis_pt[1];
    let ad = u * axis_dir[0] + v * axis_dir[1];
    let alen = ad.magnitude();
    if alen < 1.0e-9 {
        return None;
    }
    let k = ad / alen; // unit axis

    let loops: Vec<Vec<[f64; 2]>> = std::iter::once(clean_loop(outer))
        .chain(holes.iter().map(|h| clean_loop(h)))
        .filter(|l| l.len() >= 3)
        .collect();
    if loops.is_empty() {
        return None;
    }
    // Angular step count from the largest swept radius (chord error ≈ TOL).
    let mut r_max: f64 = 0.0;
    for l in &loops {
        for p in l {
            let d = to3(p) - ao;
            r_max = r_max.max((d - k * d.dot(k)).magnitude());
        }
    }
    if r_max < 1.0e-6 {
        return None;
    }
    let n = (std::f64::consts::PI * (r_max / (2.0 * TOL)).sqrt()).ceil().clamp(32.0, 360.0) as usize;
    let rot = |p: Vector3, theta: f64| {
        let d = p - ao;
        let (c, s) = (theta.cos(), theta.sin());
        ao + d * c + k.cross(d) * s + k * (k.dot(d)) * (1.0 - c) // Rodrigues
    };

    let mut pos: Vec<Vector3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for l in &loops {
        let m = l.len();
        let base = pos.len() as u32;
        for ki in 0..n {
            let theta = std::f64::consts::TAU * ki as f64 / n as f64;
            for p in l {
                pos.push(rot(to3(p), theta));
            }
        }
        let vid = |ki: usize, i: usize| base + (ki * m + i) as u32;
        for ki in 0..n {
            let kn = (ki + 1) % n; // wrap closed (full turn)
            for i in 0..m {
                let inx = (i + 1) % m;
                let (a, b, c, d) = (vid(ki, i), vid(kn, i), vid(kn, inx), vid(ki, inx));
                indices.extend([a, b, c, a, c, d]);
            }
        }
    }

    // Smooth vertex normals (accumulate incident face normals).
    let mut nrm = vec![Vector3::new(0.0_f64, 0.0, 0.0); pos.len()];
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        let fn_ = (b - a).cross(c - a);
        for &i in t {
            nrm[i as usize] += fn_;
        }
    }
    // Signed volume → flip winding + normals if inside-out.
    let mut vol = 0.0;
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
        vol += a.dot(b.cross(c));
    }
    let flip = vol < 0.0;

    let mut out = TriMesh::default();
    out.positions = pos.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect();
    out.normals = nrm
        .iter()
        .map(|nv| {
            let nv = if flip { -*nv } else { *nv };
            let nl = nv.magnitude();
            if nl > 1.0e-12 {
                [(nv.x / nl) as f32, (nv.y / nl) as f32, (nv.z / nl) as f32]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect();
    out.indices = if flip {
        indices.chunks_exact(3).flat_map(|t| [t[0], t[2], t[1]]).collect()
    } else {
        indices
    };
    Some(out)
}

/// Build the **cut tool** mesh for a signed cut `distance` (positive sweeps along the
/// normal, negative against it). The tool overshoots both caps so they never end up
/// coplanar with the body — matching [`cut_tol`]'s tool exactly, but as a mesh.
pub fn cut_tool_mesh(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
) -> Option<TriMesh> {
    let (start, length) = cut_tool_span(distance, back)?;
    extrude_tool_mesh(outer, holes, basis, start, length)
}

/// The sweep a cut of this signed `distance` (plus Direction 2 `back`) needs: where it starts
/// relative to the sketch plane and how far it runs, overshooting both ends so neither cap can
/// land coplanar with the body. `None` for a cut of no depth.
///
/// Shared so the tagged and untagged cut tools cannot drift apart — a tool tagged over a span
/// other than the one it was built over would put its surfaces in the wrong place.
fn cut_tool_span(distance: f64, back: f64) -> Option<(f64, f64)> {
    let depth = distance.abs();
    if depth < 1e-9 {
        return None;
    }
    let eps = 0.05 + depth * 0.02;
    // `back` (Direction 2) extends the cut the opposite way from `distance`.
    let b = back.max(0.0);
    Some(if distance >= 0.0 {
        (-(eps + b), depth + 2.0 * eps + b)
    } else {
        (-(depth + eps), depth + 2.0 * eps + b)
    })
}

/// [`cut_tool_mesh`] carrying the profile's exact-arc annotations, so the bore a cut leaves behind
/// is recorded as a cylinder rather than as the ring of flat strips that actually got built.
pub fn cut_tool_mesh_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
) -> Option<TriMesh> {
    let (start, length) = cut_tool_span(distance, back)?;
    extrude_tool_mesh_arcs(outer, holes, outer_arcs, hole_arcs, basis, start, length)
}

/// Serialize a triangle mesh as a **binary STL** blob (for 3D printing / mesh interchange).
/// Every triangle's normal is recomputed from its winding so the STL is self-consistent.
pub fn export_stl(mesh: &TriMesh) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(84 + mesh.indices.len() / 3 * 50);
    out.extend_from_slice(&[0u8; 80]); // 80-byte header (ignored)
    out.extend_from_slice(&((mesh.indices.len() / 3) as u32).to_le_bytes());
    for t in mesh.indices.chunks_exact(3) {
        let p = |i: u32| mesh.positions[i as usize];
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let mut n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 1e-12 {
            n = [n[0] / l, n[1] / l, n[2] / l];
        }
        for v in [n, a, b, c] {
            for k in 0..3 {
                out.extend_from_slice(&v[k].to_le_bytes());
            }
        }
        out.extend_from_slice(&[0u8, 0u8]); // attribute byte count
    }
    out
}

/// Parse an STL blob — **binary or ASCII**, auto-detected — into a flat-shaded [`TriMesh`]
/// (per-triangle normals recomputed from the winding; the file's stored normals are ignored,
/// as they're unreliable in the wild). Degenerate (zero-area) triangles are dropped. `None`
/// if the blob isn't a recognizable STL or holds no triangles.
pub fn import_stl(bytes: &[u8]) -> Option<TriMesh> {
    // ASCII files start with "solid" AND actually contain "facet" keywords (many binary
    // exporters also write "solid" into the 80-byte header, so the keyword check decides).
    let looks_ascii = bytes.len() >= 5
        && bytes[..5].eq_ignore_ascii_case(b"solid")
        && std::str::from_utf8(&bytes[..bytes.len().min(2048)]).is_ok_and(|s| s.contains("facet"));
    let tris: Vec<[[f64; 3]; 3]> = if looks_ascii {
        let text = std::str::from_utf8(bytes).ok()?;
        let mut tris = Vec::new();
        let mut cur: Vec<[f64; 3]> = Vec::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            if w.next() == Some("vertex") {
                let (x, y, z) = (w.next()?.parse().ok()?, w.next()?.parse().ok()?, w.next()?.parse().ok()?);
                cur.push([x, y, z]);
                if cur.len() == 3 {
                    tris.push([cur[0], cur[1], cur[2]]);
                    cur.clear();
                }
            }
        }
        tris
    } else {
        if bytes.len() < 84 {
            return None;
        }
        let count = u32::from_le_bytes(bytes[80..84].try_into().ok()?) as usize;
        // Guard against a corrupt count (each triangle is 50 bytes).
        if count == 0 || bytes.len() < 84 + count * 50 {
            return None;
        }
        let f = |off: usize| -> f64 { f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f64 };
        (0..count)
            .map(|i| {
                let base = 84 + i * 50 + 12; // skip the stored normal
                [
                    [f(base), f(base + 4), f(base + 8)],
                    [f(base + 12), f(base + 16), f(base + 20)],
                    [f(base + 24), f(base + 28), f(base + 32)],
                ]
            })
            .collect()
    };
    let mut mesh = TriMesh::default();
    for [a, b, c] in tris {
        let (e1, e2) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() < 1e-14 {
            continue; // zero-area sliver
        }
        push_tri(&mut mesh, a, b, c);
    }
    (mesh.indices.len() >= 3).then_some(mesh)
}

/// Intersect a triangle mesh with a plane, returning the crossing segments (world space, both
/// endpoints exactly on the plane). Used to give sketches on a plane through an imported
/// scan/mesh its **section outline** to trace and snap to (reverse engineering). Triangles
/// coplanar with the plane contribute their edges, so a flat cap at the section height still
/// yields its outline (interior duplicates are harmless for snapping).
pub fn mesh_plane_section(m: &TriMesh, origin: [f32; 3], normal: [f32; 3]) -> Vec<[[f32; 3]; 2]> {
    let n = normal;
    let dist = |p: [f32; 3]| (p[0] - origin[0]) * n[0] + (p[1] - origin[1]) * n[1] + (p[2] - origin[2]) * n[2];
    let mut out = Vec::new();
    let eps = 1e-5_f32;
    for t in m.indices.chunks_exact(3) {
        let p = [m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]];
        let d = [dist(p[0]), dist(p[1]), dist(p[2])];
        if d.iter().all(|x| x.abs() < eps) {
            // Coplanar triangle: emit its edges as section geometry.
            out.push([p[0], p[1]]);
            out.push([p[1], p[2]]);
            out.push([p[2], p[0]]);
            continue;
        }
        // Collect the crossing points of the three edges (vertex-on-plane counts once).
        let mut pts: Vec<[f32; 3]> = Vec::new();
        let mut add = |q: [f32; 3]| {
            if !pts.iter().any(|e| (0..3).all(|k| (e[k] - q[k]).abs() < eps)) {
                pts.push(q);
            }
        };
        for i in 0..3 {
            let j = (i + 1) % 3;
            let (da, db) = (d[i], d[j]);
            if da.abs() < eps {
                add(p[i]);
            }
            if (da > eps && db < -eps) || (da < -eps && db > eps) {
                let t01 = da / (da - db);
                add([
                    p[i][0] + (p[j][0] - p[i][0]) * t01,
                    p[i][1] + (p[j][1] - p[i][1]) * t01,
                    p[i][2] + (p[j][2] - p[i][2]) * t01,
                ]);
            }
        }
        if pts.len() == 2 {
            let (a, b) = (pts[0], pts[1]);
            let len2: f32 = (0..3).map(|k| (a[k] - b[k]) * (a[k] - b[k])).sum();
            if len2 > eps * eps {
                out.push([a, b]);
            }
        }
    }
    out
}

/// What [`repair_mesh`] did, so the app can report it honestly ("welded 214 verts,
/// filled 3 holes") and warn when a mesh is still open afterwards.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairReport {
    /// Vertices merged into a neighbour by the weld pass.
    pub welded: usize,
    /// Degenerate / duplicate triangles dropped.
    pub degenerate_removed: usize,
    /// Boundary loops closed by triangulation.
    pub holes_filled: usize,
    /// Boundary edges STILL open after repair (0 = watertight-topology).
    pub open_edges_left: usize,
}

/// Best-effort repair of a triangle soup so real-world STLs (scans, hobby exports) survive
/// booleans: weld near-coincident vertices, drop degenerate/duplicate faces, and close small
/// boundary loops (≤ 64 edges) by ear-clip triangulation. Conservative on purpose — big
/// openings are left alone (reported via `open_edges_left`) rather than papered over.
pub fn repair_mesh(m: &TriMesh) -> (TriMesh, RepairReport) {
    use std::collections::HashMap;
    let mut rep = RepairReport::default();
    if m.indices.len() < 3 {
        return (m.clone(), rep);
    }
    // --- Weld: quantize by a bbox-relative tolerance (exporters emit per-triangle copies
    // of every vertex; scans add real sub-micron jitter on top). ---
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let tol = (diag * 1.0e-5).max(1.0e-6);
    let key = |p: [f32; 3]| ((p[0] / tol).round() as i64, (p[1] / tol).round() as i64, (p[2] / tol).round() as i64);
    let mut uniq: Vec<[f32; 3]> = Vec::new();
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; m.positions.len()];
    for (i, p) in m.positions.iter().enumerate() {
        remap[i] = *map.entry(key(*p)).or_insert_with(|| {
            uniq.push(*p);
            (uniq.len() - 1) as u32
        });
    }
    rep.welded = m.positions.len() - uniq.len();
    // --- Faces: drop degenerate (repeated vertex after welding) and duplicate ones. ---
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let mut seen: std::collections::HashSet<[u32; 3]> = std::collections::HashSet::new();
    for t in m.indices.chunks_exact(3) {
        let f = [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]];
        if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
            rep.degenerate_removed += 1;
            continue;
        }
        let mut k = f;
        k.sort_unstable();
        if !seen.insert(k) {
            rep.degenerate_removed += 1;
            continue;
        }
        faces.push(f);
    }
    // --- Boundary loops: an undirected edge with exactly one incident face is open.
    // Chain the directed boundary edges (a→b as the face uses them) into loops. ---
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for f in &faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            *edge_count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut next_of: HashMap<u32, u32> = HashMap::new();
    for f in &faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            if edge_count[&(a.min(b), a.max(b))] == 1 {
                next_of.insert(a, b); // boundary edge, in face winding order
            }
        }
    }
    let mut fills: Vec<[u32; 3]> = Vec::new();
    let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let starts: Vec<u32> = {
        let mut s: Vec<u32> = next_of.keys().copied().collect();
        s.sort_unstable();
        s
    };
    for start in starts {
        if visited.contains(&start) {
            continue;
        }
        let mut cycle = vec![start];
        visited.insert(start);
        let mut cur = start;
        let closed = loop {
            match next_of.get(&cur) {
                Some(&nxt) if nxt == start => break true,
                Some(&nxt) if !visited.contains(&nxt) && cycle.len() <= 64 => {
                    visited.insert(nxt);
                    cycle.push(nxt);
                    cur = nxt;
                }
                _ => break false, // open path / junction / too large
            }
        };
        if !closed || cycle.len() < 3 || cycle.len() > 64 {
            continue;
        }
        // Patch triangles must traverse each boundary edge OPPOSITE to the existing face,
        // so triangulate the REVERSED loop. Project on the loop's Newell normal, ear-clip.
        cycle.reverse();
        let pts3: Vec<[f32; 3]> = cycle.iter().map(|i| uniq[*i as usize]).collect();
        let mut n = [0f64; 3];
        for i in 0..pts3.len() {
            let (p, q) = (pts3[i], pts3[(i + 1) % pts3.len()]);
            n[0] += (p[1] as f64 - q[1] as f64) * (p[2] as f64 + q[2] as f64);
            n[1] += (p[2] as f64 - q[2] as f64) * (p[0] as f64 + q[0] as f64);
            n[2] += (p[0] as f64 - q[0] as f64) * (p[1] as f64 + q[1] as f64);
        }
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nl < 1e-12 {
            continue;
        }
        let n = [n[0] / nl, n[1] / nl, n[2] / nl];
        // 2D basis in the loop plane.
        let pick = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = {
            let c = [n[1] * pick[2] - n[2] * pick[1], n[2] * pick[0] - n[0] * pick[2], n[0] * pick[1] - n[1] * pick[0]];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            [c[0] / l, c[1] / l, c[2] / l]
        };
        let v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        let p2: Vec<[f64; 2]> = pts3
            .iter()
            .map(|p| {
                let p = [p[0] as f64, p[1] as f64, p[2] as f64];
                [p[0] * u[0] + p[1] * u[1] + p[2] * u[2], p[0] * v[0] + p[1] * v[1] + p[2] * v[2]]
            })
            .collect();
        // Ear clipping (CCW in this projection by Newell construction); falls back to a fan
        // if it stalls on numerically-nasty input.
        let cross2 = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        let mut order: Vec<usize> = (0..cycle.len()).collect();
        let mut tris2: Vec<[usize; 3]> = Vec::new();
        let mut guard = 0usize;
        while order.len() > 3 && guard < 10_000 {
            guard += 1;
            let mut clipped = false;
            for i in 0..order.len() {
                let (pi, ci, ni) = (
                    order[(i + order.len() - 1) % order.len()],
                    order[i],
                    order[(i + 1) % order.len()],
                );
                if cross2(p2[pi], p2[ci], p2[ni]) <= 1e-12 {
                    continue; // reflex
                }
                let inside = order.iter().any(|&o| {
                    o != pi
                        && o != ci
                        && o != ni
                        && cross2(p2[pi], p2[ci], p2[o]) > 0.0
                        && cross2(p2[ci], p2[ni], p2[o]) > 0.0
                        && cross2(p2[ni], p2[pi], p2[o]) > 0.0
                });
                if inside {
                    continue;
                }
                tris2.push([pi, ci, ni]);
                order.remove(i);
                clipped = true;
                break;
            }
            if !clipped {
                break; // stalled — fan the rest
            }
        }
        if order.len() == 3 {
            tris2.push([order[0], order[1], order[2]]);
        } else {
            for i in 1..order.len() - 1 {
                tris2.push([order[0], order[i], order[i + 1]]);
            }
        }
        for t in tris2 {
            fills.push([cycle[t[0]], cycle[t[1]], cycle[t[2]]]);
        }
        rep.holes_filled += 1;
    }
    faces.extend(fills);
    // --- Rebuild with fresh flat normals; count what's still open. ---
    let mut out = TriMesh::default();
    for f in &faces {
        let p = |i: u32| {
            let q = uniq[i as usize];
            [q[0] as f64, q[1] as f64, q[2] as f64]
        };
        push_tri(&mut out, p(f[0]), p(f[1]), p(f[2]));
    }
    let mut final_edges: HashMap<(u32, u32), u32> = HashMap::new();
    for f in &faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            *final_edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    rep.open_edges_left = final_edges.values().filter(|c| **c == 1).count();
    (out, rep)
}

/// A primitive fitted to a clicked scan region by [`fit_region`] — the reverse-engineering
/// "click a face, get a datum" step. `rms` is the fit residual (mm), `count` the region size.
#[derive(Debug, Clone, PartialEq)]
pub enum RegionFit {
    /// A flat region: plane through `origin` (region centroid) with `normal`.
    Plane { origin: [f32; 3], normal: [f32; 3], rms: f32, count: usize },
    /// A cylindrical region: `axis` through `center` (on the axis, at the region's mid-height).
    Cylinder { center: [f32; 3], axis: [f32; 3], radius: f32, rms: f32, count: usize },
    /// A spherical region.
    Sphere { center: [f32; 3], radius: f32, rms: f32, count: usize },
}

/// Eigen-decomposition of a symmetric 3×3 matrix by cyclic Jacobi rotations.
/// Returns (eigenvalues, eigenvectors as columns), sorted DESCENDING by eigenvalue.
fn sym_eigen3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        // Largest off-diagonal element.
        let (mut p, mut q, mut big) = (0usize, 1usize, 0.0f64);
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > big {
                    big = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1e-14 {
            break;
        }
        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;
        // Apply the rotation G(p,q,θ): a = GᵀaG, v = vG.
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..3 {
            if k != p && k != q {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    // Sort descending.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let vals = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let mut vecs = [[0.0; 3]; 3];
    for (col, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vecs[r][col] = v[r][i];
        }
    }
    (vals, vecs)
}

/// Grow a smooth region from `seed_tri` (BFS across edges whose dihedral angle stays under
/// ~35°, capped at `max_dist` from the seed) and fit the best primitive — plane, cylinder,
/// or sphere — to it. Returns the fit plus the region's boundary edges (for a highlight).
/// This is "click a face on the scan → get its datum": flat face → plane, bore/boss →
/// axis + radius, ball → centre.
pub fn fit_region(m: &TriMesh, seed_tri: usize, max_dist: f32) -> Option<(RegionFit, Vec<[[f32; 3]; 2]>)> {
    use std::collections::HashMap;
    let ntri = m.indices.len() / 3;
    if seed_tri >= ntri {
        return None;
    }
    // Weld verts by position (scan meshes are flat-shaded: every triangle owns copies).
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let wtol = (diag * 1e-5).max(1e-6);
    let key = |p: [f32; 3]| ((p[0] / wtol).round() as i64, (p[1] / wtol).round() as i64, (p[2] / wtol).round() as i64);
    let mut vid: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let tri_vids: Vec<[u32; 3]> = m
        .indices
        .chunks_exact(3)
        .map(|t| {
            let mut ids = [0u32; 3];
            for (k, &i) in t.iter().enumerate() {
                let p = m.positions[i as usize];
                ids[k] = *vid.entry(key(p)).or_insert_with(|| {
                    verts.push(p);
                    (verts.len() - 1) as u32
                });
            }
            ids
        })
        .collect();
    // Edge → adjacent triangles.
    let mut edge_tris: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ti, ids) in tri_vids.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (ids[k], ids[(k + 1) % 3]);
            edge_tris.entry((a.min(b), a.max(b))).or_default().push(ti as u32);
        }
    }
    // Per-triangle normal / centroid / area.
    let tri_geo: Vec<([f32; 3], [f32; 3], f32)> = tri_vids
        .iter()
        .map(|ids| {
            let (a, b, c) = (verts[ids[0] as usize], verts[ids[1] as usize], verts[ids[2] as usize]);
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
            if l > 1e-12 {
                ([n[0] / l, n[1] / l, n[2] / l], cen, l * 0.5)
            } else {
                ([0.0, 0.0, 1.0], cen, 0.0)
            }
        })
        .collect();
    // BFS with dihedral smoothness + distance cap.
    const COS_SMOOTH: f32 = 0.82; // ~35°
    let seed_c = tri_geo[seed_tri].1;
    let d2 = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let max_d2 = max_dist * max_dist;
    let mut in_region = vec![false; ntri];
    in_region[seed_tri] = true;
    let mut queue = std::collections::VecDeque::from([seed_tri as u32]);
    while let Some(ti) = queue.pop_front() {
        let (n0, _, _) = tri_geo[ti as usize];
        let ids = tri_vids[ti as usize];
        for k in 0..3 {
            let (a, b) = (ids[k], ids[(k + 1) % 3]);
            if let Some(nbrs) = edge_tris.get(&(a.min(b), a.max(b))) {
                for &nb in nbrs {
                    if in_region[nb as usize] {
                        continue;
                    }
                    let (n1, c1, _) = tri_geo[nb as usize];
                    let dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
                    if dot > COS_SMOOTH && d2(c1, seed_c) < max_d2 {
                        in_region[nb as usize] = true;
                        queue.push_back(nb);
                    }
                }
            }
        }
    }
    // Region data: unique verts + area-weighted normal second moment.
    let mut region_verts: Vec<[f32; 3]> = Vec::new();
    let mut seen_v: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut nn = [[0.0f64; 3]; 3];
    let mut area_sum = 0.0f64;
    let mut count = 0usize;
    for ti in 0..ntri {
        if !in_region[ti] {
            continue;
        }
        count += 1;
        let (n, _, area) = tri_geo[ti];
        let w = area as f64;
        area_sum += w;
        for r in 0..3 {
            for c in 0..3 {
                nn[r][c] += w * n[r] as f64 * n[c] as f64;
            }
        }
        for &v in &tri_vids[ti] {
            if seen_v.insert(v) {
                region_verts.push(verts[v as usize]);
            }
        }
    }
    if count < 4 || area_sum < 1e-12 || region_verts.len() < 6 {
        return None;
    }
    for r in 0..3 {
        for c in 0..3 {
            nn[r][c] /= area_sum;
        }
    }
    let (_, evecs) = sym_eigen3(nn);
    let col = |c: usize| [evecs[0][c] as f32, evecs[1][c] as f32, evecs[2][c] as f32];
    let centroid = {
        let mut s = [0.0f64; 3];
        for p in &region_verts {
            for k in 0..3 {
                s[k] += p[k] as f64;
            }
        }
        [(s[0] / region_verts.len() as f64) as f32, (s[1] / region_verts.len() as f64) as f32, (s[2] / region_verts.len() as f64) as f32]
    };
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let rms = |errs: &[f32]| (errs.iter().map(|e| (e * e) as f64).sum::<f64>() / errs.len() as f64).sqrt() as f32;

    // Candidate 1 — PLANE via point PCA: normal = smallest point-scatter direction.
    let (plane_normal, plane_rms) = {
        let mut cov = [[0.0f64; 3]; 3];
        for p in &region_verts {
            let d = sub(*p, centroid);
            for r in 0..3 {
                for c in 0..3 {
                    cov[r][c] += d[r] as f64 * d[c] as f64;
                }
            }
        }
        let (_, pv) = sym_eigen3(cov);
        let n = [pv[0][2] as f32, pv[1][2] as f32, pv[2][2] as f32]; // smallest scatter
        let errs: Vec<f32> = region_verts.iter().map(|p| dot3(sub(*p, centroid), n)).collect();
        (n, rms(&errs))
    };
    // Candidate 2 — CYLINDER: axis = least-spread direction of the NORMALS (a cylinder's
    // normals fan out perpendicular to its axis), then a Kasa circle in the ⊥ plane.
    let cyl = {
        let axis = col(2);
        let ax_u = {
            let pick = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let c = [axis[1] * pick[2] - axis[2] * pick[1], axis[2] * pick[0] - axis[0] * pick[2], axis[0] * pick[1] - axis[1] * pick[0]];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            [c[0] / l, c[1] / l, c[2] / l]
        };
        let ax_v = [
            axis[1] * ax_u[2] - axis[2] * ax_u[1],
            axis[2] * ax_u[0] - axis[0] * ax_u[2],
            axis[0] * ax_u[1] - axis[1] * ax_u[0],
        ];
        let p2: Vec<[f32; 2]> = region_verts.iter().map(|p| {
            let d = sub(*p, centroid);
            [dot3(d, ax_u), dot3(d, ax_v)]
        }).collect();
        kasa_circle_2d(&p2).map(|(c2, r, _)| {
            let center = [
                centroid[0] + ax_u[0] * c2[0] + ax_v[0] * c2[1],
                centroid[1] + ax_u[1] * c2[0] + ax_v[1] * c2[1],
                centroid[2] + ax_u[2] * c2[0] + ax_v[2] * c2[1],
            ];
            let errs: Vec<f32> = p2.iter().map(|q| ((q[0] - c2[0]).powi(2) + (q[1] - c2[1]).powi(2)).sqrt() - r).collect();
            (center, axis, r, rms(&errs))
        })
    };
    // Candidate 3 — SPHERE (Kasa in 3D: x²+y²+z² + Dx + Ey + Fz + G = 0).
    let sph = {
        let n = region_verts.len() as f64;
        let mut a4 = [[0.0f64; 4]; 4];
        let mut b4 = [0.0f64; 4];
        for p in &region_verts {
            let (x, y, z) = (p[0] as f64, p[1] as f64, p[2] as f64);
            let row = [x, y, z, 1.0];
            let rhs = -(x * x + y * y + z * z);
            for r in 0..4 {
                for c in 0..4 {
                    a4[r][c] += row[r] * row[c];
                }
                b4[r] += row[r] * rhs;
            }
        }
        let _ = n;
        // Gaussian elimination with partial pivoting.
        let mut ok = true;
        for i in 0..4 {
            let mut piv = i;
            for r in (i + 1)..4 {
                if a4[r][i].abs() > a4[piv][i].abs() {
                    piv = r;
                }
            }
            a4.swap(i, piv);
            b4.swap(i, piv);
            if a4[i][i].abs() < 1e-12 {
                ok = false;
                break;
            }
            for r in (i + 1)..4 {
                let f = a4[r][i] / a4[i][i];
                for c in i..4 {
                    a4[r][c] -= f * a4[i][c];
                }
                b4[r] -= f * b4[i];
            }
        }
        if ok {
            let mut x = [0.0f64; 4];
            for i in (0..4).rev() {
                let mut s = b4[i];
                for c in (i + 1)..4 {
                    s -= a4[i][c] * x[c];
                }
                x[i] = s / a4[i][i];
            }
            let c = [(-x[0] / 2.0) as f32, (-x[1] / 2.0) as f32, (-x[2] / 2.0) as f32];
            let r2 = (c[0] as f64).powi(2) + (c[1] as f64).powi(2) + (c[2] as f64).powi(2) - x[3];
            if r2 > 0.0 {
                let r = (r2 as f32).sqrt();
                let errs: Vec<f32> = region_verts
                    .iter()
                    .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt() - r)
                    .collect();
                Some((c, r, rms(&errs)))
            } else {
                None
            }
        } else {
            None
        }
    };
    // Pick by residual, preferring the SIMPLER primitive when it's close (within 25%, plus
    // a small absolute epsilon): a plane is a degenerate cylinder is a degenerate sphere, so
    // without the preference every flat face would "win" as a huge-radius sphere — and on
    // sparsely-sampled geometry two fits can BOTH be numerically perfect (e.g. a two-ring
    // cylinder wall lies exactly on a sphere too), where only the epsilon breaks the tie.
    let eps = (diag * 1e-6).max(1e-5);
    let cyl_rms = cyl.as_ref().map(|c| c.3).unwrap_or(f32::INFINITY);
    let sph_rms = sph.as_ref().map(|s| s.2).unwrap_or(f32::INFINITY);
    let fit = if plane_rms <= cyl_rms * 1.25 + eps && plane_rms <= sph_rms * 1.25 + eps {
        RegionFit::Plane { origin: centroid, normal: plane_normal, rms: plane_rms, count }
    } else if let (true, Some((center, axis, radius, r))) = (cyl_rms <= sph_rms * 1.25 + eps, cyl) {
        RegionFit::Cylinder { center, axis, radius, rms: r, count }
    } else if let Some((center, radius, r)) = sph {
        RegionFit::Sphere { center, radius, rms: r, count }
    } else {
        RegionFit::Plane { origin: centroid, normal: plane_normal, rms: plane_rms, count }
    };
    // Boundary edges of the region (used by exactly one region triangle) — the highlight.
    let mut boundary: Vec<[[f32; 3]; 2]> = Vec::new();
    let mut edge_use: HashMap<(u32, u32), u32> = HashMap::new();
    for ti in 0..ntri {
        if !in_region[ti] {
            continue;
        }
        let ids = tri_vids[ti];
        for k in 0..3 {
            let (a, b) = (ids[k], ids[(k + 1) % 3]);
            *edge_use.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for ((a, b), uses) in edge_use {
        if uses == 1 {
            boundary.push([verts[a as usize], verts[b as usize]]);
        }
    }
    Some((fit, boundary))
}

/// A recognized curve in a mesh cross-section, in the sketch plane's 2D (uv) space.
/// Produced by [`fit_section_shapes`] from the raw triangle-crossing segments of
/// [`mesh_plane_section`] — tessellation chords become real circles/arcs/polylines the
/// sketcher can snap to and convert into sketch entities.
#[derive(Debug, Clone, PartialEq)]
pub enum SectionShape {
    /// A decimated polyline (straight runs collapsed); `closed` means last→first connects.
    Poly { pts: Vec<[f32; 2]>, closed: bool },
    /// A full circle (a closed loop whose points all sit on one circle).
    Circle { center: [f32; 2], radius: f32 },
    /// A circular arc from `start` to `end` about `center` (`ccw` = counter-clockwise).
    Arc { center: [f32; 2], radius: f32, start: [f32; 2], end: [f32; 2], ccw: bool },
}

/// Least-squares (Kasa) circle through 2D points: minimizes Σ(x²+y²+Dx+Ey+F)². Returns
/// (center, radius, max |dist−r| deviation), or None if degenerate (collinear/too few).
fn kasa_circle_2d(pts: &[[f32; 2]]) -> Option<([f32; 2], f32, f32)> {
    if pts.len() < 3 {
        return None;
    }
    let (mut sxx, mut sxy, mut syy, mut sx, mut sy, mut sxz, mut syz, mut sz, n) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, pts.len() as f64);
    for p in pts {
        let (x, y) = (p[0] as f64, p[1] as f64);
        let z = x * x + y * y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sx += x;
        sy += y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    // Normal equations for [D, E, F] in x²+y² + Dx + Ey + F = 0.
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let rhs = [-sxz, -syz, -sz];
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det3(&m);
    if d.abs() < 1e-9 {
        return None;
    }
    let col = |k: usize| {
        let mut mm = m;
        for r in 0..3 {
            mm[r][k] = rhs[r];
        }
        det3(&mm) / d
    };
    let (dd, ee, ff) = (col(0), col(1), col(2));
    let (cx, cy) = (-dd / 2.0, -ee / 2.0);
    let r2 = cx * cx + cy * cy - ff;
    if r2 <= 0.0 {
        return None;
    }
    let r = r2.sqrt();
    let mut dev = 0f64;
    for p in pts {
        let dist = ((p[0] as f64 - cx).powi(2) + (p[1] as f64 - cy).powi(2)).sqrt();
        dev = dev.max((dist - r).abs());
    }
    Some(([cx as f32, cy as f32], r as f32, dev as f32))
}

/// Max perpendicular deviation of interior points from the chord first→last.
fn line_dev(pts: &[[f32; 2]]) -> f32 {
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return 0.0;
    }
    let mut worst = 0f32;
    for p in &pts[1..pts.len() - 1] {
        worst = worst.max(((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len);
    }
    worst
}

/// Chain raw section segments into connected runs and fit clean shapes: full circles,
/// arcs, and decimated polylines, everything within `tol` of the input points. This is
/// what turns "5,000 chords across a scanned cylinder" into "a circle, r = 14.02".
pub fn fit_section_shapes(segs: &[[[f32; 2]; 2]], tol: f32) -> Vec<SectionShape> {
    use std::collections::HashMap;
    let tol = tol.max(1e-6);
    // --- Weld endpoints (quantized) and build node adjacency ---
    let q = (tol * 0.25).max(1e-6);
    let key = |p: [f32; 2]| ((p[0] / q).round() as i64, (p[1] / q).round() as i64);
    let mut nodes: Vec<[f32; 2]> = Vec::new();
    let mut idx: HashMap<(i64, i64), u32> = HashMap::new();
    let mut node_of = |p: [f32; 2], nodes: &mut Vec<[f32; 2]>| -> u32 {
        *idx.entry(key(p)).or_insert_with(|| {
            nodes.push(p);
            (nodes.len() - 1) as u32
        })
    };
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut edges: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for s in segs {
        let a = node_of(s[0], &mut nodes);
        let b = node_of(s[1], &mut nodes);
        if a == b {
            continue; // welded-away sliver
        }
        let e = (a.min(b), a.max(b));
        if !edges.insert(e) {
            continue; // duplicate (e.g. coplanar-cap double cover)
        }
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    // --- Walk chains: start at every non-degree-2 node, then sweep leftover loops ---
    let mut used: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut chains: Vec<(Vec<u32>, bool)> = Vec::new();
    let walk = |start: u32, first: u32, used: &mut std::collections::HashSet<(u32, u32)>, adj: &HashMap<u32, Vec<u32>>| {
        let mut chain = vec![start, first];
        used.insert((start.min(first), start.max(first)));
        let (mut prev, mut cur) = (start, first);
        loop {
            let nexts = &adj[&cur];
            if nexts.len() != 2 {
                return (chain, false); // hit a junction/endpoint — open chain
            }
            let nxt = if nexts[0] == prev { nexts[1] } else { nexts[0] };
            if !used.insert((cur.min(nxt), cur.max(nxt))) {
                return (chain, false);
            }
            if nxt == chain[0] {
                return (chain, true); // came home — closed loop
            }
            chain.push(nxt);
            prev = cur;
            cur = nxt;
        }
    };
    let mut starts: Vec<u32> = adj.iter().filter(|(_, v)| v.len() != 2).map(|(k, _)| *k).collect();
    starts.sort_unstable();
    for s in starts {
        for n in adj[&s].clone() {
            if !used.contains(&(s.min(n), s.max(n))) {
                chains.push(walk(s, n, &mut used, &adj));
            }
        }
    }
    let mut loop_starts: Vec<u32> = adj.keys().copied().collect();
    loop_starts.sort_unstable();
    for s in loop_starts {
        for n in adj[&s].clone() {
            if !used.contains(&(s.min(n), s.max(n))) {
                chains.push(walk(s, n, &mut used, &adj));
            }
        }
    }

    // --- Fit each chain: circle (closed) → greedy arc/line runs → decimated polyline ---
    let mut out = Vec::new();
    for (chain, closed) in chains {
        let pts: Vec<[f32; 2]> = chain.iter().map(|i| nodes[*i as usize]).collect();
        if pts.len() < 2 {
            continue;
        }
        // A closed loop that fits one circle IS that circle.
        if closed && pts.len() >= 8 {
            if let Some((c, r, dev)) = kasa_circle_2d(&pts) {
                if dev < tol {
                    out.push(SectionShape::Circle { center: c, radius: r });
                    continue;
                }
            }
        }
        // Greedy segmentation: at each position take the longest run (line or arc) that
        // stays within tol; straight runs accumulate into one polyline.
        let ring: Vec<[f32; 2]> = if closed {
            let mut v = pts.clone();
            v.push(pts[0]);
            v
        } else {
            pts.clone()
        };
        let mut poly: Vec<[f32; 2]> = Vec::new();
        let flush = |poly: &mut Vec<[f32; 2]>, out: &mut Vec<SectionShape>| {
            if poly.len() >= 2 {
                out.push(SectionShape::Poly { pts: std::mem::take(poly), closed: false });
            } else {
                poly.clear();
            }
        };
        let mut i = 0usize;
        while i + 1 < ring.len() {
            // Longest straight run from i.
            let mut le = i + 1;
            while le + 1 < ring.len() && line_dev(&ring[i..=le + 1]) < tol {
                le += 1;
            }
            // Longest arc run from i (needs enough points to be trustworthy).
            let mut ae = i + 1;
            let mut arc: Option<([f32; 2], f32)> = None;
            {
                let mut e = i + 5;
                while e < ring.len() {
                    match kasa_circle_2d(&ring[i..=e]) {
                        Some((c, r, dev)) if dev < tol && r > tol * 4.0 => {
                            ae = e;
                            arc = Some((c, r));
                            e += 1;
                        }
                        _ => break,
                    }
                }
            }
            if arc.is_some() && ae > le + 2 {
                // Arc wins (covers meaningfully more points than the straight run).
                let (c, r) = arc.unwrap();
                flush(&mut poly, &mut out);
                let (s, e) = (ring[i], ring[ae]);
                // Orientation from the midpoint of the run.
                let m = ring[(i + ae) / 2];
                let ang = |p: [f32; 2]| (p[1] - c[1]).atan2(p[0] - c[0]);
                let (a0, am, a1) = (ang(s), ang(m), ang(e));
                let norm = |x: f32| {
                    let t = x % (2.0 * std::f32::consts::PI);
                    if t < 0.0 {
                        t + 2.0 * std::f32::consts::PI
                    } else {
                        t
                    }
                };
                let ccw = norm(am - a0) < norm(a1 - a0);
                out.push(SectionShape::Arc { center: c, radius: r, start: s, end: e, ccw });
                i = ae;
            } else {
                if poly.is_empty() {
                    poly.push(ring[i]);
                }
                poly.push(ring[le]);
                i = le;
            }
        }
        // Close or emit the trailing polyline.
        if !poly.is_empty() {
            let is_whole = closed && out.is_empty() && poly.len() >= 3;
            if is_whole {
                let mut p = std::mem::take(&mut poly);
                p.pop(); // last == first (ring wrap) — drop the duplicate
                // The walk starts at an arbitrary loop node, often mid-edge — drop any
                // vertex collinear (within tol) with its neighbours around the ring.
                let mut k = 0;
                while p.len() > 3 && k < p.len() {
                    let (prev, next) = (p[(k + p.len() - 1) % p.len()], p[(k + 1) % p.len()]);
                    if line_dev(&[prev, p[k], next]) < tol {
                        p.remove(k);
                    } else {
                        k += 1;
                    }
                }
                out.push(SectionShape::Poly { pts: p, closed: true });
            } else {
                flush(&mut poly, &mut out);
            }
        }
    }
    out
}

/// A patch of mesh the sketch (or the fillet) says lies on one surface of revolution, checked to be
/// a plain complete ring between two circular rims, so the exporter can rebuild it as that surface
/// instead of the flat strips it was actually built from.
/// One stretch of a band's generatrix, in the half-plane the band is revolved from: (distance
/// from the axis, distance along it).
#[derive(Clone, Debug)]
struct Seg {
    /// A straight run — a cylinder — when `None`; otherwise the centre of the arc a rolling ball
    /// left, and its radius. Both in the same half-plane as `gen`.
    arc: Option<([f64; 2], f64)>,
    /// The two ends of this stretch.
    gen: [[f64; 2]; 2],
}

struct Band {
    /// The generatrix, end to end. Usually one stretch; several when neighbouring patches share a
    /// rim, which is how a bore and the fillets at its mouths become ONE surface of revolution
    /// rather than three that have to be stitched to each other.
    segs: Vec<Seg>,
    origin: [f64; 3],
    axis: [f64; 3],
    /// Topo vertices of the two OUTER rims, in order about the axis. A rim shared inside the chain
    /// is not here: it stops being a boundary at all.
    rims: [Vec<usize>; 2],
    /// The topo faces this band is made of, across every stretch — skipped when the flat faces
    /// are written.
    faces: Vec<usize>,
    /// For each rim, the face and loop that runs along it. That loop is replaced by the tube's own
    /// rim circle, walked the other way.
    seam: [(usize, usize); 2],
    /// What the mesh triangles of this patch contribute to the body's volume. The exact tube's own
    /// contribution is compared against it — see `mesh_to_solid`.
    patch_volume: f64,
}

/// Find the patches of a mesh that can be rebuilt as exact surfaces of revolution.
///
/// Reads the surfaces the tools recorded — a bore is a cylinder because the sketch drew a circle,
/// and a fillet's band is a torus because that is the tube the rolling ball rolled, neither of them
/// because the triangles look round — then checks the geometry really is the plain case the
/// exporter can rebuild: a complete ring between two circular rims, meeting the rest of the body
/// only along them.
///
/// Every check is a way the substitution would otherwise corrupt the shell, not a matter of taste.
/// A rim stops being forty-eight vertices and becomes one circle, so anything else holding those
/// vertices would be left pointing at geometry that no longer exists.
///
/// That plainness is a real limit, and worth stating rather than discovering. Where a band is
/// turned away it is nearly always because it is only PART of a ring — a wall cut back, a fillet
/// that runs out — which a full revolution cannot express; measured over the corpus, twenty-two of
/// the twenty-nine patches rejected for their boundary are that case, against five that are a whole
/// ring with a pinhole in it and two that are two rings still joined.
///
/// Those twenty-two are the biggest thing left, and they LOOK tidy: seventeen have every boundary
/// vertex at one of the two ends and cross between the ends exactly twice, which is to say they are
/// a plain strip — round one end, up, back along the other, down. truck sweeps a partial turn as
/// readily as a whole one, and handing the sweep's own vertices and edges back to the faces beside
/// it is straightforward enough. It was built that far, and it does not work.
///
/// What stops it is that a strip's four corners are precisely the vertices a BOOLEAN added. A mesh
/// of a circle is inscribed in it: the vertices the tool laid down are exactly on the circle, but
/// wherever something cut the wall, the new corner landed on a chord, a sagitta inside. A sweep
/// starts at one corner and arrives at the far one by exact rotation — and an inscribed corner is
/// not the rotation of another inscribed corner, so the two ends cannot both be met. Measured over
/// the corpus: eighteen strips detected, thirty-six rims to splice, FOUR whose corners happened to
/// be original vertices and matched. Every shell built from the rest failed to close, and the parts
/// that had good whole-ring cylinders lost them to the fallback.
///
/// So it is not a plumbing problem, and the next attempt should start from the geometry: the four
/// corners would have to be snapped onto the circle first — a move of about one sagitta, the same
/// order as the chord-to-arc correction this code already makes deliberately, but a change to the
/// part and therefore a decision, not an implementation detail.
///
/// TORUS bands get this far and no further, as of writing: the corpus has five, and none of them
/// lands. Three are turned away at the seam, because a fillet's rim meets the WALL it blends into
/// rather than a flat face — band against band, which would need the two tubes to share one rim
/// circle instead of each growing its own. The other two hold triangles that carry the torus tag
/// but do not sit on it within tolerance. The machinery is here and exercised; those two are what
/// is left.
fn curved_bands(mesh: &TriMesh, topo: &bevel::Topo) -> Vec<Band> {
    use std::collections::{HashMap, HashSet};
    let ntri = mesh.indices.len() / 3;
    if mesh.tri_surf.len() != ntri || mesh.surfaces.is_empty() || topo.tris.is_empty() {
        return Vec::new();
    }
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &topo.verts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1.0);
    // The weld snaps to a 1e-5 grid, so nothing is known finer than that however exact the sketch.
    let tol = (diag * 1.0e-4).max(1.0e-4);

    // Mesh triangles are not topo triangles — the weld drops any it collapses and re-orients the
    // rest — so go through the welded vertex triple, which both agree on.
    let mut vmap: HashMap<(i64, i64, i64), usize> = HashMap::new();
    for (i, p) in topo.verts.iter().enumerate() {
        vmap.insert(bevel::weld_key(*p), i);
    }
    let mut tmap: HashMap<[usize; 3], usize> = HashMap::new();
    for (i, t) in topo.tris.iter().enumerate() {
        let mut k = *t;
        k.sort_unstable();
        tmap.insert(k, i);
    }

    // The axis and the "core" each surface is a fixed distance from: the axis line for a cylinder,
    // the centre circle for a torus.
    let frame = |s: &Surf| -> Option<([f64; 3], [f64; 3], f64, f64)> {
        match *s {
            Surf::Cylinder { origin, axis, radius } => Some((origin, axis, 0.0, radius)),
            Surf::Torus { origin, axis, major, minor } => Some((origin, axis, major, minor)),
            Surf::Plane { .. } => None,
        }
    };
    // Tags that name the SAME surface are one band. A surface is recorded with a point on its axis,
    // and any point on the axis will do — so the same bore picked up a different record from every
    // feature that touched it. Compared as written, one wall read as three unrelated cylinders,
    // none of them a whole tube.
    let same = |a: &Surf, b: &Surf| {
        let (Some((o0, n0, m0, r0)), Some((o1, n1, m1, r1))) = (frame(a), frame(b)) else { return false };
        if std::mem::discriminant(a) != std::mem::discriminant(b) {
            return false;
        }
        let dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
        let d = [o1[0] - o0[0], o1[1] - o0[1], o1[2] - o0[2]];
        let al = d[0] * n0[0] + d[1] * n0[1] + d[2] * n0[2];
        let off = [d[0] - n0[0] * al, d[1] - n0[1] * al, d[2] - n0[2] * al];
        (m0 - m1).abs() < tol
            && (r0 - r1).abs() < tol
            && dot.abs() > 1.0 - 1.0e-9
            && (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]).sqrt() < tol
            // A cylinder's origin may sit anywhere along the axis; a torus's names the plane its
            // centre circle lies in, so that has to agree too.
            && (matches!(a, Surf::Cylinder { .. }) || al.abs() < tol)
    };
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (si, surf) in mesh.surfaces.iter().enumerate() {
        if frame(surf).is_none() {
            continue;
        }
        match groups.iter_mut().find(|g| same(&mesh.surfaces[g[0]], surf)) {
            Some(g) => g.push(si),
            None => groups.push(vec![si]),
        }
    }

    // Which surface each topo face lies on, for reporting who holds a rim.
    let mut face_tag: Vec<u32> = vec![NO_SURF; topo.faces.len()];
    for t in 0..ntri {
        let mut tri = [0usize; 3];
        let mut ok = true;
        for i in 0..3 {
            let q = mesh.positions[mesh.indices[t * 3 + i] as usize];
            match vmap.get(&bevel::weld_key([q[0] as f64, q[1] as f64, q[2] as f64])) {
                Some(&v) => tri[i] = v,
                None => ok = false,
            }
        }
        if !ok { continue; }
        tri.sort_unstable();
        if tri[0] == tri[1] || tri[1] == tri[2] { continue; }
        if let Some(&ti) = tmap.get(&tri) {
            if mesh.tri_surf[t] != NO_SURF {
                face_tag[topo.tri_face[ti]] = mesh.tri_surf[t];
            }
        }
    }

    let mut cands: Vec<Cand> = Vec::new();
    let mut taken: HashSet<usize> = HashSet::new(); // topo faces already claimed by a candidate
    for group in &groups {
        let surf = mesh.surfaces[group[0]];
        let Some((origin, axis, major, minor)) = frame(&surf) else { continue };
        if minor < tol {
            continue;
        }
        // The topo faces these tags cover.
        let mut faces: HashSet<usize> = HashSet::new();
        let mut ok = true;
        for t in 0..ntri {
            if !group.contains(&(mesh.tri_surf[t] as usize)) {
                continue;
            }
            let mut tri = [0usize; 3];
            for i in 0..3 {
                let p = mesh.positions[mesh.indices[t * 3 + i] as usize];
                match vmap.get(&bevel::weld_key([p[0] as f64, p[1] as f64, p[2] as f64])) {
                    Some(&v) => tri[i] = v,
                    None => ok = false,
                }
            }
            if !ok {
                break;
            }
            tri.sort_unstable();
            if tri[0] == tri[1] || tri[1] == tri[2] {
                continue; // the weld collapsed it; it carries no surface
            }
            match tmap.get(&tri) {
                Some(&ti) => {
                    faces.insert(topo.tri_face[ti]);
                }
                None => ok = false,
            }
            if !ok {
                break;
            }
        }
        if !ok || faces.is_empty() {
            continue;
        }
        let all: HashSet<usize> = faces.iter().flat_map(|&f| topo.faces[f].tris.iter().copied()).collect();

        // One tag routinely covers SEVERAL separate patches — a bore through two plates with a gap,
        // a ring of identical holes, the same wall revisited by a later feature — so judge each
        // connected patch on its own rather than the tag as a whole.
        let mut comp: HashMap<usize, usize> = HashMap::new();
        let mut order: Vec<usize> = all.iter().copied().collect();
        order.sort_unstable(); // walked in index order so the bands come out the same every run
        let mut parts: Vec<Vec<usize>> = Vec::new();
        let mut edge_tri: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for &ti in &order {
            let t = topo.tris[ti];
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edge_tri.entry(if a < b { (a, b) } else { (b, a) }).or_default().push(ti);
            }
        }
        for &ti in &order {
            if comp.contains_key(&ti) {
                continue;
            }
            let id = parts.len();
            let mut stack = vec![ti];
            let mut part = Vec::new();
            while let Some(x) = stack.pop() {
                if comp.insert(x, id).is_some() {
                    continue;
                }
                part.push(x);
                let t = topo.tris[x];
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    for &n in edge_tri.get(&(if a < b { (a, b) } else { (b, a) })).into_iter().flatten() {
                        if !comp.contains_key(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
            part.sort_unstable();
            parts.push(part);
        }

        // Where a point sits in the half-plane the surface is revolved from, and how far it is off
        // the surface itself.
        let place = |v: usize| -> ([f64; 2], f64, [f64; 3]) {
            let p = topo.verts[v];
            let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
            let al = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
            let r = [d[0] - axis[0] * al, d[1] - axis[1] * al, d[2] - axis[2] * al];
            let rl = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            ([rl, al], ((rl - major).powi(2) + (al - 0.0).powi(2) * f64::from(major > 0.0)).sqrt(), r)
        };
        // For a cylinder the surface is "radius from the axis", for a torus "minor from the centre
        // circle"; `place` returns whichever applies.
        let off_surface = |v: usize| {
            let (g, o, _) = place(v);
            if major > 0.0 { o - minor } else { g[0] - minor }
        };
        // A frame across the axis, to order each rim.
        let t0 = if axis[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = {
            let c = [
                axis[1] * t0[2] - axis[2] * t0[1],
                axis[2] * t0[0] - axis[0] * t0[2],
                axis[0] * t0[1] - axis[1] * t0[0],
            ];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1.0e-12);
            [c[0] / l, c[1] / l, c[2] / l]
        };
        let vv = [
            axis[1] * u[2] - axis[2] * u[1],
            axis[2] * u[0] - axis[0] * u[2],
            axis[0] * u[1] - axis[1] * u[0],
        ];
        let angle = |vtx: usize| {
            let (_, _, r) = place(vtx);
            (r[0] * vv[0] + r[1] * vv[1] + r[2] * vv[2]).atan2(r[0] * u[0] + r[1] * u[1] + r[2] * u[2])
        };

        for part in &parts {
            let why = |g: &str| if std::env::var("HCAD_BAND_DEBUG").is_ok() {
                eprintln!("  reject [{:?} r={:.3} part of {} tris]: {g}", std::mem::discriminant(&surf), minor, part.len());
            };
            let tagged: HashSet<usize> = part.iter().copied().collect();
            let faces: HashSet<usize> = part.iter().map(|&ti| topo.tri_face[ti]).collect();
            if faces.iter().any(|f| taken.contains(f)) {
                why("a face is already claimed by another band");
                continue;
            }
            // A face split across two patches cannot be swapped out with either.
            if faces.iter().any(|&f| topo.faces[f].tris.iter().any(|t| !tagged.contains(t))) {
                why("a face is split across two patches");
                continue;
            }
            let verts: HashSet<usize> = tagged.iter().flat_map(|&ti| topo.tris[ti]).collect();
            // Every vertex on the surface the tag names.
            // A mesh of a curved surface is INSCRIBED in it. The tool's own vertices sit exactly
            // on the surface, but every vertex a boolean adds along an intersection lands on the
            // CHORD between them — a sagitta short, always on the inside. Judging by a flat
            // tolerance asks the mesh to be something it never was: four vertices of ball's fillet
            // and twenty of bottomline's sat 1.1 to 1.2 times the tolerance in, which is to say
            // exactly where an inscribed facet puts them, and cost both parts their torus.
            //
            // So allow the inside by what this patch's OWN facets imply — a chord of length L on a
            // surface curving at `minor` falls short by about L²/8minor — and the outside by
            // nothing but noise, because there is no mechanism that puts a vertex out there.
            let sagitta = tagged
                .iter()
                .map(|&ti| {
                    let t = topo.tris[ti];
                    let e = |a: usize, b: usize| {
                        let (p, q) = (topo.verts[t[a]], topo.verts[t[b]]);
                        (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)
                    };
                    e(0, 1).max(e(1, 2)).max(e(2, 0))
                })
                .fold(0.0f64, f64::max)
                / (8.0 * minor);
            if verts.iter().any(|&v| {
                let d = off_surface(v);
                d > tol || d < -(tol + sagitta)
            }) {
                let worst = verts.iter().map(|&v| off_surface(v)).fold(0.0f64, |m, d| if d.abs() > m.abs() { d } else { m });
                why(&format!("a vertex is {worst:.2e} off the surface (tol {tol:.2e}, sagitta {sagitta:.2e})"));
                continue;
            }
            // The patch's BOUNDARY: the edges only one of its triangles holds. A plain ring has
            // exactly two, one at each end.
            //
            // Asking instead that every vertex sit at one of two heights — the obvious reading of
            // "straight band" — threw away almost everything: fifty of the corpus's bands failed it
            // and four parts came out round. A boolean subdivides a wall wherever anything else met
            // it, so a perfectly plain bore routinely carries vertices part way along. Those are
            // interior to the patch and vanish with the strips that hold them; only the two ends
            // have to be circles.
            let mut ecount: HashMap<(usize, usize), usize> = HashMap::new();
            for &ti in &tagged {
                let t = topo.tris[ti];
                for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    *ecount.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
                }
            }
            let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
            for (&(a, b), &c) in ecount.iter() {
                if c == 1 {
                    adj.entry(a).or_default().push(b);
                    adj.entry(b).or_default().push(a);
                }
            }
            // Sorted, because the walk below starts at each vertex's FIRST neighbour and that
            // decides which way round the rim it goes — and a rim walked the other way averages its
            // coordinates in the other order, which moves the tube's vertices in their last bits
            // and hands out a different STEP file every time.
            for n in adj.values_mut() {
                n.sort_unstable();
            }
            if adj.is_empty() || adj.values().any(|n| n.len() != 2) {
                why(&format!("the boundary is not a clean set of rings ({} verts, {} with a degree other than 2)",
                    adj.len(), adj.values().filter(|n| n.len() != 2).count()));
                continue;
            }
            let mut loops: Vec<Vec<usize>> = Vec::new();
            let mut walked: HashSet<usize> = HashSet::new();
            let mut starts: Vec<usize> = adj.keys().copied().collect();
            starts.sort_unstable(); // a hash order here would make the export non-reproducible
            for s0 in starts {
                if !walked.insert(s0) {
                    continue;
                }
                let (mut lp, mut prev, mut cur) = (vec![s0], s0, adj[&s0][0]);
                while cur != s0 && walked.insert(cur) {
                    lp.push(cur);
                    let n = &adj[&cur];
                    let nxt = if n[0] == prev { n[1] } else { n[0] };
                    prev = cur;
                    cur = nxt;
                }
                loops.push(lp);
            }
            if loops.len() != 2 {
                why(&format!("{} boundary loop(s), not 2", loops.len()));
                continue;
            }
            // Each end a CIRCLE: one distance from the axis and one along it, all the way round.
            let ring = |lp: &Vec<usize>| -> Option<[f64; 2]> {
                let mut c = [0.0f64; 2];
                for &v in lp {
                    let g = place(v).0;
                    c[0] += g[0];
                    c[1] += g[1];
                }
                let c = [c[0] / lp.len() as f64, c[1] / lp.len() as f64];
                lp.iter()
                    .all(|&v| {
                        let g = place(v).0;
                        (g[0] - c[0]).abs() < tol && (g[1] - c[1]).abs() < tol
                    })
                    .then_some(c)
            };
            let (Some(g0), Some(g1)) = (ring(&loops[0]), ring(&loops[1])) else {
                why("an end is not a circle");
                continue;
            };
            if ((g0[0] - g1[0]).powi(2) + (g0[1] - g1[1]).powi(2)).sqrt() < tol {
                why("the two ends are in the same place");
                continue;
            }
            // Order the rims so the generatrix runs the same way every time.
            let mut rims: [Vec<usize>; 2] = [std::mem::take(&mut loops[0]), std::mem::take(&mut loops[1])];
            let mut gen = [g0, g1];
            if (gen[1][1], gen[1][0]) < (gen[0][1], gen[0][0]) {
                rims.swap(0, 1);
                gen.swap(0, 1);
            }
            if rims.iter().any(|r| r.len() < 8) {
                why(&format!("a rim has too few vertices ({:?})", rims.iter().map(|r| r.len()).collect::<Vec<_>>()));
                continue;
            }
            for r in rims.iter_mut() {
                r.sort_by(|&a, &b| angle(a).partial_cmp(&angle(b)).unwrap_or(std::cmp::Ordering::Equal));
            }
            // What these triangles contribute to the body's volume, as they stand. The exact tube
            // has to contribute the same thing, give or take the sagitta it corrects.
            let mut patch_tris: Vec<usize> = tagged.iter().copied().collect();
            patch_tris.sort_unstable(); // f64 addition is not associative, so the order is the answer
            let patch_volume: f64 = patch_tris
                .iter()
                .map(|&ti| {
                    let t = topo.tris[ti];
                    let (a, b, c) = (topo.verts[t[0]], topo.verts[t[1]], topo.verts[t[2]]);
                    (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2])
                        + a[2] * (b[0] * c[1] - b[1] * c[0]))
                        / 6.0
                })
                .sum();
            why("a candidate");
            taken.extend(faces.iter().copied());
            let mut fs: Vec<usize> = faces.into_iter().collect();
            fs.sort_unstable();
            cands.push(Cand {
                origin,
                axis,
                // A torus's own origin names the plane its centre circle lies in, so in its own
                // half-plane that centre sits at (major, 0).
                arc: (major > 0.0).then_some(([major, 0.0], minor)),
                gen,
                rims,
                faces: fs,
                patch_volume,
            });
        }
    }

    chain_candidates(topo, cands, tol)
}

/// One patch that could be swapped for an exact surface, before anything is known about how it
/// meets the rest of the body.
struct Cand {
    origin: [f64; 3],
    axis: [f64; 3],
    arc: Option<([f64; 2], f64)>,
    gen: [[f64; 2]; 2],
    rims: [Vec<usize>; 2],
    faces: Vec<usize>,
    patch_volume: f64,
}

/// Join candidates that share a rim into single bands, and keep the ones that meet the rest of the
/// body cleanly.
///
/// A rim has to stop existing — a hundred-odd vertices become one circle — so every face still
/// holding one of those vertices has to be a face that knows about the change. Only two kinds do:
/// a flat face that walks the whole rim as one of its loops, which is handed the tube's circle
/// instead, and ANOTHER CANDIDATE that shares the rim, which is swept in the same breath so the
/// rim never becomes a boundary at all.
///
/// That second kind is what a filleted bore is made of. The wall is a cylinder, each mouth is the
/// torus a rolling ball left, and each meets the next along a circle no flat face walks — so
/// judged one at a time all three are refused, every one waiting on a rim the others cannot seam
/// either. Chained, they are one generatrix (arc, line, arc) revolved once, ending on the flat
/// faces at the far ends.
fn chain_candidates(topo: &bevel::Topo, cands: Vec<Cand>, tol: f64) -> Vec<Band> {
    use std::collections::{HashMap, HashSet};
    let mut out: Vec<Band> = Vec::new();
    if cands.is_empty() {
        return out;
    }
    let debug = std::env::var("HCAD_BAND_DEBUG").is_ok();
    let cand_faces: HashSet<usize> = cands.iter().flat_map(|c| c.faces.iter().copied()).collect();

    // Which rims are the SAME rim: two candidates meeting along one hold identical vertex sets.
    let mut by_rim: HashMap<Vec<usize>, Vec<(usize, usize)>> = HashMap::new();
    for (i, c) in cands.iter().enumerate() {
        for k in 0..2 {
            let mut key = c.rims[k].clone();
            key.sort_unstable();
            by_rim.entry(key).or_default().push((i, k));
        }
    }
    let mut link: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; cands.len()];
    // Walked in candidate order, so no hash order decides anything.
    for (i, c) in cands.iter().enumerate() {
        for k in 0..2 {
            let mut key = c.rims[k].clone();
            key.sort_unstable();
            // Exactly two: three patches meeting on one circle is not a chain.
            if let Some(v) = by_rim.get(&key) {
                if v.len() == 2 {
                    link[i][k] = Some(if v[0] == (i, k) { v[1] } else { v[0] });
                }
            }
        }
    }

    // A rim with no candidate on the far side needs a flat face walking the whole of it. Faces
    // belonging to ANY candidate are excluded: those are about to become a tube themselves, so a
    // loop of one is not something a rim can be handed to.
    let mut flat: Vec<[Option<(usize, usize)>; 2]> = vec![[None, None]; cands.len()];
    for (i, c) in cands.iter().enumerate() {
        for k in 0..2 {
            if link[i][k].is_some() {
                continue;
            }
            let want: HashSet<usize> = c.rims[k].iter().copied().collect();
            let (mut found, mut clash) = (None, false);
            for (fi, f) in topo.faces.iter().enumerate() {
                if cand_faces.contains(&fi) {
                    continue;
                }
                for (li, lp) in f.loops.iter().enumerate() {
                    if lp.len() == want.len() && lp.iter().all(|v| want.contains(v)) {
                        clash |= found.replace((fi, li)).is_some();
                    }
                }
            }
            flat[i][k] = found.filter(|_| !clash);
        }
    }

    // Walk each chain from a free end. A candidate has at most one link per rim, so the links form
    // paths and rings; a ring has no end to finish on, and is left alone.
    let mut used = vec![false; cands.len()];
    for seed in 0..cands.len() {
        if used[seed] || (link[seed][0].is_some() && link[seed][1].is_some()) {
            continue;
        }
        let head = usize::from(link[seed][0].is_some()); // the rim this chain starts on
        let mut chain: Vec<(usize, usize)> = Vec::new(); // (candidate, the rim it is entered by)
        let (mut at, mut enter) = (seed, head);
        while !used[at] {
            used[at] = true;
            chain.push((at, enter));
            match link[at][1 - enter] {
                Some((j, l)) => {
                    at = j;
                    enter = l;
                }
                None => break,
            }
        }
        let (&(first, first_rim), &(last, last_rim)) = (chain.first().unwrap(), chain.last().unwrap());
        let ends = [(first, first_rim), (last, 1 - last_rim)];
        let why = |g: &str| {
            if debug {
                let parts: Vec<String> = chain
                    .iter()
                    .map(|&(i, _)| match cands[i].arc {
                        Some((_, m)) => format!("arc {m:.3}"),
                        None => format!("line r={:.3}", cands[i].gen[0][0]),
                    })
                    .collect();
                eprintln!("  chain [{}]: {g}", parts.join(" + "));
            }
        };

        // Both ends have to land on a flat face...
        let (Some(s0), Some(s1)) = (flat[ends[0].0][ends[0].1], flat[ends[1].0][ends[1].1]) else {
            why("an end is not walked by exactly one outside loop");
            continue;
        };
        // ...and every stretch must turn about the SAME axis, or there is no one half-plane the
        // chain can be drawn in.
        // Normalised, because everything downstream multiplies by it TWICE — the rim's distance
        // along the axis is read off with a dot product and then written back with a scale — so an
        // axis 2.4e-7 long in error puts the rim 9.5e-7 out of the plane it is supposed to lie in,
        // at which point truck leaves the whole face it meets unfilled. Axes are written down by
        // whichever feature drew them and are not always unit vectors.
        let origin = cands[first].origin;
        let axis = {
            let a = cands[first].axis;
            let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            if l < 1.0e-12 {
                continue;
            }
            [a[0] / l, a[1] / l, a[2] / l]
        };
        // Judged to the model's own tolerance, not to the last bit: an axis is written down by
        // whichever feature drew it and is not always a unit vector to the last ulp — a threshold
        // of 1e-9 on the dot product rejected a candidate against ITSELF.
        if chain.iter().any(|&(i, _)| {
            let d = cands[i].axis;
            let dot = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
            dot.abs() < 1.0 - 1.0e-6 || axis_gap(cands[i].origin, d, origin) > tol
        }) {
            why("its stretches do not share one axis");
            continue;
        }

        // The generatrix, stretch by stretch, every one rewritten in the first stretch's frame.
        let segs: Vec<Seg> = chain
            .iter()
            .map(|&(i, enter)| {
                let c = &cands[i];
                let sign = (c.axis[0] * axis[0] + c.axis[1] * axis[1] + c.axis[2] * axis[2]).signum();
                let d = [c.origin[0] - origin[0], c.origin[1] - origin[1], c.origin[2] - origin[2]];
                let shift = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
                let to_chain = |g: [f64; 2]| [g[0], shift + sign * g[1]];
                Seg {
                    arc: c.arc.map(|(centre, minor)| (to_chain(centre), minor)),
                    gen: [to_chain(c.gen[enter]), to_chain(c.gen[1 - enter])],
                }
            })
            .collect();

        // Each end of the chain has to lie IN the face it meets, and truck means that to a
        // millionth of a model unit. Where the rim sits was read off the mesh — the mean of its
        // vertices' distance along the axis — and those vertices only have to agree to the model
        // tolerance, which is a hundred times looser. barthing's bore ends 1.2e-4 adrift of the
        // flat face at its mouth: truck accepts the face and then declines to fill it, which takes
        // 6.4% of the body with it and condemns the whole banded build.
        //
        // Checked rather than corrected. Moving the rim onto the plane looks like the fix and is
        // not: the plane runs through ONE vertex of the face, and at the sizes parts are drawn at
        // an f32 position is itself only good to about a millionth — so snapping a circle from the
        // mean of a hundred vertices onto a single quantised one is a step away from the plane as
        // often as toward it. It cost motormount all seven of its bores. The mean is the better
        // estimate; what is worth knowing is whether it lands close enough, and when it does not,
        // this is not a rim that can be handed to a flat face at all.
        let mut ok = true;
        for (end, seam) in [(0usize, s0), (1usize, s1)] {
            let Some((o, n)) = face_plane(topo, seam.0) else {
                ok = false;
                break;
            };
            // A plane not square to the axis cannot hold the circle however it is placed — and a
            // rim leaving through one would be an ellipse, not something this can rebuild.
            if (n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2]).abs() < 1.0 - 1.0e-6 {
                why("an end meets a face that is not square to the axis");
                ok = false;
                break;
            }
            let d = [o[0] - origin[0], o[1] - origin[1], o[2] - origin[2]];
            let al = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
            let sg = if end == 0 { segs.first() } else { segs.last() };
            let Some(sg) = sg else {
                ok = false;
                break;
            };
            let off = (sg.gen[end][1] - al).abs();
            if debug {
                eprintln!("      end {end}: face {} plane at {al:.9}, rim at {:.9}, off {off:.3e}", seam.0, sg.gen[end][1]);
            }
            if off > TRUCK_TOLERANCE {
                why(&format!("an end sits {off:.2e} off the plane of the face it meets"));
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        let faces: Vec<usize> = {
            let mut f: Vec<usize> = chain.iter().flat_map(|&(i, _)| cands[i].faces.iter().copied()).collect();
            f.sort_unstable();
            f
        };
        // Every rim the chain swallows, not only its two ends: an internal one stops existing just
        // as surely, so a third face holding one of its vertices is the same fault.
        let allowed: HashSet<usize> = faces.iter().copied().chain([s0.0, s1.0]).collect();
        if chain
            .iter()
            .flat_map(|&(i, _)| cands[i].rims.iter().flatten().copied())
            .any(|v| topo.vert_faces[v].iter().any(|f| !allowed.contains(f)))
        {
            why("a rim vertex is held by a face the chain does not cover");
            continue;
        }

        why("ACCEPTED");
        out.push(Band {
            segs,
            origin,
            axis,
            rims: [cands[ends[0].0].rims[ends[0].1].clone(), cands[ends[1].0].rims[ends[1].1].clone()],
            faces,
            seam: [s0, s1],
            patch_volume: chain.iter().map(|&(i, _)| cands[i].patch_volume).sum(),
        });
    }
    out
}


/// Rebuild a mesh as a B-rep solid, merging COPLANAR triangles into whole planar faces.
///
/// One face per triangle is a valid solid and a useless one: blocker.hcad went out as 11,668
/// planar faces and 13 MB of STEP, every flat wall shattered into hundreds of slivers, nothing
/// downstream able to grab a face. `build_topo` already groups coplanar, edge-adjacent triangles
/// into faces with ordered boundary loops — it is how the bevel finds a model's real faces — so
/// the same grouping hands the export whole walls instead. Measured: pinch 1,244 triangles → 313
/// faces, motormount 5,828 → 1,442.
///
/// A bore the sketch drew as a circle is written as a real cylinder — three
/// SURFACE_OF_REVOLUTION faces in place of forty-eight flat strips — wherever the mesh still
/// shows the plain ring the sketch described. See [`cylinder_bands`] for what "plain" has to
/// mean, and why anything less is left faceted.
///
/// Merging is *attempted*, never assumed. Each face must cover the area of the triangles it
/// replaces, and the finished solid must still enclose what the mesh did; whatever fails either
/// test drops a rung, down to the faceted build, which is exact by construction.
pub fn mesh_to_solid(mesh: &TriMesh) -> Option<KSolid> {
    if mesh.indices.len() < 12 {
        return None; // fewer than four triangles cannot bound anything
    }
    let topo = bevel::build_topo(mesh);
    if topo.tris.len() < 4 || topo.faces.is_empty() {
        return None;
    }
    let want = signed_mesh_volume(mesh).abs();
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &topo.verts {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1.0);
    // Merged, then judged as a whole. truck's triangulator is known to give up on an awkward
    // planar boundary — `extrude_tool_mesh` guards the same thing — and a face it silently drops
    // leaves a hole in the shell, which shows up as a wild volume rather than as an error. So
    // tessellate what we built and demand it still encloses what the mesh did.
    if want > 1.0e-9 {
        // Judged by tessellating what was built and seeing whether it still encloses what the mesh
        // did. With cylinders in it that has to be done FINELY: a tessellation is a polygon
        // approximation of the curve, so a coarse one of a true bore reads a couple of percent
        // light and would condemn a perfectly good solid. The cost lands only on export.
        let judge = |s: &KSolid, want: f64, tol: f64| {
            let re = tessellate(s, tol).mesh;
            !re.indices.is_empty() && (signed_mesh_volume(&re).abs() - want).abs() <= want * 1.0e-3
        };
        let bands = curved_bands(mesh, &topo);
        if !bands.is_empty() {
            // A true circle is wider than the chords that stood in for it, so the body is
            // legitimately a little different from the mesh. That is a correction, not drift — the
            // part always was meant to have a round hole — and the check has to EXPECT it, or it
            // would reject its own good work.
            //
            // The size of it is measured rather than derived: each exact tube contributes to the
            // body's volume exactly as the triangles it replaces did, so the difference between
            // those two contributions is the correction, whatever shape the band is. A formula for
            // the circle-against-polygon case would not have covered a fillet's torus.
            let corrected = want
                + bands
                    .iter()
                    .filter_map(|b| band_tube(b).map(|(_, _, v)| v - b.patch_volume))
                    .sum::<f64>();
            match mesh_brep(&topo, true, &bands) {
                Some(s) if corrected > 1.0e-9 && judge(&s, corrected, diag * 1.0e-5) => return Some(s),
                built => {
                    // A band that was accepted and still does not ship: either the shell would not
                    // close, or what was built does not enclose what the mesh did.
                    if std::env::var("HCAD_BAND_DEBUG").is_ok() {
                        match built {
                            None => eprintln!("  the {} band(s) did not close a shell", bands.len()),
                            Some(s) => {
                                let got = signed_mesh_volume(&tessellate(&s, diag * 1.0e-5).mesh).abs();
                                let tri = s.0.triangulation(diag * 1.0e-5);
                                let unfilled = tri.boundaries()[0].iter().filter(|f| f.surface().is_none()).count();
                                eprintln!(
                                    "  the {} band(s) built {} face(s), {unfilled} of them unfilled, enclosing {got:.4} against {corrected:.4} ({:+.4}%)",
                                    bands.len(),
                                    s.0.boundaries()[0].len(),
                                    100.0 * (got - corrected) / corrected.max(1.0e-9)
                                );
                            }
                        }
                    }
                }
            }
        }
        if let Some(s) = mesh_brep(&topo, true, &[]) {
            if judge(&s, want, diag * 1.0e-3) {
                return Some(s);
            }
        }
    }
    mesh_brep(&topo, false, &[])
}

/// Build a band as a real surface of revolution: its generatrix — a line for a cylinder, an arc for
/// the tube a rolling ball left — swept a whole turn about the axis, the way truck builds one
/// itself. Returns the faces, each rim as the tube walks it, and what the tube contributes to the
/// body's volume.
///
/// Hand-rolling the sweep was the obvious approach and it does not work. Three arcs through the
/// rim's own vertices, revolved, gives a shell truck ACCEPTS as a closed solid and then tessellates
/// inside-out — one face of the three, sometimes two, with no pattern to it: a 48-sided rim came out
/// right, 49 lost 88% of the volume, 50 was right again, 51 lost 44%, 60 lost 89%. The faces are
/// geometrically identical either way; what differs is that truck has to recover each boundary
/// curve's parameters on the revolved surface, and for arcs it did not build itself that search
/// lands on the wrong branch often enough to be useless. Swept truck's own way it is exact to a
/// thousandth of a percent at every radius, height and start angle tried.
fn band_tube(b: &Band) -> Option<(Vec<truck_modeling::Face>, [truck_modeling::Wire; 2], f64)> {
    let origin = Point3::new(b.origin[0], b.origin[1], b.origin[2]);
    let axis = Vector3::new(b.axis[0], b.axis[1], b.axis[2]);
    let across = if b.axis[0].abs() < 0.9 { Vector3::unit_x() } else { Vector3::unit_y() };
    let out = axis.cross(across).normalize();
    // The half-plane the surface is revolved from: `g[0]` out from the axis, `g[1]` along it.
    let at = |g: [f64; 2]| origin + out * g[0] + axis * g[1];
    // The generatrix, chained end to end and swept ONCE. Sweeping each stretch on its own and
    // stitching them afterwards cannot work: two sweeps of the same circle produce equal-looking
    // but separate edges, and a shell holding both is not closed. Swept together, the rim between
    // two stretches is one edge belonging to both by construction.
    let (first, last) = (b.segs.first()?, b.segs.last()?);
    let mut wire = truck_modeling::Wire::new();
    let mut v_prev = builder::vertex(at(first.gen[0]));
    for sg in &b.segs {
        let v_next = builder::vertex(at(sg.gen[1]));
        let e = match sg.arc {
            None => builder::line(&v_prev, &v_next),
            Some((c, minor)) => {
                // The arc between the two ends, through the point half way round its centre.
                let ang = |g: [f64; 2]| (g[1] - c[1]).atan2(g[0] - c[0]);
                let (a0, a1) = (ang(sg.gen[0]), ang(sg.gen[1]));
                // The short way round: a rolling-ball fillet never spans more than half the tube.
                let mut d = a1 - a0;
                if d > std::f64::consts::PI {
                    d -= std::f64::consts::TAU;
                } else if d < -std::f64::consts::PI {
                    d += std::f64::consts::TAU;
                }
                let mid = a0 + d * 0.5;
                builder::circle_arc(&v_prev, &v_next, at([c[0] + minor * mid.cos(), c[1] + minor * mid.sin()]))
            }
        };
        wire.push_back(e);
        v_prev = v_next;
    }
    let swept = builder::rsweep(&wire, origin, axis, truck_modeling::Rad(std::f64::consts::TAU));
    let mut faces: Vec<truck_modeling::Face> = swept.iter().cloned().collect();

    // Which side the material is on is NOT something the shell can check: the flat face beside a rim
    // is given whatever direction the tube leaves it, so a tube threaded in backwards still closes
    // into a perfectly valid solid — one with a collar of material where the hole should be.
    // Nothing complains; only the volume tells you, and it went in backwards first time round.
    //
    // So ask the volume directly. A patch's contribution to the body it bounds is the same sum that
    // gives a closed mesh its volume, and the tube is standing in for the mesh triangles it
    // replaces: if its contribution comes out with the opposite sign, it is facing the wrong way.
    let contribution = |fs: &[truck_modeling::Face]| -> Option<f64> {
        let shell: truck_modeling::Shell = fs.iter().cloned().collect();
        guard(|| {
            let scale = b.segs.iter().flat_map(|sg| sg.gen).map(|g| g[0].abs()).fold(1.0f64, f64::max);
            let mut poly = shell.triangulation(scale * 1.0e-3).to_polygon();
            poly.triangulate();
            Some(signed_mesh_volume(&polymesh_to_trimesh(&poly)))
        })
    };
    let mut vol = contribution(&faces)?;
    if (vol > 0.0) != (b.patch_volume > 0.0) {
        faces = faces.iter().map(|f| f.inverse()).collect();
        vol = -vol;
    }

    // The tube's own boundary is the two rim circles. Take them with the orientation the tube gives
    // them, so the flat face that shares a rim can simply walk them backwards.
    let key = |e: &truck_modeling::Edge| {
        let q = |p: Point3| ((p.x * 1.0e5).round() as i64, (p.y * 1.0e5).round() as i64, (p.z * 1.0e5).round() as i64);
        let (a, z) = (q(e.front().point()), q(e.back().point()));
        if a < z { (a, z) } else { (z, a) }
    };
    let mut seen: std::collections::HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = Default::default();
    for f in &faces {
        for w in f.boundaries() {
            for e in w.iter() {
                *seen.entry(key(e)).or_default() += 1;
            }
        }
    }
    let mut rims: [Vec<truck_modeling::Edge>; 2] = [Vec::new(), Vec::new()];
    for f in &faces {
        for w in f.boundaries() {
            for e in w.iter() {
                if seen.get(&key(e)).copied().unwrap_or(0) != 1 {
                    continue;
                }
                let p = e.front().point() - origin;
                let al = p.dot(axis);
                let rl = (p - axis * al).magnitude();
                let d = |g: [f64; 2]| (rl - g[0]).powi(2) + (al - g[1]).powi(2);
                rims[usize::from(d(last.gen[1]) < d(first.gen[0]))].push(e.clone());
            }
        }
    }
    // Chain each rim's edges head to tail; they are a circle, so they must form one closed run.
    let chained = |es: &Vec<truck_modeling::Edge>| -> Option<truck_modeling::Wire> {
        let q = |p: Point3| ((p.x * 1.0e5).round() as i64, (p.y * 1.0e5).round() as i64, (p.z * 1.0e5).round() as i64);
        let mut left = es.clone();
        if left.is_empty() {
            return None;
        }
        let first = left.remove(0);
        let mut at = q(first.back().point());
        let mut w = truck_modeling::Wire::new();
        w.push_back(first);
        while !left.is_empty() {
            let i = left.iter().position(|e| q(e.front().point()) == at)?;
            let e = left.remove(i);
            at = q(e.back().point());
            w.push_back(e);
        }
        w.is_closed().then_some(w)
    };
    Some((faces, [chained(&rims[0])?, chained(&rims[1])?], vol))
}


/// The loops of a flat face, widest first — the order `truck` reads as outer-then-holes.
fn face_loops(topo: &bevel::Topo, fi: usize) -> Vec<(usize, &Vec<usize>)> {
    let span = |lp: &Vec<usize>| {
        let (mut a, mut b) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &v in lp {
            for k in 0..3 {
                a[k] = a[k].min(topo.verts[v][k]);
                b[k] = b[k].max(topo.verts[v][k]);
            }
        }
        (b[0] - a[0]) + (b[1] - a[1]) + (b[2] - a[2])
    };
    let mut lps: Vec<(usize, &Vec<usize>)> = topo.faces[fi].loops.iter().enumerate().collect();
    lps.sort_by(|x, y| span(y.1).partial_cmp(&span(x.1)).unwrap_or(std::cmp::Ordering::Equal));
    lps
}

/// The plane a flat face is written with: through the first vertex of its widest loop, facing the
/// way that loop winds.
///
/// The normal comes from the winding rather than from the face record because truck reads a planar
/// face's outside off its surface — a plane disagreeing with the boundary gives an inside-out face,
/// which cost 14% of motormount's volume before this said it properly. Everything that has to
/// agree about where a face lies asks here, so it cannot drift apart.
fn face_plane(topo: &bevel::Topo, fi: usize) -> Option<([f64; 3], [f64; 3])> {
    let lps = face_loops(topo, fi);
    let (_, outer) = lps.first()?;
    let mut n = [0.0f64; 3];
    for k in 0..outer.len() {
        let (p, q) = (topo.verts[outer[k]], topo.verts[outer[(k + 1) % outer.len()]]);
        n[0] += (p[1] - q[1]) * (p[2] + q[2]);
        n[1] += (p[2] - q[2]) * (p[0] + q[0]);
        n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    (l > 1.0e-12).then(|| (topo.verts[outer[0]], [n[0] / l, n[1] / l, n[2] / l]))
}

/// truck's own absolute tolerance (`truck_base::tolerance::TOLERANCE`). Anything the kernel is
/// asked to accept has to clear this, in model units, however flat it is in proportion.
const TRUCK_TOLERANCE: f64 = 1.0e-6;

/// A truck plane through `o` whose own normal — `u` × `v` — is `n`.
///
/// Which way it faces is the point: truck reads a planar face's outside off its surface, so a
/// plane built any old way gives an inside-out face.
fn plane_through(o: [f64; 3], n: [f64; 3]) -> Option<truck_modeling::Plane> {
    let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
    let ul = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    if ul <= 1.0e-12 {
        return None;
    }
    let u = [u[0] / ul, u[1] / ul, u[2] / ul];
    let v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
    Some(truck_modeling::Plane::new(
        Point3::new(o[0], o[1], o[2]),
        Point3::new(o[0] + u[0], o[1] + u[1], o[2] + u[2]),
        Point3::new(o[0] + v[0], o[1] + v[1], o[2] + v[2]),
    ))
}

/// The shell itself: one planar face per coplanar group when `merge`, otherwise one per triangle.
///
/// Every face is stitched from ONE `Edge` per undirected vertex pair, shared with whichever face
/// meets it there — neighbours holding different edge objects give a shell that is not closed,
/// whatever it looks like.
fn mesh_brep(topo: &bevel::Topo, merge: bool, bands: &[Band]) -> Option<KSolid> {
    use std::collections::HashMap;
    guard(|| {
        let verts: Vec<truck_modeling::Vertex> =
            topo.verts.iter().map(|p| builder::vertex(Point3::new(p[0], p[1], p[2]))).collect();
        let mut edges: HashMap<(usize, usize), truck_modeling::Edge> = HashMap::new();
        let mut faces: Vec<truck_modeling::Face> = Vec::new();
        // Each band as a finished tube, built up front because the SAME edges have to appear both
        // in the tube and in the flat face that meets it: two faces holding equal-looking but
        // separate edges give a shell that is not closed, whatever it looks like.
        let tubes: Vec<(Vec<truck_modeling::Face>, [truck_modeling::Wire; 2], f64)> =
            bands.iter().map(band_tube).collect::<Option<Vec<_>>>()?;
        // Which flat loop each band replaces, and which faces are the band itself.
        let mut seam_of: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut in_band: HashMap<usize, usize> = HashMap::new();
        for (bi, b) in bands.iter().enumerate() {
            for (rim, &(f, l)) in b.seam.iter().enumerate() {
                seam_of.insert((f, l), (bi, rim));
            }
            for &f in &b.faces {
                in_band.insert(f, bi);
            }
        }
        // A rim as the flat face beside it must walk it: against the tube, so that every edge of
        // the finished shell is travelled once each way.
        let rim_wire = |bi: usize, rim: usize| {
            let mut w = truck_modeling::Wire::new();
            for e in tubes[bi].1[rim].iter().rev() {
                w.push_back(truck_modeling::Edge::inverse(e));
            }
            w
        };
        macro_rules! wire {
            ($lp:expr) => {{
                let lp: &[usize] = $lp;
                if lp.len() < 3 {
                    None
                } else {
                    let mut w = truck_modeling::Wire::new();
                    let mut ok = true;
                    for k in 0..lp.len() {
                        let (x, y) = (lp[k], lp[(k + 1) % lp.len()]);
                        if x == y {
                            ok = false;
                            break;
                        }
                        let (a, b) = if x < y { (x, y) } else { (y, x) };
                        let e = edges.entry((a, b)).or_insert_with(|| builder::line(&verts[a], &verts[b])).clone();
                        w.push_back(if x < y { e } else { e.inverse() });
                    }
                    ok.then_some(w)
                }
            }};
        }
        for (fi, f) in topo.faces.iter().enumerate() {
            if in_band.contains_key(&fi) {
                continue; // written below as a real cylinder, not as the strips it was built from
            }
            let mut merged = None;
            if merge {
                // Outer boundary first: truck reads boundary 0 as the one the rest sit inside.
                let lps = face_loops(topo, fi);
                if let Some((o, n)) = face_plane(topo, fi) {
                    // Signed loop area about that normal — outer positive, holes negative — must
                    // come to the triangles' own area. That is what catches a hole promoted to
                    // outer, a loop dropped, or a "coplanar" group that is quietly curved: the
                    // flat projection of a bent band is smaller than the band. (build_topo groups
                    // by the angle between NEIGHBOURING facets, so a long enough chain of small
                    // steps bends a long way without any pair of them ever disagreeing.)
                    let loop_area: f64 = lps
                        .iter()
                        .map(|(_, lp)| {
                            let mut s = [0.0f64; 3];
                            for k in 0..lp.len() {
                                let (p, q) = (topo.verts[lp[k]], topo.verts[lp[(k + 1) % lp.len()]]);
                                s[0] += p[1] * q[2] - p[2] * q[1];
                                s[1] += p[2] * q[0] - p[0] * q[2];
                                s[2] += p[0] * q[1] - p[1] * q[0];
                            }
                            0.5 * (s[0] * n[0] + s[1] * n[1] + s[2] * n[2])
                        })
                        .sum();
                    let tri_area: f64 = f
                        .tris
                        .iter()
                        .map(|&ti| {
                            let t = topo.tris[ti];
                            let (p, q, r) = (topo.verts[t[0]], topo.verts[t[1]], topo.verts[t[2]]);
                            let e1 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
                            let e2 = [r[0] - p[0], r[1] - p[1], r[2] - p[2]];
                            let x = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                            0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
                        })
                        .sum();
                    // ...and the face must be flat by TRUCK's reckoning, not by ours. Its
                    // tolerance is ABSOLUTE — `truck_base::tolerance::TOLERANCE`, 1e-6 in model
                    // units — and a group `build_topo` calls coplanar can be a micron out of
                    // plane, which on a 20 mm wall is a flatness of 5e-8: flat by any standard an
                    // engineer would use, and over truck's bar.
                    //
                    // A face over it is accepted by `Face::try_new` and then refused by the
                    // triangulator, which returns NO geometry for it at all. vacfitting.hcad lost
                    // 18 faces and 94.3 mm² that way — every one of them between 1.0e-6 and
                    // 5.0e-6 out — and the missing area read as a body 3.23% short, so the merged
                    // build was condemned and the part went out as 11,554 separate facets.
                    //
                    // Its triangles are each exactly planar (three points always are), so a face
                    // that fails this merges nothing and costs only its own share of the file.
                    let flat = lps
                        .iter()
                        .flat_map(|(_, lp)| lp.iter())
                        .map(|&v| {
                            let d = [topo.verts[v][0] - o[0], topo.verts[v][1] - o[1], topo.verts[v][2] - o[2]];
                            (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).abs()
                        })
                        .fold(0.0f64, f64::max);
                    if flat < TRUCK_TOLERANCE && (loop_area - tri_area).abs() <= tri_area * 1.0e-6 + 1.0e-9 {
                        let mut wires = Vec::with_capacity(lps.len());
                        let mut ok = true;
                        for (li, lp) in &lps {
                            // A loop that traces a band's rim is written as the circle it always
                            // was. Its direction was read off this very loop, so the arcs run the
                            // way the polyline did and the face keeps the orientation it had.
                            let w = match seam_of.get(&(fi, *li)) {
                                Some(&(bi, rim)) => Some(rim_wire(bi, rim)),
                                None => wire!(lp.as_slice()),
                            };
                            match w {
                                Some(w) => wires.push(w),
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok && !wires.is_empty() {
                            if let Some(pl) = plane_through(o, n) {
                                merged = truck_modeling::Face::try_new(wires, pl.into()).ok();
                            }
                        }
                    }
                }
            }
            match merged {
                Some(face) => faces.push(face),
                None => {
                    for &ti in &f.tris {
                        let t = topo.tris[ti];
                        let Some(w) = wire!(&t[..]) else { continue };
                        // A triangle ALWAYS lies on a plane — it has three points — so the faceted
                        // build was supposed to be exact by construction. `try_attach_plane` does
                        // not find that plane, it derives one and then measures the wire against it
                        // to truck's absolute 1e-6, and for a needle the derivation is the thing
                        // that fails: vacfitting.hcad carries six slivers 0.024 mm long and 1e-5 mm
                        // across, left where a fillet met a chamfer, and truck refused all six.
                        //
                        // Each refusal was dropped on the floor, which is how six triangles out of
                        // 11,554 cost the whole export: six missing faces are six holes, the shell
                        // is then not closed, `Solid::try_new` refuses it, and every rung of the
                        // ladder above falls to this one — so the user got "no exportable body"
                        // for a part that is watertight, with nothing naming the six.
                        //
                        // So hand truck the plane rather than asking it to find one. The triangle
                        // belongs to a coplanar group and lies in that group's plane, which is
                        // known independently of how thin the triangle is.
                        match builder::try_attach_plane(std::slice::from_ref(&w)) {
                            Ok(face) => faces.push(face),
                            Err(_) => match plane_through(topo.verts[t[0]], f.normal) {
                                Some(pl) => faces.push(truck_modeling::Face::new(vec![w], pl.into())),
                                // No plane even for the group: refuse the whole build rather than
                                // return a shell with a hole in it.
                                None => return None,
                            },
                        }
                    }
                }
            }
        }
        // The tubes themselves, in place of the rings of flat strips they were built from.
        for (tube, _, _) in tubes {
            faces.extend(tube);
        }
        if faces.len() < 4 {
            return None;
        }
        let shell: truck_modeling::Shell = faces.into_iter().collect();
        // Refuse a shell that is not a closed, oriented boundary rather than writing out a STEP
        // no one can open.
        truck_modeling::Solid::try_new(vec![shell]).ok().map(KSolid)
    })
}

/// Serialize the exact B-rep solid to a **STEP** (ISO 10303 AP203) string. `None` if the kernel
/// can't express it (or panics).
pub fn export_step(solid: &KSolid) -> Option<String> {
    guard(|| {
        let compressed = solid.0.compress();
        let model = truck_stepio::out::CompleteStepDisplay::new(
            truck_stepio::out::StepModel::from(&compressed),
            truck_stepio::out::StepHeaderDescriptor::default(),
        );
        Some(model.to_string())
    })
}

/// Append a triangle (with a winding-derived flat normal) to a mesh.
pub(crate) fn push_tri(mesh: &mut TriMesh, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
    let sub = |p: [f64; 3], q: [f64; 3]| [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    let (e1, e2) = (sub(b, a), sub(c, a));
    let mut n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 1e-12 {
        n = [n[0] / l, n[1] / l, n[2] / l];
    } else {
        n = [0.0, 0.0, 1.0];
    }
    let base = mesh.positions.len() as u32;
    for p in [a, b, c] {
        mesh.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        mesh.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
    }
    mesh.indices.extend([base, base + 1, base + 2]);
}

/// Triangulate a planar profile face (outer boundary + holes, all in 3D) via the kernel — used for
/// a loft's annular end caps. `reverse` flips the winding (so the two ends face opposite ways).
fn loft_cap_tris(outer: &[[f64; 3]], holes: &[Vec<[f64; 3]>], reverse: bool) -> Vec<[[f64; 3]; 3]> {
    guard(|| {
        let mk = |l: &[[f64; 3]]| -> truck_modeling::Wire {
            let verts: Vec<_> = l.iter().map(|p| builder::vertex(Point3::new(p[0], p[1], p[2]))).collect();
            let mut w = truck_modeling::Wire::new();
            let np = verts.len();
            for i in 0..np {
                w.push_back(builder::line(&verts[i], &verts[(i + 1) % np]));
            }
            w
        };
        let mut wires = vec![mk(outer)];
        for h in holes {
            if h.len() >= 3 {
                wires.push(mk(h));
            }
        }
        // truck refuses a wire more than TOLERANCE (1e-6) thick, measured across its own fitted
        // plane — which is why the sections it is handed have to be computed in f64. In f32 a
        // section is quantised onto a ~6e-7 grid, and that fuzz IS its thickness.
        let face = builder::try_attach_plane(&wires).ok()?;
        let shell: truck_modeling::Shell = std::iter::once(face).collect();
        let mut poly = shell.triangulation(TOL).to_polygon();
        poly.triangulate();
        let pos = poly.positions();
        let mut tris = Vec::new();
        for t in poly.faces().tri_faces() {
            let g = |i: usize| {
                let p = pos[i];
                [p.x, p.y, p.z]
            };
            let (a, b, c) = (g(t[0].pos), g(t[1].pos), g(t[2].pos));
            tris.push(if reverse { [a, c, b] } else { [a, b, c] });
        }
        Some(tris)
    })
    .unwrap_or_default()
}

/// Build a **lofted** solid mesh skinning between an ordered list of cross-section profiles. Each
/// profile is `(outer boundary, hole loops)` in 3D. The outer boundaries are skinned into the side
/// wall, each hole (matched by index across profiles) into an inner tube, and the two ends capped
/// with the annular profile face — a watertight mesh oriented outward. `None` with < 2 profiles.
pub fn loft_mesh(profiles: &[(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)]) -> Option<TriMesh> {
    const N: usize = 96; // resample resolution
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let scale = |a: [f64; 3], s: f64| [a[0] * s, a[1] * s, a[2] * s];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let len = |a: [f64; 3]| dot(a, a).sqrt();
    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| add(a, scale(sub(b, a), t));
    let centroid = |l: &[[f64; 3]]| scale(l.iter().fold([0.0; 3], |acc, &p| add(acc, p)), 1.0 / l.len() as f64);
    let newell = |l: &[[f64; 3]]| {
        let m = l.len();
        let mut n = [0.0; 3];
        for i in 0..m {
            let (a, b) = (l[i], l[(i + 1) % m]);
            n[0] += (a[1] - b[1]) * (a[2] + b[2]);
            n[1] += (a[2] - b[2]) * (a[0] + b[0]);
            n[2] += (a[0] - b[0]) * (a[1] + b[1]);
        }
        n
    };
    let resample = |l: &[[f64; 3]]| -> Vec<[f64; 3]> {
        let m = l.len();
        let seg: Vec<f64> = (0..m).map(|i| len(sub(l[(i + 1) % m], l[i]))).collect();
        let total: f64 = seg.iter().sum();
        if total < 1e-9 {
            return vec![l[0]; N];
        }
        let step = total / N as f64;
        let (mut si, mut acc) = (0usize, 0.0);
        (0..N)
            .map(|k| {
                let target = k as f64 * step;
                while si < m && acc + seg[si] < target {
                    acc += seg[si];
                    si += 1;
                }
                let i = si % m;
                let t = if seg[i] > 1e-9 { (target - acc) / seg[i] } else { 0.0 };
                lerp(l[i], l[(i + 1) % m], t)
            })
            .collect()
    };

    let valid: Vec<&(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)> = profiles.iter().filter(|(o, _)| o.len() >= 3).collect();
    if valid.len() < 2 {
        return None;
    }
    let (c0, cl) = (centroid(&valid[0].0), centroid(&valid[valid.len() - 1].0));
    let axis = {
        let a = sub(cl, c0);
        let la = len(a);
        if la > 1e-9 { scale(a, 1.0 / la) } else { [0.0, 0.0, 1.0] }
    };
    // Resample a set of corresponding loops (the outers, or one hole index across profiles), force a
    // winding sign relative to the axis, and rotationally align each to the previous to limit twist.
    let process = |loops: Vec<&Vec<[f64; 3]>>, want_ccw: bool| -> Vec<Vec<[f64; 3]>> {
        let mut out: Vec<Vec<[f64; 3]>> = loops
            .iter()
            .map(|l| {
                let mut r = resample(l);
                if (dot(newell(&r), axis) > 0.0) != want_ccw {
                    r.reverse();
                }
                r
            })
            .collect();
        for i in 1..out.len() {
            let prev = out[i - 1].clone();
            let mut best = (f64::MAX, 0usize);
            for off in 0..N {
                let d: f64 = (0..N).map(|k| len(sub(out[i][(k + off) % N], prev[k]))).sum();
                if d < best.0 {
                    best = (d, off);
                }
            }
            let off = best.1;
            out[i] = (0..N).map(|k| out[i][(k + off) % N]).collect();
        }
        out
    };

    let prof_outer = process(valid.iter().map(|(o, _)| o).collect(), true);
    // Holes are skinned only when every profile has the same count (matched by index) —
    // pairing 2 holes against 1 by index would skin something arbitrary.
    //
    // Dropping them SILENTLY is the problem: a loft between a 2-hole and a 1-hole profile
    // came out as a solid where a tube was drawn, with nothing to say why. Count it so the
    // caller can say so.
    let hole_count = if valid.iter().all(|(_, h)| h.len() == valid[0].1.len()) {
        valid[0].1.len()
    } else {
        if valid.iter().any(|(_, h)| !h.is_empty()) {
            LOFT_HOLE_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        0
    };
    let mut prof_holes: Vec<Vec<Vec<[f64; 3]>>> = vec![Vec::new(); valid.len()]; // [profile][hole][pt]
    for h in 0..hole_count {
        let processed = process(valid.iter().map(|(_, holes)| &holes[h]).collect(), false); // CW → inner faces the hole
        for (pi, hp) in processed.into_iter().enumerate() {
            prof_holes[pi].push(hp);
        }
    }

    let mut mesh = TriMesh::default();
    let mut skin = |a: &[[f64; 3]], b: &[[f64; 3]]| {
        for k in 0..N {
            let kn = (k + 1) % N;
            push_tri(&mut mesh, a[k], a[kn], b[kn]);
            push_tri(&mut mesh, a[k], b[kn], b[k]);
        }
    };
    for s in 0..prof_outer.len() - 1 {
        skin(&prof_outer[s], &prof_outer[s + 1]);
    }
    for h in 0..hole_count {
        for s in 0..valid.len() - 1 {
            skin(&prof_holes[s][h], &prof_holes[s + 1][h]);
        }
    }
    // End caps (annular profile faces) — only the holes that were skinned, so the tube closes.
    let last = prof_outer.len() - 1;
    for [a, b, c] in loft_cap_tris(&prof_outer[0], &prof_holes[0], true) {
        push_tri(&mut mesh, a, b, c);
    }
    for [a, b, c] in loft_cap_tris(&prof_outer[last], &prof_holes[last], false) {
        push_tri(&mut mesh, a, b, c);
    }
    // Orient outward (flip winding + normals if the signed volume came out negative).
    let mut vol = 0.0;
    for t in mesh.indices.chunks_exact(3) {
        let p = |i: u32| {
            let q = mesh.positions[i as usize];
            [q[0] as f64, q[1] as f64, q[2] as f64]
        };
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        vol += dot(a, [b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]]);
    }
    if vol < 0.0 {
        for t in mesh.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        for n in &mut mesh.normals {
            *n = [-n[0], -n[1], -n[2]];
        }
    }
    Some(mesh)
}

/// Wrap a raw triangle mesh (e.g. a mesh-boolean result) as a renderable
/// [`Tessellation`] by classifying its feature edges, so the mesh fallback renders
/// with the same sharp/tangent edge treatment as the exact kernel.
pub fn mesh_tessellation(mesh: TriMesh) -> Tessellation {
    // Prefer the **face-boundary** detector: it re-ingests the mesh into Manifold, groups coplanar
    // triangles into faces, merges tangent facets into smooth faces, and takes the boundaries between
    // smooth faces as the edges. This is topological, not angle-guessed — boolean re-tessellation
    // inside a face can't produce strays, flat edges are exact, and curve facets vanish. Falls back to
    // the angle detector (with its spur/gap cleanup) only if Manifold can't ingest the mesh.
    let (edges, tangent_edges) = mesh_bool::feature_edges_by_face(&mesh, 25.0, 8.0)
        .unwrap_or_else(|| feature_edges_opts(&mesh, 30.0, 2.0e-4, true));
    // CSG re-tessellation leaves occasional micro-facet boundaries — tiny OPEN scraps of edge
    // floating on an otherwise smooth surface. Drop those; small CLOSED loops (a real tiny
    // hole's rim) are kept.
    let edges = prune_tiny_open_fragments(edges);
    Tessellation { mesh, edges, tangent_edges }
}

/// Remove connected edge fragments that are both SHORT (total length under ~1.5% of the edge
/// set's bounding diagonal) and OPEN (have dangling ends). Real feature edges on a closed solid
/// either form loops or join a larger network; a stubby open scrap is boolean-tessellation noise.
fn prune_tiny_open_fragments(edges: Vec<[[f32; 3]; 2]>) -> Vec<[[f32; 3]; 2]> {
    use std::collections::HashMap;
    if edges.len() < 2 {
        return edges;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for e in &edges {
        for p in e {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let weld = (diag * 1.0e-5).max(1.0e-6);
    let key = |p: [f32; 3]| ((p[0] / weld).round() as i64, (p[1] / weld).round() as i64, (p[2] / weld).round() as i64);
    // Vertex ids, then union-find the segments into connected components.
    let mut ids: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let segs: Vec<(usize, usize, f32)> = edges
        .iter()
        .map(|e| {
            let n = ids.len();
            let a = *ids.entry(key(e[0])).or_insert(n);
            let n = ids.len();
            let b = *ids.entry(key(e[1])).or_insert(n);
            let d = ((e[0][0] - e[1][0]).powi(2) + (e[0][1] - e[1][1]).powi(2) + (e[0][2] - e[1][2]).powi(2)).sqrt();
            (a, b, d)
        })
        .collect();
    let mut uf: Vec<usize> = (0..ids.len()).collect();
    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    for &(a, b, _) in &segs {
        let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
        if ra != rb {
            uf[ra] = rb;
        }
    }
    // Per component: total length + whether any vertex dangles (degree 1 = open).
    let mut total: HashMap<usize, f32> = HashMap::new();
    let mut deg: HashMap<usize, usize> = HashMap::new();
    for &(a, b, d) in &segs {
        *total.entry(find(&mut uf, a)).or_default() += d;
        *deg.entry(a).or_default() += 1;
        *deg.entry(b).or_default() += 1;
    }
    let mut open: HashMap<usize, bool> = HashMap::new();
    for (&v, &dg) in &deg {
        if dg == 1 {
            open.insert(find(&mut uf, v), true);
        }
    }
    let min_len = diag * 0.015;
    // Segment count per component (a real curved rim is dozens of segments; boolean junk is 3-6).
    let mut nsegs: HashMap<usize, usize> = HashMap::new();
    for &(a, _, _) in &segs {
        *nsegs.entry(find(&mut uf, a)).or_default() += 1;
    }
    // Pass 1: drop whole components that are tiny — open scraps, and closed MICRO-LOOPS (3-6
    // segment triangles left where a flush boss meets a wall, under fillet ears, etc.). A real
    // small feature's rim is both longer and far denser in segments.
    let mut keep: Vec<bool> = edges
        .iter()
        .zip(&segs)
        .map(|(_, &(a, _, _))| {
            let root = find(&mut uf, a);
            let is_open = open.get(&root).copied().unwrap_or(false);
            let len = total.get(&root).copied().unwrap_or(0.0);
            let n = nsegs.get(&root).copied().unwrap_or(0);
            let junk_open = is_open && len < min_len;
            let junk_loop = !is_open && n <= 6 && len < diag * 0.025;
            !(junk_open || junk_loop)
        })
        .collect();
    // Pass 2: trim short DANGLING SPUR CHAINS off larger networks — walk inward from each
    // degree-1 endpoint through degree-2 vertices; if the chain ends (junction/loop) within
    // `min_len`, the whole stub is tessellation noise hanging off a real edge. Iterate so
    // nested stubs unwind.
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (si, &(a, b, _)) in segs.iter().enumerate() {
        adj.entry(a).or_default().push(si);
        adj.entry(b).or_default().push(si);
    }
    loop {
        let mut deg_now: HashMap<usize, usize> = HashMap::new();
        for (si, &(a, b, _)) in segs.iter().enumerate() {
            if keep[si] {
                *deg_now.entry(a).or_default() += 1;
                *deg_now.entry(b).or_default() += 1;
            }
        }
        let mut cut_any = false;
        for (&v, &dg) in &deg_now {
            if dg != 1 {
                continue;
            }
            // Walk the chain from this dangling end.
            let (mut cur, mut chain, mut len) = (v, Vec::new(), 0.0_f32);
            loop {
                let Some(&si) = adj.get(&cur).and_then(|es| es.iter().find(|&&si| keep[si] && !chain.contains(&si))) else { break };
                let (a, b, d) = segs[si];
                chain.push(si);
                len += d;
                cur = if a == cur { b } else { a };
                if len >= min_len || deg_now.get(&cur).copied().unwrap_or(0) != 2 {
                    break;
                }
            }
            if len < min_len && !chain.is_empty() {
                for si in chain {
                    keep[si] = false;
                }
                cut_any = true;
            }
        }
        if !cut_any {
            break;
        }
    }
    edges
        .into_iter()
        .zip(keep)
        .filter(|&(_, k)| k)
        .map(|(e, _)| e)
        .collect()
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build the profile's wires (outer CCW, holes CW) from annotated rings: arc
/// spans become **exact circular-arc edges**, everything else line edges.
/// `None` if any ring's annotations are unusable — the caller then falls back
/// to the all-lines path.
fn profile_wires(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    to_p3: &impl Fn(&[f64; 2]) -> Point3,
) -> Option<Vec<truck_modeling::Wire>> {
    let mk = |pts: &[[f64; 2]], arcs: &[ArcSpan], ccw: bool| -> Option<truck_modeling::Wire> {
        let mut segs = ring_to_segs(pts, arcs)?;
        // Winding from the source polyline (robust even when a full circle
        // collapses to two arc segments).
        if (signed_area(pts) > 0.0) != ccw {
            reverse_segs(&mut segs);
        }
        let starts = seg_starts(&segs);
        let verts: Vec<_> = starts.iter().map(|p| builder::vertex(to_p3(p))).collect();
        let m = segs.len();
        let mut w = truck_modeling::Wire::new();
        for (i, s) in segs.iter().enumerate() {
            let (v0, v1) = (&verts[i], &verts[(i + 1) % m]);
            w.push_back(match s {
                PathSeg::Line(..) => builder::line(v0, v1),
                PathSeg::Arc { transit, .. } => builder::circle_arc(v0, v1, to_p3(transit)),
            });
        }
        Some(w)
    };
    let empty: &[ArcSpan] = &[];
    let mut wires = vec![mk(outer, outer_arcs, true)?];
    for (hi, h) in holes.iter().enumerate() {
        if h.len() < 3 {
            continue;
        }
        let arcs = hole_arcs.get(hi).map_or(empty, |v| v.as_slice());
        wires.push(mk(h, arcs, false)?);
    }
    Some(wires)
}

/// [`build_solid`] with exact-arc annotations: try the arc-edge wire path first
/// (true cylindrical side faces), falling back to the sanitized all-lines path
/// if the annotations don't apply or the kernel rejects the exact wires.
fn build_solid_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<truck_modeling::Solid> {
    let any_arcs = !outer_arcs.is_empty() || hole_arcs.iter().any(|h| !h.is_empty());
    if any_arcs && outer.len() >= 3 && length.abs() >= 1e-9 {
        let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
        let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
        let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
        let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
        let base = origin + n * start_offset;
        let solid = guard(|| {
            let to_p3 = |uv: &[f64; 2]| {
                let p = base + u * uv[0] + v * uv[1];
                Point3::new(p.x, p.y, p.z)
            };
            let wires = profile_wires(outer, holes, outer_arcs, hole_arcs, &to_p3)?;
            let face = builder::try_attach_plane(&wires).ok()?;
            Some(builder::tsweep(&face, n * length))
        });
        if solid.is_some() {
            return solid;
        }
    }
    build_solid(outer, holes, basis, start_offset, length)
}

/// [`build_revolve_solid`] with exact-arc annotations — see [`build_solid_arcs`].
fn build_revolve_solid_arcs(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    outer_arcs: &[ArcSpan],
    hole_arcs: &[Vec<ArcSpan>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
    angle: f64,
) -> Option<truck_modeling::Solid> {
    let any_arcs = !outer_arcs.is_empty() || hole_arcs.iter().any(|h| !h.is_empty());
    if any_arcs && outer.len() >= 3 && angle.abs() >= 1e-6 {
        let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
        let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
        let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
        let ao = origin + u * axis_pt[0] + v * axis_pt[1];
        let axis_origin = Point3::new(ao.x, ao.y, ao.z);
        let adir = u * axis_dir[0] + v * axis_dir[1];
        let alen = (adir.x * adir.x + adir.y * adir.y + adir.z * adir.z).sqrt();
        if alen >= 1e-9 {
            let axis = adir / alen;
            let solid = guard(|| {
                let to_p3 = |uv: &[f64; 2]| {
                    let p = origin + u * uv[0] + v * uv[1];
                    Point3::new(p.x, p.y, p.z)
                };
                let wires = profile_wires(outer, holes, outer_arcs, hole_arcs, &to_p3)?;
                let face = builder::try_attach_plane(&wires).ok()?;
                let mut solid = builder::rsweep(&face, axis_origin, axis, truck_modeling::Rad(angle));
                // Same inside-out fix as the all-lines revolve path.
                if solid_signed_volume(&solid) < 0.0 {
                    solid.not();
                }
                Some(solid)
            });
            if solid.is_some() {
                return solid;
            }
        }
    }
    build_revolve_solid(outer, holes, basis, axis_pt, axis_dir, angle)
}

/// Build a prism solid from a region (outer loop + holes): place it at
/// `origin + normal*start_offset`, attach a planar face (outer CCW, holes CW),
/// and translational-sweep it by `normal*length`.
fn build_solid(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<truck_modeling::Solid> {
    // Sanitize the loops first so degenerate contours can't panic the kernel.
    let outer = clean_loop(outer);
    if outer.len() < 3 || length.abs() < 1e-9 {
        return None;
    }
    let holes: Vec<Vec<[f64; 2]>> =
        holes.iter().map(|h| clean_loop(h)).filter(|h| h.len() >= 3).collect();

    let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    let base = origin + n * start_offset;

    // Wire/face/sweep construction can still panic on geometry truck dislikes, so
    // run it under the guard and surface failures as `None`.
    guard(move || {
        let to_p3 = |uv: &[f64; 2]| {
            let p = base + u * uv[0] + v * uv[1];
            Point3::new(p.x, p.y, p.z)
        };
        let make_wire = |loop_pts: &[[f64; 2]]| {
            let verts: Vec<_> = loop_pts.iter().map(|uv| builder::vertex(to_p3(uv))).collect();
            let np = verts.len();
            let mut w = truck_modeling::Wire::new();
            for i in 0..np {
                w.push_back(builder::line(&verts[i], &verts[(i + 1) % np]));
            }
            w
        };

        // Outer boundary CCW, holes CW (truck's convention for a face with holes).
        let mut wires = vec![make_wire(&wound(&outer, true))];
        for h in &holes {
            wires.push(make_wire(&wound(h, false)));
        }
        let face = builder::try_attach_plane(&wires).ok()?;
        Some(builder::tsweep(&face, n * length))
    })
}

/// Build a solid of revolution: attach the region's planar face, then rotational-sweep it
/// around the (3D) axis through `axis_pt` along `axis_dir` (uv) by `angle` radians.
fn build_revolve_solid(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    axis_pt: [f64; 2],
    axis_dir: [f64; 2],
    angle: f64,
) -> Option<truck_modeling::Solid> {
    let outer = clean_loop(outer);
    if outer.len() < 3 || angle.abs() < 1e-6 {
        return None;
    }
    let holes: Vec<Vec<[f64; 2]>> =
        holes.iter().map(|h| clean_loop(h)).filter(|h| h.len() >= 3).collect();

    let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    // Axis line in 3D (a point and a unit direction in the sketch plane).
    let ao = origin + u * axis_pt[0] + v * axis_pt[1];
    let axis_origin = Point3::new(ao.x, ao.y, ao.z);
    let adir = u * axis_dir[0] + v * axis_dir[1];
    let alen = (adir.x * adir.x + adir.y * adir.y + adir.z * adir.z).sqrt();
    if alen < 1e-9 {
        return None;
    }
    let axis = adir / alen;

    guard(move || {
        let to_p3 = |uv: &[f64; 2]| {
            let p = origin + u * uv[0] + v * uv[1];
            Point3::new(p.x, p.y, p.z)
        };
        let make_wire = |loop_pts: &[[f64; 2]]| {
            let verts: Vec<_> = loop_pts.iter().map(|uv| builder::vertex(to_p3(uv))).collect();
            let np = verts.len();
            let mut w = truck_modeling::Wire::new();
            for i in 0..np {
                w.push_back(builder::line(&verts[i], &verts[(i + 1) % np]));
            }
            w
        };
        let mut wires = vec![make_wire(&wound(&outer, true))];
        for h in &holes {
            wires.push(make_wire(&wound(h, false)));
        }
        let face = builder::try_attach_plane(&wires).ok()?;
        // rsweep: full turn (|angle| ≈ 2π) closes the solid; a partial turn caps the ends.
        let solid = builder::rsweep(&face, axis_origin, axis, truck_modeling::Rad(angle));
        // rsweep's orientation depends on which side of the axis the profile sits and the sweep
        // sign, so the result can come out inside-out (inward-facing normals / negative volume).
        // That renders fine alone (double-sided) but, unioned with a real body, the "negative"
        // solid CANCELS it — the boss/cut would vanish. Flip to outward-facing if inverted.
        let mut solid = solid;
        if solid_signed_volume(&solid) < 0.0 {
            solid.not();
        }
        Some(solid)
    })
}

/// Signed volume of a truck solid via its triangulation (positive ⇒ outward-facing normals).
/// Used to detect and fix an inside-out revolve before it poisons a boolean.
fn solid_signed_volume(solid: &truck_modeling::Solid) -> f64 {
    guard(|| {
        let mut poly = solid.triangulation(0.1).to_polygon();
        poly.triangulate();
        let pos = poly.positions();
        let mut vol = 0.0;
        for tri in poly.faces().tri_faces() {
            let (a, b, c) = (pos[tri[0].pos], pos[tri[1].pos], pos[tri[2].pos]);
            vol += a.x * (b.y * c.z - b.z * c.y) - a.y * (b.x * c.z - b.z * c.x) + a.z * (b.x * c.y - b.y * c.x);
        }
        Some(vol / 6.0)
    })
    .unwrap_or(0.0)
}

/// Convert a truck `PolygonMesh` into a flat-shaded [`TriMesh`] (per-triangle
/// normals from the winding, so shading is correct regardless of kernel normals).
fn polymesh_to_trimesh(poly: &truck_polymesh::PolygonMesh) -> TriMesh {
    let pos = poly.positions();
    let mut out = TriMesh::default();
    for tri in poly.faces().tri_faces() {
        let p0 = pos[tri[0].pos];
        let p1 = pos[tri[1].pos];
        let p2 = pos[tri[2].pos];
        let (ux, uy, uz) = (p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        let (vx, vy, vz) = (p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
        let (mut nx, mut ny, mut nz) = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len > 1e-12 {
            nx /= len;
            ny /= len;
            nz /= len;
        } else {
            nz = 1.0;
        }
        let normal = [nx as f32, ny as f32, nz as f32];
        let base = out.positions.len() as u32;
        for p in [p0, p1, p2] {
            out.positions.push([p.x as f32, p.y as f32, p.z as f32]);
            out.normals.push(normal);
        }
        out.indices.extend([base, base + 1, base + 2]);
    }
    out
}

/// Extract the wireframe, classified. Returns `(sharp, tangent)`:
/// - **sharp**: boundary edges, or edges whose faces meet at more than
///   `sharp_deg` — the real corners of the model.
/// - **tangent**: edges whose faces meet at a gentle angle (above a tiny flat
///   threshold) — the curvature/facet lines of smooth surfaces and tangent blends.
/// Exactly-coplanar interior edges are dropped from both.
fn feature_edges(mesh: &TriMesh, sharp_deg: f64) -> (Vec<[[f32; 3]; 2]>, Vec<[[f32; 3]; 2]>) {
    feature_edges_opts(mesh, sharp_deg, 1.0e-6, false)
}

/// `feature_edges`, parameterised for the two mesh sources:
/// - `rel` is the vertex-merge tolerance as a fraction of the mesh's bounding-box diagonal, so it
///   scales with the part (a fixed grid over- or under-merges as the model size changes). Two
///   vertices closer than `rel · diag` fuse — recovering the shared-edge adjacency that flat-shading
///   and CSG float error split apart. Mesh-boolean output needs a looser tolerance than clean truck
///   meshes.
/// - `manifold_only` keeps **only** edges shared by exactly two faces and drops boundary/non-manifold
///   edges. Mesh-boolean (CSG) output can leave stray boundary slivers that would draw as a starburst,
///   so the mesh-fallback path turns this on; the exact (truck) path leaves it off.
fn feature_edges_opts(
    mesh: &TriMesh,
    sharp_deg: f64,
    rel: f32,
    manifold_only: bool,
) -> (Vec<[[f32; 3]; 2]>, Vec<[[f32; 3]; 2]>) {
    use std::collections::HashMap;
    if mesh.positions.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Bbox-relative merge grid: cell size scales with the part so coincident vertices fuse reliably
    // at any model scale (a fixed grid is too coarse for tiny parts, too fine for big ones).
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let cell = (diag * rel).max(1.0e-6);
    let scale = 1.0 / cell;
    // Merge duplicated (flat-shaded) / near-coincident vertices by quantized position.
    let quant = |p: [f32; 3]| {
        ((p[0] * scale).round() as i64, (p[1] * scale).round() as i64, (p[2] * scale).round() as i64)
    };
    let mut canon: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut canon_pos: Vec<[f32; 3]> = Vec::new();
    let mut vid = vec![0usize; mesh.positions.len()];
    for (i, p) in mesh.positions.iter().enumerate() {
        let id = *canon.entry(quant(*p)).or_insert_with(|| {
            canon_pos.push(*p);
            canon_pos.len() - 1
        });
        vid[i] = id;
    }

    // Gather the face normals incident to each undirected edge.
    let mut emap: HashMap<(usize, usize), Vec<[f32; 3]>> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        let (ia, ib, ic) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let normal = mesh.normals[ia]; // flat normal, same for all 3 verts
        let (a, b, c) = (vid[ia], vid[ib], vid[ic]);
        for (i, j) in [(a, b), (b, c), (c, a)] {
            let key = if i < j { (i, j) } else { (j, i) };
            emap.entry(key).or_default().push(normal);
        }
    }

    let cos_sharp = sharp_deg.to_radians().cos();
    let cos_flat = 1.0_f64.to_radians().cos(); // below this angle ⇒ coplanar, drop
    let mut sharp_ids: Vec<(usize, usize)> = Vec::new();
    let mut tangent = Vec::new();
    for ((i, j), normals) in emap {
        // The widest angle between any incident pair of faces = the smallest dot.
        let mut min_dot = 1.0_f32;
        for a in 0..normals.len() {
            for b in (a + 1)..normals.len() {
                let d = normals[a][0] * normals[b][0]
                    + normals[a][1] * normals[b][1]
                    + normals[a][2] * normals[b][2];
                min_dot = min_dot.min(d);
            }
        }
        if normals.len() != 2 {
            // Boundary (1 face): a lone normal can't give a dihedral — keep on the exact (truck) path
            // where it's a real open edge, drop on the CSG path (seam slivers). Non-manifold (≥3
            // faces): a CSG cut can make a real edge where 3 faces meet — KEEP it if some incident
            // pair forms a real corner, so the cut's edges aren't lost (they were dropped before,
            // which then let the spur-prune eat the whole chain).
            match normals.len() {
                1 if !manifold_only => sharp_ids.push((i, j)),
                n if n >= 3 && (min_dot as f64) < cos_sharp => sharp_ids.push((i, j)),
                _ => {}
            }
            continue;
        }
        let md = min_dot as f64;
        if md < cos_sharp {
            sharp_ids.push((i, j)); // a real corner
        } else if md < cos_flat {
            tangent.push([canon_pos[i], canon_pos[j]]); // smooth/curvature edge
        } // else coplanar interior → drop
    }
    // Clean boolean-seam artifacts: prune short stray/spur paths (the pop-out segments) and bridge
    // tiny gaps where a loop lost a segment at a seam.
    let sharp = clean_feature_edges(&sharp_ids, &canon_pos, diag);
    (sharp, tangent)
}

/// Tidy the raw sharp-edge set extracted from a (mesh-boolean) mesh, using the invariant that on a
/// *closed solid* real feature edges never dead-end — they close into loops or meet at corners. So a
/// dangling (degree-1) endpoint is always a boolean-seam artifact.
/// 1. **Prune spurs** — any degree-2 chain running from a dangling end to a junction is a stray that
///    pokes off the real edge network; remove it whatever its length (this kills the long "sticking
///    out" segments the short-only prune missed). Iterated, so nested spurs unwind.
/// 2. **Resolve isolated open paths** — a connected piece with exactly two dangling ends and no
///    junction is either a short stray (drop it) or a loop that lost a segment at a seam: if its two
///    ends nearly meet, close it so the circle reads continuous; if they're far apart it's not a loop
///    at all, so drop it.
fn clean_feature_edges(ids: &[(usize, usize)], pos: &[[f32; 3]], diag: f32) -> Vec<[[f32; 3]; 2]> {
    use std::collections::{HashMap, HashSet};
    let elen = |a: usize, b: usize| -> f32 {
        let (p, q) = (pos[a], pos[b]);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    let mut edges: Vec<(usize, usize)> = ids.to_vec();
    let adjacency = |edges: &[(usize, usize)]| {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (ei, &(a, b)) in edges.iter().enumerate() {
            adj.entry(a).or_default().push(ei);
            adj.entry(b).or_default().push(ei);
        }
        adj
    };

    // --- Step 1: prune spurs (dangle → … → junction), any length. Iterate so unwinding a spur that
    // exposes a new dangling end keeps pruning.
    loop {
        let adj = adjacency(&edges);
        let deg = |v: usize| adj.get(&v).map_or(0, |e| e.len());
        let mut remove: HashSet<usize> = HashSet::new();
        for (&v, es) in &adj {
            if es.len() != 1 {
                continue; // start only from dangling ends
            }
            let mut chain = Vec::new();
            let mut len = 0.0f32;
            let mut cur = v;
            let mut e = es[0];
            let reached_junction = loop {
                let (a, b) = edges[e];
                let other = if a == cur { b } else { a };
                chain.push(e);
                len += elen(a, b);
                match deg(other) {
                    2 => match adj[&other].iter().copied().find(|&x| x != e) {
                        Some(n) => {
                            cur = other;
                            e = n;
                        }
                        None => break false,
                    },
                    d if d >= 3 => break true, // attached to the real network → spur
                    _ => break false,          // another dangle → isolated path (step 2)
                }
                if chain.len() > edges.len() {
                    break false; // safety
                }
            };
            // Only prune SHORT spurs. A *long* chain that dead-ends is a real edge that lost a
            // neighbour at a non-manifold/boolean seam — deleting it would erase a real cut edge.
            if reached_junction && len < diag * 0.08 {
                for c in chain {
                    remove.insert(c);
                }
            }
        }
        if remove.is_empty() {
            break;
        }
        edges = edges.iter().enumerate().filter(|(i, _)| !remove.contains(i)).map(|(_, e)| *e).collect();
    }

    // --- Step 2: classify connected components; resolve isolated open paths.
    let adj = adjacency(&edges);
    let deg = |v: usize| adj.get(&v).map_or(0, |e| e.len());
    let mut comp_of = vec![usize::MAX; edges.len()];
    let mut ncomp = 0;
    for start in 0..edges.len() {
        if comp_of[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        comp_of[start] = ncomp;
        while let Some(ei) = stack.pop() {
            let (a, b) = edges[ei];
            for v in [a, b] {
                for &ne in &adj[&v] {
                    if comp_of[ne] == usize::MAX {
                        comp_of[ne] = ncomp;
                        stack.push(ne);
                    }
                }
            }
        }
        ncomp += 1;
    }
    let stray_max = diag * 0.05; // a short isolated piece is a seam stray
    let gap_max = diag * 0.2; // a loop that lost a segment has its two ends near each other
    let mut drop_comp: HashSet<usize> = HashSet::new();
    let mut close: Vec<(usize, usize)> = Vec::new();
    for c in 0..ncomp {
        let cedges: Vec<usize> = (0..edges.len()).filter(|&i| comp_of[i] == c).collect();
        let mut verts: HashSet<usize> = HashSet::new();
        for &ei in &cedges {
            verts.insert(edges[ei].0);
            verts.insert(edges[ei].1);
        }
        let dangles: Vec<usize> = verts.iter().copied().filter(|&v| deg(v) == 1).collect();
        let has_junction = verts.iter().any(|&v| deg(v) >= 3);
        if dangles.len() == 2 && !has_junction {
            let length: f32 = cedges.iter().map(|&ei| { let (a, b) = edges[ei]; elen(a, b) }).sum();
            let gap = elen(dangles[0], dangles[1]);
            if length < stray_max || gap > gap_max {
                drop_comp.insert(c); // short stray, or a long path that isn't a loop
            } else {
                close.push((dangles[0], dangles[1])); // a loop that lost a segment → close it
            }
        }
    }
    let mut out: Vec<(usize, usize)> =
        edges.iter().enumerate().filter(|(i, _)| !drop_comp.contains(&comp_of[*i])).map(|(_, e)| *e).collect();
    out.extend(close);
    out.iter().map(|&(a, b)| [pos[a], pos[b]]).collect()
}

/// A copy of `m` translated by `d` (normals unchanged) — pattern instances.
pub fn translate_mesh(m: &TriMesh, d: [f64; 3]) -> TriMesh {
    let mut out = m.clone();
    for p in &mut out.positions {
        p[0] += d[0] as f32;
        p[1] += d[1] as f32;
        p[2] += d[2] as f32;
    }
    out
}

/// A copy of `m` rotated by `angle` radians about the axis through `pt` along `axis`
/// (Rodrigues; normals rotate too) — circular-pattern instances.
pub fn rotate_mesh(m: &TriMesh, pt: [f64; 3], axis: [f64; 3], angle: f64) -> TriMesh {
    let al = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if al < 1e-12 {
        return m.clone();
    }
    let k = [axis[0] / al, axis[1] / al, axis[2] / al];
    let (s, c) = angle.sin_cos();
    let rot = |v: [f64; 3]| -> [f64; 3] {
        let kv = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
        let kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        [
            v[0] * c + kv[0] * s + k[0] * kd * (1.0 - c),
            v[1] * c + kv[1] * s + k[1] * kd * (1.0 - c),
            v[2] * c + kv[2] * s + k[2] * kd * (1.0 - c),
        ]
    };
    let mut out = m.clone();
    for p in &mut out.positions {
        let v = [p[0] as f64 - pt[0], p[1] as f64 - pt[1], p[2] as f64 - pt[2]];
        let r = rot(v);
        *p = [(r[0] + pt[0]) as f32, (r[1] + pt[1]) as f32, (r[2] + pt[2]) as f32];
    }
    for n in &mut out.normals {
        let r = rot([n[0] as f64, n[1] as f64, n[2] as f64]);
        *n = [r[0] as f32, r[1] as f32, r[2] as f32];
    }
    out
}

/// Build the SHELL cutting tool: the body's inner surface, offset inward by `thickness`.
/// Subtracting it from the body leaves walls of that thickness. Faces listed in `open`
/// (a point on the face + its outward unit normal) are REMOVED: vertices on them push
/// *outward* by `overshoot` instead, so the tool pokes through and the cavity opens there.
///
/// Vertices are welded by position, then each is moved to satisfy "every adjacent face's
/// plane shifts in by `thickness`" (least-squares over the distinct adjacent face normals) —
/// so box corners land exactly and walls stay uniform. Offsets are capped at 3× thickness
/// to keep shallow creases from exploding. `None` if the mesh is empty.
pub fn shell_tool(body: &TriMesh, thickness: f64, open: &[([f64; 3], [f64; 3])], overshoot: f64) -> Option<TriMesh> {
    if body.positions.is_empty() || thickness <= 0.0 {
        return None;
    }
    // Bounding diagonal for tolerances.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &body.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a] as f64);
            hi[a] = hi[a].max(p[a] as f64);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1e-6);
    let plane_tol = diag * 1e-4 + 1e-6;

    // Weld vertices by quantised position (tessellations may duplicate per-face).
    let q = diag * 1e-6;
    let key = |p: [f32; 3]| ((p[0] as f64 / q).round() as i64, (p[1] as f64 / q).round() as i64, (p[2] as f64 / q).round() as i64);
    let mut weld: std::collections::HashMap<(i64, i64, i64), usize> = std::collections::HashMap::new();
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let map: Vec<usize> = body
        .positions
        .iter()
        .map(|&p| {
            *weld.entry(key(p)).or_insert_with(|| {
                verts.push([p[0] as f64, p[1] as f64, p[2] as f64]);
                verts.len() - 1
            })
        })
        .collect();

    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| {
        let l = dot(a, a).sqrt();
        if l > 1e-12 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0; 3] }
    };

    // Classify triangles: on a removed face, or a keeper (offsets inward).
    let tris: Vec<[usize; 3]> = body.indices.chunks(3).map(|t| [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]]).collect();
    let mut tri_removed: Vec<Option<usize>> = vec![None; tris.len()]; // index into `open`
    let mut tri_n: Vec<[f64; 3]> = Vec::with_capacity(tris.len());
    for (ti, t) in tris.iter().enumerate() {
        let n = norm(cross(sub(verts[t[1]], verts[t[0]]), sub(verts[t[2]], verts[t[0]])));
        tri_n.push(n);
        for (oi, (op, on)) in open.iter().enumerate() {
            let on = norm(*on);
            if dot(n, on) > 0.985 && t.iter().all(|&vi| (dot(sub(verts[vi], *op), on)).abs() < plane_tol) {
                tri_removed[ti] = Some(oi);
                break;
            }
        }
    }

    // Per vertex: the distinct adjacent KEEPER face normals (dedup near-parallel), and the
    // distinct removed faces it touches.
    let mut keep_normals: Vec<Vec<[f64; 3]>> = vec![Vec::new(); verts.len()];
    let mut open_normals: Vec<Vec<[f64; 3]>> = vec![Vec::new(); verts.len()];
    for (ti, t) in tris.iter().enumerate() {
        let n = tri_n[ti];
        if dot(n, n) < 0.5 {
            continue; // degenerate sliver
        }
        for &vi in t {
            let bucket = if tri_removed[ti].is_some() { &mut open_normals[vi] } else { &mut keep_normals[vi] };
            if !bucket.iter().any(|m| dot(*m, n) > 0.999) {
                bucket.push(n);
            }
        }
    }

    // Solve each vertex's inward offset x: n_i · x = thickness for every keeper normal,
    // least-squares via regularised 3×3 normal equations. Then add the outward overshoot
    // for removed faces.
    let mut out_verts: Vec<[f64; 3]> = Vec::with_capacity(verts.len());
    let cap = thickness * 3.0;
    for vi in 0..verts.len() {
        let mut m = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for n in &keep_normals[vi] {
            for r in 0..3 {
                for c in 0..3 {
                    m[r][c] += n[r] * n[c];
                }
                b[r] += n[r] * thickness;
            }
        }
        let lam = 1e-7 * (m[0][0] + m[1][1] + m[2][2]).max(1e-9);
        for r in 0..3 {
            m[r][r] += lam;
        }
        // Cramer's rule.
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let mut x = [0.0f64; 3];
        if det.abs() > 1e-18 {
            for col in 0..3 {
                let mut mm = m;
                for r in 0..3 {
                    mm[r][col] = b[r];
                }
                let d = mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1]) - mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0])
                    + mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]);
                x[col] = d / det;
            }
        }
        let xl = dot(x, x).sqrt();
        if xl > cap {
            x = [x[0] * cap / xl, x[1] * cap / xl, x[2] * cap / xl];
        }
        let mut p = sub(verts[vi], x); // move INWARD (x satisfies n·x = +t on outward normals)
        for on in &open_normals[vi] {
            p = [p[0] + on[0] * overshoot, p[1] + on[1] * overshoot, p[2] + on[2] * overshoot];
        }
        out_verts.push(p);
    }

    // Rebuild with the moved vertices (same topology; flat per-face normals).
    let mut mesh = TriMesh::default();
    for t in &tris {
        push_tri(&mut mesh, out_verts[t[0]], out_verts[t[1]], out_verts[t[2]]);
    }
    (!mesh.indices.is_empty()).then_some(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagnostic: where is the tolerance cliff below which truck's boolean runs away on an
    /// exact-arc (NURBS) tool? This is how `hworks-app`'s `ARC_BOOL_MIN_TOL` was chosen —
    /// re-run it after a truck upgrade to re-derive the floor. Run with:
    ///   cargo test -p hworks-geometry diag_nurbs_boolean_runaway -- --ignored --nocapture
    ///
    /// Each case runs on its own thread: a runaway boolean cannot be cancelled, so the
    /// thread is abandoned (it spins until the process exits) rather than joined.
    #[test]
    #[ignore]
    fn diag_nurbs_boolean_runaway() {
        let basis = PlaneBasis { origin: [0.0, 0.0, 15.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        // A 30x30 slab, all lines, as the base body.
        let slab: Vec<[f64; 2]> = vec![[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]];
        let base_basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        // A circle tessellated to `n` points, annotated as one full-circle arc span.
        let circle = |cx: f64, cy: f64, r: f64, n: usize| -> (Vec<[f64; 2]>, Vec<ArcSpan>) {
            let pts = (0..n)
                .map(|k| {
                    let a = std::f64::consts::TAU * k as f64 / n as f64;
                    [cx + r * a.cos(), cy + r * a.sin()]
                })
                .collect();
            (pts, vec![ArcSpan { first_edge: 0, count: n, center: [cx, cy], radius: r }])
        };
        let run = |label: String, f: Box<dyn FnOnce() -> String + Send>| {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || { let _ = tx.send(f()); });
            match rx.recv_timeout(std::time::Duration::from_secs(45)) {
                Ok(m) => eprintln!("  OK      {label}: {m}"),
                Err(_) => eprintln!("  RUNAWAY {label} (>45s)"),
            }
        };
        // The sweep that located the cliff: everything at or below 0.005 ran away; 0.01 came
        // back in 21 s, 0.02 in 14 s, 0.05 in 10 s (debug build).
        for tol in [1.0e-4_f64, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 0.02, 0.05] {
            let base = extrude_solid_arcs(&slab, &[], &[], &[], &base_basis, 15.0).expect("slab");
            let (pts, arcs) = circle(15.0, 15.0, 11.25, 64);
            let b = basis.clone();
            run(format!("cut NURBS disc tol={tol}"), Box::new(move || {
                let t0 = std::time::Instant::now();
                let r = cut_tol_arcs(&base, &pts, &[], &arcs, &[], &b, -2.5, 0.0, tol).is_some();
                format!("some={r} in {:?}", t0.elapsed())
            }));
        }
        // Same disc as a plain faceted tool (no arc annotations) — the control.
        for tol in [1.0e-4_f64, 0.05] {
            let base = extrude_solid_arcs(&slab, &[], &[], &[], &base_basis, 15.0).expect("slab");
            let (pts, _) = circle(15.0, 15.0, 11.25, 128);
            let b = basis.clone();
            run(format!("cut FACETED disc n=128 tol={tol}"), Box::new(move || {
                format!("some={}", cut_tol(&base, &pts, &[], &b, -2.5, 0.0, tol).is_some())
            }));
        }
        // A NURBS boss union onto the slab.
        for tol in [1.0e-4_f64, 0.02] {
            let base = extrude_solid_arcs(&slab, &[], &[], &[], &base_basis, 15.0).expect("slab");
            let (pts, arcs) = circle(15.0, 15.0, 5.0, 64);
            let b = PlaneBasis { origin: [0.0, 0.0, 15.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
            run(format!("union NURBS boss tol={tol}"), Box::new(move || {
                let boss = extrude_solid_arcs(&pts, &[], &arcs, &[], &b, 5.0).expect("boss");
                format!("some={}", union_tol(&base, &boss, tol).is_some())
            }));
        }
    }

    /// Diagnostic: load a real scan and report why it won't cut. Run with:
    ///   cargo test -p hworks-geometry diag_scan_topology -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diag_scan_topology() {
        let path = std::env::var("HCAD_STL").unwrap_or_else(|_| "../../saved files/0707_01_mesh.stl".to_string());
        let bytes = std::fs::read(&path).expect("read STL");
        let m = import_stl(&bytes).expect("parse STL");
        let (v, t, bnd, nonman) = crate::mesh_bool::weld_edge_stats(&m);
        eprintln!("RAW: {} tris | welded {v} verts, {t} tris | boundary edges {bnd} | non-manifold edges {nonman}", m.indices.len() / 3);
        eprintln!("RAW is_manifold (Manifold ingest): {}", is_manifold(&m));
        let t0 = std::time::Instant::now();
        let (r, rep) = repair_mesh(&m);
        eprintln!("repair took {:?}: {rep:?}", t0.elapsed());
        let (v2, t2, bnd2, nonman2) = crate::mesh_bool::weld_edge_stats(&r);
        eprintln!("REPAIRED: welded {v2} verts, {t2} tris | boundary edges {bnd2} | non-manifold edges {nonman2}");
        eprintln!("REPAIRED is_manifold: {}", is_manifold(&r));
        let bbox = |mesh: &TriMesh| {
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for p in &mesh.positions {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        };
        let vol = |mesh: &TriMesh| {
            let mut v = 0.0f64;
            for t in mesh.indices.chunks_exact(3) {
                let p = |i: u32| {
                    let q = mesh.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                v += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1])) / 6.0;
            }
            v.abs()
        };
        let (blo, bhi) = bbox(&m);
        let bbox_vol = (bhi[0] - blo[0]) as f64 * (bhi[1] - blo[1]) as f64 * (bhi[2] - blo[2]) as f64;
        eprintln!("INPUT bbox {blo:?}..{bhi:?} | bbox_vol {bbox_vol:.0} | input surface vol (bogus if open) {:.0}", vol(&m));
        for res in [96usize, 128] {
            let t1 = std::time::Instant::now();
            match remesh_solid(&m, res) {
                Some(s) => {
                    let sv = vol(&s);
                    let (wv, wt, wb, wn) = crate::mesh_bool::weld_edge_stats(&s);
                    // boxiness = output volume / bbox: should match the actual shape's fill
                    // fraction, NOT ~1.0 (which would mean it filled the whole box).
                    eprintln!(
                        "REMESH res={res}: {} tris in {:?} | manifold {} | vol {:.0} | boxiness {:.2} | welded {wv}v {wt}t bnd={wb} nonman={wn}",
                        s.indices.len() / 3,
                        t1.elapsed(),
                        is_manifold(&s),
                        sv,
                        sv / bbox_vol
                    );
                }
                None => eprintln!("REMESH res={res}: returned None (falls back to raw)"),
            }
        }
    }

    fn xy_plane() -> PlaneBasis {
        PlaneBasis {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        }
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }
    fn circle(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
        (0..n).map(|k| { let a = std::f64::consts::TAU * k as f64 / n as f64; [cx + r * a.cos(), cy + r * a.sin()] }).collect()
    }

    #[test]
    fn revolve_boss_keeps_the_existing_body() {
        // Reproduces the app's "revolve a profile while a body already exists" case: a cylinder
        // (extruded along Z) plus a torus (a circle revolved around the perpendicular Y axis).
        // The union MUST contain both — the bug was the cylinder vanishing, leaving only the ring.
        let cyl = extrude_tool_mesh(&circle(0.0, 0.0, 5.0, 48), &[], &plane_at(-10.0), 0.0, 20.0).expect("cylinder");
        let torus = revolve_tool_mesh(&circle(20.0, 0.0, 2.0, 32), &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::TAU).expect("torus");
        let (cv, tv) = (mesh_vol(&cyl), mesh_vol(&torus));
        let u = mesh_union(&cyl, &torus);
        let uv = mesh_vol(&u);
        assert!(uv > cv + tv * 0.5, "mesh union dropped a body: union {uv:.1}, cyl {cv:.1}, torus {tv:.1}");
        // Exact-kernel union too.
        let cyl_s = extrude_solid(&circle(0.0, 0.0, 5.0, 48), &[], &plane_at(-10.0), 20.0).expect("cyl solid");
        let tor_s = revolve_solid(&circle(20.0, 0.0, 2.0, 32), &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::TAU).expect("torus solid");
        let us = union(&cyl_s, &tor_s).expect("exact union builds");
        let usv = mesh_vol(&tessellate(&us, 0.1).mesh);
        assert!(usv > cv + tv * 0.5, "exact union dropped a body: union {usv:.1}, cyl {cv:.1}, torus {tv:.1}");
    }

    fn circle3(cx: f64, cy: f64, cz: f64, r: f64, n: usize) -> Vec<[f64; 3]> {
        (0..n).map(|k| { let a = std::f64::consts::TAU * k as f64 / n as f64; [cx + r * a.cos(), cy + r * a.sin(), cz] }).collect()
    }

    /// Full-circle [`ArcSpan`] over an `n`-gon polyline (what the sketch layer
    /// produces for a plain circle region).
    fn full_span(cx: f64, cy: f64, r: f64, n: usize) -> Vec<ArcSpan> {
        vec![ArcSpan { first_edge: 0, count: n, center: [cx, cy], radius: r }]
    }

    #[test]
    fn exact_arc_extrude_is_a_true_cylinder() {
        // The same 64-gon profile, extruded with and without arc annotations. The
        // annotated one must produce a compact exact B-rep (two cylindrical side
        // faces + caps), not one wall face per polyline facet.
        let poly = circle(0.0, 0.0, 5.0, 64);
        let faceted = extrude_solid(&poly, &[], &xy_plane(), 10.0).expect("prism");
        let exact = extrude_solid_arcs(&poly, &[], &full_span(0.0, 0.0, 5.0, 64), &[], &xy_plane(), 10.0)
            .expect("exact cylinder");
        let count = |s: &KSolid| export_step(s).map_or(usize::MAX, |st| st.matches("FACE_SURFACE").count());
        let (fa, ex) = (count(&faceted), count(&exact));
        assert!(ex <= 6, "exact cylinder should have a handful of faces, got {ex}");
        assert!(fa >= 60, "sanity: faceted prism should have ~66 faces, got {fa}");
        // And the volume is still a cylinder's.
        let vol = mesh_vol(&tessellate(&exact, 0.02).mesh);
        let want = std::f64::consts::PI * 25.0 * 10.0;
        assert!((vol - want).abs() / want < 0.01, "cylinder volume {vol:.2}, want {want:.2}");
    }

    #[test]
    fn exact_arc_hole_gives_an_exact_bore() {
        // A plate with a circular hole: the hole's arc annotation must survive as
        // exact cylindrical bore faces.
        let hole = circle(0.0, 0.0, 2.0, 48);
        let solid = extrude_solid_arcs(
            &rect(-10.0, -10.0, 10.0, 10.0),
            &[hole],
            &[],
            &[full_span(0.0, 0.0, 2.0, 48)],
            &xy_plane(),
            5.0,
        )
        .expect("plate with bore");
        let step = export_step(&solid).expect("step");
        let faces = step.matches("FACE_SURFACE").count();
        assert!(faces <= 12, "plate+bore should be ~8 faces, got {faces}");
        let vol = mesh_vol(&tessellate(&solid, 0.02).mesh);
        let want = 20.0 * 20.0 * 5.0 - std::f64::consts::PI * 4.0 * 5.0;
        assert!((vol - want).abs() / want < 0.01, "bore volume {vol:.2}, want {want:.2}");
    }

    #[test]
    fn partial_arc_span_builds_a_half_round() {
        // A semicircular profile: 33 rim samples (32 arc edges) closed by one
        // chord edge. The span covers only the rim edges.
        let n = 33;
        let mut poly: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / (n - 1) as f64;
                [3.0 * a.cos(), 3.0 * a.sin()]
            })
            .collect();
        poly.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        let spans = vec![ArcSpan { first_edge: 0, count: n - 1, center: [0.0, 0.0], radius: 3.0 }];
        let solid = extrude_solid_arcs(&poly, &[], &spans, &[], &xy_plane(), 4.0).expect("half round");
        let step = export_step(&solid).expect("step");
        let faces = step.matches("FACE_SURFACE").count();
        assert!(faces <= 6, "half-round should be ~4 faces, got {faces}");
        let vol = mesh_vol(&tessellate(&solid, 0.02).mesh);
        let want = 0.5 * std::f64::consts::PI * 9.0 * 4.0;
        assert!((vol - want).abs() / want < 0.01, "half-round volume {vol:.2}, want {want:.2}");
    }

    #[test]
    fn exact_arc_cut_bores_a_hole() {
        // Cut a round hole through a block with the arc-annotated tool.
        let block = extrude_solid(&rect(-10.0, -10.0, 10.0, 10.0), &[], &xy_plane(), 5.0).unwrap();
        let hole = circle(0.0, 0.0, 2.0, 48);
        let cut = cut_tol_arcs(&block, &hole, &[], &full_span(0.0, 0.0, 2.0, 48), &[], &xy_plane(), 5.0, 0.0, TOL)
            .expect("cut with exact tool");
        let vol = mesh_vol(&tessellate(&cut, 0.02).mesh);
        let want = 20.0 * 20.0 * 5.0 - std::f64::consts::PI * 4.0 * 5.0;
        assert!((vol - want).abs() / want < 0.01, "cut volume {vol:.2}, want {want:.2}");
    }

    #[test]
    fn exact_arc_revolve_makes_a_torus() {
        // Revolve an arc-annotated circle profile around the Y axis → a torus with
        // exact cross-section: V = 2π²·R·r².
        let prof = circle(20.0, 0.0, 2.0, 48);
        let torus = revolve_solid_arcs(
            &prof,
            &[],
            &full_span(20.0, 0.0, 2.0, 48),
            &[],
            &xy_plane(),
            [0.0, 0.0],
            [0.0, 1.0],
            std::f64::consts::TAU,
        )
        .expect("exact torus");
        let vol = mesh_vol(&tessellate(&torus, 0.02).mesh);
        let want = 2.0 * std::f64::consts::PI.powi(2) * 20.0 * 4.0;
        assert!((vol - want).abs() / want < 0.02, "torus volume {vol:.2}, want {want:.2}");
    }

    /// Sectioning a cube through its middle must yield exactly its square outline: every
    /// segment on the plane, and the total length equal to the perimeter (4 × 20). Guards the
    /// scan-tracing feature (sketch on a plane through a reference mesh, snap to the outline).
    #[test]
    fn mesh_plane_section_of_a_cube_is_its_square_outline() {
        let sq = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let cube = extrude_tool_mesh(&sq, &[], &basis, 0.0, 20.0).expect("cube");
        let segs = mesh_plane_section(&cube, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
        assert!(!segs.is_empty(), "mid-cube section must not be empty");
        let mut total = 0.0f64;
        for s in &segs {
            for p in s {
                assert!((p[2] - 10.0).abs() < 1e-4, "section point off the plane: {p:?}");
                assert!(p[0].abs() < 10.0 + 1e-3 && p[1].abs() < 10.0 + 1e-3, "outside the cube: {p:?}");
            }
            total += (((s[0][0] - s[1][0]).powi(2) + (s[0][1] - s[1][1]).powi(2)) as f64).sqrt();
        }
        assert!((total - 80.0).abs() < 0.5, "section length {total:.2} should be the 80 mm perimeter");
        // Off the body entirely → nothing.
        assert!(mesh_plane_section(&cube, [0.0, 0.0, 50.0], [0.0, 0.0, 1.0]).is_empty());
    }

    /// A dense NON-manifold mesh (an open subdivided sheet, >20k tris) must NOT drag the cut
    /// into the O(n²) BSP fallback — the boolean skips it, returns the base unchanged, and
    /// bumps the dense-skip counter so the app can warn instead of hanging.
    #[test]
    fn dense_nonmanifold_cut_skips_bsp_instead_of_hanging() {
        // A flat sheet subdivided into a grid — all boundary edges, so Manifold declines it.
        let g = 120usize; // 2*g*g = 28 800 triangles > BSP_MAX_TRIS
        let mut sheet = TriMesh::default();
        let step = 1.0f32;
        let vid = |x: usize, y: usize| (y * (g + 1) + x) as u32;
        for y in 0..=g {
            for x in 0..=g {
                sheet.positions.push([x as f32 * step, y as f32 * step, 0.0]);
                sheet.normals.push([0.0, 0.0, 1.0]);
            }
        }
        for y in 0..g {
            for x in 0..g {
                sheet.indices.extend([vid(x, y), vid(x + 1, y), vid(x + 1, y + 1)]);
                sheet.indices.extend([vid(x, y), vid(x + 1, y + 1), vid(x, y + 1)]);
            }
        }
        assert!(sheet.indices.len() / 3 > 20_000, "sheet must exceed the BSP guard");
        assert!(!is_manifold(&sheet), "an open sheet isn't manifold");
        let _ = take_dense_skip_count(); // clear
        // A tool box overlapping the sheet.
        let sq = [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]];
        let basis = PlaneBasis { origin: [0.0, 0.0, -5.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let tool = extrude_tool_mesh(&sq, &[], &basis, 0.0, 10.0).expect("tool");
        let t0 = std::time::Instant::now();
        let out = mesh_difference(&sheet, &tool);
        let elapsed = t0.elapsed();
        assert!(elapsed.as_secs() < 2, "must fail fast, not grind BSP (took {elapsed:?})");
        assert_eq!(out.indices.len(), sheet.indices.len(), "base returned unchanged");
        assert_eq!(take_dense_skip_count(), 1, "the skip must be recorded for the app to warn");
    }

    /// `fit_region` must recognize what was clicked: a cylinder's wall → Cylinder with the
    /// right axis/radius; its flat cap → Plane with the right normal; a ball → Sphere with
    /// the right centre/radius. This is the click-a-scan-face-get-a-datum feature's core.
    #[test]
    fn fit_region_recognizes_cylinder_cap_and_sphere() {
        // Cylinder r=8, z∈[0,30], 64-gon — via the extrude tool (side + caps).
        let n = 64;
        let circle: Vec<[f64; 2]> = (0..n)
            .map(|k| {
                let a = k as f64 / n as f64 * std::f64::consts::TAU;
                [8.0 * a.cos(), 8.0 * a.sin()]
            })
            .collect();
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let cyl = extrude_tool_mesh(&circle, &[], &basis, 0.0, 30.0).expect("cylinder");
        let tri_at = |m: &TriMesh, pred: &dyn Fn([f32; 3], [f32; 3]) -> bool| -> usize {
            for (ti, t) in m.indices.chunks_exact(3).enumerate() {
                let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
                let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if l < 1e-12 {
                    continue;
                }
                let n = [n[0] / l, n[1] / l, n[2] / l];
                let cen = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
                if pred(cen, n) {
                    return ti;
                }
            }
            panic!("no seed triangle matched");
        };
        // Wall seed: a mid-height triangle whose normal is radial (≈ no z).
        let wall_seed = tri_at(&cyl, &|c, n| n[2].abs() < 0.1 && c[2] > 5.0 && c[2] < 25.0);
        let (fit, boundary) = fit_region(&cyl, wall_seed, 100.0).expect("wall fit");
        match fit {
            RegionFit::Cylinder { axis, radius, center, rms, .. } => {
                assert!(axis[2].abs() > 0.99, "axis along z, got {axis:?}");
                assert!((radius - 8.0).abs() < 0.05, "radius {radius}");
                assert!((center[0].powi(2) + center[1].powi(2)).sqrt() < 0.05, "centre on the z axis: {center:?}");
                assert!(rms < 0.05, "wall rms {rms}");
            }
            other => panic!("wall should fit a cylinder, got {other:?}"),
        }
        assert!(!boundary.is_empty(), "region boundary must be reported for the highlight");
        // Cap seed: a top-face triangle (normal +z).
        let cap_seed = tri_at(&cyl, &|_, n| n[2] > 0.99);
        let (fit, _) = fit_region(&cyl, cap_seed, 100.0).expect("cap fit");
        match fit {
            RegionFit::Plane { normal, origin, rms, .. } => {
                assert!(normal[2].abs() > 0.99, "cap normal along z, got {normal:?}");
                assert!((origin[2] - 30.0).abs() < 0.05, "cap at z=30, got {origin:?}");
                assert!(rms < 0.02, "cap rms {rms}");
            }
            other => panic!("cap should fit a plane, got {other:?}"),
        }
        // Sphere r=9 centred at (3,-2,5), UV-tessellated.
        let (r, c0, nu, nv) = (9.0f32, [3.0f32, -2.0, 5.0], 48usize, 24usize);
        let mut sph = TriMesh::default();
        let pt = |iu: usize, iv: usize| {
            let (th, ph) = (iu as f32 / nu as f32 * std::f32::consts::TAU, iv as f32 / nv as f32 * std::f32::consts::PI);
            [c0[0] + r * ph.sin() * th.cos(), c0[1] + r * ph.sin() * th.sin(), c0[2] + r * ph.cos()]
        };
        for iu in 0..nu {
            for iv in 0..nv {
                let (a, b, c, d) = (pt(iu, iv), pt(iu + 1, iv), pt(iu + 1, iv + 1), pt(iu, iv + 1));
                for tri in [[a, b, c], [a, c, d]] {
                    let base = sph.positions.len() as u32;
                    for p in tri {
                        sph.positions.push(p);
                        sph.normals.push([0.0, 0.0, 1.0]);
                    }
                    sph.indices.extend([base, base + 1, base + 2]);
                }
            }
        }
        // Seed at the equator (iv = nv/2) — the iv=0 row is pole-degenerate (zero-area tris).
        let (fit, _) = fit_region(&sph, nv, 100.0).expect("sphere fit");
        match fit {
            RegionFit::Sphere { center, radius, rms, .. } => {
                assert!((radius - r).abs() < 0.1, "sphere radius {radius}");
                assert!(
                    ((center[0] - c0[0]).powi(2) + (center[1] - c0[1]).powi(2) + (center[2] - c0[2]).powi(2)).sqrt() < 0.1,
                    "sphere centre {center:?}"
                );
                assert!(rms < 0.05, "sphere rms {rms}");
            }
            other => panic!("ball should fit a sphere, got {other:?}"),
        }
    }

    /// Surface fidelity: remeshing a SPHERE must land vertices on the true surface with
    /// sub-voxel accuracy (the exact-distance band). The voxel-quantized SDF this replaces
    /// put the isosurface on voxel centers — stair-steps with error ≈ a full voxel.
    #[test]
    fn remesh_solid_surface_lands_on_the_true_surface() {
        // UV sphere, r = 10, 48×24 — plenty smooth relative to the voxel size below.
        let (r, nu, nv) = (10.0f32, 48usize, 24usize);
        let mut sph = TriMesh::default();
        let pt = |iu: usize, iv: usize| {
            let (th, ph) = (
                iu as f32 / nu as f32 * std::f32::consts::TAU,
                iv as f32 / nv as f32 * std::f32::consts::PI,
            );
            [r * ph.sin() * th.cos(), r * ph.sin() * th.sin(), r * ph.cos()]
        };
        for iu in 0..nu {
            for iv in 0..nv {
                let (a, b, c, d) = (pt(iu, iv), pt(iu + 1, iv), pt(iu + 1, iv + 1), pt(iu, iv + 1));
                let push = |m: &mut TriMesh, p: [f32; 3], q: [f32; 3], s: [f32; 3]| {
                    let base = m.positions.len() as u32;
                    for v in [p, q, s] {
                        m.positions.push(v);
                        m.normals.push([0.0, 0.0, 1.0]);
                    }
                    m.indices.extend([base, base + 1, base + 2]);
                };
                push(&mut sph, a, b, c);
                push(&mut sph, a, c, d);
            }
        }
        let res = 64usize;
        let h = 2.0 * r / res as f32; // voxel size ≈ 0.3125
        let solid = remesh_solid(&sph, res).expect("remesh sphere");
        assert!(is_manifold(&solid));
        let mut worst = 0.0f32;
        for p in &solid.positions {
            let rad = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            worst = worst.max((rad - r).abs());
        }
        // Sub-voxel: the quantized-SDF version erred by ~a full voxel (stair-steps); the
        // exact-distance band should keep every vertex well inside half a voxel of truth.
        assert!(worst < h * 0.5, "worst radial error {worst:.3} vs voxel {h:.3} — surface not landing on the true sphere");
    }

    /// `remesh_solid` must reproduce the input SHAPE, not fill its bounding box — the regression
    /// guard for the SDF-sign bug that surfaced the complement (a scan came out a solid block).
    /// An octahedron fills only ~1/6 of its bbox, so a bbox-fill bug (≈100%) is unmistakable.
    #[test]
    fn remesh_solid_reproduces_shape_not_bounding_box() {
        let s = 10.0f32;
        // Octahedron: 6 axis vertices, 8 faces, wound outward.
        let v = [[s, 0.0, 0.0], [-s, 0.0, 0.0], [0.0, s, 0.0], [0.0, -s, 0.0], [0.0, 0.0, s], [0.0, 0.0, -s]];
        let faces = [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], // top pyramid
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5], // bottom pyramid
        ];
        let mut oct = TriMesh::default();
        for f in faces {
            let base = oct.positions.len() as u32;
            for &vi in &f {
                oct.positions.push(v[vi]);
                oct.normals.push([0.0, 0.0, 1.0]);
            }
            oct.indices.extend([base, base + 1, base + 2]);
        }
        let vol = |m: &TriMesh| {
            let mut vv = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let p = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                vv += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1])) / 6.0;
            }
            vv.abs()
        };
        let bbox_vol = (2.0 * s as f64).powi(3); // 8000
        let true_vol = 4.0 / 3.0 * (s as f64).powi(3); // octahedron ≈ 1333 (17% of bbox)
        let solid = remesh_solid(&oct, 64).expect("remesh octahedron");
        assert!(is_manifold(&solid), "remeshed octahedron must be manifold");
        let vs = vol(&solid);
        // The output must be the OCTAHEDRON, not the bounding box: comfortably under half the
        // bbox (a bbox-fill bug lands near 100%), and in a plausible band around the true volume
        // (voxel remesh rounds the sharp vertices outward, so allow generous headroom).
        assert!(vs < bbox_vol * 0.5, "remesh filled the bbox (vol {vs:.0} / bbox {bbox_vol:.0}) — SDF sign regression?");
        assert!(vs > true_vol * 0.6 && vs < true_vol * 2.2, "octahedron volume {vs:.0} implausible vs true {true_vol:.0}");
    }

    /// `remesh_solid` must turn a CLOSED-but-non-manifold mesh (the common scan defect: edges
    /// shared by >2 faces) into a watertight solid that CUTS via the fast Manifold path. Build
    /// a cube, then duplicate some faces so several edges are non-manifold; remesh, confirm it's
    /// manifold and ~cube-sized, then difference a tool and confirm material was removed.
    #[test]
    fn remesh_solid_makes_a_nonmanifold_mesh_cuttable() {
        let sq = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let cube = extrude_tool_mesh(&sq, &[], &basis, 0.0, 20.0).expect("cube");
        // Closed but non-manifold: append duplicates of the first few triangles, so their edges
        // are shared by 3+ faces (Manifold declines it → the reason a scan won't cut).
        let mut bad = cube.clone();
        for t in cube.indices.chunks_exact(3).take(4) {
            let base = bad.positions.len() as u32;
            for &vi in t {
                bad.positions.push(cube.positions[vi as usize]);
                bad.normals.push([0.0, 0.0, 1.0]);
            }
            bad.indices.extend([base, base + 1, base + 2]);
        }
        assert!(!is_manifold(&bad), "duplicated faces make it non-manifold");

        let solid = remesh_solid(&bad, 48).expect("remesh");
        assert!(is_manifold(&solid), "remeshed body must be a clean manifold");
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let p = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                v += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1])) / 6.0;
            }
            v.abs()
        };
        // Roughly the 20³ = 8000 cube (voxel remesh inflates a little at coarse res).
        let vs = vol(&solid);
        assert!(vs > 8000.0 * 0.7 && vs < 8000.0 * 1.5, "remeshed volume {vs:.0} near 8000");

        // Now CUT it — a corner tool — and confirm material was removed via the fast path.
        let _ = take_dense_skip_count();
        let tsq = [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0]];
        let tbasis = PlaneBasis { origin: [0.0, 0.0, -1.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let tool = extrude_tool_mesh(&tsq, &[], &tbasis, 0.0, 25.0).expect("tool");
        let cut = mesh_difference(&solid, &tool);
        assert_eq!(take_dense_skip_count(), 0, "a manifold body must cut, not skip");
        assert!(is_manifold(&cut), "cut result stays manifold");
        assert!(vol(&cut) < vs * 0.95, "the cut must actually remove material ({:.0} → {:.0})", vs, vol(&cut));
    }

    /// Repair must close a punctured cube: delete two triangles (one quad face hole), then
    /// `repair_mesh` welds, fills the loop, and the result is manifold with ~unchanged volume.
    /// An already-clean mesh must pass through unharmed.
    #[test]
    fn repair_mesh_closes_a_punctured_cube() {
        let sq = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let cube = extrude_tool_mesh(&sq, &[], &basis, 0.0, 20.0).expect("cube");
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let p = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                v += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1])) / 6.0;
            }
            v.abs()
        };
        let v0 = vol(&cube);
        // Clean mesh: repair is a no-op topologically (weld dedupes exporter-duplicated
        // verts, but fills nothing and stays manifold).
        let (clean, rep) = repair_mesh(&cube);
        assert_eq!(rep.holes_filled, 0, "clean cube needs no fill: {rep:?}");
        assert_eq!(rep.open_edges_left, 0);
        assert!(is_manifold(&clean), "clean cube must stay manifold");
        assert!((vol(&clean) - v0).abs() < v0 * 0.001);
        // Puncture: drop two triangles that share the cube's top-front corner region.
        let mut holed = cube.clone();
        let ntri = holed.indices.len() / 3;
        let drop_a = 0usize;
        let drop_b = 1usize;
        let mut idx = Vec::new();
        for (t, tri) in holed.indices.chunks_exact(3).enumerate() {
            if t != drop_a && t != drop_b {
                idx.extend_from_slice(tri);
            }
        }
        holed.indices = idx;
        assert!(!is_manifold(&holed), "puncturing must break manifoldness");
        let (fixed, rep) = repair_mesh(&holed);
        assert!(rep.holes_filled >= 1, "the hole must be filled: {rep:?}");
        assert_eq!(rep.open_edges_left, 0, "nothing should remain open: {rep:?}");
        assert!(is_manifold(&fixed), "repaired cube must be manifold");
        let vf = vol(&fixed);
        assert!((vf - v0).abs() < v0 * 0.01, "volume {vf:.1} vs original {v0:.1}");
        assert_eq!(fixed.indices.len() / 3, ntri, "hole filled with the same-shaped patch");
    }

    /// Chaining + fitting must turn raw section chords into recognized shapes: a square's
    /// segments (subdivided, shuffled) become ONE closed 4-point polyline, and a 96-gon's
    /// chords become ONE circle with the right centre/radius — not thousands of segments.
    #[test]
    fn section_shapes_recognize_squares_and_circles() {
        // Square perimeter, each side chopped into 8 pieces, order scrambled.
        let mut segs: Vec<[[f32; 2]; 2]> = Vec::new();
        let corners = [[-10.0f32, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            for k in 0..8 {
                let t0 = k as f32 / 8.0;
                let t1 = (k + 1) as f32 / 8.0;
                let lerp = |t: f32| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                segs.push([lerp(t0), lerp(t1)]);
            }
        }
        segs.reverse();
        segs.swap(3, 17);
        let shapes = fit_section_shapes(&segs, 0.05);
        assert_eq!(shapes.len(), 1, "one shape for the square, got {shapes:?}");
        match &shapes[0] {
            SectionShape::Poly { pts, closed } => {
                assert!(*closed, "square outline should close");
                assert_eq!(pts.len(), 4, "decimated to the 4 corners, got {}", pts.len());
                for p in pts {
                    assert!((p[0].abs() - 10.0).abs() < 0.05 && (p[1].abs() - 10.0).abs() < 0.05, "corner off: {p:?}");
                }
            }
            other => panic!("expected a closed polyline, got {other:?}"),
        }

        // 96-gon approximating a circle: centre (5, -3), r = 14.
        let n = 96;
        let circ: Vec<[[f32; 2]; 2]> = (0..n)
            .map(|k| {
                let p = |k: i32| {
                    let a = k as f32 / n as f32 * std::f32::consts::TAU;
                    [5.0 + 14.0 * a.cos(), -3.0 + 14.0 * a.sin()]
                };
                [p(k), p(k + 1)]
            })
            .collect();
        let shapes = fit_section_shapes(&circ, 0.05);
        assert_eq!(shapes.len(), 1, "one shape for the circle, got {shapes:?}");
        match &shapes[0] {
            SectionShape::Circle { center, radius } => {
                assert!((center[0] - 5.0).abs() < 0.05 && (center[1] + 3.0).abs() < 0.05, "centre {center:?}");
                assert!((radius - 14.0).abs() < 0.05, "radius {radius}");
            }
            other => panic!("expected a circle, got {other:?}"),
        }
    }

    /// A stadium/slot outline (two straight sides + two semicircular ends, tessellated)
    /// must come back as arcs AND lines — the mixed case the greedy segmentation exists for.
    #[test]
    fn section_shapes_split_a_slot_into_arcs_and_lines() {
        let (r, half) = (5.0f32, 10.0f32); // slot: straight from x=-10..10, end caps r=5
        let mut pts: Vec<[f32; 2]> = Vec::new();
        for k in 0..=16 {
            let t = k as f32 / 16.0;
            pts.push([-half + 2.0 * half * t, -r]); // bottom edge left→right
        }
        for k in 1..=24 {
            let a = -std::f32::consts::FRAC_PI_2 + k as f32 / 24.0 * std::f32::consts::PI;
            pts.push([half + r * a.cos(), r * a.sin()]); // right cap, CCW
        }
        for k in 1..=16 {
            let t = k as f32 / 16.0;
            pts.push([half - 2.0 * half * t, r]); // top edge right→left
        }
        for k in 1..24 {
            let a = std::f32::consts::FRAC_PI_2 + k as f32 / 24.0 * std::f32::consts::PI;
            pts.push([-half + r * a.cos(), r * a.sin()]); // left cap, CCW (open: last≠first)
        }
        let segs: Vec<[[f32; 2]; 2]> = (0..pts.len())
            .map(|i| [pts[i], pts[(i + 1) % pts.len()]])
            .collect();
        let shapes = fit_section_shapes(&segs, 0.05);
        let arcs = shapes.iter().filter(|s| matches!(s, SectionShape::Arc { .. })).count();
        let others = shapes.len() - arcs;
        assert_eq!(arcs, 2, "two end-cap arcs, got {shapes:#?}");
        assert!(others >= 1, "the straight sides must appear too, got {shapes:#?}");
        for s in &shapes {
            if let SectionShape::Arc { center, radius, .. } = s {
                assert!((radius - r).abs() < 0.1, "cap radius {radius} (want {r})");
                assert!((center[0].abs() - half).abs() < 0.1 && center[1].abs() < 0.1, "cap centre {center:?}");
            }
        }
    }

    #[test]
    fn stl_import_roundtrips_a_cylinder_and_reads_ascii() {
        // Binary: export a cylinder, re-import, and the triangle count and volume survive.
        let cyl = extrude_solid(&circle(0.0, 0.0, 5.0, 32), &[], &plane_at(0.0), 10.0).unwrap();
        let mesh = tessellate(&cyl, 0.1).mesh;
        let blob = export_stl(&mesh);
        let back = import_stl(&blob).expect("binary STL parses");
        assert_eq!(back.indices.len(), mesh.indices.len(), "triangle count survives the roundtrip");
        assert!((mesh_vol(&back) - mesh_vol(&mesh)).abs() < 1e-3, "volume survives the roundtrip");

        // ASCII: a single-triangle solid in the text dialect.
        let ascii = b"solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid t\n";
        let tri = import_stl(ascii).expect("ascii STL parses");
        assert_eq!(tri.indices.len(), 3, "one triangle");
        assert_eq!(tri.positions[1], [1.0, 0.0, 0.0]);

        // Garbage in → None, not a panic.
        assert!(import_stl(b"not an stl at all").is_none());
    }

    #[test]
    fn step_and_stl_export_are_well_formed() {
        let cyl = extrude_solid(&circle(0.0, 0.0, 5.0, 32), &[], &plane_at(0.0), 10.0).unwrap();
        let step = export_step(&cyl).expect("step export");
        assert!(step.contains("ISO-10303-21") && step.contains("CLOSED_SHELL"), "STEP looks malformed:\n{}", &step[..step.len().min(200)]);
        let mesh = tessellate(&cyl, 0.1).mesh;
        let stl = export_stl(&mesh);
        assert_eq!(stl.len(), 84 + (mesh.indices.len() / 3) * 50, "binary STL size wrong");
    }

    #[test]
    fn a_mesh_exports_as_whole_walls_not_loose_triangles() {
        // Anything with a fillet is built by the mesh kernel and has no exact B-rep, so its STEP
        // comes from `mesh_to_solid`. That used to emit one planar face PER TRIANGLE: a box went
        // out as 12 faces where it has 6, and a real part as ~11,700 faces and 13 MB, with every
        // flat wall shattered into slivers nothing downstream could grab.
        let sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]];
        let m = extrude_tool_mesh(&sq, &[], &xy_plane(), 0.0, 4.0).expect("box mesh");
        assert_eq!(m.indices.len() / 3, 12, "a box tessellates to 12 triangles");
        let solid = mesh_to_solid(&m).expect("box -> solid");
        let step = export_step(&solid).expect("step");
        let faces = step.matches("FACE_SURFACE").count();
        assert_eq!(faces, 6, "a box has six walls; got {faces} faces");

        // ...and the walls are whole even when one is pierced, which is the case that needs the
        // boundary's holes carried onto the face rather than triangulated away.
        let bore = circle(5.0, 3.0, 1.5, 24);
        let m = extrude_tool_mesh(&sq, &[bore], &xy_plane(), 0.0, 4.0).expect("pierced box mesh");
        let solid = mesh_to_solid(&m).expect("pierced box -> solid");
        let step = export_step(&solid).expect("step");
        let faces = step.matches("FACE_SURFACE").count();
        // 4 outer walls + top + bottom (each with the bore as a hole) + one strip per bore facet.
        assert!(faces <= 6 + 24 + 2, "pierced box should be ~30 faces, got {faces}");
        assert!(faces >= 6, "sanity: at least the six walls");

        // Whatever it merged, the solid still has to enclose what the mesh did. A face that lost a
        // hole, or one truck quietly failed to build, shows up here and nowhere else.
        let want = mesh_vol(&m);
        let got = mesh_vol(&tessellate(&solid, 0.01).mesh);
        assert!((got - want).abs() <= want * 1.0e-3, "solid encloses {got:.4}, mesh {want:.4}");
    }

    /// A face tagged on the way into Manifold is still identifiable on the way out, through a
    /// CHAIN of booleans.
    ///
    /// This is the keystone of carrying surfaces rather than recovering them. HCAD never needs to
    /// reverse-engineer a cylinder out of triangles — it knows a bore is a cylinder when it builds
    /// the tool. What it lacks is a way to say so through the booleans, and Manifold has one:
    /// `reserve_ids` allocates original IDs, `MeshGLOptions::runs` attaches a block of triangles to
    /// each, and `run_original_id` reads them back on the far side. Then a STEP export can emit one
    /// real surface per tagged face instead of a ring of flat strips, with no fitting, no
    /// segmentation and no tolerance to guess at.
    ///
    /// Three things had to be true, and are: the tags attach at all, they survive a boolean, and
    /// they survive a SEQUENCE of them — the real timeline is extrude, cut, extrude, cut, and tags
    /// that only lasted one operation would be useless. Measured here: six faces tagged on box A
    /// are all still traceable after difference, union, difference.
    ///
    /// Vertices must be WELDED before Manifold sees them, or it refuses the mesh as not a solid and
    /// the tags never get a chance.
    /// A prism records the surface of every face it builds, and those records survive the app's own
    /// booleans — not just a hand-rolled Manifold call.
    ///
    /// This is the plumbing the STEP export needs: it can ask a triangle what surface it lies on
    /// instead of inferring one. Nothing consumes it yet, so the only thing that can go wrong
    /// silently is the tags quietly disappearing, which is what this watches.
    #[test]
    fn a_prism_tags_its_faces_and_the_booleans_keep_them() {
        let sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let a = direct_prism_mesh(&sq, &[], &xy_plane(), 0.0, 4.0).expect("prism");
        // Six faces: two caps and four walls, every triangle accounted for.
        assert_eq!(a.tri_surf.len(), a.indices.len() / 3, "every triangle needs a tag slot");
        assert_eq!(a.surfaces.len(), 6, "a box has six planes, got {:?}", a.surfaces.len());
        assert!(a.tri_surf.iter().all(|&s| s != NO_SURF), "some triangle came out untagged");

        // Each tag must actually describe its triangles: every vertex on the recorded plane.
        for t in 0..a.indices.len() / 3 {
            let Some(Surf::Plane { origin, normal }) = a.surf_of(t) else {
                panic!("triangle {t} lost its plane");
            };
            for i in 0..3 {
                let p = a.positions[a.indices[t * 3 + i] as usize];
                let d = (p[0] as f64 - origin[0]) * normal[0]
                    + (p[1] as f64 - origin[1]) * normal[1]
                    + (p[2] as f64 - origin[2]) * normal[2];
                assert!(d.abs() < 1.0e-6, "triangle {t} sits {d:.2e} off the plane it claims");
            }
        }
        // ...and the normals point OUT: a point just past a face must be outside the solid.
        let vol = signed_mesh_volume(&a).abs();
        assert!((vol - 400.0).abs() < 1.0e-6, "prism volume {vol}");

        // Through a real boolean, by the app's own entry point.
        let b = direct_prism_mesh(&[[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]], &[], &xy_plane(), 2.0, 4.0)
            .expect("tool");
        let cut = mesh_difference(&a, &b);
        assert!(!cut.indices.is_empty(), "the difference built nothing");
        assert_eq!(cut.tri_surf.len(), cut.indices.len() / 3, "tags did not survive the difference");
        let tagged = cut.tri_surf.iter().filter(|&&s| s != NO_SURF).count();
        assert!(
            tagged * 4 >= cut.indices.len() / 3,
            "only {tagged} of {} triangles came back tagged",
            cut.indices.len() / 3
        );
        // The four original side walls are still identifiable in the result.
        let mut kept: Vec<Surf> = Vec::new();
        for t in 0..cut.indices.len() / 3 {
            if let Some(s) = cut.surf_of(t) {
                if !kept.contains(&s) {
                    kept.push(s);
                }
            }
        }
        assert!(kept.len() >= 6, "expected the box's six planes to survive, kept {}", kept.len());
        // And every surviving tag still describes its triangles after the boolean re-meshed them.
        for t in 0..cut.indices.len() / 3 {
            let Some(Surf::Plane { origin, normal }) = cut.surf_of(t) else { continue };
            for i in 0..3 {
                let p = cut.positions[cut.indices[t * 3 + i] as usize];
                let d = (p[0] as f64 - origin[0]) * normal[0]
                    + (p[1] as f64 - origin[1]) * normal[1]
                    + (p[2] as f64 - origin[2]) * normal[2];
                assert!(d.abs() < 1.0e-4, "after the boolean, triangle {t} sits {d:.2e} off its plane");
            }
        }
    }


    /// A 48-gon approximating the circle at `c` with radius `r`, plus the annotation saying so.
    fn arc_loop(c: [f64; 2], r: f64, n: usize) -> (Vec<[f64; 2]>, ArcSpan) {
        let pts = (0..n)
            .map(|i| {
                let a = i as f64 / n as f64 * std::f64::consts::TAU;
                [c[0] + r * a.cos(), c[1] + r * a.sin()]
            })
            .collect();
        (pts, ArcSpan { first_edge: 0, count: n, center: c, radius: r })
    }

    /// Every triangle carrying `s`, and the radial distance of their vertices from its axis.
    fn tris_on(m: &TriMesh, s: Surf) -> (usize, f64, f64) {
        let (mut n, mut lo, mut hi) = (0usize, f64::MAX, f64::MIN);
        for t in 0..m.indices.len() / 3 {
            if m.surf_of(t) != Some(s) {
                continue;
            }
            n += 1;
            let Surf::Cylinder { origin, axis, .. } = s else { continue };
            for i in 0..3 {
                let p = m.positions[m.indices[t * 3 + i] as usize];
                let d = [p[0] as f64 - origin[0], p[1] as f64 - origin[1], p[2] as f64 - origin[2]];
                let al = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
                let rad = [d[0] - axis[0] * al, d[1] - axis[1] * al, d[2] - axis[2] * al];
                let rr = (rad[0] * rad[0] + rad[1] * rad[1] + rad[2] * rad[2]).sqrt();
                lo = lo.min(rr);
                hi = hi.max(rr);
            }
        }
        (n, lo, hi)
    }

    /// The one cylinder a mesh records, if it records exactly one.
    fn sole_cylinder(m: &TriMesh) -> Surf {
        let cyls: Vec<Surf> = m.surfaces.iter().copied().filter(|s| matches!(s, Surf::Cylinder { .. })).collect();
        assert_eq!(cyls.len(), 1, "expected exactly one cylinder, got {cyls:?}");
        cyls[0]
    }

    /// An extrude carries its sketch's arcs onto the mesh: a bore comes back as ONE cylinder, and
    /// the booleans that follow keep it.
    ///
    /// This is the step that makes the surface registry worth anything. Until now a mesh could
    /// record surfaces but nothing put a curved one there, so the best an exporter could do with a
    /// bore was the ring of flat strips it was handed. The sketch knew all along — an [`ArcSpan`]
    /// says which edges lie on which circle — and this is that knowledge surviving the build.
    ///
    /// The tagging happens at the mesh entry point, not inside a prism builder, because on real
    /// parts truck's triangulation answers every time: across usercylinder, blocker and motormount
    /// not one of 83 prisms reached `direct_prism_mesh`. Tags written in there would be dead code.
    #[test]
    fn an_extrudes_bore_is_recorded_as_one_cylinder() {
        let (bore, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let plate = extrude_tool_mesh_arcs(
            &rect(0.0, 0.0, 40.0, 40.0),
            &[bore.iter().rev().copied().collect()],
            &[],
            &[vec![ArcSpan { first_edge: 0, count: 48, center: span.center, radius: span.radius }]],
            &xy_plane(),
            0.0,
            5.0,
        )
        .expect("plate with a bore");

        // Two caps, four outer walls, and the bore as a single cylinder — not 48 strips.
        assert_eq!(
            plate.surfaces.len(),
            7,
            "a plate with one bore has 7 surfaces, got {:?}",
            plate.surfaces
        );
        let cyl = sole_cylinder(&plate);
        let Surf::Cylinder { origin, axis, radius } = cyl else { unreachable!() };
        assert!((radius - 6.0).abs() < 1.0e-6, "bore radius came out {radius}");
        assert!(
            (origin[0] - 20.0).abs() < 1.0e-9 && (origin[1] - 20.0).abs() < 1.0e-9,
            "bore axis passes through {origin:?}, not the sketch centre"
        );
        assert!(axis[2].abs() > 0.999, "bore axis {axis:?} is not the sweep direction");

        // Every triangle of the bore wall, and nothing else, carries it.
        let (n, lo, hi) = tris_on(&plate, cyl);
        assert_eq!(n, 96, "a 48-sided bore wall is 96 triangles, {n} carry the cylinder");
        // Positions are f32, so "on the cylinder" is judged at that precision, not f64's.
        assert!(
            (lo - 6.0).abs() < 1.0e-4 && (hi - 6.0).abs() < 1.0e-4,
            "triangles claiming the cylinder sit at radius {lo}..{hi}, not 6"
        );
        // ...and nothing came out unaccounted for: a prism is all caps and walls.
        let untagged = plate.tri_surf.iter().filter(|&&s| s == NO_SURF).count();
        assert_eq!(untagged, 0, "{untagged} triangles of a plain prism went untagged");

        // Through a real boolean, by the app's own entry point: cut a notch in a corner, far from
        // the bore, and the bore must still be a cylinder of the same size in the same place.
        let notch = extrude_tool_mesh(&rect(-1.0, -1.0, 5.0, 5.0), &[], &xy_plane(), 2.0, 4.0).expect("notch");
        let cut = mesh_difference(&plate, &notch);
        assert!(!cut.indices.is_empty(), "the difference built nothing");
        let after = sole_cylinder(&cut);
        assert_eq!(after, cyl, "the boolean changed the bore's surface: {after:?}");
        let (n2, lo2, hi2) = tris_on(&cut, after);
        assert!(n2 >= 90, "only {n2} of the bore's 96 triangles came through the boolean tagged");
        assert!(
            (lo2 - 6.0).abs() < 1.0e-3 && (hi2 - 6.0).abs() < 1.0e-3,
            "after the boolean the bore's triangles sit at radius {lo2}..{hi2}"
        );
    }


    /// A boolean on a tagged mesh survives the weld dropping a triangle, and the tags stay on the
    /// triangles they describe.
    ///
    /// The weld that prepares a mesh for Manifold removes any triangle it collapsed — that is the
    /// point of it — so the welded list is SHORTER than the source and welded triangle j is no
    /// longer source triangle j. Reading the source's tag array at a welded index therefore slides
    /// every tag past the first gap onto the wrong triangle, and then runs off the end.
    ///
    /// Not a hypothetical: the moment extrudes started tagging real profiles, this took down the
    /// rebuild of eight saved parts outright (extruderpart, fillererror3, filletpolygon,
    /// holegenieupgrade, roundfilleterror, roundfilleterror3, sliver, testpart2) and silently
    /// misfiled tags on the rest. It was invisible before only because nothing put a tag on a mesh
    /// that had a sliver in it.
    #[test]
    fn a_weld_that_drops_a_triangle_keeps_the_tags_on_the_right_ones() {
        let (bore, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let mut plate = extrude_tool_mesh_arcs(
            &rect(0.0, 0.0, 40.0, 40.0),
            &[bore.iter().rev().copied().collect()],
            &[],
            &[vec![span]],
            &xy_plane(),
            0.0,
            5.0,
        )
        .expect("plate with a bore");
        let cyl = sole_cylinder(&plate);
        let (before, _, _) = tris_on(&plate, cyl);

        // Slip a collapsed triangle in near the FRONT, so the weld's drop shifts every tag behind
        // it — the worst case, not a harmless one at the end.
        let v = plate.indices[0];
        plate.indices.splice(3..3, [v, v, v]);
        plate.tri_surf.insert(1, NO_SURF);
        assert_eq!(plate.tri_surf.len(), plate.indices.len() / 3, "the test's own mesh is out of step");

        let notch = extrude_tool_mesh(&rect(-1.0, -1.0, 5.0, 5.0), &[], &xy_plane(), 2.0, 4.0).expect("notch");
        let cut = mesh_difference(&plate, &notch);
        assert!(!cut.indices.is_empty(), "the difference built nothing");

        let after = sole_cylinder(&cut);
        assert_eq!(after, cyl, "the bore's surface changed across the boolean: {after:?}");
        let (n, lo, hi) = tris_on(&cut, after);
        assert!(n * 10 >= before * 9, "only {n} of the bore's {before} triangles came through tagged");
        // The real damage of a slipped tag isn't a missing one — it's a confident wrong one, a
        // triangle on a flat face claiming to be on the bore.
        assert!(
            (lo - 6.0).abs() < 1.0e-3 && (hi - 6.0).abs() < 1.0e-3,
            "triangles claiming the bore sit at radius {lo}..{hi}, so a tag landed on the wrong face"
        );
    }

    /// Dropping triangles takes their tags with them, so the survivors keep the right ones.
    ///
    /// Same failure as the weld, on the other side of the seam: the cleanup passes rewrite the
    /// index buffer without knowing tags exist. A tag array left at its old length describes a mesh
    /// that no longer exists.
    #[test]
    fn dropping_triangles_carries_their_tags_along() {
        let sq = rect(0.0, 0.0, 10.0, 10.0);
        let mut m = direct_prism_mesh(&sq, &[], &xy_plane(), 0.0, 4.0).expect("prism");
        let before: Vec<Option<Surf>> = (0..m.indices.len() / 3).map(|t| m.surf_of(t)).collect();
        assert!(before.iter().all(|s| s.is_some()), "a plain prism should come out fully tagged");

        // A triangle naming one vertex three times: no area, no surface, nothing to keep.
        let v = m.indices[0];
        m.indices.splice(3..3, [v, v, v]);
        m.tri_surf.insert(1, NO_SURF);
        let dropped = drop_duplicate_vertex_triangles(&mut m);
        assert_eq!(dropped, 1, "the collapsed triangle should have been the only casualty");
        assert_eq!(m.tri_surf.len(), m.indices.len() / 3, "the tag array outlived the triangles");
        let after: Vec<Option<Surf>> = (0..m.indices.len() / 3).map(|t| m.surf_of(t)).collect();
        assert_eq!(after, before, "the surviving triangles came back on different surfaces");
    }

    /// A bore the sketch drew as a circle is exported as a real cylinder.
    ///
    /// The whole point of carrying surfaces. The mesh kernel can only build a bore as a ring of
    /// flat strips — forty-eight of them, each a separate face in the STEP — so a hole that the
    /// user drew as a circle arrived downstream as a many-sided prism: unusable for a
    /// cylindricity callout, for a toolpath, or for mating in an assembly. The sketch knew the
    /// radius all along; this is it reaching the file.
    ///
    /// A true circle is WIDER than the chords standing in for it, so the body legitimately loses a
    /// little volume here. That is a correction, not drift: the part always was meant to have a
    /// round hole.
    #[test]
    fn a_bore_exports_as_a_real_cylinder() {
        let (bore, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let plate = extrude_tool_mesh_arcs(
            &rect(0.0, 0.0, 40.0, 40.0),
            &[bore.iter().rev().copied().collect()],
            &[],
            &[vec![span]],
            &xy_plane(),
            0.0,
            5.0,
        )
        .expect("plate with a bore");

        let solid = mesh_to_solid(&plate).expect("the plate should rebuild as a B-rep");
        let step = export_step(&solid).expect("STEP");
        assert_eq!(
            step.matches("SURFACE_OF_REVOLUTION").count(),
            3,
            "a full turn is written as three faces; got {}",
            step.matches("SURFACE_OF_REVOLUTION").count()
        );
        // Six flat faces would be a bore still made of strips. Two caps and four walls is the
        // whole of the rest of a plate.
        let planes = step.matches("= PLANE(").count();
        assert_eq!(planes, 6, "expected 6 flat faces beside the bore, got {planes}");

        // The same geometry every time it is asked. Finding a band means walking edge and face
        // maps, and Rust seeds every HashMap separately — WITHIN one process — so an export that
        // read hash order would hand out a different shape each time. That is the fault that made
        // fillererror3 rebuild three different ways. (The header carries a timestamp, which is
        // truck's business and not geometry, so the comparison starts at the data.)
        let body = |t: &str| t[t.find("DATA;").unwrap_or(0)..].to_string();
        for _ in 0..3 {
            let again = mesh_to_solid(&plate).and_then(|s| export_step(&s)).expect("STEP");
            assert!(body(&again) == body(&step), "the same part exported two different solids");
        }

        // The solid is the right shape, not merely the right entities: a true bore, not a prism.
        let re = tessellate(&solid, 0.002).mesh;
        let want = 40.0 * 40.0 * 5.0 - std::f64::consts::PI * 36.0 * 5.0;
        let got = signed_mesh_volume(&re).abs();
        assert!(
            (got - want).abs() < want * 1.0e-4,
            "volume {got:.4}, want {want:.4} — that is the polygon, not the circle"
        );
    }

    /// A bore a later cut has bitten into is NOT written as a cylinder.
    ///
    /// The tag still says "cylinder" and is still true of the triangles that carry it — but they no
    /// longer make the plain tube a revolved face would rebuild. Writing one anyway would fill the
    /// opening back in: the STEP would show a part that was never modelled, and nothing in the
    /// entity counts would look wrong.
    ///
    /// Several of the checks could catch this; measured, it is the first — the cut leaves the
    /// band's own faces holding triangles that are not on the cylinder at all, so the tag no longer
    /// describes a surface that can be swapped out whole.
    #[test]
    fn a_bore_a_cut_has_opened_is_not_written_as_a_cylinder() {
        let (bore, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let plate = extrude_tool_mesh_arcs(
            &rect(0.0, 0.0, 40.0, 40.0),
            &[bore.iter().rev().copied().collect()],
            &[],
            &[vec![span]],
            &xy_plane(),
            0.0,
            5.0,
        )
        .expect("plate");
        // A slot straight through one side of the bore, so the ring is no longer closed.
        let slot = extrude_tool_mesh(&rect(14.0, 18.0, 26.0, 22.0), &[], &xy_plane(), 1.0, 3.0).expect("slot");
        let cut = mesh_difference(&plate, &slot);
        assert!(
            cut.surfaces.iter().any(|s| matches!(s, Surf::Cylinder { .. })),
            "the cut should leave the cylinder TAG in place — that is what makes this a real test"
        );

        let solid = mesh_to_solid(&cut).expect("B-rep");
        let step = export_step(&solid).expect("STEP");
        assert_eq!(
            step.matches("SURFACE_OF_REVOLUTION").count(),
            0,
            "an opened bore was written as a closed cylinder, filling the slot back in"
        );
        // And the solid still is what the mesh was.
        let re = tessellate(&solid, 0.01).mesh;
        let (got, want) = (signed_mesh_volume(&re).abs(), signed_mesh_volume(&cut).abs());
        assert!((got - want).abs() < want * 1.0e-3, "volume {got:.4} against the mesh's {want:.4}");
    }


    /// A boss is a cylinder too, and comes out the right way round.
    ///
    /// Which side of a tube the material is on cannot be checked by building the solid: the flat
    /// face beside a rim is given whatever direction the tube leaves it, so a tube threaded in
    /// backwards still closes into a perfectly valid solid — one with a collar of material where
    /// the hole should be, or a hole where the boss should be. Nothing about the shell complains;
    /// only the volume tells you. It went in backwards first time round, and every bore in the
    /// corpus came out filled.
    #[test]
    fn a_boss_exports_as_a_cylinder_of_material_not_a_hole() {
        let (circ, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let plate = extrude_tool_mesh(&rect(0.0, 0.0, 40.0, 40.0), &[], &xy_plane(), 0.0, 5.0).expect("plate");
        let boss = extrude_tool_mesh_arcs(&circ, &[], &[span], &[], &xy_plane(), 4.0, 6.0).expect("boss");
        let body = mesh_union(&plate, &boss);
        let solid = mesh_to_solid(&body).expect("B-rep");
        let step = export_step(&solid).expect("STEP");
        assert_eq!(step.matches("SURFACE_OF_REVOLUTION").count(), 3, "the boss did not come out round");

        // Plate, plus the boss standing proud of it. Backwards, this would be the plate MINUS a
        // bore, and the difference is far too big to hide inside a tolerance.
        let want = 40.0 * 40.0 * 5.0 + std::f64::consts::PI * 36.0 * 5.0;
        let got = signed_mesh_volume(&tessellate(&solid, 0.002).mesh).abs();
        assert!(
            (got - want).abs() < want * 1.0e-4,
            "volume {got:.4}, want {want:.4} — the tube went in inside out"
        );
    }

    /// PROBE: can a PARTIAL sweep's own edges be shared with hand-built flat faces?
    ///
    /// A whole ring shares only its two rims, and both come from the sweep. A partial one also
    /// shares two straight ends with ordinary faces — and truck's topology is identity-based, so
    /// those faces have to be built from the sweep's own vertices, not from equal-looking ones.
    #[test]
    #[ignore]
    fn diag_partial_sweep_sharing() {
        let (r, h, ang) = (10.0_f64, 4.0_f64, std::f64::consts::FRAC_PI_2);
        let axis = Vector3::unit_z();
        let seed = builder::vertex(Point3::new(r, 0.0, 0.0));
        let rise = builder::tsweep(&seed, Vector3::new(0.0, 0.0, h));
        let swept = builder::rsweep(&rise, Point3::origin(), axis, truck_modeling::Rad(ang));
        let wall: Vec<truck_modeling::Face> = swept.iter().cloned().collect();
        eprintln!("PARTIAL: wall is {} face(s)", wall.len());

        // Its boundary: two arcs (bottom and top) and the two straight ends.
        let key = |e: &truck_modeling::Edge| {
            let q = |p: Point3| ((p.x * 1.0e5).round() as i64, (p.y * 1.0e5).round() as i64, (p.z * 1.0e5).round() as i64);
            let (a, z) = (q(e.front().point()), q(e.back().point()));
            if a < z { (a, z) } else { (z, a) }
        };
        let mut seen: std::collections::HashMap<_, usize> = Default::default();
        for f in &wall {
            for w in f.boundaries() {
                for e in w.iter() {
                    *seen.entry(key(e)).or_default() += 1;
                }
            }
        }
        let mut bnd: Vec<truck_modeling::Edge> = Vec::new();
        for f in &wall {
            for w in f.boundaries() {
                for e in w.iter() {
                    if seen.get(&key(e)).copied().unwrap_or(0) == 1 {
                        bnd.push(e.clone());
                    }
                }
            }
        }
        let flat = |e: &truck_modeling::Edge| (e.front().point().z - e.back().point().z).abs() < 1.0e-9;
        let (arcs, ends): (Vec<_>, Vec<_>) = bnd.iter().cloned().partition(flat);
        eprintln!("  boundary: {} arc edge(s), {} straight end(s)", arcs.len(), ends.len());
        let lo: Vec<_> = arcs.iter().filter(|e| e.front().point().z.abs() < 1.0e-9).cloned().collect();
        let hi: Vec<_> = arcs.iter().filter(|e| (e.front().point().z - h).abs() < 1.0e-9).cloned().collect();

        // The axis edge both caps share, built from the sweep's OWN end vertices.
        let (c0, c1) = (builder::vertex(Point3::origin()), builder::vertex(Point3::new(0.0, 0.0, h)));
        let spine = builder::line(&c0, &c1);
        // A cap is: axis -> out along one end's bottom vertex -> round the arc -> back to the axis.
        let cap = |arc: &Vec<truck_modeling::Edge>, at: &truck_modeling::Vertex, inv: bool| {
            let chain = |es: &Vec<truck_modeling::Edge>| {
                let q = |p: Point3| ((p.x * 1.0e5).round() as i64, (p.y * 1.0e5).round() as i64, (p.z * 1.0e5).round() as i64);
                let mut left = es.clone();
                let mut w = truck_modeling::Wire::new();
                let first = left.remove(0);
                let mut cur = q(first.back().point());
                w.push_back(first);
                while !left.is_empty() {
                    let Some(i) = left.iter().position(|e| q(e.front().point()) == cur) else { break };
                    let e = left.remove(i);
                    cur = q(e.back().point());
                    w.push_back(e);
                }
                w
            };
            let run = chain(arc);
            let mut w = truck_modeling::Wire::new();
            w.push_back(builder::line(at, &run.front_vertex().unwrap().clone()));
            for e in run.iter() {
                w.push_back(e.clone());
            }
            w.push_back(builder::line(&run.back_vertex().unwrap().clone(), at));
            let _ = inv;
            builder::try_attach_plane(&[w])
        };
        eprintln!("  bottom cap: {:?}", cap(&lo, &c0, false).is_ok());
        eprintln!("  top cap:    {:?}", cap(&hi, &c1, true).is_ok());
        eprintln!("  spine + ends available: {} / {}", ends.len(), 2);
        let _ = spine;
    }

    /// Half a bore is not a bore: a wall cut back to a half-pipe stays faceted.
    ///
    /// This is what the boundary-loop count is for, and the case that decided its shape. Half a
    /// tube still has two flat, complete-looking arcs at top and bottom, so asking only "are the
    /// ends circles?" would wave it through and write a whole cylinder where half the material had
    /// been taken away. What separates them is that a partial ring's boundary is ONE loop — up the
    /// cut end, round the top, down the other end, back along the bottom — where a real tube's is
    /// two.
    #[test]
    fn half_a_bore_is_not_written_as_a_whole_one() {
        let (bore, span) = arc_loop([20.0, 20.0], 6.0, 48);
        let plate = extrude_tool_mesh_arcs(
            &rect(0.0, 0.0, 40.0, 40.0),
            &[bore.iter().rev().copied().collect()],
            &[],
            &[vec![span]],
            &xy_plane(),
            0.0,
            5.0,
        )
        .expect("plate");
        // Take one whole side away, through the full thickness.
        let side = extrude_tool_mesh(&rect(20.0, -1.0, 41.0, 41.0), &[], &xy_plane(), -1.0, 7.0).expect("side");
        let cut = mesh_difference(&plate, &side);
        assert!(
            cut.surfaces.iter().any(|s| matches!(s, Surf::Cylinder { .. })),
            "the cut should leave the cylinder TAG in place — that is what makes this a real test"
        );
        let solid = mesh_to_solid(&cut).expect("B-rep");
        let step = export_step(&solid).expect("STEP");
        assert_eq!(
            step.matches("SURFACE_OF_REVOLUTION").count(),
            0,
            "a half-pipe was written as a whole cylinder, putting back material that was cut away"
        );
        let re = tessellate(&solid, 0.01).mesh;
        let (got, want) = (signed_mesh_volume(&re).abs(), signed_mesh_volume(&cut).abs());
        assert!((got - want).abs() < want * 1.0e-3, "volume {got:.4} against the mesh's {want:.4}");
    }

    /// A flat across a bore stays flat: the chord is NOT swallowed into the cylinder it cuts.
    ///
    /// This is the trap that decides how the tagging has to work. A D-shaped bore's flat has both
    /// its endpoints exactly on the circle and faces straight along the radius — geometrically it
    /// is indistinguishable from one facet of the polygon approximating that circle. No radius
    /// test, however tight, can separate them. Which edges belong to the arc is something only the
    /// sketch knows, so triangles are placed by WHICH PROFILE EDGE swept them, and the annotation
    /// is believed about exactly the edges it names.
    ///
    /// Get this wrong and the flat exports as part of the round hole — a D-bore silently becomes a
    /// plain one, and the part no longer keys onto its shaft.
    #[test]
    fn a_flat_across_a_bore_is_not_mistaken_for_the_bore() {
        // A 48-gon with a run of 12 edges replaced by one straight chord: the D-bore.
        let (full, _) = arc_loop([0.0, 0.0], 6.0, 48);
        let kept = 36;
        let mut d_bore: Vec<[f64; 2]> = full[..=kept].to_vec(); // 37 points ⇒ 36 arc edges + 1 chord
        // Points 0..=36 leave the closing edge 36→0 as the flat.
        assert_eq!(d_bore.len(), 37);
        let chord = (
            d_bore[kept],
            d_bore[0],
            ((d_bore[kept][0] - d_bore[0][0]).powi(2) + (d_bore[kept][1] - d_bore[0][1]).powi(2)).sqrt(),
        );
        assert!(chord.2 > 5.0, "the flat should be a long chord, it is {}", chord.2);
        d_bore.reverse(); // holes run the other way round
        // After reversing, the flat is edge 0 and the arc is edges 1..=36.
        let spans = vec![ArcSpan { first_edge: 1, count: 36, center: [0.0, 0.0], radius: 6.0 }];

        let part = extrude_tool_mesh_arcs(
            &rect(-15.0, -15.0, 15.0, 15.0),
            &[d_bore],
            &[],
            &[spans],
            &xy_plane(),
            0.0,
            4.0,
        )
        .expect("plate with a D-bore");

        let cyl = sole_cylinder(&part);
        let (n, _, hi) = tris_on(&part, cyl);
        assert_eq!(n, 72, "the 36 arc edges are 72 triangles, {n} carry the cylinder");
        assert!((hi - 6.0).abs() < 1.0e-4, "a triangle on the cylinder reaches radius {hi}");

        // The flat has a plane of its own, and its triangles are on it.
        let flat: Vec<Surf> = part
            .surfaces
            .iter()
            .copied()
            .filter(|s| match s {
                // A wall (not a cap) anchored on the bore's rim: the outer square's walls are
                // out at radius 21, so only the chord qualifies.
                Surf::Plane { origin, normal } => {
                    normal[2].abs() < 0.01 && (origin[0].hypot(origin[1]) - 6.0).abs() < 0.01
                }
                _ => false,
            })
            .collect();
        assert!(!flat.is_empty(), "the flat got no plane of its own: {:?}", part.surfaces);
        let on_flat = (0..part.indices.len() / 3)
            .filter(|&t| part.surf_of(t).is_some_and(|s| flat.contains(&s)))
            .count();
        assert_eq!(on_flat, 2, "the flat is one quad — 2 triangles — but {on_flat} sit on it");
    }

    #[test]
    fn manifold_carries_face_tags_through_a_chain_of_booleans() {
        use manifold3d::{Manifold, MeshGL};
        let boxes = |x0: f64, y0: f64, x1: f64, y1: f64, z0: f64, h: f64| {
            crate::extrude_tool_mesh(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], &[], &xy_plane(), z0, h)
        };
        let a = boxes(0.0, 0.0, 10.0, 10.0, 0.0, 10.0).expect("box a");
        let b = boxes(5.0, 5.0, 15.0, 15.0, 2.0, 12.0).expect("box b");
        let c = boxes(2.0, 2.0, 4.0, 4.0, 8.0, 6.0).expect("box c");
        let d = boxes(7.0, 1.0, 9.0, 3.0, -1.0, 5.0).expect("box d");

        // Tag every coplanar face of a mesh with its own original ID. Runs index into the FLAT
        // tri_verts array and carry one ID each, with a sentinel at the end.
        let tag = |m: &TriMesh| -> (Manifold, std::ops::Range<u32>) {
            let ntri = m.indices.len() / 3;
            let plane_of = |t: usize| {
                let g = |i: usize| m.positions[m.indices[t * 3 + i] as usize];
                let (p, q, r) = (g(0), g(1), g(2));
                let (e1, e2) = (
                    [q[0] - p[0], q[1] - p[1], q[2] - p[2]],
                    [r[0] - p[0], r[1] - p[1], r[2] - p[2]],
                );
                let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
                let n = [n[0] / l, n[1] / l, n[2] / l];
                let d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
                [(n[0] * 1e4) as i64, (n[1] * 1e4) as i64, (n[2] * 1e4) as i64, (d * 1e4) as i64]
            };
            let mut order: Vec<usize> = (0..ntri).collect();
            order.sort_by_key(|&t| plane_of(t));
            let (mut verts, mut tris) = (Vec::<f32>::new(), Vec::<u32>::new());
            let (mut run_index, mut run_ids) = (Vec::<u32>::new(), Vec::<u32>::new());
            let mut weld: std::collections::HashMap<(i64, i64, i64), u32> = Default::default();
            let base = manifold3d::reserve_ids(ntri as u32);
            let (mut last, mut nfaces) = (None, 0u32);
            for &t in &order {
                let pl = plane_of(t);
                if last != Some(pl) {
                    run_index.push(tris.len() as u32);
                    run_ids.push(base + nfaces);
                    nfaces += 1;
                    last = Some(pl);
                }
                for i in 0..3 {
                    let p = m.positions[m.indices[t * 3 + i] as usize];
                    let k = ((p[0] * 1e4) as i64, (p[1] * 1e4) as i64, (p[2] * 1e4) as i64);
                    let id = *weld.entry(k).or_insert_with(|| {
                        verts.extend_from_slice(&p);
                        (verts.len() / 3 - 1) as u32
                    });
                    tris.push(id);
                }
            }
            run_index.push(tris.len() as u32); // sentinel
            let opts = manifold3d::MeshGLOptions::new().runs(&run_index, &run_ids);
            let mgl = MeshGL::new_with_options(&verts, 3, &tris, opts).expect("tagged MeshGL");
            (Manifold::from_meshgl(&mgl).expect("tagged Manifold"), base..base + nfaces)
        };

        let (ma, a_ids) = tag(&a);
        let (mb, _) = tag(&b);
        let (mc, _) = tag(&c);
        let (md, _) = tag(&d);
        let a_ids: std::collections::BTreeSet<u32> = a_ids.collect();
        assert_eq!(a_ids.len(), 6, "a box has six faces to tag");

        let mut man = ma.difference(&mb);
        for (label, other, is_cut) in [("union C", &mc, false), ("difference D", &md, true)] {
            let ids: std::collections::BTreeSet<u32> = man.to_meshgl().run_original_id().into_iter().collect();
            assert!(
                ids.is_superset(&a_ids),
                "before {label}, only {} of A's 6 faces are still traceable",
                ids.intersection(&a_ids).count()
            );
            man = if is_cut { man.difference(other) } else { man.union(other) };
        }
        let ids: std::collections::BTreeSet<u32> = man.to_meshgl().run_original_id().into_iter().collect();
        assert!(
            ids.is_superset(&a_ids),
            "after three booleans only {} of A's 6 faces trace back — ids present: {ids:?}",
            ids.intersection(&a_ids).count()
        );
    }

    #[test]
    fn mesh_to_solid_exports_faceted_step() {
        // A mesh-only body (here a loft, which has no exact B-rep) → faceted solid → STEP.
        let m = loft_mesh(&[(circle3(0.0, 0.0, 0.0, 5.0, 24), vec![]), (circle3(0.0, 0.0, 10.0, 3.0, 24), vec![])]).unwrap();
        let solid = mesh_to_solid(&m).expect("faceted solid from mesh");
        let step = export_step(&solid).expect("step from faceted solid");
        assert!(step.contains("ISO-10303-21"), "faceted STEP malformed");
        // The frustum has 24 facet strips and two caps. It is NOT the ~100 faces this asked for
        // before: a strip is one face now, not the two triangles it is drawn with, and each flat
        // cap is one face rather than its whole fan. Still faceted — truck has no cone to put
        // there — just not shattered.
        let faces = step.matches("FACE_SURFACE").count();
        let tris = m.indices.len() / 3;
        assert!(faces < tris, "{faces} faces from {tris} triangles — nothing merged");
        assert!(faces >= 24, "a 24-segment frustum needs at least its 24 side strips, got {faces}");
        // And it still encloses the frustum it came from.
        let want = mesh_vol(&m);
        let got = mesh_vol(&tessellate(&solid, 0.01).mesh);
        assert!((got - want).abs() <= want * 1.0e-3, "solid encloses {got:.4}, mesh {want:.4}");
    }

    #[test]
    fn loft_two_circles_is_a_clean_frustum() {
        // Loft a r=5 circle at z=0 to a r=2 circle at z=10 → a cone frustum. Must be watertight
        // and have the frustum volume V = π·h·(R²+R·r+r²)/3.
        let a = circle3(0.0, 0.0, 0.0, 5.0, 40);
        let b = circle3(0.0, 0.0, 10.0, 2.0, 24);
        let m = loft_mesh(&[(a, vec![]), (b, vec![])]).expect("loft builds");
        assert!(is_manifold(&m), "loft result isn't watertight");
        let want = std::f64::consts::PI * 10.0 * (25.0 + 10.0 + 4.0) / 3.0;
        let got = mesh_vol(&m);
        assert!((got - want).abs() / want < 0.02, "frustum volume {got:.1} (want {want:.1})");
    }

    #[test]
    fn loft_two_annuli_keeps_the_hole() {
        // Loft a ring (outer 5, hole 3) at z=0 to a ring (outer 8, hole 4) at z=10. The result must
        // be a hollow tapered tube — watertight, with volume = outer frustum − inner frustum.
        let frustum = |big: f64, small: f64| std::f64::consts::PI * 10.0 * (big * big + big * small + small * small) / 3.0;
        let p0 = (circle3(0.0, 0.0, 0.0, 5.0, 48), vec![circle3(0.0, 0.0, 0.0, 3.0, 40)]);
        let p1 = (circle3(0.0, 0.0, 10.0, 8.0, 48), vec![circle3(0.0, 0.0, 10.0, 4.0, 40)]);
        let m = loft_mesh(&[p0, p1]).expect("annulus loft builds");
        assert!(is_manifold(&m), "annular loft isn't watertight (hole not skinned/capped)");
        let want = frustum(5.0, 8.0) - frustum(3.0, 4.0);
        let got = mesh_vol(&m);
        assert!((got - want).abs() / want < 0.03, "hollow loft volume {got:.1} (want {want:.1})");
    }

    #[test]
    fn coaxial_revolve_cut_groove_stays_manifold() {
        // From the user's "bad revolve.hcad": a cylinder (r=57.29, axis +Y, h=189.2) with a torus
        // groove cut into its wall — the torus tube centre is at r=57.96 about the SAME Y axis,
        // minor r=21.35. truck's tessellation of this large revolve came out non-watertight, so
        // Manifold rejected the difference (→ lossy BSP → torn surface / OOM). The direct full-turn
        // revolve mesh is watertight at any scale, so the cut now stays 2-manifold.
        let top = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] };
        let front = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let cyl = extrude_tool_mesh(&circle(0.0, 0.0, 57.29, 128), &[], &top, 0.0, 189.2).unwrap();
        let torus = revolve_tool_mesh(&circle(-57.96, 106.4, 21.35, 128), &[], &front, [0.0, 0.627], [0.0, 211.6], std::f64::consts::TAU).unwrap();
        assert!(is_manifold(&torus), "the file's torus must be a watertight manifold now");
        let d = mesh_difference(&cyl, &torus);
        assert!(!d.indices.is_empty() && is_manifold(&d), "coaxial groove cut not a clean manifold");
    }

    #[test]
    fn revolve_overlapping_union_is_clean() {
        // The user's actual case: a torus revolved so it straddles the cylinder wall (a bead
        // around the cylinder). The union must go through Manifold and stay 2-manifold — if it
        // falls back to BSP it leaves overlapping shells (the torn/striped surface).
        let cyl = extrude_tool_mesh(&circle(0.0, 0.0, 5.0, 64), &[], &plane_at(-10.0), 0.0, 20.0).unwrap();
        let torus = revolve_tool_mesh(&circle(5.0, 0.0, 2.0, 48), &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::TAU).unwrap();
        let u = mesh_union(&cyl, &torus);
        assert!(!u.indices.is_empty(), "overlapping union empty");
        assert!(is_manifold(&u), "overlapping union isn't manifold → BSP fallback → torn shells");
    }

    #[test]
    fn revolve_mesh_is_a_valid_manifold() {
        // A full-turn revolve must ingest as a 2-manifold, or every boolean with it falls back to
        // the lossy BSP CSG → torn/overlapping shells. (Cylinder for contrast.)
        let cyl = extrude_tool_mesh(&circle(0.0, 0.0, 5.0, 48), &[], &plane_at(-10.0), 0.0, 20.0).unwrap();
        assert!(is_manifold(&cyl), "extrude mesh isn't manifold");
        let torus = revolve_tool_mesh(&circle(20.0, 0.0, 2.0, 32), &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::TAU).unwrap();
        assert!(is_manifold(&torus), "full-turn revolve mesh isn't manifold (booleans will tear)");
    }
    fn plane_at(z: f64) -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, z], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    fn mesh_vol(m: &TriMesh) -> f64 {
        let mut v = 0.0;
        for t in m.indices.chunks_exact(3) {
            let p: Vec<[f64; 3]> = t.iter().map(|&i| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] }).collect();
            v += p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
        }
        (v / 6.0).abs()
    }

    #[test]
    fn revolve_rectangle_full_turn_is_a_washer() {
        // Rectangle u∈[1,2], v∈[0,4] revolved 360° about the v-axis (u=0) → a cylindrical washer:
        // inner r=1, outer r=2, height 4. Volume = π(2²−1²)·4 = 12π ≈ 37.70.
        let prof = rect(1.0, 0.0, 2.0, 4.0);
        let m = revolve_tool_mesh(&prof, &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::TAU)
            .expect("full revolve builds");
        let want = std::f64::consts::PI * 3.0 * 4.0;
        assert!((mesh_vol(&m) - want).abs() < 1.0, "washer volume {} (want {want})", mesh_vol(&m));
    }

    #[test]
    fn revolve_half_turn_is_half_volume() {
        let prof = rect(1.0, 0.0, 2.0, 4.0);
        let m = revolve_tool_mesh(&prof, &[], &xy_plane(), [0.0, 0.0], [0.0, 1.0], std::f64::consts::PI)
            .expect("half revolve builds");
        let want = std::f64::consts::PI * 3.0 * 4.0 / 2.0;
        assert!((mesh_vol(&m) - want).abs() < 1.0, "half washer volume {} (want {want})", mesh_vol(&m));
    }

    #[test]
    fn stacked_bosses_regenerate_with_small_overlap() {
        // base box 4×4×2, then two more bosses stacked on top faces — the
        // "after two extrusions" case. Each union uses the app's small overlap.
        let ov = 2.0e-3;
        let tol = 1.0e-4;
        let base = extrude_solid(&rect(0.0, 0.0, 4.0, 4.0), &[], &plane_at(0.0), 2.0).unwrap();
        let boss1 = extrude_solid_with_overlap(&rect(1.0, 1.0, 3.0, 3.0), &[], &plane_at(2.0), 2.0, ov).unwrap();
        let s1 = union_tol(&base, &boss1, tol).expect("first boss unions");
        let boss2 = extrude_solid_with_overlap(&rect(1.5, 1.5, 2.5, 2.5), &[], &plane_at(4.0), 2.0, ov).unwrap();
        let s2 = union_tol(&s1, &boss2, tol).expect("second boss unions");
        assert!(tessellate(&s2, 0.05).edges.len() > 12);
    }

    #[test]
    fn degenerate_contour_is_cleaned_not_crashed() {
        // A loop with a duplicate vertex and an antenna spike — truck would panic
        // ("wire is not simple") on this raw, but clean_loop fixes it to a square.
        let pts = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0], // duplicate
            [2.0, 2.0],
            [1.0, 3.0], // spike tip
            [2.0, 2.0], // back — spike
            [0.0, 2.0],
        ];
        let solid = extrude_solid(&pts, &[], &xy_plane(), 1.0).expect("cleaned square extrudes");
        // Cleaned to a 4-edge square prism → 12 feature edges.
        assert_eq!(tessellate(&solid, 0.05).edges.len(), 12);
    }

    #[test]
    fn clean_loop_removes_spikes_and_duplicates() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 3.0], [2.0, 2.0], [0.0, 2.0]];
        assert_eq!(clean_loop(&pts).len(), 4, "should reduce to a clean quad");
    }

    #[test]
    fn thread_depth_survives_off_face_placement() {
        // Replay of holegenieupgrade.hcad: the Hole Genie placement point sits a few
        // thousandths ABOVE the face it was clicked on (face-pick snap error). The depth clamp
        // used to cast its exit ray from a 1e-3 step, hit the ENTRY face 0.004 away, read the
        // body as paper-thin, and cap the thread at the 0.5 minimum — "the hole only goes a
        // small depth and stops".
        let block = extrude_tool_mesh(&[[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]], &[], &xy_plane(), 0.0, 20.0)
            .expect("block");
        let origin = [15.0, 15.0, 20.0039]; // slightly OFF the top face, like the logged file
        let out = threaded_hole(&block, origin, [0.0, 0.0, 1.0], 5.0, 0.8, 9.0, true, true).expect("thread");
        // Signed volume (divergence theorem): the block is exactly 30×30×20 = 18000. A 9-deep
        // Ø5 bore removes ~πr²·9 ≈ 177 (the thread ridges union a fraction back). The old bug
        // clamped the hole to 0.5 deep — removing barely ~10 — so require a healthy chunk gone.
        let volume = |m: &TriMesh| -> f64 {
            let mut v = 0.0;
            for t in m.indices.chunks(3) {
                let g = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                let (a, b, c) = (g(t[0]), g(t[1]), g(t[2]));
                v += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2])
                    + a[2] * (b[0] * c[1] - c[0] * b[1]))
                    / 6.0;
            }
            v.abs()
        };
        let removed = volume(&block) - volume(&out);
        assert!(
            removed > 80.0,
            "thread only removed {removed:.1} of material — the depth clamp cut the hole short (expected ~120+ for a 9-deep Ø5 tap)"
        );

        // THROUGH hole: ask for a depth past the far side (block is 20 tall, ask 25). The bore
        // must punch clean out the bottom — the old clamp left a paper-thin floor skin. A full
        // Ø5 bore through 20 removes ~πr²·20 ≈ 393 (threads union a fraction back).
        let out = threaded_hole(&block, origin, [0.0, 0.0, 1.0], 5.0, 0.8, 25.0, true, true).expect("through");
        let removed = volume(&block) - volume(&out);
        assert!(
            removed > 250.0,
            "through-tap removed only {removed:.1} — the bore didn't punch out the far side"
        );
    }

    /// Every Hole Genie boolean must go through Manifold — the lossy BSP fallback tears surfaces
    /// and fired the "1 boolean(s) used the lossy fallback" banner on EVERY thread.
    #[test]
    fn hole_genie_booleans_stay_manifold() {
        let _ = take_fallback_count(); // reset the counter
        let block = extrude_tool_mesh(&[[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]], &[], &xy_plane(), 0.0, 20.0)
            .expect("block");
        let _tap = threaded_hole(&block, [15.0, 15.0, 20.0], [0.0, 0.0, 1.0], 5.0, 0.8, 9.0, true, true).expect("tap");
        assert_eq!(take_fallback_count(), 0, "internal tap used the lossy BSP fallback");
        let _ext = threaded_hole(&block, [15.0, 15.0, 20.0], [0.0, 0.0, 1.0], 5.0, 0.8, 9.0, false, true).expect("ext");
        assert_eq!(take_fallback_count(), 0, "external thread used the lossy BSP fallback");
    }

    #[test]
    fn extrude_square_makes_a_box() {
        let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let solid = extrude_solid(&square, &[], &xy_plane(), 2.0).expect("extrude");
        let t = tessellate(&solid, 0.05);
        assert!(t.mesh.indices.len() >= 36, "got {} indices", t.mesh.indices.len());
        assert_eq!(t.mesh.indices.len() % 3, 0);
        // A closed box has 12 feature edges.
        assert_eq!(t.edges.len(), 12, "box should have 12 edges, got {}", t.edges.len());
    }

    #[test]
    fn boss_on_a_top_face_unions_into_a_stepped_solid() {
        // 4×4×2 base on the XY plane.
        let base = extrude_solid(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], &[], &xy_plane(), 2.0)
            .expect("base");
        // A 2×2 boss on the top face (z = 2), overlapping back into the base.
        let top = PlaneBasis {
            origin: [0.0, 0.0, 2.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let boss = extrude_solid_with_overlap(&[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]], &[], &top, 2.0, 0.1)
            .expect("boss");
        let combined = union(&base, &boss).expect("union should succeed");
        let t = tessellate(&combined, 0.05);
        assert!(t.mesh.indices.len() % 3 == 0 && !t.mesh.positions.is_empty());
        assert!(t.edges.len() > 12, "a stepped solid has more than 12 edges, got {}", t.edges.len());
    }

    #[test]
    fn cut_into_a_top_face_with_negative_distance() {
        // 4×4×2 base; cut downward from the top face (body is on the −normal side).
        let base = extrude_solid(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], &[], &xy_plane(), 2.0)
            .expect("base");
        let top = PlaneBasis {
            origin: [0.0, 0.0, 2.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        // Negative distance ⇒ tool sweeps against the normal, i.e. down into the body.
        let result = cut(&base, &[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]], &[], &top, -2.0)
            .expect("downward cut should succeed");
        let t = tessellate(&result, 0.05);
        assert!(t.edges.len() > 12, "pocketed solid should have extra edges, got {}", t.edges.len());
    }

    #[test]
    fn extrude_a_square_with_a_hole_makes_a_frame() {
        let outer = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let hole = vec![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let solid = extrude_solid(&outer, std::slice::from_ref(&hole), &xy_plane(), 2.0)
            .expect("frame should extrude");
        let t = tessellate(&solid, 0.05);
        assert!(t.edges.len() > 12, "frame should have inner+outer edges, got {}", t.edges.len());
        assert!(t.mesh.indices.len() % 3 == 0 && !t.mesh.positions.is_empty());
    }

    #[test]
    fn cutting_a_pocket_reduces_volume_and_adds_edges() {
        // Base: 4×4 box, 2 tall.
        let base = extrude_solid(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], &[], &xy_plane(), 2.0)
            .expect("base");
        // Cut a centered 2×2 pocket straight through.
        let pocket = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let result = cut(&base, &pocket, &[], &xy_plane(), 2.0).expect("cut should succeed");
        let t = tessellate(&result, 0.05);
        // A box with a rectangular through-hole has more than the 12 edges of a plain box.
        assert!(t.edges.len() > 12, "cut result should have extra edges, got {}", t.edges.len());
        assert!(t.mesh.indices.len() % 3 == 0 && !t.mesh.positions.is_empty());
    }

    #[test]
    fn direction_two_extends_the_prism_both_ways() {
        // A 2×2 square, Direction 1 = 3 (z: 0..3), Direction 2 `back` = 1 (z: -1..0).
        // The both-directions prism (start = -back, length = d + back) must span z ∈ [-1, 3].
        let sq = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let (d, back) = (3.0_f64, 1.0_f64);
        let m = extrude_tool_mesh(&sq, &[], &xy_plane(), -back, d + back).expect("prism");
        let (mut zlo, mut zhi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &m.positions {
            zlo = zlo.min(p[2]);
            zhi = zhi.max(p[2]);
        }
        assert!((zlo - -1.0).abs() < 1e-4, "Direction 2 should reach z=-1, got {zlo}");
        assert!((zhi - 3.0).abs() < 1e-4, "Direction 1 should reach z=3, got {zhi}");
    }

    #[test]
    fn cylinder_rims_are_clean_closed_loops() {
        // A plain cylinder's top + bottom rims must each be a clean closed loop in the displayed
        // edges — no dangling vertices (a dangle is the "circle edge break").
        use std::collections::HashMap;
        let cyl = extrude_tool_mesh(&circle(0.0, 0.0, 20.0, 48), &[], &plane_at(0.0), 0.0, 50.0).expect("cyl");
        let tess = mesh_tessellation(cyl);
        let key = |p: [f32; 3]| ((p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64, (p[2] * 1e3).round() as i64);
        let mut deg: HashMap<(i64, i64, i64), u32> = HashMap::new();
        for e in &tess.edges {
            *deg.entry(key(e[0])).or_default() += 1;
            *deg.entry(key(e[1])).or_default() += 1;
        }
        assert!(deg.values().all(|&d| d == 2), "every rim vertex should have degree 2 (closed loops)");
        assert_eq!(tess.edges.len(), 96, "two 48-segment rims");
    }

    #[test]
    fn cleanup_prunes_a_short_spur() {
        // A closed 10×10 square plus a tiny spur off a corner (a boolean-seam sliver). The short spur
        // is pruned; the four loop edges stay.
        let pos = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0], [10.4, 10.4, 0.0]];
        let ids = vec![(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)]; // (2,4) is a ~0.57mm spur
        let out = clean_feature_edges(&ids, &pos, 14.14);
        assert_eq!(out.len(), 4, "the short spur should be pruned, the square loop kept");
    }

    #[test]
    fn cleanup_keeps_a_long_dangling_chain() {
        // A square plus a LONG chain dead-ending off a corner. A long dead-end is a real edge that
        // lost a neighbour at a non-manifold seam (not a stray), so it must be KEPT — deleting it is
        // what erased real cut edges.
        let pos = vec![
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
            [25.0, 25.0, 0.0], [40.0, 40.0, 0.0], // a long 2-segment chain off corner 2
        ];
        let ids = vec![(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5)];
        let out = clean_feature_edges(&ids, &pos, 56.6);
        assert_eq!(out.len(), 6, "a long dangling chain is a real edge and must be kept");
    }

    #[test]
    fn cleanup_closes_a_gapped_loop() {
        // A finely-faceted circle (a real curved rim) that lost ONE segment at a seam: an isolated
        // open path whose two ends are one facet apart. It must be closed back into a full loop.
        let n = 48usize;
        let r = 10.0f32;
        let pos: Vec<[f32; 3]> =
            (0..n).map(|k| { let a = std::f32::consts::TAU * k as f32 / n as f32; [r * a.cos(), r * a.sin(), 0.0] }).collect();
        let ids: Vec<(usize, usize)> = (0..n - 1).map(|k| (k, k + 1)).collect(); // missing (n-1, 0)
        let out = clean_feature_edges(&ids, &pos, 2.0 * r);
        assert_eq!(out.len(), n, "the one-facet gap should be closed back into the full loop");
    }

    #[test]
    fn cleanup_drops_a_long_floating_stray() {
        // An isolated open path whose ends are far apart isn't a loop — it's a stray streak. Drop it.
        let pos = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
        let ids = vec![(0, 1), (1, 2), (2, 3)];
        let out = clean_feature_edges(&ids, &pos, 30.0);
        assert!(out.is_empty(), "a long straight floating path is a stray, not a loop");
    }
}

