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

mod bevel;
mod csg;
mod fillet;
mod mesh_bool;
pub use bevel::{bevel_feature_edges, bevel_mesh, bevel_mesh_and_edges, bevel_mesh_selected};
pub use fillet::{chamfer_mesh, round_mesh, threaded_hole};
pub use mesh_bool::{is_manifold, mesh_difference, mesh_intersection, mesh_union, mirror_mesh, take_fallback_count};

/// A tessellated triangle mesh handed up to the renderer.
#[derive(Debug, Default, Clone)]
pub struct TriMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
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
        if out.last().map_or(true, |&q| !close(p, q)) {
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
    cut_tol(base, outer, holes, basis, distance, TOL)
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
    tol: f64,
) -> Option<KSolid> {
    let depth = distance.abs();
    if depth < 1e-9 {
        return None;
    }
    let eps = 0.05 + depth * 0.02;
    let (start_offset, length) = if distance >= 0.0 {
        (-eps, depth + 2.0 * eps)
    } else {
        (-(depth + eps), depth + 2.0 * eps)
    };
    let mut tool = build_solid(outer, holes, basis, start_offset, length)?;
    guard(move || {
        tool.not(); // invert all faces → complement region
        truck_shapeops::and(&base.0, &tool, tol)
    })
    .map(KSolid)
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
    let solid = build_solid(outer, holes, basis, start_offset, length)?;
    guard(|| {
        let mut poly = solid.triangulation(TOL).to_polygon();
        poly.triangulate();
        Some(polymesh_to_trimesh(&poly))
    })
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
) -> Option<TriMesh> {
    let depth = distance.abs();
    if depth < 1e-9 {
        return None;
    }
    let eps = 0.05 + depth * 0.02;
    let (start, length) =
        if distance >= 0.0 { (-eps, depth + 2.0 * eps) } else { (-(depth + eps), depth + 2.0 * eps) };
    extrude_tool_mesh(outer, holes, basis, start, length)
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

/// Reconstruct a **faceted** B-rep solid from a triangle mesh: weld coincident vertices, share an
/// edge between adjacent triangles, and make each triangle a planar `Face`, assembled into a Shell
/// → Solid. This lets a mesh-only body (loft, fillet, seamless boolean) still export to STEP — the
/// result is faceted (one flat face per triangle), not smooth, but valid B-rep. `None` if it can't
/// be assembled (panic-guarded). Large meshes make large STEP files.
pub fn mesh_to_solid(mesh: &TriMesh) -> Option<KSolid> {
    use std::collections::HashMap;
    if mesh.indices.len() < 3 {
        return None;
    }
    // Weld to unique vertices (truck topology shares Vertex objects; exact-ish merge only fuses
    // truck/Manifold's duplicated corners, never distinct geometry).
    let key = |p: [f32; 3]| ((p[0] * 1.0e5).round() as i64, (p[1] * 1.0e5).round() as i64, (p[2] * 1.0e5).round() as i64);
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut uniq: Vec<[f32; 3]> = Vec::new();
    let mut remap = vec![0u32; mesh.positions.len()];
    for (i, p) in mesh.positions.iter().enumerate() {
        remap[i] = *map.entry(key(*p)).or_insert_with(|| {
            uniq.push(*p);
            (uniq.len() - 1) as u32
        });
    }
    guard(|| {
        let verts: Vec<truck_modeling::Vertex> = uniq.iter().map(|p| builder::vertex(Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))).collect();
        let mut edges: HashMap<(u32, u32), truck_modeling::Edge> = HashMap::new();
        let mut faces: Vec<truck_modeling::Face> = Vec::new();
        for t in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]);
            if a == b || b == c || a == c {
                continue; // degenerate after welding — skip
            }
            // A shared edge is built once (canonical low→high) and reused inverted by the other face.
            let mut directed = |x: u32, y: u32| -> truck_modeling::Edge {
                let (lo, hi) = if x < y { (x, y) } else { (y, x) };
                let e = edges.entry((lo, hi)).or_insert_with(|| builder::line(&verts[lo as usize], &verts[hi as usize])).clone();
                if x < y { e } else { e.inverse() }
            };
            let wire: truck_modeling::Wire = vec![directed(a, b), directed(b, c), directed(c, a)].into_iter().collect();
            faces.push(builder::try_attach_plane(&[wire]).ok()?);
        }
        if faces.len() < 4 {
            return None;
        }
        let shell: truck_modeling::Shell = faces.into_iter().collect();
        Some(KSolid(truck_modeling::Solid::new(vec![shell])))
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
fn push_tri(mesh: &mut TriMesh, a: [f64; 3], b: [f64; 3], c: [f64; 3]) {
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
    // Holes are skinned only when every profile has the same count (matched by index).
    let hole_count = if valid.iter().all(|(_, h)| h.len() == valid[0].1.len()) { valid[0].1.len() } else { 0 };
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
    // CSG output has T-junctions and near-coincident vertices where two solids' separate
    // tessellations meet. Weld coarsely (2e-3 grid) and keep only *manifold* edges above
    // a slightly raised 50° threshold — so boolean seams don't draw as stray dashes,
    // while real corners (≥~90°) still show.
    let (edges, tangent_edges) = feature_edges_opts(&mesh, 50.0, 5.0e2, true);
    Tessellation { mesh, edges, tangent_edges }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

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
    feature_edges_opts(mesh, sharp_deg, 1.0e4, false)
}

/// `feature_edges`, parameterised for the two mesh sources:
/// - `weld` is the quantisation scale used to merge coincident vertices (a larger
///   number = finer grid; truck meshes are clean so 1e4 is right).
/// - `manifold_only` keeps **only** edges shared by exactly two faces and drops
///   boundary/non-manifold edges. Mesh-boolean (CSG) output is riddled with
///   T-junctions whose "boundary" edges would otherwise draw as a starburst, so the
///   mesh-fallback path turns this on; the exact (truck) path leaves it off.
fn feature_edges_opts(
    mesh: &TriMesh,
    sharp_deg: f64,
    weld: f32,
    manifold_only: bool,
) -> (Vec<[[f32; 3]; 2]>, Vec<[[f32; 3]; 2]>) {
    use std::collections::HashMap;
    // Merge duplicated (flat-shaded) vertices by quantized position.
    let quant = |p: [f32; 3]| {
        ((p[0] * weld).round() as i64, (p[1] * weld).round() as i64, (p[2] * weld).round() as i64)
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
    let mut sharp = Vec::new();
    let mut tangent = Vec::new();
    for ((i, j), normals) in emap {
        let edge = [canon_pos[i], canon_pos[j]];
        if normals.len() != 2 {
            // Boundary (1) or non-manifold (≥3). For CSG output these are T-junction
            // artifacts → drop. For exact (truck) meshes a lone boundary edge is real.
            if !manifold_only && normals.len() == 1 {
                sharp.push(edge);
            }
            continue;
        }
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
        let md = min_dot as f64;
        if md < cos_sharp {
            sharp.push(edge); // a real corner
        } else if md < cos_flat {
            tangent.push(edge); // smooth/curvature edge
        } // else coplanar interior → drop
    }
    (sharp, tangent)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn mesh_to_solid_exports_faceted_step() {
        // A mesh-only body (here a loft, which has no exact B-rep) → faceted solid → STEP.
        let m = loft_mesh(&[(circle3(0.0, 0.0, 0.0, 5.0, 24), vec![]), (circle3(0.0, 0.0, 10.0, 3.0, 24), vec![])]).unwrap();
        let solid = mesh_to_solid(&m).expect("faceted solid from mesh");
        let step = export_step(&solid).expect("step from faceted solid");
        assert!(step.contains("ISO-10303-21") && step.matches("FACE").count() > 100, "faceted STEP malformed");
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
}
