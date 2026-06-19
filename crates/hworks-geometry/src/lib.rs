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

mod csg;
mod fillet;
mod mesh_bool;
pub use fillet::round_mesh;
pub use mesh_bool::{mesh_difference, mesh_intersection, mesh_union};

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

/// Tolerance for boolean operations and tessellation.
const TOL: f64 = 0.05;

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

/// Like [`extrude_solid`] but the prism starts `back` units *behind* the plane,
/// so a boss built on a face overlaps the body it sits on — avoiding a coplanar
/// shared face that would make the following union fail.
pub fn extrude_solid_with_overlap(
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    basis: &PlaneBasis,
    distance: f64,
    back: f64,
) -> Option<KSolid> {
    build_solid(outer, holes, basis, -back, distance + back).map(KSolid)
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
    fn plane_at(z: f64) -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, z], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
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
