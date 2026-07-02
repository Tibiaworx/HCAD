//! Triangle-mesh booleans, backed by the **Manifold** library (the robust CSG engine
//! used by OpenSCAD and others). Manifold guarantees 2-manifold output and dissolves
//! coincident faces cleanly — exactly the cases truck's exact B-rep boolean rejects
//! (a boss flush on a cut floor, stacked same-footprint extrudes, …).
//!
//! truck's tessellation flat-shades (every triangle owns its three vertices), so we
//! **weld** coincident vertices before handing a mesh to Manifold — it needs shared
//! topology to know the surface is connected. If Manifold ever declines a boolean we
//! fall back to the self-contained BSP CSG so an operation never silently vanishes.

use crate::{csg, TriMesh};
use manifold3d::{Manifold, MeshGL};
use std::collections::HashMap;

/// Weld coincident vertices (truck flat-shades, so shared corners are duplicated) and
/// return Manifold-ready flat vertex properties `[x,y,z, …]` + triangle indices.
///
/// The tolerance is **bounding-box-relative** and the merge is **neighbour-checked**, both for one
/// reason: a full-turn revolve's seam. truck computes the 0 and 2π vertices from cos/sin, and the
/// 2π rotation error grows with distance from the axis — on a big part (major radius tens of mm)
/// the two "identical" seam verts can sit microns apart, far enough that a fixed grid leaves an
/// open seam → NotManifold → the lossy BSP CSG (torn surface, or an OOM on dense meshes). Scaling
/// the tolerance with the model and searching the 27 neighbour cells fuses the seam at any size
/// without merging genuinely distinct (mm-scale) geometry.
fn weld(m: &TriMesh) -> (Vec<f32>, Vec<u32>) {
    weld_tol(m, 3.0e-5)
}

/// `weld` with a caller-chosen bbox-relative tolerance factor (exposed for diagnostics).
fn weld_tol(m: &TriMesh, rel: f32) -> (Vec<f32>, Vec<u32>) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    // `rel` of the model size: comfortably exceeds the seam gap yet stays far below any real
    // feature; floored so tiny models still merge exact duplicates.
    let tol = (diag * rel).max(1.0e-5);
    let inv = 1.0 / tol;
    let cell = |c: f32| (c * inv).floor() as i64;
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    let mut props: Vec<f32> = Vec::new();
    let mut remap = vec![0u32; m.positions.len()];
    for (i, p) in m.positions.iter().enumerate() {
        let (cx, cy, cz) = (cell(p[0]), cell(p[1]), cell(p[2]));
        // Cell size == tol, so any vertex within tol lies in one of the 27 surrounding cells.
        let mut hit: Option<u32> = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ids) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &id in ids {
                            let b = id as usize * 3;
                            if (props[b] - p[0]).abs() < tol && (props[b + 1] - p[1]).abs() < tol && (props[b + 2] - p[2]).abs() < tol {
                                hit = Some(id);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }
        let id = hit.unwrap_or_else(|| {
            let id = (props.len() / 3) as u32;
            props.extend_from_slice(&[p[0], p[1], p[2]]);
            grid.entry((cx, cy, cz)).or_default().push(id);
            id
        });
        remap[i] = id;
    }
    let tris: Vec<u32> = m.indices.iter().map(|&i| remap[i as usize]).collect();
    (props, tris)
}

/// True if `m` can be ingested as a valid 2-manifold solid (welds coincident verts first).
/// A `false` here means a boolean with this operand will fall back to the lossy BSP CSG.
pub fn is_manifold(m: &TriMesh) -> bool {
    to_manifold(m).is_some()
}

/// Test-only: the Manifold difference WITHOUT the BSP fallback (so a failing case can be inspected
/// without the BSP CSG exploding). `None` ⇒ Manifold (and its retries) couldn't do it.
#[cfg(test)]
pub fn manifold_difference_only(a: &TriMesh, b: &TriMesh) -> Option<TriMesh> {
    manifold_boolean(a, b, Op::Difference)
}

/// Test-only edge topology after welding: (verts, tris, boundary edges [used once], non-manifold
/// edges [used >2×]). A watertight 2-manifold has every edge used exactly twice → both 0.
#[cfg(test)]
pub fn weld_edge_stats_tol(m: &TriMesh, rel: f32) -> (usize, usize, usize, usize) {
    let (props, tris) = weld_tol(m, rel);
    let mut edge: HashMap<(u32, u32), i32> = HashMap::new();
    for t in tris.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let k = if a < b { (a, b) } else { (b, a) };
            *edge.entry(k).or_insert(0) += 1;
        }
    }
    let boundary = edge.values().filter(|&&c| c == 1).count();
    let nonman = edge.values().filter(|&&c| c > 2).count();
    (props.len() / 3, tris.len() / 3, boundary, nonman)
}

#[cfg(test)]
pub fn weld_edge_stats(m: &TriMesh) -> (usize, usize, usize, usize) {
    let (props, tris) = weld(m);
    let mut edge: HashMap<(u32, u32), i32> = HashMap::new();
    for t in tris.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let k = if a < b { (a, b) } else { (b, a) };
            *edge.entry(k).or_insert(0) += 1;
        }
    }
    let boundary = edge.values().filter(|&&c| c == 1).count();
    let nonman = edge.values().filter(|&&c| c > 2).count();
    (props.len() / 3, tris.len() / 3, boundary, nonman)
}

/// Build a `Manifold` from a triangle mesh; `None` if empty or not a valid solid.
fn to_manifold(m: &TriMesh) -> Option<Manifold> {
    if m.indices.len() < 3 {
        return None;
    }
    let (props, tris) = weld(m);
    let meshgl = MeshGL::new(&props, 3, &tris).ok()?;
    Manifold::from_meshgl(&meshgl).ok()
}

/// Convert a `Manifold` back to a flat-shaded triangle mesh (per-face normals).
fn from_manifold(man: &Manifold) -> TriMesh {
    let mgl = man.to_meshgl();
    let nprop = mgl.num_prop().max(3);
    let vp = mgl.vert_properties();
    let tris = mgl.tri_verts();
    let pos = |v: u32| {
        let b = v as usize * nprop;
        [vp[b], vp[b + 1], vp[b + 2]]
    };
    let mut out = TriMesh::default();
    for t in tris.chunks_exact(3) {
        let (p0, p1, p2) = (pos(t[0]), pos(t[1]), pos(t[2]));
        let n = face_normal(p0, p1, p2);
        let base = out.positions.len() as u32;
        for p in [p0, p1, p2] {
            out.positions.push(p);
            out.normals.push(n);
        }
        out.indices.extend([base, base + 1, base + 2]);
    }
    out
}

/// **Face-boundary feature edges** — the FreeCAD-style detector. Ingest the mesh into Manifold, which
/// groups coplanar-connected triangles into faces (exact for flats; per-facet for curves). Merge
/// facet groups that meet tangentially (dihedral below `crease_deg`) into *smooth faces*, then the
/// real edges are exactly the boundaries between different smooth faces.
///
/// Why this beats per-edge dihedral thresholding:
/// - **No boolean-seam strays are possible** — re-tessellation inside a flat face shares that face's
///   id, so it never crosses a face boundary.
/// - **Flat-face edges are exact** — no threshold; a box edge is a face boundary, full stop.
/// - **Curve facets vanish** — the 48 facets of a cylinder wall merge into one smooth face, so no
///   starburst and no per-facet noise; the angle is used once per face-pair, not per triangle.
///
/// Returns `(sharp, tangent)` in world positions; `tangent` collects boundaries that are gentle
/// (between `tangent_deg` and `crease_deg`) so they can be shown optionally. `None` if the mesh can't
/// be ingested (caller falls back to the angle detector).
pub fn feature_edges_by_face(mesh: &TriMesh, crease_deg: f64, tangent_deg: f64) -> Option<(Vec<[[f32; 3]; 2]>, Vec<[[f32; 3]; 2]>)> {
    let man = to_manifold(mesh)?.as_original();
    let mgl = man.to_meshgl();
    let nprop = mgl.num_prop().max(3);
    let vp = mgl.vert_properties();
    let tris = mgl.tri_verts();
    let fid = mgl.face_id();
    if tris.len() < 3 || fid.len() * 3 != tris.len() {
        return None;
    }
    let ntri = tris.len() / 3;
    let pos = |v: u32| { let b = v as usize * nprop; [vp[b], vp[b + 1], vp[b + 2]] };
    let dot = |u: [f32; 3], w: [f32; 3]| u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    // Per-triangle normal + dense face-id.
    let tnorm: Vec<[f32; 3]> = (0..ntri).map(|t| face_normal(pos(tris[t * 3]), pos(tris[t * 3 + 1]), pos(tris[t * 3 + 2]))).collect();
    let mut fmap: HashMap<u32, usize> = HashMap::new();
    let face_of: Vec<usize> = (0..ntri).map(|t| { let n = fmap.len(); *fmap.entry(fid[t]).or_insert(n) }).collect();
    // Edge → the (≤2) triangles that share it. Manifold shares vertices, so an index pair is exact.
    let mut emap: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for t in 0..ntri {
        let vs = [tris[t * 3], tris[t * 3 + 1], tris[t * 3 + 2]];
        for k in 0..3 {
            let (i, j) = (vs[k], vs[(k + 1) % 3]);
            emap.entry(if i < j { (i, j) } else { (j, i) }).or_default().push(t);
        }
    }
    // Union-find: merge tangent-connected face groups into smooth faces.
    let mut uf: Vec<usize> = (0..fmap.len()).collect();
    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    let cos_crease = crease_deg.to_radians().cos() as f32;
    for ts in emap.values() {
        if ts.len() == 2 && face_of[ts[0]] != face_of[ts[1]] && dot(tnorm[ts[0]], tnorm[ts[1]]) > cos_crease {
            let (ra, rb) = (find(&mut uf, face_of[ts[0]]), find(&mut uf, face_of[ts[1]]));
            if ra != rb {
                uf[ra] = rb;
            }
        }
    }
    // Edges = boundaries between different smooth faces.
    let cos_tan = tangent_deg.to_radians().cos() as f32;
    let (mut sharp, mut tangent) = (Vec::new(), Vec::new());
    for ((i, j), ts) in &emap {
        let edge = [pos(*i), pos(*j)];
        let boundary = match ts.len() {
            2 => find(&mut uf, face_of[ts[0]]) != find(&mut uf, face_of[ts[1]]),
            1 => true, // a true boundary edge (open shell) — keep
            _ => false,
        };
        if !boundary {
            continue;
        }
        // Classify by the dihedral across the boundary: a hard crease vs a gentle (tangent) meeting.
        let hard = ts.len() != 2 || dot(tnorm[ts[0]], tnorm[ts[1]]) < cos_tan;
        if hard {
            sharp.push(edge);
        } else {
            tangent.push(edge);
        }
    }
    Some((sharp, tangent))
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 1e-12 {
        [n[0] / l, n[1] / l, n[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[derive(Clone, Copy)]
enum Op {
    Union,
    Difference,
    Intersection,
}

/// Count of booleans that fell back to the lossy BSP CSG this regen (Manifold rejected them).
/// The app reads + resets this after a rebuild to warn that a result is unreliable.
static BSP_FALLBACKS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Read and reset the BSP-fallback counter (number of booleans Manifold couldn't do).
pub fn take_fallback_count() -> u32 {
    BSP_FALLBACKS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// Shift every vertex by `d` — a sub-micron nudge to break exact coincident/tangent faces (e.g. a
/// revolve grazing a boss wall) that make Manifold's boolean fail.
fn nudged(m: &TriMesh, d: [f32; 3]) -> TriMesh {
    TriMesh {
        positions: m.positions.iter().map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]]).collect(),
        normals: m.normals.clone(),
        indices: m.indices.clone(),
    }
}

/// Run one Manifold boolean attempt; `None` if an operand won't ingest or the op errors.
fn manifold_try(a: &TriMesh, b: &TriMesh, op: Op) -> Option<TriMesh> {
    let (ma, mb) = (to_manifold(a)?, to_manifold(b)?);
    let r = match op {
        Op::Union => ma.union(&mb),
        Op::Difference => ma.difference(&mb),
        Op::Intersection => ma.intersection(&mb),
    };
    if r.status().is_err() {
        return None;
    }
    let mesh = from_manifold(&r);
    (!mesh.indices.is_empty()).then_some(mesh)
}

/// Manifold boolean with a tangency-breaking retry; `None` only if every attempt fails (then the
/// caller drops to the BSP CSG). The retries nudge `b` by a few sub-micron offsets — when the two
/// solids share a tangent/coincident band (a concentric revolve grazing the boss wall), the exact
/// coincidence is what trips Manifold up, and a tiny perturbation makes it resolve cleanly.
fn manifold_boolean(a: &TriMesh, b: &TriMesh, op: Op) -> Option<TriMesh> {
    if let Some(m) = manifold_try(a, b, op) {
        return Some(m);
    }
    // Asymmetric, irrational-ish nudges so no offset lands back on another coincidence.
    for d in [[1.7e-4, 1.1e-4, 1.3e-4], [-2.3e-4, 1.9e-4, -1.5e-4], [3.1e-4, -2.7e-4, 2.1e-4]] {
        if let Some(m) = manifold_try(a, &nudged(b, d), op) {
            return Some(m);
        }
    }
    None
}

/// Boolean **union** of two triangle meshes (Manifold; lossy BSP CSG fallback as last resort).
pub fn mesh_union(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Union).unwrap_or_else(|| {
        BSP_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        csg::bsp_union(a, b)
    })
}

/// Boolean **difference** `a − b` of two triangle meshes (Manifold; lossy BSP CSG last resort).
pub fn mesh_difference(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Difference).unwrap_or_else(|| {
        BSP_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        csg::bsp_difference(a, b)
    })
}

/// Boolean **intersection** `a ∩ b` of two triangle meshes (Manifold; empty on failure).
pub fn mesh_intersection(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Intersection).unwrap_or_default()
}

/// Reflect a mesh across the plane through `origin` with `normal`. Reflection reverses
/// orientation, so triangle winding is swapped (and normals reflected+negated) to keep the
/// surface outward-facing — ready to union with the original for a mirror.
pub fn mirror_mesh(mesh: &TriMesh, origin: [f64; 3], normal: [f64; 3]) -> TriMesh {
    let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if nl < 1e-12 {
        return mesh.clone();
    }
    let n = [normal[0] / nl, normal[1] / nl, normal[2] / nl];
    let o = origin;
    let reflect_pt = |p: &[f32; 3]| -> [f32; 3] {
        let d = [p[0] as f64 - o[0], p[1] as f64 - o[1], p[2] as f64 - o[2]];
        let dot = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
        [
            (p[0] as f64 - 2.0 * dot * n[0]) as f32,
            (p[1] as f64 - 2.0 * dot * n[1]) as f32,
            (p[2] as f64 - 2.0 * dot * n[2]) as f32,
        ]
    };
    let reflect_nrm = |m: &[f32; 3]| -> [f32; 3] {
        let dot = m[0] as f64 * n[0] + m[1] as f64 * n[1] + m[2] as f64 * n[2];
        // Reflected then negated (winding is also swapped) so it points back outward.
        [
            -((m[0] as f64 - 2.0 * dot * n[0]) as f32),
            -((m[1] as f64 - 2.0 * dot * n[1]) as f32),
            -((m[2] as f64 - 2.0 * dot * n[2]) as f32),
        ]
    };
    let mut out = TriMesh {
        positions: mesh.positions.iter().map(reflect_pt).collect(),
        normals: mesh.normals.iter().map(reflect_nrm).collect(),
        indices: Vec::with_capacity(mesh.indices.len()),
    };
    for t in mesh.indices.chunks_exact(3) {
        out.indices.extend([t[0], t[2], t[1]]); // swap winding to restore orientation
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{extrude_tool_mesh, PlaneBasis};

    fn xy() -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    fn volume(m: &TriMesh) -> f64 {
        let mut v = 0.0;
        for t in m.indices.chunks_exact(3) {
            let p: Vec<[f64; 3]> = t
                .iter()
                .map(|&i| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                })
                .collect();
            v += (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
                - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
                + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]))
                / 6.0;
        }
        v.abs()
    }

    #[test]
    fn manifold_union_of_two_truck_prisms() {
        // Two 4x4 square prisms (truck-extruded), one shifted 2 in x → overlap half.
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let a = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let sq2 = [[2.0, 0.0], [6.0, 0.0], [6.0, 4.0], [2.0, 4.0]];
        let b = extrude_tool_mesh(&sq2, &[], &xy(), 0.0, 4.0).unwrap();
        let u = mesh_union(&a, &b);
        // 4*4*4 + 4*4*4 - 2*4*4 (overlap) = 64 + 64 - 32 = 96.
        assert!((volume(&u) - 96.0).abs() < 0.5, "union volume was {}", volume(&u));
    }

    #[test]
    fn manifold_flush_boss_on_floor() {
        // A boss whose base is coincident with the body's top face (z=4) — the case the
        // exact kernel rejects. Manifold must union it cleanly.
        let big = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let body = extrude_tool_mesh(&big, &[], &xy(), 0.0, 4.0).unwrap();
        let small = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        // Boss dips 0.01 into the body (z 3.99..8) so it's a real union with a near-flush base.
        let boss = extrude_tool_mesh(&small, &[], &xy(), 3.99, 4.01).unwrap();
        let u = mesh_union(&body, &boss);
        let expect = 10.0 * 10.0 * 4.0 + 4.0 * 4.0 * 4.01 - 4.0 * 4.0 * 0.01;
        assert!((volume(&u) - expect).abs() < 1.0, "flush-boss volume was {} (want {expect})", volume(&u));
    }

    #[test]
    fn face_provenance_and_coplanar_grouping() {
        use std::collections::HashSet;
        // Manifold exposes per-face provenance the FreeCAD-style edge detector relies on. Two unioned
        // boxes: (run_original_id, face_id) must key their 12 faces uniquely. And a fresh single ingest
        // must group coplanar triangles into faces (box → 6, cylinder → caps + per-facet walls).
        let a = to_manifold(&extrude_tool_mesh(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], &[], &xy(), 0.0, 10.0).unwrap()).unwrap().as_original();
        let b = to_manifold(&extrude_tool_mesh(&[[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]], &[], &xy(), 5.0, 10.0).unwrap()).unwrap().as_original();
        let u = a.union(&b);
        let mgl = u.to_meshgl();
        let (nrun, ntri) = (mgl.num_run(), mgl.num_tri());
        let (roid, ridx, faceid) = (mgl.run_original_id(), mgl.run_index(), mgl.face_id());
        let mut tri_oid = vec![0u32; ntri];
        for i in 0..nrun {
            let s = ridx[i] as usize / 3;
            let e = if i + 1 < ridx.len() { ridx[i + 1] as usize / 3 } else { ntri };
            for t in s..e {
                tri_oid[t] = roid.get(i).copied().unwrap_or(0);
            }
        }
        let keys: HashSet<(u32, u32)> = (0..ntri).map(|t| (tri_oid[t], faceid[t])).collect();
        assert_eq!(keys.len(), 12, "two unioned boxes have 12 provenance-keyed faces");

        let boxm = to_manifold(&extrude_tool_mesh(&[[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]], &[], &xy(), 0.0, 8.0).unwrap()).unwrap().as_original();
        let bfaces: HashSet<u32> = boxm.to_meshgl().face_id().iter().copied().collect();
        assert_eq!(bfaces.len(), 6, "a fresh box ingests to 6 coplanar faces");

        let circle: Vec<[f64; 2]> = (0..48).map(|k| { let a = std::f64::consts::TAU * k as f64 / 48.0; [10.0 * a.cos(), 10.0 * a.sin()] }).collect();
        let cm = to_manifold(&extrude_tool_mesh(&circle, &[], &xy(), 0.0, 20.0).unwrap()).unwrap().as_original();
        let cfaces: HashSet<u32> = cm.to_meshgl().face_id().iter().copied().collect();
        assert_eq!(cfaces.len(), 50, "cylinder: 2 caps + 48 wall facets");
    }
}
