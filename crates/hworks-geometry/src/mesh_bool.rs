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
    // Drop triangles the weld made degenerate (two corners merged): Manifold rejects a
    // MeshGL containing them, so a dense mesh with edges shorter than the weld tolerance
    // (e.g. marching-cubes output slivers) would spuriously read as "not manifold".
    let mut tris: Vec<u32> = Vec::with_capacity(m.indices.len());
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]);
        if a != b && b != c && a != c {
            tris.extend([a, b, c]);
        }
    }
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

/// Above this combined triangle count, the O(n²)-ish BSP CSG fallback is a multi-minute
/// grind (or an OOM) that looks like a hang — so when Manifold declines a dense operand we
/// SKIP the BSP entirely and count it (see [`take_dense_skip_count`]). The app turns that
/// count into "this mesh isn't a closed solid — Solidify it to cut" rather than freezing.
const BSP_MAX_TRIS: usize = 20_000;

/// Count of booleans SKIPPED because Manifold declined a mesh too dense for the BSP fallback.
static DENSE_SKIPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Read and reset the dense-skip counter (booleans skipped to avoid a BSP hang on a big
/// non-manifold mesh). The app warns the user and points them at Solidify.
pub fn take_dense_skip_count() -> u32 {
    DENSE_SKIPS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

fn too_dense_for_bsp(a: &TriMesh, b: &TriMesh) -> bool {
    (a.indices.len() + b.indices.len()) / 3 > BSP_MAX_TRIS
}

/// Boolean **union** of two triangle meshes (Manifold; lossy BSP CSG fallback as last resort).
pub fn mesh_union(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Union).unwrap_or_else(|| {
        if too_dense_for_bsp(a, b) {
            DENSE_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return a.clone(); // don't grind BSP on a dense mesh — keep the base, warn upstream
        }
        BSP_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        csg::bsp_union(a, b)
    })
}

/// Boolean **difference** `a − b` of two triangle meshes (Manifold; lossy BSP CSG last resort).
pub fn mesh_difference(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Difference).unwrap_or_else(|| {
        if too_dense_for_bsp(a, b) {
            DENSE_SKIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return a.clone(); // cut can't apply to a non-solid dense mesh — keep the base uncut
        }
        BSP_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        csg::bsp_difference(a, b)
    })
}

/// Boolean **intersection** `a ∩ b` of two triangle meshes (Manifold; empty on failure).
pub fn mesh_intersection(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Intersection).unwrap_or_default()
}

/// Triangle vs axis-aligned voxel box overlap (Akenine-Möller separating-axis test). Box is
/// centred at `c` with half-extent `r` on each axis. Used for CONSERVATIVE voxel rasterization:
/// every voxel a triangle actually passes through is marked (no sampling gaps), so the shell is
/// watertight without a dilation that would bridge real concavities.
/// Squared distance from point `p` to triangle `abc` (Ericson, *Real-Time Collision
/// Detection*): closest point via barycentric region tests. Used to sharpen the voxel SDF
/// with EXACT surface distances near the wall, so the remeshed surface lands on the true
/// scan surface instead of quantized voxel centers (which read as stair-steps).
fn point_tri_dist2(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let cl = point_tri_closest(p, a, b, c);
    let d = [p[0] - cl[0], p[1] - cl[1], p[2] - cl[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}

/// Closest point on triangle `abc` to `p` (Ericson, *Real-Time Collision Detection*).
fn point_tri_closest(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let sub = |x: [f32; 3], y: [f32; 3]| [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    let dot = |x: [f32; 3], y: [f32; 3]| x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    
    if d1 <= 0.0 && d2 <= 0.0 {
        a // vertex A
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            b // vertex B
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let v = d1 / (d1 - d3);
                [a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v] // edge AB
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    c // vertex C
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        let w = d2 / (d2 - d6);
                        [a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w] // edge AC
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            [b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w, b[2] + (c[2] - b[2]) * w] // edge BC
                        } else {
                            // interior: project onto the triangle plane
                            let denom = 1.0 / (va + vb + vc);
                            let v = vb * denom;
                            let w = vc * denom;
                            [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
                        }
                    }
                }
            }
        }
    }
}

fn tri_box_overlap(c: [f32; 3], r: f32, v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> bool {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (p0, p1, p2) = (sub(v0, c), sub(v1, c), sub(v2, c));
    // 1) AABB of the triangle vs the box (3 axes).
    for k in 0..3 {
        let (mn, mx) = (p0[k].min(p1[k]).min(p2[k]), p0[k].max(p1[k]).max(p2[k]));
        if mn > r || mx < -r {
            return false;
        }
    }
    let e = [sub(p1, p0), sub(p2, p1), sub(p0, p2)];
    // 2) Triangle-normal plane vs box.
    let nrm = [
        e[0][1] * e[1][2] - e[0][2] * e[1][1],
        e[0][2] * e[1][0] - e[0][0] * e[1][2],
        e[0][0] * e[1][1] - e[0][1] * e[1][0],
    ];
    let d = nrm[0] * p0[0] + nrm[1] * p0[1] + nrm[2] * p0[2];
    let rad = r * (nrm[0].abs() + nrm[1].abs() + nrm[2].abs());
    if d.abs() > rad {
        return false;
    }
    // 3) Nine edge × axis cross-product separating axes.
    let verts = [p0, p1, p2];
    for ei in &e {
        for axis in 0..3 {
            // axis unit vector cross edge → the test axis.
            let a = match axis {
                0 => [0.0, -ei[2], ei[1]],
                1 => [ei[2], 0.0, -ei[0]],
                _ => [-ei[1], ei[0], 0.0],
            };
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for v in &verts {
                let p = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
                mn = mn.min(p);
                mx = mx.max(p);
            }
            let rr = r * (a[0].abs() + a[1].abs() + a[2].abs());
            if mn > rr || mx < -rr {
                return false;
            }
        }
    }
    true
}

/// Voxel-remesh a triangle mesh into a **watertight, 2-manifold solid** so a scan (or any
/// non-manifold soup) can be cut. `res` is the voxel count on the longest bounding-box axis
/// (≈ 64 fast/coarse … 256 slow/fine). Pipeline: rasterize the surface into an occupancy grid
/// → morphological close (seal small holes) → flood-fill outside → fill the interior → signed
/// distance transform → hand the SDF to Manifold's level-set surfacer (guaranteed-manifold
/// output). Lossy (resolution-limited, softens sharp detail), which is the right trade for
/// making a scan solid. `None` on a degenerate/empty mesh.
pub fn remesh_solid(m: &TriMesh, res: usize) -> Option<TriMesh> {
    if m.indices.len() < 3 || m.positions.is_empty() {
        return None;
    }
    let res = res.clamp(16, 400);
    // Padded bounding box (2-voxel margin so the surface never touches the grid border —
    // the flood fill needs a guaranteed-outside shell of empty voxels).
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let longest = extent[0].max(extent[1]).max(extent[2]);
    if !(longest.is_finite() && longest > 0.0) {
        return None;
    }
    let h = longest / res as f32; // voxel size
    let pad = 3;
    let origin = [lo[0] - pad as f32 * h, lo[1] - pad as f32 * h, lo[2] - pad as f32 * h];
    let dim = |e: f32| ((e / h).ceil() as usize) + 2 * pad + 1;
    let (nx, ny, nz) = (dim(extent[0]), dim(extent[1]), dim(extent[2]));
    let n = nx * ny * nz;
    // Guard against a pathological grid (a very thin, huge part at high res).
    if n > 64_000_000 {
        return None;
    }
    let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
    let vox = |p: [f32; 3]| {
        [
            ((p[0] - origin[0]) / h) as i64,
            ((p[1] - origin[1]) / h) as i64,
            ((p[2] - origin[2]) / h) as i64,
        ]
    };

    // 1) CONSERVATIVE rasterization: mark every voxel each triangle actually passes through
    // (triangle-box overlap), so the shell is watertight with NO sampling gaps — and thus no
    // need for a dilation, which would bridge the model's real concavities into solid.
    let mut wall = vec![false; n];
    // Which triangles pass through each wall voxel — feeds the exact-distance band below,
    // which is what makes the output surface smooth instead of voxel-stepped.
    let mut wall_tris: HashMap<usize, Vec<u32>> = HashMap::new();
    let half = h * 0.5;
    for (ti, t) in m.indices.chunks_exact(3).enumerate() {
        let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
        let (mut tlo, mut thi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in [a, b, c] {
            for k in 0..3 {
                tlo[k] = tlo[k].min(p[k]);
                thi[k] = thi[k].max(p[k]);
            }
        }
        let lo_v = vox(tlo);
        let hi_v = vox(thi);
        for z in lo_v[2].max(0)..=hi_v[2].min(nz as i64 - 1) {
            for y in lo_v[1].max(0)..=hi_v[1].min(ny as i64 - 1) {
                for x in lo_v[0].max(0)..=hi_v[0].min(nx as i64 - 1) {
                    let center = [
                        origin[0] + (x as f32 + 0.5) * h,
                        origin[1] + (y as f32 + 0.5) * h,
                        origin[2] + (z as f32 + 0.5) * h,
                    ];
                    if tri_box_overlap(center, half, a, b, c) {
                        let i = at(x as usize, y as usize, z as usize);
                        wall[i] = true;
                        wall_tris.entry(i).or_default().push(ti as u32);
                    }
                }
            }
        }
    }
    // Seal genuine mesh holes (missing surface) without bridging concavities: a morphological
    // CLOSE (dilate then erode by the same radius) fills gaps up to ~2·seal voxels and restores
    // the surface elsewhere. `seal = 1` closes the common 1–2 voxel scan cracks; a conservative
    // shell rarely needs more. (Concavities wider than 2 voxels are preserved by the erode.)
    let seal = 1usize;
    let grow = |src: &[bool], want: bool| {
        // one-voxel morphological step: `want=true` dilate, `want=false` erode.
        let mut out = src.to_vec();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let i = at(x, y, z);
                    if src[i] == want {
                        continue;
                    }
                    let nb = |x: usize, y: usize, z: usize| src[at(x, y, z)] == want;
                    let touches = (x > 0 && nb(x - 1, y, z))
                        || (x + 1 < nx && nb(x + 1, y, z))
                        || (y > 0 && nb(x, y - 1, z))
                        || (y + 1 < ny && nb(x, y + 1, z))
                        || (z > 0 && nb(x, y, z - 1))
                        || (z + 1 < nz && nb(x, y, z + 1))
                        // grid border counts as "empty" for erosion so the solid never
                        // touches the padded bounds.
                        || (!want && (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz));
                    if touches {
                        out[i] = want;
                    }
                }
            }
        }
        out
    };
    let mut wall_c = wall.clone();
    for _ in 0..seal {
        wall_c = grow(&wall_c, true);
    }
    for _ in 0..seal {
        wall_c = grow(&wall_c, false);
    }
    // Re-add the original wall so erosion never re-opens a thin true surface.
    for i in 0..n {
        if wall[i] {
            wall_c[i] = true;
        }
    }

    // 2) Flood-fill "outside" from the (guaranteed-empty) corner through non-wall voxels.
    let mut outside = vec![false; n];
    let mut stack = vec![at(0, 0, 0)];
    outside[at(0, 0, 0)] = true;
    while let Some(idx) = stack.pop() {
        let z = idx / (nx * ny);
        let y = (idx % (nx * ny)) / nx;
        let x = idx % nx;
        let push = |x: usize, y: usize, z: usize, stack: &mut Vec<usize>, outside: &mut Vec<bool>| {
            let i = at(x, y, z);
            if !outside[i] && !wall_c[i] {
                outside[i] = true;
                stack.push(i);
            }
        };
        if x > 0 { push(x - 1, y, z, &mut stack, &mut outside); }
        if x + 1 < nx { push(x + 1, y, z, &mut stack, &mut outside); }
        if y > 0 { push(x, y - 1, z, &mut stack, &mut outside); }
        if y + 1 < ny { push(x, y + 1, z, &mut stack, &mut outside); }
        if z > 0 { push(x, y, z - 1, &mut stack, &mut outside); }
        if z + 1 < nz { push(x, y, z + 1, &mut stack, &mut outside); }
    }

    // 3) Solid = everything the outside flood couldn't reach (wall + enclosed interior).
    let solid: Vec<bool> = (0..n).map(|i| !outside[i]).collect();
    if !solid.iter().any(|&s| s) {
        return None; // nothing enclosed — likely a hole bigger than the close could seal
    }

    // 5) Signed distance field: unsigned chamfer distance to the solid/empty boundary, signed
    // negative inside. Multi-source BFS from every boundary voxel (distance in voxel units).
    let mut dist = vec![u32::MAX; n];
    let mut q = std::collections::VecDeque::new();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = at(x, y, z);
                let s = solid[i];
                let boundary = (x > 0 && solid[at(x - 1, y, z)] != s)
                    || (x + 1 < nx && solid[at(x + 1, y, z)] != s)
                    || (y > 0 && solid[at(x, y - 1, z)] != s)
                    || (y + 1 < ny && solid[at(x, y + 1, z)] != s)
                    || (z > 0 && solid[at(x, y, z - 1)] != s)
                    || (z + 1 < nz && solid[at(x, y, z + 1)] != s);
                if boundary {
                    dist[i] = 0;
                    q.push_back(i);
                }
            }
        }
    }
    while let Some(idx) = q.pop_front() {
        let d = dist[idx] + 1;
        let z = idx / (nx * ny);
        let y = (idx % (nx * ny)) / nx;
        let x = idx % nx;
        let relax = |x: usize, y: usize, z: usize, dist: &mut Vec<u32>, q: &mut std::collections::VecDeque<usize>| {
            let i = at(x, y, z);
            if d < dist[i] {
                dist[i] = d;
                q.push_back(i);
            }
        };
        if x > 0 { relax(x - 1, y, z, &mut dist, &mut q); }
        if x + 1 < nx { relax(x + 1, y, z, &mut dist, &mut q); }
        if y > 0 { relax(x, y - 1, z, &mut dist, &mut q); }
        if y + 1 < ny { relax(x, y + 1, z, &mut dist, &mut q); }
        if z > 0 { relax(x, y, z - 1, &mut dist, &mut q); }
        if z + 1 < nz { relax(x, y, z + 1, &mut dist, &mut q); }
    }
    // Signed field in world units. Manifold's `from_sdf` takes the region where the value is
    // POSITIVE as the solid, so inside is positive here, outside negative, ~0 at the surface.
    let mut sdf: Vec<f32> = (0..n)
        .map(|i| {
            let d = if dist[i] == u32::MAX { res as f32 } else { dist[i] as f32 };
            (if solid[i] { d } else { -d }) * h
        })
        .collect();

    // Sharpen the band: near the wall, replace the voxel-quantized BFS distance with the
    // EXACT distance to the real triangle surface. Without this, the isosurface snaps to
    // voxel centers and the whole model reads as stair-steps.
    //
    // The near-surface SIGN can't come from the voxel classification (a wall voxel whose
    // centre sits just outside the true surface is classified "solid" — half-voxel sign
    // errors that re-terrace the surface). It comes from the closest triangle's geometric
    // normal instead — with a global MAJORITY VOTE against the flood-fill classification, so
    // a mesh with inverted (or locally inconsistent) winding can't flip the model inside out:
    // if most band voxels' normal verdicts disagree with the robust flood fill, flip them all.
    const BAND: i64 = 2;
    // Parallel over z-slices (read-only inputs, per-thread output) — this is the hottest
    // loop of the remesh and scales linearly with cores.
    let nthreads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(16);
    let chunk = nz.div_ceil(nthreads.max(1)).max(1);
    // (voxel, exact dist, normal verdict: +1 out / -1 in / 0 unknown)
    let band: Vec<(usize, f32, i8)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for tid in 0..nthreads {
            let (z0, z1) = (tid * chunk, ((tid + 1) * chunk).min(nz));
            if z0 >= z1 {
                continue;
            }
            let (wall, wall_tris, dist) = (&wall, &wall_tris, &dist);
            handles.push(scope.spawn(move || {
                let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
                let mut out: Vec<(usize, f32, i8)> = Vec::new();
                for z in z0..z1 {
                    for y in 0..ny {
                        for x in 0..nx {
                            let i = at(x, y, z);
                            if dist[i] > BAND as u32 {
                                continue; // outside the band — coarse BFS distance is fine there
                            }
                            let center = [
                                origin[0] + (x as f32 + 0.5) * h,
                                origin[1] + (y as f32 + 0.5) * h,
                                origin[2] + (z as f32 + 0.5) * h,
                            ];
                            let mut best = f32::INFINITY;
                            let mut best_tri: Option<usize> = None;
                            for dz in -BAND..=BAND {
                                for dy in -BAND..=BAND {
                                    for dx in -BAND..=BAND {
                                        let (qx, qy, qz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                                        if qx < 0 || qy < 0 || qz < 0 || qx >= nx as i64 || qy >= ny as i64 || qz >= nz as i64 {
                                            continue;
                                        }
                                        let qi = at(qx as usize, qy as usize, qz as usize);
                                        if !wall[qi] {
                                            continue;
                                        }
                                        if let Some(tris) = wall_tris.get(&qi) {
                                            for &ti in tris {
                                                let t = &m.indices[ti as usize * 3..ti as usize * 3 + 3];
                                                let d2 = point_tri_dist2(
                                                    center,
                                                    m.positions[t[0] as usize],
                                                    m.positions[t[1] as usize],
                                                    m.positions[t[2] as usize],
                                                );
                                                if d2 < best {
                                                    best = d2;
                                                    best_tri = Some(ti as usize);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if let (true, Some(ti)) = (best.is_finite(), best_tri) {
                                let t = &m.indices[ti * 3..ti * 3 + 3];
                                let (a, b, c) = (m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
                                let cl = point_tri_closest(center, a, b, c);
                                let to_p = [center[0] - cl[0], center[1] - cl[1], center[2] - cl[2]];
                                let nrm = [
                                    (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                                    (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
                                ];
                                let s = to_p[0] * nrm[0] + to_p[1] * nrm[1] + to_p[2] * nrm[2];
                                let nl2 = nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2];
                                // Ambiguous when the point sits (nearly) on the triangle plane/edge
                                // or the triangle is degenerate — fall back to the classification.
                                let verdict = if nl2 < 1e-20 || s.abs() < 1e-12 { 0 } else if s > 0.0 { 1 } else { -1 };
                                out.push((i, best.sqrt(), verdict));
                            }
                        }
                    }
                }
                out
            }));
        }
        handles.into_iter().flat_map(|hnd| hnd.join().unwrap()).collect()
    });
    // Majority vote: does "normal says outside" line up with "flood fill says outside"?
    let (mut agree, mut disagree) = (0usize, 0usize);
    for &(i, _, v) in &band {
        if v == 0 {
            continue;
        }
        if (v > 0) != solid[i] {
            agree += 1;
        } else {
            disagree += 1;
        }
    }
    let flip = disagree > agree; // consistently inverted winding — trust the flood fill's orientation
    for (i, exact, v) in band {
        let inside = match v {
            0 => solid[i],
            _ => (v < 0) != flip,
        };
        sdf[i] = if inside { exact } else { -exact };
    }

    // 6) Surface the level set with Manifold (guaranteed 2-manifold output). The closure
    // trilinearly samples the SDF grid; outside the grid it returns a large positive value.
    let sample = move |x: f64, y: f64, z: f64| -> f64 {
        // SDF values live at voxel CENTERS (origin + (i+0.5)·h) — offset by half a voxel so
        // the trilinear interpolation is anchored correctly (without this the whole surface
        // shifts by h/2 per axis).
        let gx = (x as f32 - origin[0]) / h - 0.5;
        let gy = (y as f32 - origin[1]) / h - 0.5;
        let gz = (z as f32 - origin[2]) / h - 0.5;
        if gx < 0.0 || gy < 0.0 || gz < 0.0 || gx >= (nx - 1) as f32 || gy >= (ny - 1) as f32 || gz >= (nz - 1) as f32 {
            return -(longest as f64); // safely outside (negative, matching the inside-positive convention)
        }
        let (x0, y0, z0) = (gx as usize, gy as usize, gz as usize);
        let (fx, fy, fz) = (gx - x0 as f32, gy - y0 as f32, gz - z0 as f32);
        let s = |x: usize, y: usize, z: usize| sdf[at(x, y, z)];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(s(x0, y0, z0), s(x0 + 1, y0, z0), fx);
        let c10 = lerp(s(x0, y0 + 1, z0), s(x0 + 1, y0 + 1, z0), fx);
        let c01 = lerp(s(x0, y0, z0 + 1), s(x0 + 1, y0, z0 + 1), fx);
        let c11 = lerp(s(x0, y0 + 1, z0 + 1), s(x0 + 1, y0 + 1, z0 + 1), fx);
        let c0 = lerp(c00, c10, fy);
        let c1 = lerp(c01, c11, fy);
        lerp(c0, c1, fz) as f64
    };
    let bounds = (
        [origin[0] as f64, origin[1] as f64, origin[2] as f64],
        [
            (origin[0] + nx as f32 * h) as f64,
            (origin[1] + ny as f32 * h) as f64,
            (origin[2] + nz as f32 * h) as f64,
        ],
    );
    // Tight tolerance: with the exact-distance band the field is sub-voxel accurate near the
    // surface, so let the surfacer place vertices precisely rather than to a coarse budget.
    let man = Manifold::from_sdf(sample, bounds, h as f64, 0.0, (h * 0.05) as f64);
    if man.status().is_err() {
        return None;
    }
    // Simplify within a small fraction of the voxel size: marching emits sliver edges far
    // shorter than the boolean pipeline's weld tolerance, which would collapse them into
    // degenerate/boundary edges on re-ingestion ("not manifold" downstream). Collapsing them
    // here keeps the surface (deviation ≤ h/10) and trims the triangle count substantially.
    let man = man.simplify((h * 0.1) as f64);
    if man.status().is_err() {
        return None;
    }
    let out = from_manifold(&man);
    (!out.indices.is_empty()).then_some(out)
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
