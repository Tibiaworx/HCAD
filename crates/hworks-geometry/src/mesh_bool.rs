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
fn weld(m: &TriMesh) -> (Vec<f32>, Vec<u32>, Vec<usize>) {
    weld_tol(m, 3.0e-5)
}

/// `weld` with a caller-chosen bbox-relative tolerance factor (exposed for diagnostics).
fn weld_tol(m: &TriMesh, rel: f32) -> (Vec<f32>, Vec<u32>, Vec<usize>) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    // ...but never coarser than the mesh's own detail. A fillet's facets are far smaller than a
    // thousandth of the part it sits on, and welding across one collapses it: the two ends of a
    // real edge become one vertex and the triangles that shared it tear open. That is how a
    // perfectly built fillet — watertight, consistently wound, nothing degenerate — arrived at the
    // kernel as a torn surface and came back rejected, on nothing more than how many facets it
    // happened to have. Edges already below the floor are degenerate, and merging those is the
    // point, so they don't hold the tolerance down.
    let mut shortest = f32::INFINITY;
    for t in m.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (p, q) = (m.positions[a as usize], m.positions[b as usize]);
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            if d > 1.0e-5 {
                shortest = shortest.min(d);
            }
        }
    }
    // `rel` of the model size: comfortably exceeds the seam gap yet stays far below any real
    // feature; floored so tiny models still merge exact duplicates.
    let tol = (diag * rel).min(shortest * 0.3).max(1.0e-5);
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
    //
    // Dropping them breaks the 1:1 correspondence with the source, so `from` records which
    // source triangle each survivor came from. A caller that carries per-triangle data — the
    // surface tags — has no other way to keep it attached to the right triangle.
    let mut tris: Vec<u32> = Vec::with_capacity(m.indices.len());
    let mut from: Vec<usize> = Vec::with_capacity(m.indices.len() / 3);
    for (t, v) in m.indices.chunks_exact(3).enumerate() {
        let (a, b, c) = (remap[v[0] as usize], remap[v[1] as usize], remap[v[2] as usize]);
        if a != b && b != c && a != c {
            tris.extend([a, b, c]);
            from.push(t);
        }
    }
    (props, tris, from)
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
    let (props, tris, _) = weld_tol(m, rel);
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
    let (props, tris, _) = weld(m);
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
    let (props, tris, _) = weld(m);
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
    let pos = |v: u32| {
        let b = v as usize * nprop;
        [vp[b], vp[b + 1], vp[b + 2]]
    };
    let dot = |u: [f32; 3], w: [f32; 3]| u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    // WELD vertices before building adjacency. Manifold shares indices within one clean
    // solid, but a flush boolean leaves DUPLICATE vertices along its seam — edges there never
    // pair up by raw index, every unpaired edge read as an open boundary, and the seam was
    // drawn as a sharp ring around the part (sliver.hcad: a bottom ring boss snapped to the
    // tube's own radii painted a 202-segment ring at its top). Welded, those edges pair,
    // their dihedral is zero, and the ring merges away like any coplanar continuation.
    let nvert = vp.len() / nprop;
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for v in 0..nvert {
        let p = pos(v as u32);
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = (((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)) as f64).sqrt();
    let cell = (diag * 1e-6).max(1e-9);
    let mut wmap: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut wid: Vec<u32> = Vec::with_capacity(nvert);
    for v in 0..nvert {
        let p = pos(v as u32);
        let key = (
            (p[0] as f64 / cell).round() as i64,
            (p[1] as f64 / cell).round() as i64,
            (p[2] as f64 / cell).round() as i64,
        );
        let n = wmap.len() as u32;
        wid.push(*wmap.entry(key).or_insert(n));
    }
    // Per-triangle normal, plus a NEEDLE flag: a hair-thin triangle's normal is numerical
    // noise, so a needle neither blocks a merge nor gets its edges drawn.
    let needle_floor = (diag * 1e-5) as f32;
    let mut tnorm: Vec<[f32; 3]> = Vec::with_capacity(ntri);
    let mut is_needle: Vec<bool> = Vec::with_capacity(ntri);
    for t in 0..ntri {
        let (a, b, c) = (pos(tris[t * 3]), pos(tris[t * 3 + 1]), pos(tris[t * 3 + 2]));
        tnorm.push(face_normal(a, b, c));
        let e = |x: [f32; 3], y: [f32; 3]| {
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        };
        let (ab, bc, ca) = (e(a, b), e(b, c), e(c, a));
        let longest = ab.max(bc).max(ca);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        let area2 = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        is_needle.push(longest < 1e-12 || area2 / longest < needle_floor);
    }
    // Dense face ids and welded-edge adjacency.
    let mut fmap: HashMap<u32, usize> = HashMap::new();
    let face_of: Vec<usize> = (0..ntri)
        .map(|t| {
            let n = fmap.len();
            *fmap.entry(fid[t]).or_insert(n)
        })
        .collect();
    let mut emap: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for t in 0..ntri {
        let vs = [wid[tris[t * 3] as usize], wid[tris[t * 3 + 1] as usize], wid[tris[t * 3 + 2] as usize]];
        for k in 0..3 {
            let (i, j) = (vs[k], vs[(k + 1) % 3]);
            if i == j {
                continue; // an edge the weld collapsed
            }
            emap.entry(if i < j { (i, j) } else { (j, i) }).or_default().push(t);
        }
    }
    // Union-find: merge everything smoother than the crease into smooth faces, recording each
    // distinct group-pair's dihedral — those are the face's curvature STEPS, and the step
    // statistics are what tells a tangent SEAM from ordinary tessellation later.
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
        if ts.len() != 2 {
            continue;
        }
        let (t0, t1) = (ts[0], ts[1]);
        let (g0, g1) = (face_of[t0], face_of[t1]);
        if g0 == g1 {
            continue;
        }
        let d = if is_needle[t0] || is_needle[t1] { 1.0 } else { dot(tnorm[t0], tnorm[t1]) };
        if d > cos_crease {
            let (ra, rb) = (find(&mut uf, g0), find(&mut uf, g1));
            if ra != rb {
                uf[ra] = rb;
            }
        }
    }
    // Per-triangle neighbour steps, for the DIRECTIONAL rate test below: each triangle's
    // dihedral to each edge-neighbour, with that shared edge's direction. Group-level rate
    // statistics were tried and fail on twisted quads — a curved band's quad halves are each
    // their own coplanar group, so the halves' diagonals masquerade as boundaries and drown
    // every honest signal (six full rings of false seams on one filleted cylinder).
    const COPLANAR_DEG: f64 = 0.25;
    let cos_coplanar = COPLANAR_DEG.to_radians().cos() as f32;
    let mut rep: Vec<u32> = vec![u32::MAX; wmap.len()];
    for (raw, &w) in wid.iter().enumerate() {
        if rep[w as usize] == u32::MAX {
            rep[w as usize] = raw as u32;
        }
    }
    let wpos = |w: u32| pos(rep[w as usize]);
    let mut nbrs: Vec<Vec<(f32, [f32; 3])>> = vec![Vec::new(); ntri];
    let mut tri_adj: Vec<Vec<usize>> = vec![Vec::new(); ntri];
    for ((i, j), ts) in &emap {
        if ts.len() != 2 {
            continue;
        }
        let (t0, t1) = (ts[0], ts[1]);
        if is_needle[t0] || is_needle[t1] {
            continue;
        }
        let pa = wpos(*i);
        let pb = wpos(*j);
        let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if l < 1e-12 {
            continue;
        }
        let dir = [d[0] / l, d[1] / l, d[2] / l];
        let ang = (dot(tnorm[t0], tnorm[t1]) as f64).clamp(-1.0, 1.0).acos().to_degrees() as f32;
        nbrs[t0].push((ang, dir));
        nbrs[t1].push((ang, dir));
        tri_adj[t0].push(t1);
        tri_adj[t1].push(t0);
    }
    // The step a triangle's surface takes ACROSS an edge of direction `e_dir`: the widest
    // sub-crease dihedral among the edges of its 2-HOP patch that run parallel to it. Two
    // hops, because on a band of twisted quads the parallel next-ring edge belongs to the
    // triangle's diagonal partner, never to the triangle itself (a triangle's own edges are
    // mutually ~60 degrees — a 1-hop test measures nothing, ctx was 0.0 across the board).
    // Sub-crease, so a nearby RIM's 90 degrees cannot inflate the budget; and edges at nearly
    // the candidate's own angle are its own seam continuing sideways, not the surface
    // stepping, so they do not count either.
    let crossing = |t: usize, e_dir: [f32; 3], own: f32| -> f32 {
        let take = |u: usize, best: &mut f32| {
            for (a, d) in &nbrs[u] {
                if *a < crease_deg as f32
                    && dot(*d, e_dir).abs() > 0.8
                    && (*a - own).abs() > 0.15 * own + 0.05
                    && *a > *best
                {
                    *best = *a;
                }
            }
        };
        let mut best = 0.0f32;
        take(t, &mut best);
        for &u in &tri_adj[t] {
            take(u, &mut best);
        }
        best
    };
    // Emit. Boundaries between different smooth faces are edges as before (sharp past the
    // tangent threshold). NEW: inside one smooth face, an edge whose dihedral is a RATE
    // BREAK — clearly gentler than the face's own median facet step, yet not merely the
    // coplanar continuation a flush seam leaves — is a tangent SEAM: the line where a fillet
    // band lands on the face it blends into. That is the boundary the old detector merged
    // away, which is why every body reported tangent=0 and fillet seams had to be dragged
    // through regens as stored polylines.
    let cos_tan = tangent_deg.to_radians().cos() as f32;
    let probe: Option<(f32, f32)> = std::env::var("HCAD_EDGE_PROBE")
        .ok()
        .and_then(|v| {
            let mut it = v.split(',').filter_map(|x| x.trim().parse::<f32>().ok());
            Some((it.next()?, it.next()?))
        });
    let (mut sharp, mut tangent) = (Vec::new(), Vec::new());
    for ((i, j), ts) in &emap {
        let edge = [wpos(*i), wpos(*j)];
        if let Some((py, pr)) = probe {
            let hit = edge.iter().all(|p| {
                (p[1] - py).abs() < 5e-3 && ((p[0] * p[0] + p[2] * p[2]).sqrt() - pr).abs() < 5e-2
            });
            if hit {
                let d = if ts.len() == 2 { dot(tnorm[ts[0]], tnorm[ts[1]]) } else { f32::NAN };
                let a = (d as f64).clamp(-1.0, 1.0).acos().to_degrees();
                let roots: Vec<usize> = ts.iter().map(|&t| find(&mut uf, face_of[t])).collect();
                let needles = ts.iter().filter(|&&t| is_needle[t]).count();
                eprintln!(
                    "  PROBE ({:.3},{:.3},{:.3})-({:.3},{:.3},{:.3}) owners={} needles={needles} dihedral={a:.3} roots={roots:?}",
                    edge[0][0], edge[0][1], edge[0][2], edge[1][0], edge[1][1], edge[1][2], ts.len()
                );
            }
        }
        match ts.len() {
            1 => {
                if !is_needle[ts[0]] {
                    sharp.push(edge); // a true open boundary — the weld already paired fakes
                }
            }
            2 => {
                let (t0, t1) = (ts[0], ts[1]);
                if is_needle[t0] || is_needle[t1] {
                    continue;
                }
                let (g0, g1) = (face_of[t0], face_of[t1]);
                if g0 == g1 {
                    continue;
                }
                let d = dot(tnorm[t0], tnorm[t1]);
                let (r0, r1) = (find(&mut uf, g0), find(&mut uf, g1));
                if r0 != r1 {
                    if d < cos_tan {
                        sharp.push(edge);
                    } else {
                        tangent.push(edge);
                    }
                } else if d < cos_coplanar && d > cos_tan {
                    // Inside one smooth face: gentler than the tangent threshold, but NOT the
                    // dead-flat continuation a welded flush seam leaves. A SEAM where the
                    // angle breaks the rate either side steps at ACROSS this edge — a fillet
                    // band meets its face at half the band's own step, while the face beyond
                    // is flat (or curves some other way) in the crossing direction.
                    let ed = [edge[1][0] - edge[0][0], edge[1][1] - edge[0][1], edge[1][2] - edge[0][2]];
                    let el = (ed[0] * ed[0] + ed[1] * ed[1] + ed[2] * ed[2]).sqrt();
                    if el < 1e-12 {
                        continue;
                    }
                    let e_dir = [ed[0] / el, ed[1] / el, ed[2] / el];
                    let a = (d as f64).clamp(-1.0, 1.0).acos().to_degrees() as f32;
                    let ctx = crossing(t0, e_dir, a).max(crossing(t1, e_dir, a));
                    if std::env::var("HCAD_EDGE_DEBUG").is_ok() {
                        eprintln!("    cand ({:.2},{:.2},{:.2}) angle {a:.2} ctx {ctx:.2} n0={} n1={} -> {}",
                            edge[0][0], edge[0][1], edge[0][2], nbrs[t0].len(), nbrs[t1].len(),
                            if a < 0.6 * ctx { "TANGENT" } else { "no" });
                    }
                    if a < 0.6 * ctx {
                        tangent.push(edge);
                    }
                }
            }
            _ => {}
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
    // The tags come along, shifted with everything else: a nudge is a translation, so a face that
    // was a cylinder still is one, a hair further over.
    let shift = |s: &crate::Surf| match *s {
        crate::Surf::Plane { origin, normal } => crate::Surf::Plane {
            origin: [origin[0] + d[0] as f64, origin[1] + d[1] as f64, origin[2] + d[2] as f64],
            normal,
        },
        crate::Surf::Cylinder { origin, axis, radius } => crate::Surf::Cylinder {
            origin: [origin[0] + d[0] as f64, origin[1] + d[1] as f64, origin[2] + d[2] as f64],
            axis,
            radius,
        },
        crate::Surf::Torus { origin, axis, major, minor } => crate::Surf::Torus {
            origin: [origin[0] + d[0] as f64, origin[1] + d[1] as f64, origin[2] + d[2] as f64],
            axis,
            major,
            minor,
        },
    };
    TriMesh {
        positions: m.positions.iter().map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]]).collect(),
        normals: m.normals.clone(),
        indices: m.indices.clone(),
        surfaces: m.surfaces.iter().map(shift).collect(),
        tri_surf: m.tri_surf.clone(),
    }
}

/// Build a `Manifold` from a mesh, attaching one Manifold *original ID* per tagged surface so the
/// far side of a boolean can still say which surface each triangle came from. `reg` collects the
/// `id -> Surf` mapping the caller needs to read them back.
///
/// Runs must be CONTIGUOUS blocks of the flat index array, so the triangles are re-ordered to group
/// them by tag. Order carries no meaning in a boolean operand.
///
/// Meshes with nothing tagged take the plain path — no runs, no cost.
fn to_manifold_tagged(m: &TriMesh, reg: &mut HashMap<u32, crate::Surf>) -> Option<Manifold> {
    if m.indices.len() < 3 {
        return None;
    }
    let ntri = m.indices.len() / 3;
    if m.tri_surf.len() != ntri || m.surfaces.is_empty() {
        return to_manifold(m);
    }
    let (props, tris, from) = weld(m);
    // Walk the WELDED triangles, not the source ones: the weld drops any it made degenerate, so
    // source triangle t is not welded triangle t. `from` maps each survivor back to the source it
    // came from, which is where its tag is. Indexing the welded list by a source index instead ran
    // off its end and took eight parts' rebuilds down with it.
    //
    // Group by tag, untagged last. `sort_by_key` is stable, so triangles keep their relative order
    // within a face and a re-run gives the same mesh.
    let mut order: Vec<usize> = (0..from.len()).collect();
    order.sort_by_key(|&j| m.tri_surf[from[j]]);
    let base = manifold3d::reserve_ids(m.surfaces.len() as u32 + 1);
    let mut grouped: Vec<u32> = Vec::with_capacity(tris.len());
    let (mut run_index, mut run_ids) = (Vec::<u32>::new(), Vec::<u32>::new());
    let mut last: Option<u32> = None;
    for &j in &order {
        let tag = m.tri_surf[from[j]];
        if last != Some(tag) {
            run_index.push(grouped.len() as u32);
            // NO_SURF gets an id too — Manifold wants every triangle inside a run — it just never
            // reaches `reg`, so it reads back as "nobody said".
            let id = if tag == crate::NO_SURF { base + m.surfaces.len() as u32 } else { base + tag };
            run_ids.push(id);
            if tag != crate::NO_SURF {
                reg.insert(id, m.surfaces[tag as usize]);
            }
            last = Some(tag);
        }
        grouped.extend_from_slice(&tris[j * 3..j * 3 + 3]);
    }
    run_index.push(grouped.len() as u32); // sentinel
    let opts = manifold3d::MeshGLOptions::new().runs(&run_index, &run_ids);
    let meshgl = MeshGL::new_with_options(&props, 3, &grouped, opts).ok()?;
    Manifold::from_meshgl(&meshgl).ok()
}

/// Convert a `Manifold` back to a flat-shaded mesh, restoring the surface tags from the original
/// IDs `to_manifold_tagged` attached. Triangles the boolean created along an intersection belong to
/// whichever operand's surface they lie on, which is exactly what Manifold reports.
fn from_manifold_tagged(man: &Manifold, reg: &HashMap<u32, crate::Surf>) -> TriMesh {
    let mut out = from_manifold(man);
    if reg.is_empty() {
        return out;
    }
    let mgl = man.to_meshgl();
    let (run_index, run_ids) = (mgl.run_index(), mgl.run_original_id());
    let ntri = out.indices.len() / 3;
    if run_ids.is_empty() || run_index.len() < run_ids.len() {
        return out;
    }
    // `from_manifold` walks `tri_verts` in order, so triangle t sits at flat offset 3t and the run
    // it belongs to is the last one starting at or before that.
    let mut tri_surf = vec![crate::NO_SURF; ntri];
    let mut surfaces: Vec<crate::Surf> = Vec::new();
    for (r, &id) in run_ids.iter().enumerate() {
        let Some(s) = reg.get(&id) else { continue };
        let slot = match surfaces.iter().position(|x| x == s) {
            Some(i) => i as u32,
            None => {
                surfaces.push(*s);
                (surfaces.len() - 1) as u32
            }
        };
        let start = run_index[r] as usize / 3;
        let end = (run_index.get(r + 1).copied().unwrap_or(run_index[r]) as usize / 3).min(ntri);
        for e in tri_surf.iter_mut().take(end).skip(start) {
            *e = slot;
        }
    }
    if !surfaces.is_empty() {
        out.surfaces = surfaces;
        out.tri_surf = tri_surf;
    }
    out
}

/// Run one Manifold boolean attempt; `None` if an operand won't ingest or the op errors.
fn manifold_try(a: &TriMesh, b: &TriMesh, op: Op) -> Option<TriMesh> {
    // One registry for both operands: the ids are Manifold's, the surfaces are ours, and the
    // mapping only has to live as long as this call.
    let mut reg: HashMap<u32, crate::Surf> = HashMap::new();
    let (ma, mb) = (to_manifold_tagged(a, &mut reg)?, to_manifold_tagged(b, &mut reg)?);
    let r = match op {
        Op::Union => ma.union(&mb),
        Op::Difference => ma.difference(&mb),
        Op::Intersection => ma.intersection(&mb),
    };
    if r.status().is_err() {
        return None;
    }
    let mesh = from_manifold_tagged(&r, &reg);
    (!mesh.indices.is_empty()).then_some(mesh)
}

/// Give every vertex position an id, merging points closer together than a whisker of the part's
/// own size. Meshes reach a boolean from two directions — Manifold stores one position per vertex,
/// but a freshly built extrude computes a corner once for its cap and again for its wall — so
/// matching bit patterns would call an ordinary solid torn. The tolerance is relative because HCAD
/// parts run from millimetres to hundreds of units: it has to sit above f32 rounding noise at the
/// coordinates in play and below the smallest feature a sketch can carry.
fn weld_ids(m: &TriMesh) -> Vec<usize> {
    use std::collections::HashMap;
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let tol = (diag * 1.0e-6).max(1.0e-12);
    let cell = |p: [f32; 3]| [(p[0] / tol).floor() as i64, (p[1] / tol).floor() as i64, (p[2] / tol).floor() as i64];
    let mut buckets: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut exact: HashMap<[u32; 3], usize> = HashMap::new();
    let mut rep: Vec<[f32; 3]> = Vec::new();
    let mut ids = Vec::with_capacity(m.positions.len());
    for p in &m.positions {
        // A boolean result repeats each vertex exactly, so nearly every lookup lands here.
        let bits = [p[0] + 0.0, p[1] + 0.0, p[2] + 0.0].map(f32::to_bits); // −0.0 is 0.0
        if let Some(&id) = exact.get(&bits) {
            ids.push(id);
            continue;
        }
        let c = cell(*p);
        let mut found = None;
        // Search the neighbouring cells too: two copies of one point can straddle a cell boundary.
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = buckets.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else { continue };
                    for &id in list {
                        let q = rep[id];
                        let d2 = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2);
                        if d2 <= tol * tol {
                            found = Some(id);
                            break 'search;
                        }
                    }
                }
            }
        }
        let id = found.unwrap_or_else(|| {
            rep.push(*p);
            buckets.entry(c).or_default().push(rep.len() - 1);
            rep.len() - 1
        });
        exact.insert(bits, id);
        ids.push(id);
    }
    ids
}

/// How many edges do **not** carry exactly two faces. A closed solid has none: one face means a
/// hole in the surface, four means two surfaces meeting along it.
///
/// This is the only cheap check that sees a *sheet sealed inside a part* — two solids joined flush
/// can come back with both of their shared faces still in the middle. Volume can't see it (a sheet
/// has none), the bounding box can't, and it cuts like a wall the user can't remove.
fn torn_edges(m: &TriMesh) -> usize {
    use std::collections::HashMap;
    let ids = weld_ids(m);
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let v = [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]];
        for &(x, y) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *edges.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    edges.values().filter(|&&c| c != 2).count()
}

/// How many separate pieces the mesh is in (flood-fill across shared vertices).
fn shell_count(m: &TriMesh) -> usize {
    let ids = weld_ids(m);
    let n = ids.iter().copied().max().map_or(0, |m| m + 1);
    let mut uf: Vec<usize> = (0..n).collect();
    fn find(uf: &mut [usize], mut x: usize) -> usize {
        while uf[x] != x {
            uf[x] = uf[uf[x]];
            x = uf[x];
        }
        x
    }
    for t in m.indices.chunks_exact(3) {
        let v = [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]];
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2])] {
            let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
            if ra != rb {
                uf[ra] = rb;
            }
        }
    }
    let mut roots: Vec<usize> = (0..n).map(|i| find(&mut uf, i)).collect();
    roots.sort_unstable();
    roots.dedup();
    roots.len()
}

/// Is the point inside the solid? Ray parity along a direction chosen not to line up with any
/// axis or facet, so it doesn't graze an edge.
fn inside_solid(m: &TriMesh, p: [f32; 3]) -> bool {
    let dir = [0.573_257_f32, 0.577_350, 0.581_443];
    let mut crossings = 0u32;
    for t in m.indices.chunks_exact(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
        // Möller–Trumbore, forward hits only.
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let pv = [dir[1] * e2[2] - dir[2] * e2[1], dir[2] * e2[0] - dir[0] * e2[2], dir[0] * e2[1] - dir[1] * e2[0]];
        let det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
        if det.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let tv = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
        let u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let qv = [tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]];
        let v = (dir[0] * qv[0] + dir[1] * qv[1] + dir[2] * qv[2]) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv > 0.0 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// Does the result hold a **sheet sealed inside the part**?
///
/// A face of a solid has material on exactly one side of it. A face with the same answer on both
/// sides bounds nothing — it is a sheet in the middle of the part, left behind when a join keeps
/// both of its shared faces. Buried in material it is invisible; once a later cut opens the space
/// around it, the user meets it as a thin wall in the middle of the cut that will not go away,
/// because a cut removes material and this is not material.
///
/// This is narrower than a torn edge, which is what makes it the right trigger: two lumps of solid
/// meeting along a line also leave an edge carrying four faces, but every one of those faces still
/// bounds material and the part is sound. Only faces along a torn edge can be sheets, so only
/// those are probed, and no more than [`SHEET_PROBES`] of them — each probe reads the whole mesh,
/// and a sheet is made of many faces, so the first handful finds one whenever there is one.
fn seals_a_sheet(m: &TriMesh) -> bool {
    /// Cap on faces probed per result: enough to find a sheet, few enough that a mesh with many
    /// legitimate pinches doesn't pay a full inside/outside test for every one of them.
    const SHEET_PROBES: usize = 128;

    use std::collections::{HashMap, HashSet};
    let ids = weld_ids(m);
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let v = [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]];
        for &(x, y) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *edges.entry((x.min(y), x.max(y))).or_default() += 1;
        }
    }
    let torn: HashSet<(usize, usize)> = edges.iter().filter(|(_, &c)| c != 2).map(|(&e, _)| e).collect();
    if torn.is_empty() {
        return false;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &m.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    // Far enough off the surface to clear a coincident copy of it, near enough to stay inside the
    // material a sound face bounds.
    let eps = diag * 1.0e-5;
    let mut probed = 0;
    for t in m.indices.chunks_exact(3) {
        let v = [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]];
        if !torn.contains(&(v[0].min(v[1]), v[0].max(v[1])))
            && !torn.contains(&(v[1].min(v[2]), v[1].max(v[2])))
            && !torn.contains(&(v[2].min(v[0]), v[2].max(v[0])))
        {
            continue;
        }
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| m.positions[i as usize]);
        let n = face_normal(a, b, c);
        let mid = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0];
        let off = |s: f32| [mid[0] + n[0] * s, mid[1] + n[1] * s, mid[2] + n[2] * s];
        if inside_solid(m, off(eps)) == inside_solid(m, off(-eps)) {
            return true; // the same on both sides: it divides nothing
        }
        probed += 1;
        if probed >= SHEET_PROBES {
            break;
        }
    }
    false
}

/// Enclosed volume, via the divergence theorem. Used to hold a repair to its job: nudging an
/// operand may only resolve a degeneracy, never move the shape.
fn enclosed_volume(m: &TriMesh) -> f64 {
    let mut v = 0.0;
    for t in m.indices.chunks_exact(3) {
        let p = [t[0], t[1], t[2]].map(|i| m.positions[i as usize].map(f64::from));
        v += (p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1])
            - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0])
            + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]))
            / 6.0;
    }
    v.abs()
}

/// Manifold boolean with a tangency-breaking retry; `None` only if every attempt fails (then the
/// caller drops to the BSP CSG). The retries nudge `b` by a few sub-micron offsets — when the two
/// solids share a tangent/coincident band (a concentric revolve grazing the boss wall), the exact
/// coincidence is what trips Manifold up, and a tiny perturbation makes it resolve cleanly.
///
/// The same coincidence can also make it return **success with a torn surface** — join a foot flush
/// under a plate with the same bores through both and the two shared faces both survive, sealing a
/// sheet inside the part. So a result is judged, not just accepted: a torn one is treated exactly
/// like a failure and sent round the same retries.
/// Cheap screen: does the mesh anywhere FOLD BACK on itself — two real-sized triangles on
/// one edge with strongly opposed normals? Every film (a wall of thickness ~the nudge scale
/// left by a flush boolean) has such a rim somewhere; so do a few legitimate shapes (a
/// knife-edge wedge, a pinch), which is why this only SCREENS — the verdict comes from
/// measuring actual thin material, which costs more.
fn fold_screen(m: &TriMesh) -> bool {
    fold_count(m) > 0
}

/// How many opposed real-sized pairs `fold_screen` would trip on — countable, so a boolean's
/// result can be compared against its own operands: damage CARRIED IN is not repairable by
/// nudging, but a result that folds MORE than its operands did is new damage this boolean
/// made, however pinched the operands already were (a legitimately pinched body must not
/// grandfather every later flush join).
fn fold_count(m: &TriMesh) -> usize {
    let ids = weld_ids(m);
    let p = |i: u32| {
        let q = m.positions[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    let mut owners: HashMap<(usize, usize), Vec<[f64; 3]>> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let n = [
            (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
            (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
        ];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let e = |x: [f64; 3], y: [f64; 3]| {
            let d = [y[0] - x[0], y[1] - x[1], y[2] - x[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let longest = e(a, b).max(e(b, c)).max(e(c, a));
        if nl < 1e-12 || longest < 1e-9 || nl / longest < 1e-3 {
            continue; // a needle
        }
        let nrm = [n[0] / nl, n[1] / nl, n[2] / nl];
        let (ia, ib, ic) = (ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]);
        for (u, v) in [(ia, ib), (ib, ic), (ic, ia)] {
            let k = if u <= v { (u, v) } else { (v, u) };
            owners.entry(k).or_default().push(nrm);
        }
    }
    let mut count = 0usize;
    for own in owners.values() {
        for i in 0..own.len() {
            for j in i + 1..own.len() {
                let d = own[i][0] * own[j][0] + own[i][1] * own[j][1] + own[i][2] * own[j][2];
                if d < -0.85 {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Total area of FILM in `m` near the zone `[zone_lo, zone_hi]`: triangles whose centroid,
/// marched inward along -normal, meets the opposite surface within `max_thick`. A film is
/// the nudge-scale wall a flush boolean can leave standing (both its sides count, so the
/// area overstates by 2x — fine, it is only compared against itself). Restricted to the
/// zone the other operand touched, because that is the only place this boolean can have
/// created one, and a full-mesh scan would cost O(n^2) on every retry.
fn thin_film_area(m: &TriMesh, zone_lo: [f32; 3], zone_hi: [f32; 3], max_thick: f64) -> f64 {
    let p = |i: u32| {
        let q = m.positions[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    let tris: Vec<[[f64; 3]; 3]> = m.indices.chunks_exact(3).map(|t| [p(t[0]), p(t[1]), p(t[2])]).collect();
    let margin = 0.05f64;
    let mut area = 0.0f64;
    // EXACT folds first: a sheet of literally zero thickness — two real-sized triangles on one
    // welded edge with opposed normals — is invisible to the ray march below, whose self-hit
    // guard (t > 1e-6) rejects the opposite face at distance ~0. Count the area of every
    // in-zone triangle participating in such a fold, and remember it so the ray pass does not
    // count it twice.
    let ids = weld_ids(m);
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, t) in m.indices.chunks_exact(3).enumerate() {
        let (ia, ib, ic) = (ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]);
        for (u, v) in [(ia, ib), (ib, ic), (ic, ia)] {
            let k = if u <= v { (u, v) } else { (v, u) };
            edge_tris.entry(k).or_default().push(ti);
        }
    }
    let tri_geom = |ti: usize| -> Option<([f64; 3], [f64; 3], f64)> {
        let t = &tris[ti];
        let n = [
            (t[1][1] - t[0][1]) * (t[2][2] - t[0][2]) - (t[1][2] - t[0][2]) * (t[2][1] - t[0][1]),
            (t[1][2] - t[0][2]) * (t[2][0] - t[0][0]) - (t[1][0] - t[0][0]) * (t[2][2] - t[0][2]),
            (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]),
        ];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let e = |x: [f64; 3], y: [f64; 3]| {
            let d = [y[0] - x[0], y[1] - x[1], y[2] - x[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let longest = e(t[0], t[1]).max(e(t[1], t[2])).max(e(t[2], t[0]));
        if nl < 1e-12 || longest < 1e-9 || nl / longest < 1e-3 {
            return None; // a needle
        }
        Some(([n[0] / nl, n[1] / nl, n[2] / nl], n, nl))
    };
    let mut folded: Vec<bool> = vec![false; tris.len()];
    for own in edge_tris.values() {
        if own.len() != 2 {
            continue;
        }
        let (Some((na, _, _)), Some((nb, _, _))) = (tri_geom(own[0]), tri_geom(own[1])) else { continue };
        if na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2] >= -0.95 {
            continue;
        }
        // Opposed normals alone also describe a legitimate KNIFE WEDGE — a corner-sliver
        // fillet tool tapers to exactly that at its tangent lines. A film's two sides are
        // COINCIDENT sheets: every vertex of each triangle lies within film thickness of the
        // other's plane. A wedge's faces separate away from the shared edge and fail this.
        let coincident = |ti: usize, tj: usize| -> bool {
            let Some((nj, _, _)) = tri_geom(tj) else { return false };
            let q0 = tris[tj][0];
            tris[ti].iter().all(|v| {
                let d = (v[0] - q0[0]) * nj[0] + (v[1] - q0[1]) * nj[1] + (v[2] - q0[2]) * nj[2];
                d.abs() < max_thick
            })
        };
        if coincident(own[0], own[1]) && coincident(own[1], own[0]) {
            folded[own[0]] = true;
            folded[own[1]] = true;
        }
    }
    for (ti, t) in tris.iter().enumerate() {
        if !folded[ti] {
            continue;
        }
        let cen = [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
            (t[0][2] + t[1][2] + t[2][2]) / 3.0,
        ];
        if (0..3).any(|k| cen[k] < zone_lo[k] as f64 - margin || cen[k] > zone_hi[k] as f64 + margin) {
            continue;
        }
        if let Some((_, _, nl)) = tri_geom(ti) {
            area += nl * 0.5;
        }
    }
    for (ti, t) in tris.iter().enumerate() {
        if folded[ti] {
            continue; // already counted by the fold pass
        }
        let cen = [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
            (t[0][2] + t[1][2] + t[2][2]) / 3.0,
        ];
        if (0..3).any(|k| cen[k] < zone_lo[k] as f64 - margin || cen[k] > zone_hi[k] as f64 + margin) {
            continue;
        }
        let n = [
            (t[1][1] - t[0][1]) * (t[2][2] - t[0][2]) - (t[1][2] - t[0][2]) * (t[2][1] - t[0][1]),
            (t[1][2] - t[0][2]) * (t[2][0] - t[0][0]) - (t[1][0] - t[0][0]) * (t[2][2] - t[0][2]),
            (t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0]),
        ];
        let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if nl < 1e-12 {
            continue;
        }
        let dir = [-n[0] / nl, -n[1] / nl, -n[2] / nl];
        let mut best = f64::MAX;
        for (ui, u) in tris.iter().enumerate() {
            if ui == ti {
                continue;
            }
            let e1 = [u[1][0] - u[0][0], u[1][1] - u[0][1], u[1][2] - u[0][2]];
            let e2 = [u[2][0] - u[0][0], u[2][1] - u[0][1], u[2][2] - u[0][2]];
            let pv = [
                dir[1] * e2[2] - dir[2] * e2[1],
                dir[2] * e2[0] - dir[0] * e2[2],
                dir[0] * e2[1] - dir[1] * e2[0],
            ];
            let det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let tv = [cen[0] - u[0][0], cen[1] - u[0][1], cen[2] - u[0][2]];
            let uu = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) * inv;
            if !(-1e-9..=1.0 + 1e-9).contains(&uu) {
                continue;
            }
            let qv = [
                tv[1] * e1[2] - tv[2] * e1[1],
                tv[2] * e1[0] - tv[0] * e1[2],
                tv[0] * e1[1] - tv[1] * e1[0],
            ];
            let vv = (dir[0] * qv[0] + dir[1] * qv[1] + dir[2] * qv[2]) * inv;
            if vv < -1e-9 || uu + vv > 1.0 + 1e-9 {
                continue;
            }
            let tt = (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]) * inv;
            if tt > 1e-6 && tt < best {
                // Only an OPPOSING surface makes a film: the far side of a thin wall faces
                // back along the ray. A ray grazing into a perpendicular neighbouring face —
                // a band facet next to its own tangent line passes within a sagitta of it —
                // is geometry meeting, not a wall standing.
                let un = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                let ul = (un[0] * un[0] + un[1] * un[1] + un[2] * un[2]).sqrt();
                if ul > 1e-12 && (un[0] * dir[0] + un[1] * dir[1] + un[2] * dir[2]) / ul > 0.7 {
                    best = tt;
                }
            }
        }
        if best < max_thick {
            area += nl * 0.5;
        }
    }
    area
}

fn manifold_boolean(a: &TriMesh, b: &TriMesh, op: Op) -> Option<TriMesh> {
    let first = manifold_try(a, b, op);
    let bbox = |m: &TriMesh| {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &m.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    };
    // Anything this boolean created lives where operand b was; a film is offset-scale, so
    // anything under 8e-4 across is not a wall anyone drew.
    let (zlo, zhi) = bbox(b);
    // A film is offset-scale; anything under 8e-4 across is not a wall anyone drew. The
    // clearance-scale doubled sheets a FLUSH NO-OP bakes (1.4e-3 apart — beyond any nudge's
    // repair) are prevented upstream instead: a boolean that changed nothing returns the
    // body untouched.
    let film_t: f64 = 8.0e-4;
    let dbg = std::env::var("HCAD_BOOL_DEBUG").is_ok();
    if let Some(m) = &first {
        // A boolean that did not change the body returns the body — bit for bit. A flush
        // no-op (a boss drawn over material that is already there, its walls snapped to the
        // body's own) still re-tessellates through Manifold and bakes doubled sheets along
        // every coincident wall: sliver.hcad's bottom ring boss added 121 zero-thickness fold
        // pairs while changing the volume by nothing. No sub-micron nudge can repair sheets a
        // clearance-scale gap apart; not building them is the fix.
        if (enclosed_volume(m) - enclosed_volume(a)).abs() < enclosed_volume(a).max(1e-9) * 1e-6 {
            return Some(a.clone());
        }
        if !seals_a_sheet(m) {
            // Success is still judged for FILM: a flush difference can succeed with the
            // coincident wall left standing at offset thickness (sliver.hcad: a 1.5e-4 sheet
            // over a fillet's whole band — "the thin wall on the fillet"). The fold screen is
            // O(n) and almost always clean; only a hit pays for the thickness measure, and
            // only measured film sends a success into the retries.
            if !fold_screen(m) {
                return first;
            }
            if fold_screen(a) || fold_screen(b) {
                return first; // damage (or a knife-edge shape) carried in — not repairable here
            }
            let fa = thin_film_area(m, zlo, zhi, film_t);
            if dbg {
                eprintln!("BOOL first ok, folds, film {fa:.5}, torn {}", torn_edges(m));
            }
            if fa < 1.0e-4 {
                return first;
            }
        } else if seals_a_sheet(a) || seals_a_sheet(b) {
            // A sheet already carried by an operand comes back out of every attempt, so nudging
            // only trades one damaged mesh for another. Hand back what the caller would have had.
            return first;
        }
    } else if dbg {
        eprintln!("BOOL first FAILED to ingest/run");
    }
    let joined = first.as_ref().map(shell_count);
    let held = first.as_ref().map(enclosed_volume);
    let first_film = first.as_ref().map(|m| thin_film_area(m, zlo, zhi, film_t));
    let first_torn = first.as_ref().map(|m| torn_edges(m));
    let mut best: Option<(f64, usize, TriMesh)> = None; // (film area, torn edges, mesh)
    let mut fallback = None;
    // Asymmetric, irrational-ish nudges so no offset lands back on another coincidence — and
    // BOTH signs of each, because for a flush face the sign decides what the offset leaves
    // behind: nudged off the face, the offset stands as a film of wall; nudged into it, the
    // same offset is a sub-tolerance overcut that shows nothing. Which sign is which depends
    // on the face's orientation, so offer the pair and let the film measure pick.
    let base: [[f32; 3]; 3] = [[1.7e-4, 1.1e-4, 1.3e-4], [-2.3e-4, 1.9e-4, -1.5e-4], [3.1e-4, -2.7e-4, 2.1e-4]];
    let negated: [[f32; 3]; 3] = [[-1.7e-4, -1.1e-4, -1.3e-4], [2.3e-4, -1.9e-4, 1.5e-4], [-3.1e-4, 2.7e-4, -2.1e-4]];
    let mut queue: Vec<[f32; 3]> = base.to_vec();
    let mut qi = 0;
    while qi < queue.len() {
        let d = queue[qi];
        qi += 1;
        let Some(m) = manifold_try(a, &nudged(b, d), op) else { continue };
        let sound = !seals_a_sheet(&m)
            // A nudge big enough to part two solids that were touching also comes back sound — as
            // two separate bodies. Breaking the part in half is not a repair.
            && !matches!(joined, Some(k) if shell_count(&m) > k)
            // Nor is building a different shape. An offset this small can only move a hair of
            // material; more than that means the nudge landed the operands in a different
            // arrangement, and what the caller asked for is the un-nudged result.
            && !matches!(held, Some(v) if (enclosed_volume(&m) - v).abs() > v * 1.0e-3);
        if sound {
            let film = thin_film_area(&m, zlo, zhi, film_t);
            let torn = torn_edges(&m);
            // Film-free and pinch-free is as good as it gets — stop looking. (A sound result
            // can still pinch — two walls meeting exactly along a line, which is a real shape
            // and not the damage being repaired.)
            if film < 1.0e-4 && torn == 0 {
                if dbg {
                    eprintln!("BOOL nudge {d:?} clean (film {film:.5}) — taken");
                }
                return Some(m);
            }
            if dbg {
                eprintln!("BOOL nudge {d:?}: film {film:.5} torn {torn}");
            }
            // A candidate may not buy its film reduction by TEARING: an offset that overlaps a
            // flush face sheds slivers finer than the weld and the surface stops being closed —
            // a worse defect than the film (it is what makes an export invalid). Anything that
            // pinches more than the un-nudged result is out.
            if matches!(first_torn, Some(ft) if torn > ft) {
                continue;
            }
            let better = match &best {
                None => true,
                Some((bf, bt, _)) => film < *bf || (film == *bf && torn < *bt),
            };
            if better {
                best = Some((film, torn, m));
            }
            // Still chasing a film after the base offsets: for a flush face the SIGN of the
            // offset decides film-or-clean, so queue the negations — but only when there is a
            // film to beat, so the failure path (a mesh Manifold will not ingest at all, e.g.
            // the dense-sheet guard) keeps its original three fast attempts.
            if qi == queue.len() && queue.len() == base.len() && matches!(&best, Some((bf, _, _)) if *bf >= 1.0e-4) {
                queue.extend_from_slice(&negated);
            }
        } else if fallback.is_none() {
            fallback = Some(m);
        }
    }
    // The least-film candidate wins, but only if it beats what the un-nudged attempt already
    // had — an offset the caller never asked for has to buy something.
    match (best, first, first_film) {
        (Some((bf, _, m)), Some(f), Some(ff)) => Some(if bf < ff { m } else { f }),
        (Some((_, _, m)), Some(f), None) => Some(if seals_a_sheet(&f) { m } else { f }),
        (Some((_, _, m)), None, _) => Some(m),
        (None, f, _) => f.or(fallback),
    }
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
    // No tags on the far side. A reflected plane or cylinder is a perfectly good plane or
    // cylinder, but it is not the one recorded here, and a tag that describes the wrong geometry
    // is worse than none — the exporter would write a surface the triangles do not lie on.
    let mut out = TriMesh {
        positions: mesh.positions.iter().map(reflect_pt).collect(),
        normals: mesh.normals.iter().map(reflect_nrm).collect(),
        indices: Vec::with_capacity(mesh.indices.len()),
        ..Default::default()
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
    fn a_block_stacked_exactly_flush_merges_into_one_solid() {
        // Two blocks meeting exactly at z=4 — the everyday "extrude up from the face I just made".
        // They only touch, so a boolean can call the pair disjoint and hand back both bodies with
        // the shared cap still in the middle: a sheet sealed inside the part that no later cut can
        // remove, because it is not material and nothing bounds it.
        let sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let lower = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let upper = extrude_tool_mesh(&sq, &[], &xy(), 4.0, 2.0).unwrap();
        let u = mesh_union(&lower, &upper);
        assert!((volume(&u) - 600.0).abs() < 0.5, "stacked volume was {}", volume(&u));
        assert_eq!(torn_edges(&u), 0, "the shared face at z=4 was left inside the solid");
    }

    /// Load a fixture solid from `testdata` (vertex/face OBJ, no normals or texture coords).
    fn fixture(name: &str) -> TriMesh {
        let path = format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
        let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut m = TriMesh::default();
        let mut verts: Vec<[f32; 3]> = Vec::new();
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            match it.next() {
                Some("v") => {
                    let p: Vec<f32> = it.take(3).map(|x| x.parse().expect("vertex")).collect();
                    verts.push([p[0], p[1], p[2]]);
                }
                Some("f") => {
                    for x in it.take(3) {
                        let v = verts[x.parse::<usize>().expect("index") - 1];
                        m.indices.push(m.positions.len() as u32);
                        m.positions.push(v);
                    }
                }
                _ => {}
            }
        }
        m
    }

    #[test]
    fn a_flush_join_does_not_seal_a_sheet_inside_the_part() {
        // motormount.hcad's join, as the two solids that were actually handed to the boolean. A
        // foot sits exactly flush under the plate at y=0 and the same bolt holes run through both,
        // so each bore is the same cylinder twice over. Manifold reports success and hands back a
        // mesh whose bore rims carry four faces: two bores plus BOTH solids' shared caps. Those
        // caps are a sheet sealed inside the part — the user meets it as a thin wall in the middle
        // of a later cut that nothing will remove, because it is not material and a cut only
        // removes material.
        let (plate, foot) = (fixture("flush_join_plate.obj"), fixture("flush_join_foot.obj"));
        assert!(!seals_a_sheet(&plate), "fixture plate is already damaged");
        assert!(!seals_a_sheet(&foot), "fixture foot is already damaged");
        let u = mesh_union(&plate, &foot);
        assert!(!seals_a_sheet(&u), "the join left a sheet inside the part");
        // ...and the repair must join them, not sidestep the problem by parting them.
        assert_eq!(shell_count(&u), 1, "the join left the part in pieces");
        let expect = volume(&plate) + volume(&foot);
        assert!((volume(&u) - expect).abs() < expect * 1e-4, "union volume was {} (want {expect})", volume(&u));
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
