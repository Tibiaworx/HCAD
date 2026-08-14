//! Mesh bevel — Blender-style topology *surgery* (no CSG), prototype.
//!
//! Our CSG fillet leaves facets/dimples at 3-edge corners because bounding a sphere with a
//! box always shows a flat face, and tangent booleans are degenerate. Blender's bevel avoids
//! this by working on mesh topology: it offsets the faces adjacent to a bevelled edge, rebuilds
//! a strip where the sharp edge was, and stitches a small *vertex patch* at corners where
//! several bevelled edges meet — placing every new vertex analytically, never via a boolean.
//!
//! Step 1 (this file): rebuild flat-face topology from a triangle soup — welded vertices,
//! coplanar faces, the edges between faces, and the edges/faces meeting at each vertex. The
//! actual offset/strip/vertex-patch surgery builds on top of this.
#![allow(dead_code)] // work-in-progress prototype — items wired up in later steps

use crate::TriMesh;
use std::collections::{HashMap, HashSet};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 1e-12 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// A flat face: the welded triangles that are coplanar and edge-connected, plus a normal
/// and its boundary loop(s). Each loop is an ordered vertex ring, CCW about `normal`
/// (so the face interior is to the left when walking it); a face with a hole has two loops.
#[derive(Debug, Clone)]
pub struct Face {
    pub tris: Vec<usize>,
    pub normal: V3,
    pub loops: Vec<Vec<usize>>,
}

/// A model edge between two faces (or a boundary edge with one), as the welded vertex pair.
#[derive(Debug, Clone)]
pub struct TopoEdge {
    pub a: usize,
    pub b: usize,
    pub faces: Vec<usize>, // 1 (boundary) or 2 (interior); >2 = non-manifold
}

/// Reconstructed flat-face topology of a triangle mesh.
#[derive(Debug, Clone)]
pub struct Topo {
    pub verts: Vec<V3>,
    pub tris: Vec<[usize; 3]>, // welded triangle indices
    pub faces: Vec<Face>,
    pub tri_face: Vec<usize>, // which face each triangle belongs to
    pub edges: Vec<TopoEdge>, // welded mesh edges shared across *different* faces
    pub vert_edges: Vec<Vec<usize>>, // model-edge indices incident to each vertex
    pub vert_faces: Vec<Vec<usize>>, // face indices incident to each vertex
}

impl Topo {
    /// The model-edge index joining welded vertices `a` and `b`, if any.
    pub fn edge_between(&self, a: usize, b: usize) -> Option<usize> {
        self.vert_edges[a].iter().copied().find(|&ei| {
            let e = &self.edges[ei];
            (e.a == a && e.b == b) || (e.a == b && e.b == a)
        })
    }
}

/// Weld coincident vertices on a 1e-5 grid and return welded positions + per-input remap.
fn weld(mesh: &TriMesh) -> (Vec<V3>, Vec<usize>) {
    let key = |p: V3| {
        ((p[0] * 1.0e5).round() as i64, (p[1] * 1.0e5).round() as i64, (p[2] * 1.0e5).round() as i64)
    };
    let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut verts: Vec<V3> = Vec::new();
    let mut remap = vec![0usize; mesh.positions.len()];
    for (i, p) in mesh.positions.iter().enumerate() {
        let p = [p[0] as f64, p[1] as f64, p[2] as f64];
        let id = *map.entry(key(p)).or_insert_with(|| {
            verts.push(p);
            verts.len() - 1
        });
        remap[i] = id;
    }
    (verts, remap)
}

/// Union–find for coplanar triangle grouping.
struct Uf {
    parent: Vec<usize>,
}
impl Uf {
    fn new(n: usize) -> Self {
        Uf { parent: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != r {
            let next = self.parent[c];
            self.parent[c] = r;
            c = next;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

/// Rebuild the flat-face topology of a (watertight, flat-shaded) triangle mesh.
pub fn build_topo(mesh: &TriMesh) -> Topo {
    let (verts, remap) = weld(mesh);
    let mut tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
        .collect();
    // Guarantee outward, consistent winding so face normals point out — `edge_sign` relies on it.
    orient_consistently(&verts, &mut tris);

    // Per-triangle normal.
    let tnorm: Vec<V3> = tris
        .iter()
        .map(|t| norm(cross(sub(verts[t[1]], verts[t[0]]), sub(verts[t[2]], verts[t[0]]))))
        .collect();

    // Map each undirected welded edge → the triangles touching it.
    let ekey = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edge_tris.entry(ekey(a, b)).or_default().push(ti);
        }
    }

    // Group coplanar, edge-adjacent triangles into faces (union–find).
    let mut uf = Uf::new(tris.len());
    for ts in edge_tris.values() {
        if ts.len() == 2 && dot(tnorm[ts[0]], tnorm[ts[1]]) > 0.9995 {
            uf.union(ts[0], ts[1]);
        }
    }
    let mut root_to_face: HashMap<usize, usize> = HashMap::new();
    let mut faces: Vec<Face> = Vec::new();
    let mut tri_face = vec![0usize; tris.len()];
    for ti in 0..tris.len() {
        let r = uf.find(ti);
        let fi = *root_to_face.entry(r).or_insert_with(|| {
            faces.push(Face { tris: Vec::new(), normal: [0.0, 0.0, 0.0], loops: Vec::new() });
            faces.len() - 1
        });
        faces[fi].tris.push(ti);
        tri_face[ti] = fi;
    }
    // Face normal = (area-weighted) average of its triangle normals.
    for f in &mut faces {
        let mut n = [0.0; 3];
        for &ti in &f.tris {
            for k in 0..3 {
                n[k] += tnorm[ti][k];
            }
        }
        f.normal = norm(n);
    }

    // Boundary loops per face: directed boundary half-edges are those whose reverse is not
    // also a directed edge *within the same face*. Chain them tip-to-tail into ordered rings.
    for f in &mut faces {
        let mut dir: HashMap<usize, Vec<usize>> = HashMap::new(); // a -> [b...] directed half-edges
        let mut present: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for &ti in &f.tris {
            let t = tris[ti];
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                present.insert((a, b));
            }
        }
        // A directed edge is on the boundary when its opposite is absent (interior shared edges
        // appear in both directions and cancel).
        for &(a, b) in &present {
            if !present.contains(&(b, a)) {
                dir.entry(a).or_default().push(b);
            }
        }
        // Walk chains until all boundary half-edges are consumed.
        while let Some((&start, _)) = dir.iter().find(|(_, v)| !v.is_empty()) {
            let mut loop_v = vec![start];
            let mut cur = start;
            loop {
                let nexts = dir.get_mut(&cur).unwrap();
                let nv = nexts.pop().unwrap();
                if nexts.is_empty() {
                    dir.remove(&cur);
                }
                if nv == start {
                    break;
                }
                loop_v.push(nv);
                cur = nv;
                if dir.get(&cur).is_none_or(|v| v.is_empty()) {
                    break; // open chain (shouldn't happen on a watertight face)
                }
            }
            f.loops.push(loop_v);
        }
    }

    // Model edges: welded mesh edges whose two triangles belong to *different* faces.
    let mut edges: Vec<TopoEdge> = Vec::new();
    for (&(a, b), ts) in &edge_tris {
        let mut fs: Vec<usize> = ts.iter().map(|&t| tri_face[t]).collect();
        fs.sort_unstable();
        fs.dedup();
        if fs.len() >= 2 {
            edges.push(TopoEdge { a, b, faces: fs });
        }
    }

    // Vertex stars: incident model edges and faces per welded vertex.
    let mut vert_edges = vec![Vec::<usize>::new(); verts.len()];
    let mut vert_faces = vec![Vec::<usize>::new(); verts.len()];
    for (ei, e) in edges.iter().enumerate() {
        vert_edges[e.a].push(ei);
        vert_edges[e.b].push(ei);
    }
    for (fi, f) in faces.iter().enumerate() {
        for &ti in &f.tris {
            for &vi in &tris[ti] {
                if !vert_faces[vi].contains(&fi) {
                    vert_faces[vi].push(fi);
                }
            }
        }
    }

    Topo { verts, tris, faces, tri_face, edges, vert_edges, vert_faces }
}

/// Rolling-ball *setback* on each adjacent face for the edge between faces with outward
/// normals `na`, `nb`: the in-face distance from the sharp edge back to the ball's contact
/// line. For a fillet radius `r` and exterior dihedral β (= angle between the outward
/// normals), `s = r·tan(β/2)` — so a 90° box edge sets back exactly `r`. Returns `None` for a
/// reflex/concave edge (β handling differs there; handled in a later step).
pub fn edge_setback(na: V3, nb: V3, r: f64) -> Option<f64> {
    let beta = dot(na, nb).clamp(-1.0, 1.0).acos();
    if beta <= 1e-4 || beta >= std::f64::consts::PI - 1e-4 {
        return None; // coplanar or fully folded
    }
    Some(r * (beta * 0.5).tan())
}

/// Inset one face's boundary loops by the per-edge rolling-ball setback: each loop edge slides
/// `s` into the face interior, and consecutive offset lines are intersected to place the new
/// corner. The result is the flat remnant of the face after bevelling; its boundary is where
/// the edge cylinders and corner patches will attach. Returns one 3D point per loop vertex.
///
/// `selected[ei]` gates which model edges actually round — a non-selected boundary edge keeps a
/// zero setback, so the face stays put there (the edge remains sharp).
pub fn inset_loops(topo: &Topo, fi: usize, r: f64, selected: &[bool]) -> Vec<Vec<V3>> {
    let f = &topo.faces[fi];
    let n = f.normal;
    let mut out = Vec::with_capacity(f.loops.len());
    for lp in &f.loops {
        let m = lp.len();
        // Per directed loop edge i (lp[i] -> lp[i+1]): unit direction and inward setback.
        let mut dir = vec![[0.0; 3]; m];
        let mut setback = vec![0.0f64; m];
        for i in 0..m {
            let a = topo.verts[lp[i]];
            let b = topo.verts[lp[(i + 1) % m]];
            dir[i] = norm(sub(b, a));
            // The model edge across lp[i]->lp[i+1] gives the dihedral (and thus the setback).
            setback[i] = topo
                .edge_between(lp[i], lp[(i + 1) % m])
                .and_then(|ei| {
                    if !selected[ei] {
                        return Some(0.0); // sharp edge: face doesn't move here
                    }
                    let fs = &topo.edges[ei].faces;
                    let other = fs.iter().copied().find(|&g| g != fi)?;
                    edge_setback(n, topo.faces[other].normal, r)
                })
                .unwrap_or(0.0);
        }
        // New corner at lp[i] = intersection of offset lines of edges (i-1) and (i).
        let mut ring = Vec::with_capacity(m);
        for i in 0..m {
            let v = topo.verts[lp[i]];
            let prev = (i + m - 1) % m;
            let w_in = norm(cross(n, dir[prev])); // inward (interior is left of a CCW loop edge)
            let w_out = norm(cross(n, dir[i]));
            let p_in = add(v, scale(w_in, setback[prev])); // a point on the incoming offset line
            let p_out = add(v, scale(w_out, setback[i]));
            // When the two boundary edges are collinear (a mid-edge vertex left by triangulation)
            // the offset lines are near-parallel — the intersection runs off to infinity, which
            // would draw a feature edge shooting across the part. Cap how far the inset corner may
            // travel from the vertex (a sane corner is within a few × the setback); past that,
            // fall back to a plain perpendicular step.
            let s = setback[prev].max(setback[i]);
            let pt = match line_intersect(p_in, dir[prev], p_out, dir[i], n) {
                Some(p) if dot(sub(p, v), sub(p, v)).sqrt() <= 2.5 * s + 1e-9 => p,
                _ => p_out,
            };
            ring.push(pt);
        }
        out.push(ring);
    }
    out
}

/// Intersection of two coplanar lines (point `p` dir `d`, point `q` dir `e`) in the plane with
/// normal `n`. Returns `None` if (near) parallel.
fn line_intersect(p: V3, d: V3, q: V3, e: V3, n: V3) -> Option<V3> {
    let denom = dot(cross(d, e), n);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = dot(cross(sub(q, p), e), n) / denom;
    Some(add(p, scale(d, t)))
}

/// Spherical linear interpolation between two unit vectors.
fn slerp(a: V3, b: V3, t: f64) -> V3 {
    let d = dot(a, b).clamp(-1.0, 1.0);
    let om = d.acos();
    if om < 1e-6 {
        return norm(add(scale(a, 1.0 - t), scale(b, t)));
    }
    let s = om.sin();
    norm(add(scale(a, ((1.0 - t) * om).sin() / s), scale(b, (t * om).sin() / s)))
}

/// Solve the 3×3 system `M x = rhs` by Cramer's rule, where **rows** of `M` are `m0,m1,m2` —
/// i.e. `dot(m0, x) = rhs[0]`, `dot(m1, x) = rhs[1]`, `dot(m2, x) = rhs[2]`. (Every caller wants
/// this "3 planes through a point, each at a prescribed signed distance" form — the row-based
/// system — not "x is the combination of m0,m1,m2 that sums to rhs", which is what a naive
/// per-component `dot(rhs, cross(...))` formula actually computes; that reads deceptively close
/// to right since it *is* valid Cramer's rule, just for a transposed system.) The standard
/// closed form for the row-based solve is `x = Σᵢ rhs[i]·(the other two rows' cross product) /
/// det`, cyclically: `rhs[0]·(m1×m2) + rhs[1]·(m2×m0) + rhs[2]·(m0×m1)`.
fn solve3(m0: V3, m1: V3, m2: V3, rhs: V3) -> Option<V3> {
    let det = dot(m0, cross(m1, m2));
    if det.abs() < 1e-9 {
        return None;
    }
    let x = add(add(scale(cross(m1, m2), rhs[0]), scale(cross(m2, m0), rhs[1])), scale(cross(m0, m1), rhs[2]));
    Some(scale(x, 1.0 / det))
}

/// The rolling-ball centre for a 3-face corner: the point at signed distance `sign*r` from each
/// incident face plane (through `v`, outward normal `nf`) — `sign=-1` for a convex corner (the
/// sphere sits in the material, inscribed against the outward faces), `sign=+1` for a concave one
/// (the sphere sits in the open notch). It lies on all three incident edge cylinders' axes, so
/// the corner sphere blends smoothly into each. `None` if degenerate.
fn corner_centre(v: V3, na: V3, nb: V3, nc: V3, r: f64, sign: f64) -> Option<V3> {
    let x = solve3(na, nb, nc, [sign * r, sign * r, sign * r])?;
    Some(add(v, x))
}

/// Fill a spherical-triangle corner boundary (all `boundary` points lying on the sphere of
/// radius `rad` centred at `center`) with a smooth dome: concentric rings slerped from the
/// boundary toward a pole on the sphere, capped by a fan. Every generated point lies *exactly*
/// on the sphere, so the corner reads as a true sphere cap — Blender's `tri_corner` /
/// `snap_to_superellipsoid` case for the circular profile (a pure radial projection). The
/// outermost ring IS the passed boundary, so it welds to the adjoining edge strips by position
/// identity — no cracks. Emits into `b`; winding is fixed globally by `Build::finish`, so
/// triangle order here is free.
fn emit_sphere_dome(b: &mut Build, boundary: &[V3], center: V3, rad: f64, bands: usize) {
    let n = boundary.len();
    let dirs: Vec<V3> = boundary.iter().map(|&p| norm(sub(p, center))).collect();
    // Pole = the sphere point in the mean boundary direction (the dome's apex, inside the
    // spherical triangle since it spans well under a hemisphere).
    let mut pole_dir = [0.0; 3];
    for &d in &dirs {
        pole_dir = add(pole_dir, d);
    }
    pole_dir = norm(pole_dir);
    let on_sphere = |d: V3| add(center, scale(d, rad));

    // ring[0] = the boundary itself (shared with the edge strips); ring[k] slerps each
    // boundary direction a fraction k/bands toward the pole, staying on the sphere.
    let mut rings: Vec<Vec<usize>> = Vec::with_capacity(bands.max(1));
    rings.push(boundary.iter().map(|&p| b.v(p)).collect());
    for k in 1..bands {
        let t = k as f64 / bands as f64;
        rings.push(dirs.iter().map(|&d| b.v(on_sphere(slerp(d, pole_dir, t)))).collect());
    }
    // Quad strips between consecutive concentric rings.
    for k in 0..rings.len() - 1 {
        for i in 0..n {
            let j = (i + 1) % n;
            let (a0, a1) = (rings[k][i], rings[k][j]);
            let (b0, b1) = (rings[k + 1][i], rings[k + 1][j]);
            b.tri(a0, a1, b1);
            b.tri(a0, b1, b0);
        }
    }
    // Cap the innermost ring with a fan to the single pole vertex.
    let pole = b.v(on_sphere(pole_dir));
    let inner = rings.last().unwrap();
    for i in 0..n {
        b.tri(inner[i], inner[(i + 1) % n], pole);
    }
}

/// Output-mesh accumulator: collects raw triangles, then welds + orients into a `TriMesh`.
struct Build {
    pos: Vec<V3>,
    idx: Vec<[usize; 3]>,
}
impl Build {
    fn new() -> Self {
        Build { pos: Vec::new(), idx: Vec::new() }
    }
    fn v(&mut self, p: V3) -> usize {
        self.pos.push(p);
        self.pos.len() - 1
    }
    fn tri(&mut self, a: usize, b: usize, c: usize) {
        self.idx.push([a, b, c]);
    }
    /// Fan-triangulate a convex ring (indices already pushed).
    fn fan(&mut self, ring: &[usize]) {
        for i in 1..ring.len() - 1 {
            self.tri(ring[0], ring[i], ring[i + 1]);
        }
    }
    /// Weld coincident vertices, make winding globally consistent + outward, build a `TriMesh`.
    fn finish(self) -> TriMesh {
        // Weld on a 1e-5 grid.
        let key = |p: V3| ((p[0] * 1e5).round() as i64, (p[1] * 1e5).round() as i64, (p[2] * 1e5).round() as i64);
        let mut map: HashMap<(i64, i64, i64), usize> = HashMap::new();
        let mut verts: Vec<V3> = Vec::new();
        let mut remap = vec![0usize; self.pos.len()];
        for (i, &p) in self.pos.iter().enumerate() {
            let id = *map.entry(key(p)).or_insert_with(|| {
                verts.push(p);
                verts.len() - 1
            });
            remap[i] = id;
        }
        let mut tris: Vec<[usize; 3]> = self
            .idx
            .iter()
            .map(|t| [remap[t[0]], remap[t[1]], remap[t[2]]])
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
            .collect();

        orient_consistently(&verts, &mut tris);

        // Flat (per-triangle) normals.
        let mut positions = Vec::with_capacity(tris.len() * 3);
        let mut normals = Vec::with_capacity(tris.len() * 3);
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for t in &tris {
            let (a, b, c) = (verts[t[0]], verts[t[1]], verts[t[2]]);
            let nf = norm(cross(sub(b, a), sub(c, a)));
            let base = positions.len() as u32;
            for &p in &[a, b, c] {
                positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                normals.push([nf[0] as f32, nf[1] as f32, nf[2] as f32]);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        TriMesh { positions, normals, indices }
    }
}

/// Flood-fill a consistent winding across shared edges, then flip everything if the resulting
/// signed volume is negative (so normals face outward).
fn orient_consistently(verts: &[V3], tris: &mut [[usize; 3]]) {
    // Map undirected edge -> the (≤2) triangles on it.
    let ekey = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut edge_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edge_tris.entry(ekey(a, b)).or_default().push(ti);
        }
    }
    // BFS; a neighbour that traverses the shared edge the *same* direction must be flipped.
    let dir_same = |t: &[usize; 3], a: usize, b: usize| {
        [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].iter().any(|&(x, y)| x == a && y == b)
    };
    let mut seen = vec![false; tris.len()];
    for start in 0..tris.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        while let Some(ti) = stack.pop() {
            let t = tris[ti];
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                for &nb in &edge_tris[&ekey(a, b)] {
                    if seen[nb] {
                        continue;
                    }
                    // Consistent neighbours traverse the shared edge in the opposite direction.
                    if dir_same(&tris[nb], a, b) {
                        tris[nb].swap(1, 2);
                    }
                    seen[nb] = true;
                    stack.push(nb);
                }
            }
        }
    }
    // Global flip if inside-out (signed volume via tetrahedra from origin).
    let vol: f64 = tris
        .iter()
        .map(|t| dot(verts[t[0]], cross(verts[t[1]], verts[t[2]])))
        .sum();
    if vol < 0.0 {
        for t in tris.iter_mut() {
            t.swap(1, 2);
        }
    }
}

/// Ear-clip triangulate a simple (possibly concave) planar polygon lying in the plane with
/// normal `n`. Returns vertex-index triples wound the same way as the input loop. Handles the
/// concave notch a terminal-edge splice cuts into a face (see `run_surgery` step 1). Degenerate
/// leftovers that can't be clipped fall back to a fan, so the caller always gets a full cover.
fn ear_clip(pts: &[V3], n: V3) -> Vec<[usize; 3]> {
    let m = pts.len();
    if m < 3 {
        return Vec::new();
    }
    // 2D basis in the plane: (u, w, n) right-handed, so a CCW-about-n loop is CCW in (u, w).
    let seed = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = norm(sub(seed, scale(n, dot(seed, n))));
    let w = cross(n, u);
    let p2: Vec<[f64; 2]> = pts.iter().map(|&p| [dot(p, u), dot(p, w)]).collect();
    let x2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let mut area2 = 0.0;
    for i in 0..m {
        let (a, b) = (p2[i], p2[(i + 1) % m]);
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    // Clip in CCW order regardless of the input's winding; un-flip the output at the end.
    let flip = area2 < 0.0;
    let mut idx: Vec<usize> = (0..m).collect();
    if flip {
        idx.reverse();
    }
    let eps = area2.abs() * 1e-9 + 1e-12;
    let mut out: Vec<[usize; 3]> = Vec::new();
    while idx.len() > 3 {
        let k = idx.len();
        let mut clipped = false;
        for i in 0..k {
            let (ia, ib, ic) = (idx[(i + k - 1) % k], idx[i], idx[(i + 1) % k]);
            let (a, b, c) = (p2[ia], p2[ib], p2[ic]);
            if x2(a, b, c) <= eps {
                continue; // reflex or degenerate corner — not an ear
            }
            let contains_other = idx.iter().any(|&j| {
                j != ia
                    && j != ib
                    && j != ic
                    && x2(a, b, p2[j]) >= -eps
                    && x2(b, c, p2[j]) >= -eps
                    && x2(c, a, p2[j]) >= -eps
            });
            if contains_other {
                continue;
            }
            out.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break; // pathological input — close the remainder as a fan below
        }
    }
    for i in 1..idx.len().saturating_sub(1) {
        out.push([idx[0], idx[i], idx[i + 1]]);
    }
    if flip {
        for t in &mut out {
            t.swap(1, 2);
        }
    }
    out
}

/// Direction a model edge is traversed within face `fi`'s CCW boundary loop (so the face
/// interior is on its left). `None` if the edge isn't on that face's boundary.
fn loop_dir(topo: &Topo, fi: usize, a: usize, b: usize) -> Option<V3> {
    for lp in &topo.faces[fi].loops {
        let m = lp.len();
        for i in 0..m {
            let (x, y) = (lp[i], lp[(i + 1) % m]);
            if (x, y) == (a, b) || (x, y) == (b, a) {
                return Some(norm(sub(topo.verts[y], topo.verts[x])));
            }
        }
    }
    None
}

/// Convexity of a model edge: `-1` convex (a round bulges outward), `+1` concave (a round fills
/// an inside corner), `0` flat/degenerate. Decided locally: walking the edge with face A's
/// interior on the left, the edge is convex when face B folds *away* from that interior.
pub fn edge_sign(topo: &Topo, e: &TopoEdge) -> i32 {
    let (fa, fb) = (e.faces[0], e.faces[1]);
    let na = topo.faces[fa].normal;
    let nb = topo.faces[fb].normal;
    let t = match loop_dir(topo, fa, e.a, e.b) {
        Some(t) => t,
        None => return 0,
    };
    let inward = cross(na, t); // into face A
    let d = dot(inward, nb);
    if d < -1e-9 {
        -1
    } else if d > 1e-9 {
        1
    } else {
        0
    }
}

/// The arc where edge `e`'s rolling-ball surface terminates at vertex `vi` — `seg+1` points
/// from the face-A corner to the face-B corner, lying on the edge's fillet cylinder. Both the
/// edge strip and the corner patch call this, so their shared boundary is bit-identical and
/// welds without cracks. `None` if the edge is flat/degenerate at this vertex.
///
/// At a vertex where *another* selected edge also converges (e.g. two perpendicular fillet
/// edges meeting at a box corner), the far face's inset corner (`cb`) is a **mitred** point —
/// pulled along this edge's own direction by that other edge's setback, not by anything to do
/// with this edge's cylinder. Interpolating "along" linearly from `ca` to that drifted `cb`
/// smears the whole strip sideways along the edge, so it bulges past the true wall (the "block"
/// artifact). Since `o` sits exactly at `vi`'s own along-position by construction, every
/// *interior* sample is built at `along = 0` — a pure radial fan pinned at this vertex — so
/// the cylinder stays correct along the edge's length; only the very last band (bridging to the
/// mitred `cb`, needed so the flat far face still welds without a crack) absorbs the drift, and
/// it does that over a short, local hop instead of the whole strip.
fn edge_end_ring(topo: &Topo, e: &TopoEdge, vi: usize, cpt: &dyn Fn(usize, usize) -> V3, r: f64, seg: usize) -> Option<Vec<V3>> {
    let (fa, fb) = (e.faces[0], e.faces[1]);
    let (na, nb) = (topo.faces[fa].normal, topo.faces[fb].normal);
    let sigma = edge_sign(topo, e) as f64;
    if sigma == 0.0 {
        return None;
    }
    let t = norm(sub(topo.verts[e.b], topo.verts[e.a]));
    // Axis line of the fillet cylinder: at signed distance σ·r from both faces, parallel to t.
    // `off` only depends on the two face normals and the edge direction (not on a position), so
    // it's valid at either endpoint — anchor `o` at `vi` itself (not always `e.a`), so `dot(off,
    // t) == 0` really does put `o` at exactly *this* vertex's along-position, matching the
    // interior sweep below. (Anchoring at `e.a` unconditionally — the previous bug — silently
    // relied on the old along-interpolation to drag the far end's ring back to vertex `e.b`; once
    // the interior sweep stopped interpolating along, that made the `e.b` ring collapse onto
    // `e.a`'s position instead.)
    let off = solve3(na, nb, t, [sigma * r, sigma * r, 0.0])?;
    let o = add(topo.verts[vi], off);
    // Radial direction of each face's corner point at this vertex (its along-component is
    // discarded for the interior sweep — see the doc comment above).
    let radial = |p: V3| -> V3 { norm(sub(sub(p, o), scale(t, dot(sub(p, o), t)))) };
    let (ca, cb) = (cpt(vi, fa), cpt(vi, fb));
    let (da, db) = (radial(ca), radial(cb));
    let mut pts = Vec::with_capacity(seg + 1);
    for j in 0..=seg {
        // Pin the two ends to the exact corner points so the flat-face seam can't crack; every
        // interior arc point sweeps radially at along=0 (this vertex's own cross-section).
        if j == 0 {
            pts.push(ca);
        } else if j == seg {
            pts.push(cb);
        } else {
            let p = j as f64 / seg as f64;
            let d = slerp(da, db, p);
            pts.push(add(o, scale(d, r)));
        }
    }
    Some(pts)
}

/// Squared distance from point `p` to segment `a`–`b`.
fn pt_seg_dist2(p: V3, a: V3, b: V3) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 1e-18 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = add(a, scale(ab, t));
    let d = sub(p, q);
    dot(d, d)
}

/// Does model edge `e` lie on one of the world-space `picked` edge polylines? Matched by its
/// midpoint sitting on a polyline segment (model edges are sub-segments of the picked edges).
fn edge_is_picked(topo: &Topo, e: &TopoEdge, picked: &[Vec<[f64; 3]>]) -> bool {
    let mid = scale(add(topo.verts[e.a], topo.verts[e.b]), 0.5);
    let tol2 = 1.0e-4; // 0.01 units
    picked.iter().any(|poly| {
        if poly.windows(2).any(|w| pt_seg_dist2(mid, w[0], w[1]) < tol2) {
            return true;
        }
        // Wrap heal: older documents stored a closed loop WITHOUT repeating its first point, so
        // the closing segment is missing — that one edge never rounds (a notch on the body, and
        // its seam detours as a chevron). If the polyline's end gap is on the scale of its own
        // segments it was meant as a loop: test the closing pair too. (A straight 2-point pick
        // "wraps" onto its own segment reversed — harmless.)
        if poly.len() >= 3 {
            let (first, last) = (poly[0], poly[poly.len() - 1]);
            let gd = sub(last, first);
            let gap2 = dot(gd, gd);
            if gap2 > tol2 {
                let mut seg2: Vec<f64> = poly.windows(2).map(|w| { let d = sub(w[1], w[0]); dot(d, d) }).collect();
                seg2.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med2 = seg2[seg2.len() / 2];
                // gap ≤ ~1.75× the median segment length (compared squared).
                if gap2 <= med2 * 3.1 {
                    return pt_seg_dist2(mid, last, first) < tol2;
                }
            }
        }
        false
    })
}

/// Round **every** edge of a solid (the all-edges prototype). See [`bevel_mesh_selected`].
pub fn bevel_mesh(mesh: &TriMesh, r: f64, seg: usize) -> Option<TriMesh> {
    bevel_mesh_selected(mesh, r, seg, &[])
}

/// Shared front half of every bevel op: rebuild topology, decide which model edges round, and
/// place each face's inset corners. `None` if nothing is selected. Computing this once lets the
/// surgery and the feature-edge emission share a single (relatively costly) `build_topo` pass.
type BevelPrep = (Topo, Vec<bool>, HashMap<(usize, usize), V3>);
fn bevel_prep(mesh: &TriMesh, r: f64, picked: &[Vec<[f64; 3]>]) -> Option<BevelPrep> {
    let topo = build_topo(mesh);
    let all = picked.is_empty();
    let selected: Vec<bool> = topo.edges.iter().map(|e| all || edge_is_picked(&topo, e, picked)).collect();
    if !selected.iter().any(|&s| s) {
        return None;
    }
    // corner(v, f): where face f's boundary insets to at vertex v (rolling-ball setback, zero on
    // sharp edges). Single source of truth for flat faces, edge strips, patches AND tangent edges.
    let mut corner: HashMap<(usize, usize), V3> = HashMap::new();
    for fi in 0..topo.faces.len() {
        let rings = inset_loops(&topo, fi, r, &selected);
        for (li, lp) in topo.faces[fi].loops.iter().enumerate() {
            for (k, &vi) in lp.iter().enumerate() {
                corner.insert((vi, fi), rings[li][k]);
            }
        }
    }
    Some((topo, selected, corner))
}

/// The selectable feature edges from a prepared bevel: the inset boundary segment along each
/// selected model edge (a fillet's tangent edges / a chamfer's hard edges), one per adjacent
/// face. Follows the rounded body and chains up for picking. Each segment carries the normal of
/// the flat face it borders — the segment is only valid while the final surface UNDER it still
/// faces that way (a later cut whose wall merely grazes the line must not keep it alive).
fn emit_feature_edges(topo: &Topo, selected: &[bool], corner: &HashMap<(usize, usize), V3>) -> Vec<([[f32; 3]; 2], [f32; 3])> {
    let f32a = |p: V3| [p[0] as f32, p[1] as f32, p[2] as f32];
    let mut out = Vec::new();
    for fi in 0..topo.faces.len() {
        let n = f32a(topo.faces[fi].normal);
        for lp in &topo.faces[fi].loops {
            let m = lp.len();
            for k in 0..m {
                let (a, b) = (lp[k], lp[(k + 1) % m]);
                if topo.edge_between(a, b).is_some_and(|ei| selected[ei]) {
                    let pa = corner.get(&(a, fi)).copied().unwrap_or(topo.verts[a]);
                    let pb = corner.get(&(b, fi)).copied().unwrap_or(topo.verts[b]);
                    out.push(([f32a(pa), f32a(pb)], n));
                }
            }
        }
    }
    out
}

/// How much of face `fi` the inset turns inside-out — the area of its triangles that end up facing
/// against the face normal.
///
/// A flat face is re-emitted with its ORIGINAL triangulation and every vertex moved to its inset
/// corner (step 1 below). That is only valid while the move stays injective — which it is exactly
/// as long as each boundary vertex's setback is small next to the spacing of its neighbours along
/// the boundary. Where a selected edge ENDS on a tessellated CURVED wall, it isn't: the setback is
/// the full fillet radius while the neighbouring boundary vertices are facet chords away, a small
/// fraction of it. The edge's end vertices are then dragged clean across several neighbours, the
/// inset outline crosses itself, and the triangles hanging off those vertices flip over — leaving
/// two coplanar sheets stacked in the same plane, which z-fight and read as stripes down the
/// fillet's run-out. (Measured on the boss-sector junction at r=2.15: 5.49 units of face folded
/// back, 96.4 units of triangle area in a plane that should hold 85.4.)
///
/// Volume cannot see this: a fold adds and removes material in equal measure, so the body still
/// weighs exactly right and every Pappus check passes with the defect present. Orientation is what
/// catches it. Judged against the FACE's normal, not each triangle's own before-normal, so a
/// sliver's noisy normal can't cast the deciding vote.
fn inset_fold_area(topo: &Topo, fi: usize, cpt: &dyn Fn(usize, usize) -> V3) -> f64 {
    let n = topo.faces[fi].normal;
    topo.faces[fi]
        .tris
        .iter()
        .filter_map(|&ti| {
            let t = topo.tris[ti];
            let before = cross(sub(topo.verts[t[1]], topo.verts[t[0]]), sub(topo.verts[t[2]], topo.verts[t[0]]));
            let after = cross(sub(cpt(t[1], fi), cpt(t[0], fi)), sub(cpt(t[2], fi), cpt(t[0], fi)));
            (dot(before, n) > 1e-12 && dot(after, n) < -1e-12).then(|| dot(after, after).sqrt() * 0.5)
        })
        .sum()
}

/// How many selected model edges the CSG round can still finish in an interactive budget.
///
/// Handing a folded bevel to `round_mesh` only helps if it comes back. It unions one fillet solid
/// per picked segment, and the cost is not linear — measured on the ring-and-sector body at r=0.5:
/// 1 segment 26 ms, 2 segments 55 ms, 4 segments 135 ms, **8 segments 61 s**. So a lone terminal
/// edge (the shape that folds in practice — a fillet running out onto a curved wall) is cheap to
/// hand over, while a whole picked rim would wedge the UI, which regenerates on the main thread.
/// Past this bound the folded mesh is emitted as-is: a striped run-out is a bad look, but it is a
/// far better one than a frozen application.
const CSG_ROUND_MAX_EDGES: usize = 4;

/// Escape hatch for measuring the two engines against each other on the same document:
/// `HCAD_BEVEL_NO_CSG_HANDOVER=1` keeps the folded surgery result instead of declining.
fn csg_handover_enabled() -> bool {
    std::env::var("HCAD_BEVEL_NO_CSG_HANDOVER").is_err()
}

/// The surgery half of a prepared bevel: flat faces inset to their corners, a convex/concave
/// cylinder strip per selected edge, and a fanned patch per corner. `None` if the result isn't
/// watertight (a partial-vertex T-junction the prototype can't split → caller falls back to CSG).
fn run_surgery(topo: &Topo, selected: &[bool], corner: &HashMap<(usize, usize), V3>, r: f64, seg: usize) -> Option<TriMesh> {
    let verts = &topo.verts;
    let cpt = |vi: usize, fi: usize| -> V3 { corner.get(&(vi, fi)).copied().unwrap_or(verts[vi]) };
    // A face the inset turns inside-out can only be emitted as a self-overlapping sheet — see
    // `inset_fold_area`. Hand those to the CSG round, which does this shape properly (on the
    // boss-sector junction: volume to 0.7%, the face plane covered exactly once, nothing through
    // the walls) — but only while the pick is small enough for it to finish, per
    // `CSG_ROUND_MAX_EDGES`. The threshold is on fold AREA, not on any fold at all, so a sliver
    // flipped by rounding noise doesn't push a good bevel onto the slower path; a real fold runs
    // to whole multiples of r², a numerical one to a millionth of it.
    let folded: f64 = (0..topo.faces.len()).map(|fi| inset_fold_area(topo, fi, &cpt)).sum();
    if folded > 0.01 * r * r
        && selected.iter().filter(|&&s| s).count() <= CSG_ROUND_MAX_EDGES
        && csg_handover_enabled()
    {
        return None;
    }
    let mut b = Build::new();

    // 0) Terminal-edge splices. At a vertex where exactly ONE selected edge ends (its other
    //    incident edges all sharp, all bordering one shared untouched face `fs`), the fillet
    //    arc's end ring lies entirely IN `fs`'s plane whenever the edge meets that face
    //    squarely (the overwhelmingly common case — any box edge). The old approach left `fs`
    //    covering its full original polygon and stretched a separate, coplanar corner-patch
    //    membrane over the arc↔corner region to close the mesh — topologically watertight but
    //    geometrically a double-covered sheet (renders as self-clipping at the fillet's ends).
    //    The right shape: `fs` itself gets the quarter-arc NOTCH cut out of its outline — the
    //    ring replaces the original corner vertex in its boundary loop — so the region is
    //    covered exactly once and the fillet strip welds edge-to-edge into the face. This is
    //    what Blender's bevel does for terminal edges (its faces are rebuilt polygons, so the
    //    notch falls out naturally there). Non-square meetings (oblique corner: ring not in
    //    fs's plane) keep the membrane fallback.
    let mut splices: HashMap<usize, (usize, Vec<V3>)> = HashMap::new(); // vertex → (face, arc ring)
    for vi in 0..topo.verts.len() {
        let inc = &topo.vert_edges[vi];
        let sel: Vec<usize> = inc.iter().copied().filter(|&ei| selected[ei]).collect();
        if sel.len() != 1 {
            continue;
        }
        let e = &topo.edges[sel[0]];
        if e.faces.len() != 2 {
            continue;
        }
        let (fa, fb) = (e.faces[0], e.faces[1]);
        let mut others: Vec<usize> = topo.vert_faces[vi].iter().copied().filter(|&f| f != fa && f != fb).collect();
        others.sort_unstable();
        others.dedup();
        if others.len() != 1 {
            continue; // not a clean 3-face corner — membrane fallback
        }
        let fs = others[0];
        // Faces with holes are fine: the notched loop gets hole-bridged triangulation below.
        if !topo.faces[fs].loops.iter().any(|lp| lp.contains(&vi)) {
            continue; // the vertex isn't on any boundary loop of fs — not a splice shape
        }
        if inc.iter().any(|&ei| ei != sel[0] && !topo.edges[ei].faces.contains(&fs)) {
            continue; // a sharp edge here not bordering fs — not the simple terminal shape
        }
        let Some(ring) = edge_end_ring(topo, e, vi, &cpt, r, seg) else { continue };
        // The whole ring must lie in fs's plane, or the notch wouldn't be flat.
        let n = topo.faces[fs].normal;
        let v = verts[vi];
        let eps = 1e-6 * r.max(1e-6);
        if ring.iter().any(|&p| dot(sub(p, v), n).abs() > eps) {
            continue;
        }
        splices.insert(vi, (fs, ring));
    }

    // 0.5) Weld corners — Blender's M_NONE case: exactly TWO selected edges meet at this
    //    vertex, and their end rings are pinned to the same two mitred face corners (the
    //    shared face's corner and the far point where the other two faces meet). Each ring's
    //    INTERIOR points sit on its own edge's cylinder cross-section, and the two cylinders
    //    disagree in between — so ending each strip on its own ring leaves a twisted gap that
    //    a fanned patch can only paper over (the "wing" fins reported against
    //    fillererror3.hcad's top-rim fillet). Blender's answer: no corner geometry at all —
    //    blend the two rings point-by-point (mid_v3_v3v3 in bmesh_bevel.cc) and terminate
    //    BOTH strips on the single shared blended arc. Bit-identical shared points ⇒ the two
    //    strips join seamlessly around the corner; the corner patch is skipped entirely.
    let mut ring_override: HashMap<(usize, usize), Vec<V3>> = HashMap::new(); // (edge, vertex) → ring
    let mut welded: HashSet<usize> = HashSet::new();
    for vi in 0..topo.verts.len() {
        if splices.contains_key(&vi) {
            continue;
        }
        let sel: Vec<usize> = topo.vert_edges[vi].iter().copied().filter(|&ei| selected[ei]).collect();
        if sel.len() != 2 {
            continue;
        }
        let (e1, e2) = (&topo.edges[sel[0]], &topo.edges[sel[1]]);
        let (Some(r1), Some(r2)) = (
            edge_end_ring(topo, e1, vi, &cpt, r, seg),
            edge_end_ring(topo, e2, vi, &cpt, r, seg),
        ) else {
            continue;
        };
        let d2 = |p: V3, q: V3| dot(sub(p, q), sub(p, q));
        // Orient ring 2 to run the same way as ring 1, then require both endpoints to
        // genuinely coincide (they're the same mitred corners, computed from the same
        // offset intersections — anything else isn't a weldable corner; fall back).
        let flipped = d2(r2[0], r1[0]) > d2(r2[r2.len() - 1], r1[0]);
        let r2o: Vec<V3> = if flipped { r2.iter().rev().copied().collect() } else { r2.clone() };
        let tol2 = (1.0e-6 * (1.0 + r)).powi(2);
        if d2(r2o[0], r1[0]) > tol2 || d2(r2o[r2o.len() - 1], r1[r1.len() - 1]) > tol2 {
            continue;
        }
        let blended: Vec<V3> = r1.iter().zip(&r2o).map(|(&p, &q)| scale(add(p, q), 0.5)).collect();
        // Store each edge's override in that edge's OWN ring orientation (edge 2 saw the
        // blend through its possibly-reversed lens), or its strip would twist.
        ring_override.insert((sel[0], vi), blended.clone());
        ring_override.insert(
            (sel[1], vi),
            if flipped { blended.iter().rev().copied().collect() } else { blended },
        );
        welded.insert(vi);
    }

    // 1) Flat faces: reuse the original triangulation with each vertex moved to its inset
    //    corner — except spliced faces, whose boundary loop gets the arc notch cut in and is
    //    re-triangulated (ear clip; the notch makes the polygon concave).
    for fi in 0..topo.faces.len() {
        let f = &topo.faces[fi];
        let has_splice = f.loops.iter().any(|lp| lp.iter().any(|w| splices.get(w).is_some_and(|(sf, _)| *sf == fi)));
        if !has_splice {
            for &ti in &f.tris {
                let t = topo.tris[ti];
                let a = b.v(cpt(t[0], fi));
                let c = b.v(cpt(t[1], fi));
                let d = b.v(cpt(t[2], fi));
                b.tri(a, c, d);
            }
            continue;
        }
        // Every loop of the face, with arc notches spliced in where a terminal fillet ends
        // on it and every other vertex moved to its inset corner.
        let splice_loop = |lp: &Vec<usize>| -> Vec<V3> {
            let m = lp.len();
            let mut poly: Vec<V3> = Vec::new();
            for i in 0..m {
                let wv = lp[i];
                match splices.get(&wv) {
                    Some((sf, ring)) if *sf == fi => {
                        // Orient the ring so it continues from the loop's incoming edge.
                        let prev = verts[lp[(i + m - 1) % m]];
                        let d_first = pt_seg_dist2(ring[0], prev, verts[wv]);
                        let d_last = pt_seg_dist2(ring[ring.len() - 1], prev, verts[wv]);
                        if d_first <= d_last {
                            poly.extend(ring.iter().copied());
                        } else {
                            poly.extend(ring.iter().rev().copied());
                        }
                    }
                    _ => poly.push(cpt(wv, fi)),
                }
            }
            poly
        };
        if f.loops.len() == 1 {
            let poly = splice_loop(&f.loops[0]);
            for t in ear_clip(&poly, f.normal) {
                let a = b.v(poly[t[0]]);
                let c = b.v(poly[t[1]]);
                let d = b.v(poly[t[2]]);
                b.tri(a, c, d);
            }
            continue;
        }
        // Holes present (excut's chamfered/counterbored top face): project every loop into
        // the face plane, bridge the holes into the outer, ear-clip, lift back. Without this
        // the face used to fall back to full-square triangles + a coplanar membrane over the
        // arc — the visible "thin square corner" flap.
        let n = f.normal;
        let seed = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let u = norm(sub(seed, scale(n, dot(seed, n))));
        let w = cross(n, u);
        let loops3: Vec<Vec<V3>> = f.loops.iter().map(splice_loop).collect();
        let to2 = |p: V3| [dot(p, u), dot(p, w)];
        let loops2: Vec<Vec<[f64; 2]>> = loops3.iter().map(|l| l.iter().map(|&p| to2(p)).collect()).collect();
        let area2 = |l: &Vec<[f64; 2]>| -> f64 {
            let mut a = 0.0;
            for k in 0..l.len() {
                let (p, q) = (l[k], l[(k + 1) % l.len()]);
                a += p[0] * q[1] - q[0] * p[1];
            }
            (a * 0.5).abs()
        };
        let outer_i = (0..loops2.len()).max_by(|&i, &j| area2(&loops2[i]).partial_cmp(&area2(&loops2[j])).unwrap()).unwrap();
        let holes2: Vec<Vec<[f64; 2]>> = (0..loops2.len()).filter(|&i| i != outer_i).map(|i| loops2[i].clone()).collect();
        let merged = crate::bridge_holes(&loops2[outer_i], &holes2);
        // Lift 2D back to the face plane through a reference vertex.
        let r0 = loops3[outer_i][0];
        let r02 = to2(r0);
        let lift0 = |q: [f64; 2]| add(r0, add(scale(u, q[0] - r02[0]), scale(w, q[1] - r02[1])));
        // The planar lift drops any tiny off-plane component a loop vertex carried — the same
        // vertex emitted by the neighbouring wall then no longer welds, cracking the mesh
        // (and failing the watertight check). Snap each lifted point back onto the exact
        // original loop vertex it came from.
        let lift = |q: [f64; 2]| -> V3 {
            let p = lift0(q);
            for l in &loops3 {
                for &v0 in l {
                    if dot(sub(p, v0), sub(p, v0)) < 1e-10 {
                        return v0;
                    }
                }
            }
            p
        };
        // earcut emits CCW in (u, w); the surrounding surface follows the topo loop's own
        // winding — mismatched face triangles fail the watertight self-check. Match it.
        let outer_signed = {
            let l = &loops2[outer_i];
            let mut a2 = 0.0;
            for k in 0..l.len() {
                let (p, q) = (l[k], l[(k + 1) % l.len()]);
                a2 += p[0] * q[1] - q[0] * p[1];
            }
            a2
        };
        for t in crate::earcut_simple(&merged) {
            let a = b.v(lift(merged[t[0]]));
            let c = b.v(lift(merged[t[1]]));
            let d = b.v(lift(merged[t[2]]));
            if outer_signed >= 0.0 {
                b.tri(a, c, d);
            } else {
                b.tri(a, d, c);
            }
        }
    }

    // 2) Edge strips: only for selected edges. Connect the end ring at v0 to the end ring at
    //    v1 — a welded corner's shared blended arc overrides the edge's own ring there, so
    //    the two strips meeting at that corner share bit-identical end geometry.
    for (ei, e) in topo.edges.iter().enumerate() {
        if !selected[ei] {
            continue;
        }
        let ring_at = |vi: usize| -> Option<Vec<V3>> {
            if let Some(o) = ring_override.get(&(ei, vi)) {
                return Some(o.clone());
            }
            edge_end_ring(topo, e, vi, &cpt, r, seg)
        };
        let r0 = match ring_at(e.a) {
            Some(p) => p,
            None => continue,
        };
        let r1 = match ring_at(e.b) {
            Some(p) => p,
            None => continue,
        };
        let mut prev: Option<(usize, usize)> = None;
        for j in 0..=seg {
            let p0 = b.v(r0[j]);
            let p1 = b.v(r1[j]);
            if let Some((q0, q1)) = prev {
                b.tri(q0, p0, p1);
                b.tri(q0, p1, q1);
            }
            prev = Some((p0, p1));
        }
    }

    // 2.5) Sharp-edge transition wedges: a face only insets near a SELECTED edge on its OWN
    //    boundary — a face with none gets left completely alone (`cpt` returns the vertex
    //    unchanged). So at a vertex where a face's OWN corner *did* move (some OTHER, possibly
    //    distant, selected edge touches this vertex on a different face) but a SHARP edge here
    //    leads toward a face that never moved, the two faces sharing that sharp edge disagree
    //    about where their common corner sits — the sharp edge's own two endpoints are the SAME
    //    two points from each face's perspective (a==a, b==b), but its "F1 side" and "F2 side"
    //    positions can differ. That's a real crack, not just a cosmetic seam: this is exactly
    //    why every single-selected-edge fillet has silently been falling back to CSG (a genuine,
    //    pre-existing gap the corner patch alone can't close, since patching only ever runs at
    //    the vertex that moved — never propagates along the untouched edge to whichever vertex,
    //    possibly far away, doesn't). Close it here: treat the sharp edge as a trivial two-point
    //    "ring" per endpoint (the two faces' own corner positions) and stitch them exactly like
    //    a one-segment edge strip — same triangle pattern as step 2, just without any arc, since
    //    a sharp edge has none. When one end already agrees (the overwhelmingly common case —
    //    every ordinary untouched edge), that half degenerates to a zero-area triangle and
    //    `Build::finish` drops it automatically; nothing is emitted when both ends agree.
    for (ei, e) in topo.edges.iter().enumerate() {
        if selected[ei] || e.faces.len() != 2 {
            continue;
        }
        let (f1, f2) = (e.faces[0], e.faces[1]);
        // At a spliced end, the spliced face's rebuilt outline already runs through the
        // touched face's corner exactly, so both sides agree there — collapse that end to
        // the touched face's corner instead of bridging a gap that no longer exists.
        let end_ring = |vi: usize| -> [V3; 2] {
            if let Some((sf, _)) = splices.get(&vi) {
                if *sf == f1 {
                    let p = cpt(vi, f2);
                    return [p, p];
                }
                if *sf == f2 {
                    let p = cpt(vi, f1);
                    return [p, p];
                }
            }
            [cpt(vi, f1), cpt(vi, f2)]
        };
        let r0 = end_ring(e.a);
        let r1 = end_ring(e.b);
        let gap = |p: V3, q: V3| dot(sub(p, q), sub(p, q)) > 1e-14;
        if !gap(r0[0], r0[1]) && !gap(r1[0], r1[1]) {
            continue; // both ends already agree — the ordinary case, nothing to add
        }
        let (q0, q1) = (b.v(r0[0]), b.v(r1[0]));
        let (p0, p1) = (b.v(r0[1]), b.v(r1[1]));
        b.tri(q0, p0, p1);
        b.tri(q0, p1, q1);
    }

    // 3) Corner patches: at any vertex touching ≥1 selected edge, chain ALL incident edges into
    //    one boundary loop and fan it. A selected edge contributes its rounded arc; a sharp edge
    //    contributes its two face corners joined through the original vertex (the end-cap).
    for vi in 0..topo.verts.len() {
        let inc = &topo.vert_edges[vi];
        if !inc.iter().any(|&ei| selected[ei]) {
            continue; // wholly sharp vertex: untouched
        }
        if splices.contains_key(&vi) {
            continue; // terminal-edge splice: the notched face already covers this corner
        }
        if welded.contains(&vi) {
            continue; // weld corner: both strips share the blended arc — nothing to patch
        }
        let mut arcs: Vec<(usize, usize, Vec<V3>)> = Vec::new();
        for &ei in inc {
            let e = &topo.edges[ei];
            let pts = if selected[ei] {
                match edge_end_ring(topo, e, vi, &cpt, r, seg) {
                    Some(p) => p,
                    None => continue,
                }
            } else {
                // Sharp edge: just its two face corners. Where the neighbouring rounded edges
                // pulled both faces to the same point (a rim corner), these coincide and dedup
                // away; where they differ (an isolated edge) the patch can't close and we bail.
                vec![cpt(vi, e.faces[0]), cpt(vi, e.faces[1])]
            };
            arcs.push((e.faces[0], e.faces[1], pts));
        }
        if arcs.len() < 2 {
            continue;
        }
        // Walk arcs face-to-face to build the closed boundary ring of points.
        let mut used = vec![false; arcs.len()];
        let start_face = arcs[0].0;
        let mut cur_face = arcs[0].0;
        let mut boundary: Vec<V3> = Vec::new();
        let mut ok = true;
        for _ in 0..arcs.len() {
            let found = arcs.iter().enumerate().position(|(i, (f0, f1, _))| !used[i] && (*f0 == cur_face || *f1 == cur_face));
            let ai = match found {
                Some(ai) => ai,
                None => {
                    ok = false;
                    break;
                }
            };
            used[ai] = true;
            let (f0, f1, pts) = &arcs[ai];
            let (oriented, next_face): (Vec<V3>, usize) = if *f0 == cur_face {
                (pts.clone(), *f1)
            } else {
                (pts.iter().rev().cloned().collect(), *f0)
            };
            let skip = if boundary.is_empty() { 0 } else { 1 };
            boundary.extend(oriented.into_iter().skip(skip));
            cur_face = next_face;
        }
        if !ok || cur_face != start_face {
            continue;
        }
        boundary.pop(); // closing duplicate
        // Collapse consecutive duplicates (e.g. the shared original vertex from several sharp edges).
        boundary.dedup_by(|a, b| dot(sub(*a, *b), sub(*a, *b)) < 1e-14);
        if boundary.len() > 1 && dot(sub(boundary[0], boundary[boundary.len() - 1]), sub(boundary[0], boundary[boundary.len() - 1])) < 1e-14 {
            boundary.pop();
        }
        let n = boundary.len();
        if n < 3 {
            continue;
        }

        // Domed corner (Blender's tri_corner idea, adapted): a clean 3-face corner where ALL
        // incident edges are rounded (boundary is three arcs, no sharp-edge corner points).
        // The flat-centroid fan below drops its apex to the boundary's vector *average*, which
        // sits well inside the rounded volume — the visible "pinch". Instead dome it: the
        // corner's symmetry centre is the inscribed-sphere centre (equidistant from the three
        // faces), and pushing the interior out onto a sphere through the boundary's mean
        // distance gives a smoothly bulging cap that welds to the three edge arcs at its rim.
        // (Not a *perfect* rolling-ball octant — the edge arcs trace their own cylinders and
        // bulge slightly off any single corner sphere; a true octant needs the strips to stop
        // short of the vertex, a larger rework — but strictly rounder than the flat fan, and
        // watertight by construction since only the boundary ring is shared.)
        let all_rounded = arcs.iter().all(|(_, _, p)| p.len() > 2);
        let dome = (all_rounded && topo.vert_faces[vi].len() == 3)
            .then(|| {
                let fs = &topo.vert_faces[vi];
                let (n1, n2, n3) = (topo.faces[fs[0]].normal, topo.faces[fs[1]].normal, topo.faces[fs[2]].normal);
                // Prefer the sign whose centre sits on the *interior* side (nearer the boundary
                // centroid than the raw vertex is) — convex vs concave corner.
                let mut bc = [0.0; 3];
                for p in &boundary {
                    bc = add(bc, *p);
                }
                bc = scale(bc, 1.0 / boundary.len() as f64);
                [-1.0_f64, 1.0]
                    .into_iter()
                    .filter_map(|sign| corner_centre(verts[vi], n1, n2, n3, r, sign))
                    .map(|c| {
                        let rad = boundary.iter().map(|&p| dot(sub(p, c), sub(p, c)).sqrt()).sum::<f64>() / boundary.len() as f64;
                        (c, rad, dot(sub(bc, c), sub(bc, c)))
                    })
                    // The correct centre is the one the boundary is centred ON (the sphere the
                    // cap wraps around): the nearer centre→boundary-centroid. Its opposite-sign
                    // twin sits far on the other side of the part and would fling the pole out.
                    .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
                    .filter(|&(_, rad, _)| rad > 1e-6)
                    .map(|(c, rad, _)| (c, rad))
            })
            .flatten();
        if let Some((center, rad)) = dome {
            emit_sphere_dome(&mut b, &boundary, center, rad, (seg / 2).max(1));
            continue;
        }

        // Prefer fanning from a genuine point already ON the boundary — specifically
        // the one point *shared* between this vertex's two rounded (selected-edge)
        // arcs, when there are exactly two — rather than the whole loop's flat vector
        // average. The average is dragged inward by every arc simultaneously (a point
        // spread around a curved boundary averages to somewhere *inside* the volume
        // that boundary bounds), which is exactly the reported "corners aren't clipped
        // right": a visibly pinched, flat patch instead of a rounded one. A point that
        // is already part of a real rounded arc has no such problem — it's already on
        // the correct surface. This is a conservative, low-risk fix (Blender's own
        // bevel handles this exact "two rounded edges meeting through a straight/sharp
        // run" case specially too — see TODO.md): it changes *which* vertex is used as
        // the fan's centre, never invents a new position, so it can't introduce a new
        // degenerate/duplicate vertex the way earlier fuller attempts did.
        let full_arcs: Vec<usize> = (0..arcs.len()).filter(|&i| arcs[i].2.len() > 2).collect();
        let shared_apex = (full_arcs.len() == 2).then(|| (arcs[full_arcs[0]].0, arcs[full_arcs[0]].1, arcs[full_arcs[1]].0, arcs[full_arcs[1]].1)).and_then(|(a0, a1, b0, b1)| {
            let shared_face = if a0 == b0 || a0 == b1 { a0 } else if a1 == b0 || a1 == b1 { a1 } else { return None };
            let target = cpt(vi, shared_face);
            boundary.iter().position(|&p| dot(sub(p, target), sub(p, target)) < 1e-14)
        });

        match shared_apex {
            Some(m) => {
                let ring: Vec<usize> = boundary.iter().map(|&p| b.v(p)).collect();
                for step in 1..n - 1 {
                    let i = (m + step) % n;
                    let j = (m + step + 1) % n;
                    b.tri(ring[m], ring[i], ring[j]);
                }
            }
            None => {
                let mut cen = [0.0; 3];
                for p in &boundary {
                    cen = add(cen, *p);
                }
                let apex = b.v(scale(cen, 1.0 / n as f64));
                let ring: Vec<usize> = boundary.iter().map(|&p| b.v(p)).collect();
                for i in 0..n {
                    b.tri(ring[i], ring[(i + 1) % n], apex);
                }
            }
        }
    }

    let out = b.finish();
    if is_closed(&out) {
        Some(out)
    } else {
        if std::env::var("HCAD_BEVEL_DEBUG").is_ok() {
            // Report the open edges so a watertight failure is diagnosable.
            let key = |i: u32| {
                let p = out.positions[i as usize];
                ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
            };
            let mut edges: std::collections::HashMap<((i64, i64, i64), (i64, i64, i64)), i32> = std::collections::HashMap::new();
            for t in out.indices.chunks_exact(3) {
                for &(a, b2) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                    let (ka, kb) = (key(a), key(b2));
                    let e = if ka < kb { (ka, kb) } else { (kb, ka) };
                    *edges.entry(e).or_insert(0) += 1;
                }
            }
            let mut shown = 0;
            for (e, c) in &edges {
                if *c != 2 && shown < 12 {
                    eprintln!("  OPEN edge x{} at ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2})", c,
                        e.0.0 as f64 / 1e4, e.0.1 as f64 / 1e4, e.0.2 as f64 / 1e4,
                        e.1.0 as f64 / 1e4, e.1.1 as f64 / 1e4, e.1.2 as f64 / 1e4);
                    shown += 1;
                }
            }
        }
        None
    }
}

/// The **selectable feature edges** a bevel produces — see [`emit_feature_edges`]. `picked`
/// empty = every edge.
pub fn bevel_feature_edges(mesh: &TriMesh, r: f64, picked: &[Vec<[f64; 3]>]) -> Vec<[[f32; 3]; 2]> {
    match bevel_prep(mesh, r, picked) {
        Some((topo, selected, corner)) => emit_feature_edges(&topo, &selected, &corner).into_iter().map(|(e, _)| e).collect(),
        None => Vec::new(),
    }
}

/// Bevel a solid by `r` with `seg` arc segments via mesh surgery (no CSG). `picked` is the set
/// of world-space edge polylines to round; an **empty** list rounds every edge. `seg = 1` gives
/// a flat (chamfer) profile. `None` if a corner ring can't be resolved (caller → CSG).
pub fn bevel_mesh_selected(mesh: &TriMesh, r: f64, seg: usize, picked: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    let (topo, selected, corner) = bevel_prep(mesh, r, picked)?;
    run_surgery(&topo, &selected, &corner, r, seg)
}

/// Both the surgery mesh and the selectable feature edges from a **single** topology pass — what
/// the app's regen wants. The mesh is `None` when the surgery can't close (caller falls back to
/// CSG), but the edges are still returned: they sit at the same contact lines either way. Each
/// edge carries the bordering flat face's normal, so a later cut can invalidate seams whose
/// supporting surface is gone even when the cut's own wall grazes the line (see the app's clip).
pub fn bevel_mesh_and_edges(mesh: &TriMesh, r: f64, seg: usize, picked: &[Vec<[f64; 3]>]) -> (Option<TriMesh>, Vec<([[f32; 3]; 2], [f32; 3])>) {
    match bevel_prep(mesh, r, picked) {
        Some((topo, selected, corner)) => {
            let edges = emit_feature_edges(&topo, &selected, &corner);
            let mesh = run_surgery(&topo, &selected, &corner, r, seg);
            (mesh, edges)
        }
        None => (None, Vec::new()),
    }
}

/// Every welded edge shared by exactly two triangles (closed, 2-manifold).
fn is_closed(m: &TriMesh) -> bool {
    // Weld on the SAME 1e-5 grid `Build::finish` used. A coarser grid here (was 1e-4) merges
    // vertices finish kept distinct, so a sub-1e-4 crack would be papered over and an
    // actually-open surgery mesh wrongly pass as watertight.
    let key = |i: u32| {
        let p = m.positions[i as usize];
        ((p[0] * 1e5).round() as i64, (p[1] * 1e5).round() as i64, (p[2] * 1e5).round() as i64)
    };
    let mut id: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut next = 0usize;
    let mut vid = |i: u32| *id.entry(key(i)).or_insert_with(|| { next += 1; next - 1 });
    let mut edges: HashMap<(usize, usize), i32> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let v = [vid(t[0]), vid(t[1]), vid(t[2])];
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *edges.entry(if a < b { (a, b) } else { (b, a) }).or_insert(0) += 1;
        }
    }
    !edges.is_empty() && edges.values().all(|&c| c == 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{extrude_tool_mesh, PlaneBasis};

    fn xy() -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    #[test]
    fn solve3_solves_the_row_based_system() {
        // dot(m_i, x) = rhs[i] for each row — NOT "x is the combination of m0,m1,m2 summing to
        // rhs" (a different, transposed system that a naive per-axis formula can silently solve
        // instead — the bug that broke edge_end_ring's axis offset and corner_centre's sphere).
        let (m0, m1, m2) = ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]);
        let rhs = [-2.0, -2.0, 0.0];
        let x = solve3(m0, m1, m2, rhs).expect("non-degenerate axis-aligned system");
        assert!((x[0] - 0.0).abs() < 1e-9, "x={x:?}");
        assert!((x[1] - -2.0).abs() < 1e-9, "x={x:?}");
        assert!((x[2] - -2.0).abs() < 1e-9, "x={x:?}");
        for (m, r) in [(m0, rhs[0]), (m1, rhs[1]), (m2, rhs[2])] {
            assert!((dot(m, x) - r).abs() < 1e-9, "dot({m:?}, {x:?}) should be {r}, was {}", dot(m, x));
        }

        // A non-axis-aligned, less trivial case — three mutually skew-ish planes.
        let (m0, m1, m2) = ([1.0, 0.2, 0.0], [0.1, 1.0, 0.3], [0.0, -0.2, 1.0]);
        let rhs = [3.0, -1.5, 2.0];
        let x = solve3(m0, m1, m2, rhs).expect("well-conditioned system");
        for (m, r) in [(m0, rhs[0]), (m1, rhs[1]), (m2, rhs[2])] {
            assert!((dot(m, x) - r).abs() < 1e-6, "dot({m:?}, {x:?}) should be {r}, was {}", dot(m, x));
        }
    }

    #[test]
    fn box_topology_is_6_faces_12_edges_8_verts() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let topo = build_topo(&cube);
        assert_eq!(topo.verts.len(), 8, "cube has 8 corners");
        assert_eq!(topo.faces.len(), 6, "cube has 6 flat faces");
        // 12 model edges, each shared by exactly 2 faces.
        assert_eq!(topo.edges.len(), 12, "cube has 12 edges");
        assert!(topo.edges.iter().all(|e| e.faces.len() == 2), "every cube edge joins 2 faces");
        // Every corner of a cube is a 3-edge, 3-face vertex.
        assert!(topo.vert_edges.iter().all(|e| e.len() == 3), "each cube corner has 3 edges");
        assert!(topo.vert_faces.iter().all(|f| f.len() == 3), "each cube corner touches 3 faces");
        // Each face is a single 4-vertex boundary loop.
        for f in &topo.faces {
            assert_eq!(f.loops.len(), 1, "cube face has one boundary loop");
            assert_eq!(f.loops[0].len(), 4, "cube face loop has 4 corners");
        }
    }

    #[test]
    fn frame_top_face_has_two_loops() {
        // A square tube (frame prism): outer square with a square hole, extruded up. Its top
        // and bottom faces are annuli — one outer loop + one inner (hole) loop each.
        let outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole: Vec<[f64; 2]> = vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let tube = extrude_tool_mesh(&outer, &[hole], &xy(), 0.0, 5.0).unwrap();
        let topo = build_topo(&tube);
        let annuli = topo.faces.iter().filter(|f| f.loops.len() == 2).count();
        assert_eq!(annuli, 2, "top and bottom of a square tube are 2-loop annuli");
    }

    /// Signed volume (divergence theorem) — sane only for a consistently-wound closed mesh.
    fn mesh_volume(m: &TriMesh) -> f64 {
        let mut v = 0.0;
        for t in m.indices.chunks_exact(3) {
            let p: Vec<[f64; 3]> = t.iter().map(|&i| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] }).collect();
            v += p[0][0] * (p[1][1] * p[2][2] - p[1][2] * p[2][1]) - p[0][1] * (p[1][0] * p[2][2] - p[1][2] * p[2][0]) + p[0][2] * (p[1][0] * p[2][1] - p[1][1] * p[2][0]);
        }
        (v / 6.0).abs()
    }

    /// Weld output positions and confirm every edge is shared by exactly two triangles.
    fn is_watertight(m: &TriMesh) -> bool {
        let key = |i: u32| {
            let p = m.positions[i as usize];
            ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
        };
        let mut id = std::collections::HashMap::new();
        let mut next = 0usize;
        let mut vid = |i: u32| *id.entry(key(i)).or_insert_with(|| { next += 1; next - 1 });
        let mut edges: std::collections::HashMap<(usize, usize), i32> = std::collections::HashMap::new();
        for t in m.indices.chunks_exact(3) {
            let v = [vid(t[0]), vid(t[1]), vid(t[2])];
            for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
                let k = if a < b { (a, b) } else { (b, a) };
                *edges.entry(k).or_insert(0) += 1;
            }
        }
        edges.values().all(|&c| c == 2)
    }

    #[test]
    fn bevel_cube_is_watertight_and_unsharp() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let r = 0.6;
        let rounded = bevel_mesh(&cube, r, 4).expect("box bevels");
        assert!(is_watertight(&rounded), "bevelled cube must be a closed surface");
        // No vertex sits at an original sharp corner any more (each corner is now a sphere
        // whose nearest surface point is r·(1-1/√3)≈0.25 away from the corner along the diagonal).
        for corner in [[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]] {
            let min_d = rounded
                .positions
                .iter()
                .map(|p| {
                    let d = [p[0] as f64 - corner[0], p[1] as f64 - corner[1], p[2] as f64 - corner[2]];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(min_d > 0.2, "corner {corner:?} should be rounded away (min dist {min_d})");
        }
    }

    #[test]
    fn cube_corner_domes_out_instead_of_pinching_flat() {
        // A fully-rounded cube: each corner is a 3-edge convex corner. The old flat fan dropped
        // the corner apex to the boundary's vector *average* — pulled well inside the rounded
        // volume, the visible pinch. The dome (Blender's tri_corner idea) pushes the corner
        // interior out onto a cap wrapping the corner's symmetry centre (r,r,r). Concretely:
        // every corner-area vertex must stay at least ~r from that centre — the three inset
        // face corners sit at exactly r, the edge arcs bulge past it, and the domed interior
        // rides out on the fitted cap; only a collapsed flat fan puts a vertex nearer (its
        // apex sat at ~0.8·r from the centre).
        let side = 8.0;
        let sq = [[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, side).unwrap();
        let r = 1.5;
        let seg = 8;
        let rounded = bevel_mesh(&cube, r, seg).expect("cube bevels");
        assert!(is_watertight(&rounded));

        let center = [r, r, r];
        let corner_dists: Vec<f64> = rounded
            .positions
            .iter()
            // Genuinely in the origin corner's octant and inset off every flat face.
            .filter(|p| (0..3).all(|k| (p[k] as f64) < r + 1e-4) && (0..3).all(|k| (p[k] as f64) > 1e-4))
            .map(|p| {
                let d = [p[0] as f64 - center[0], p[1] as f64 - center[1], p[2] as f64 - center[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect();
        assert!(corner_dists.len() > 6, "expected a dome of corner vertices, found {}", corner_dists.len());
        let min_d = corner_dists.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_d > r * 0.9, "corner vertices should ride out on the cap (min dist {min_d:.3} from centre, r={r}) — a flat fan collapses the apex to ~0.8·r");
    }

    #[test]
    fn bevel_l_prism_with_concave_edge_is_watertight() {
        // An L-shaped prism: the reflex corner (2,2) gives one concave vertical edge whose two
        // ends are mixed corners (1 concave + 2 convex) — the same difficulty as a pocket rim.
        let l = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]];
        let prism = extrude_tool_mesh(&l, &[], &xy(), 0.0, 3.0).unwrap();
        let topo = build_topo(&prism);
        // The vertical edge at the reflex corner must be detected concave; the rest convex.
        let concave = topo.edges.iter().filter(|e| edge_sign(&topo, e) > 0).count();
        assert_eq!(concave, 1, "L-prism has exactly one concave (vertical) edge");
        let rounded = bevel_mesh(&prism, 0.4, 4).expect("L-prism bevels");
        assert!(is_watertight(&rounded), "bevelled L-prism must be a closed surface");
    }

    #[test]
    fn bevel_one_box_edge_closes_or_falls_back() {
        // An isolated edge with sharp neighbours is the hard partial-vertex (T-junction) case.
        // The engine must NOT emit a cracked mesh: either it closes, or it returns None so the
        // caller can fall back to CSG (which handles a lone edge fine).
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let picked = vec![vec![[0.0, 0.0, 4.0], [4.0, 0.0, 4.0]]];
        if let Some(m) = bevel_mesh_selected(&cube, 0.5, 4, &picked) {
            assert!(is_watertight(&m), "if it returns a mesh, it must be closed");
        }
    }

    #[test]
    fn bevel_pocket_rim_loop_selective_is_watertight() {
        // The realistic selective case: round just the rim loop of a blind pocket. Each rim
        // corner is 2 rounded edges + 1 sharp vertical edge whose two walls move together, so
        // there's no T-junction and the engine should close it.
        let outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let block = extrude_tool_mesh(&outer, &[], &xy(), 0.0, 5.0).unwrap();
        let pk = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let tool = extrude_tool_mesh(&pk, &[], &xy(), 3.0, 5.0).unwrap();
        let body = crate::mesh_difference(&block, &tool);
        // The four rim edges at z=5 around the pocket opening.
        let rim = vec![
            vec![[3.0, 3.0, 5.0], [7.0, 3.0, 5.0]],
            vec![[7.0, 3.0, 5.0], [7.0, 7.0, 5.0]],
            vec![[7.0, 7.0, 5.0], [3.0, 7.0, 5.0]],
            vec![[3.0, 7.0, 5.0], [3.0, 3.0, 5.0]],
        ];
        let rounded = bevel_mesh_selected(&body, 0.3, 3, &rim).expect("rim loop bevels");
        assert!(is_watertight(&rounded), "selective rim-loop bevel must be closed");
    }

    /// Replica of the app's reproject_plane_on_mesh: snap a +z sketch plane at `o` onto the
    /// nearest parallel body face under the origin. Returns the snapped z (or o.z if none).
    fn reproject_z(mesh: &TriMesh, o: [f64; 3]) -> f64 {
        let n = [0.0, 0.0, 1.0];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let o_n = dot(o, n);
        let p = |i: u32| { let q = mesh.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] };
        let in_tri = |ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64| {
            let s = |px: f64, py: f64, qx: f64, qy: f64, rx: f64, ry: f64| (rx - qx) * (py - qy) - (ry - qy) * (px - qx);
            let d1 = s(0.0, 0.0, ax, ay, bx, by);
            let d2 = s(0.0, 0.0, bx, by, cx, cy);
            let d3 = s(0.0, 0.0, cx, cy, ax, ay);
            !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
        };
        let mut best: Option<f64> = None;
        for t in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let tn = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let l = dot(tn, tn).sqrt();
            if l < 1e-12 || (tn[2] / l).abs() < 0.9 {
                continue;
            }
            let to2 = |q: [f64; 3]| (q[0] - o[0], q[1] - o[1]);
            let (a2, b2, c2) = (to2(a), to2(b), to2(c));
            if !in_tri(a2.0, a2.1, b2.0, b2.1, c2.0, c2.1) {
                continue;
            }
            let off = a[2];
            if best.map_or(true, |bo| (off - o_n).abs() < (bo - o_n).abs()) {
                best = Some(off);
            }
        }
        best.unwrap_or(o_n)
    }

    #[test]
    fn reproject_finds_top_face_on_bevelled_body() {
        // Block 12×12×6 with a pocket [2,6] cut from the top, rim filleted (the user's body).
        let outer = [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0]];
        let block = extrude_tool_mesh(&outer, &[], &xy(), 0.0, 6.0).unwrap();
        let pk = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let tool1 = extrude_tool_mesh(&pk, &[], &xy(), 3.0, 6.5).unwrap();
        let pocketed = crate::mesh_difference(&block, &tool1);
        let rim = vec![
            vec![[2.0, 2.0, 6.0], [6.0, 2.0, 6.0]], vec![[6.0, 2.0, 6.0], [6.0, 6.0, 6.0]],
            vec![[6.0, 6.0, 6.0], [2.0, 6.0, 6.0]], vec![[2.0, 6.0, 6.0], [2.0, 2.0, 6.0]],
        ];
        let body = bevel_mesh_selected(&pocketed, 0.4, 3, &rim).expect("rim fillets");
        // A second cut sketched on the top frame (origin at (9,9,6)) must reproject to z=6.
        let z = reproject_z(&body, [9.0, 9.0, 6.0]);
        assert!((z - 6.0).abs() < 0.01, "reproject snapped cut plane to z={z}, not the top (6)");
    }

    #[test]
    fn fillet_cylinder_boss_on_base_top_rim() {
        // The user's case: a cylinder boss unioned onto a base block, then fillet just the
        // cylinder's top rim. The union makes the topology messier than a lone cylinder.
        let base = extrude_tool_mesh(&[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], &[], &xy(), 0.0, 5.0).unwrap();
        let n = 32;
        let circ: Vec<[f64; 2]> = (0..n)
            .map(|i| { let a = std::f64::consts::TAU * i as f64 / n as f64; [10.0 + 4.0 * a.cos(), 10.0 + 4.0 * a.sin()] })
            .collect();
        let boss = extrude_tool_mesh(&circ, &[], &xy(), 4.5, 7.5).unwrap(); // offset 4.5, height 7.5 → top z=12
        let body = crate::mesh_union(&base, &boss);
        let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], 12.0]).collect();
        rim.push(rim[0]);
        let picked = vec![rim];
        // A rim on a tessellated CURVED wall used to be refused outright; the surgery handles it,
        // so it must now SUCCEED and stay watertight even with the union's messier topology.
        let m = bevel_mesh_selected(&body, 0.7, 12, &picked).expect("the surgery declined a boss rim");
        assert!(crate::mesh_bool::is_manifold(&m), "the filleted boss is not manifold");
        // The fillet band hugs the ideal torus round the boss (axis circle r 4−0.7 at z 12−0.7).
        let (r, mut worst) = (0.7f64, 0.0f64);
        for q in &m.positions {
            let (x, y, z) = (q[0] as f64 - 10.0, q[1] as f64 - 10.0, q[2] as f64);
            let d = (x * x + y * y).sqrt();
            if z < 12.0 - r - 1e-6 || d < 4.0 - r - 1e-6 {
                continue;
            }
            let (ax, ay) = ((4.0 - r) * x / d.max(1e-12), (4.0 - r) * y / d.max(1e-12));
            worst = worst.max((((x - ax).powi(2) + (y - ay).powi(2) + (z - (12.0 - r)).powi(2)).sqrt() - r).abs());
        }
        assert!(worst < r * 0.02, "the boss fillet strays {worst:.4} from the ideal torus");
        let top_edges = bevel_feature_edges(&body, 0.7, &picked);
        assert!(top_edges.len() > 30, "rim tangent edges emitted");

        // Bottom rim: the circle where the cylinder meets the box top (z=5) — an annulus face.
        let mut brim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], 5.0]).collect();
        brim.push(brim[0]);
        let bedges = bevel_feature_edges(&body, 0.5, &[brim]);
        let seg_len = |s: &[[f32; 3]; 2]| {
            let d = [s[1][0] - s[0][0], s[1][1] - s[0][1], s[1][2] - s[0][2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        let longest = bedges.iter().map(seg_len).fold(0.0f32, f32::max);
        let long_count = bedges.iter().filter(|s| seg_len(s) > 1.5).count();
        eprintln!("bottom rim: {} edges, longest={longest:.2}, long(>1.5)={long_count}", bedges.len());
        // Rim chords are ~0.8 long; a segment shooting across the box would be »1.5.
        assert_eq!(long_count, 0, "no spurious long feature edges (longest {longest:.2})");
    }

    /// A rim on a tessellated curved wall fillets the right way round whichever way it bends:
    /// convex rims lose material, concave rims gain it, and both go through the surgery.
    ///
    /// This used to assert the concave half DECLINED — there was a guard in `run_surgery` that
    /// refused exactly this shape. The guard began life refusing *any* rim on a curved wall, on the
    /// belief that per-facet strips crease into a striped band; that was disproved for convex rims
    /// (`fillet_cylinder_top_rim`) and the guard was narrowed to concave ones. It is now gone
    /// altogether: the concave case is as accurate as the convex one — see
    /// `a_concave_curved_rim_fills_the_corner`, which measures the volume, the surface and the
    /// creasing.
    #[test]
    fn a_curved_rim_fillets_the_right_way_round_whichever_way_it_bends() {
        let n = 32;
        let circ = |cx: f64, cy: f64, r: f64| -> Vec<[f64; 2]> {
            (0..n)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / n as f64;
                    [cx + r * a.cos(), cy + r * a.sin()]
                })
                .collect()
        };
        // A boss on a plate: its TOP rim is convex, its BASE rim (where it meets the plate) is
        // concave. Both run along the same tessellated cylindrical wall, so the old guard refused
        // both and the no-guard version accepted both.
        // Same body as `fillet_cylinder_boss_on_base_top_rim`, whose top rim is known to fillet.
        let plate = extrude_tool_mesh(&[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], &[], &xy(), 0.0, 5.0).unwrap();
        let boss = extrude_tool_mesh(&circ(10.0, 10.0, 4.0), &[], &xy(), 4.5, 7.5).unwrap();
        let body = crate::mesh_union(&plate, &boss);
        let ring = |z: f64| -> Vec<Vec<[f64; 3]>> {
            let mut v: Vec<[f64; 3]> = circ(10.0, 10.0, 4.0).iter().map(|p| [p[0], p[1], z]).collect();
            v.push(v[0]);
            vec![v]
        };
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        // Convex top rim (z = 12): taken, and it REMOVES material.
        let top = bevel_mesh_selected(&body, 0.7, 12, &ring(12.0)).expect("a convex curved rim must be filleted by the surgery");
        assert!(crate::mesh_bool::is_manifold(&top), "the convex rim fillet is not manifold");
        assert!(vol(&top) < vol(&body), "rounding a convex rim must remove material, not add it");

        // Concave base rim (z = 5): also taken, and it ADDS material — the fillet fills the notch
        // where the boss meets the plate instead of cutting into it.
        let base = bevel_mesh_selected(&body, 0.7, 12, &ring(5.0)).expect("a concave curved rim must be filleted by the surgery");
        assert!(crate::mesh_bool::is_manifold(&base), "the concave rim fillet is not manifold");
        assert!(vol(&base) > vol(&body), "rounding a concave rim must add material, not remove it");
        // Either way the feature edges are emitted, so the rim stays selectable.
        assert!(bevel_feature_edges(&body, 0.7, &ring(5.0)).len() > n, "the concave rim's seam edges must still be emitted");
    }

    /// The surgery fillets a CONCAVE rim on a curved wall: the circular junction where a raised
    /// feature meets the part below it (the boss-base rim from roundfilleterror{,2,3}.hcad).
    ///
    /// `run_surgery` used to decline this shape outright, and the CSG round cannot do a rim mixing
    /// arcs with corners either, so there was no way at all to round such a junction. The three
    /// measurements below are the ones that would catch the failure it was declined for — a fillet
    /// of the wrong sign (material taken OUT of the corner), a band that misses the true fillet
    /// surface, and the creasing that a striped, per-facet band shows as.
    #[test]
    fn a_concave_curved_rim_fills_the_corner() {
        let n = 32;
        let circ: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [10.0 + 4.0 * a.cos(), 10.0 + 4.0 * a.sin()]
            })
            .collect();
        let plate = extrude_tool_mesh(&[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], &[], &xy(), 0.0, 5.0).unwrap();
        let boss = extrude_tool_mesh(&circ, &[], &xy(), 4.5, 7.5).unwrap();
        let body = crate::mesh_union(&plate, &boss);
        let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], 5.0]).collect();
        rim.push(rim[0]);

        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        let r = 0.7f64;
        let m = bevel_mesh_selected(&body, r, 12, &[rim]).expect("a concave curved rim must fillet");
        assert!(crate::mesh_bool::is_manifold(&m), "the concave rim fillet is not manifold");

        // A concave fillet FILLS the corner, so the body gains the swept corner region: area
        // r²(1−π/4) at radius 4 + r(5/6−π/4)/(1−π/4) OUT from the boss wall (Pappus).
        let pi = std::f64::consts::PI;
        let area = r * r * (1.0 - pi / 4.0);
        let ubar = r * (5.0 / 6.0 - pi / 4.0) / (1.0 - pi / 4.0);
        let want = 2.0 * pi * (4.0 + ubar) * area;
        let got = vol(&m) - vol(&body);
        assert!(got > 0.0, "a concave fillet must ADD material; this removed {:.3}", -got);
        assert!((got - want).abs() < want * 0.02, "added {got:.4}, the exact answer is {want:.4}");

        // ...and the band it adds is the actual fillet surface, not just the right amount of
        // material: every vertex inside the band's box lies on the ideal concave torus (axis circle
        // radius 4+r at z=5+r, tube radius r). Measures 0.12% of r — the same fidelity the convex
        // rim holds in `fillet_cylinder_top_rim`.
        let mut worst = 0.0f64;
        for q in &m.positions {
            let (x, y, z) = (q[0] as f64 - 10.0, q[1] as f64 - 10.0, q[2] as f64);
            let d = (x * x + y * y).sqrt();
            if !(5.0 - 1e-6..=5.0 + r + 1e-6).contains(&z) || !(4.0 - 1e-6..=4.0 + r + 1e-6).contains(&d) {
                continue; // outside the fillet band's box — plain wall or plate top
            }
            let (ax, ay) = ((4.0 + r) * x / d.max(1e-12), (4.0 + r) * y / d.max(1e-12));
            worst = worst.max((((x - ax).powi(2) + (y - ay).powi(2) + (z - (5.0 + r)).powi(2)).sqrt() - r).abs());
        }
        assert!(worst < r * 0.02, "the concave band strays {worst:.4} from the ideal torus");

        // No striping: a band creased into per-facet strips shows up as a mass of sharp display
        // edges (the symptom this shape was declined for was the count trebling). A real fillet
        // does the opposite — it REPLACES the rim's own sharp ring with a smooth band, so the
        // count falls (76 → 44 here).
        let before = crate::mesh_tessellation(body.clone()).edges.len();
        let after = crate::mesh_tessellation(m.clone()).edges.len();
        assert!(after <= before, "the fillet added sharp edges ({before} → {after}) — the band is striped");
    }

    /// roundfilleterror2.hcad's shape: a flat ring (annulus 5..8, z 0..2) with a quarter-annulus
    /// sector standing on it (z 2..6.5), walls flush. The junction loop mixes two CONCAVE straight
    /// radial edges with the ring top's two CONVEX 270° arcs.
    fn ring_and_sector() -> (TriMesh, Vec<Vec<[f64; 3]>>) {
        let n = 128;
        let pt = |i: usize, rad: f64| -> [f64; 2] {
            let a = std::f64::consts::TAU * (i % n) as f64 / n as f64;
            [rad * a.cos(), rad * a.sin()]
        };
        let ring_o: Vec<[f64; 2]> = (0..n).map(|i| pt(i, 8.0)).collect();
        let ring_i: Vec<[f64; 2]> = (0..n).map(|i| pt(i, 5.0)).collect();
        let ring = extrude_tool_mesh(&ring_o, &[ring_i.iter().rev().copied().collect()], &xy(), 0.0, 2.0).unwrap();
        // Quarter sector, 0°..90°, on the same tessellation so its walls are flush with the ring's.
        let q = n / 4;
        let mut sect: Vec<[f64; 2]> = (0..=q).map(|i| pt(i, 8.0)).collect();
        sect.extend((0..=q).rev().map(|i| pt(i, 5.0)));
        let sector = extrude_tool_mesh(&sect, &[], &xy(), 2.0, 4.5).unwrap();
        let body = crate::mesh_union(&ring, &sector);
        // The junction loop at z=2: outer 270° arc, straight radial at 0°, inner 270° arc,
        // straight radial at 90° — as four picked polylines.
        let at = |i: usize, rad: f64| { let p = pt(i, rad); [p[0], p[1], 2.0] };
        let picked = vec![
            (q..=n).map(|i| at(i, 8.0)).collect::<Vec<_>>(),
            (q..=n).map(|i| at(i, 5.0)).collect::<Vec<_>>(),
            vec![at(0, 5.0), at(0, 8.0)],
            vec![at(q, 5.0), at(q, 8.0)],
        ];
        (body, picked)
    }

    /// A fillet that RUNS OUT onto a tessellated curved wall hands over to the CSG round.
    ///
    /// roundfilleterror2.hcad as saved: one concave straight radial edge under the sector, whose
    /// two ends land on the outer (r=8) and inner (r=5) cylinder walls. The surgery insets that
    /// edge's end vertices by the full radius while their neighbours on the wall are facet chords
    /// away — so it drags them across several neighbours and folds the ring-top face back over
    /// itself, stacking two coplanar sheets that z-fight into stripes down the run-out. It weighs
    /// correctly the whole time (a fold cancels itself out), which is why only an orientation test
    /// catches it. Rather than ship that, the surgery declines and `round_mesh` takes over.
    #[test]
    fn a_fillet_running_out_onto_a_curved_wall_hands_over_to_csg() {
        let (body, picked) = ring_and_sector();
        let edge = vec![picked[2].clone()]; // the straight radial edge at angle 0
        let r = 2.15f64;
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        // A lone terminal edge is one model edge — cheap for the CSG round to take (26 ms), well
        // inside `CSG_ROUND_MAX_EDGES`.
        let (_, sel, _) = bevel_prep(&body, r, &edge).expect("prep");
        assert_eq!(sel.iter().filter(|&&s| s).count(), 1, "a lone radial edge should select one model edge");
        assert!(bevel_mesh_selected(&body, r, 12, &edge).is_none(), "the surgery must decline a fillet that folds its run-out");

        let m = crate::round_mesh(&body, r, &edge).expect("the CSG round must handle a lone terminal edge");
        assert!(crate::mesh_bool::is_manifold(&m), "the CSG run-out is not manifold");
        let want = (1.0 - std::f64::consts::PI / 4.0) * r * r * 3.0;
        let got = vol(&m) - vol(&body);
        assert!((got - want).abs() < want * 0.05, "a concave fillet must add {want:.4}, this moved {got:+.4}");

        // The point of the exercise: the ring-top plane is covered ONCE. The surgery stacked 96.4
        // units of triangle where 85.4 belong; anything near that again is the fold returning.
        let mut area = 0.0f64;
        for t in m.indices.chunks_exact(3) {
            let g = |i: u32| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] };
            let (a, b2, c) = (g(t[0]), g(t[1]), g(t[2]));
            if [a, b2, c].iter().all(|p| (p[2] - 2.0).abs() < 1e-6) {
                let n = cross(sub(b2, a), sub(c, a));
                area += dot(n, n).sqrt() * 0.5;
            }
        }
        let want_area = std::f64::consts::PI * (64.0 - 25.0) * 0.75 - 3.0 * r;
        assert!(area < want_area * 1.02, "the ring-top plane holds {area:.2} of triangle where {want_area:.2} belongs — the sheets are stacked again");

        // ...and the fillet stays INSIDE the walls it runs out into. The fill is swept with flat
        // ends square to the edge, so before `trim_fill_to_walls` its far corner stood proud of
        // the cylinder that curves away from it — the endpoint displaced by the fillet's own
        // radius, 0.4 outside an r=8 wall, which is the tab the junction showed.
        // The walls are 128-gons, so a facet sits a sagitta inside the ideal circle and a fillet
        // tracking the wall faithfully reads as very slightly inside it too. Allow exactly that
        // and no more — the tab this catches stood 0.4 out, two orders above it.
        let sagitta = |rad: f64| rad * (1.0 - (std::f64::consts::PI / 128.0).cos());
        let mut worst = 0.0f64;
        for p in &m.positions {
            let d = ((p[0] as f64).powi(2) + (p[1] as f64).powi(2)).sqrt();
            worst = worst.max(d - 8.0); // the faceted wall is INSIDE r=8, so nothing may exceed it
            if p[2] as f64 > 2.0 + 1e-6 {
                worst = worst.max(5.0 - sagitta(5.0) - d); // nothing may hang into the bore
            }
        }
        assert!(worst < 1e-3, "the fillet stands {worst:.4} proud of the walls — the run-out isn't trimmed");
    }

    /// The junction loop of roundfilleterror2.hcad, which the curved-wall guard used to refuse:
    /// two CONCAVE straight radial edges and two CONVEX 270° arcs picked as one selection. Each
    /// piece must pull its own way — the arcs cutting material off the ring's rims, the straights
    /// filling the notch under the sector — so the total is the signed Pappus sum of the four.
    ///
    /// (The whole loop was once read as evidence the surgery had the sign of a concave fillet
    /// wrong, on a measurement of about −61 at r=2.2. It doesn't: the two convex arcs simply
    /// outweigh the two short straights, and −57 is what this loop is supposed to give.)
    ///
    /// KNOWN LIMIT, deliberately not asserted away: the two straight edges still run out onto the
    /// curved walls, so their four ends fold the ring-top face exactly as
    /// `a_fillet_running_out_onto_a_curved_wall_hands_over_to_csg` describes. This pick is far too
    /// big to hand to the CSG round — 194 model edges against a measured cliff at 8 — so the
    /// folded mesh is emitted rather than freezing the app. What this test pins is the part that
    /// IS right: each piece pulls its own way and the total is the signed sum. It cannot see the
    /// fold, because volume never can.
    #[test]
    fn mixed_convex_and_concave_junction_loop_fillets() {
        let (body, picked) = ring_and_sector();
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        let pi = std::f64::consts::PI;
        // Radii up to the ring's own 2.0 wall height. Past that the fillet is taller than the
        // feature it stands on and the pieces run into each other — the volume stays right, but
        // the band self-creases, which is the geometry being impossible rather than the surgery
        // going wrong.
        for &r in &[0.5f64, 1.0, 1.5, 1.9] {
            let area = r * r * (1.0 - pi / 4.0);
            let ubar = r * (5.0 / 6.0 - pi / 4.0) / (1.0 - pi / 4.0);
            // Pappus per piece: the convex arcs remove, the concave straights add.
            let outer = -1.5 * pi * (8.0 - ubar) * area;
            let inner = -1.5 * pi * (5.0 + ubar) * area;
            let straights = 2.0 * area * 3.0;
            let want = outer + inner + straights;
            let m = bevel_mesh_selected(&body, r, 12, &picked)
                .unwrap_or_else(|| panic!("r={r}: the surgery declined the mixed junction loop"));
            assert!(crate::mesh_bool::is_manifold(&m), "r={r}: the filleted junction is not manifold");
            let got = vol(&m) - vol(&body);
            assert!(
                (got - want).abs() < want.abs() * 0.02,
                "r={r}: the loop moved {got:+.4}, the signed sum of its four pieces is {want:+.4} \
                 (outer arc {outer:+.3}, inner arc {inner:+.3}, straights {straights:+.3})"
            );
        }
    }

    #[test]
    #[ignore]
    fn diag_both_radial_edges() {
        // The user's current state: ONE fillet, r=2.35, on BOTH radial edges of the sector.
        let (body, picked) = ring_and_sector();
        let edges = vec![picked[2].clone(), picked[3].clone()];
        let r = 2.35f64;
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| { let q = m.positions[i as usize]; [q[0] as f64, q[1] as f64, q[2] as f64] };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        let m = crate::round_mesh(&body, r, &edges).expect("CSG round");
        eprintln!("volume added {:+.4} (untrimmed ideal {:+.4}), manifold={}",
            vol(&m) - vol(&body), 2.0 * (1.0 - std::f64::consts::PI / 4.0) * r * r * 3.0, crate::mesh_bool::is_manifold(&m));
        // Where does it sit relative to the two walls? Anything above the ring top (z>2) must
        // stay in the annulus band 5..8.
        let (mut out, mut into_bore) = (0.0f64, 0.0f64);
        let (mut n_out, mut n_bore) = (0usize, 0usize);
        let mut worst_bore: [f64; 3] = [0.0; 3];
        for q in &m.positions {
            let p = [q[0] as f64, q[1] as f64, q[2] as f64];
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if d > 8.0 + 1e-3 { n_out += 1; out = out.max(d - 8.0); }
            if p[2] > 2.0 + 1e-6 && d < 5.0 - 1e-3 {
                n_bore += 1;
                if 5.0 - d > into_bore { into_bore = 5.0 - d; worst_bore = p; }
            }
        }
        // The bore is a 128-gon, so a fillet tracking it faithfully reads a sagitta inside the
        // ideal circle — 0.0015 at r=5. Anything materially past that is a real intrusion.
        eprintln!("outside r=8 wall: {n_out} verts, worst {out:.4}");
        eprintln!(
            "into  r=5 bore : {n_bore} verts, worst {into_bore:.4} at ({:.3},{:.3},{:.3})  [128-gon sagitta {:.4}]",
            worst_bore[0], worst_bore[1], worst_bore[2],
            5.0 * (1.0 - (std::f64::consts::PI / 128.0).cos())
        );
    }

    #[test]
    #[ignore]
    fn diag_csg_round_cost_vs_pick_size() {
        // Declining sends the caller to the CSG round. That is only a safe place to send work if
        // it finishes — so measure how it scales with the number of picked segments.
        let (body, _) = ring_and_sector();
        let n = 128;
        let pt = |i: usize, rad: f64| -> [f64; 3] {
            let a = std::f64::consts::TAU * (i % n) as f64 / n as f64;
            [rad * a.cos(), rad * a.sin(), 2.0]
        };
        for &k in &[1usize, 2, 4, 8, 16, 32] {
            let chain: Vec<[f64; 3]> = (n / 4..=n / 4 + k).map(|i| pt(i, 8.0)).collect();
            let t0 = std::time::Instant::now();
            let got = crate::round_mesh(&body, 0.5, std::slice::from_ref(&chain));
            eprintln!("{k:3} segment pick: {:>8.2?}  {}", t0.elapsed(), if got.is_some() { "OK" } else { "declined" });
            if t0.elapsed().as_secs() > 20 {
                eprintln!("    (stopping the sweep — already past any interactive budget)");
                break;
            }
        }
    }

    #[test]
    fn fillet_cylinder_top_rim() {
        // A cylinder (32-gon prism), filleted on its top rim — a rim on a tessellated CURVED wall.
        //
        // This used to assert the surgery DECLINED, on the belief that its per-facet strips crease
        // into a striped surface. It doesn't: the fillet it produces holds the ideal torus and
        // removes the exact volume, so the test now checks that fidelity instead of the refusal.
        // Refusing was costing every curved-wall fillet the good engine.
        let n = 32;
        let circ: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [5.0 + 4.0 * a.cos(), 5.0 + 4.0 * a.sin()]
            })
            .collect();
        let cyl = extrude_tool_mesh(&circ, &[], &xy(), 0.0, 6.0).unwrap();
        // Pick the whole top rim (z=6): one polyline around the circle.
        let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], 6.0]).collect();
        rim.push(rim[0]);
        let picked = vec![rim];
        let vol = |m: &TriMesh| {
            let mut v = 0.0f64;
            for t in m.indices.chunks_exact(3) {
                let g = |i: u32| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                };
                v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
            }
            v.abs()
        };
        let pi = std::f64::consts::PI;
        // Across radii from a fraction of a facet chord (0.784) to several times it.
        for r in [0.2f64, 0.5, 1.0, 2.0, 3.5] {
            let m = bevel_mesh_selected(&cyl, r, 12, &picked)
                .unwrap_or_else(|| panic!("the surgery declined a curved-wall rim at r={r}"));
            assert!(crate::mesh_bool::is_manifold(&m), "r={r}: the filleted cylinder is not manifold");

            // Every vertex of the fillet band lies on the ideal torus: axis circle of radius 4−r
            // at height 6−r, tube radius r. This is what "striped and notched" would violate.
            let mut worst = 0.0f64;
            for q in &m.positions {
                let (x, y, z) = (q[0] as f64 - 5.0, q[1] as f64 - 5.0, q[2] as f64);
                let d = (x * x + y * y).sqrt();
                if z < 6.0 - r - 1e-6 || d < 4.0 - r - 1e-6 {
                    continue; // below or inside the band — plain wall / cap
                }
                let (ax, ay) = ((4.0 - r) * x / d.max(1e-12), (4.0 - r) * y / d.max(1e-12));
                let dev = (((x - ax).powi(2) + (y - ay).powi(2) + (z - (6.0 - r)).powi(2)).sqrt() - r).abs();
                worst = worst.max(dev);
            }
            assert!(worst < r * 0.02, "r={r}: the fillet band strays {worst:.4} from the ideal torus");

            // ...and it removes the exact corner volume (Pappus): area r²(1−π/4), centroid
            // r(5/6−π/4)/(1−π/4) in from the wall. A faceted or notched band would not.
            let area = r * r * (1.0 - pi / 4.0);
            let ubar = r * (5.0 / 6.0 - pi / 4.0) / (1.0 - pi / 4.0);
            let want = 2.0 * pi * (4.0 - ubar) * area;
            let got = vol(&cyl) - vol(&m);
            assert!((got - want).abs() < want * 0.02, "r={r}: removed {got:.4}, exact answer is {want:.4}");
        }
        // Two tangent circles (top-flat↔torus and torus↔side), ~32 segments each.
        let edges = bevel_feature_edges(&cyl, 0.5, &picked);
        assert!(edges.len() > 30, "should emit a ring of tangent edges, got {}", edges.len());
    }

    #[test]
    fn bevel_is_deterministic() {
        // Rust seeds each HashMap randomly, so build_topo iterates edges/verts differently every
        // run. If any geometry decision depends on that order, the watertight outcome flips —
        // exactly the "sometimes takes, sometimes falls back" the user sees. Run the same bevel
        // many times; the result (and watertightness) must be identical every time.
        let base = extrude_tool_mesh(&[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], &[], &xy(), 0.0, 5.0).unwrap();
        let n = 32;
        let circ: Vec<[f64; 2]> = (0..n)
            .map(|i| { let a = std::f64::consts::TAU * i as f64 / n as f64; [10.0 + 4.0 * a.cos(), 10.0 + 4.0 * a.sin()] })
            .collect();
        let boss = extrude_tool_mesh(&circ, &[], &xy(), 4.5, 7.5).unwrap();
        let body = crate::mesh_union(&base, &boss);
        let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], 12.0]).collect();
        rim.push(rim[0]);
        let picked = vec![rim];
        let first = bevel_mesh_selected(&body, 0.7, 3, &picked);
        let (some0, tris0) = (first.is_some(), first.as_ref().map(|m| m.indices.len()).unwrap_or(0));
        for _ in 0..40 {
            let r = bevel_mesh_selected(&body, 0.7, 3, &picked);
            assert_eq!(r.is_some(), some0, "bevel success flipped between identical runs (non-deterministic)");
            assert_eq!(r.map(|m| m.indices.len()).unwrap_or(0), tris0, "bevel output size varies between identical runs");
        }
    }

    #[test]
    fn mesh_union_is_deterministic() {
        // If Manifold's rebuild varies run-to-run, the bevel sees a different body each regen and
        // its watertight outcome can flip. Confirm the union is stable.
        let base = extrude_tool_mesh(&[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]], &[], &xy(), 0.0, 5.0).unwrap();
        let circ: Vec<[f64; 2]> = (0..32).map(|i| { let a = std::f64::consts::TAU * i as f64 / 32.0; [10.0 + 4.0 * a.cos(), 10.0 + 4.0 * a.sin()] }).collect();
        let boss = extrude_tool_mesh(&circ, &[], &xy(), 4.5, 7.5).unwrap();
        let base0 = crate::mesh_union(&base, &boss);
        for _ in 0..20 {
            let u = crate::mesh_union(&base, &boss);
            assert_eq!(u.indices.len(), base0.indices.len(), "mesh_union triangle count varies run-to-run");
        }
    }

    #[test]
    fn cut_after_bevel_reaches_full_depth() {
        // Reproduce the "shallow cut after chamfer" report: bevel a block, then difference a
        // pocket tool from its top. The cut must reach the intended floor — i.e. the result has
        // geometry down at the pocket floor z≈2, not just a shallow nick near the top.
        let sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let block = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 6.0).unwrap();
        let beveled = bevel_mesh(&block, 0.4, 2).expect("block bevels");
        // Pocket tool z∈[2,6.5] over [3,7]² (overlap above the top).
        let pk = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let tool = extrude_tool_mesh(&pk, &[], &xy(), 2.0, 6.5).unwrap();
        let cut = crate::mesh_difference(&beveled, &tool);
        // Full-depth cut removes the 4×4×4 = 64 pocket: 597.1 − 64 ≈ 533.
        assert!((mesh_volume(&cut) - 533.0).abs() < 5.0, "cut after bevel went shallow: vol {}", mesh_volume(&cut));
    }

    #[test]
    fn cut_after_selective_rim_fillet_full_depth() {
        // The user's path: block with a pocket, fillet the pocket rim, then make another cut.
        // The rim-filleted body (concave + selective bevel) must still cut to full depth.
        let outer = [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0]];
        let block = extrude_tool_mesh(&outer, &[], &xy(), 0.0, 6.0).unwrap();
        let pk = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let tool1 = extrude_tool_mesh(&pk, &[], &xy(), 3.0, 6.5).unwrap();
        let pocketed = crate::mesh_difference(&block, &tool1);
        let rim = vec![
            vec![[2.0, 2.0, 6.0], [6.0, 2.0, 6.0]],
            vec![[6.0, 2.0, 6.0], [6.0, 6.0, 6.0]],
            vec![[6.0, 6.0, 6.0], [2.0, 6.0, 6.0]],
            vec![[2.0, 6.0, 6.0], [2.0, 2.0, 6.0]],
        ];
        let filleted = bevel_mesh_selected(&pocketed, 0.4, 3, &rim).expect("rim fillets");
        // A second pocket on the far side, full through-ish: z 2..6.5 over [8,11]².
        let pk2 = [[8.0, 8.0], [11.0, 8.0], [11.0, 11.0], [8.0, 11.0]];
        let tool2 = extrude_tool_mesh(&pk2, &[], &xy(), 2.0, 6.5).unwrap();
        let cut2 = crate::mesh_difference(&filleted, &tool2);
        // Second pocket removes 3×3×4 = 36 from the rim-filleted body.
        let expect = mesh_volume(&filleted) - 36.0;
        assert!((mesh_volume(&cut2) - expect).abs() < 6.0, "second cut went shallow: vol {} want {expect}", mesh_volume(&cut2));
    }

    #[test]
    fn bevel_csg_pocket_is_watertight() {
        // A real CSG body: a block with a blind rectangular pocket. Its rim corners are mixed
        // (2 convex + 1 concave) and its edges are subdivided by triangulation — the actual case
        // the CSG fillet notched. This is the end-to-end target.
        let outer = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let block = extrude_tool_mesh(&outer, &[], &xy(), 0.0, 5.0).unwrap();
        let pocket = [[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        // Tool fills the pocket volume z∈[3,5] so the difference cuts a real blind pocket.
        let tool = extrude_tool_mesh(&pocket, &[], &xy(), 3.0, 5.0).unwrap();
        let body = crate::mesh_difference(&block, &tool);
        let rounded = bevel_mesh(&body, 0.3, 3).expect("pocket bevels");
        assert!(is_watertight(&rounded), "bevelled CSG pocket must be a closed surface");
    }

    #[test]
    fn cube_faces_inset_by_radius() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let topo = build_topo(&cube);
        let r = 0.5;
        let sel = vec![true; topo.edges.len()];
        // Every face is a [0,4]² square at 90° edges → its inset is the [0.5,3.5]² square.
        for fi in 0..topo.faces.len() {
            let rings = inset_loops(&topo, fi, r, &sel);
            assert_eq!(rings.len(), 1);
            let ring = &rings[0];
            assert_eq!(ring.len(), 4);
            // The in-plane coordinates of the 4 inset corners span exactly [0.5, 3.5].
            // Check by collecting the two axes that vary on this face.
            for axis in 0..3 {
                let vals: Vec<f64> = ring.iter().map(|p| p[axis]).collect();
                let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if (hi - lo) > 1e-6 {
                    // a varying (in-plane) axis: must be inset to [0.5, 3.5]
                    assert!((lo - 0.5).abs() < 1e-6, "inset lo on axis {axis}: {lo}");
                    assert!((hi - 3.5).abs() < 1e-6, "inset hi on axis {axis}: {hi}");
                }
            }
        }
    }

    #[test]
    fn top_perimeter_fillet_never_bulges_past_the_walls() {
        // Regression for saved files/fillererror.hcad: filleting all 4 edges of a box's top-face
        // perimeter (two pairs of PARALLEL selected edges meeting at right angles) produced a
        // "block" — a chunk of the surgery mesh sticking out past the box's own flat side walls,
        // well inside the fillet's vertical band, not just near the rounded corners. Root cause
        // was two bugs: `solve3` solved the wrong (transposed) 3x3 system, so `edge_end_ring`'s
        // cylinder-axis anchor `o` landed off-axis; and `edge_end_ring` anchored `o` at the
        // edge's `e.a` endpoint unconditionally instead of the vertex (`vi`) actually being
        // built, which (once `solve3` was fixed) collapsed the far end's ring onto the near
        // end's position. Every vertex of the beveled result must stay within (or on) the box's
        // original bounding box — a fillet only ever removes material, never adds it.
        let (x0, x1) = (10.300506591787663, 30.300506591807387);
        let (z0, z1) = (7.451744079581647, 27.451744079596413);
        let outer = [[x0, -z0], [x1, -z0], [x1, -z1], [x0, -z1]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] };
        let dist = 10.874906539916992;
        let mesh = crate::extrude_tool_mesh(&outer, &[], &basis, 0.0, dist).unwrap();

        let radius = 4.893707752227783;
        let top_rim = vec![vec![[x1, dist, z1], [x1, dist, z0], [x0, dist, z0], [x0, dist, z1]]];
        let seg = 12;
        let beveled = bevel_mesh_selected(&mesh, radius, seg, &top_rim).expect("top-rim fillet builds");
        assert!(is_watertight(&beveled), "beveled box must stay a closed surface");

        let tol = 1.0e-4;
        for p in &beveled.positions {
            assert!(p[0] as f64 <= x1 + tol && p[0] as f64 >= x0 - tol, "x={} outside [{x0},{x1}]", p[0]);
            assert!(p[2] as f64 <= z1 + tol && p[2] as f64 >= z0 - tol, "z={} outside [{z0},{z1}] — the block bug", p[2]);
            assert!(p[1] as f64 <= dist as f64 + tol && p[1] as f64 >= -tol, "y={} outside [0,{dist}]", p[1]);
        }
    }

    #[test]
    fn weld_corner_uses_the_blended_shared_profile() {
        // A corner where exactly two rounded edges meet (a box's top-rim fillet —
        // fillererror.hcad / fillererror2 / fillererror3 loop-picks) is Blender's M_NONE
        // "weld" case: each edge's own end ring sits on its own cylinder's cross-section,
        // and the two cross-sections DISAGREE between the shared mitred endpoints — so
        // terminating each strip on its own ring leaves a twisted gap that a fanned patch
        // could only paper over (the reported "wing" fins). The correct construction
        // (bmesh_bevel.cc, mid_v3_v3v3) terminates BOTH strips on the single point-by-point
        // BLEND of the two rings, with no corner patch at all. So: every mesh vertex near
        // the corner must lie on {the blended arc} ∪ {the two strips' own geometry},
        // and the fanned patch's old apex points must not exist.
        let (x0, x1) = (10.300506591787663, 30.300506591807387);
        let (z0, z1) = (7.451744079581647, 27.451744079596413);
        let outer = [[x0, -z0], [x1, -z0], [x1, -z1], [x0, -z1]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] };
        let dist = 10.874906539916992;
        let mesh = crate::extrude_tool_mesh(&outer, &[], &basis, 0.0, dist).unwrap();
        let radius = 4.893707752227783;
        let top_rim = vec![vec![[x1, dist, z1], [x1, dist, z0], [x0, dist, z0], [x0, dist, z1]]];
        let seg = 12;

        // Independently recompute the allowed positions at the (x1, dist, z1) corner with
        // the same low-level machinery: the two edges' own rings (strip interior columns
        // still use them away from the corner), their point-by-point blend (the shared
        // terminal arc), and the sharp vertical edge's corners.
        let (topo, selected, corner) = bevel_prep(&mesh, radius, &top_rim).expect("prep succeeds");
        let cpt = |vi: usize, fi: usize| -> V3 { corner.get(&(vi, fi)).copied().unwrap_or(topo.verts[vi]) };
        let vi = topo
            .verts
            .iter()
            .position(|&v| (v[0] - x1).abs() < 1e-6 && (v[1] - dist).abs() < 1e-6 && (v[2] - z1).abs() < 1e-6)
            .expect("corner vertex exists");
        let mut rings: Vec<Vec<V3>> = Vec::new();
        let mut allowed: Vec<V3> = Vec::new();
        for &ei in &topo.vert_edges[vi] {
            let e = &topo.edges[ei];
            if selected[ei] {
                let ring = edge_end_ring(&topo, e, vi, &cpt, radius, seg).expect("selected edge has a ring");
                allowed.extend(ring.iter().copied());
                rings.push(ring);
            } else {
                allowed.push(cpt(vi, e.faces[0]));
                allowed.push(cpt(vi, e.faces[1]));
            }
        }
        assert_eq!(rings.len(), 2, "a weld corner has exactly two rounded edges");
        let d2 = |p: V3, q: V3| dot(sub(p, q), sub(p, q));
        let r2o: Vec<V3> = if d2(rings[1][0], rings[0][0]) <= d2(rings[1][seg], rings[0][0]) {
            rings[1].clone()
        } else {
            rings[1].iter().rev().copied().collect()
        };
        for (p, q) in rings[0].iter().zip(&r2o) {
            allowed.push(scale(add(*p, *q), 0.5)); // the blended shared arc
        }

        let beveled = bevel_mesh_selected(&mesh, radius, seg, &top_rim).expect("top-rim fillet builds");
        let near_corner: Vec<[f32; 3]> = beveled
            .positions
            .iter()
            .copied()
            .filter(|p| (p[0] as f64 - x1).abs() < radius && (p[1] as f64 - dist).abs() < radius && (p[2] as f64 - z1).abs() < radius)
            .collect();
        assert!(near_corner.len() > 4, "expected several corner-area vertices, found {}", near_corner.len());
        for p in &near_corner {
            let pd = [p[0] as f64, p[1] as f64, p[2] as f64];
            let closest = allowed.iter().map(|&a| dot(sub(a, pd), sub(a, pd))).fold(f64::INFINITY, f64::min).sqrt();
            assert!(closest < 1.0e-4, "corner-area vertex {pd:?} isn't ring/blend/corner geometry (closest {closest:.4}) — stray corner-patch geometry is back");
        }
    }

    #[test]
    fn single_edge_fillet_succeeds_via_surgery_on_every_box_edge() {
        // Regression for saved files/fillererror3.hcad: filleting a SINGLE edge of a box
        // always declined the surgery path (silently falling back to CSG, which produced
        // a self-intersecting "bowtie" mesh for this exact box+radius) — for every edge, on
        // any box, at any size, since bevel.rs was first introduced. Root cause: a face
        // only insets near a SELECTED edge on its own boundary; a face with no selected
        // edge anywhere on it is left completely alone. So at a vertex where some OTHER,
        // possibly distant, selected edge moved a DIFFERENT face's corner, a SHARP edge
        // here leading toward the untouched face has two different endpoint positions
        // depending which of its two faces you ask — a real crack, not just a seam. The
        // corner patch alone can't fix this: it only ever runs at the vertex that moved,
        // never propagates along the untouched sharp edge to its other end. Fixed by
        // treating every sharp edge with a real gap at either end as a trivial one-segment
        // "edge strip" (its two faces' own corner positions, no arc) stitched exactly like
        // a rounded edge's strip — using only already-computed positions, so it can't
        // introduce a new, badly-placed vertex.
        let (x0, x1) = (0.0_f64, 40.0);
        let (z0, z1) = (0.0_f64, 40.0);
        let dist = 26.0_f64;
        let outer = [[x0, -z0], [x1, -z0], [x1, -z1], [x0, -z1]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] };
        let mesh = crate::extrude_tool_mesh(&outer, &[], &basis, 0.0, dist).unwrap();
        let r = 5.0;
        let seg = 8;

        let corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)];
        let mut edges: Vec<Vec<[f64; 3]>> = Vec::new();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            edges.push(vec![[a.0, 0.0, a.1], [b.0, 0.0, b.1]]);
            edges.push(vec![[a.0, dist, a.1], [b.0, dist, b.1]]);
        }
        for &(cx, cz) in &corners {
            edges.push(vec![[cx, 0.0, cz], [cx, dist, cz]]);
        }
        assert_eq!(edges.len(), 12, "sanity: a box has 12 edges");
        let box_vol = (x1 - x0) * (z1 - z0) * dist;
        for chain in &edges {
            let picked = vec![chain.clone()];
            let beveled = bevel_mesh_selected(&mesh, r, seg, &picked);
            assert!(beveled.is_some(), "single-edge fillet declined surgery for {chain:?}");
            let beveled = beveled.unwrap();
            assert!(is_watertight(&beveled), "single-edge fillet result not watertight for {chain:?}");

            // The filleted edge's two END CORNERS must be gone from the mesh: the untouched
            // side face at each end gets the quarter-arc NOTCH spliced into its outline
            // (replacing that corner), so nothing references the original corner any more.
            // The old membrane approach kept the corner AND laid a coplanar patch over the
            // same region — topologically closed but visibly self-clipping at the ends.
            for end in [&chain[0], &chain[chain.len() - 1]] {
                let stale = beveled.positions.iter().any(|p| {
                    (p[0] as f64 - end[0]).abs() < 1e-6
                        && (p[1] as f64 - end[1]).abs() < 1e-6
                        && (p[2] as f64 - end[2]).abs() < 1e-6
                });
                assert!(!stale, "original corner {end:?} still present — the coplanar double-cover is back");
            }

            // Volume: exactly a quarter-round groove removed along the full edge length
            // (the polygonal arc under-fills the true circle slightly; 2% covers it).
            let edge_len = {
                let (a, b) = (chain[0], chain[chain.len() - 1]);
                dot(sub(b, a), sub(b, a)).sqrt()
            };
            let removed = (1.0 - std::f64::consts::PI / 4.0) * r * r * edge_len;
            let vol = mesh_volume(&beveled);
            assert!(
                (vol - (box_vol - removed)).abs() < removed * 0.05,
                "volume {vol:.2} should be box {box_vol:.2} minus groove {removed:.2} for {chain:?}"
            );
        }
    }

    #[test]
    fn single_edge_fillet_on_fillererror3_box() {
        // The exact geometry from saved files/fillererror3.hcad.
        let (x0, x1) = (0.0_f64, 5.708608627319336);
        let (z0, z1) = (0.0_f64, 3.6159682273864746);
        let dist = 2.903108596801758_f64;
        let outer = [[x0, -z0], [x1, -z0], [x1, -z1], [x0, -z1]];
        let basis = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] };
        let mesh = crate::extrude_tool_mesh(&outer, &[], &basis, 0.0, dist).unwrap();
        let r = 1.3064;
        let seg = 8;
        let picked = vec![vec![[x0, dist, z0], [x1, dist, z0]]];
        let beveled = bevel_mesh_selected(&mesh, r, seg, &picked).expect("fillererror3's box+radius should surgery-bevel");
        assert!(is_watertight(&beveled));
    }
}
