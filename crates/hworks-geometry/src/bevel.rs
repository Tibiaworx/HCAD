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
use std::collections::HashMap;

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
                if dir.get(&cur).map_or(true, |v| v.is_empty()) {
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

/// The rolling-ball centre for a 3-face convex corner: the point at signed distance `-r` from
/// each incident face plane (through `v`, outward normal `nf`). It lies on all three incident
/// edge cylinders' axes, so the corner sphere blends smoothly into each. `None` if degenerate.
fn corner_centre(v: V3, na: V3, nb: V3, nc: V3, r: f64) -> Option<V3> {
    let x = solve3(na, nb, nc, [-r, -r, -r])?;
    Some(add(v, x))
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

/// The surgery half of a prepared bevel: flat faces inset to their corners, a convex/concave
/// cylinder strip per selected edge, and a fanned patch per corner. `None` if the result isn't
/// watertight (a partial-vertex T-junction the prototype can't split → caller falls back to CSG).
fn run_surgery(topo: &Topo, selected: &[bool], corner: &HashMap<(usize, usize), V3>, r: f64, seg: usize) -> Option<TriMesh> {
    // The surgery insets flat faces and stitches arc strips + corner patches — sound on real
    // (flat-face) corners, but on a TESSELLATED CURVED wall (a chain of near-tangent micro-facets,
    // e.g. a cylinder rim) the per-facet strips and corner patches crease into sliver geometry
    // that still closes watertight — a striped, notched surface. Detect that regime and decline:
    // a selected edge sharing an endpoint with a near-tangent (facet-seam) edge is running across
    // a curved wall, and the CSG round handles those smoothly.
    let cos_facet = 20.0_f64.to_radians().cos();
    for (ei, e) in topo.edges.iter().enumerate() {
        if !selected[ei] {
            continue;
        }
        for &v in &[e.a, e.b] {
            for &oi in &topo.vert_edges[v] {
                if oi == ei || selected[oi] {
                    continue;
                }
                let o = &topo.edges[oi];
                if o.faces.len() == 2
                    && dot(topo.faces[o.faces[0]].normal, topo.faces[o.faces[1]].normal) > cos_facet
                {
                    return None;
                }
            }
        }
    }
    let verts = &topo.verts;
    let cpt = |vi: usize, fi: usize| -> V3 { corner.get(&(vi, fi)).copied().unwrap_or(verts[vi]) };
    let mut b = Build::new();

    // 1) Flat faces: reuse the original triangulation with each vertex moved to its inset corner.
    for fi in 0..topo.faces.len() {
        for &ti in &topo.faces[fi].tris {
            let t = topo.tris[ti];
            let a = b.v(cpt(t[0], fi));
            let c = b.v(cpt(t[1], fi));
            let d = b.v(cpt(t[2], fi));
            b.tri(a, c, d);
        }
    }

    // 2) Edge strips: only for selected edges. Connect the end ring at v0 to the end ring at v1.
    for (ei, e) in topo.edges.iter().enumerate() {
        if !selected[ei] {
            continue;
        }
        let r0 = match edge_end_ring(topo, e, e.a, &cpt, r, seg) {
            Some(p) => p,
            None => continue,
        };
        let r1 = match edge_end_ring(topo, e, e.b, &cpt, r, seg) {
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

    // 3) Corner patches: at any vertex touching ≥1 selected edge, chain ALL incident edges into
    //    one boundary loop and fan it. A selected edge contributes its rounded arc; a sharp edge
    //    contributes its two face corners joined through the original vertex (the end-cap).
    for vi in 0..topo.verts.len() {
        let inc = &topo.vert_edges[vi];
        if !inc.iter().any(|&ei| selected[ei]) {
            continue; // wholly sharp vertex: untouched
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

    let out = b.finish();
    if is_closed(&out) {
        Some(out)
    } else {
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
    let key = |i: u32| {
        let p = m.positions[i as usize];
        ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64)
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
        // A rim on a tessellated CURVED wall must DECLINE the mesh surgery (its per-facet strips
        // crease into a striped, notched surface) — the caller then uses the smooth CSG round.
        // Either way the topology-only feature edges are emitted so the rounded rim is selectable.
        assert!(
            bevel_mesh_selected(&body, 0.7, 3, &picked).is_none(),
            "curved-wall rim must decline surgery (creased strips) and fall back to CSG"
        );
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

    #[test]
    fn fillet_cylinder_top_rim() {
        // A cylinder (32-gon prism). A rim on a tessellated curved wall must DECLINE the mesh
        // surgery (per-facet strips crease into a striped/notched surface) so the caller uses the
        // smooth CSG round — but the feature edges are still emitted so the rim stays selectable.
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
        assert!(
            bevel_mesh_selected(&cyl, 0.5, 3, &picked).is_none(),
            "curved-wall rim must decline surgery and fall back to CSG"
        );
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
}
