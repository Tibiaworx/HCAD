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
pub fn inset_loops(topo: &Topo, fi: usize, r: f64) -> Vec<Vec<V3>> {
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
            ring.push(line_intersect(p_in, dir[prev], p_out, dir[i], n).unwrap_or(v));
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

/// Solve the 3×3 system `M x = rhs` by Cramer's rule (rows of `M` are `m0,m1,m2`).
fn solve3(m0: V3, m1: V3, m2: V3, rhs: V3) -> Option<V3> {
    let det = dot(m0, cross(m1, m2));
    if det.abs() < 1e-9 {
        return None;
    }
    let dx = dot(rhs, cross(m1, m2));
    let dy = dot(m0, cross(rhs, m2));
    let dz = dot(m0, cross(m1, rhs));
    Some([dx / det, dy / det, dz / det])
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
fn edge_end_ring(topo: &Topo, e: &TopoEdge, vi: usize, cpt: &dyn Fn(usize, usize) -> V3, r: f64, seg: usize) -> Option<Vec<V3>> {
    let (fa, fb) = (e.faces[0], e.faces[1]);
    let (na, nb) = (topo.faces[fa].normal, topo.faces[fb].normal);
    let sigma = edge_sign(topo, e) as f64;
    if sigma == 0.0 {
        return None;
    }
    let t = norm(sub(topo.verts[e.b], topo.verts[e.a]));
    // Axis line of the fillet cylinder: at signed distance σ·r from both faces, parallel to t.
    let off = solve3(na, nb, t, [sigma * r, sigma * r, 0.0])?;
    let o = add(topo.verts[e.a], off);
    // Radial direction + along-axis coordinate of each face's corner point at this vertex.
    let radial = |p: V3| -> (V3, f64) {
        let along = dot(sub(p, o), t);
        let rp = sub(sub(p, o), scale(t, along));
        (norm(rp), along)
    };
    let (ca, cb) = (cpt(vi, fa), cpt(vi, fb));
    let (da, aa) = radial(ca);
    let (db, ab) = radial(cb);
    let mut pts = Vec::with_capacity(seg + 1);
    for j in 0..=seg {
        // Pin the two ends to the exact corner points so the flat-face seam can't crack; only
        // interior arc points come from the cylinder formula (cylinder & patch share these).
        if j == 0 {
            pts.push(ca);
        } else if j == seg {
            pts.push(cb);
        } else {
            let p = j as f64 / seg as f64;
            let along = aa + (ab - aa) * p;
            let d = slerp(da, db, p);
            pts.push(add(add(o, scale(t, along)), scale(d, r)));
        }
    }
    Some(pts)
}

/// Prototype bevel: round **every** edge of a solid by `r` with `seg` arc segments, by mesh
/// surgery instead of CSG. Flat faces inset by the rolling-ball setback (keeping their original
/// triangulation, so non-convex faces and holes are fine); each edge becomes a cylinder strip
/// that bulges outward (convex) or fills inward (concave); each corner is a patch stitched to
/// the incident edge rings. No booleans, so corners carry no facets/dimples. Returns `None` on
/// a body this prototype can't resolve (a degenerate edge, or a corner whose ring won't close).
pub fn bevel_mesh(mesh: &TriMesh, r: f64, seg: usize) -> Option<TriMesh> {
    let topo = build_topo(mesh);

    // corner(v, f): where face f's boundary insets to at vertex v (rolling-ball setback). The
    // single source of truth for flat faces, edge strips and corner patches → guaranteed shared.
    let mut corner: HashMap<(usize, usize), V3> = HashMap::new();
    for fi in 0..topo.faces.len() {
        let rings = inset_loops(&topo, fi, r);
        for (li, lp) in topo.faces[fi].loops.iter().enumerate() {
            for (k, &vi) in lp.iter().enumerate() {
                corner.insert((vi, fi), rings[li][k]);
            }
        }
    }
    let verts = &topo.verts;
    let cpt = |vi: usize, fi: usize| -> V3 { corner.get(&(vi, fi)).copied().unwrap_or(verts[vi]) };

    let mut b = Build::new();

    // 1) Flat faces: reuse the original triangulation with each vertex moved to its inset corner
    //    (interior vertices, if any, stay put). Robust for non-convex faces and faces with holes.
    for fi in 0..topo.faces.len() {
        for &ti in &topo.faces[fi].tris {
            let t = topo.tris[ti];
            let a = b.v(cpt(t[0], fi));
            let c = b.v(cpt(t[1], fi));
            let d = b.v(cpt(t[2], fi));
            b.tri(a, c, d);
        }
    }

    // 2) Edge strips: connect the edge's end ring at v0 to its end ring at v1.
    for e in &topo.edges {
        let r0 = match edge_end_ring(&topo, e, e.a, &cpt, r, seg) {
            Some(p) => p,
            None => continue, // flat edge: nothing to round
        };
        let r1 = edge_end_ring(&topo, e, e.b, &cpt, r, seg)?;
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

    // 3) Corner patches: chain the incident edge rings into one boundary loop around v, fan it.
    for vi in 0..topo.verts.len() {
        let inc = &topo.vert_edges[vi];
        if inc.len() < 3 {
            continue; // mid-edge or boundary vertex: the abutting strips already share the ring
        }
        // Each incident bevelled edge gives an arc from its face-A corner to its face-B corner.
        let mut arcs: Vec<(usize, usize, Vec<V3>)> = Vec::new();
        for &ei in inc {
            let e = &topo.edges[ei];
            if let Some(pts) = edge_end_ring(&topo, e, vi, &cpt, r, seg) {
                arcs.push((e.faces[0], e.faces[1], pts));
            }
        }
        if arcs.len() < 3 {
            continue;
        }
        // Walk arcs face-to-face to build the closed boundary ring of points.
        let mut used = vec![false; arcs.len()];
        let start_face = arcs[0].0;
        let mut cur_face = arcs[0].0;
        let mut boundary: Vec<V3> = Vec::new();
        let mut ok = true;
        for _ in 0..arcs.len() {
            // Find an unused arc touching cur_face.
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
            // Orient the arc to start at cur_face.
            let (oriented, next_face): (Vec<V3>, usize) = if *f0 == cur_face {
                (pts.clone(), *f1)
            } else {
                (pts.iter().rev().cloned().collect(), *f0)
            };
            // Append, skipping the shared first point (== boundary's last).
            let skip = if boundary.is_empty() { 0 } else { 1 };
            boundary.extend(oriented.into_iter().skip(skip));
            cur_face = next_face;
        }
        if !ok || cur_face != start_face || boundary.len() < 3 {
            continue;
        }
        // Drop the closing duplicate (last == first), then fan from the centroid.
        boundary.pop();
        let n = boundary.len();
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

    Some(b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{extrude_tool_mesh, PlaneBasis};

    fn xy() -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
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
    fn cube_faces_inset_by_radius() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let cube = extrude_tool_mesh(&sq, &[], &xy(), 0.0, 4.0).unwrap();
        let topo = build_topo(&cube);
        let r = 0.5;
        // Every face is a [0,4]² square at 90° edges → its inset is the [0.5,3.5]² square.
        for fi in 0..topo.faces.len() {
            let rings = inset_loops(&topo, fi, r);
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
}
