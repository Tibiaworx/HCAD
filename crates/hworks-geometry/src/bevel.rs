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

/// A flat face: the welded triangles that are coplanar and edge-connected, plus a normal.
#[derive(Debug, Clone)]
pub struct Face {
    pub tris: Vec<usize>,
    pub normal: V3,
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
    let tris: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]])
        .collect();

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
            faces.push(Face { tris: Vec::new(), normal: [0.0, 0.0, 0.0] });
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

    Topo { verts, tris, faces, tri_face, edges }
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
    }
}
