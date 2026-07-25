//! A self-contained BSP-tree mesh boolean (a Rust port of the proven `csg.js`
//! algorithm). It operates on triangle meshes, not B-reps, so — unlike truck's
//! exact boolean — it copes with **coincident/coplanar faces** (a boss sitting on
//! the floor of a cut, walls flush with a pocket, etc.). We use it only as a
//! *fallback* when the exact kernel rejects a boolean, so the operation still
//! applies instead of silently vanishing.
//!
//! The algorithm: each solid becomes a set of convex polygons; a BSP tree built
//! from one solid's polygons is used to clip the other's, classifying every piece
//! as inside/outside. Union/difference/intersection are then a fixed sequence of
//! clip + invert operations (see `Node`).

use crate::TriMesh;

/// Polygons closer than this (in world units) to a splitting plane count as lying
/// *on* it. Our models are tens of units, so 1e-5 is comfortably sub-facet.
const EPSILON: f64 = 1e-5;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt();
    if l > 1e-12 {
        scale(a, 1.0 / l)
    } else {
        [0.0, 0.0, 0.0]
    }
}
fn lerp(a: V3, b: V3, t: f64) -> V3 {
    add(a, scale(sub(b, a), t))
}

#[derive(Clone, Copy)]
struct Vertex {
    pos: V3,
}

#[derive(Clone, Copy)]
struct Plane {
    normal: V3,
    w: f64,
}

impl Plane {
    fn from_points(a: V3, b: V3, c: V3) -> Option<Plane> {
        let n = norm(cross(sub(b, a), sub(c, a)));
        if n == [0.0, 0.0, 0.0] {
            return None; // degenerate triangle
        }
        Some(Plane { normal: n, w: dot(n, a) })
    }
    fn flip(&mut self) {
        self.normal = scale(self.normal, -1.0);
        self.w = -self.w;
    }
}

#[derive(Clone)]
struct Polygon {
    vertices: Vec<Vertex>,
    plane: Plane,
}

impl Polygon {
    fn new(vertices: Vec<Vertex>) -> Option<Polygon> {
        let plane = Plane::from_points(vertices[0].pos, vertices[1].pos, vertices[2].pos)?;
        Some(Polygon { vertices, plane })
    }
    fn flip(&mut self) {
        self.vertices.reverse();
        self.plane.flip();
    }
}

// Classification of a point/polygon relative to a plane.
const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// Split `polygon` by `plane`, appending the pieces to the four buckets. Coplanar
/// polygons go to coplanar-front/back by normal alignment (this is what lets a
/// flush face survive a boolean instead of crashing it).
fn split_polygon(
    plane: &Plane,
    polygon: &Polygon,
    coplanar_front: &mut Vec<Polygon>,
    coplanar_back: &mut Vec<Polygon>,
    front: &mut Vec<Polygon>,
    back: &mut Vec<Polygon>,
) {
    let mut polygon_type = 0u8;
    let mut types: Vec<u8> = Vec::with_capacity(polygon.vertices.len());
    for v in &polygon.vertices {
        let t = dot(plane.normal, v.pos) - plane.w;
        let kind = if t < -EPSILON {
            BACK
        } else if t > EPSILON {
            FRONT
        } else {
            COPLANAR
        };
        polygon_type |= kind;
        types.push(kind);
    }

    match polygon_type {
        COPLANAR => {
            if dot(plane.normal, polygon.plane.normal) > 0.0 {
                coplanar_front.push(polygon.clone());
            } else {
                coplanar_back.push(polygon.clone());
            }
        }
        FRONT => front.push(polygon.clone()),
        BACK => back.push(polygon.clone()),
        _ => {
            // SPANNING: cut the polygon into a front part and a back part.
            let (mut f, mut b): (Vec<Vertex>, Vec<Vertex>) = (Vec::new(), Vec::new());
            let n = polygon.vertices.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (ti, tj) = (types[i], types[j]);
                let (vi, vj) = (polygon.vertices[i], polygon.vertices[j]);
                if ti != BACK {
                    f.push(vi);
                }
                if ti != FRONT {
                    b.push(vi);
                }
                if (ti | tj) == SPANNING {
                    let t = (plane.w - dot(plane.normal, vi.pos))
                        / dot(plane.normal, sub(vj.pos, vi.pos));
                    let v = Vertex { pos: lerp(vi.pos, vj.pos, t) };
                    f.push(v);
                    b.push(v);
                }
            }
            if f.len() >= 3 {
                if let Some(p) = Polygon::new(f) {
                    front.push(p);
                }
            }
            if b.len() >= 3 {
                if let Some(p) = Polygon::new(b) {
                    back.push(p);
                }
            }
        }
    }
}

/// A node of the BSP tree.
struct Node {
    plane: Option<Plane>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
    polygons: Vec<Polygon>,
}

impl Node {
    fn new() -> Node {
        Node { plane: None, front: None, back: None, polygons: Vec::new() }
    }

    fn invert(&mut self) {
        for p in &mut self.polygons {
            p.flip();
        }
        if let Some(pl) = &mut self.plane {
            pl.flip();
        }
        if let Some(f) = &mut self.front {
            f.invert();
        }
        if let Some(b) = &mut self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Remove all parts of `polygons` that are inside this BSP tree.
    fn clip_polygons(&self, polygons: Vec<Polygon>) -> Vec<Polygon> {
        let Some(plane) = self.plane else {
            return polygons;
        };
        // Coplanar-front pieces join `front`, coplanar-back join `back` (csg.js).
        let (mut cf, mut cb, mut front, mut back) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for p in &polygons {
            split_polygon(&plane, p, &mut cf, &mut cb, &mut front, &mut back);
        }
        front.extend(cf);
        back.extend(cb);
        let mut front = match &self.front {
            Some(n) => n.clip_polygons(front),
            None => front,
        };
        let back = match &self.back {
            Some(n) => n.clip_polygons(back),
            None => Vec::new(), // no back tree → everything behind is inside → discard
        };
        front.extend(back);
        front
    }

    /// Clip this tree's polygons to `other`'s solid.
    fn clip_to(&mut self, other: &Node) {
        self.polygons = other.clip_polygons(std::mem::take(&mut self.polygons));
        if let Some(f) = &mut self.front {
            f.clip_to(other);
        }
        if let Some(b) = &mut self.back {
            b.clip_to(other);
        }
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        let mut out = self.polygons.clone();
        if let Some(f) = &self.front {
            out.extend(f.all_polygons());
        }
        if let Some(b) = &self.back {
            out.extend(b.all_polygons());
        }
        out
    }

    /// Build (or extend) the tree from a set of polygons, using the first as the
    /// splitting plane. Iterative front/back accumulation, recursive descent.
    fn build(&mut self, polygons: Vec<Polygon>) {
        self.build_depth(polygons, 0);
    }

    /// `build` picks `polygons[0]`'s plane as the split (no balancing), so a near-coplanar
    /// fan produces an almost-linear tree whose depth ≈ polygon count — 20k-deep recursion
    /// (each frame allocating Vecs) can blow the stack. Cap the depth: past the limit, keep
    /// the remaining polygons as this node's own coplanar set rather than recursing. The BSP
    /// is the lossy CSG fallback already, so a slightly coarser leaf is acceptable next to a
    /// crash.
    fn build_depth(&mut self, polygons: Vec<Polygon>, depth: u32) {
        const MAX_DEPTH: u32 = 3000;
        if polygons.is_empty() {
            return;
        }
        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane);
        }
        let plane = self.plane.unwrap();
        // Coplanar pieces (either orientation) belong to this node's own polygons.
        let (mut cf, mut cb, mut front, mut back) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for p in &polygons {
            split_polygon(&plane, p, &mut cf, &mut cb, &mut front, &mut back);
        }
        self.polygons.extend(cf);
        self.polygons.extend(cb);
        if depth >= MAX_DEPTH {
            // Bail out flat: stop splitting, keep the rest here.
            self.polygons.extend(front);
            self.polygons.extend(back);
            return;
        }
        if !front.is_empty() {
            self.front.get_or_insert_with(|| Box::new(Node::new())).build_depth(front, depth + 1);
        }
        if !back.is_empty() {
            self.back.get_or_insert_with(|| Box::new(Node::new())).build_depth(back, depth + 1);
        }
    }
}

fn node_from(polygons: Vec<Polygon>) -> Node {
    let mut n = Node::new();
    n.build(polygons);
    n
}

/// `a ∪ b` on polygon soups (csg.js union).
fn union_polys(a: Vec<Polygon>, b: Vec<Polygon>) -> Vec<Polygon> {
    let mut na = node_from(a);
    let mut nb = node_from(b);
    na.clip_to(&nb);
    nb.clip_to(&na);
    nb.invert();
    nb.clip_to(&na);
    nb.invert();
    na.build(nb.all_polygons());
    na.all_polygons()
}

/// `a − b` on polygon soups (csg.js subtract).
fn subtract_polys(a: Vec<Polygon>, b: Vec<Polygon>) -> Vec<Polygon> {
    let mut na = node_from(a);
    let mut nb = node_from(b);
    na.invert();
    na.clip_to(&nb);
    nb.clip_to(&na);
    nb.invert();
    nb.clip_to(&na);
    nb.invert();
    na.build(nb.all_polygons());
    na.invert();
    na.all_polygons()
}

/// Triangle mesh → convex polygon soup (one triangle per polygon).
fn mesh_to_polys(m: &TriMesh) -> Vec<Polygon> {
    let mut out = Vec::with_capacity(m.indices.len() / 3);
    for tri in m.indices.chunks_exact(3) {
        let v = |i: u32| {
            let p = m.positions[i as usize];
            Vertex { pos: [p[0] as f64, p[1] as f64, p[2] as f64] }
        };
        if let Some(p) = Polygon::new(vec![v(tri[0]), v(tri[1]), v(tri[2])]) {
            out.push(p);
        }
    }
    out
}

/// Polygon soup → triangle mesh (fan-triangulate, flat per-face normals).
fn polys_to_mesh(polys: &[Polygon]) -> TriMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for p in polys {
        if p.vertices.len() < 3 {
            continue;
        }
        let n = [p.plane.normal[0] as f32, p.plane.normal[1] as f32, p.plane.normal[2] as f32];
        let base = positions.len() as u32;
        for v in &p.vertices {
            positions.push([v.pos[0] as f32, v.pos[1] as f32, v.pos[2] as f32]);
            normals.push(n);
        }
        for k in 1..(p.vertices.len() as u32 - 1) {
            indices.push(base);
            indices.push(base + k);
            indices.push(base + k + 1);
        }
    }
    TriMesh { positions, normals, indices }
}

/// Boolean **union** of two triangle meshes (BSP fallback — see [`crate::mesh_union`]).
pub(crate) fn bsp_union(a: &TriMesh, b: &TriMesh) -> TriMesh {
    polys_to_mesh(&union_polys(mesh_to_polys(a), mesh_to_polys(b)))
}

/// Boolean **difference** `a − b` (BSP fallback — see [`crate::mesh_difference`]).
pub(crate) fn bsp_difference(a: &TriMesh, b: &TriMesh) -> TriMesh {
    polys_to_mesh(&subtract_polys(mesh_to_polys(a), mesh_to_polys(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box mesh from `lo`..`hi` (12 triangles, outward normals).
    fn box_mesh(lo: [f32; 3], hi: [f32; 3]) -> TriMesh {
        let c = [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ];
        // Faces wound CCW seen from outside.
        let faces = [
            [0, 3, 2, 1], // -z
            [4, 5, 6, 7], // +z
            [0, 1, 5, 4], // -y
            [2, 3, 7, 6], // +y
            [1, 2, 6, 5], // +x
            [0, 4, 7, 3], // -x
        ];
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for f in faces {
            let base = positions.len() as u32;
            for &i in &f {
                positions.push(c[i]);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let normals = vec![[0.0, 0.0, 1.0]; positions.len()];
        TriMesh { positions, normals, indices }
    }

    fn volume(m: &TriMesh) -> f64 {
        // Signed volume via the divergence theorem (sum of tetra from origin).
        let mut v = 0.0;
        for t in m.indices.chunks_exact(3) {
            let p: Vec<V3> = t
                .iter()
                .map(|&i| {
                    let q = m.positions[i as usize];
                    [q[0] as f64, q[1] as f64, q[2] as f64]
                })
                .collect();
            v += dot(p[0], cross(p[1], p[2])) / 6.0;
        }
        v.abs()
    }

    #[test]
    fn union_of_two_overlapping_unit_cubes() {
        // Cubes overlapping by half in x → combined volume 1 + 1 − 0.5 = 1.5.
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_mesh([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);
        let u = bsp_union(&a, &b);
        assert!(!u.indices.is_empty(), "union produced no geometry");
        assert!((volume(&u) - 1.5).abs() < 1e-3, "union volume was {}", volume(&u));
    }

    #[test]
    fn difference_carves_a_notch() {
        // A unit cube minus a half-overlapping cube → volume 1 − 0.5 = 0.5.
        let a = box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = box_mesh([0.5, 0.25, 0.25], [1.5, 0.75, 0.75]);
        let d = bsp_difference(&a, &b);
        assert!((volume(&d) - (1.0 - 0.5 * 0.5 * 0.5)).abs() < 1e-3, "difference volume was {}", volume(&d));
    }

    #[test]
    fn boss_flush_on_a_floor_unions_coplanar_faces() {
        // The exact case truck chokes on: a small box sitting *on* a big box's top
        // face (shared coplanar plane at z=1). Mesh CSG must still union them.
        let body = box_mesh([0.0, 0.0, 0.0], [4.0, 4.0, 1.0]);
        // Boss overlaps slightly into the body so it's a real union (z 0.9..3).
        let boss = box_mesh([1.0, 1.0, 0.9], [3.0, 3.0, 3.0]);
        let u = bsp_union(&body, &boss);
        let expect = 4.0 * 4.0 * 1.0 + 2.0 * 2.0 * 2.1 - 2.0 * 2.0 * 0.1;
        assert!((volume(&u) - expect).abs() < 1e-2, "flush-boss union volume was {} (want {expect})", volume(&u));
    }
}
