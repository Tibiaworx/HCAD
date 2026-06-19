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
fn weld(m: &TriMesh) -> (Vec<f32>, Vec<u32>) {
    // 1e-5 grid: truck emits *identical* f32 values for a shared vertex, so this only
    // ever merges true duplicates, never distinct geometry (models are tens of units).
    let key = |p: [f32; 3]| {
        ((p[0] * 1.0e5).round() as i64, (p[1] * 1.0e5).round() as i64, (p[2] * 1.0e5).round() as i64)
    };
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut props: Vec<f32> = Vec::new();
    let mut remap = vec![0u32; m.positions.len()];
    for (i, p) in m.positions.iter().enumerate() {
        let id = *map.entry(key(*p)).or_insert_with(|| {
            props.extend_from_slice(&[p[0], p[1], p[2]]);
            (props.len() / 3 - 1) as u32
        });
        remap[i] = id;
    }
    let tris: Vec<u32> = m.indices.iter().map(|&i| remap[i as usize]).collect();
    (props, tris)
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
}

/// Run a Manifold boolean; `None` if either operand can't be ingested or the result
/// isn't a valid manifold (so the caller can fall back to the BSP CSG).
fn manifold_boolean(a: &TriMesh, b: &TriMesh, op: Op) -> Option<TriMesh> {
    let (ma, mb) = (to_manifold(a)?, to_manifold(b)?);
    let r = match op {
        Op::Union => ma.union(&mb),
        Op::Difference => ma.difference(&mb),
    };
    if r.status().is_err() {
        return None;
    }
    let mesh = from_manifold(&r);
    (!mesh.indices.is_empty()).then_some(mesh)
}

/// Boolean **union** of two triangle meshes (Manifold; BSP CSG fallback).
pub fn mesh_union(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Union).unwrap_or_else(|| csg::bsp_union(a, b))
}

/// Boolean **difference** `a − b` of two triangle meshes (Manifold; BSP CSG fallback).
pub fn mesh_difference(a: &TriMesh, b: &TriMesh) -> TriMesh {
    manifold_boolean(a, b, Op::Difference).unwrap_or_else(|| csg::bsp_difference(a, b))
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
}
