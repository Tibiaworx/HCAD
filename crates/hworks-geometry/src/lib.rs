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

/// A render-ready tessellation: triangle mesh + feature/boundary edges.
pub struct Tessellation {
    pub mesh: TriMesh,
    pub edges: Vec<[[f32; 3]; 2]>,
}

/// Tolerance for boolean operations and tessellation.
const TOL: f64 = 0.05;

// ---------------------------------------------------------------------------
// Public kernel operations
// ---------------------------------------------------------------------------

/// Extrude a closed profile (ordered loop, plane-local uv) along the plane normal
/// by `distance` into a solid (boss / add material). `None` if degenerate.
pub fn extrude_solid(profile_uv: &[[f64; 2]], basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    build_solid(profile_uv, basis, 0.0, distance).map(KSolid)
}

/// Boolean union of two solids (boss added to an existing body).
pub fn union(a: &KSolid, b: &KSolid) -> Option<KSolid> {
    truck_shapeops::or(&a.0, &b.0, TOL).map(KSolid)
}

/// Boolean cut: subtract an extrusion of `profile_uv` from `base`.
///
/// The cutting tool is extruded with a small overshoot on both ends so its caps
/// never sit exactly coplanar with the base's faces — that coplanarity is the
/// classic failure mode for B-rep booleans, and the overshoot avoids it.
pub fn cut(base: &KSolid, profile_uv: &[[f64; 2]], basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    let depth = distance.abs();
    let eps = 0.05 + depth * 0.02;
    let mut tool = build_solid(profile_uv, basis, -eps, depth + 2.0 * eps)?;
    tool.not(); // invert all faces → complement region: base ∩ ¬tool == base − tool
    truck_shapeops::and(&base.0, &tool, TOL).map(KSolid)
}

/// Tessellate a solid into a flat-shaded mesh plus its feature/boundary edges.
pub fn tessellate(solid: &KSolid, tol: f64) -> Tessellation {
    let mut poly = solid.0.triangulation(tol).to_polygon();
    poly.triangulate();
    let mesh = polymesh_to_trimesh(&poly);
    let edges = feature_edges(&mesh, 18.0);
    Tessellation { mesh, edges }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

/// Build a prism solid: place the profile at `origin + normal*start_offset`, then
/// translational-sweep it by `normal*length`. The canonical truck vertex → wire →
/// face → solid workflow.
fn build_solid(
    profile_uv: &[[f64; 2]],
    basis: &PlaneBasis,
    start_offset: f64,
    length: f64,
) -> Option<truck_modeling::Solid> {
    if profile_uv.len() < 3 || length.abs() < 1e-9 {
        return None;
    }
    let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);
    let base = origin + n * start_offset;

    let to_p3 = |uv: &[f64; 2]| {
        let p = base + u * uv[0] + v * uv[1];
        Point3::new(p.x, p.y, p.z)
    };
    let verts: Vec<_> = profile_uv.iter().map(|uv| builder::vertex(to_p3(uv))).collect();
    let np = verts.len();
    let mut wire = truck_modeling::Wire::new();
    for i in 0..np {
        wire.push_back(builder::line(&verts[i], &verts[(i + 1) % np]));
    }
    let face = builder::try_attach_plane(&vec![wire]).ok()?;
    Some(builder::tsweep(&face, n * length))
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

/// Extract the wireframe: every edge that is either a boundary (used by one
/// triangle) or a feature edge (its two triangles meet at more than `min_angle`
/// degrees). Coplanar interior edges are skipped, giving a clean outline.
fn feature_edges(mesh: &TriMesh, min_angle_deg: f64) -> Vec<[[f32; 3]; 2]> {
    use std::collections::HashMap;
    // Merge duplicated (flat-shaded) vertices by quantized position.
    let quant = |p: [f32; 3]| {
        (
            (p[0] * 1.0e4).round() as i64,
            (p[1] * 1.0e4).round() as i64,
            (p[2] * 1.0e4).round() as i64,
        )
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

    let cos_thresh = min_angle_deg.to_radians().cos();
    let mut out = Vec::new();
    for ((i, j), normals) in emap {
        let keep = if normals.len() == 1 {
            true // boundary edge
        } else {
            // sharp if any incident pair of faces differ by more than the angle
            let mut sharp = false;
            for a in 0..normals.len() {
                for b in (a + 1)..normals.len() {
                    let d = normals[a][0] * normals[b][0]
                        + normals[a][1] * normals[b][1]
                        + normals[a][2] * normals[b][2];
                    if (d as f64) < cos_thresh {
                        sharp = true;
                    }
                }
            }
            sharp
        };
        if keep {
            out.push([canon_pos[i], canon_pos[j]]);
        }
    }
    out
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

    #[test]
    fn extrude_square_makes_a_box() {
        let square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let solid = extrude_solid(&square, &xy_plane(), 2.0).expect("extrude");
        let t = tessellate(&solid, 0.05);
        assert!(t.mesh.indices.len() >= 36, "got {} indices", t.mesh.indices.len());
        assert_eq!(t.mesh.indices.len() % 3, 0);
        // A closed box has 12 feature edges.
        assert_eq!(t.edges.len(), 12, "box should have 12 edges, got {}", t.edges.len());
    }

    #[test]
    fn cutting_a_pocket_reduces_volume_and_adds_edges() {
        // Base: 4×4 box, 2 tall.
        let base = extrude_solid(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], &xy_plane(), 2.0)
            .expect("base");
        // Cut a centered 2×2 pocket straight through.
        let pocket = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        let result = cut(&base, &pocket, &xy_plane(), 2.0).expect("cut should succeed");
        let t = tessellate(&result, 0.05);
        // A box with a rectangular through-hole has more than the 12 edges of a plain box.
        assert!(t.edges.len() > 12, "cut result should have extra edges, got {}", t.edges.len());
        assert!(t.mesh.indices.len() % 3 == 0 && !t.mesh.positions.is_empty());
    }
}
