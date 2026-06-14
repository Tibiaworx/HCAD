//! `hworks-geometry` — Layer 1: the geometry kernel seam.
//!
//! This crate hides the concrete CAD kernel (initially [`truck`]) behind the
//! [`GeometryKernel`] trait so the rest of HCAD never depends on it directly.
//! That seam is what lets us swap in OpenCASCADE later without touching the
//! sketcher, document, or UI. See `DESIGN.md` §3 and §7.
//!
//! At milestone **M0** this is a stub: the trait shape is sketched out but no
//! kernel is wired in yet. Real solids (extrude/cut/tessellate) land at **M3**.

/// A tessellated triangle mesh handed up to the renderer.
#[derive(Debug, Default, Clone)]
pub struct TriMesh {
    /// Flat list of vertex positions (x, y, z).
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices into `positions`.
    pub indices: Vec<u32>,
}

/// The kernel abstraction. Concrete impls (truck, later OCCT) live behind this.
///
/// Intentionally tiny for now — it grows with the roadmap (revolve, fillet, …).
pub trait GeometryKernel {
    /// The kernel's native solid representation.
    type Solid;

    /// Extrude a closed profile into a solid (boss / add material).
    fn extrude(&self, profile: &Profile, distance: f64) -> Self::Solid;

    /// Subtract an extruded profile from an existing solid (cut).
    fn cut(&self, base: &Self::Solid, profile: &Profile, distance: f64) -> Self::Solid;

    /// Tessellate a solid into a triangle mesh for rendering.
    fn tessellate(&self, solid: &Self::Solid, tolerance: f64) -> TriMesh;
}

/// A closed 2D profile in a plane's local coordinates, ready to feed the kernel.
///
/// Populated by `hworks-sketch` once a sketch's outer loop is closed. Stub for now.
#[derive(Debug, Default, Clone)]
pub struct Profile {
    /// Ordered boundary points of the outer loop (local UV).
    pub outer: Vec<[f64; 2]>,
}

// ---------------------------------------------------------------------------
// truck-backed extrusion (M3)
//
// This is the first real wiring of the `truck` kernel. We take a closed 2D
// profile in a plane's local (u, v) coordinates, place it in 3D, build a B-rep
// face, and translational-sweep it into a solid — the canonical truck workflow
// (vertex → wire → face → solid). The solid is then tessellated for rendering.
// ---------------------------------------------------------------------------

use truck_meshalgo::prelude::*;
use truck_modeling::{builder, Point3, Vector3};

/// A plane expressed as a 3D origin and orthonormal in-plane axes (`u`, `v`)
/// plus its `normal`. Mirrors `hworks_document::Plane` but in `f64` world space.
#[derive(Debug, Clone)]
pub struct PlaneBasis {
    pub origin: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub normal: [f64; 3],
}

/// The result of an extrude: a render-ready triangle mesh plus the prism's
/// wireframe edges (for drawing a clean outline over the shaded faces).
pub struct ExtrudeResult {
    pub mesh: TriMesh,
    pub edges: Vec<[[f32; 3]; 2]>,
}

/// Extrude a closed profile (ordered loop, plane-local uv) along the plane normal
/// by `distance`, returning a tessellated mesh + wireframe. `None` if the profile
/// is degenerate or the kernel could not build a planar face from it.
pub fn extrude(profile_uv: &[[f64; 2]], basis: &PlaneBasis, distance: f64) -> Option<ExtrudeResult> {
    if profile_uv.len() < 3 || distance.abs() < 1e-9 {
        return None;
    }
    let origin = Vector3::new(basis.origin[0], basis.origin[1], basis.origin[2]);
    let u = Vector3::new(basis.u[0], basis.u[1], basis.u[2]);
    let v = Vector3::new(basis.v[0], basis.v[1], basis.v[2]);
    let n = Vector3::new(basis.normal[0], basis.normal[1], basis.normal[2]);

    let world = |uv: &[f64; 2]| origin + u * uv[0] + v * uv[1];
    let to_p3 = |uv: &[f64; 2]| {
        let p = world(uv);
        Point3::new(p.x, p.y, p.z)
    };

    // Vertex → wire → face → solid.
    let verts: Vec<_> = profile_uv.iter().map(|uv| builder::vertex(to_p3(uv))).collect();
    let np = verts.len();
    let mut wire = truck_modeling::Wire::new();
    for i in 0..np {
        wire.push_back(builder::line(&verts[i], &verts[(i + 1) % np]));
    }
    let face = builder::try_attach_plane(&vec![wire]).ok()?;
    let solid = builder::tsweep(&face, n * distance);

    // Tessellate the B-rep into a triangle mesh.
    let tol = (distance.abs() * 0.01).max(0.005);
    let mut poly = solid.triangulation(tol).to_polygon();
    poly.triangulate();
    let mesh = polymesh_to_trimesh(&poly);

    // Prism wireframe from the profile: bottom loop, top loop, verticals.
    let mut edges = Vec::with_capacity(np * 3);
    let arr = |w: Vector3| [w.x as f32, w.y as f32, w.z as f32];
    for i in 0..np {
        let a = &profile_uv[i];
        let b = &profile_uv[(i + 1) % np];
        let (a0, b0) = (world(a), world(b));
        let (a1, b1) = (a0 + n * distance, b0 + n * distance);
        edges.push([arr(a0), arr(b0)]); // bottom edge
        edges.push([arr(a1), arr(b1)]); // top edge
        edges.push([arr(a0), arr(a1)]); // vertical edge
    }

    Some(ExtrudeResult { mesh, edges })
}

/// Convert a truck `PolygonMesh` into a flat-shaded [`TriMesh`]. We compute a
/// per-triangle normal from the winding so shading is correct regardless of
/// whether the kernel supplied vertex normals.
fn polymesh_to_trimesh(poly: &truck_polymesh::PolygonMesh) -> TriMesh {
    let pos = poly.positions();
    let mut out = TriMesh::default();
    for tri in poly.faces().tri_faces() {
        let p0 = pos[tri[0].pos];
        let p1 = pos[tri[1].pos];
        let p2 = pos[tri[2].pos];
        // Manual cross product of (p1-p0) × (p2-p0).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_square_makes_a_box() {
        // Unit square on the XY plane, extruded 1 unit along +Z.
        let basis = PlaneBasis {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let res = extrude(&square, &basis, 1.0).expect("extrude should succeed");

        // A box must tessellate to at least 12 triangles (6 quad faces × 2).
        assert!(res.mesh.indices.len() >= 36, "got {} indices", res.mesh.indices.len());
        assert_eq!(res.mesh.indices.len() % 3, 0);
        assert_eq!(res.mesh.positions.len(), res.mesh.normals.len());

        // All vertices lie within the unit cube (with a little tolerance).
        for p in &res.mesh.positions {
            for c in p {
                assert!(*c >= -0.01 && *c <= 1.01, "vertex out of box: {p:?}");
            }
        }
        // 4 profile edges × 3 (bottom/top/vertical) = 12 wireframe segments.
        assert_eq!(res.edges.len(), 12);
    }
}
