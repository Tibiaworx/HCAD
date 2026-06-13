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
