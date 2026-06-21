//! `hworks-document` — Layer 3: the parametric feature tree. The source of truth.
//!
//! The document is the *recipe* that produces the geometry, not the geometry
//! itself: an ordered timeline of features (planes, sketches, extrudes, cuts)
//! that regenerate when an earlier step is edited. See `DESIGN.md` §4.
//!
//! At milestone **M0** the only feature kind that exists is the datum [`Plane`],
//! and a fresh document starts with the three standard reference planes. The
//! Bevy app renders *from* this document — it never owns the model itself.

use hworks_sketch::Sketch;
use serde::{Deserialize, Serialize};

/// A stable identifier for a feature. Never reused, so downstream references
/// stay valid across regeneration. (Topological naming, `DESIGN.md` §4.3, builds
/// on this idea at the entity level from M5.)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureId(pub u64);

/// A datum plane: an origin and two orthonormal in-plane axes (u, v). The plane
/// normal is `u × v`. Sketches are drawn in (u, v) coordinates.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Plane {
    pub name: String,
    pub origin: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
}

/// A geometric reference to the plane a sketch was made on — a reference plane
/// or a planar face of the body. Recording it (rather than a face index) is the
/// start of surviving regeneration: after a rebuild, the face is re-matched by
/// geometry (normal + a point on it). See `DESIGN.md` §4.3. (M5 groundwork.)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlaneRef {
    pub origin: [f64; 3],
    pub u: [f64; 3],
    pub v: [f64; 3],
    pub normal: [f64; 3],
}

/// The kinds of feature a timeline can hold. Grows along the roadmap.
///
/// Operation features are *self-contained* — they carry the sketch and the plane
/// they were drawn on — so the whole timeline can be replayed from scratch to
/// regenerate the solid (M6). That replay is what makes editing parametric.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FeatureKind {
    Plane(Plane),
    /// A standalone sketch on a plane/face that hasn't been extruded yet. Kept in
    /// the timeline so you can return to it. Its regions are re-resolved from the
    /// sketch at regenerate time, so editing the sketch updates downstream.
    Sketch { sketch: Sketch, plane: PlaneRef },
    /// Boss extrude: add material by sweeping the chosen `regions` (the "Selected
    /// Contours") of `sketch`. An empty list means "all closed regions".
    Extrude { sketch: Sketch, regions: Vec<usize>, plane: PlaneRef, distance: f64 },
    /// Cut: subtract the chosen swept `regions` from the body (empty = all).
    Cut { sketch: Sketch, regions: Vec<usize>, plane: PlaneRef, distance: f64 },
    /// Fillet: round body edges by `radius` (a mesh round). `edges` are the picked edge
    /// polylines (world space) to round; empty means "round every edge".
    Fillet { radius: f64, #[serde(default)] edges: Vec<Vec<[f64; 3]>> },
    /// Chamfer: flat-bevel the picked `edges` (world-space polylines) by `distance`.
    Chamfer { distance: f64, edges: Vec<Vec<[f64; 3]>> },
    /// Mirror: reflect the whole body across `plane` and union it with the original
    /// (a symmetric part). The plane is recorded so it survives regeneration.
    Mirror { plane: PlaneRef },
    /// Threaded hole / thread (the "Hole Genie"): at `origin` on a face with outward normal
    /// `axis`, a thread of `major_d` × `pitch` over `depth`. `internal` taps a hole; false
    /// threads an existing boss. `rh` = right-handed.
    Thread { origin: [f64; 3], axis: [f64; 3], major_d: f64, pitch: f64, depth: f64, internal: bool, rh: bool },
    // Revolve / … arrive at M8+.
}

/// One node in the feature timeline.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Feature {
    pub id: FeatureId,
    pub kind: FeatureKind,
}

/// The whole part: an ordered timeline plus a rollback position.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Document {
    pub features: Vec<Feature>,
    /// Features at indices `>= rollback` are "rolled back" (suppressed).
    pub rollback: usize,
    next_id: u64,
}

impl Document {
    /// A new document seeded with the three standard reference planes
    /// (Front = XY, Top = XZ, Right = YZ), exactly like SolidWorks.
    pub fn with_default_planes() -> Self {
        let mut doc = Document::default();
        doc.push_plane(Plane {
            name: "Front".into(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0], // +X
            v: [0.0, 1.0, 0.0], // +Y  (normal +Z)
        });
        doc.push_plane(Plane {
            name: "Top".into(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],  // +X
            v: [0.0, 0.0, -1.0], // -Z  (normal +Y)
        });
        doc.push_plane(Plane {
            name: "Right".into(),
            origin: [0.0, 0.0, 0.0],
            u: [0.0, 0.0, -1.0], // -Z
            v: [0.0, 1.0, 0.0],  // +Y  (normal +X)
        });
        doc
    }

    fn push_plane(&mut self, plane: Plane) -> FeatureId {
        self.add_feature(FeatureKind::Plane(plane))
    }

    /// Append a feature to the timeline, returning its stable id.
    pub fn add_feature(&mut self, kind: FeatureKind) -> FeatureId {
        let id = FeatureId(self.next_id);
        self.next_id += 1;
        self.features.push(Feature { id, kind });
        self.rollback = self.features.len();
        id
    }

    /// One display label per feature, in timeline order (for the feature-tree
    /// panel). Extrudes/cuts are numbered in the order they appear.
    pub fn tree_labels(&self) -> Vec<String> {
        let (mut sk, mut ex, mut ct) = (0, 0, 0);
        self.features
            .iter()
            .map(|f| match &f.kind {
                FeatureKind::Plane(p) => format!("[plane]  {}", p.name),
                FeatureKind::Sketch { .. } => {
                    sk += 1;
                    format!("[sketch] Sketch{sk}")
                }
                FeatureKind::Extrude { distance, .. } => {
                    ex += 1;
                    format!("[boss]   Extrude{ex}  h={distance:.1}")
                }
                FeatureKind::Cut { distance, .. } => {
                    ct += 1;
                    format!("[cut]    Cut{ct}  h={distance:.1}")
                }
                FeatureKind::Fillet { radius, edges } => {
                    let scope = if edges.is_empty() { "all".to_string() } else { format!("{}", edges.len()) };
                    format!("[fillet] Fillet  r={radius:.2} ({scope})")
                }
                FeatureKind::Chamfer { distance, edges } => {
                    format!("[chamfer] Chamfer  d={distance:.2} ({})", edges.len())
                }
                FeatureKind::Mirror { .. } => "[mirror] Mirror".to_string(),
                FeatureKind::Thread { major_d, pitch, internal, .. } => {
                    let kind = if *internal { "tap" } else { "ext" };
                    format!("[thread] Thread {kind}  M{major_d:.1}×{pitch:.2}")
                }
            })
            .collect()
    }

    /// Iterate the datum planes currently in the document.
    pub fn planes(&self) -> impl Iterator<Item = (&FeatureId, &Plane)> {
        self.features.iter().filter_map(|f| match &f.kind {
            FeatureKind::Plane(p) => Some((&f.id, p)),
            _ => None,
        })
    }
}
