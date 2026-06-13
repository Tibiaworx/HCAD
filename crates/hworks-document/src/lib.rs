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

/// A stable identifier for a feature. Never reused, so downstream references
/// stay valid across regeneration. (Topological naming, `DESIGN.md` §4.3, builds
/// on this idea at the entity level from M5.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureId(pub u64);

/// A datum plane: an origin and two orthonormal in-plane axes (u, v). The plane
/// normal is `u × v`. Sketches are drawn in (u, v) coordinates.
#[derive(Debug, Clone)]
pub struct Plane {
    pub name: String,
    pub origin: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
}

/// The kinds of feature a timeline can hold. Grows along the roadmap.
#[derive(Debug, Clone)]
pub enum FeatureKind {
    Plane(Plane),
    Sketch(Sketch),
    // Extrude / Cut / Revolve / … arrive at M3+.
}

/// One node in the feature timeline.
#[derive(Debug, Clone)]
pub struct Feature {
    pub id: FeatureId,
    pub kind: FeatureKind,
}

/// The whole part: an ordered timeline plus a rollback position.
#[derive(Debug, Default, Clone)]
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
        let id = FeatureId(self.next_id);
        self.next_id += 1;
        self.features.push(Feature { id, kind: FeatureKind::Plane(plane) });
        self.rollback = self.features.len();
        id
    }

    /// Iterate the datum planes currently in the document.
    pub fn planes(&self) -> impl Iterator<Item = (&FeatureId, &Plane)> {
        self.features.iter().filter_map(|f| match &f.kind {
            FeatureKind::Plane(p) => Some((&f.id, p)),
            _ => None,
        })
    }
}
