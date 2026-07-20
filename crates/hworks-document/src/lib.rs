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
    /// For a user-created offset plane: how it was built (base + distance), so it can be re-edited.
    /// `None` for the three standard datum planes.
    #[serde(default)]
    pub offset: Option<PlaneOffset>,
}

/// How a construction plane was built — a base plane/face and a signed offset along its normal —
/// kept so the plane can be reopened and edited after creation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PlaneOffset {
    /// Base origin + axes the plane was offset from (the face/plane geometry at creation time).
    pub base_origin: [f32; 3],
    pub base_u: [f32; 3],
    pub base_v: [f32; 3],
    pub base_name: String,
    pub distance: f32,
    pub flip: bool,
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
    /// True if the sketch was made on a fixed datum plane (Front/Top/Right), not a body face.
    /// Datum planes never move, so regeneration must NOT reproject them onto a body face — doing
    /// so would snap a centre-plane sketch onto a parallel cap and displace the profile.
    #[serde(default)]
    pub datum: bool,
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
    /// Contours") of `sketch`. An empty list means "all closed regions". `back` is the optional
    /// Direction-2 distance (≥0): the prism also extends `back` the opposite way (both directions).
    /// `thin` > 0 makes it a **thin feature**: extrude a wall of that thickness along the profile
    /// (a pipe/box shell) instead of filling the region. `thin_side`: 0 = outward (profile is the
    /// inner wall), 1 = inward (outer wall), 2 = mid-plane (split evenly).
    /// `region_pts` are interior SAMPLE POINTS of the chosen regions (one per pick, sketch
    /// uv): region INDICES shift when a crossing-heavy sketch re-solves (the arrangement
    /// re-orders), so regeneration re-resolves the selection by point when samples exist.
    Extrude { sketch: Sketch, regions: Vec<usize>, #[serde(default)] region_pts: Vec<[f64; 2]>, plane: PlaneRef, distance: f64, #[serde(default)] back: f64, #[serde(default)] thin: f64, #[serde(default)] thin_side: u8 },
    /// Cut: subtract the chosen swept `regions` from the body (empty = all). `back` = Direction 2.
    /// `thin`/`thin_side` as for [`Extrude`] — subtract a wall instead of the filled region.
    Cut { sketch: Sketch, regions: Vec<usize>, #[serde(default)] region_pts: Vec<[f64; 2]>, plane: PlaneRef, distance: f64, #[serde(default)] back: f64, #[serde(default)] thin: f64, #[serde(default)] thin_side: u8 },
    /// Revolve: sweep the chosen `regions` of `sketch` around an axis line (a point + direction in
    /// the sketch's uv plane) by `angle` radians (τ = full turn). `cut` subtracts the swept solid
    /// from the body (a lathe groove/bore) instead of adding it (a revolve boss).
    Revolve { sketch: Sketch, regions: Vec<usize>, #[serde(default)] region_pts: Vec<[f64; 2]>, plane: PlaneRef, axis_pt: [f64; 2], axis_dir: [f64; 2], angle: f64, #[serde(default)] cut: bool },
    /// Fillet: round body edges by `radius` (a mesh round). `edges` are the picked edge
    /// polylines (world space) to round; empty means "round every edge".
    Fillet { radius: f64, #[serde(default)] edges: Vec<Vec<[f64; 3]>> },
    /// Chamfer: flat-bevel the picked `edges` (world-space polylines) by `distance`.
    Chamfer { distance: f64, edges: Vec<Vec<[f64; 3]>> },
    /// Loft: skin a solid between an ordered list of cross-section profiles (each a sketch on its
    /// own plane). Builds a smooth body connecting the profiles — the construction-plane payoff.
    /// `cut` subtracts the lofted solid from the body (a tapered pocket/bore) instead of adding it.
    Loft { profiles: Vec<LoftProfile>, #[serde(default)] cut: bool },
    /// Mirror: reflect the whole body across `plane` and union it with the original
    /// (a symmetric part). The plane is recorded so it survives regeneration.
    Mirror { plane: PlaneRef },
    /// Pattern: repeat the *tool* of an earlier material feature (`seed` = its timeline index —
    /// an Extrude/Cut/Revolve/Loft) `count` times total, re-applying the seed's op (boss unions,
    /// cut subtracts) at each instance. Linear: step `spacing` along the world direction `dir`.
    /// Circular: rotate `step` radians per instance about the world axis through `axis_pt`
    /// along `axis_dir`.
    Pattern {
        seed: usize,
        circular: bool,
        dir: [f64; 3],
        spacing: f64,
        axis_pt: [f64; 3],
        axis_dir: [f64; 3],
        step: f64,
        count: u32,
    },
    /// Shell: hollow the body leaving walls of `thickness`. Each entry in `open` is a picked
    /// face to REMOVE (a point on the face + its outward normal) — the cavity opens through it,
    /// like SolidWorks' "faces to remove". Empty = a fully enclosed hollow.
    Shell { thickness: f64, #[serde(default)] open: Vec<([f64; 3], [f64; 3])> },
    /// Sweep: sweep a profile sketch region along an open path sketched on (usually) another
    /// plane. The profile is carried along the path with rotation-minimising frames (no twist).
    /// `cut` subtracts the swept solid instead of adding it.
    Sweep { profile: LoftProfile, path_sketch: Sketch, path_plane: PlaneRef, #[serde(default)] cut: bool },
    /// Threaded hole / thread (the "Hole Genie"): at `origin` on a face with outward normal
    /// `axis`, a thread of `major_d` × `pitch` over `depth`. `internal` taps a hole; false
    /// threads an existing boss. `rh` = right-handed.
    Thread { origin: [f64; 3], axis: [f64; 3], major_d: f64, pitch: f64, depth: f64, internal: bool, rh: bool },
    /// An imported triangle mesh (an STL) as a **solid body feature**: the first solid feature
    /// it becomes the body, later it unions in — and everything downstream (cuts, fillets,
    /// booleans) applies to it like any other body. `data` is the deflate-compressed binary
    /// STL, base64 — self-contained in the file. `scale` multiplies the raw (unitless) STL
    /// coordinates into mm. Mesh-kernel only (no exact B-rep for a scan).
    /// `rot_deg` (XYZ Euler, degrees, about the origin) then `offset` (mm) place the mesh —
    /// so a scan/import can be aligned to the datum planes without editing the source file.
    ImportMesh {
        data: String,
        name: String,
        #[serde(default = "default_mesh_scale")]
        scale: f64,
        #[serde(default)]
        rot_deg: [f64; 3],
        #[serde(default)]
        offset: [f64; 3],
        /// Voxel-remesh resolution for making a non-manifold scan cuttable. 0 = off (use the
        /// raw+repaired mesh); >0 = rebuild as a watertight solid at this voxel resolution
        /// (voxels on the longest axis). Lossy but robust — the fix for cutting into scans.
        #[serde(default)]
        solidify: u32,
    },
    /// An imported triangle mesh as **reference only** — a 3D scan to build parts onto or
    /// reverse-engineer. Renders as a translucent ghost, contributes NOTHING to the solid,
    /// and its sketch-plane cross-sections become snappable reference curves. Same embedded
    /// `data` format as [`ImportMesh`].
    RefMesh {
        data: String,
        name: String,
        #[serde(default = "default_mesh_scale")]
        scale: f64,
        #[serde(default = "default_ref_mesh_opacity")]
        opacity: f32,
        #[serde(default)]
        rot_deg: [f64; 3],
        #[serde(default)]
        offset: [f64; 3],
        /// Section-curve fit tolerance (mm): how far the fitted circles/arcs/lines may
        /// deviate from the raw cross-section. Clean exports → small; noisy scans → larger.
        #[serde(default = "default_section_tol")]
        section_tol: f64,
    },
    /// Reference image ("sketch picture"): a raster pinned to `plane` to trace over — not geometry,
    /// just a visual underlay. `data` is the base64-encoded PNG/JPG; `px_w`/`px_h` the source pixel
    /// size (for aspect ratio). `width`/`height` are the physical size on the plane (mm); `center` is
    /// the uv offset of the image centre from the plane origin; `rot` rotates it in-plane (rad);
    /// `opacity` is 0..1; `flip_h`/`flip_v` mirror it. Size starts at a default and is set by typing a
    /// dimension or the click-to-calibrate tool.
    RefImage {
        plane: PlaneRef,
        data: String,
        px_w: u32,
        px_h: u32,
        center: [f64; 2],
        rot: f64,
        width: f64,
        height: f64,
        opacity: f32,
        #[serde(default)]
        flip_h: bool,
        #[serde(default)]
        flip_v: bool,
    },
    // Revolve / … arrive at M8+.
}

/// One cross-section of a loft: a sketch on a plane, and which of its closed regions to skin.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LoftProfile {
    pub sketch: Sketch,
    pub plane: PlaneRef,
    pub region: usize,
}

/// One node in the feature timeline.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Feature {
    pub id: FeatureId,
    pub kind: FeatureKind,
    /// Visual-only visibility toggle (hide/show for planes, sketches, reference
    /// images — features whose viewport presence is decorative). Hidden features
    /// still regenerate: this never changes the solid.
    #[serde(default)]
    pub hidden: bool,
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
            offset: None,
        });
        doc.push_plane(Plane {
            name: "Top".into(),
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],  // +X
            v: [0.0, 0.0, -1.0], // -Z  (normal +Y)
            offset: None,
        });
        doc.push_plane(Plane {
            name: "Right".into(),
            origin: [0.0, 0.0, 0.0],
            u: [0.0, 0.0, -1.0], // -Z
            v: [0.0, 1.0, 0.0],  // +Y  (normal +X)
            offset: None,
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
        self.features.push(Feature { id, kind, hidden: false });
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
                FeatureKind::Revolve { angle, cut, .. } => {
                    if *cut {
                        ct += 1;
                        format!("[rev]    RevCut{ct}  {:.0}°", angle.to_degrees())
                    } else {
                        ex += 1;
                        format!("[rev]    Revolve{ex}  {:.0}°", angle.to_degrees())
                    }
                }
                FeatureKind::Fillet { radius, edges } => {
                    let scope = if edges.is_empty() { "all".to_string() } else { format!("{}", edges.len()) };
                    format!("[fillet] Fillet  r={radius:.2} ({scope})")
                }
                FeatureKind::Chamfer { distance, edges } => {
                    format!("[chamfer] Chamfer  d={distance:.2} ({})", edges.len())
                }
                FeatureKind::Loft { profiles, cut } => {
                    if *cut {
                        ct += 1;
                        format!("[loft]   LoftCut{ct}  ({} profiles)", profiles.len())
                    } else {
                        ex += 1;
                        format!("[loft]   Loft{ex}  ({} profiles)", profiles.len())
                    }
                }
                FeatureKind::Mirror { .. } => "[mirror] Mirror".to_string(),
                FeatureKind::Pattern { circular, count, .. } => {
                    let kind = if *circular { "Circular" } else { "Linear" };
                    format!("[pattern] {kind} Pattern  ×{count}")
                }
                FeatureKind::Shell { thickness, open } => {
                    format!("[shell]  Shell  t={thickness:.2} ({} open)", open.len())
                }
                FeatureKind::Sweep { cut, .. } => {
                    if *cut {
                        ct += 1;
                        format!("[sweep]  SweepCut{ct}")
                    } else {
                        ex += 1;
                        format!("[sweep]  Sweep{ex}")
                    }
                }
                FeatureKind::Thread { major_d, pitch, internal, .. } => {
                    let kind = if *internal { "tap" } else { "ext" };
                    format!("[thread] Thread {kind}  M{major_d:.1}×{pitch:.2}")
                }
                FeatureKind::RefImage { width, height, .. } => {
                    format!("[image]  Picture  {width:.0}×{height:.0}")
                }
                FeatureKind::ImportMesh { name, .. } => format!("[mesh]   {name}"),
                FeatureKind::RefMesh { name, .. } => format!("[scan]   {name}"),
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

    /// Like [`planes`] but with each plane's hide/show state, so drawing and
    /// viewport picking can skip hidden planes (a hidden plane must be neither
    /// visible nor clickable, while keeping every plane's tree order stable).
    pub fn planes_vis(&self) -> impl Iterator<Item = (&FeatureId, &Plane, bool)> {
        self.features.iter().filter_map(|f| match &f.kind {
            FeatureKind::Plane(p) => Some((&f.id, p, f.hidden)),
            _ => None,
        })
    }
}

fn default_mesh_scale() -> f64 {
    1.0
}
fn default_ref_mesh_opacity() -> f32 {
    0.35
}

fn default_section_tol() -> f64 {
    0.05
}

// ===================== Assemblies (Phase 1) =====================

fn quat_identity() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// One placed part instance inside an assembly.
///
/// The part is referenced the **hybrid** way: `source` is the `.hcad` path *relative to the
/// assembly file* (so a copied project folder keeps working), and `cached` is an embedded copy
/// of the part document — the fallback when the source file is missing, and what actually
/// regenerates. Opening an assembly refreshes each cache from its source file when present.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Component {
    /// Stable instance id (never reused) — selections and future mates point at this.
    pub id: u64,
    pub name: String,
    /// Source `.hcad` path relative to the assembly file; empty = embedded only.
    pub source: String,
    /// Embedded copy of the part's document.
    pub cached: Document,
    /// Placement: translation + unit quaternion `[x, y, z, w]`, part-local → assembly.
    pub translation: [f64; 3],
    #[serde(default = "quat_identity")]
    pub rotation: [f64; 4],
    /// Fixed components can't be dragged (the assembly's anchor).
    #[serde(default)]
    pub fixed: bool,
    #[serde(default)]
    pub hidden: bool,
}

/// An assembly document (`.hasm`): placed component instances. Mates arrive in Phase 2.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Assembly {
    pub components: Vec<Component>,
    /// Mates between component faces (Phase 2). Solved sequentially with relaxation.
    #[serde(default)]
    pub mates: Vec<Mate>,
    next_id: u64,
    #[serde(default)]
    next_mate_id: u64,
}

impl Assembly {
    /// Add a component instance; returns its stable id. The first component lands fixed
    /// (an assembly needs an anchor), later ones float.
    pub fn add_component(&mut self, name: String, source: String, cached: Document, translation: [f64; 3]) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.components.push(Component {
            id,
            name,
            source,
            cached,
            translation,
            rotation: quat_identity(),
            fixed: self.components.is_empty(),
            hidden: false,
        });
        id
    }

    pub fn component(&self, id: u64) -> Option<&Component> {
        self.components.iter().find(|c| c.id == id)
    }

    pub fn component_mut(&mut self, id: u64) -> Option<&mut Component> {
        self.components.iter_mut().find(|c| c.id == id)
    }

    /// Add a mate; returns its stable id.
    pub fn add_mate(&mut self, kind: u8, value: f64, flip: bool, a: MateRef, b: MateRef) -> u64 {
        self.next_mate_id += 1;
        let id = self.next_mate_id;
        self.mates.push(Mate { id, kind, value, flip, a, b });
        id
    }
}

/// One side of a mate: a geometric sample on a component's face — a point on it plus its
/// normal (planar) or axis (cylindrical), all in PART-LOCAL space. Sampling geometry rather
/// than face indices survives regeneration (mesh face ids are unstable across rebuilds).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MateRef {
    /// The component instance this face belongs to.
    pub comp: u64,
    /// 0 = planar face, 1 = cylindrical face.
    pub kind: u8,
    /// A point ON the face (planar), or a point on the cylinder AXIS (cylindrical).
    pub point: [f64; 3],
    /// The face normal (planar) or the axis direction (cylindrical), unit.
    pub dir: [f64; 3],
}

/// A mate constrains two components' faces. `kind`: 0 = Coincident (planes flush),
/// 1 = Distance (planes parallel at `value` mm), 2 = Concentric (cylinder axes align),
/// 3 = Parallel (normals align only). `flip` reverses the facing direction.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Mate {
    pub id: u64,
    pub kind: u8,
    pub value: f64,
    #[serde(default)]
    pub flip: bool,
    pub a: MateRef,
    pub b: MateRef,
}
