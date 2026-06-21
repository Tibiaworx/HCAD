//! HCAD — Layer 4: the Bevy viewport + a SolidWorks-style egui shell.
//!
//! UI pass (post-M4): the app is now mouse-driven.
//!  - **Top toolbar** (CommandManager): sketch tools + feature operations.
//!  - **Left panel** (FeatureManager / PropertyManager): the feature tree, and
//!    when a boss/cut is being configured, an editable depth field + OK/Cancel.
//!  - **Bottom status bar**: mode, tool, cursor coordinates, feature count, units.
//!  - Standard-view + zoom-fit buttons.
//!
//! Keyboard shortcuts remain as accelerators (S/L/C/R/X · E boss · D cut · Esc).
//! Drawing happens in the 3D viewport; egui panels capture the pointer so clicks
//! on the UI never leak into the sketch.

use bevy::asset::RenderAssetUsages;
use bevy::gizmos::config::{GizmoConfigGroup, GizmoConfigStore};
use bevy::log::tracing_subscriber::Layer;
use bevy::log::{BoxedLayer, LogPlugin};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use hworks_document::{Document, FeatureKind, Plane, PlaneRef};
use hworks_geometry::{
    chamfer_mesh, cut_tol, cut_tool_mesh, extrude_solid, extrude_solid_with_overlap,
    extrude_tool_mesh, mesh_difference, mesh_tessellation, mesh_union, mirror_mesh, round_mesh,
    tessellate, union_tol, KSolid, PlaneBasis, Tessellation, TriMesh,
};

/// Gizmo group rendered ON TOP of the solid (depth-biased), for the extrude preview,
/// the drag arrow, and the cut-depth indicator — so they show through the model.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct OverlayGizmos;

/// Gizmo group for the drawn sketch profile geometry, given a thicker line width than the
/// default group (grid, markers, dimensions) for better visibility.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct ProfileGizmos;

/// How far a boss reaches back into the body it's built on (flush path). Kept below
/// the 0.03 tessellation tolerance so any overhang lip is sub-facet (invisible),
/// while big enough to keep the union robust paired with the tight tolerance.
const BOSS_OVERLAP: f64 = 0.01;
use hworks_sketch::{
    point_in_poly, tessellate_arc_slot, tessellate_slot, tessellate_spline, text_contours, Constraint,
    DimAxis, Sketch, SketchEntity,
};

mod text;

/// Default boss/cut depth used by the keyboard accelerators (the UI lets you edit it).
const EXTRUDE_DISTANCE: f64 = 2.0;
const PLANE_SIZE: f32 = 8.0;
const SNAP: f32 = 0.18;
/// Max entities held in the Select-tool selection (enough for an Equal across many lines).
const MAX_SEL: usize = 32;
/// How long (seconds) an edge's key points flash after it's selected.
const FLASH_SECS: f32 = 1.2;
/// Two adjacent edge segments merge into one chain while the turn between them
/// stays under ~60° (dot ≥ this). Sharp model corners (~90°) break the chain, so a
/// box edge stays a single edge while a tessellated circle walks into a full loop.
const EDGE_CONTINUE_COS: f32 = 0.5;
/// Screen-space pixel radius for picking a model edge under the cursor.
const EDGE_PICK_PX: f32 = 9.0;

/// A tracing layer that writes every log event to `run.log` (truncated each launch),
/// unbuffered, so failures show up live. Added on top of the default stdout layer.
fn file_log_layer(_app: &mut App) -> Option<BoxedLayer> {
    // `&File` writes are unbuffered syscalls, so each event lands on disk immediately.
    let file = std::fs::File::create("run.log").ok()?;
    let layer = bevy::log::tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(std::sync::Arc::new(file));
    Some(layer.boxed())
}

fn main() {
    // GPU selection: the integrated Radeon's 2020-era driver crashes (device
    // lost) under DX12, so we let Bevy use the default (discrete) GPU, which is
    // stable. On hybrid laptops that can flicker (cross-GPU present); the proper
    // fix is to run on the integrated GPU after updating its driver — see README.
    let power_preference = match std::env::var("HCAD_GPU").as_deref() {
        Ok("integrated") | Ok("low") => PowerPreference::LowPower,
        Ok("discrete") | Ok("high") => PowerPreference::HighPerformance,
        _ => PowerPreference::HighPerformance,
    };

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "HCAD".into(),
                        present_mode: bevy::window::PresentMode::Fifo,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        power_preference,
                        ..default()
                    }),
                    ..default()
                })
                // Mirror all logs to a line-flushed `run.log` (alongside stdout) so a
                // failed cut/regenerate is inspectable while the app is still running.
                .set(LogPlugin {
                    custom_layer: file_log_layer,
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin::default())
        .init_gizmo_group::<OverlayGizmos>()
        .init_gizmo_group::<ProfileGizmos>()
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .insert_resource(DocRes(Document::with_default_planes()))
        .init_resource::<SketchSession>()
        .init_resource::<Part>()
        .init_resource::<UiState>()
        .init_resource::<UiBlocking>()
        .init_resource::<FontPreviews>()
        .init_resource::<History>()
        .init_resource::<EdgeSelection>()
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .add_systems(
            Update,
            (
                (
                    sketch_interaction,
                    handle_keys,
                    history_keys,
                    apply_history,
                    handle_file_io,
                    handle_edit_sketch,
                    handle_exit_sketch,
                    do_solid_op,
                    apply_fillet,
                    apply_chamfer,
                    apply_mirror_feature,
                    do_regenerate,
                    fillet_preview,
                    chamfer_preview,
                    mirror_preview,
                ),
                (
                    handle_new_part,
                    highlight_face,
                    update_plane_visibility,
                    orbit_camera,
                    draw_world_axes,
                    draw_body_edges,
                    draw_sketch,
                    tick_edge_flash,
                    draw_edge_selection,
                ),
            ),
        )
        .run();
}

// ---------------------------------------------------------------------------
// Resources & state
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct DocRes(Document);

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Tool {
    Select,
    #[default]
    Line,
    Circle,
    Rectangle,
    Slot,
    Polygon,
    Spline,
    Text,
    Dimension,
    Pattern,
    Mirror,
}

/// Pattern-tool variant.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum PatternMode {
    /// Repeat the selection along one or two directions (rows × columns).
    #[default]
    Linear,
    /// Repeat the selection around a centre point.
    Circular,
    /// Tile copies of the selection to fill a chosen closed region.
    Fill,
}

/// Rectangle-tool variant.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum RectMode {
    /// Two opposite corners (axis-aligned).
    #[default]
    Corner,
    /// Centre then a corner; adds X-pattern construction diagonals + a centre point.
    Center,
    /// Three clicks: the first two anchor one side, the third pulls out a parallelogram.
    Parallelogram,
}

/// Slot-tool variant.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum SlotMode {
    /// Two end centres, then width (the centre line runs end-to-end).
    #[default]
    Straight,
    /// Centre of the slot, then one end, then width (the line grows from the centre).
    Centerpoint,
    /// Two ends, a bend point (the centre line arcs), then width.
    Arc,
}

/// An in-progress drag of a placed Text entity's on-canvas handle.
#[derive(Clone, Copy)]
enum TextHandle {
    /// Drag to scale: the entity index. Height follows the cursor's distance from origin.
    Scale(usize),
    /// Drag to rotate: the entity index. Rotation follows the cursor's angle about origin.
    Rotate(usize),
}

/// A body edge the cursor can snap to while sketching: a straight edge (its uv
/// endpoints) or a rounded edge / fillet arc (its centre + radius in plane uv).
#[derive(Clone, Copy)]
enum EdgeSnap {
    Line([Vec2; 2]),
    Arc { center: Vec2, radius: f32, a: Vec2, b: Vec2 },
}

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Line => "Line",
            Tool::Circle => "Circle",
            Tool::Rectangle => "Rectangle",
            Tool::Slot => "Slot",
            Tool::Polygon => "Polygon",
            Tool::Text => "Text",
            Tool::Spline => "Spline",
            Tool::Dimension => "Dimension",
            Tool::Pattern => "Pattern",
            Tool::Mirror => "Mirror",
        }
    }
}

/// A requested boolean solid operation, carrying its depth.
#[derive(Clone, Copy)]
enum SolidOp {
    Boss(f64),
    Cut(f64),
}

#[derive(Clone, Copy, PartialEq)]
enum OpKind {
    Boss,
    Cut,
}

/// An action chosen from a feature-tree right-click menu (applied after the
/// tree's immutable borrow ends).
#[derive(Clone, Copy)]
enum TreeAction {
    Select(usize),
    Edit(usize),
    ExtrudeBoss(usize),
    ExtrudeCut(usize),
    Delete(usize),
}

/// A camera action chosen from the right-click context menu.
#[derive(Clone, Copy)]
enum ViewAction {
    NormalToSketch,
    Normal(Vec3),
    Iso,
    Fit,
    ExitSketch,
}

/// PropertyManager state for a boss/cut being configured in the UI.
#[derive(Clone)]
struct PendingOp {
    kind: OpKind,
    depth: f32,
    /// Direction 1 "reverse" toggle — extrude/cut against the sketch normal.
    reverse: bool,
}

#[derive(Resource, Default)]
struct UiState {
    pending: Option<PendingOp>,
    /// egui style applied once.
    themed: bool,
    /// "New Part" was clicked; consumed by `handle_new_part`.
    new_part: bool,
    /// The model needs rebuilding from the timeline; consumed by `do_regenerate`.
    regen: bool,
    /// Selected feature index in the tree (for editing).
    selected: Option<usize>,
    /// If set, a viewport right-click context menu is open at this screen position.
    context_pos: Option<egui::Pos2>,
    /// Undo/redo requested (from buttons or Ctrl+Z / Ctrl+Y).
    undo_request: bool,
    redo_request: bool,
    /// Save / open requested (from buttons or Ctrl+S / Ctrl+O).
    save_request: bool,
    open_request: bool,
    /// Request to (re)open a feature's sketch for editing.
    edit_sketch_request: Option<usize>,
    /// Show smooth/tangent edges in the viewport (off = SolidWorks-style removed).
    show_tangent_edges: bool,
    /// Active CommandManager tab.
    active_tab: Tab,
    /// Tracks sketch-mode transitions so the tab can auto-switch.
    was_sketching: bool,
    /// Pending depth value for the selected feature's editor (applied on Apply).
    edit_depth: f32,
    /// Which feature `edit_depth` currently mirrors.
    edit_depth_for: Option<usize>,
    /// The last operation failure to surface to the user (shown as a banner until
    /// dismissed or a clean regenerate clears it). Set by `do_regenerate`.
    last_error: Option<String>,
    /// Build the whole model with the robust **mesh** kernel (Manifold) instead of the
    /// exact B-rep kernel. This fuses coincident/coplanar faces — so adjacent features
    /// with shared walls merge *seamlessly* — at the cost of triangulated (mesh) faces.
    seamless: bool,
    /// The Fillet tool's PropertyManager is open, configuring this radius. While set, the
    /// viewport shows a live rounded preview of the current body.
    pending_fillet: Option<f32>,
    /// The radius the on-screen fillet preview was last built at (so it only recomputes
    /// when the value actually changes).
    fillet_shown: Option<f32>,
    /// A confirmed fillet to append to the timeline; consumed by `apply_fillet`.
    fillet_request: Option<f64>,
    /// While the Fillet PM is open, the edges (world-space polylines) the user has picked
    /// to round. Empty = round every edge. (Shared by the chamfer tool too.)
    fillet_edges: Vec<Vec<[f64; 3]>>,
    /// Chamfer tool: the bevel distance the PM is configuring (mirrors the fillet state).
    pending_chamfer: Option<f32>,
    chamfer_shown: Option<f32>,
    chamfer_request: Option<f64>,
    /// Mirror feature: the chosen mirror plane while the PM is open (0=Front, 1=Top,
    /// 2=Right), the plane currently previewed, and a confirmed mirror to append.
    pending_mirror: Option<u8>,
    mirror_shown: Option<u8>,
    mirror_request: Option<u8>,
}

/// A `PlaneRef` for one of the three standard reference planes (0=Front, 1=Top, 2=Right),
/// matching `Document::with_default_planes`.
fn standard_plane_ref(which: u8) -> PlaneRef {
    match which {
        1 => PlaneRef { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0] }, // Top (XZ)
        2 => PlaneRef { origin: [0.0; 3], u: [0.0, 0.0, -1.0], v: [0.0, 1.0, 0.0], normal: [1.0, 0.0, 0.0] }, // Right (YZ)
        _ => PlaneRef { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }, // Front (XY)
    }
}

/// CommandManager tabs (SolidWorks-style), to declutter the toolbar.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Tab {
    #[default]
    Features,
    Sketch,
}

/// `.0`: egui wants the pointer (suppress viewport drawing/orbit). `.1`: egui wants the
/// keyboard, e.g. a text field is focused (suppress sketch keyboard shortcuts).
#[derive(Resource, Default)]
struct UiBlocking(bool, bool);

/// System fonts registered with egui so the Text tool's font dropdown can render each
/// name in its own typeface. Populated once, the first time the Text panel is shown.
#[derive(Resource, Default)]
struct FontPreviews {
    /// Registration has been requested (`set_fonts` called).
    done: bool,
    /// The registered fonts are actually bound in egui (true from the frame *after*
    /// `set_fonts`, since it only takes effect on the next pass). Only render names in
    /// their own typeface once this is set, or egui panics on the unbound family.
    ready: bool,
    families: std::collections::HashSet<String>,
}

#[derive(Resource, Default)]
struct Part {
    solid: Option<KSolid>,
    /// Last tessellation, kept in world space so faces can be ray-picked.
    mesh: Option<TriMesh>,
    /// Body feature edges, drawn as an overlay (gizmos) so they can't z-fight.
    edges: Vec<[[f32; 3]; 2]>,
    /// Smooth/tangent edges (curvature lines) — hidden unless the user shows them.
    tangent_edges: Vec<[[f32; 3]; 2]>,
}

/// A picked model edge (or edge loop) in view mode: the ordered world-space
/// vertices of the chain, whether it closes on itself, and a short-lived flash of
/// its key points (midpoint + endpoints, or a loop's quadrant points).
#[derive(Resource, Default)]
struct EdgeSelection {
    chain: Vec<Vec3>,
    closed: bool,
    /// Seconds remaining on the key-point flash (counts down to 0).
    flash: f32,
    flash_points: Vec<Vec3>,
}

impl EdgeSelection {
    /// Select a chain and (re)start the key-point flash.
    fn set(&mut self, chain: Vec<Vec3>, closed: bool) {
        self.flash_points = chain_flash_points(&chain, closed);
        self.chain = chain;
        self.closed = closed;
        self.flash = FLASH_SECS;
    }
    fn clear(&mut self) {
        self.chain.clear();
        self.flash_points.clear();
        self.flash = 0.0;
    }
}

#[derive(Clone)]
struct ActivePlane {
    name: String,
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    n: Vec3,
}

impl ActivePlane {
    fn from_doc(p: &Plane) -> Self {
        let u = Vec3::from_array(p.u);
        let v = Vec3::from_array(p.v);
        Self { name: p.name.clone(), origin: Vec3::from_array(p.origin), u, v, n: u.cross(v) }
    }
    fn to_world(&self, uv: Vec2) -> Vec3 {
        self.origin + self.u * uv.x + self.v * uv.y
    }
}

#[derive(Resource, Default)]
struct SketchSession {
    plane: Option<ActivePlane>,
    tool: Tool,
    construction: bool,
    /// Line-tool variant: when set, a line grows symmetrically from its first click
    /// (the click is the midpoint), with a Midpoint relation pinning the centre.
    line_midpoint: bool,
    /// The most recently drawn construction *centre line* as `[a, mid, b]` point indices.
    /// Its PropertyManager edits the two half-lengths and can equalise them. `mid` is the
    /// pivot; the endpoints slide along the line's direction when a length changes, and
    /// dragging `mid` with the Select tool translates the whole line.
    center_line: Option<[usize; 3]>,
    /// The last half-length the user typed for the centre line — "Make sides equal" sets
    /// both halves to this number (rather than averaging).
    center_line_len: Option<f32>,
    /// An open on-canvas centre-line length editor: `(endpoint index, edit buffer)`.
    center_line_edit: Option<(usize, f32)>,
    /// A dimension (constraint index) selected on the canvas — highlighted, and removable
    /// with Delete / the red ✕ in the panel.
    selected_dim: Option<usize>,
    /// Pattern tool: which variant, and its parameters (linear rows/cols + spacing,
    /// circular centre/count/angle, fill spacing/margin). Operates on the current
    /// selection (entities to copy; for Fill, a chosen closed region is the boundary).
    pattern_mode: PatternMode,
    pat_count1: u32,
    pat_count2: u32,
    pat_spacing1: f32,
    pat_spacing2: f32,
    pat_circ_count: u32,
    pat_circ_angle: f32,
    pat_circ_center: Vec2,
    pat_center_set: bool,
    /// While true, the next canvas click sets the circular-pattern centre (snapped to a
    /// point / endpoint), instead of selecting geometry.
    pattern_pick_center: bool,
    pat_fill_spacing: f32,
    pat_fill_margin: f32,
    pattern_init: bool,
    /// Per-operation undo/redo *within* the sketch (so Ctrl+Z reverts the last line /
    /// dimension / drag, not the whole sketch feature). `undo_baseline` holds the last
    /// recorded stable state; `undo_fp` its fingerprint, so a change is detected cheaply.
    undo_sketch: Vec<Sketch>,
    redo_sketch: Vec<Sketch>,
    undo_baseline: Option<Sketch>,
    undo_fp: u64,
    /// Circle-tool variant: when set, the first click anchors a point on the rim and the
    /// circle grows to the cursor (the two clicks are opposite ends of a diameter).
    circle_perimeter: bool,
    /// Spline tool: the points placed so far for the in-progress spline.
    spline_pts: Vec<Vec2>,
    /// Spline-tool variant: false ⇒ through-points (Catmull-Rom), true ⇒ control-points.
    spline_control: bool,
    /// Rectangle-tool variant (corner / centre / parallelogram).
    rect_mode: RectMode,
    /// Slot-tool variant (straight / centrepoint / arc).
    slot_mode: SlotMode,
    /// Polygon tool: number of sides (left-hand parameter), default 6.
    polygon_sides: usize,
    /// Text tool parameters (left-hand panel). `text_arc` is the text-on-arc radius (0 =
    /// straight); `text_mirror` reverses the text. `text_font_init` lazily fills the font
    /// name from the system default the first time the Text tool is opened.
    text_string: String,
    text_font: String,
    text_font_init: bool,
    text_bold: bool,
    text_italic: bool,
    text_spacing: f64,
    text_height: f32,
    text_arc: f64,
    text_mirror: bool,
    /// An on-canvas handle drag in progress on a placed Text entity.
    text_handle: Option<TextHandle>,
    pending: Option<Vec2>,
    /// Second anchored point — the parallelogram's first side / the slot's second centre.
    pending_b: Option<Vec2>,
    /// Third anchored point — the arc slot's bend point.
    pending_c: Option<Vec2>,
    /// First point picked by the Dimension tool (point index).
    dim_first: Option<usize>,
    /// A just-placed dimension awaiting a typed value (the Distance constraint index),
    /// its editing buffer, and a one-shot focus request for the Modify box.
    dim_edit: Option<usize>,
    dim_buf: f64,
    dim_edit_focus: bool,
    /// If the open Modify box dimensions a single line (entity index), clicking a
    /// second line converts it into an angle dimension between the two lines.
    dim_line: Option<usize>,
    /// A dimension whose offset is being dragged with the Select tool (constraint index).
    dim_drag: Option<usize>,
    /// Timestamp of the last Select-tool click, for double-click-to-edit detection.
    last_click_t: f32,
    /// Live dimension input while drawing (length for a line, radius for a circle).
    live_buf: f32,
    /// Request keyboard focus on the live-input field (set when a draw starts).
    request_live_focus: bool,
    drag: Option<usize>,
    dirty: bool,
    op_request: Option<SolidOp>,
    cursor_uv: Option<Vec2>,
    /// The cursor's raw position on the sketch plane, before snapping. Used by the
    /// polygon tool so its radius tracks the mouse instead of jumping to a far snap.
    cursor_raw_uv: Option<Vec2>,
    /// Drag-over box select (Select tool): the anchor corner where the drag started on
    /// empty space; the opposite corner follows the cursor until release.
    box_select: Option<Vec2>,
    /// Which closed contours (indices into `sketch.regions()`) are selected for
    /// extrude/cut — the "Selected Contours". Empty means "all closed regions".
    selected_contours: Vec<usize>,
    /// Sketch entities selected (with the Select tool) for applying a constraint.
    selected_entities: Vec<usize>,
    /// Snap/inference points for the entity currently under the cursor (line
    /// midpoint, circle centre + quadrants). Recomputed each frame on hover.
    inference_points: Vec<Vec2>,
    /// User toggle to disable the snap/inference points.
    hide_inference: bool,
    /// Entity (line/circle) under the cursor in Select mode, highlighted on hover.
    hover_entity: Option<usize>,
    /// Reference snap points (uv) from the body's edges lying in the sketch plane —
    /// each in-plane edge's endpoints and midpoint. Lets new sketch geometry snap to
    /// existing model features (e.g. an edge midpoint) to stay square/aligned.
    reference_points: Vec<Vec2>,
    /// Circular edges on the sketch face: (centre uv, radius). Lets the Circle tool
    /// snap its radius so a new circle matches existing round geometry exactly.
    reference_circles: Vec<(Vec2, f32)>,
    /// The body edge (straight or arc) currently under the cursor while sketching,
    /// highlighted.
    hover_edge: Option<EdgeSnap>,
    /// The edge the cursor is snapped *onto* this frame (so placing a point adds a
    /// point-on-edge relation); and the same remembered for the line's start point.
    cursor_edge: Option<EdgeSnap>,
    pending_edge: Option<EdgeSnap>,
    /// Snap tolerance in world units, scaled to the zoom so snapping feels the same
    /// at any scale (a fixed tolerance is unusable on a large part). Set each frame.
    snap_dist: f32,
    /// True while the user is dragging the extrude direction arrow to set the depth.
    arrow_drag: bool,
    /// If editing an existing feature's sketch, its feature index (else a new sketch).
    editing: Option<usize>,
    /// Request to leave sketch mode and commit the sketch to the timeline.
    exit_request: bool,
    /// Request to leave sketch mode and discard the changes (no commit).
    cancel_request: bool,
    /// Edited dimensions / added relations are staged until Apply re-solves.
    needs_apply: bool,
    sketch: Sketch,
}

/// Undo/redo stacks of whole-document snapshots.
#[derive(Resource, Default)]
struct History {
    undo: Vec<Document>,
    redo: Vec<Document>,
}

impl History {
    /// Snapshot the document before a mutation (caps the stack; clears redo).
    fn snapshot(&mut self, doc: &Document) {
        self.undo.push(doc.clone());
        if self.undo.len() > 64 {
            self.undo.remove(0);
        }
        self.redo.clear();
    }
}

#[derive(Component)]
struct OrbitCamera {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { focus: Vec3::ZERO, radius: 12.0, yaw: 0.8, pitch: -0.55 }
    }
}

#[derive(Component)]
struct SolidPart;

/// A reference plane's quad — shown only while starting a part (no body, not yet
/// sketching), so it can't be accidentally selected once you're modeling.
#[derive(Component)]
struct RefPlane;

/// The translucent overlay that highlights the hovered / active face.
#[derive(Component)]
struct FaceHighlight;

/// Handle to the highlight mesh so it can be rebuilt in place each frame.
#[derive(Resource)]
struct HighlightMesh(Handle<Mesh>);

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gizmo_store: ResMut<GizmoConfigStore>,
    doc: Res<DocRes>,
) {
    // Overlay gizmos draw in front of the solid so the extrude preview/arrow and the
    // cut-depth indicator are visible through the model.
    gizmo_store.config_mut::<OverlayGizmos>().0.depth_bias = -1.0;
    // Drawn sketch lines are a touch thicker than the grid/markers for visibility.
    gizmo_store.config_mut::<ProfileGizmos>().0.line.width = 3.2;

    let plane_mesh = meshes.add(Rectangle::new(PLANE_SIZE, PLANE_SIZE));
    let colors = [
        Color::srgba(0.85, 0.25, 0.25, 0.16),
        Color::srgba(0.25, 0.75, 0.30, 0.16),
        Color::srgba(0.25, 0.45, 0.90, 0.16),
    ];
    for (i, (_id, plane)) in doc.0.planes().enumerate() {
        let ap = ActivePlane::from_doc(plane);
        let rotation = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));
        let material = materials.add(StandardMaterial {
            base_color: colors[i % colors.len()],
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            double_sided: true,
            // Distinct bias per plane so the three transparent quads sort
            // deterministically instead of flickering as the camera moves.
            depth_bias: i as f32,
            ..default()
        });
        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(material),
            Transform { translation: ap.origin, rotation, ..default() },
            Name::new(plane.name.clone()),
            RefPlane,
        ));
    }

    commands.spawn((
        DirectionalLight { illuminance: 6_000.0, shadows_enabled: false, ..default() },
        Transform::from_xyz(6.0, 10.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let cam = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        // Wide depth range so large parts (and a far-backed-out camera) don't clip.
        Projection::from(PerspectiveProjection { near: 0.02, far: 100_000.0, ..default() }),
        camera_transform(&cam),
        cam,
        AmbientLight { color: Color::WHITE, brightness: 250.0, ..default() },
    ));

    // Reusable translucent overlay for face highlighting (hidden until needed).
    let mut empty = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    empty.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    empty.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    let hl_mesh = meshes.add(empty);
    let hl_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.25, 0.6, 1.0, 0.35),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(hl_mesh.clone()),
        MeshMaterial3d(hl_material),
        Visibility::Hidden,
        FaceHighlight,
        Name::new("FaceHighlight"),
    ));
    commands.insert_resource(HighlightMesh(hl_mesh));

    println!("HCAD ready — mouse-driven UI. Click a reference plane to start sketching.");
}

fn camera_transform(cam: &OrbitCamera) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let translation = cam.focus + rotation * Vec3::new(0.0, 0.0, cam.radius);
    Transform { translation, rotation, ..default() }
}

/// Focus point + camera radius that frame the current body — or the reference
/// planes when there's no body yet. Used by Zoom-to-Fit so it works at any scale.
fn fit_view(part: &Part) -> (Vec3, f32) {
    if let Some(mesh) = &part.mesh {
        if !mesh.positions.is_empty() {
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for p in &mesh.positions {
                let v = Vec3::from_array(*p);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let center = (lo + hi) * 0.5;
            let radius = ((hi - lo).length() * 0.9).max(4.0);
            return (center, radius);
        }
    }
    (Vec3::ZERO, 14.0)
}

/// Aim the orbit camera straight down a plane/face normal (a "Normal To" view),
/// keeping the current radius. Setting yaw/pitch (not the transform directly)
/// means the user can keep orbiting smoothly afterwards.
fn look_along(cam: &mut OrbitCamera, focus: Vec3, normal: Vec3) {
    let n = normal.normalize_or_zero();
    if n != Vec3::ZERO {
        cam.yaw = n.x.atan2(n.z);
        cam.pitch = (-n.y).asin().clamp(-1.54, 1.54);
    }
    cam.focus = focus;
}

// ---------------------------------------------------------------------------
// egui shell
// ---------------------------------------------------------------------------

/// A bare painted down-triangle dropdown trigger — no button frame, and no missing-glyph
/// box (it's drawn, not a font character). Anchor a `Popup::menu` on the returned response.
fn dropdown_arrow(ui: &mut egui::Ui, hover: &str) -> egui::Response {
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(14.0, ui.spacing().interact_size.y), egui::Sense::click());
    let resp = resp.on_hover_text(hover);
    let c = rect.center();
    let col = if resp.hovered() { ui.visuals().strong_text_color() } else { ui.visuals().text_color() };
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x - 4.0, c.y - 2.5),
            egui::pos2(c.x + 4.0, c.y - 2.5),
            egui::pos2(c.x, c.y + 3.0),
        ],
        col,
        egui::Stroke::NONE,
    ));
    resp
}

fn ui_system(
    mut contexts: EguiContexts,
    mut session: ResMut<SketchSession>,
    mut ui_state: ResMut<UiState>,
    mut blocking: ResMut<UiBlocking>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
    mut cam_q: Query<(&mut Transform, &mut OrbitCamera)>,
    cam_read: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    part: Res<Part>,
    mut font_previews: ResMut<FontPreviews>,
    edge_sel: Res<EdgeSelection>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    // The first time the Text tool is active, register the system fonts with egui so the
    // font dropdown can render each name in its own typeface. `set_fonts` only applies on
    // the next pass, so previews stay off (`ready == false`) until the following frame.
    if session.tool == Tool::Text && session.plane.is_some() && !font_previews.done {
        register_system_fonts(ctx, &mut font_previews);
    } else if font_previews.done && !font_previews.ready {
        font_previews.ready = true;
    }

    // Sanitize every value that feeds an egui number widget — a NaN/∞ crashes egui's
    // "smart aim" code. (The solver already rejects NaN; this is the last line.)
    for c in &mut session.sketch.constraints {
        if let Constraint::Distance { value, .. } = c {
            if !value.is_finite() {
                *value = 1.0;
            }
        }
    }
    for e in &mut session.sketch.entities {
        if let SketchEntity::Circle { radius, .. } = e {
            if !radius.is_finite() {
                *radius = 1.0;
            }
        }
    }
    if !session.live_buf.is_finite() {
        session.live_buf = 0.0;
    }
    if let Some(op) = ui_state.pending.as_mut() {
        if !op.depth.is_finite() {
            op.depth = EXTRUDE_DISTANCE as f32;
        }
    }

    // Apply a CAD-ish dark style once.
    if !ui_state.themed {
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(6.0, 6.0);
        style.spacing.button_padding = egui::vec2(10.0, 5.0);
        style.visuals.selection.bg_fill = egui::Color32::from_rgb(59, 110, 165);
        style.visuals.panel_fill = egui::Color32::from_rgb(34, 36, 41);
        ctx.set_style(style);
        ui_state.themed = true;
    }

    let in_sketch = session.plane.is_some();
    let has_profile = !session.sketch.regions().is_empty();
    // A standalone Sketch feature is selected → it can be extruded from the Features tab.
    let selected_sketch = ui_state.selected.filter(|&i| {
        matches!(doc.0.features.get(i).map(|f| &f.kind), Some(FeatureKind::Sketch { .. }))
    });
    let can_extrude = (in_sketch && has_profile) || selected_sketch.is_some();

    // Auto-switch tab on entering/leaving a sketch. Leaving a sketch drops back to
    // the Select tool — no reason to stay on a drawing tool once in Features.
    if in_sketch != ui_state.was_sketching {
        ui_state.active_tab = if in_sketch { Tab::Sketch } else { Tab::Features };
        if !in_sketch {
            session.tool = Tool::Select;
        }
        ui_state.was_sketching = in_sketch;
    }

    // ---- Menu bar (File / Edit / View / Insert / Tools) ----
    // Placeholder menus for now; the entries are stubs wired to real actions where one
    // already exists (New/Open/Save/Undo/Redo) and left inert otherwise so they can be
    // populated later.
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Part").clicked() {
                    ui_state.new_part = true;
                    ui.close();
                }
                if ui.button("Open…").clicked() {
                    ui_state.open_request = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Save").clicked() {
                    ui_state.save_request = true;
                    ui.close();
                }
                let _ = ui.button("Save As…"); // TODO: distinct Save-As path
                ui.separator();
                let _ = ui.button("Exit"); // TODO: graceful shutdown
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(!history.undo.is_empty(), egui::Button::new("Undo")).clicked() {
                    ui_state.undo_request = true;
                    ui.close();
                }
                if ui.add_enabled(!history.redo.is_empty(), egui::Button::new("Redo")).clicked() {
                    ui_state.redo_request = true;
                    ui.close();
                }
                ui.separator();
                let _ = ui.button("Cut"); // TODO
                let _ = ui.button("Copy"); // TODO
                let _ = ui.button("Paste"); // TODO
                let _ = ui.button("Delete"); // TODO
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut ui_state.show_tangent_edges, "Tangent edges");
                ui.checkbox(&mut ui_state.seamless, "Seamless");
                ui.separator();
                let _ = ui.button("Zoom to Fit"); // TODO
                let _ = ui.button("Isometric"); // TODO
                let _ = ui.button("Front"); // TODO
                let _ = ui.button("Top"); // TODO
                let _ = ui.button("Right"); // TODO
            });
            ui.menu_button("Insert", |ui| {
                let _ = ui.button("Sketch"); // TODO: start a sketch
                let _ = ui.button("Boss Extrude…"); // TODO
                let _ = ui.button("Cut Extrude…"); // TODO
                ui.separator();
                let _ = ui.button("Fillet…"); // TODO
                let _ = ui.button("Chamfer…"); // TODO
                let _ = ui.button("Reference Plane…"); // TODO
            });
            ui.menu_button("Tools", |ui| {
                let _ = ui.button("Measure…"); // TODO
                let _ = ui.button("Mass Properties…"); // TODO
                ui.separator();
                let _ = ui.button("Options…"); // TODO
            });
        });
    });

    // ---- Top toolbar (CommandManager) ----
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        // Row 1: global actions + view controls.
        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("HCAD").strong().size(16.0));
            if ui.button("New Part").on_hover_text("Clear the model and start over").clicked() {
                ui_state.new_part = true;
            }
            if ui.add_enabled(!history.undo.is_empty(), egui::Button::new("Undo")).on_hover_text("Undo (Ctrl+Z)").clicked() {
                ui_state.undo_request = true;
            }
            if ui.add_enabled(!history.redo.is_empty(), egui::Button::new("Redo")).on_hover_text("Redo (Ctrl+Y)").clicked() {
                ui_state.redo_request = true;
            }
            ui.separator();
            if ui.button("Open").on_hover_text("Open a part (Ctrl+O)").clicked() {
                ui_state.open_request = true;
            }
            if ui.button("Save").on_hover_text("Save the part (Ctrl+S)").clicked() {
                ui_state.save_request = true;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut ui_state.show_tangent_edges, "Tangent edges")
                    .on_hover_text("Show smooth curvature edges (off = SolidWorks-style tangent edges removed)");
                if ui
                    .checkbox(&mut ui_state.seamless, "Seamless")
                    .on_hover_text(
                        "Build with the robust mesh kernel so adjacent features with shared/coincident \
                         walls merge without a seam (curved faces become triangulated)",
                    )
                    .changed()
                {
                    ui_state.regen = true; // rebuild with the chosen kernel
                }
                ui.separator();
                if ui.button("Fit").on_hover_text("Zoom to fit the part").clicked() {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        let (focus, radius) = fit_view(&part);
                        orbit.focus = focus;
                        orbit.radius = radius;
                        *tf = camera_transform(&orbit);
                    }
                }
                for (name, yaw, pitch) in [
                    ("Iso", 0.8_f32, -0.55_f32),
                    ("Right", 1.5708, 0.0),
                    ("Top", 0.0, -1.553),
                    ("Front", 0.0, 0.0),
                ] {
                    if ui.button(name).on_hover_text(format!("{name} view")).clicked() {
                        if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                            orbit.yaw = yaw;
                            orbit.pitch = pitch;
                            *tf = camera_transform(&orbit);
                        }
                    }
                }
                ui.label(egui::RichText::new("View").weak().small());
            });
        });
        ui.separator();

        // Row 2: CommandManager tabs + the active tab's tools.
        ui.horizontal_wrapped(|ui| {
            if ui.selectable_label(ui_state.active_tab == Tab::Features, "Features").clicked() {
                ui_state.active_tab = Tab::Features;
                session.tool = Tool::Select;
            }
            if ui.selectable_label(ui_state.active_tab == Tab::Sketch, "Sketch").clicked() {
                ui_state.active_tab = Tab::Sketch;
            }
            ui.separator();

            match ui_state.active_tab {
                Tab::Sketch => {
                    ui.add_enabled_ui(in_sketch, |ui| {
                        if ui
                            .selectable_label(session.tool == Tool::Select, "Select")
                            .on_hover_text("Select & drag points — geometry re-solves (S)")
                            .clicked()
                        {
                            session.tool = Tool::Select;
                            session.pending = None;
                        }
                        // Line tool + a ▾ dropdown of its variants (plain / construction /
                        // midpoint). Picking a variant also activates the Line tool.
                        let line_on = session.tool == Tool::Line;
                        if ui
                            .selectable_label(line_on, "Line")
                            .on_hover_text("Draw line segments; endpoints snap to close loops (L)")
                            .clicked()
                        {
                            session.tool = Tool::Line;
                            session.pending = None;
                        }
                        let is_ccl = session.construction && session.line_midpoint;
                        let is_mid = session.line_midpoint && !session.construction;
                        let is_con = session.construction && !session.line_midpoint;
                        let is_plain = !session.construction && !session.line_midpoint;
                        egui::Popup::menu(&dropdown_arrow(ui, "Line variants")).show(|ui| {
                            if ui.selectable_label(line_on && is_plain, "Line").clicked() {
                                session.tool = Tool::Line;
                                session.construction = false;
                                session.line_midpoint = false;
                                session.pending = None;
                            }
                            if ui
                                .selectable_label(line_on && is_con, "Construction Line")
                                .on_hover_text("A guide line — not part of the extrude profile")
                                .clicked()
                            {
                                session.tool = Tool::Line;
                                session.construction = true;
                                session.line_midpoint = false;
                                session.pending = None;
                            }
                            if ui
                                .selectable_label(line_on && is_mid, "Midpoint Line")
                                .on_hover_text("Line grows symmetrically from the first click (its midpoint)")
                                .clicked()
                            {
                                session.tool = Tool::Line;
                                session.construction = false;
                                session.line_midpoint = true;
                                session.pending = None;
                            }
                            if ui
                                .selectable_label(line_on && is_ccl, "Construction Center Line")
                                .on_hover_text("A construction centre line — click the centre, then an end. Edit each half-length in the panel.")
                                .clicked()
                            {
                                session.tool = Tool::Line;
                                session.construction = true;
                                session.line_midpoint = true;
                                session.pending = None;
                            }
                        });
                        // Circle tool + a ▾ dropdown: centre circle, or perimeter circle.
                        let circle_on = session.tool == Tool::Circle;
                        if ui
                            .selectable_label(circle_on, "Circle")
                            .on_hover_text("Click centre, then radius (C)")
                            .clicked()
                        {
                            session.tool = Tool::Circle;
                            session.pending = None;
                        }
                        let perim = session.circle_perimeter;
                        egui::Popup::menu(&dropdown_arrow(ui, "Circle variants")).show(|ui| {
                            if ui
                                .selectable_label(circle_on && !perim, "Circle")
                                .on_hover_text("Centre, then radius")
                                .clicked()
                            {
                                session.tool = Tool::Circle;
                                session.circle_perimeter = false;
                                session.pending = None;
                            }
                            if ui
                                .selectable_label(circle_on && perim, "Perimeter Circle")
                                .on_hover_text("Click a point on the rim, then drag to the opposite rim (a diameter)")
                                .clicked()
                            {
                                session.tool = Tool::Circle;
                                session.circle_perimeter = true;
                                session.pending = None;
                            }
                        });
                        // Rectangle tool + a ▾ dropdown: corner / centre / parallelogram.
                        let rect_on = session.tool == Tool::Rectangle;
                        if ui
                            .selectable_label(rect_on, "Rectangle")
                            .on_hover_text("Click two opposite corners (R)")
                            .clicked()
                        {
                            session.tool = Tool::Rectangle;
                            session.pending = None;
                            session.pending_b = None;
                        }
                        let rm = session.rect_mode;
                        egui::Popup::menu(&dropdown_arrow(ui, "Rectangle variants")).show(|ui| {
                            let mut pick = |ui: &mut egui::Ui, m: RectMode, name: &str, tip: &str| {
                                if ui.selectable_label(rect_on && rm == m, name).on_hover_text(tip).clicked() {
                                    session.tool = Tool::Rectangle;
                                    session.rect_mode = m;
                                    session.pending = None;
                                    session.pending_b = None;
                                }
                            };
                            pick(ui, RectMode::Corner, "Corner Rectangle", "Two opposite corners");
                            pick(ui, RectMode::Center, "Center Rectangle", "Centre, then a corner (adds X construction diagonals)");
                            pick(ui, RectMode::Parallelogram, "Parallelogram", "Two points anchor one side, then pull out the rest");
                        });
                        // Spline tool + a ▾ dropdown: through-points or control-points.
                        let spline_on = session.tool == Tool::Spline;
                        if ui
                            .selectable_label(spline_on, "Spline")
                            .on_hover_text("Click points for a smooth curve; Enter to finish, click the first point to close")
                            .clicked()
                        {
                            session.tool = Tool::Spline;
                            session.pending = None;
                        }
                        let ctrl = session.spline_control;
                        egui::Popup::menu(&dropdown_arrow(ui, "Spline variants")).show(|ui| {
                            if ui
                                .selectable_label(spline_on && !ctrl, "Spline (through points)")
                                .on_hover_text("Curve passes through each clicked point")
                                .clicked()
                            {
                                session.tool = Tool::Spline;
                                session.spline_control = false;
                                session.spline_pts.clear();
                            }
                            if ui
                                .selectable_label(spline_on && ctrl, "Spline (control points)")
                                .on_hover_text("Clicked points form a control hull the curve only approaches")
                                .clicked()
                            {
                                session.tool = Tool::Spline;
                                session.spline_control = true;
                                session.spline_pts.clear();
                            }
                        });
                        // Slot tool + a ▾ dropdown: straight / centrepoint / arc.
                        let slot_on = session.tool == Tool::Slot;
                        if ui
                            .selectable_label(slot_on, "Slot")
                            .on_hover_text("Click two centres for the slot line, then move out to set its width")
                            .clicked()
                        {
                            session.tool = Tool::Slot;
                            session.pending = None;
                            session.pending_b = None;
                            session.pending_c = None;
                        }
                        let sm = session.slot_mode;
                        egui::Popup::menu(&dropdown_arrow(ui, "Slot variants")).show(|ui| {
                            let mut pick = |ui: &mut egui::Ui, m: SlotMode, name: &str, tip: &str| {
                                if ui.selectable_label(slot_on && sm == m, name).on_hover_text(tip).clicked() {
                                    session.tool = Tool::Slot;
                                    session.slot_mode = m;
                                    session.pending = None;
                                    session.pending_b = None;
                                    session.pending_c = None;
                                }
                            };
                            pick(ui, SlotMode::Straight, "Straight Slot", "Two end centres, then the width");
                            pick(ui, SlotMode::Centerpoint, "Centerpoint Slot", "Centre, then one end, then the width");
                            pick(ui, SlotMode::Arc, "3-Point Arc Slot", "Two ends, bend into an arc, then the width");
                        });
                        // Polygon tool: click centre, then a vertex. Side count lives in the
                        // left-hand parameter panel.
                        if ui
                            .selectable_label(session.tool == Tool::Polygon, "Polygon")
                            .on_hover_text("Click the centre, then a vertex — sides set in the panel on the left")
                            .clicked()
                        {
                            if session.polygon_sides == 0 {
                                session.polygon_sides = 6;
                            }
                            session.tool = Tool::Polygon;
                            session.pending = None;
                        }
                        // Text tool: parameters (font, style, arc, …) live in the left panel.
                        if ui
                            .selectable_label(session.tool == Tool::Text, "Text")
                            .on_hover_text("Place outlined text — font and options in the panel on the left")
                            .clicked()
                        {
                            init_text_defaults(&mut session);
                            session.tool = Tool::Text;
                            session.pending = None;
                        }
                        if ui
                            .selectable_label(session.tool == Tool::Dimension, "Dimension")
                            .on_hover_text("Click two points to set an exact distance (M)")
                            .clicked()
                        {
                            session.tool = Tool::Dimension;
                            session.pending = None;
                        }
                        // Pattern tool + a ▾ dropdown: linear / circular / fill. Select the
                        // entities to repeat first; parameters live in the left panel.
                        if ui
                            .selectable_label(session.tool == Tool::Pattern, "Pattern")
                            .on_hover_text("Repeat the selected sketch geometry — options in the panel on the left")
                            .clicked()
                        {
                            init_pattern_defaults(&mut session);
                            session.tool = Tool::Pattern;
                            session.pending = None;
                        }
                        egui::Popup::menu(&dropdown_arrow(ui, "Pattern variants")).show(|ui| {
                            let pat_on = session.tool == Tool::Pattern;
                            let mut pick = |ui: &mut egui::Ui, m: PatternMode, name: &str, tip: &str| {
                                if ui.selectable_label(pat_on && session.pattern_mode == m, name).on_hover_text(tip).clicked() {
                                    init_pattern_defaults(&mut session);
                                    session.pattern_mode = m;
                                    session.tool = Tool::Pattern;
                                    session.pending = None;
                                }
                            };
                            pick(ui, PatternMode::Linear, "Linear Pattern", "Rows × columns at a set spacing");
                            pick(ui, PatternMode::Circular, "Circular Pattern", "Copies revolved around a centre");
                            pick(ui, PatternMode::Fill, "Fill Pattern", "Tile copies to fill a selected closed region");
                        });
                        // Mirror: reflect the selection across a selected line (a construction
                        // centre line is the natural axis).
                        if ui
                            .selectable_label(session.tool == Tool::Mirror, "Mirror")
                            .on_hover_text("Reflect selected geometry across a selected line — options in the panel on the left")
                            .clicked()
                        {
                            session.tool = Tool::Mirror;
                            session.pending = None;
                        }
                        let mut snap = !session.hide_inference;
                        if ui
                            .checkbox(&mut snap, "Snap pts")
                            .on_hover_text("Show midpoint / circle centre + quadrant snap points on hover")
                            .changed()
                        {
                            session.hide_inference = !snap;
                        }
                    });
                    if in_sketch {
                        ui.separator();
                        let confirm = if session.editing.is_some() { "Confirm edit" } else { "Finish sketch" };
                        if ui
                            .add(egui::Button::new(confirm).fill(egui::Color32::from_rgb(40, 110, 70)))
                            .on_hover_text("Apply changes and keep the sketch (Esc)")
                            .clicked()
                        {
                            session.exit_request = true;
                        }
                        if ui.button("Cancel").on_hover_text("Discard changes since opening this sketch").clicked() {
                            session.cancel_request = true;
                        }
                    } else {
                        ui.label(egui::RichText::new("Click a plane or face to start a sketch.").italics().weak());
                    }
                }
                Tab::Features => {
                    ui.add_enabled_ui(can_extrude, |ui| {
                        if ui.button("Extrude Boss").on_hover_text("Add material from the sketch (E)").clicked() {
                            if let Some(i) = selected_sketch.filter(|_| !in_sketch) {
                                ui_state.edit_sketch_request = Some(i);
                            }
                            ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32, reverse: false });
                        }
                        if ui.button("Extrude Cut").on_hover_text("Remove material from the sketch (D)").clicked() {
                            if let Some(i) = selected_sketch.filter(|_| !in_sketch) {
                                ui_state.edit_sketch_request = Some(i);
                            }
                            ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32, reverse: false });
                        }
                    });
                    if !can_extrude {
                        ui.label(egui::RichText::new("Select a sketch or draw a closed profile.").italics().weak());
                    }
                    ui.add_enabled_ui(part.mesh.is_some() && !in_sketch, |ui| {
                        // Seed the edge set from a pre-selected edge (click an edge, then the tool).
                        let seed = |ui_state: &mut UiState| {
                            ui_state.fillet_edges.clear();
                            if edge_sel.chain.len() >= 2 {
                                ui_state.fillet_edges.push(
                                    edge_sel.chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect(),
                                );
                            }
                        };
                        if ui.button("Fillet").on_hover_text("Round picked edges by a radius — click edges on the body").clicked() {
                            ui_state.pending_fillet = Some(0.2);
                            ui_state.fillet_shown = None;
                            ui_state.pending_chamfer = None;
                            seed(&mut ui_state);
                        }
                        if ui.button("Chamfer").on_hover_text("Flat-bevel picked edges by a distance — click edges on the body").clicked() {
                            ui_state.pending_chamfer = Some(0.2);
                            ui_state.chamfer_shown = None;
                            ui_state.pending_fillet = None;
                            seed(&mut ui_state);
                        }
                        if ui.button("Mirror").on_hover_text("Reflect the whole body across a plane and union it (a symmetric part)").clicked() {
                            ui_state.pending_mirror = Some(0);
                            ui_state.mirror_shown = None;
                            ui_state.pending_fillet = None;
                            ui_state.pending_chamfer = None;
                        }
                    });
                }
            }
        });
        ui.add_space(2.0);
    });

    // ---- Left panel: PropertyManager (if configuring) else FeatureManager ----
    egui::SidePanel::left("left_panel").default_width(240.0).show(ctx, |ui| {
      egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        // Polygon-tool parameters (SolidWorks-style PropertyManager): the side count.
        if session.tool == Tool::Polygon && session.plane.is_some() {
            ui.heading("Polygon");
            ui.separator();
            if session.polygon_sides == 0 {
                session.polygon_sides = 6;
            }
            ui.horizontal(|ui| {
                ui.label("Sides");
                let mut n = session.polygon_sides as u32;
                if ui
                    .add(egui::DragValue::new(&mut n).range(3..=64).speed(0.1))
                    .on_hover_text("Number of sides of the regular polygon")
                    .changed()
                {
                    session.polygon_sides = n.clamp(3, 64) as usize;
                }
            });
            ui.label(
                egui::RichText::new("Click the centre, then a vertex. The dashed circle is\nconstruction geometry — snap to it or dimension its radius.")
                    .weak()
                    .small(),
            );
            ui.separator();
        }
        // Construction centre-line parameters: the two half-lengths + an equalise button.
        if session.tool == Tool::Line
            && session.construction
            && session.line_midpoint
            && session.plane.is_some()
        {
            ui.heading("Centre Line");
            ui.separator();
            // Drop a stale reference if the line was removed (e.g. new sketch / undo).
            if let Some([a, mid, b]) = session.center_line {
                let n = session.sketch.points.len();
                if a >= n || mid >= n || b >= n {
                    session.center_line = None;
                }
            }
            match session.center_line {
                None => {
                    ui.label(
                        egui::RichText::new("Click the centre, then an end point. Each half-length\nappears here to edit; this line is construction geometry.")
                            .weak()
                            .small(),
                    );
                }
                Some([a, mid, b]) => {
                    let p = |i: usize| Vec2::new(session.sketch.points[i].x as f32, session.sketch.points[i].y as f32);
                    let (pm, pa, pb) = (p(mid), p(a), p(b));
                    let mut la = (pa - pm).length();
                    let mut lb = (pb - pm).length();
                    // Move an endpoint to a new half-length, sliding along its current
                    // direction from the (fixed) centre. Falls back to +X if degenerate.
                    let mut set_len = |sk: &mut Sketch, end: usize, len: f32| {
                        let m = Vec2::new(sk.points[mid].x as f32, sk.points[mid].y as f32);
                        let e = Vec2::new(sk.points[end].x as f32, sk.points[end].y as f32);
                        let dir = (e - m).normalize_or_zero();
                        let dir = if dir == Vec2::ZERO { Vec2::X } else { dir };
                        let np = m + dir * len.max(0.001);
                        sk.points[end].x = np.x as f64;
                        sk.points[end].y = np.y as f64;
                    };
                    let mut changed = false;
                    let mut entered: Option<f32> = None;
                    ui.horizontal(|ui| {
                        ui.label("Side A");
                        if ui.add(egui::DragValue::new(&mut la).speed(0.05).range(0.001..=10_000.0).suffix(" mm")).changed() {
                            set_len(&mut session.sketch, a, la);
                            entered = Some(la);
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Side B");
                        if ui.add(egui::DragValue::new(&mut lb).speed(0.05).range(0.001..=10_000.0).suffix(" mm")).changed() {
                            set_len(&mut session.sketch, b, lb);
                            entered = Some(lb);
                            changed = true;
                        }
                    });
                    if let Some(v) = entered {
                        session.center_line_len = Some(v);
                    }
                    if ui
                        .button("Make sides equal")
                        .on_hover_text("Set both halves to the length you last entered (or their average)")
                        .clicked()
                    {
                        // Use the number the user last typed; fall back to the average.
                        let target = session.center_line_len.unwrap_or((la + lb) * 0.5);
                        set_len(&mut session.sketch, a, target);
                        set_len(&mut session.sketch, b, target);
                        session.center_line_len = Some(target);
                        changed = true;
                    }
                    if changed {
                        session.dirty = true;
                        session.needs_apply = true;
                    }
                }
            }
            ui.separator();
        }
        // Pattern-tool parameters (operates on the current selection).
        if session.tool == Tool::Pattern && session.plane.is_some() {
            init_pattern_defaults(&mut session);
            let mode = session.pattern_mode;
            ui.heading(match mode {
                PatternMode::Linear => "Linear Pattern",
                PatternMode::Circular => "Circular Pattern",
                PatternMode::Fill => "Fill Pattern",
            });
            ui.separator();
            // Mode picker (mirrors the toolbar dropdown).
            ui.horizontal(|ui| {
                for (m, name) in [
                    (PatternMode::Linear, "Linear"),
                    (PatternMode::Circular, "Circular"),
                    (PatternMode::Fill, "Fill"),
                ] {
                    if ui.selectable_label(mode == m, name).clicked() {
                        session.pattern_mode = m;
                    }
                }
            });
            ui.separator();

            let seed_count = session
                .selected_entities
                .iter()
                .filter(|&&i| {
                    !matches!(
                        session.sketch.entities.get(i),
                        None | Some(SketchEntity::Line { reference: true, .. }) | Some(SketchEntity::Text { .. })
                    )
                })
                .count();
            ui.label(format!("Entities to repeat: {seed_count}"));
            ui.label(egui::RichText::new("Click sketch geometry to (de)select it.").weak().small());
            ui.add_space(4.0);

            match session.pattern_mode {
                PatternMode::Linear => {
                    egui::Grid::new("lin_pat").num_columns(2).show(ui, |ui| {
                        ui.label("Direction 1 count");
                        ui.add(egui::DragValue::new(&mut session.pat_count1).range(1..=200).speed(0.1));
                        ui.end_row();
                        ui.label("Direction 1 spacing");
                        ui.add(egui::DragValue::new(&mut session.pat_spacing1).speed(0.05).suffix(" mm"));
                        ui.end_row();
                        ui.label("Direction 2 count");
                        ui.add(egui::DragValue::new(&mut session.pat_count2).range(1..=200).speed(0.1));
                        ui.end_row();
                        ui.label("Direction 2 spacing");
                        ui.add(egui::DragValue::new(&mut session.pat_spacing2).speed(0.05).suffix(" mm"));
                        ui.end_row();
                    });
                    ui.label(egui::RichText::new("Direction 1 is the sketch X axis, direction 2 the Y axis.").weak().small());
                }
                PatternMode::Circular => {
                    egui::Grid::new("circ_pat").num_columns(2).show(ui, |ui| {
                        ui.label("Instances");
                        ui.add(egui::DragValue::new(&mut session.pat_circ_count).range(2..=360).speed(0.1));
                        ui.end_row();
                        ui.label("Total angle");
                        ui.add(egui::DragValue::new(&mut session.pat_circ_angle).range(1.0..=360.0).speed(0.5).suffix("°"));
                        ui.end_row();
                    });
                    ui.horizontal(|ui| {
                        let mut cx = session.pat_circ_center.x;
                        let mut cy = session.pat_circ_center.y;
                        ui.label("Centre");
                        if ui.add(egui::DragValue::new(&mut cx).speed(0.05).prefix("x ")).changed() {
                            session.pat_circ_center.x = cx;
                            session.pat_center_set = true;
                        }
                        if ui.add(egui::DragValue::new(&mut cy).speed(0.05).prefix("y ")).changed() {
                            session.pat_circ_center.y = cy;
                            session.pat_center_set = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Centre on selection").clicked() {
                            let seeds: Vec<usize> = session.selected_entities.clone();
                            session.pat_circ_center = selection_centroid(&session.sketch, &seeds);
                            session.pat_center_set = true;
                            session.pattern_pick_center = false;
                        }
                        if ui.button("Origin").clicked() {
                            session.pat_circ_center = Vec2::ZERO;
                            session.pat_center_set = true;
                            session.pattern_pick_center = false;
                        }
                    });
                    let picking = session.pattern_pick_center;
                    if ui
                        .selectable_label(picking, "🖈 Pick centre on canvas")
                        .on_hover_text("Then click a point / endpoint in the sketch to use as the revolve centre")
                        .clicked()
                    {
                        session.pattern_pick_center = !picking;
                    }
                    if session.pattern_pick_center {
                        ui.label(egui::RichText::new("Click a point or endpoint to set the centre…").color(egui::Color32::from_rgb(120, 200, 255)).small());
                    } else if !session.pat_center_set {
                        ui.label(egui::RichText::new("Centre defaults to the selection's centroid.").weak().small());
                    }
                }
                PatternMode::Fill => {
                    egui::Grid::new("fill_pat").num_columns(2).show(ui, |ui| {
                        ui.label("Spacing");
                        ui.add(egui::DragValue::new(&mut session.pat_fill_spacing).range(0.05..=1000.0).speed(0.05).suffix(" mm"));
                        ui.end_row();
                        ui.label("Border margin");
                        ui.add(egui::DragValue::new(&mut session.pat_fill_margin).range(0.0..=1000.0).speed(0.05).suffix(" mm"));
                        ui.end_row();
                    });
                    let region = session.selected_contours.first().is_some();
                    ui.label(if region {
                        egui::RichText::new("Boundary region: selected ✓").color(egui::Color32::from_rgb(90, 200, 120))
                    } else {
                        egui::RichText::new("Click inside a closed region to set the fill boundary.").color(egui::Color32::from_rgb(230, 170, 60))
                    });
                }
            }

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Apply").fill(egui::Color32::from_rgb(40, 110, 70))).clicked() {
                    match apply_pattern(&mut session) {
                        Ok(_) => {
                            session.selected_entities.clear();
                            session.selected_contours.clear();
                        }
                        Err(msg) => ui_state.last_error = Some(msg),
                    }
                }
                if ui.button("Done").clicked() {
                    session.tool = Tool::Select;
                    session.selected_entities.clear();
                }
            });
            ui.separator();
        }
        // Mirror-tool parameters (operates on the current selection).
        if session.tool == Tool::Mirror && session.plane.is_some() {
            ui.heading("Mirror");
            ui.separator();
            let axis = mirror_axis(&session);
            match axis {
                Some((ai, _, _)) => {
                    let kind = if matches!(session.sketch.entities.get(ai), Some(SketchEntity::Line { construction: true, .. })) {
                        "construction centre line"
                    } else {
                        "line"
                    };
                    ui.label(egui::RichText::new(format!("Mirror line: selected ({kind}) ✓")).color(egui::Color32::from_rgb(90, 200, 120)));
                    let n = mirror_seeds(&session, ai).len();
                    ui.label(format!("Geometry to mirror: {n}"));
                }
                None => {
                    ui.label(egui::RichText::new("Select a line (or construction centre line) as the mirror axis.").color(egui::Color32::from_rgb(230, 170, 60)));
                }
            }
            ui.label(
                egui::RichText::new("Click the geometry to reflect and the line to mirror across.\nA construction centre line is used as the axis if selected.")
                    .weak()
                    .small(),
            );
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Apply").fill(egui::Color32::from_rgb(40, 110, 70))).clicked() {
                    match apply_mirror(&mut session) {
                        Ok(_) => session.selected_entities.clear(),
                        Err(msg) => ui_state.last_error = Some(msg),
                    }
                }
                if ui.button("Done").clicked() {
                    session.tool = Tool::Select;
                    session.selected_entities.clear();
                }
            });
            ui.separator();
        }
        // Text-tool parameters (also edit the selected text entity live).
        if session.tool == Tool::Text && session.plane.is_some() {
            init_text_defaults(&mut session);
            ui.heading("Text");
            ui.separator();
            // If a single placed text is selected, edits apply to it; otherwise they set
            // up the next placement.
            let sel_text = session
                .selected_entities
                .iter()
                .copied()
                .find(|&i| matches!(session.sketch.entities.get(i), Some(SketchEntity::Text { .. })));

            let mut rebake = false; // font/style/string/spacing changed → re-outline
            let mut xform = false; // height/arc/mirror changed → just re-solve

            ui.label("Text");
            let mut s = session.text_string.clone();
            if ui.add(egui::TextEdit::multiline(&mut s).desired_rows(2)).changed() {
                session.text_string = s;
                rebake = true;
            }

            let cur_font = session.text_font.clone();
            // Render each name in its own typeface, but only once the fonts are actually
            // bound in egui (a frame after registration) — else the family lookup panics.
            let previews_ready = font_previews.ready;
            let preview = |fam: &str| -> egui::WidgetText {
                if previews_ready && font_previews.families.contains(fam) {
                    egui::RichText::new(fam)
                        .font(egui::FontId::new(16.0, egui::FontFamily::Name(std::sync::Arc::from(fam))))
                        .into()
                } else {
                    egui::RichText::new(fam).into()
                }
            };
            egui::ComboBox::from_label("Font")
                .selected_text(if cur_font.is_empty() { egui::WidgetText::from("—") } else { preview(&cur_font) })
                .show_ui(ui, |ui| {
                    for fam in text::families() {
                        let sel = session.text_font == fam;
                        if ui.selectable_label(sel, preview(&fam)).clicked() {
                            session.text_font = fam;
                            rebake = true;
                        }
                    }
                });

            ui.horizontal(|ui| {
                let mut b = session.text_bold;
                if ui.toggle_value(&mut b, "Bold").changed() {
                    session.text_bold = b;
                    rebake = true;
                }
                let mut it = session.text_italic;
                if ui.toggle_value(&mut it, "Italic").changed() {
                    session.text_italic = it;
                    rebake = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Spacing");
                let mut sp = session.text_spacing;
                if ui.add(egui::DragValue::new(&mut sp).range(-0.3..=2.0).speed(0.005)).changed() {
                    session.text_spacing = sp;
                    rebake = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Height");
                let mut h = session.text_height;
                if ui.add(egui::DragValue::new(&mut h).range(0.05..=1000.0).speed(0.02)).changed() {
                    session.text_height = h.max(0.05);
                    xform = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Arc R");
                let mut a = session.text_arc;
                if ui.add(egui::DragValue::new(&mut a).speed(0.05)).on_hover_text("Text-on-arc radius (0 = straight; ± curves up/down)").changed() {
                    session.text_arc = a;
                    xform = true;
                }
            });
            let mut mir = session.text_mirror;
            if ui.checkbox(&mut mir, "Reverse (mirror)").changed() {
                session.text_mirror = mir;
                xform = true;
            }

            // Push edits onto the selected entity.
            if let Some(idx) = sel_text {
                let (nh, na, nm, nt, nf, nb, ni, ns) = (
                    session.text_height.max(0.05) as f64,
                    session.text_arc,
                    session.text_mirror,
                    session.text_string.clone(),
                    session.text_font.clone(),
                    session.text_bold,
                    session.text_italic,
                    session.text_spacing,
                );
                if let Some(SketchEntity::Text { height, arc, mirror, text, font, bold, italic, spacing, .. }) =
                    session.sketch.entities.get_mut(idx)
                {
                    *height = nh;
                    *arc = na;
                    *mirror = nm;
                    *text = nt;
                    *font = nf;
                    *bold = nb;
                    *italic = ni;
                    *spacing = ns;
                }
                if rebake {
                    rebake_text(&mut session, idx);
                }
                if rebake || xform {
                    session.dirty = true;
                }
            }
            ui.label(
                egui::RichText::new("Click in the sketch to place. Select it, then drag the\nsquare to scale or the circle to rotate.")
                    .weak()
                    .small(),
            );
            ui.separator();
        }
        // Fillet PropertyManager: a single radius, with a live rounded preview.
        if let Some(mut r) = ui_state.pending_fillet {
            if !r.is_finite() {
                r = 0.2;
            }
            ui.heading("Fillet");
            let mut commit = false;
            let mut cancel = false;
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70)))
                    .clicked()
                {
                    commit = true;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55)))
                    .clicked()
                {
                    cancel = true;
                }
            });
            ui.separator();
            // Cap the radius to under half the body's smallest dimension — a steeper fillet
            // deforms the geometry and can abort the kernel.
            let max_r = part
                .mesh
                .as_ref()
                .map(|m| {
                    let (lo, hi) = mesh_bbox(m);
                    (((hi - lo).min_element()) * 0.45).max(0.02)
                })
                .unwrap_or(1000.0);
            if r > max_r {
                r = max_r;
                ui_state.fillet_shown = None;
            }
            ui.horizontal(|ui| {
                ui.label("Radius");
                if ui.add(egui::DragValue::new(&mut r).range(0.01..=max_r as f64).speed(0.02).suffix(" mm")).changed() {
                    ui_state.fillet_shown = None; // radius changed → refresh preview
                }
            });
            let n_edges = ui_state.fillet_edges.len();
            ui.horizontal(|ui| {
                ui.strong(format!("Edges to fillet  ({n_edges})"));
                if n_edges > 0 && ui.small_button("Clear").clicked() {
                    ui_state.fillet_edges.clear();
                    ui_state.fillet_shown = None;
                }
            });
            // SolidWorks-style selection list: one row per picked edge, removable.
            let mut remove: Option<usize> = None;
            egui::ScrollArea::vertical().id_salt("fillet_edge_list").max_height(140.0).show(ui, |ui| {
                for i in 0..n_edges {
                    ui.horizontal(|ui| {
                        if ui.small_button("✕").on_hover_text("Remove this edge").clicked() {
                            remove = Some(i);
                        }
                        ui.label(format!("Edge {}", i + 1));
                    });
                }
            });
            if let Some(i) = remove {
                ui_state.fillet_edges.remove(i);
                ui_state.fillet_shown = None;
            }
            ui.label(
                egui::RichText::new(if n_edges == 0 {
                    "Click edges on the body to pick which to round.\n(No edges picked = round every edge.)"
                } else {
                    "Click more edges to add, or a picked edge again to remove."
                })
                .weak()
                .small(),
            );
            ui.separator();
            if commit {
                ui_state.fillet_request = Some(r as f64);
                ui_state.pending_fillet = None;
                ui_state.fillet_shown = None;
            } else if cancel {
                ui_state.pending_fillet = None;
                ui_state.fillet_shown = None;
                ui_state.fillet_edges.clear();
                ui_state.regen = true; // rebuild without the preview
            } else {
                ui_state.pending_fillet = Some(r.max(0.01));
            }
        }
        // Chamfer PropertyManager: a single bevel distance + the same edge list.
        if let Some(mut d) = ui_state.pending_chamfer {
            if !d.is_finite() {
                d = 0.2;
            }
            ui.heading("Chamfer");
            let mut commit = false;
            let mut cancel = false;
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70))).clicked() {
                    commit = true;
                }
                if ui.add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55))).clicked() {
                    cancel = true;
                }
            });
            ui.separator();
            let max_d = part
                .mesh
                .as_ref()
                .map(|m| {
                    let (lo, hi) = mesh_bbox(m);
                    (((hi - lo).min_element()) * 0.45).max(0.02)
                })
                .unwrap_or(1000.0);
            if d > max_d {
                d = max_d;
                ui_state.chamfer_shown = None;
            }
            ui.horizontal(|ui| {
                ui.label("Distance");
                if ui.add(egui::DragValue::new(&mut d).range(0.01..=max_d as f64).speed(0.02).suffix(" mm")).changed() {
                    ui_state.chamfer_shown = None;
                }
            });
            let n_edges = ui_state.fillet_edges.len();
            ui.horizontal(|ui| {
                ui.strong(format!("Edges to chamfer  ({n_edges})"));
                if n_edges > 0 && ui.small_button("Clear").clicked() {
                    ui_state.fillet_edges.clear();
                    ui_state.chamfer_shown = None;
                }
            });
            let mut remove: Option<usize> = None;
            egui::ScrollArea::vertical().id_salt("chamfer_edge_list").max_height(140.0).show(ui, |ui| {
                for i in 0..n_edges {
                    ui.horizontal(|ui| {
                        if ui.small_button("✕").on_hover_text("Remove this edge").clicked() {
                            remove = Some(i);
                        }
                        ui.label(format!("Edge {}", i + 1));
                    });
                }
            });
            if let Some(i) = remove {
                ui_state.fillet_edges.remove(i);
                ui_state.chamfer_shown = None;
            }
            ui.label(egui::RichText::new("Click edges on the body to bevel them.").weak().small());
            ui.separator();
            if commit {
                ui_state.chamfer_request = Some(d as f64);
                ui_state.pending_chamfer = None;
                ui_state.chamfer_shown = None;
            } else if cancel {
                ui_state.pending_chamfer = None;
                ui_state.chamfer_shown = None;
                ui_state.fillet_edges.clear();
                ui_state.regen = true;
            } else {
                ui_state.pending_chamfer = Some(d.max(0.01));
            }
        }
        if let Some(mut which) = ui_state.pending_mirror {
            ui.heading("Mirror");
            let mut commit = false;
            let mut cancel = false;
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70))).clicked() {
                    commit = true;
                }
                if ui.add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55))).clicked() {
                    cancel = true;
                }
            });
            ui.separator();
            ui.label("Mirror plane");
            for (k, name) in [(0u8, "Front (XY)"), (1, "Top (XZ)"), (2, "Right (YZ)")] {
                if ui.radio(which == k, name).clicked() {
                    which = k;
                    ui_state.mirror_shown = None;
                }
            }
            ui.label(egui::RichText::new("The body is reflected across this plane and unioned with the original.").weak().small());
            ui.separator();
            if commit {
                ui_state.mirror_request = Some(which);
                ui_state.pending_mirror = None;
                ui_state.mirror_shown = None;
            } else if cancel {
                ui_state.pending_mirror = None;
                ui_state.mirror_shown = None;
                ui_state.regen = true;
            } else {
                ui_state.pending_mirror = Some(which);
            }
        }
        if let Some(mut op) = ui_state.pending.clone() {
            // PropertyManager laid out like SolidWorks' Boss-Extrude.
            // Guard the depth: a non-finite value would crash egui's DragValue.
            if !op.depth.is_finite() {
                op.depth = EXTRUDE_DISTANCE as f32;
            }
            op.depth = op.depth.clamp(0.1, 10_000.0);
            let mut keep = true;
            let mut commit = false;
            ui.horizontal(|ui| {
                ui.heading(match op.kind {
                    OpKind::Boss => "Boss-Extrude",
                    OpKind::Cut => "Cut-Extrude",
                });
            });
            // OK (green ✔) / Cancel (red ✗) row, as in the PropertyManager header.
            ui.horizontal(|ui| {
                if ui
                    .add(egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE))
                        .fill(egui::Color32::from_rgb(40, 140, 70)))
                    .clicked()
                {
                    commit = true;
                }
                if ui
                    .add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE))
                        .fill(egui::Color32::from_rgb(170, 55, 55)))
                    .clicked()
                {
                    keep = false;
                }
            });
            ui.separator();

            // From — the sketch plane (only option for now).
            egui::CollapsingHeader::new("From").default_open(true).show(ui, |ui| {
                ui.add_enabled(false, egui::Button::new("Sketch Plane             ▼"));
            });

            // Direction 1 — end condition, reverse, and depth (D1).
            egui::CollapsingHeader::new("Direction 1").default_open(true).show(ui, |ui| {
                ui.add_enabled(false, egui::Button::new("Blind                    ▼"))
                    .on_disabled_hover_text("End condition (only Blind for now)");
                ui.checkbox(&mut op.reverse, "Reverse direction");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("D1").strong());
                    ui.add(
                        egui::DragValue::new(&mut op.depth).speed(0.1).range(0.1..=10_000.0).suffix(" mm"),
                    );
                });
            });

            // Direction 2 / Thin Feature — present (like SolidWorks) but not yet built.
            let mut off = false;
            egui::CollapsingHeader::new("Direction 2").default_open(false).show(ui, |ui| {
                ui.add_enabled(false, egui::Checkbox::new(&mut off, "Not implemented yet"));
            });
            egui::CollapsingHeader::new("Thin Feature").default_open(false).show(ui, |ui| {
                ui.add_enabled(false, egui::Checkbox::new(&mut off, "Not implemented yet"));
            });

            // Selected Contours — the closed regions this op will use (empty = all).
            egui::CollapsingHeader::new("Selected Contours").default_open(true).show(ui, |ui| {
                let nreg = session.sketch.regions().len();
                let picked: Vec<usize> =
                    session.selected_contours.iter().copied().filter(|&i| i < nreg).collect();
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_min_width(190.0);
                    if picked.is_empty() {
                        ui.colored_label(
                            egui::Color32::from_rgb(150, 180, 255),
                            format!("All contours ({nreg})"),
                        );
                    } else {
                        for &i in &picked {
                            ui.colored_label(
                                egui::Color32::from_rgb(150, 180, 255),
                                format!("Contour {}", i + 1),
                            );
                        }
                    }
                });
                ui.label(
                    egui::RichText::new("Select tool (S): click closed areas to pick contours.")
                        .italics()
                        .weak()
                        .small(),
                );
                if !picked.is_empty() && ui.button("Clear contours").clicked() {
                    session.selected_contours.clear();
                }
            });

            if commit {
                // Reverse flips the sweep direction (signed distance).
                let d = if op.reverse { -(op.depth as f64) } else { op.depth as f64 };
                session.op_request = Some(match op.kind {
                    OpKind::Boss => SolidOp::Boss(d),
                    OpKind::Cut => SolidOp::Cut(d),
                });
                keep = false;
            }
            ui_state.pending = if keep { Some(op) } else { None };
        } else if in_sketch {
            // Sketch panel: edit dimensions / relations, then Apply to re-solve.
            ui.heading("Sketch");

            // Live dimension entry while drawing a line/circle.
            if let Some(start) = session.pending {
                let live = session.cursor_uv.map(|c| (c - start).length()).unwrap_or(0.0);
                let mut focus_now = session.request_live_focus;
                match session.tool {
                    Tool::Line => {
                        ui.label(egui::RichText::new("Line length").strong());
                        ui.horizontal(|ui| {
                            let resp = ui.add(
                                egui::DragValue::new(&mut session.live_buf).speed(0.05).range(0.0..=10_000.0).suffix(" mm"),
                            );
                            if focus_now {
                                resp.request_focus();
                                focus_now = false;
                            }
                            if !resp.has_focus() {
                                session.live_buf = live;
                            }
                            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                let len = session.live_buf;
                                commit_line_length(&mut session, len);
                            }
                        });
                        ui.label(egui::RichText::new("type a length + Enter, or click the end point").italics().weak());
                        ui.separator();
                    }
                    Tool::Circle => {
                        // Snap the live radius to a matching existing circular edge.
                        let snapped_live = snap_radius(live, &session.reference_circles, session.snap_dist.max(SNAP));
                        ui.label(egui::RichText::new("Circle radius").strong());
                        ui.horizontal(|ui| {
                            let resp = ui.add(
                                egui::DragValue::new(&mut session.live_buf).speed(0.05).range(0.01..=10_000.0).suffix(" mm"),
                            );
                            if focus_now {
                                resp.request_focus();
                                focus_now = false;
                            }
                            if !resp.has_focus() {
                                session.live_buf = snapped_live;
                            }
                            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                let r = session.live_buf;
                                commit_circle_radius(&mut session, r);
                            }
                        });
                        ui.label(
                            egui::RichText::new(format!("circumference {:.1} mm", std::f32::consts::TAU * session.live_buf))
                                .weak(),
                        );
                        ui.separator();
                    }
                    Tool::Rectangle => {
                        if let Some(cur) = session.cursor_uv {
                            let (w, h) = ((cur.x - start.x).abs(), (cur.y - start.y).abs());
                            ui.label(egui::RichText::new(format!("Rectangle {w:.1} × {h:.1} mm")).strong());
                            ui.separator();
                        }
                    }
                    _ => {}
                }
                session.request_live_focus = focus_now;
            }

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(session.needs_apply, egui::Button::new("Apply").fill(egui::Color32::from_rgb(40, 110, 70)))
                    .on_hover_text("Re-solve the sketch with your edited dimensions / relations")
                    .clicked()
                {
                    session.dirty = true;
                    session.needs_apply = false;
                }
                if session.needs_apply {
                    ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "unapplied changes");
                } else {
                    ui.colored_label(egui::Color32::from_rgb(90, 200, 120), "up to date");
                }
            });
            ui.separator();

            // All driving dimensions (distance / radius / angle / point-line), each with a
            // red ✕ to delete it.
            let dims: Vec<usize> = session
                .sketch
                .constraints
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    matches!(
                        c,
                        Constraint::Distance { .. }
                            | Constraint::Radius { .. }
                            | Constraint::Angle { .. }
                            | Constraint::PointLineDistance { .. }
                    )
                    .then_some(i)
                })
                .collect();
            if dims.is_empty() {
                ui.label(
                    egui::RichText::new("Dimension tool: click a line, or two points.")
                        .italics()
                        .weak(),
                );
            } else {
                ui.label(egui::RichText::new("Dimensions").strong());
                let mut changed = false;
                let mut delete_dim: Option<usize> = None;
                let sel = session.selected_dim;
                egui::ScrollArea::vertical().max_height(220.0).id_salt("dims").show(ui, |ui| {
                    for (k, i) in dims.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui
                                .small_button(egui::RichText::new("X").color(egui::Color32::from_rgb(220, 50, 50)).strong())
                                .on_hover_text("Delete dimension")
                                .clicked()
                            {
                                delete_dim = Some(*i);
                            }
                            let tag = ui.selectable_label(sel == Some(*i), format!("D{}", k + 1));
                            if tag.clicked() {
                                session.selected_dim = if sel == Some(*i) { None } else { Some(*i) };
                            }
                            match session.sketch.constraints.get_mut(*i) {
                                Some(Constraint::Distance { value, .. })
                                | Some(Constraint::PointLineDistance { value, .. }) => {
                                    if ui.add(egui::DragValue::new(value).speed(0.05).range(0.01..=10_000.0).suffix(" mm")).changed() {
                                        changed = true;
                                    }
                                }
                                Some(Constraint::Radius { value, diameter, .. }) => {
                                    let dia = *diameter;
                                    let mut shown = if dia { *value * 2.0 } else { *value };
                                    let prefix = if dia { "Ø" } else { "R" };
                                    if ui.add(egui::DragValue::new(&mut shown).speed(0.05).range(0.01..=10_000.0).prefix(prefix).suffix(" mm")).changed() {
                                        if let Some(Constraint::Radius { value, .. }) = session.sketch.constraints.get_mut(*i) {
                                            *value = if dia { shown * 0.5 } else { shown };
                                        }
                                        changed = true;
                                    }
                                }
                                Some(Constraint::Angle { value, .. }) => {
                                    let mut deg = value.to_degrees();
                                    if ui.add(egui::DragValue::new(&mut deg).speed(0.5).range(0.1..=359.9).suffix("°")).changed() {
                                        if let Some(Constraint::Angle { value, .. }) = session.sketch.constraints.get_mut(*i) {
                                            *value = deg.to_radians();
                                        }
                                        changed = true;
                                    }
                                }
                                _ => {}
                            }
                        });
                    }
                });
                if let Some(i) = delete_dim {
                    if i < session.sketch.constraints.len() {
                        session.sketch.constraints.remove(i);
                    }
                    // Constraint indices shift after a remove → drop any stale references.
                    session.selected_dim = None;
                    session.dim_edit = None;
                    changed = true;
                }
                if changed {
                    session.needs_apply = true;
                }
            }

            // Circle radii (editable — a circle's radius is its dimension).
            let circles: Vec<usize> = session
                .sketch
                .entities
                .iter()
                .enumerate()
                .filter_map(|(i, e)| matches!(e, SketchEntity::Circle { .. }).then_some(i))
                .collect();
            if !circles.is_empty() {
                ui.label(egui::RichText::new("Circles").strong());
                let mut changed = false;
                for (k, i) in circles.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("C{}", k + 1));
                        if let Some(SketchEntity::Circle { radius, .. }) = session.sketch.entities.get_mut(*i) {
                            if ui
                                .add(egui::DragValue::new(radius).speed(0.05).range(0.01..=10_000.0).prefix("R ").suffix(" mm"))
                                .changed()
                            {
                                changed = true;
                            }
                        }
                    });
                }
                if changed {
                    session.needs_apply = true;
                }
            }

            // Geometric constraints on the selected entities (Select tool).
            ui.separator();
            let sel = session.selected_entities.clone();
            let lines: Vec<(usize, usize)> = sel.iter().filter_map(|&i| entity_line(&session.sketch, i)).collect();
            let circles: Vec<(usize, f64)> = sel.iter().filter_map(|&i| entity_circle(&session.sketch, i)).collect();
            let two_lines = lines.len() == 2;
            let many_lines = lines.len() >= 2;
            let many_circles = circles.len() >= 2;
            let line_circle = lines.len() == 1 && circles.len() == 1;
            ui.label(egui::RichText::new(format!("Selected Entities — {}", sel.len())).strong());
            if sel.is_empty() {
                ui.label(
                    egui::RichText::new("Select tool: click lines/circles to relate (2 lines, 2 circles, or a line + circle).")
                        .italics()
                        .weak(),
                );
            }
            // ---- Add Relations ----
            let mut applied: Option<Constraint> = None;
            let mut dist_request = false;
            let mut angle_request = false;
            let mut equal_request = false;
            // Entity indices of the selected lines (parallel to `lines`), so we can tell
            // which is a body-edge reference line when dimensioning.
            let line_entities: Vec<usize> =
                sel.iter().copied().filter(|&i| entity_line(&session.sketch, i).is_some()).collect();
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(two_lines, egui::Button::new("Parallel")).clicked() {
                    applied = Some(Constraint::Parallel(lines[0].0, lines[0].1, lines[1].0, lines[1].1));
                }
                if ui.add_enabled(two_lines, egui::Button::new("Perpendicular")).clicked() {
                    applied = Some(Constraint::Perpendicular(lines[0].0, lines[0].1, lines[1].0, lines[1].1));
                }
                if ui.add_enabled(many_lines || many_circles, egui::Button::new("Equal"))
                    .on_hover_text("Make all selected lines equal length (or all circles equal radius). If one is already dimensioned, that size drives the rest.")
                    .clicked()
                {
                    equal_request = true;
                }
                if ui.add_enabled(line_circle, egui::Button::new("Tangent")).clicked() {
                    applied = Some(Constraint::Tangent {
                        a: lines[0].0,
                        b: lines[0].1,
                        center: circles[0].0,
                        radius: circles[0].1,
                    });
                }
                if ui.add_enabled(two_lines, egui::Button::new("Distance"))
                    .on_hover_text("Dimension the perpendicular distance between two lines (e.g. a line off a body edge)")
                    .clicked()
                {
                    dist_request = true;
                }
                if ui.add_enabled(two_lines, egui::Button::new("Angle"))
                    .on_hover_text("Angle dimension between the two selected lines (the first stays fixed when edited)")
                    .clicked()
                {
                    angle_request = true;
                }
            });
            // Angle dimension between exactly the two selected lines (more reliable than the
            // click-to-convert flow, which can grab a nearby line). First line is the
            // reference that stays put when the angle is later edited.
            if angle_request && two_lines {
                if let Some(ci) = add_angle_dim(&mut session.sketch, line_entities[0], line_entities[1]) {
                    session.selected_entities.clear();
                    open_dim_edit(&mut session, ci, None);
                    session.needs_apply = true;
                }
            }
            // Distance dimension between the two selected lines (point-to-line). The body
            // edge (a reference line) is the base; an endpoint of the other line is driven.
            if dist_request && two_lines {
                let is_ref = |i: usize| matches!(session.sketch.entities.get(i), Some(SketchEntity::Line { reference: true, .. }));
                let base_i = if is_ref(line_entities[0]) { 0 } else if is_ref(line_entities[1]) { 1 } else { 0 };
                let other_i = 1 - base_i;
                let (ba, bb) = lines[base_i];
                let pp = lines[other_i].0;
                let a2 = Vec2::new(session.sketch.points[ba].x as f32, session.sketch.points[ba].y as f32);
                let b2 = Vec2::new(session.sketch.points[bb].x as f32, session.sketch.points[bb].y as f32);
                let p2 = Vec2::new(session.sketch.points[pp].x as f32, session.sketch.points[pp].y as f32);
                let (foot, _) = point_line_geometry(p2, a2, b2);
                let value = (p2 - foot).length().max(0.001) as f64;
                let ci = session.sketch.constraints.len();
                session.sketch.constraints.push(Constraint::PointLineDistance { p: pp, a: ba, b: bb, value, offset: 0.0 });
                session.selected_entities.clear();
                open_dim_edit(&mut session, ci, None);
                session.needs_apply = true;
            }
            // Equal across the whole selection. Whichever item already carries a size
            // dimension drives the rest through the chain; >2 sized items is ambiguous,
            // so we refuse it rather than fight the solver.
            if equal_request {
                if many_circles {
                    let defined = circles.iter().filter(|(c, _)| has_radius(&session.sketch, *c)).count();
                    if defined > 2 {
                        ui_state.last_error = Some(
                            "Equal: more than two selected circles already have a radius — too many defined items. Remove a radius dimension and try again.".into(),
                        );
                    } else {
                        let first = circles[0].0;
                        for (c, _) in circles.iter().skip(1) {
                            let dup = session.sketch.constraints.iter().any(|k| {
                                matches!(k, Constraint::EqualRadius { a, b } if (*a == first && *b == *c) || (*a == *c && *b == first))
                            });
                            if !dup {
                                session.sketch.constraints.push(Constraint::EqualRadius { a: first, b: *c });
                            }
                        }
                        session.selected_entities.clear();
                        session.needs_apply = true;
                    }
                } else if many_lines {
                    let defined = lines.iter().filter(|(a, b)| has_distance(&session.sketch, *a, *b)).count();
                    if defined > 2 {
                        ui_state.last_error = Some(
                            "Equal: more than two selected lines already have a length dimension — too many defined items. Remove a dimension and try again.".into(),
                        );
                    } else {
                        let first = lines[0];
                        for l in lines.iter().skip(1) {
                            let dup = session.sketch.constraints.iter().any(|k| {
                                matches!(k, Constraint::Equal(a, b, c, d)
                                    if (*a == first.0 && *b == first.1 && *c == l.0 && *d == l.1)
                                        || (*a == l.0 && *b == l.1 && *c == first.0 && *d == first.1))
                            });
                            if !dup {
                                session.sketch.constraints.push(Constraint::Equal(first.0, first.1, l.0, l.1));
                            }
                        }
                        session.selected_entities.clear();
                        session.needs_apply = true;
                    }
                }
            }

            // ---- Existing Relations on the selection (view + delete) ----
            let sel_points: std::collections::HashSet<usize> =
                sel.iter().flat_map(|&i| entity_points(&session.sketch, i)).collect();
            if !sel_points.is_empty() {
                let related: Vec<usize> = session
                    .sketch
                    .constraints
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| constraint_points(c).iter().any(|p| sel_points.contains(p)))
                    .map(|(i, _)| i)
                    .collect();
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Existing Relations").strong());
                if related.is_empty() {
                    ui.label(egui::RichText::new("none").italics().weak());
                }
                let mut delete: Option<usize> = None;
                egui::ScrollArea::vertical().max_height(150.0).id_salt("relations").show(ui, |ui| {
                    for &ci in &related {
                        ui.horizontal(|ui| {
                            if ui.small_button(egui::RichText::new("X").color(egui::Color32::from_rgb(220, 50, 50)).strong())
                                .on_hover_text("Delete relation").clicked() {
                                delete = Some(ci);
                            }
                            ui.label(constraint_label(&session.sketch.constraints[ci]));
                        });
                    }
                });
                if let Some(ci) = delete {
                    session.sketch.constraints.remove(ci);
                    session.needs_apply = true;
                }
            }

            if !sel.is_empty() && ui.button("Clear selection").clicked() {
                session.selected_entities.clear();
            }
            if let Some(c) = applied {
                session.sketch.constraints.push(c);
                session.selected_entities.clear();
                session.needs_apply = true;
            }
        } else {
            // ---- FeatureManager design tree (SolidWorks-style) ----
            let nplanes = doc.0.features.iter().filter(|f| matches!(f.kind, FeatureKind::Plane(_))).count();
            let rollback = doc.0.rollback;

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("⬡ Part").strong().size(15.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let shown = rollback.saturating_sub(nplanes);
                    let total = doc.0.features.len().saturating_sub(nplanes);
                    ui.label(egui::RichText::new(format!("{shown} / {total}")).weak().small())
                        .on_hover_text("Active features (drag the blue rollback bar to change)");
                });
            });
            ui.separator();

            let mut action: Option<TreeAction> = None;
            // Rows of solid features, with their on-screen rects, for the rollback bar.
            let mut feat_rows: Vec<(usize, egui::Rect)> = Vec::new();

            egui::ScrollArea::vertical().max_height(340.0).show(ui, |ui| {
                // Datum planes + origin, shown flat at the top like SolidWorks.
                for (_id, p) in doc.0.planes() {
                    ui.label(egui::RichText::new(format!("▱  {} Plane", p.name)).weak());
                }
                ui.label(egui::RichText::new("⊕  Origin").weak());
                ui.add_space(3.0);

                let (mut sk, mut ex, mut ct) = (0u32, 0u32, 0u32);
                for (i, f) in doc.0.features.iter().enumerate() {
                    let suppressed = i >= rollback;
                    let selected = ui_state.selected == Some(i);
                    let styled = |s: String| {
                        let mut rt = egui::RichText::new(s);
                        if selected {
                            rt = rt.color(egui::Color32::from_rgb(120, 180, 255)).strong();
                        }
                        if suppressed {
                            rt = rt.weak().strikethrough();
                        }
                        rt
                    };

                    let row = match &f.kind {
                        FeatureKind::Plane(_) => continue,
                        FeatureKind::Sketch { .. } => {
                            sk += 1;
                            let resp = ui.selectable_label(selected, styled(format!("✎ Sketch{sk}")));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit sketch").clicked() {
                                    action = Some(TreeAction::Edit(i));
                                    ui.close();
                                }
                                if ui.button("Extrude boss").clicked() {
                                    action = Some(TreeAction::ExtrudeBoss(i));
                                    ui.close();
                                }
                                if ui.button("Extrude cut").clicked() {
                                    action = Some(TreeAction::ExtrudeCut(i));
                                    ui.close();
                                }
                                if ui.button("Delete").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Extrude { distance, .. } | FeatureKind::Cut { distance, .. } => {
                            let (label, child, icon) = match &f.kind {
                                FeatureKind::Extrude { .. } => {
                                    ex += 1;
                                    (format!("Boss-Extrude{ex}  (h {distance:.1})"), format!("✎ Sketch of Extrude{ex}"), "⬢")
                                }
                                _ => {
                                    ct += 1;
                                    (format!("Cut-Extrude{ct}  (h {distance:.1})"), format!("✎ Sketch of Cut{ct}"), "⬣")
                                }
                            };
                            let resp = egui::CollapsingHeader::new(styled(format!("{icon} {label}")))
                                .id_salt(i)
                                .default_open(false)
                                .show(ui, |ui| {
                                    let child_resp =
                                        ui.selectable_label(false, egui::RichText::new(child).weak());
                                    child_resp.context_menu(|ui| {
                                        if ui.button("Edit sketch").clicked() {
                                            action = Some(TreeAction::Edit(i));
                                            ui.close();
                                        }
                                    });
                                });
                            if resp.header_response.clicked() {
                                ui_state.selected = Some(i);
                            }
                            resp.header_response.context_menu(|ui| {
                                if ui.button("Edit feature (depth)").clicked() {
                                    action = Some(TreeAction::Select(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.header_response.rect
                        }
                        FeatureKind::Fillet { radius, edges } => {
                            let scope = if edges.is_empty() { "all edges".to_string() } else { format!("{} edge(s)", edges.len()) };
                            let resp = ui.selectable_label(selected, styled(format!("⬤ Fillet  (r {radius:.2}, {scope})")));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Chamfer { distance, edges } => {
                            let resp = ui.selectable_label(selected, styled(format!("⬤ Chamfer  (d {distance:.2}, {} edge(s))", edges.len())));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Mirror { .. } => {
                            let resp = ui.selectable_label(selected, styled("◧ Mirror".to_string()));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                    };
                    feat_rows.push((i, row));
                }

                // ---- Draggable rollback bar, in the tree (SolidWorks-style) ----
                if !feat_rows.is_empty() {
                    let bar_y = match feat_rows.iter().filter(|(d, _)| *d < rollback).last() {
                        Some((_, last)) => last.bottom() + 2.0,
                        None => feat_rows[0].1.top() - 2.0,
                    };
                    let x = ui.max_rect().x_range();
                    let bar_rect = egui::Rect::from_x_y_ranges(x.clone(), (bar_y - 3.0)..=(bar_y + 3.0));
                    let resp = ui.interact(bar_rect, ui.id().with("rollback_bar"), egui::Sense::drag());
                    let col = egui::Color32::from_rgb(70, 140, 220);
                    ui.painter().hline(x, bar_y, egui::Stroke::new(2.0, col));
                    ui.painter().circle_filled(egui::pos2(bar_rect.left() + 7.0, bar_y), 4.0, col);
                    if resp.hovered() || resp.dragged() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
                    }
                    if resp.dragged() {
                        if let Some(pos) = resp.interact_pointer_pos() {
                            let mut new_rb = nplanes; // keep datum planes; suppress all solids
                            for (d, rect) in &feat_rows {
                                if rect.center().y < pos.y {
                                    new_rb = d + 1;
                                }
                            }
                            if new_rb != doc.0.rollback {
                                doc.0.rollback = new_rb;
                                ui_state.regen = true;
                            }
                        }
                    }
                }
            });

            // Apply the tree context-menu action (after the borrow above ends).
            if let Some(act) = action {
                match act {
                    TreeAction::Select(i) => {
                        // "Edit feature" → reopen the sketch and the PropertyManager
                        // prefilled with the feature's kind/depth (no separate editor).
                        let op = doc.0.features.get(i).and_then(|f| match &f.kind {
                            FeatureKind::Extrude { distance, .. } => Some((OpKind::Boss, *distance)),
                            FeatureKind::Cut { distance, .. } => Some((OpKind::Cut, *distance)),
                            _ => None,
                        });
                        if let Some((kind, dist)) = op {
                            ui_state.edit_sketch_request = Some(i);
                            ui_state.pending =
                                Some(PendingOp { kind, depth: (dist.abs() as f32).max(0.1), reverse: dist < 0.0 });
                        } else {
                            ui_state.selected = Some(i);
                        }
                    }
                    TreeAction::Edit(i) => ui_state.edit_sketch_request = Some(i),
                    TreeAction::ExtrudeBoss(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32, reverse: false });
                    }
                    TreeAction::ExtrudeCut(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32, reverse: false });
                    }
                    TreeAction::Delete(i) => {
                        if i < doc.0.features.len() {
                            history.snapshot(&doc.0);
                            doc.0.features.remove(i);
                            if doc.0.rollback > doc.0.features.len() {
                                doc.0.rollback = doc.0.features.len();
                            }
                            ui_state.selected = None;
                            ui_state.regen = true;
                        }
                    }
                }
            }

            // (Feature depth is edited in the PropertyManager via "Edit feature".)
            if doc.0.features.iter().any(|f| matches!(f.kind, FeatureKind::Extrude { .. } | FeatureKind::Cut { .. })) {
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("Right-click a feature → Edit feature / Edit sketch.")
                        .italics()
                        .weak()
                        .small(),
                );
            }
        }
      });
    });

    // ---- Bottom status bar ----
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            let mode = match &session.plane {
                Some(ap) => format!("Sketch: {}", ap.name),
                None => "View".to_string(),
            };
            ui.label(mode);
            ui.separator();
            ui.label(format!("Tool: {}", session.tool.label()));
            ui.separator();
            match session.cursor_uv {
                Some(uv) => ui.label(format!("x {:.2}  y {:.2}", uv.x, uv.y)),
                None => ui.label("x —  y —"),
            };
            ui.separator();
            if in_sketch {
                if has_profile {
                    ui.colored_label(egui::Color32::from_rgb(90, 200, 120), "● profile ready");
                } else {
                    ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "○ profile open");
                }
                ui.separator();
                let nreg = session.sketch.regions().len();
                if nreg > 0 {
                    let picked = session.selected_contours.iter().filter(|&&i| i < nreg).count();
                    let sel = if picked == 0 { format!("all {nreg}") } else { format!("{picked}/{nreg}") };
                    ui.label(format!("contours {sel}"));
                    if nreg > 1 {
                        ui.label(
                            egui::RichText::new("(Select tool: click closed areas)")
                                .weak()
                                .small(),
                        );
                    }
                    ui.separator();
                }
            }
            ui.label(format!("{} features", doc.0.features.len()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label("mm");
            });
        });
    });

    // ---- Right-click viewport context menu (view alignment) ----
    // Only over the 3D viewport — not when right-clicking inside an egui panel
    // (the feature tree has its own context menus there).
    if ctx.input(|i| i.pointer.secondary_clicked()) && !ctx.is_pointer_over_area() {
        ui_state.context_pos = ctx.input(|i| i.pointer.interact_pos());
    }
    if let Some(pos) = ui_state.context_pos {
        let mut close = false;
        let mut action: Option<ViewAction> = None;
        egui::Area::new(egui::Id::new("viewport_ctx"))
            .fixed_pos(pos)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(160.0);
                    if in_sketch {
                        if ui.button("Normal to sketch").clicked() {
                            action = Some(ViewAction::NormalToSketch);
                            close = true;
                        }
                        ui.separator();
                    }
                    for (label, act) in [
                        ("Front", ViewAction::Normal(Vec3::Z)),
                        ("Top", ViewAction::Normal(Vec3::Y)),
                        ("Right", ViewAction::Normal(Vec3::X)),
                        ("Isometric", ViewAction::Iso),
                        ("Zoom to fit", ViewAction::Fit),
                    ] {
                        if ui.button(label).clicked() {
                            action = Some(act);
                            close = true;
                        }
                    }
                    if in_sketch {
                        ui.separator();
                        if ui.button("Exit sketch").clicked() {
                            action = Some(ViewAction::ExitSketch);
                            close = true;
                        }
                    }
                });
            });

        // Apply the chosen action after the menu closure (avoids double borrows).
        if let Some(act) = action {
            match act {
                ViewAction::NormalToSketch => {
                    if let (Some(ap), Ok((mut tf, mut orbit))) =
                        (session.plane.clone(), cam_q.single_mut())
                    {
                        look_along(&mut orbit, ap.origin, ap.n);
                        *tf = camera_transform(&orbit);
                    }
                }
                ViewAction::Normal(n) => {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        look_along(&mut orbit, Vec3::ZERO, n);
                        *tf = camera_transform(&orbit);
                    }
                }
                ViewAction::Iso => {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        orbit.focus = Vec3::ZERO;
                        orbit.yaw = 0.8;
                        orbit.pitch = -0.55;
                        *tf = camera_transform(&orbit);
                    }
                }
                ViewAction::Fit => {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        let (focus, radius) = fit_view(&part);
                        orbit.focus = focus;
                        orbit.radius = radius;
                        *tf = camera_transform(&orbit);
                    }
                }
                ViewAction::ExitSketch => {
                    session.exit_request = true;
                }
            }
        }

        // Close on any button, a left click elsewhere, or Escape.
        if close || ctx.input(|i| i.pointer.primary_clicked() || i.key_pressed(egui::Key::Escape)) {
            ui_state.context_pos = None;
        }
    }

    // Dimension value labels floating at each dimension's midpoint in the viewport.
    // Clone the plane so the centre-line label code below can also mutate `session`.
    if let (Some(ap), Ok((camera, cam_gt))) = (session.plane.clone(), cam_read.single()) {
        // Every measurement label is interactive: single-click selects it (Delete or the
        // panel ✕ removes it), double-click opens its Modify box. Returns the label's
        // response so the caller can act on the click.
        let sel_dim = session.selected_dim;
        let label_at = |ctx: &egui::Context, id: egui::Id, world: Vec3, text: String, selected: bool| -> Option<egui::Response> {
            camera.world_to_viewport(cam_gt, world).ok().map(|screen| {
                let col = if selected {
                    egui::Color32::from_rgb(255, 205, 90)
                } else {
                    egui::Color32::from_rgb(150, 215, 255)
                };
                egui::Area::new(id)
                    .fixed_pos(egui::pos2(screen.x, screen.y))
                    .order(egui::Order::Foreground)
                    .show(ctx, |ui| {
                        // Never wrap — keep the value on one line even near a screen edge
                        // (otherwise "90.0°" stacks one character per row).
                        ui.add(
                            egui::Label::new(egui::RichText::new(text).color(col).strong())
                                .wrap_mode(egui::TextWrapMode::Extend)
                                .sense(egui::Sense::click()),
                        )
                    })
                    .inner
            })
        };
        // Deferred click actions (the constraint loop holds an immutable borrow of session).
        let mut dim_action: Option<(usize, bool)> = None; // (constraint index, double-clicked?)
        let mut dia_action: Option<usize> = None; // circle centre to attach a Ø dim to
        let mut act = |resp: Option<egui::Response>, ci: usize, slot: &mut Option<(usize, bool)>| {
            if let Some(r) = resp {
                if r.double_clicked() {
                    *slot = Some((ci, true));
                } else if r.clicked() {
                    *slot = Some((ci, false));
                }
            }
        };
        // Centres that carry a driving radius/diameter dimension (so the generic Ø
        // callout below can defer to the explicit dimension and avoid double labels).
        let mut dimensioned_centers: Vec<usize> = Vec::new();
        for (k, c) in session.sketch.constraints.iter().enumerate() {
            let on = sel_dim == Some(k);
            match c {
                Constraint::Distance { a, b, value, offset, axis } => {
                    if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                        let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                        let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                        let (_, _, lab) = distance_dim_geometry(a2, b2, *offset as f32, *axis);
                        act(label_at(ctx, egui::Id::new(("dimlabel", k)), ap.to_world(lab), format!("{value:.2}"), on), k, &mut dim_action);
                    }
                }
                Constraint::Radius { center, value, diameter } => {
                    dimensioned_centers.push(*center);
                    if let Some(c) = session.sketch.points.get(*center) {
                        let cu = Vec2::new(c.x as f32, c.y as f32);
                        let r = *value as f32;
                        let edge = cu + Vec2::new(r * 0.707, r * 0.707);
                        let text = if *diameter { format!("Ø{:.2}", value * 2.0) } else { format!("R{value:.2}") };
                        act(label_at(ctx, egui::Id::new(("radlabel", k)), ap.to_world(edge), text, on), k, &mut dim_action);
                    }
                }
                Constraint::Angle { a, b, c, d, value, offset } => {
                    let pts = (
                        session.sketch.points.get(*a),
                        session.sketch.points.get(*b),
                        session.sketch.points.get(*c),
                        session.sketch.points.get(*d),
                    );
                    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = pts {
                        let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                        let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                        let c2 = Vec2::new(pc.x as f32, pc.y as f32);
                        let d2 = Vec2::new(pd.x as f32, pd.y as f32);
                        let (_, lab) = angle_dim_geometry(a2, b2, c2, d2, *offset as f32);
                        act(label_at(ctx, egui::Id::new(("anglabel", k)), ap.to_world(lab), format!("{:.1}°", value.to_degrees()), on), k, &mut dim_action);
                    }
                }
                Constraint::PointLineDistance { p, a, b, value, .. } => {
                    let pts = (session.sketch.points.get(*p), session.sketch.points.get(*a), session.sketch.points.get(*b));
                    if let (Some(pp), Some(pa), Some(pb)) = pts {
                        let p2 = Vec2::new(pp.x as f32, pp.y as f32);
                        let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                        let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                        let (_, lab) = point_line_geometry(p2, a2, b2);
                        act(label_at(ctx, egui::Id::new(("pldlabel", k)), ap.to_world(lab), format!("{value:.2}"), on), k, &mut dim_action);
                    }
                }
                _ => {}
            }
        }
        // Diameter callout (Ø) for circles without an explicit radius dimension. Double-
        // click promotes it to a real (editable, deletable) diameter dimension.
        for (k, e) in session.sketch.entities.iter().enumerate() {
            if let SketchEntity::Circle { center, radius, .. } = e {
                if dimensioned_centers.contains(center) {
                    continue;
                }
                if let Some(c) = session.sketch.points.get(*center) {
                    let cu = Vec2::new(c.x as f32, c.y as f32);
                    let edge = cu + Vec2::new(*radius as f32 * 0.707, *radius as f32 * 0.707);
                    let resp = label_at(ctx, egui::Id::new(("dialabel", k)), ap.to_world(edge), format!("Ø{:.2}", radius * 2.0), false);
                    if resp.is_some_and(|r| r.double_clicked()) {
                        dia_action = Some(*center);
                    }
                }
            }
        }
        // Centre-line half-length callouts: each is an interactive label sat at the middle
        // of its half. Double-click one to open a Modify box and type an exact length.
        if let Some([a, mid, b]) = session.center_line {
            let n = session.sketch.points.len();
            if a < n && mid < n && b < n {
                let pv = |i: usize| Vec2::new(session.sketch.points[i].x as f32, session.sketch.points[i].y as f32);
                let (pm, pa, pb) = (pv(mid), pv(a), pv(b));
                let mut open_edit: Option<(usize, f32)> = None;
                let mut half = |end: usize, ep: Vec2| {
                    let len = (ep - pm).length();
                    let labpos = (pm + ep) * 0.5;
                    if let Ok(screen) = camera.world_to_viewport(cam_gt, ap.to_world(labpos)) {
                        egui::Area::new(egui::Id::new(("cllabel", end)))
                            .fixed_pos(egui::pos2(screen.x, screen.y))
                            .order(egui::Order::Foreground)
                            .show(ctx, |ui| {
                                let resp = ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!("{len:.2}"))
                                            .color(egui::Color32::from_rgb(210, 160, 245))
                                            .strong(),
                                    )
                                    .sense(egui::Sense::click()),
                                );
                                if resp.double_clicked() {
                                    open_edit = Some((end, len));
                                }
                            });
                    }
                };
                half(a, pa);
                half(b, pb);
                if let Some(e) = open_edit {
                    session.center_line_edit = Some(e);
                }
            } else {
                session.center_line = None;
                session.center_line_edit = None;
            }
        }
        // The centre-line length Modify box (opened by double-clicking a half label).
        if let Some((end, mut buf)) = session.center_line_edit {
            if let Some([a, mid, b]) = session.center_line {
                let valid = end == a || end == b;
                let pv = |i: usize| Vec2::new(session.sketch.points[i].x as f32, session.sketch.points[i].y as f32);
                let labpos = valid.then(|| (pv(mid) + pv(end)) * 0.5);
                let screen = labpos.and_then(|uv| camera.world_to_viewport(cam_gt, ap.to_world(uv)).ok());
                if let Some(s) = screen {
                    let r = ctx.screen_rect();
                    let pos = egui::pos2(
                        s.x.clamp(r.left() + 8.0, (r.right() - 120.0).max(r.left() + 8.0)),
                        s.y.clamp(r.top() + 96.0, (r.bottom() - 40.0).max(r.top() + 96.0)),
                    );
                    let mut close = false;
                    egui::Area::new(egui::Id::new("cl_modify"))
                        .fixed_pos(pos)
                        .order(egui::Order::Foreground)
                        .show(ctx, |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let resp = ui.add_sized(
                                        egui::vec2(78.0, ui.spacing().interact_size.y),
                                        egui::DragValue::new(&mut buf).speed(0.1).range(0.001..=1_000_000.0).max_decimals(2).suffix(" mm"),
                                    );
                                    resp.request_focus();
                                    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                                    if ui.button("OK").clicked() || enter {
                                        close = true;
                                    }
                                });
                            });
                        });
                    // Slide the endpoint to the typed length along its current direction.
                    let m = pv(mid);
                    let e = pv(end);
                    let dir = (e - m).normalize_or_zero();
                    let dir = if dir == Vec2::ZERO { Vec2::X } else { dir };
                    let np = m + dir * buf.max(0.001);
                    session.sketch.points[end].x = np.x as f64;
                    session.sketch.points[end].y = np.y as f64;
                    session.center_line_len = Some(buf.max(0.001));
                    session.dirty = true;
                    session.center_line_edit = Some((end, buf));
                    if close || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                        session.center_line_edit = None;
                        session.needs_apply = true;
                    }
                } else {
                    session.center_line_edit = None;
                }
            } else {
                session.center_line_edit = None;
            }
        }
        // Apply the deferred measurement-label clicks now that the read borrows are gone.
        if let Some((ci, double)) = dim_action {
            if double {
                session.selected_dim = Some(ci);
                open_dim_edit(&mut session, ci, None);
            } else {
                // Toggle selection.
                session.selected_dim = if session.selected_dim == Some(ci) { None } else { Some(ci) };
            }
        }
        if let Some(center) = dia_action {
            let radius = session
                .sketch
                .entities
                .iter()
                .find_map(|e| match e {
                    SketchEntity::Circle { center: c, radius, .. } if *c == center => Some(*radius),
                    _ => None,
                })
                .unwrap_or(1.0);
            if let Some(ci) = add_radius_dim(&mut session.sketch, center, radius) {
                session.selected_dim = Some(ci);
                open_dim_edit(&mut session, ci, None);
            }
        }
    }

    // ---- Modify box: type the exact value for a just-placed dimension ----
    if let Some(ci) = session.dim_edit {
        // A snapshot of which kind of dimension this is, so the box can show the right
        // units, toggle buttons, and label position without holding a borrow.
        #[derive(Clone, Copy)]
        enum DimKind {
            Distance(DimAxis),
            Radius(bool), // diameter?
            Angle,
            PointLine,
        }
        let kind = match session.sketch.constraints.get(ci) {
            Some(Constraint::Distance { axis, .. }) => Some(DimKind::Distance(*axis)),
            Some(Constraint::Radius { diameter, .. }) => Some(DimKind::Radius(*diameter)),
            Some(Constraint::Angle { .. }) => Some(DimKind::Angle),
            Some(Constraint::PointLineDistance { .. }) => Some(DimKind::PointLine),
            _ => None,
        };
        // Label/box anchor in plane uv, computed from the constraint's points.
        let pt = |i: usize| session.sketch.points.get(i).copied().map(|p| Vec2::new(p.x as f32, p.y as f32));
        let label_uv = match session.sketch.constraints.get(ci) {
            Some(Constraint::Distance { a, b, offset, axis, .. }) => match (pt(*a), pt(*b)) {
                (Some(a2), Some(b2)) => Some(distance_dim_geometry(a2, b2, *offset as f32, *axis).2),
                _ => None,
            },
            Some(Constraint::Radius { center, value, .. }) => pt(*center).map(|cu| {
                let r = *value as f32;
                cu + Vec2::new(r * 0.707, r * 0.707)
            }),
            Some(Constraint::Angle { a, b, c, d, offset, .. }) => {
                match (pt(*a), pt(*b), pt(*c), pt(*d)) {
                    (Some(a2), Some(b2), Some(c2), Some(d2)) => {
                        Some(angle_dim_geometry(a2, b2, c2, d2, *offset as f32).1)
                    }
                    _ => None,
                }
            }
            Some(Constraint::PointLineDistance { p, a, b, .. }) => match (pt(*p), pt(*a), pt(*b)) {
                (Some(pp), Some(a2), Some(b2)) => Some(point_line_geometry(pp, a2, b2).1),
                _ => None,
            },
            _ => None,
        };
        let screen_pos = match (kind, session.plane.as_ref(), cam_read.single().ok(), label_uv) {
            (Some(kind), Some(ap), Some((camera, cam_gt)), Some(uv)) => camera
                .world_to_viewport(cam_gt, ap.to_world(uv))
                .ok()
                .map(|s| {
                    // Keep the Modify box on-screen (below the toolbar, inside the right
                    // edge) even when the dimension itself sits near/off a screen corner.
                    let r = ctx.screen_rect();
                    let x = s.x.clamp(r.left() + 8.0, (r.right() - 175.0).max(r.left() + 8.0));
                    let y = s.y.clamp(r.top() + 96.0, (r.bottom() - 40.0).max(r.top() + 96.0));
                    (kind, egui::pos2(x, y))
                }),
            _ => None,
        };
        if let Some((kind, pos)) = screen_pos {
            let mut close = false;
            // What the box edits is the *displayed* number: a diameter shows 2·r, an
            // angle shows degrees. Convert back to the stored value on apply.
            let (suffix, speed) = match kind {
                DimKind::Angle => ("°", 1.0),
                _ => (" mm", 0.1),
            };
            // Re-set dim_buf from the displayed value when an axis/diameter toggle has
            // just changed the meaning of the number; signalled via dim_edit_focus path.
            egui::Area::new(egui::Id::new("dim_modify"))
                .fixed_pos(pos)
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        // Keep the whole row on one line so the number isn't wrapped.
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        ui.horizontal(|ui| {
                            // A fixed, roomy width so the value (e.g. "100.00 mm") reads
                            // inline instead of spilling decimals onto a second line.
                            let resp = ui.add_sized(
                                egui::vec2(78.0, ui.spacing().interact_size.y),
                                egui::DragValue::new(&mut session.dim_buf)
                                    .speed(speed)
                                    .range(0.001..=1_000_000.0)
                                    .max_decimals(2)
                                    .suffix(suffix),
                            );
                            if session.dim_edit_focus {
                                resp.request_focus();
                                session.dim_edit_focus = false;
                            }
                            if resp.changed() {
                                let v = session.dim_buf.max(0.001);
                                // For an angle, keep the *first* (reference) line fixed so
                                // only the second line swings to the new angle — like SW.
                                let mut angle_ref: Option<[usize; 2]> = None;
                                match session.sketch.constraints.get_mut(ci) {
                                    Some(Constraint::Distance { value, .. }) => *value = v,
                                    Some(Constraint::Radius { value, diameter, .. }) => {
                                        *value = if *diameter { v * 0.5 } else { v };
                                    }
                                    Some(Constraint::Angle { value, a, b, .. }) => {
                                        *value = (v as f64).to_radians();
                                        angle_ref = Some([*a, *b]);
                                    }
                                    Some(Constraint::PointLineDistance { value, .. }) => *value = v,
                                    _ => {}
                                }
                                match angle_ref {
                                    Some(fixed) => session.sketch.solve_with_fixed(&fixed),
                                    None => session.sketch.solve(),
                                }
                            }
                            // Per-kind toggle controls (Ø/R, or aligned/H/V).
                            match kind {
                                DimKind::Radius(diameter) => {
                                    let mut new_diam: Option<bool> = None;
                                    if ui.selectable_label(diameter, "Ø").on_hover_text("Diameter").clicked() {
                                        new_diam = Some(true);
                                    }
                                    if ui.selectable_label(!diameter, "R").on_hover_text("Radius").clicked() {
                                        new_diam = Some(false);
                                    }
                                    if let Some(nd) = new_diam {
                                        let mut buf = session.dim_buf;
                                        if let Some(Constraint::Radius { value, diameter, .. }) = session.sketch.constraints.get_mut(ci) {
                                            *diameter = nd;
                                            buf = if nd { *value * 2.0 } else { *value };
                                        }
                                        session.dim_buf = buf;
                                    }
                                }
                                DimKind::Distance(axis) => {
                                    let mut new_axis: Option<DimAxis> = None;
                                    if ui.selectable_label(axis == DimAxis::Aligned, "⤢").on_hover_text("Aligned").clicked() {
                                        new_axis = Some(DimAxis::Aligned);
                                    }
                                    if ui.selectable_label(axis == DimAxis::Horizontal, "↔").on_hover_text("Horizontal").clicked() {
                                        new_axis = Some(DimAxis::Horizontal);
                                    }
                                    if ui.selectable_label(axis == DimAxis::Vertical, "↕").on_hover_text("Vertical").clicked() {
                                        new_axis = Some(DimAxis::Vertical);
                                    }
                                    if let Some(na) = new_axis {
                                        set_distance_axis(&mut session, ci, na);
                                    }
                                }
                                DimKind::Angle | DimKind::PointLine => {}
                            }
                            if ui.small_button("✔").on_hover_text("Apply (Enter)").clicked() {
                                close = true;
                            }
                        });
                    });
                });
            if close || ctx.input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Escape)) {
                session.dim_edit = None;
            }
        } else {
            session.dim_edit = None; // can't place the box → just drop it
        }
    }

    // ---- Error banner: a failed operation (e.g. a boolean the kernel rejected) ----
    if let Some(msg) = ui_state.last_error.clone() {
        let mut dismiss = false;
        let screen = ctx.screen_rect();
        egui::Area::new(egui::Id::new("error_banner"))
            .order(egui::Order::Foreground)
            .fixed_pos(egui::pos2(screen.center().x - 300.0, screen.top() + 48.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .fill(egui::Color32::from_rgb(80, 22, 22))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(220, 90, 90)))
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.set_max_width(600.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("⚠").size(18.0).color(egui::Color32::from_rgb(255, 170, 80)));
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(255, 225, 225)).strong());
                            if ui.small_button("✕").on_hover_text("Dismiss").clicked() {
                                dismiss = true;
                            }
                        });
                    });
            });
        if dismiss {
            ui_state.last_error = None;
        }
    }

    blocking.0 = ctx.wants_pointer_input() || ctx.is_pointer_over_area();
    blocking.1 = ctx.wants_keyboard_input();
    Ok(())
}

// ---------------------------------------------------------------------------
// Ray / plane math + point helpers
// ---------------------------------------------------------------------------

fn ray_plane(ap: &ActivePlane, ray: &Ray3d) -> Option<(f32, Vec2)> {
    let dir = ray.direction.as_vec3();
    let denom = ap.n.dot(dir);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (ap.origin - ray.origin).dot(ap.n) / denom;
    if t <= 0.0 {
        return None;
    }
    let hit = ray.origin + dir * t;
    let d = hit - ap.origin;
    Some((t, Vec2::new(d.dot(ap.u), d.dot(ap.v))))
}

fn cursor_ray(window: &Window, camera: &Camera, cam_transform: &GlobalTransform) -> Option<Ray3d> {
    let cursor = window.cursor_position()?;
    camera.viewport_to_world(cam_transform, cursor).ok()
}

fn nearest_point(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, p) in sketch.points.iter().enumerate() {
        let d = Vec2::new(p.x as f32, p.y as f32).distance(uv);
        if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

fn get_or_add_point(sketch: &mut Sketch, uv: Vec2, snap: f32) -> usize {
    nearest_point(sketch, uv, snap).unwrap_or_else(|| sketch.add_point(uv.x as f64, uv.y as f64))
}

/// Where to place the endpoint `i` while it's being dragged: snap onto the nearest *other*
/// sketch point, body-projected reference point (corner/centre), or on-hover inference
/// point within the snap tolerance; otherwise the raw cursor `uv`.
fn snap_drag_target(session: &SketchSession, i: usize, uv: Vec2) -> Vec2 {
    let snap = session.snap_dist;
    let mut best: Option<(Vec2, f32)> = None;
    let mut consider = |p: Vec2| {
        let d = p.distance(uv);
        if d <= snap && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((p, d));
        }
    };
    for (j, p) in session.sketch.points.iter().enumerate() {
        if j != i {
            consider(Vec2::new(p.x as f32, p.y as f32));
        }
    }
    for p in &session.reference_points {
        consider(*p);
    }
    for p in &session.inference_points {
        consider(*p);
    }
    best.map(|(p, _)| p).unwrap_or(uv)
}

/// Like `get_or_add_point`, but if the position coincides with a body-projected
/// reference snap point (a corner/centre), the new point is *locked* there — so the
/// endpoint stays constrained to that 3D feature through later solves.
fn get_or_add_point_ref(session: &mut SketchSession, uv: Vec2, snap: f32) -> usize {
    if let Some(i) = nearest_point(&session.sketch, uv, snap) {
        return i;
    }
    let tol = (snap * 0.25).max(1e-3);
    let on_ref = session.reference_points.iter().any(|r| r.distance(uv) <= tol);
    if on_ref {
        session.sketch.add_fixed_point(uv.x as f64, uv.y as f64)
    } else {
        session.sketch.add_point(uv.x as f64, uv.y as f64)
    }
}

/// If point `p` sits on a sketch circle's rim (e.g. a line endpoint just snapped to it),
/// record a point-on-circle relation so the endpoint follows later radius/centre edits.
/// `tol` should be a hair under the snap distance so only genuine rim landings qualify.
fn maybe_add_point_on_circle(sketch: &mut Sketch, p: usize, tol: f32) {
    let pp = sketch.points[p];
    let on = sketch.entities.iter().find_map(|e| match e {
        SketchEntity::Circle { center, radius, .. } if *center != p => {
            let c = sketch.points[*center];
            let d = (((pp.x - c.x).powi(2) + (pp.y - c.y).powi(2)).sqrt() - *radius).abs() as f32;
            (d <= tol).then_some(*center)
        }
        _ => None,
    });
    if let Some(center) = on {
        let exists = sketch.constraints.iter().any(
            |c| matches!(c, Constraint::PointOnCircle { p: q, center: cc } if *q == p && *cc == center),
        );
        if !exists {
            sketch.constraints.push(Constraint::PointOnCircle { p, center });
        }
    }
}

/// Where point `p` landed on another sketch line, if anywhere within `tol`.
enum LineHit {
    /// On an endpoint (`point index`) — bind coincident so it stays put.
    Endpoint(usize),
    /// On the midpoint — bind with a Midpoint relation so it stays centred.
    Midpoint(usize, usize),
    /// On the span — pin with PointOnLine so it can still slide *along* the line.
    Span(usize, usize),
}

/// If point `p` lands on another sketch line, give the connection a real relation so it
/// survives later edits/moves. A span landing slides along the line (`PointOnLine`); a
/// snap/end/mid landing is pinned (`Coincident` / `Midpoint`). `tol` ≈ a hair under snap.
fn maybe_add_point_on_sketch_line(sketch: &mut Sketch, p: usize, tol: f32) {
    let pp = Vec2::new(sketch.points[p].x as f32, sketch.points[p].y as f32);
    let mut hit: Option<LineHit> = None;
    for e in &sketch.entities {
        if let SketchEntity::Line { a, b, reference: false, .. } = e {
            if *a == p || *b == p {
                continue; // p is an endpoint of this line — already structurally joined
            }
            let va = Vec2::new(sketch.points[*a].x as f32, sketch.points[*a].y as f32);
            let vb = Vec2::new(sketch.points[*b].x as f32, sketch.points[*b].y as f32);
            let len = (vb - va).length();
            if len < 1e-6 || closest_on_segment(pp, va, vb).distance(pp) > tol {
                continue;
            }
            // Parameter along the line (0 = a, 1 = b); classify the landing.
            let t = ((pp - va).dot(vb - va) / (len * len)).clamp(0.0, 1.0);
            let edge_t = (tol / len).min(0.25); // "near an end" window, in parameter units
            hit = Some(if t <= edge_t {
                LineHit::Endpoint(*a)
            } else if t >= 1.0 - edge_t {
                LineHit::Endpoint(*b)
            } else if ((t - 0.5).abs() * len) <= tol {
                LineHit::Midpoint(*a, *b)
            } else {
                LineHit::Span(*a, *b)
            });
            break;
        }
    }
    match hit {
        Some(LineHit::Endpoint(end)) => {
            let dup = sketch.constraints.iter().any(|c| {
                matches!(c, Constraint::Coincident(x, y) if (*x == p && *y == end) || (*x == end && *y == p))
            });
            if !dup {
                sketch.constraints.push(Constraint::Coincident(p, end));
            }
        }
        Some(LineHit::Midpoint(a, b)) => {
            let dup = sketch.constraints.iter().any(|c| {
                matches!(c, Constraint::Midpoint { mid, a: x, b: y } if *mid == p && *x == a && *y == b)
            });
            if !dup {
                sketch.constraints.push(Constraint::Midpoint { mid: p, a, b });
            }
        }
        Some(LineHit::Span(a, b)) => {
            let dup = sketch.constraints.iter().any(
                |c| matches!(c, Constraint::PointOnLine { p: q, a: x, b: y } if *q == p && *x == a && *y == b),
            );
            if !dup {
                sketch.constraints.push(Constraint::PointOnLine { p, a, b });
            }
        }
        None => {}
    }
}

/// Add (or find) a locked reference line at plane-uv endpoints `a`,`b` — the projection
/// of a picked 3D body edge. Returns the entity index, or `None` if degenerate.
fn add_or_get_reference_line(session: &mut SketchSession, a: Vec2, b: Vec2) -> Option<usize> {
    if a.distance(b) < 1e-4 {
        return None;
    }
    let eps = 1e-3;
    for (i, e) in session.sketch.entities.iter().enumerate() {
        if let SketchEntity::Line { a: pa, b: pb, reference: true, .. } = e {
            let va = Vec2::new(session.sketch.points[*pa].x as f32, session.sketch.points[*pa].y as f32);
            let vb = Vec2::new(session.sketch.points[*pb].x as f32, session.sketch.points[*pb].y as f32);
            if (va.distance(a) < eps && vb.distance(b) < eps) || (va.distance(b) < eps && vb.distance(a) < eps) {
                return Some(i);
            }
        }
    }
    let pa = session.sketch.add_fixed_point(a.x as f64, a.y as f64);
    let pb = session.sketch.add_fixed_point(b.x as f64, b.y as f64);
    Some(session.sketch.add_reference_line(pa, pb))
}

/// Closest point on segment a–b to `p` (clamped to the segment).
fn closest_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b - a;
    let t = if ab.length_squared() > 1e-9 {
        ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    a + ab * t
}

/// The point on a snap edge (straight or arc) nearest to `uv`.
fn edge_snap_point(es: EdgeSnap, uv: Vec2) -> Vec2 {
    match es {
        EdgeSnap::Line([a, b]) => closest_on_segment(uv, a, b),
        EdgeSnap::Arc { center, radius, .. } => {
            let to = uv - center;
            if to.length() > 1e-4 {
                center + to.normalize() * radius
            } else {
                center + Vec2::new(radius, 0.0)
            }
        }
    }
}

/// If point `p` landed on a body `edge` (straight or arc), pin `p` to it: a straight edge
/// becomes a locked reference line + point-on-line relation; an arc becomes a point-on-arc
/// relation. Skipped when `p` is already locked (it snapped to a corner — coincident there
/// is enough).
fn maybe_add_point_on_edge(session: &mut SketchSession, p: usize, edge: Option<EdgeSnap>) {
    let Some(edge) = edge else { return };
    if session.sketch.points.get(p).map_or(true, |pt| pt.fixed) {
        return;
    }
    let already = |c: &Constraint| match c {
        Constraint::PointOnLine { p: q, .. } | Constraint::PointOnArc { p: q, .. } => *q == p,
        _ => false,
    };
    if session.sketch.constraints.iter().any(already) {
        return;
    }
    match edge {
        EdgeSnap::Line([a, b]) => {
            if let Some(ent) = add_or_get_reference_line(session, a, b) {
                if let Some((ra, rb)) = entity_line(&session.sketch, ent) {
                    session.sketch.constraints.push(Constraint::PointOnLine { p, a: ra, b: rb });
                }
            }
        }
        EdgeSnap::Arc { center, radius, .. } => {
            session.sketch.constraints.push(Constraint::PointOnArc {
                p,
                cx: center.x as f64,
                cy: center.y as f64,
                radius: radius as f64,
            });
        }
    }
}

/// Index of the line/circle entity nearest to `uv`, within `thresh`.
fn nearest_entity(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<usize> {
    let p = |i: usize| Vec2::new(sketch.points[i].x as f32, sketch.points[i].y as f32);
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in sketch.entities.iter().enumerate() {
        let d = match e {
            SketchEntity::Line { a, b, .. } => {
                let (a, b) = (p(*a), p(*b));
                let ab = b - a;
                let t = if ab.length_squared() > 1e-9 {
                    ((uv - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (uv - (a + ab * t)).length()
            }
            SketchEntity::Circle { center, radius, .. } => {
                ((uv - p(*center)).length() - *radius as f32).abs()
            }
            SketchEntity::Spline { points, closed, control, .. } => {
                let pts: Vec<[f64; 2]> =
                    points.iter().filter_map(|&i| sketch.points.get(i)).map(|q| [q.x, q.y]).collect();
                if pts.len() < 2 {
                    continue;
                }
                let poly = tessellate_spline(&pts, *closed, *control);
                let mut dmin = f32::MAX;
                for w in poly.windows(2) {
                    let a = Vec2::new(w[0][0] as f32, w[0][1] as f32);
                    let b = Vec2::new(w[1][0] as f32, w[1][1] as f32);
                    let ab = b - a;
                    let t = if ab.length_squared() > 1e-9 {
                        ((uv - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    dmin = dmin.min((uv - (a + ab * t)).length());
                }
                dmin
            }
            SketchEntity::Slot { a, b, radius, mid, .. } => match (sketch.points.get(*a), sketch.points.get(*b)) {
                (Some(pa), Some(pb)) => {
                    let pm = mid.and_then(|m| sketch.points.get(m)).map(|p| [p.x, p.y]);
                    let poly = match pm {
                        Some(pm) => tessellate_arc_slot([pa.x, pa.y], pm, [pb.x, pb.y], *radius),
                        None => tessellate_slot([pa.x, pa.y], [pb.x, pb.y], *radius),
                    };
                    let mut dmin = f32::MAX;
                    for w in poly.windows(2) {
                        let a2 = Vec2::new(w[0][0] as f32, w[0][1] as f32);
                        let b2 = Vec2::new(w[1][0] as f32, w[1][1] as f32);
                        let ab = b2 - a2;
                        let t = if ab.length_squared() > 1e-9 {
                            ((uv - a2).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        dmin = dmin.min((uv - (a2 + ab * t)).length());
                    }
                    dmin
                }
                _ => continue,
            },
            SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. } => {
                let o = match sketch.points.get(*origin) {
                    Some(o) => [o.x, o.y],
                    None => continue,
                };
                let mut dmin = f32::MAX;
                for l in text_contours(o, contours, *height, *rotation, *mirror, *arc) {
                    let n = l.len();
                    for k in 0..n {
                        let a = Vec2::new(l[k][0] as f32, l[k][1] as f32);
                        let b = Vec2::new(l[(k + 1) % n][0] as f32, l[(k + 1) % n][1] as f32);
                        let ab = b - a;
                        let t = if ab.length_squared() > 1e-9 {
                            ((uv - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        dmin = dmin.min((uv - (a + ab * t)).length());
                    }
                }
                dmin
            }
            SketchEntity::Point { .. } => continue,
        };
        if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// Snap/inference points for an entity: a line's midpoint, or a circle's centre
/// and four quadrant points. These let the user snap new geometry to centres.
fn inference_points(sketch: &Sketch, i: usize) -> Vec<Vec2> {
    let p = |j: usize| Vec2::new(sketch.points[j].x as f32, sketch.points[j].y as f32);
    match sketch.entities.get(i) {
        Some(SketchEntity::Line { a, b, .. }) => vec![(p(*a) + p(*b)) * 0.5],
        Some(SketchEntity::Circle { center, radius, .. }) => {
            let c = p(*center);
            let r = *radius as f32;
            vec![
                c,
                c + Vec2::new(0.0, r),
                c - Vec2::new(0.0, r),
                c + Vec2::new(r, 0.0),
                c - Vec2::new(r, 0.0),
            ]
        }
        _ => vec![],
    }
}

/// Snap `uv` to the nearest inference point within `thresh`, else return it.
fn snap_to_inference(uv: Vec2, points: &[Vec2], thresh: f32) -> Vec2 {
    nearest_within(uv, points, thresh).unwrap_or(uv)
}

/// The nearest of `points` to `uv` within `thresh`, if any.
fn nearest_within(uv: Vec2, points: &[Vec2], thresh: f32) -> Option<Vec2> {
    points
        .iter()
        .copied()
        .filter(|p| p.distance(uv) <= thresh)
        .min_by(|a, b| a.distance(uv).total_cmp(&b.distance(uv)))
}

/// Snap a circle's `raw` radius to a nearby reference circle's radius (within
/// `thresh`), so a new circle matches existing round geometry exactly.
fn snap_radius(raw: f32, circles: &[(Vec2, f32)], thresh: f32) -> f32 {
    circles
        .iter()
        .map(|&(_, r)| r)
        .filter(|r| (r - raw).abs() <= thresh)
        .min_by(|a, b| (a - raw).abs().total_cmp(&(b - raw).abs()))
        .unwrap_or(raw)
}

/// The endpoint indices of a line entity, if `i` is a line.
fn entity_line(sketch: &Sketch, i: usize) -> Option<(usize, usize)> {
    match sketch.entities.get(i) {
        Some(SketchEntity::Line { a, b, .. }) => Some((*a, *b)),
        _ => None,
    }
}

/// The (centre index, radius) of a circle entity, if `i` is a circle.
fn entity_circle(sketch: &Sketch, i: usize) -> Option<(usize, f64)> {
    match sketch.entities.get(i) {
        Some(SketchEntity::Circle { center, radius, .. }) => Some((*center, *radius)),
        _ => None,
    }
}

/// The point indices an entity is built on (a line's ends, a circle's centre).
fn entity_points(sketch: &Sketch, i: usize) -> Vec<usize> {
    match sketch.entities.get(i) {
        Some(SketchEntity::Line { a, b, .. }) => vec![*a, *b],
        Some(SketchEntity::Circle { center, .. }) => vec![*center],
        Some(SketchEntity::Point { at }) => vec![*at],
        Some(SketchEntity::Spline { points, .. }) => points.clone(),
        Some(SketchEntity::Slot { a, b, .. }) => vec![*a, *b],
        Some(SketchEntity::Text { origin, .. }) => vec![*origin],
        None => vec![],
    }
}

/// The point indices a constraint references — used to find the relations that
/// touch a selected entity.
fn constraint_points(c: &Constraint) -> Vec<usize> {
    match c {
        Constraint::Coincident(a, b)
        | Constraint::Horizontal(a, b)
        | Constraint::Vertical(a, b)
        | Constraint::Distance { a, b, .. }
        | Constraint::EqualRadius { a, b } => vec![*a, *b],
        Constraint::Midpoint { mid, a, b } => vec![*mid, *a, *b],
        Constraint::Parallel(a, b, c, d)
        | Constraint::Perpendicular(a, b, c, d)
        | Constraint::Equal(a, b, c, d) => vec![*a, *b, *c, *d],
        Constraint::Tangent { a, b, center, .. } => vec![*a, *b, *center],
        Constraint::Radius { center, .. } => vec![*center],
        Constraint::Angle { a, b, c, d, .. } => vec![*a, *b, *c, *d],
        Constraint::PointLineDistance { p, a, b, .. } => vec![*p, *a, *b],
        Constraint::PointOnCircle { p, center } => vec![*p, *center],
        Constraint::PointOnLine { p, a, b } => vec![*p, *a, *b],
        Constraint::PointOnArc { p, .. } => vec![*p],
    }
}

/// A short human label for a relation (for the Existing Relations list).
fn constraint_label(c: &Constraint) -> String {
    match c {
        Constraint::Coincident(..) => "Coincident".into(),
        Constraint::Horizontal(..) => "Horizontal".into(),
        Constraint::Vertical(..) => "Vertical".into(),
        Constraint::Midpoint { .. } => "Midpoint".into(),
        Constraint::Distance { value, axis, .. } => match axis {
            DimAxis::Aligned => format!("Distance  {value:.2}"),
            DimAxis::Horizontal => format!("Horizontal  {value:.2}"),
            DimAxis::Vertical => format!("Vertical  {value:.2}"),
        },
        Constraint::Parallel(..) => "Parallel".into(),
        Constraint::Perpendicular(..) => "Perpendicular".into(),
        Constraint::Equal(..) => "Equal length".into(),
        Constraint::Tangent { .. } => "Tangent".into(),
        Constraint::EqualRadius { .. } => "Equal radius".into(),
        Constraint::Radius { value, diameter, .. } => {
            if *diameter {
                format!("Diameter  {:.2}", value * 2.0)
            } else {
                format!("Radius  {value:.2}")
            }
        }
        Constraint::Angle { value, .. } => format!("Angle  {:.1}°", value.to_degrees()),
        Constraint::PointLineDistance { value, .. } => format!("Distance  {value:.2}"),
        Constraint::PointOnCircle { .. } => "On circle".into(),
        Constraint::PointOnLine { .. } => "On edge".into(),
        Constraint::PointOnArc { .. } => "On arc".into(),
    }
}

/// A cheap fingerprint of a sketch's editable state (point positions, entity/constraint
/// counts, and dimension values) — used to notice when an operation changed the sketch so
/// a per-operation undo step can be recorded. Quantised so float jitter doesn't churn it.
fn sketch_fingerprint(s: &Sketch) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x100000001b3);
    };
    mix(s.points.len() as u64);
    mix(s.entities.len() as u64);
    mix(s.constraints.len() as u64);
    for p in &s.points {
        mix((p.x * 1.0e4).round() as i64 as u64);
        mix((p.y * 1.0e4).round() as i64 as u64);
        mix(p.fixed as u64);
    }
    for c in &s.constraints {
        let v = match c {
            Constraint::Distance { value, .. }
            | Constraint::PointLineDistance { value, .. }
            | Constraint::Radius { value, .. }
            | Constraint::Angle { value, .. } => *value,
            _ => 0.0,
        };
        mix((v * 1.0e4).round() as i64 as u64);
    }
    h
}

/// Index of the sketch region whose interior (inside the outer loop, outside any
/// hole) contains `uv`.
fn region_at(sketch: &Sketch, uv: Vec2) -> Option<usize> {
    let p = [uv.x as f64, uv.y as f64];
    sketch
        .regions()
        .iter()
        .position(|r| point_in_poly(p, &r.outer) && !r.holes.iter().any(|h| point_in_poly(p, h)))
}

/// Möller–Trumbore ray/triangle intersection. Returns the ray parameter `t` of a
/// front-facing hit, if any.
fn ray_triangle(orig: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    const EPS: f32 = 1e-7;
    let e1 = b - a;
    let e2 = c - a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < EPS {
        return None;
    }
    let inv = 1.0 / det;
    let tv = orig - a;
    let u = tv.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tv.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > EPS).then_some(t)
}

/// Ray-pick a planar face of a tessellated body. Returns the hit distance and the
/// sketch plane on that face (origin at the face centroid, normal facing the
/// camera, with derived in-plane axes).
fn pick_face(mesh: &TriMesh, ray: &Ray3d) -> Option<(f32, ActivePlane)> {
    let dir = ray.direction.as_vec3();
    let pos = &mesh.positions;

    let mut best_t = f32::INFINITY;
    let mut hit = Vec3::ZERO;
    let mut normal = Vec3::ZERO;
    for tri in mesh.indices.chunks(3) {
        let a = Vec3::from_array(pos[tri[0] as usize]);
        let b = Vec3::from_array(pos[tri[1] as usize]);
        let c = Vec3::from_array(pos[tri[2] as usize]);
        if let Some(t) = ray_triangle(ray.origin, dir, a, b, c) {
            if t < best_t {
                best_t = t;
                hit = ray.origin + dir * t;
                normal = (b - a).cross(c - a).normalize_or_zero();
            }
        }
    }
    if !best_t.is_finite() || normal == Vec3::ZERO {
        return None;
    }
    // Normal should face the camera.
    let mut n = normal;
    if n.dot(dir) > 0.0 {
        n = -n;
    }

    // Average the centroids of all triangles coplanar with the hit → face origin.
    let plane_d = n.dot(hit);
    let mut sum = Vec3::ZERO;
    let mut count = 0.0_f32;
    for tri in mesh.indices.chunks(3) {
        let a = Vec3::from_array(pos[tri[0] as usize]);
        let b = Vec3::from_array(pos[tri[1] as usize]);
        let c = Vec3::from_array(pos[tri[2] as usize]);
        let mut tn = (b - a).cross(c - a).normalize_or_zero();
        if tn.dot(n) < 0.0 {
            tn = -tn;
        }
        let centroid = (a + b + c) / 3.0;
        if tn.dot(n) > 0.99 && (n.dot(centroid) - plane_d).abs() < 0.01 {
            sum += centroid;
            count += 1.0;
        }
    }
    let mut origin = if count > 0.0 { sum / count } else { hit };
    origin -= n * (n.dot(origin) - plane_d); // snap exactly onto the plane

    // In-plane axes from the normal.
    let seed = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let u = (seed - n * seed.dot(n)).normalize();
    let v = n.cross(u).normalize();

    Some((best_t, ActivePlane { name: "Face".into(), origin, u, v, n }))
}

/// An `ActivePlane` (app-side) from a stored `PlaneRef` (document-side).
fn active_plane_from_ref(p: &PlaneRef, name: &str) -> ActivePlane {
    let f = |a: [f64; 3]| Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32);
    ActivePlane { name: name.into(), origin: f(p.origin), u: f(p.u), v: f(p.v), n: f(p.normal) }
}

/// `PlaneRef` (document-side) from an active plane (app-side).
fn plane_ref(ap: &ActivePlane) -> PlaneRef {
    PlaneRef {
        origin: [ap.origin.x as f64, ap.origin.y as f64, ap.origin.z as f64],
        u: [ap.u.x as f64, ap.u.y as f64, ap.u.z as f64],
        v: [ap.v.x as f64, ap.v.y as f64, ap.v.z as f64],
        normal: [ap.n.x as f64, ap.n.y as f64, ap.n.z as f64],
    }
}

/// Average of a mesh's vertices — used to decide which side of a sketch plane the
/// body sits on (so a cut removes material in the right direction).
fn mesh_centroid(mesh: &TriMesh) -> Vec3 {
    if mesh.positions.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = mesh.positions.iter().map(|p| Vec3::from_array(*p)).sum();
    sum / mesh.positions.len() as f32
}

/// Axis-aligned bounding box (min, max) of a mesh — for diagnosing how a boss sits
/// relative to the body when a boolean fails.
fn mesh_bbox(mesh: &TriMesh) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in &mesh.positions {
        let v = Vec3::from_array(*p);
        lo = lo.min(v);
        hi = hi.max(v);
    }
    (lo, hi)
}

/// All triangles of `mesh` lying on the plane through `point` with `normal`
/// (either winding) — i.e. one flat face of the body.
fn gather_face(mesh: &TriMesh, point: Vec3, normal: Vec3) -> Vec<[Vec3; 3]> {
    let d = normal.dot(point);
    let pos = &mesh.positions;
    let mut out = Vec::new();
    for tri in mesh.indices.chunks(3) {
        let a = Vec3::from_array(pos[tri[0] as usize]);
        let b = Vec3::from_array(pos[tri[1] as usize]);
        let c = Vec3::from_array(pos[tri[2] as usize]);
        let tn = (b - a).cross(c - a).normalize_or_zero();
        if tn.dot(normal).abs() > 0.99 && (normal.dot((a + b + c) / 3.0) - d).abs() < 0.02 {
            out.push([a, b, c]);
        }
    }
    out
}

/// Build the highlight overlay mesh from a face's triangles, lifted slightly
/// toward the camera (along `n`) to avoid z-fighting with the body.
fn build_face_mesh(tris: &[[Vec3; 3]], n: Vec3) -> Mesh {
    let off = n * 0.025;
    let normal = [n.x, n.y, n.z];
    let mut positions = Vec::with_capacity(tris.len() * 3);
    let mut normals = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        for vtx in t {
            let p = *vtx + off;
            positions.push([p.x, p.y, p.z]);
            normals.push(normal);
        }
    }
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh
}

/// Highlight the face under the cursor (view mode) or the active sketch face.
fn highlight_face(
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    part: Res<Part>,
    session: Res<SketchSession>,
    blocking: Res<UiBlocking>,
    hl: Res<HighlightMesh>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut vis_q: Query<&mut Visibility, With<FaceHighlight>>,
) {
    let Ok(mut vis) = vis_q.single_mut() else { return };
    let Some(mesh) = part.mesh.as_ref() else {
        *vis = Visibility::Hidden;
        return;
    };

    // Which face to highlight: the active sketch face, or the hovered one.
    let target: Option<(Vec3, Vec3)> = if let Some(ap) = &session.plane {
        (ap.name == "Face").then_some((ap.origin, ap.n))
    } else if !blocking.0 {
        windows
            .single()
            .ok()
            .zip(cam_q.single().ok())
            .and_then(|(w, (cam, gt))| cursor_ray(w, cam, gt))
            .and_then(|ray| pick_face(mesh, &ray))
            .map(|(_, ap)| (ap.origin, ap.n))
    } else {
        None
    };

    match target {
        Some((origin, n)) => {
            let tris = gather_face(mesh, origin, n);
            if tris.is_empty() {
                *vis = Visibility::Hidden;
            } else {
                if let Some(slot) = meshes.get_mut(&hl.0) {
                    *slot = build_face_mesh(&tris, n);
                }
                *vis = Visibility::Visible;
            }
        }
        None => *vis = Visibility::Hidden,
    }
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

fn sketch_interaction(
    buttons: Res<ButtonInput<MouseButton>>,
    blocking: Res<UiBlocking>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cam_q: Query<(&Camera, &GlobalTransform, &mut Transform, &mut OrbitCamera)>,
    doc: Res<DocRes>,
    part: Res<Part>,
    mut session: ResMut<SketchSession>,
    mut edge_sel: ResMut<EdgeSelection>,
    mut ui_state: ResMut<UiState>,
    time: Res<Time>,
) {
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_gt, mut cam_tf, mut orbit)) = cam_q.single_mut() else { return };
    let Some(ray) = cursor_ray(window, camera, cam_gt) else { return };

    // Fillet/chamfer edge picking runs *before* the egui pointer-block: while the panel is
    // open, a focused widget makes egui greedy for the pointer, which would otherwise eat
    // the click. Gating on an actual body-edge hit keeps it safe — a hit means the cursor
    // is on the body, not the panel.
    if (ui_state.pending_fillet.is_some() || ui_state.pending_chamfer.is_some())
        && buttons.just_pressed(MouseButton::Left)
    {
        if let Some(cursor) = window.cursor_position() {
            if let Some(si) = pick_edge(&part.edges, camera, cam_gt, cursor, EDGE_PICK_PX) {
                let (chain, closed) = edge_chain(&part.edges, si);
                if chain.len() >= 2 {
                    toggle_fillet_edge(&mut ui_state, &chain);
                    edge_sel.set(chain, closed);
                }
            }
        }
        return;
    }

    if blocking.0 {
        return;
    }

    let active_uv = session.plane.as_ref().and_then(|ap| ray_plane(ap, &ray).map(|(_, uv)| uv));

    // Snap tolerance scaled to the zoom, so snapping feels consistent at any scale.
    let snap = (orbit.radius * (SNAP / 12.0)).clamp(0.03, 200.0);
    session.snap_dist = snap;

    // Inference/snap points for the entity under the cursor (shown on hover).
    session.inference_points.clear();
    if session.plane.is_some() && !session.hide_inference {
        if let Some(e) = active_uv.and_then(|uv| nearest_entity(&session.sketch, uv, snap * 2.5)) {
            session.inference_points = inference_points(&session.sketch, e);
        }
    }
    // Reference snap points/circles from the body's edges lying in the sketch plane,
    // so new geometry can be snapped to existing features (endpoints, edge midpoints,
    // and circular-edge centres + radii).
    session.reference_points.clear();
    session.reference_circles.clear();
    if let Some(ap) = session.plane.clone() {
        let in_plane = |w: Vec3| (w - ap.origin).dot(ap.n).abs() < 1e-3;
        let to_uv = |w: Vec3| {
            let d = w - ap.origin;
            Vec2::new(d.dot(ap.u), d.dot(ap.v))
        };
        // Vertices already accounted for by a detected circular edge (full circle or
        // arc), so its many tessellation segments aren't each emitted as straight points.
        let mut circle_verts: Vec<Vec3> = Vec::new();
        for (i, e) in part.edges.iter().enumerate() {
            let (a, b) = (Vec3::from_array(e[0]), Vec3::from_array(e[1]));
            if !in_plane(a) || !in_plane(b) || a.distance(b) < 1e-6 {
                continue;
            }
            // Skip a segment only if BOTH ends belong to a handled circle/arc — a
            // radial edge that merely touches an arc endpoint must still be processed.
            let on_circle = |p: Vec3| circle_verts.iter().any(|c| c.distance(p) < 1e-3);
            if on_circle(a) && on_circle(b) {
                continue;
            }
            let (chain, closed) = edge_chain(&part.edges, i);
            let uvs: Vec<Vec2> = chain.iter().map(|p| to_uv(*p)).collect();
            // Fast path: a whole closed circle → centre + 4 quadrant points.
            let whole_circle = if closed && uvs.len() >= 8 {
                fit_circle(&uvs).filter(|&(c, r)| {
                    r > 1e-3 && uvs.iter().map(|p| (p.distance(c) - r).abs()).fold(0.0_f32, f32::max) < r * 0.05
                })
            } else {
                None
            };
            if let Some((c, r)) = whole_circle {
                session.reference_circles.push((c, r));
                session.reference_points.push(c);
                session.reference_points.push(c + Vec2::new(r, 0.0));
                session.reference_points.push(c + Vec2::new(-r, 0.0));
                session.reference_points.push(c + Vec2::new(0.0, r));
                session.reference_points.push(c + Vec2::new(0.0, -r));
                circle_verts.extend(chain);
            } else {
                // Segment a mixed/long chain into arcs + straight runs, emitting only a
                // few key points per part (so a slot's arc isn't a wall of midpoints).
                let segs = segment_chain(&uvs);
                let has_curve = segs.iter().any(|(_, _, f)| f.is_some());
                for (s, e, fit) in &segs {
                    match fit {
                        Some((c, r)) => {
                            session.reference_circles.push((*c, *r));
                            session.reference_points.push(*c);
                            session.reference_points.push(uvs[*s]);
                            session.reference_points.push(uvs[*e]);
                            session.reference_points.push(uvs[(*s + *e) / 2]);
                        }
                        None => {
                            session.reference_points.push(uvs[*s]);
                            session.reference_points.push(uvs[*e]);
                            session.reference_points.push((uvs[*s] + uvs[*e]) * 0.5);
                        }
                    }
                }
                // Only consume the chain if it contained a curve (so straight box edges
                // sharing corners are each still processed; duplicates are deduped).
                if has_curve {
                    circle_verts.extend(chain);
                }
            }
        }
        // Drop duplicate points (edges share corner vertices).
        session.reference_points.sort_by(|p, q| {
            p.x.partial_cmp(&q.x).unwrap().then(p.y.partial_cmp(&q.y).unwrap())
        });
        session.reference_points.dedup_by(|p, q| p.distance(*q) < 1e-3);
    }

    // The working cursor snaps to an inference point, a reference point, or onto the
    // boundary of a sketch circle near it — so lines drawn to a circle actually
    // connect (land on the rim) instead of stopping a hair short.
    if session.plane.is_some() {
        let mut snaps = session.inference_points.clone();
        snaps.extend_from_slice(&session.reference_points);
        // Every existing sketch point (line endpoints, circle centres) is a snap
        // target — so a circle placed on a line's end shares its exact coordinates.
        for p in &session.sketch.points {
            snaps.push(Vec2::new(p.x as f32, p.y as f32));
        }
        // Circle perimeters (quadrants + the nearest point on the rim) are *lower priority*
        // than explicit geometry: a line or point in front of a circle must win, so the
        // cursor isn't yanked past the line onto the rim behind it. These go in a separate
        // pool that's only consulted when nothing stronger is within range.
        let mut rim_snaps: Vec<Vec2> = Vec::new();
        for e in &session.sketch.entities {
            // Construction circles (e.g. a polygon's circumscribed circle) deliberately do
            // *not* seed the quadrant/rim snap cloud — otherwise every later point gets
            // yanked onto earlier polygons' guide circles. Their centre is still a normal
            // snappable point.
            if let SketchEntity::Circle { center, radius, construction: false } = e {
                if let Some(c) = session.sketch.points.get(*center) {
                    let (cc, r) = (Vec2::new(c.x as f32, c.y as f32), *radius as f32);
                    rim_snaps.push(cc + Vec2::new(r, 0.0));
                    rim_snaps.push(cc + Vec2::new(-r, 0.0));
                    rim_snaps.push(cc + Vec2::new(0.0, r));
                    rim_snaps.push(cc + Vec2::new(0.0, -r));
                    if let Some(uv) = active_uv {
                        let to = uv - cc;
                        if to.length() > 1e-4 {
                            rim_snaps.push(cc + to.normalize() * r); // nearest point on the rim
                        }
                    }
                }
            }
        }
        // Body (reference) circles get the same nearest-rim snap, so lines can land on
        // a projected circular edge anywhere on its rim, not just its 4 quadrant points.
        if let Some(uv) = active_uv {
            for (cc, r) in &session.reference_circles {
                let to = uv - *cc;
                if to.length() > 1e-4 {
                    rim_snaps.push(*cc + to.normalize() * *r);
                }
            }
        }
        // The nearest point *along* each sketch line's span — so a line is snappable
        // anywhere on its length (not just its ends) and reliably beats a circle behind it.
        if let Some(uv) = active_uv {
            for e in &session.sketch.entities {
                if let SketchEntity::Line { a, b, .. } = e {
                    if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                        let va = Vec2::new(pa.x as f32, pa.y as f32);
                        let vb = Vec2::new(pb.x as f32, pb.y as f32);
                        let cp = closest_on_segment(uv, va, vb);
                        if cp.distance(uv) <= snap {
                            snaps.push(cp);
                        }
                    }
                }
            }
        }
        // Body edges: pick the edge under the cursor (same screen-space pick the Select
        // tool uses), split its smooth chain into straight runs + arcs, take the run the
        // cursor is nearest, highlight it, and add a snap target on it — so a line can run
        // *along* a straight edge, or land an endpoint on a straight edge or a fillet arc.
        session.hover_edge = None;
        if let (Some(uv), Some(ap), Some(cursor)) =
            (active_uv, session.plane.clone(), window.cursor_position())
        {
            if let Some(i) = pick_edge(&part.edges, camera, cam_gt, cursor, 12.0) {
                let (chain, _) = edge_chain(&part.edges, i);
                let to_uv = |w: Vec3| {
                    let d = w - ap.origin;
                    Vec2::new(d.dot(ap.u), d.dot(ap.v))
                };
                let uvs: Vec<Vec2> = chain.iter().map(|w| to_uv(*w)).collect();
                if uvs.len() >= 2 {
                    // The run (straight or arc) whose polyline is closest to the cursor.
                    let runs = segment_chain(&uvs);
                    let mut best: Option<(usize, f32)> = None;
                    for (ri, (s, e, _)) in runs.iter().enumerate() {
                        let mut d = f32::MAX;
                        for w in uvs[*s..=*e].windows(2) {
                            d = d.min(closest_on_segment(uv, w[0], w[1]).distance(uv));
                        }
                        if best.map_or(true, |(_, bd)| d < bd) {
                            best = Some((ri, d));
                        }
                    }
                    if let Some((ri, _)) = best {
                        let (s, e, fit) = runs[ri];
                        let es = match fit {
                            None => EdgeSnap::Line([uvs[s], uvs[e]]),
                            Some((c, r)) => EdgeSnap::Arc { center: c, radius: r, a: uvs[s], b: uvs[e] },
                        };
                        session.hover_edge = Some(es);
                        snaps.push(edge_snap_point(es, uv));
                    }
                }
            }
        }
        // Strong geometry (points, line spans, straight edges) wins; circle perimeters are
        // only used when nothing stronger is within range.
        session.cursor_uv = active_uv.map(|uv| {
            nearest_within(uv, &snaps, snap)
                .or_else(|| nearest_within(uv, &rim_snaps, snap))
                .unwrap_or(uv)
        });
        session.cursor_raw_uv = active_uv;
        // Is the (snapped) cursor sitting on the hovered edge? If so, placing a point
        // here will add a point-on-edge relation.
        session.cursor_edge = match (session.cursor_uv, session.hover_edge) {
            (Some(uv), Some(es)) if edge_snap_point(es, uv).distance(uv) <= snap * 0.5 => Some(es),
            _ => None,
        };
    } else {
        session.hover_edge = None;
        session.cursor_edge = None;
    }

    // Line tool (and the slot's centre line) — snap the in-progress segment to a 90° step
    // off the world axes *and* off any line the start point is joined to (so it stays
    // square / perpendicular to what it connects to, at any length). This yields to a
    // genuine point / corner / circle-rim target so the line can still close on one.
    let square_tool =
        session.tool == Tool::Line || (session.tool == Tool::Slot && session.pending_b.is_none());
    if square_tool {
        if let (Some(start), Some(cur)) = (session.pending, session.cursor_uv) {
            // Strong targets win over the square snap so connections aren't pulled off.
            let near = |p: Vec2, t: f32| p.distance(cur) <= t;
            let strong = snap * 0.6;
            let on_strong = nearest_point(&session.sketch, cur, strong).is_some()
                || session.reference_points.iter().any(|r| near(*r, strong))
                || session.inference_points.iter().any(|r| near(*r, strong))
                || session.reference_circles.iter().any(|(c, r)| (cur.distance(*c) - *r).abs() <= strong)
                || session.sketch.entities.iter().any(|e| matches!(e, SketchEntity::Circle { center, radius, .. }
                    if session.sketch.points.get(*center).is_some_and(|c| (cur.distance(Vec2::new(c.x as f32, c.y as f32)) - *radius as f32).abs() <= strong)));
            let v = cur - start;
            if !on_strong && v.length() > 1e-4 {
                // Reference directions: the world axes, plus the direction of any sketch
                // line the start point belongs to (its perpendicular/parallel snaps too).
                let mut dirs: Vec<Vec2> = vec![Vec2::X, Vec2::Y];
                if let Some(pi) = nearest_point(&session.sketch, start, snap * 0.5) {
                    for e in &session.sketch.entities {
                        if let SketchEntity::Line { a, b, reference: false, .. } = e {
                            if *a == pi || *b == pi {
                                let pa = session.sketch.points[*a];
                                let pb = session.sketch.points[*b];
                                let d = (Vec2::new(pb.x as f32, pb.y as f32) - Vec2::new(pa.x as f32, pa.y as f32)).normalize_or_zero();
                                if d != Vec2::ZERO {
                                    dirs.push(d);
                                }
                            }
                        }
                    }
                }
                let step = std::f32::consts::FRAC_PI_2; // 90°
                let ang = v.y.atan2(v.x);
                let mut best: Option<(f32, Vec2)> = None; // (angle error, snapped direction)
                for d in &dirs {
                    let base = d.y.atan2(d.x);
                    let snapped = base + ((ang - base) / step).round() * step;
                    let mut err = (ang - snapped).abs();
                    err = err.min(std::f32::consts::TAU - err);
                    if best.map_or(true, |(be, _)| err < be) {
                        best = Some((err, Vec2::new(snapped.cos(), snapped.sin())));
                    }
                }
                if let Some((err, dir)) = best {
                    if err < 5.0_f32.to_radians() {
                        // Project the cursor onto the chosen axis (length follows the cursor).
                        session.cursor_uv = Some(start + dir * v.dot(dir));
                    }
                }
            }
        }
    }
    // Never let a non-finite cursor through — it poisons placed points and egui.
    if session.cursor_uv.is_some_and(|c| !c.is_finite()) {
        session.cursor_uv = None;
    }

    // Entity under the cursor (Select tool) for hover highlighting.
    session.hover_entity = if session.plane.is_some() && matches!(session.tool, Tool::Select | Tool::Pattern | Tool::Mirror) {
        active_uv.and_then(|uv| nearest_entity(&session.sketch, uv, snap * 1.5))
    } else {
        None
    };

    let just_pressed = buttons.just_pressed(MouseButton::Left);
    let pressed = buttons.pressed(MouseButton::Left);
    let just_released = buttons.just_released(MouseButton::Left);

    // Circular-pattern centre pick: the next click sets the revolve centre from the snapped
    // cursor (a point / endpoint), instead of selecting geometry.
    if session.pattern_pick_center {
        if session.tool == Tool::Pattern && session.pattern_mode == PatternMode::Circular {
            if just_pressed {
                if let Some(uv) = session.cursor_uv {
                    session.pat_circ_center = uv;
                    session.pat_center_set = true;
                }
                session.pattern_pick_center = false;
                return; // consume the click so it doesn't also toggle a selection
            }
        } else {
            session.pattern_pick_center = false; // mode/tool changed → cancel the pick
        }
    }

    // While the dimension Modify box is open it stays put on the measurement bar (the
    // dimension line is positioned by a default offset, draggable later with the Select
    // tool). A click confirms it; if the open dimension is a line length and a *second*
    // line is clicked, it becomes an angle dimension.
    if let Some(ci) = session.dim_edit {
        if just_pressed {
            // Angle conversion: open dim is a line length and a different line is clicked.
            // Pick the nearest *line* under the cursor (a sketch line in preference to a
            // body-edge reference line) so the angle is between the lines you actually click.
            let convert = session.dim_line.and_then(|l1| {
                let uv = active_uv?;
                let mut best: Option<(usize, bool, f32)> = None; // (entity, is_ref, dist)
                for (i, e) in session.sketch.entities.iter().enumerate() {
                    if i == l1 {
                        continue;
                    }
                    if let SketchEntity::Line { a, b, reference, .. } = e {
                        if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                            let va = Vec2::new(pa.x as f32, pa.y as f32);
                            let vb = Vec2::new(pb.x as f32, pb.y as f32);
                            let d = closest_on_segment(uv, va, vb).distance(uv);
                            if d <= snap * 2.0 {
                                // Prefer a non-reference line; among equals, the nearer one.
                                let better = best.map_or(true, |(_, bref, bd)| {
                                    (*reference, d) < (bref, bd) || (*reference == bref && d < bd)
                                });
                                if better {
                                    best = Some((i, *reference, d));
                                }
                            }
                        }
                    }
                }
                best.map(|(i, _, _)| (l1, i))
            });
            if let Some((l1, l2)) = convert {
                if let Some(ci2) = convert_length_to_angle(&mut session, ci, l1, l2) {
                    session.dim_edit = Some(ci2);
                    session.dim_line = None;
                    let deg = match session.sketch.constraints.get(ci2) {
                        Some(Constraint::Angle { value, .. }) => value.to_degrees(),
                        _ => 0.0,
                    };
                    session.dim_buf = deg;
                    session.dim_edit_focus = true;
                    return;
                }
            }
            session.dim_edit = None;
            session.dim_line = None;
        }
        return;
    }

    // While a boss/cut is being configured, grabbing its direction arrow and dragging
    // sets the depth live (which shows in the panel and the feature tree on commit).
    if ui_state.pending.is_some() && session.plane.is_some() {
        if extrude_arrow_drag(&mut session, &mut ui_state, window, camera, cam_gt, &ray, just_pressed, pressed, just_released) {
            return;
        }
    }

    if session.plane.is_none() {
        if just_pressed {
            // A click near a body edge selects that edge/loop (and flashes its key
            // points) instead of starting a sketch. Edges are thin, faces are wide,
            // so clicking the open part of a face still enters sketch mode below.
            if !part.edges.is_empty() {
                if let Some(cursor) = window.cursor_position() {
                    if let Some(si) = pick_edge(&part.edges, camera, cam_gt, cursor, EDGE_PICK_PX) {
                        let (chain, closed) = edge_chain(&part.edges, si);
                        if chain.len() >= 2 {
                            edge_sel.set(chain, closed);
                            return;
                        }
                    }
                }
            }

            let mut best: Option<(f32, ActivePlane)> = None;
            // Reference planes — only while starting the part (they're hidden once
            // a body exists, so you sketch on faces from then on).
            if part.solid.is_none() {
                for (_id, p) in doc.0.planes() {
                    let ap = ActivePlane::from_doc(p);
                    if let Some((t, uv)) = ray_plane(&ap, &ray) {
                        let half = PLANE_SIZE * 0.5;
                        if uv.x.abs() <= half && uv.y.abs() <= half {
                            if best.as_ref().map_or(true, |(bt, _)| t < *bt) {
                                best = Some((t, ap));
                            }
                        }
                    }
                }
            }
            // Planar faces of the body (M5) — usually in front, so they win.
            if let Some(mesh) = &part.mesh {
                if let Some((t, ap)) = pick_face(mesh, &ray) {
                    if best.as_ref().map_or(true, |(bt, _)| t < *bt) {
                        best = Some((t, ap));
                    }
                }
            }
            if let Some((_, ap)) = best {
                // Snap face-on, but via the orbit state so the user can keep orbiting.
                edge_sel.clear();
                orbit.radius = orbit.radius.max(6.0);
                look_along(&mut orbit, ap.origin, ap.n);
                *cam_tf = camera_transform(&orbit);
                session.sketch.clear();
                session.pending = None;
                session.dim_first = None;
                session.cursor_uv = None;
                session.drag = None;
                session.selected_contours.clear();
                session.selected_entities.clear();
                session.needs_apply = false;
                session.undo_sketch.clear();
                session.redo_sketch.clear();
                session.undo_baseline = Some(session.sketch.clone());
                session.undo_fp = sketch_fingerprint(&session.sketch);
                info!("Sketching on the {} plane.", ap.name);
                session.plane = Some(ap);
            } else {
                // Clicked empty space (no edge, no face) → deselect.
                edge_sel.clear();
            }
        }
        return;
    }

    // Switching tools drops any in-progress dimension pick (the Modify box itself can
    // stay open — it's modal and closes on its own confirm/cancel, and double-clicking
    // a dimension with the Select tool reopens it).
    if session.tool != Tool::Dimension {
        session.dim_first = None;
        session.dim_line = None;
    }
    if session.tool != Tool::Spline && !session.spline_pts.is_empty() {
        session.spline_pts.clear(); // leaving the spline tool drops the in-progress curve
    }
    if session.tool != Tool::Rectangle && session.tool != Tool::Slot {
        session.pending_b = None; // the second anchor belongs to the parallelogram / slot
    }
    if session.tool != Tool::Slot {
        session.pending_c = None; // the arc-slot bend point belongs only to the slot tool
    }
    if !matches!(session.tool, Tool::Select | Tool::Text | Tool::Pattern | Tool::Mirror) {
        // Keep a selection alive into the Text tool (live-edit the picked text) and the
        // Pattern / Mirror tools (the selection is what gets repeated / reflected).
        session.selected_entities.clear();
    }
    if !matches!(session.tool, Tool::Select | Tool::Pattern | Tool::Mirror) {
        session.box_select = None;
    }

    match session.tool {
        Tool::Select | Tool::Pattern | Tool::Mirror => {
            if just_pressed {
                // A Text entity's on-canvas handles take priority: grab the square to
                // scale or the circle to rotate it.
                if let Some(uv) = active_uv {
                    let mut grabbed = None;
                    for i in 0..session.sketch.entities.len() {
                        if let Some((sc, rot, _)) = text_handles(&session.sketch, i) {
                            if rot.distance(uv) <= snap * 1.2 {
                                grabbed = Some(TextHandle::Rotate(i));
                                break;
                            }
                            if sc.distance(uv) <= snap * 1.2 {
                                grabbed = Some(TextHandle::Scale(i));
                                break;
                            }
                        }
                    }
                    if let Some(h) = grabbed {
                        session.text_handle = Some(h);
                        let idx = match h {
                            TextHandle::Scale(i) | TextHandle::Rotate(i) => i,
                        };
                        session.selected_entities.clear();
                        session.selected_entities.push(idx);
                        // Mirror the entity's params into the panel fields (own them first
                        // so the immutable borrow ends before we write back).
                        let params = match session.sketch.entities.get(idx) {
                            Some(SketchEntity::Text { text, font, bold, italic, spacing, height, arc, mirror, .. }) => {
                                Some((text.clone(), font.clone(), *bold, *italic, *spacing, *height as f32, *arc, *mirror))
                            }
                            _ => None,
                        };
                        if let Some((t, f, b, it, sp, hh, ar, mi)) = params {
                            session.text_string = t;
                            session.text_font = f;
                            session.text_bold = b;
                            session.text_italic = it;
                            session.text_spacing = sp;
                            session.text_height = hh;
                            session.text_arc = ar;
                            session.text_mirror = mi;
                        }
                        return;
                    }
                }
                // Double-click a dimension to reopen its Modify box (like SolidWorks).
                let now = time.elapsed_secs();
                let double = now - session.last_click_t < 0.4;
                session.last_click_t = now;
                if double {
                    if let Some(ci) = active_uv.and_then(|uv| dim_at(&session.sketch, uv, snap * 2.0)) {
                        open_dim_edit(&mut session, ci, None);
                        return;
                    }
                }
                // Clicking a dimension's label (not on a point) grabs it to drag its
                // offset — reposition the dimension line off the geometry.
                let on_dim = active_uv
                    .filter(|uv| nearest_point(&session.sketch, *uv, snap).is_none())
                    .and_then(|uv| dim_at(&session.sketch, uv, snap * 2.0));
                if let Some(ci) = on_dim {
                    session.dim_drag = Some(ci);
                }
                // Locked reference points (projected from the body) can't be dragged.
                let hit = (on_dim.is_none())
                    .then(|| {
                        active_uv
                            .and_then(|uv| nearest_point(&session.sketch, uv, snap))
                            .filter(|i| !session.sketch.points[*i].fixed)
                    })
                    .flatten();
                session.drag = hit;
                if on_dim.is_none() && hit.is_none() {
                    if let Some(uv) = active_uv {
                        // Priority: clicking *on* a line/circle (tight) selects it for
                        // a constraint; clicking *inside* a closed area selects that
                        // contour; a looser grab still catches a nearby line otherwise.
                        let on_entity = nearest_entity(&session.sketch, uv, snap * 0.6);
                        let region = region_at(&session.sketch, uv);
                        let near_entity = nearest_entity(&session.sketch, uv, snap * 1.5);
                        if let Some(e) = on_entity.or(region.is_none().then_some(()).and(near_entity)) {
                            // (de)select the entity for a constraint. Many can be selected
                            // (e.g. Equal across several lines); pairwise relations just
                            // stay disabled unless exactly two are chosen.
                            if let Some(pos) = session.selected_entities.iter().position(|&x| x == e) {
                                session.selected_entities.remove(pos);
                            } else {
                                session.selected_entities.push(e);
                                while session.selected_entities.len() > MAX_SEL {
                                    session.selected_entities.remove(0);
                                }
                            }
                        } else if let Some(r) = region {
                            // Inside a closed region → toggle it as a Selected Contour.
                            session.selected_entities.clear();
                            if let Some(pos) = session.selected_contours.iter().position(|&x| x == r) {
                                session.selected_contours.remove(pos);
                            } else {
                                session.selected_contours.push(r);
                            }
                        } else {
                            // Nothing in the sketch under the cursor — try a 3D body edge.
                            // Project it into the plane as a locked reference line and
                            // select it (so it can be made parallel/perp/distance to a line).
                            let picked = window.cursor_position().and_then(|cur| {
                                pick_edge(&part.edges, camera, cam_gt, cur, 10.0)
                                    .map(|i| part.edges[i])
                                    .or_else(|| {
                                        pick_edge(&part.tangent_edges, camera, cam_gt, cur, 10.0)
                                            .map(|i| part.tangent_edges[i])
                                    })
                            });
                            if let (Some(seg), Some(ap)) = (picked, session.plane.clone()) {
                                let to_uv = |w: Vec3| {
                                    let d = w - ap.origin;
                                    Vec2::new(d.dot(ap.u), d.dot(ap.v))
                                };
                                let a2 = to_uv(Vec3::from_array(seg[0]));
                                let b2 = to_uv(Vec3::from_array(seg[1]));
                                if let Some(e) = add_or_get_reference_line(&mut session, a2, b2) {
                                    if !session.selected_entities.contains(&e) {
                                        session.selected_entities.push(e);
                                        while session.selected_entities.len() > MAX_SEL {
                                            session.selected_entities.remove(0);
                                        }
                                    }
                                } else {
                                    session.selected_entities.clear();
                                }
                            } else {
                                // Empty space → start a drag-over box select.
                                session.selected_entities.clear();
                                session.box_select = active_uv;
                            }
                        }
                    }
                }
            }
            // Dragging a Text handle: scale (height from origin distance) or rotate.
            if let Some(h) = session.text_handle {
                if pressed {
                    if let Some(uv) = active_uv {
                        apply_text_handle(&mut session, h, uv);
                    }
                }
                if just_released {
                    session.text_handle = None;
                }
            } else if let Some(i) = session.drag {
                if pressed {
                    if let Some(uv) = active_uv {
                        // Snap the dragged endpoint onto a nearby *other* point / corner.
                        let target = snap_drag_target(&session, i, uv);
                        // Grabbing a centre line by its midpoint translates the whole line
                        // (both ends ride along) so it can be repositioned and snapped.
                        let cl_mid = session.center_line.filter(|[_, mid, _]| *mid == i);
                        if let Some([a, mid, b]) = cl_mid {
                            let old = Vec2::new(session.sketch.points[mid].x as f32, session.sketch.points[mid].y as f32);
                            let d = target - old;
                            for p in [a, mid, b] {
                                if let Some(q) = session.sketch.points.get_mut(p) {
                                    q.x += d.x as f64;
                                    q.y += d.y as f64;
                                }
                            }
                            session.dirty = true;
                        } else {
                            if let Some(p) = session.sketch.points.get_mut(i) {
                                p.x = target.x as f64;
                                p.y = target.y as f64;
                            }
                            session.sketch.solve_with_fixed(&[i]);
                        }
                    }
                }
                if just_released {
                    // If we dropped this point onto another, bind them coincident so the
                    // snap *survives* later edits/moves (not just a one-frame overlap).
                    // Skip the centre line's own three points (mid carries its ends).
                    let is_cl = session.center_line.is_some_and(|t| t.contains(&i));
                    if !is_cl {
                        let pi = session.sketch.points.get(i).map(|p| Vec2::new(p.x as f32, p.y as f32));
                        if let Some(pi) = pi {
                            let tol = session.snap_dist.max(1e-3);
                            let target = session.sketch.points.iter().enumerate().find(|(j, p)| {
                                *j != i && Vec2::new(p.x as f32, p.y as f32).distance(pi) <= tol
                            }).map(|(j, _)| j);
                            if let Some(j) = target {
                                let exists = session.sketch.constraints.iter().any(|c| {
                                    matches!(c, Constraint::Coincident(x, y) if (*x == i && *y == j) || (*x == j && *y == i))
                                });
                                if !exists {
                                    session.sketch.constraints.push(Constraint::Coincident(i, j));
                                    session.sketch.solve();
                                    session.dirty = true;
                                }
                            }
                        }
                    }
                    session.drag = None;
                }
            }
            // Dragging a dimension repositions its line off the geometry (display only).
            if let Some(ci) = session.dim_drag {
                if pressed {
                    if let Some(uv) = active_uv {
                        set_dim_offset_from_cursor(&mut session, ci, uv);
                    }
                }
                if just_released {
                    session.dim_drag = None;
                }
            }
            // Drag-over box select: on release, select every entity whose endpoints all
            // fall inside the box (a window select).
            if let Some(start) = session.box_select {
                if just_released {
                    if let Some(end) = active_uv {
                        let (lo, hi) = (start.min(end), start.max(end));
                        if (hi - lo).length() > session.snap_dist * 0.5 {
                            let in_box = |uv: Vec2| uv.cmpge(lo).all() && uv.cmple(hi).all();
                            let mut sel = Vec::new();
                            for (i, _) in session.sketch.entities.iter().enumerate() {
                                let pts = entity_points(&session.sketch, i);
                                if !pts.is_empty()
                                    && pts.iter().all(|&p| {
                                        session
                                            .sketch
                                            .points
                                            .get(p)
                                            .is_some_and(|q| in_box(Vec2::new(q.x as f32, q.y as f32)))
                                    })
                                {
                                    sel.push(i);
                                }
                            }
                            session.selected_entities = sel;
                        }
                    }
                    session.box_select = None;
                }
            }
        }
        Tool::Line | Tool::Circle | Tool::Rectangle | Tool::Slot | Tool::Polygon | Tool::Text if just_pressed => {
            // Use the snapped cursor so endpoints land on midpoints / quadrants / centres.
            if let Some(uv) = session.cursor_uv {
                place_point(&mut session, uv);
            }
        }
        Tool::Spline if just_pressed => {
            if let Some(uv) = session.cursor_uv {
                // Clicking the first point (with ≥3 placed) closes the loop and commits.
                if session.spline_pts.len() >= 3 && session.spline_pts[0].distance(uv) <= snap {
                    commit_spline(&mut session, true);
                } else {
                    session.spline_pts.push(uv);
                    session.request_live_focus = true;
                }
            }
        }
        Tool::Dimension if just_pressed => {
            if let Some(uv) = active_uv {
                // Clicking an existing dimension's label reopens it for editing.
                if let Some(ci) = dim_at(&session.sketch, uv, snap * 2.0) {
                    session.dim_first = None;
                    open_dim_edit(&mut session, ci, None);
                } else {
                    // Smart pick: a point starts/continues a point-to-point distance;
                    // a line dimensions its length (and can become an angle with a 2nd
                    // line); a circle dimensions its radius/diameter. A new dimension
                    // opens the Modify box so the exact value can be typed straight away.
                    let mut line_ctx = None;
                    let created = if let Some(p) = nearest_point(&session.sketch, uv, snap * 1.5) {
                        match session.dim_first.take() {
                            Some(first) if first != p => add_distance_dim(&mut session.sketch, first, p),
                            _ => {
                                session.dim_first = Some(p);
                                None
                            }
                        }
                    } else if let Some(e) = nearest_entity(&session.sketch, uv, snap * 2.0) {
                        session.dim_first = None;
                        if let Some((a, b)) = entity_line(&session.sketch, e) {
                            line_ctx = Some(e);
                            add_distance_dim(&mut session.sketch, a, b)
                        } else if let Some((center, radius)) = entity_circle(&session.sketch, e) {
                            add_radius_dim(&mut session.sketch, center, radius)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(ci) = created {
                        open_dim_edit(&mut session, ci, line_ctx);
                    }
                }
            }
        }
        _ => {}
    }

    if session.drag.is_none() && session.dirty {
        session.sketch.solve();
        session.dirty = false;
    }

    // ---- Per-operation sketch undo: snapshot a *settled* change ----
    // Only record between gestures (nothing in-progress), so one click / drag / dimension
    // edit is one undo step. The state is already solved here, so no jitter step is logged.
    if session.plane.is_some() {
        let settled = !buttons.pressed(MouseButton::Left)
            && session.pending.is_none()
            && session.pending_b.is_none()
            && session.pending_c.is_none()
            && session.drag.is_none()
            && session.text_handle.is_none()
            && session.dim_drag.is_none()
            && session.dim_edit.is_none()
            && session.spline_pts.is_empty()
            && session.box_select.is_none();
        if settled {
            let fp = sketch_fingerprint(&session.sketch);
            if session.undo_baseline.is_none() {
                session.undo_baseline = Some(session.sketch.clone());
                session.undo_fp = fp;
            } else if fp != session.undo_fp {
                let snapshot = session.sketch.clone();
                if let Some(prev) = session.undo_baseline.replace(snapshot) {
                    session.undo_sketch.push(prev);
                    if session.undo_sketch.len() > 200 {
                        session.undo_sketch.remove(0);
                    }
                }
                session.redo_sketch.clear();
                session.undo_fp = fp;
            }
        }
    }
}

/// Commit the in-progress line at an exact `length` (toward the current cursor),
/// adding a locking distance dimension. Used by the live dimension input.
fn commit_line_length(session: &mut SketchSession, length: f32) {
    let (Some(start), Some(cur)) = (session.pending, session.cursor_uv) else { return };
    let mut dir = (cur - start).normalize_or_zero();
    if dir == Vec2::ZERO {
        dir = Vec2::X;
    }
    let end = start + dir * length;
    let snap = session.snap_dist;
    let a = get_or_add_point(&mut session.sketch, start, snap);
    let b = get_or_add_point(&mut session.sketch, end, snap);
    session.sketch.add_line(a, b, session.construction);
    let off = (length as f64 * 0.2).max(0.5);
    session.sketch.constraints.push(Constraint::Distance { a, b, value: length as f64, offset: off, axis: DimAxis::Aligned });
    let tol = (snap * 0.6).max(1e-3);
    maybe_add_point_on_circle(&mut session.sketch, a, tol);
    maybe_add_point_on_circle(&mut session.sketch, b, tol);
    session.pending = None;
    session.dirty = true;
}

/// Commit the in-progress circle at an exact `radius`.
fn commit_circle_radius(session: &mut SketchSession, radius: f32) {
    let Some(center) = session.pending else { return };
    let c = get_or_add_point(&mut session.sketch, center, session.snap_dist);
    session.sketch.add_circle(c, radius.max(0.01) as f64);
    session.pending = None;
    session.dirty = true;
}

/// Commit the in-progress spline (≥2 points) as a Spline entity. `closed` wraps it.
fn commit_spline(session: &mut SketchSession, closed: bool) {
    let pts = std::mem::take(&mut session.spline_pts);
    if pts.len() < 2 {
        return;
    }
    let snap = session.snap_dist;
    let points: Vec<usize> = pts.iter().map(|p| get_or_add_point(&mut session.sketch, *p, snap)).collect();
    session.sketch.entities.push(SketchEntity::Spline {
        points,
        closed,
        construction: session.construction,
        control: session.spline_control,
    });
    session.dirty = true;
}

/// True if a distance dimension already exists between points `a` and `b`.
fn has_distance(sketch: &Sketch, a: usize, b: usize) -> bool {
    sketch.constraints.iter().any(|c| {
        matches!(c, Constraint::Distance { a: x, b: y, .. } if (*x == a && *y == b) || (*x == b && *y == a))
    })
}

/// True if the circle centred at point `center` already carries a driving radius dimension.
fn has_radius(sketch: &Sketch, center: usize) -> bool {
    sketch
        .constraints
        .iter()
        .any(|c| matches!(c, Constraint::Radius { center: x, .. } if *x == center))
}

/// Add a distance dimension between two points at their current distance (or return
/// the existing one for that pair). Returns the constraint index so the caller can
/// open the Modify box on it. `None` only for a degenerate same-point pick.
fn add_distance_dim(sketch: &mut Sketch, a: usize, b: usize) -> Option<usize> {
    if a == b {
        return None;
    }
    if let Some(i) = sketch.constraints.iter().position(|c| {
        matches!(c, Constraint::Distance { a: x, b: y, .. } if (*x == a && *y == b) || (*x == b && *y == a))
    }) {
        return Some(i);
    }
    let (pa, pb) = (sketch.points[a], sketch.points[b]);
    let d = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt();
    sketch.constraints.push(Constraint::Distance { a, b, value: d, offset: (d * 0.2).max(0.5), axis: DimAxis::Aligned });
    Some(sketch.constraints.len() - 1)
}

/// Add a driving radius dimension to the circle centred at `center` (or return the
/// existing one). Returns the constraint index so the Modify box can open on it.
fn add_radius_dim(sketch: &mut Sketch, center: usize, radius: f64) -> Option<usize> {
    if let Some(i) = sketch
        .constraints
        .iter()
        .position(|c| matches!(c, Constraint::Radius { center: x, .. } if *x == center))
    {
        return Some(i);
    }
    sketch.constraints.push(Constraint::Radius { center, value: radius, diameter: true });
    Some(sketch.constraints.len() - 1)
}

/// Add a new point at `xf(existing point)` and return its index — used to clone an
/// entity through a translation/rotation when patterning.
fn xf_point(sketch: &mut Sketch, pi: usize, xf: &impl Fn(Vec2) -> Vec2) -> usize {
    let p = sketch.points[pi];
    let np = xf(Vec2::new(p.x as f32, p.y as f32));
    sketch.add_point(np.x as f64, np.y as f64)
}

/// Append a copy of entity `idx` with all its points mapped through `xf` (a pattern
/// instance). Reference lines and text are skipped. New geometry is unconstrained.
fn duplicate_entity(sketch: &mut Sketch, idx: usize, xf: &impl Fn(Vec2) -> Vec2) {
    let Some(e) = sketch.entities.get(idx).cloned() else { return };
    let new = match e {
        SketchEntity::Line { a, b, construction, reference: false } => {
            let (a, b) = (xf_point(sketch, a, xf), xf_point(sketch, b, xf));
            SketchEntity::Line { a, b, construction, reference: false }
        }
        SketchEntity::Circle { center, radius, construction } => {
            let center = xf_point(sketch, center, xf);
            SketchEntity::Circle { center, radius, construction }
        }
        SketchEntity::Slot { a, b, radius, construction, mid } => {
            let a = xf_point(sketch, a, xf);
            let b = xf_point(sketch, b, xf);
            let mid = mid.map(|m| xf_point(sketch, m, xf));
            SketchEntity::Slot { a, b, radius, construction, mid }
        }
        SketchEntity::Spline { points, closed, construction, control } => {
            let points = points.iter().map(|&pi| xf_point(sketch, pi, xf)).collect();
            SketchEntity::Spline { points, closed, construction, control }
        }
        SketchEntity::Point { at } => SketchEntity::Point { at: xf_point(sketch, at, xf) },
        _ => return, // reference line / text: not patterned
    };
    sketch.entities.push(new);
}

/// The outline polylines of an entity in plane-uv — used to draw a ghost pattern preview.
/// A line is one open segment; a circle / slot a closed loop; a spline its tessellation.
fn entity_preview_polylines(sketch: &Sketch, idx: usize) -> Vec<Vec<Vec2>> {
    let pt = |i: usize| sketch.points.get(i).map(|p| Vec2::new(p.x as f32, p.y as f32));
    match sketch.entities.get(idx) {
        Some(SketchEntity::Line { a, b, .. }) => match (pt(*a), pt(*b)) {
            (Some(a), Some(b)) => vec![vec![a, b]],
            _ => vec![],
        },
        Some(SketchEntity::Circle { center, radius, .. }) => match pt(*center) {
            Some(c) => {
                const SEG: usize = 48;
                let r = *radius as f32;
                vec![(0..=SEG)
                    .map(|k| {
                        let a = std::f32::consts::TAU * k as f32 / SEG as f32;
                        c + Vec2::new(r * a.cos(), r * a.sin())
                    })
                    .collect()]
            }
            None => vec![],
        },
        Some(SketchEntity::Slot { a, b, radius, mid, .. }) => match (pt(*a), pt(*b)) {
            (Some(pa), Some(pb)) => {
                let poly = match mid.and_then(|m| pt(m)) {
                    Some(pm) => tessellate_arc_slot([pa.x as f64, pa.y as f64], [pm.x as f64, pm.y as f64], [pb.x as f64, pb.y as f64], *radius),
                    None => tessellate_slot([pa.x as f64, pa.y as f64], [pb.x as f64, pb.y as f64], *radius),
                };
                let mut v: Vec<Vec2> = poly.iter().map(|p| Vec2::new(p[0] as f32, p[1] as f32)).collect();
                if let Some(f) = v.first().copied() {
                    v.push(f); // close the loop
                }
                vec![v]
            }
            _ => vec![],
        },
        Some(SketchEntity::Spline { points, closed, control, .. }) => {
            let pts: Vec<[f64; 2]> = points.iter().filter_map(|&i| pt(i)).map(|p| [p.x as f64, p.y as f64]).collect();
            if pts.len() >= 2 {
                let poly = tessellate_spline(&pts, *closed, *control);
                vec![poly.iter().map(|p| Vec2::new(p[0] as f32, p[1] as f32)).collect()]
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}

/// Centroid of all points referenced by the given entities.
fn selection_centroid(sketch: &Sketch, entities: &[usize]) -> Vec2 {
    let mut sum = Vec2::ZERO;
    let mut n = 0.0;
    for &i in entities {
        for p in entity_points(sketch, i) {
            if let Some(q) = sketch.points.get(p) {
                sum += Vec2::new(q.x as f32, q.y as f32);
                n += 1.0;
            }
        }
    }
    if n > 0.0 {
        sum / n
    } else {
        Vec2::ZERO
    }
}

/// One pattern instance as an affine map: `p' = center + R(theta)·(p − center) + off`.
/// Translations leave `center`/`theta` zero; rotations leave `off` zero.
#[derive(Clone, Copy)]
struct Xf {
    center: Vec2,
    cos: f32,
    sin: f32,
    off: Vec2,
}

impl Xf {
    fn translate(off: Vec2) -> Self {
        Xf { center: Vec2::ZERO, cos: 1.0, sin: 0.0, off }
    }
    fn rotate(center: Vec2, theta: f32) -> Self {
        Xf { center, cos: theta.cos(), sin: theta.sin(), off: Vec2::ZERO }
    }
    fn apply(&self, p: Vec2) -> Vec2 {
        let d = p - self.center;
        self.center + Vec2::new(d.x * self.cos - d.y * self.sin, d.x * self.sin + d.y * self.cos) + self.off
    }
}

/// The selected entities that can serve as a pattern seed (no reference lines / text).
fn pattern_seeds(session: &SketchSession) -> Vec<usize> {
    session
        .selected_entities
        .iter()
        .copied()
        .filter(|&i| {
            !matches!(
                session.sketch.entities.get(i),
                None | Some(SketchEntity::Line { reference: true, .. }) | Some(SketchEntity::Text { .. })
            )
        })
        .collect()
}

/// The instance transforms for the current pattern (excluding the seed/original). Shared by
/// the live preview and Apply so they always match. `Err` carries a banner-friendly reason.
fn pattern_instances(session: &SketchSession, seeds: &[usize]) -> Result<Vec<Xf>, String> {
    let mut xfs = Vec::new();
    match session.pattern_mode {
        PatternMode::Linear => {
            if seeds.is_empty() {
                return Err("Pattern: select the sketch geometry to repeat first.".into());
            }
            let (n1, n2) = (session.pat_count1.max(1), session.pat_count2.max(1));
            let (s1, s2) = (session.pat_spacing1, session.pat_spacing2);
            for i in 0..n1 {
                for j in 0..n2 {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    xfs.push(Xf::translate(Vec2::new(i as f32 * s1, j as f32 * s2)));
                }
            }
        }
        PatternMode::Circular => {
            if seeds.is_empty() {
                return Err("Pattern: select the sketch geometry to repeat first.".into());
            }
            let count = session.pat_circ_count.max(2);
            let center = if session.pat_center_set {
                session.pat_circ_center
            } else {
                selection_centroid(&session.sketch, seeds)
            };
            let full = (session.pat_circ_angle - 360.0).abs() < 0.5;
            let divs = if full { count as f32 } else { (count.max(2) - 1) as f32 };
            let step = session.pat_circ_angle.to_radians() / divs;
            for k in 1..count {
                xfs.push(Xf::rotate(center, step * k as f32));
            }
        }
        PatternMode::Fill => {
            let regions = session.sketch.regions();
            let Some(&bi) = session.selected_contours.first() else {
                return Err("Fill: click inside a closed region to choose the boundary to fill.".into());
            };
            let Some(region) = regions.get(bi) else {
                return Err("Fill: the chosen boundary region no longer exists.".into());
            };
            if seeds.is_empty() {
                return Err("Fill: select a small shape (the seed) to tile with.".into());
            }
            let seedc = selection_centroid(&session.sketch, seeds);
            let outer = &region.outer;
            let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for p in outer {
                lo[0] = lo[0].min(p[0]);
                lo[1] = lo[1].min(p[1]);
                hi[0] = hi[0].max(p[0]);
                hi[1] = hi[1].max(p[1]);
            }
            let sp = session.pat_fill_spacing.max(0.05) as f64;
            let m = session.pat_fill_margin.max(0.0) as f64;
            let inside = |x: f64, y: f64| {
                point_in_poly([x, y], outer) && !region.holes.iter().any(|h| point_in_poly([x, y], h))
            };
            let ok = |x: f64, y: f64| {
                inside(x, y)
                    && (m <= 0.0
                        || (inside(x + m, y) && inside(x - m, y) && inside(x, y + m) && inside(x, y - m)))
            };
            // Cap the instance count so a tiny spacing can't spawn a runaway preview.
            let mut y = lo[1];
            while y <= hi[1] && xfs.len() < 5000 {
                let mut x = lo[0];
                while x <= hi[0] && xfs.len() < 5000 {
                    if ok(x, y) {
                        let off = Vec2::new(x as f32 - seedc.x, y as f32 - seedc.y);
                        if off.length() > 1e-3 {
                            xfs.push(Xf::translate(off));
                        }
                    }
                    x += sp;
                }
                y += sp;
            }
        }
    }
    if xfs.is_empty() {
        return Err("Pattern: nothing was generated (check counts / spacing / boundary).".into());
    }
    Ok(xfs)
}

/// Generate a pattern of the selected geometry. Returns the number of copies made, or an
/// error string for the banner. The seed geometry stays; new instances are appended.
fn apply_pattern(session: &mut SketchSession) -> Result<usize, String> {
    let seeds = pattern_seeds(session);
    let xfs = pattern_instances(session, &seeds)?;
    for xf in &xfs {
        let map = |p: Vec2| xf.apply(p);
        for &e in &seeds {
            duplicate_entity(&mut session.sketch, e, &map);
        }
    }
    session.dirty = true;
    Ok(xfs.len() * seeds.len())
}

/// Reflect point `p` across the infinite line through `a`–`b`.
fn reflect_across(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let d = (b - a).normalize_or_zero();
    if d == Vec2::ZERO {
        return p;
    }
    let v = p - a;
    a + d * (2.0 * v.dot(d)) - v
}

/// Among the selected entities, the one to use as the mirror axis: a construction line is
/// preferred (the natural centre line), else any line. Returns its (entity index, endpoints).
fn mirror_axis(session: &SketchSession) -> Option<(usize, Vec2, Vec2)> {
    let ep = |i: usize| {
        entity_line(&session.sketch, i).map(|(a, b)| {
            let pa = session.sketch.points[a];
            let pb = session.sketch.points[b];
            (Vec2::new(pa.x as f32, pa.y as f32), Vec2::new(pb.x as f32, pb.y as f32))
        })
    };
    // Prefer a construction line; otherwise the first selected line.
    let con = session.selected_entities.iter().copied().find(|&i| {
        matches!(session.sketch.entities.get(i), Some(SketchEntity::Line { construction: true, .. }))
    });
    let axis = con.or_else(|| session.selected_entities.iter().copied().find(|&i| entity_line(&session.sketch, i).is_some()))?;
    let (a, b) = ep(axis)?;
    Some((axis, a, b))
}

/// Entities to mirror = selected geometry minus the axis line (and minus reference/text).
fn mirror_seeds(session: &SketchSession, axis: usize) -> Vec<usize> {
    session
        .selected_entities
        .iter()
        .copied()
        .filter(|&i| {
            i != axis
                && !matches!(
                    session.sketch.entities.get(i),
                    None | Some(SketchEntity::Line { reference: true, .. }) | Some(SketchEntity::Text { .. })
                )
        })
        .collect()
}

/// Reflect the selected geometry across the selected axis line, appending the copies.
fn apply_mirror(session: &mut SketchSession) -> Result<usize, String> {
    let Some((axis, a, b)) = mirror_axis(session) else {
        return Err("Mirror: select a line (or construction centre line) to mirror across.".into());
    };
    let seeds = mirror_seeds(session, axis);
    if seeds.is_empty() {
        return Err("Mirror: also select the geometry to mirror.".into());
    }
    let map = |p: Vec2| reflect_across(p, a, b);
    for &e in &seeds {
        duplicate_entity(&mut session.sketch, e, &map);
    }
    session.dirty = true;
    Ok(seeds.len())
}

/// The index of a dimension whose label sits within `tol` (plane uv) of `uv` — used
/// to click an existing dimension and reopen its Modify box.
fn dim_at(sketch: &Sketch, uv: Vec2, tol: f32) -> Option<usize> {
    let pt = |i: usize| sketch.points.get(i).copied().map(|p| Vec2::new(p.x as f32, p.y as f32));
    sketch.constraints.iter().enumerate().find_map(|(i, c)| {
        let anchor = match c {
            Constraint::Distance { a, b, offset, axis, .. } => match (pt(*a), pt(*b)) {
                (Some(a2), Some(b2)) => Some(distance_dim_geometry(a2, b2, *offset as f32, *axis).2),
                _ => None,
            },
            Constraint::Radius { center, value, .. } => {
                pt(*center).map(|cu| cu + Vec2::new(*value as f32 * 0.707, *value as f32 * 0.707))
            }
            Constraint::Angle { a, b, c, d, offset, .. } => match (pt(*a), pt(*b), pt(*c), pt(*d)) {
                (Some(a2), Some(b2), Some(c2), Some(d2)) => Some(angle_dim_geometry(a2, b2, c2, d2, *offset as f32).1),
                _ => None,
            },
            Constraint::PointLineDistance { p, a, b, .. } => match (pt(*p), pt(*a), pt(*b)) {
                (Some(pp), Some(a2), Some(b2)) => Some(point_line_geometry(pp, a2, b2).1),
                _ => None,
            },
            _ => None,
        };
        anchor.filter(|p| (*p - uv).length() <= tol).map(|_| i)
    })
}

/// Open the Modify box on dimension `ci`, seeding the edit buffer in the units the box
/// shows (diameter ×2, angle in degrees). `line` carries a single-line context so a
/// length dim can still morph into an angle if a second line is then clicked.
fn open_dim_edit(session: &mut SketchSession, ci: usize, line: Option<usize>) {
    let buf = match session.sketch.constraints.get(ci) {
        Some(Constraint::Distance { value, .. }) => *value,
        Some(Constraint::Radius { value, diameter, .. }) => {
            if *diameter {
                value * 2.0
            } else {
                *value
            }
        }
        Some(Constraint::Angle { value, .. }) => value.to_degrees(),
        Some(Constraint::PointLineDistance { value, .. }) => *value,
        _ => return,
    };
    session.dim_edit = Some(ci);
    session.dim_buf = buf;
    session.dim_edit_focus = true;
    session.dim_line = line;
}

/// Replace a freshly-placed line-length dimension (`ci`) with an angle dimension
/// between lines `l1` and `l2`. Returns the new constraint index.
/// Add an angle dimension between two line entities (or return the existing one). The
/// stored angle is kept positive by ordering the lines so the directed angle is ≥ 0; the
/// first line in that order is the reference that stays put when the dimension is edited.
fn add_angle_dim(sketch: &mut Sketch, l1: usize, l2: usize) -> Option<usize> {
    let (a, b) = entity_line(sketch, l1)?;
    let (c, d) = entity_line(sketch, l2)?;
    if let Some(i) = sketch.constraints.iter().position(|k| {
        matches!(k, Constraint::Angle { a: x, b: y, c: z, d: w, .. }
            if (*x == a && *y == b && *z == c && *w == d) || (*x == c && *y == d && *z == a && *w == b))
    }) {
        return Some(i);
    }
    let p = |i: usize| {
        let q = sketch.points[i];
        Vec2::new(q.x as f32, q.y as f32)
    };
    let vertex = line_intersection(p(a), p(b), p(c), p(d)).unwrap_or((p(a) + p(b) + p(c) + p(d)) * 0.25);
    // Order each line so its *first* point is the one nearer the vertex; then (second −
    // first) is the ray pointing away from the vertex, and the directed angle between the
    // two rays is the actual wedge between the lines as drawn (not its supplement).
    let (a, b) = if (p(a) - vertex).length() <= (p(b) - vertex).length() { (a, b) } else { (b, a) };
    let (c, d) = if (p(c) - vertex).length() <= (p(d) - vertex).length() { (c, d) } else { (d, c) };
    let (v1, v2) = (p(b) - p(a), p(d) - p(c));
    let mut ang = (v1.x * v2.y - v1.y * v2.x).atan2(v1.x * v2.x + v1.y * v2.y) as f64;
    let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
    if ang < 0.0 {
        // Reverse the line order so the wedge angle reads positive.
        ang = -ang;
        std::mem::swap(&mut aa, &mut cc);
        std::mem::swap(&mut bb, &mut dd);
    }
    let off = (v1.length().min(v2.length()) * 0.4).max(1.0) as f64;
    sketch.constraints.push(Constraint::Angle { a: aa, b: bb, c: cc, d: dd, value: ang, offset: off });
    Some(sketch.constraints.len() - 1)
}

fn convert_length_to_angle(session: &mut SketchSession, ci: usize, l1: usize, l2: usize) -> Option<usize> {
    // Drop the length dim we just made (it's the last one) so the angle replaces it.
    if ci + 1 == session.sketch.constraints.len() {
        session.sketch.constraints.pop();
    }
    // Reuse the shared builder so the click-convert and the relations-panel "Angle" button
    // produce identical (vertex-first, positive-wedge) angle dimensions.
    add_angle_dim(&mut session.sketch, l1, l2)
}

/// The two endpoints of a linear dimension's offset line plus its label anchor, all
/// in plane uv. `offset` pushes the line off the measured geometry (perpendicular for
/// an aligned dim; vertical/horizontal displacement for a projected one). Shared by
/// the drawing, labelling, Modify-box, and offset-drag code so they stay in sync.
fn distance_dim_geometry(a2: Vec2, b2: Vec2, offset: f32, axis: DimAxis) -> (Vec2, Vec2, Vec2) {
    match axis {
        DimAxis::Aligned => {
            let dir = (b2 - a2).normalize_or_zero();
            let perp = Vec2::new(-dir.y, dir.x) * offset;
            (a2 + perp, b2 + perp, (a2 + b2) * 0.5 + perp)
        }
        DimAxis::Horizontal => {
            let y = (a2.y + b2.y) * 0.5 + offset;
            (Vec2::new(a2.x, y), Vec2::new(b2.x, y), Vec2::new((a2.x + b2.x) * 0.5, y))
        }
        DimAxis::Vertical => {
            let x = (a2.x + b2.x) * 0.5 + offset;
            (Vec2::new(x, a2.y), Vec2::new(x, b2.y), Vec2::new(x, (a2.y + b2.y) * 0.5))
        }
    }
}

/// Set a dimension's display offset (how far its line sits off the geometry) from the
/// cursor — used when dragging a placed dimension with the Select tool.
fn set_dim_offset_from_cursor(session: &mut SketchSession, ci: usize, uv: Vec2) {
    let pt = |i: usize| session.sketch.points.get(i).copied().map(|p| Vec2::new(p.x as f32, p.y as f32));
    let new_off = match session.sketch.constraints.get(ci) {
        Some(Constraint::Distance { a, b, axis, .. }) => match (pt(*a), pt(*b)) {
            (Some(a2), Some(b2)) => {
                let mid = (a2 + b2) * 0.5;
                Some(match axis {
                    DimAxis::Aligned => {
                        let dir = (b2 - a2).normalize_or_zero();
                        (uv - mid).dot(Vec2::new(-dir.y, dir.x)) as f64
                    }
                    DimAxis::Horizontal => (uv.y - mid.y) as f64,
                    DimAxis::Vertical => (uv.x - mid.x) as f64,
                })
            }
            _ => None,
        },
        Some(Constraint::Angle { a, b, c, d, .. }) => match (pt(*a), pt(*b), pt(*c), pt(*d)) {
            (Some(a2), Some(b2), Some(c2), Some(d2)) => {
                let (vertex, _) = angle_dim_geometry(a2, b2, c2, d2, 0.0);
                Some((uv - vertex).length().max(0.5) as f64)
            }
            _ => None,
        },
        _ => None,
    };
    match (new_off, session.sketch.constraints.get_mut(ci)) {
        (Some(o), Some(Constraint::Distance { offset, .. })) => *offset = o,
        (Some(o), Some(Constraint::Angle { offset, .. })) => *offset = o,
        _ => {}
    }
}

/// Switch a distance dimension between aligned / horizontal / vertical, recomputing
/// its driving value from the current geometry so the toggle doesn't jerk the sketch.
fn set_distance_axis(session: &mut SketchSession, ci: usize, new: DimAxis) {
    let measured = if let Some(Constraint::Distance { a, b, .. }) = session.sketch.constraints.get(ci) {
        match (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
            (Some(pa), Some(pb)) => {
                let (dx, dy) = (pa.x - pb.x, pa.y - pb.y);
                Some(match new {
                    DimAxis::Aligned => (dx * dx + dy * dy).sqrt(),
                    DimAxis::Horizontal => dx.abs(),
                    DimAxis::Vertical => dy.abs(),
                })
            }
            _ => None,
        }
    } else {
        None
    };
    if let Some(Constraint::Distance { value, axis, .. }) = session.sketch.constraints.get_mut(ci) {
        *axis = new;
        if let Some(m) = measured {
            *value = m.max(0.001);
        }
    }
    if let Some(m) = measured {
        session.dim_buf = m.max(0.001);
    }
    session.sketch.solve();
}

/// Foot of the perpendicular from `p` onto the line through (a,b), and the dimension
/// label anchor (midpoint of the leader). Used by point-to-line distance dimensions.
fn point_line_geometry(p: Vec2, a: Vec2, b: Vec2) -> (Vec2, Vec2) {
    let ab = b - a;
    let t = if ab.length_squared() > 1e-9 { (p - a).dot(ab) / ab.length_squared() } else { 0.0 };
    let foot = a + ab * t;
    (foot, (p + foot) * 0.5)
}

/// Intersection of the infinite lines through (p1,p2) and (p3,p4), if not parallel.
fn line_intersection(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Option<Vec2> {
    let (d1, d2) = (p2 - p1, p4 - p3);
    let denom = d1.x * d2.y - d1.y * d2.x;
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / denom;
    Some(p1 + d1 * t)
}

/// The vertex and label anchor of an angle dimension between lines (a→b) and (c→d).
/// The label sits `offset` out along the bisector from the vertex, on the side *between*
/// the two lines as they emanate from the vertex (so the arc spans the lines you picked).
fn angle_dim_geometry(a2: Vec2, b2: Vec2, c2: Vec2, d2: Vec2, offset: f32) -> (Vec2, Vec2) {
    let vertex = line_intersection(a2, b2, c2, d2).unwrap_or((a2 + b2 + c2 + d2) * 0.25);
    // Rays pointing away from the vertex along each line (toward the line's far endpoint),
    // so the bisector lands in the wedge actually between the two drawn lines.
    let ray = |p: Vec2, q: Vec2| {
        let far = if (p - vertex).length() >= (q - vertex).length() { p } else { q };
        (far - vertex).normalize_or_zero()
    };
    let dir1 = ray(a2, b2);
    let dir2 = ray(c2, d2);
    let mut bisect = (dir1 + dir2).normalize_or_zero();
    if bisect == Vec2::ZERO {
        bisect = Vec2::new(-dir1.y, dir1.x);
    }
    (vertex, vertex + bisect * offset)
}

/// After a line `a→b` is committed, lock in the relation the user clearly intended so later
/// edits keep the sketch square:
///   * a near-axis line gets a `Horizontal` / `Vertical` constraint;
///   * otherwise a line meeting an existing line at ~90° (sharing an endpoint *or* touching
///     it at a corner / T-junction / crossing) gets a `Perpendicular` constraint.
/// The point *connection* itself is already preserved — snapped endpoints share one point
/// index — so this only adds the angular relation. Tolerances are tight (≈2°): the 90°
/// preview snap already projects an intended-square line exactly onto its axis, so this
/// fires for those and leaves a deliberately diagonal line alone.
fn add_square_relations(sketch: &mut Sketch, a: usize, b: usize) {
    let p = |i: usize| Vec2::new(sketch.points[i].x as f32, sketch.points[i].y as f32);
    let v = p(b) - p(a);
    let len = v.length();
    if len < 1e-5 {
        return;
    }
    let sin_tol = 2.0_f32.to_radians().sin();
    let axis_tol = len * sin_tol; // perpendicular gap from the axis at this length
    // Horizontal / vertical — these also make two square lines implicitly perpendicular.
    if v.y.abs() <= axis_tol {
        if !sketch.constraints.iter().any(|c| matches!(c, Constraint::Horizontal(x, y) if (*x == a && *y == b) || (*x == b && *y == a))) {
            sketch.constraints.push(Constraint::Horizontal(a, b));
        }
        return;
    }
    if v.x.abs() <= axis_tol {
        if !sketch.constraints.iter().any(|c| matches!(c, Constraint::Vertical(x, y) if (*x == a && *y == b) || (*x == b && *y == a))) {
            sketch.constraints.push(Constraint::Vertical(a, b));
        }
        return;
    }
    // Otherwise: perpendicular to a *connected* non-reference sketch line. "Connected"
    // means sharing an endpoint OR touching geometrically — either line's endpoint landing
    // on the other (an L-corner, a T-junction, or a crossing). This is what keeps a 90°
    // joint square when the sketch is later dragged around, instead of going free.
    let dirn = v / len;
    let touch_tol = (len * 0.02).clamp(1e-3, 0.1); // small absolute proximity for a touch
    let lines: Vec<(usize, usize)> = sketch
        .entities
        .iter()
        .filter_map(|e| match e {
            SketchEntity::Line { a: c, b: d, reference: false, .. }
                if !((*c == a && *d == b) || (*c == b && *d == a)) =>
            {
                Some((*c, *d))
            }
            _ => None,
        })
        .collect();
    let (pa, pb) = (p(a), p(b));
    for (c, d) in lines {
        let w = p(d) - p(c);
        let wl = w.length();
        if wl < 1e-5 {
            continue;
        }
        // Connected if they share a point, or an endpoint of one lands on the other.
        let shares = c == a || c == b || d == a || d == b;
        let touches = shares
            || closest_on_segment(pa, p(c), p(d)).distance(pa) <= touch_tol
            || closest_on_segment(pb, p(c), p(d)).distance(pb) <= touch_tol
            || closest_on_segment(p(c), pa, pb).distance(p(c)) <= touch_tol
            || closest_on_segment(p(d), pa, pb).distance(p(d)) <= touch_tol;
        if !touches {
            continue;
        }
        if (dirn.dot(w / wl)).abs() <= sin_tol {
            let dup = sketch.constraints.iter().any(|k| {
                matches!(k, Constraint::Perpendicular(a1, b1, c1, d1)
                    if (*a1 == a && *b1 == b && *c1 == c && *d1 == d)
                        || (*a1 == c && *b1 == d && *c1 == a && *d1 == b))
            });
            if !dup {
                sketch.constraints.push(Constraint::Perpendicular(a, b, c, d));
            }
            return;
        }
    }
}

fn place_point(session: &mut SketchSession, uv: Vec2) {
    let snap = session.snap_dist;
    match session.tool {
        Tool::Line if session.line_midpoint => {
            // Midpoint line: the first click is the centre; the line grows symmetrically.
            if let Some(center) = session.pending.take() {
                let e1 = center * 2.0 - uv; // mirror of the cursor about the centre
                let a = get_or_add_point(&mut session.sketch, e1, snap);
                let b = get_or_add_point_ref(session, uv, snap);
                let mid = get_or_add_point(&mut session.sketch, center, snap);
                session.sketch.add_line(a, b, session.construction);
                if session.construction {
                    // Construction centre line: leave the two halves free (no Midpoint
                    // constraint) so the PropertyManager can dimension each side and
                    // equalise them by sliding the endpoints along the line.
                    session.center_line = Some([a, mid, b]);
                } else {
                    session.sketch.constraints.push(Constraint::Midpoint { mid, a, b });
                }
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Line => {
            if let Some(start) = session.pending.take() {
                let a = get_or_add_point_ref(session, start, snap);
                let b = get_or_add_point_ref(session, uv, snap);
                session.sketch.add_line(a, b, session.construction);
                // Persist the square/perpendicular relation the 90° snap implied, so resizing
                // keeps the sketch square (the shared point already keeps lines connected).
                add_square_relations(&mut session.sketch, a, b);
                // If either endpoint landed on a circle rim, link it parametrically.
                let tol = (snap * 0.6).max(1e-3);
                maybe_add_point_on_circle(&mut session.sketch, a, tol);
                maybe_add_point_on_circle(&mut session.sketch, b, tol);
                // If either endpoint landed on the span of another sketch line, pin it
                // there so the snap is a real relation that survives later moves.
                maybe_add_point_on_sketch_line(&mut session.sketch, a, tol);
                maybe_add_point_on_sketch_line(&mut session.sketch, b, tol);
                // If either endpoint landed on a body edge, add a point-on-edge relation
                // (two on one line make it collinear with — snapped along — the edge).
                let start_edge = session.pending_edge.take();
                maybe_add_point_on_edge(session, a, start_edge);
                let end_edge = session.cursor_edge;
                maybe_add_point_on_edge(session, b, end_edge);
                session.dirty = true;
                // A construction line is a one-shot: revert to the regular line tool after
                // drawing one (re-pick "Construction Line" for another).
                session.construction = false;
            } else {
                session.pending = Some(uv);
                session.pending_edge = session.cursor_edge; // remember the start's edge
                session.request_live_focus = true;
            }
        }
        Tool::Circle if session.circle_perimeter => {
            // Perimeter circle: the two clicks are opposite ends of a diameter, so the
            // centre is their midpoint and the radius is half the distance.
            if let Some(p1) = session.pending.take() {
                let center = (p1 + uv) * 0.5;
                let radius = ((uv - p1).length() * 0.5).max(0.01);
                let c = get_or_add_point(&mut session.sketch, center, snap);
                session.sketch.add_circle(c, radius as f64);
                session.dirty = true;
            } else {
                session.pending = Some(uv); // first rim point
                session.request_live_focus = true;
            }
        }
        Tool::Circle => {
            if let Some(center) = session.pending.take() {
                let radius = snap_radius(center.distance(uv), &session.reference_circles, snap);
                let c = get_or_add_point_ref(session, center, snap);
                session.sketch.add_circle(c, radius as f64);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Rectangle => match session.rect_mode {
            RectMode::Corner => {
                if let Some(c0) = session.pending.take() {
                    rect_axis_aligned(&mut session.sketch, c0, uv, session.construction);
                    session.dirty = true;
                } else {
                    session.pending = Some(uv);
                }
            }
            RectMode::Center => {
                if let Some(center) = session.pending.take() {
                    commit_center_rect(session, center, uv);
                    session.dirty = true;
                } else {
                    session.pending = Some(uv);
                }
            }
            RectMode::Parallelogram => {
                // 1st click → A; 2nd → B (anchors a side); 3rd → C (pulls out the shape).
                if session.pending.is_none() {
                    session.pending = Some(uv);
                } else if session.pending_b.is_none() {
                    session.pending_b = Some(uv);
                } else {
                    let a = session.pending.take().unwrap();
                    let b = session.pending_b.take().unwrap();
                    commit_parallelogram(&mut session.sketch, a, b, uv, session.construction);
                    session.dirty = true;
                }
            }
        },
        Tool::Slot => match session.slot_mode {
            SlotMode::Straight => {
                if session.pending.is_none() {
                    session.pending = Some(uv);
                } else if session.pending_b.is_none() {
                    session.pending_b = Some(uv);
                } else {
                    let a = session.pending.take().unwrap();
                    let b = session.pending_b.take().unwrap();
                    commit_slot(session, a, b, None, perp_dist(uv, a, b));
                }
            }
            SlotMode::Centerpoint => {
                if session.pending.is_none() {
                    session.pending = Some(uv); // centre
                } else if session.pending_b.is_none() {
                    session.pending_b = Some(uv); // one end
                } else {
                    let center = session.pending.take().unwrap();
                    let end = session.pending_b.take().unwrap();
                    let a = center * 2.0 - end; // mirrored end
                    commit_slot(session, a, end, None, perp_dist(uv, a, end));
                }
            }
            SlotMode::Arc => {
                if session.pending.is_none() {
                    session.pending = Some(uv); // end A
                } else if session.pending_b.is_none() {
                    session.pending_b = Some(uv); // end B
                } else if session.pending_c.is_none() {
                    session.pending_c = Some(uv); // bend point
                } else {
                    let a = session.pending.take().unwrap();
                    let b = session.pending_b.take().unwrap();
                    let p = session.pending_c.take().unwrap();
                    commit_slot(session, a, b, Some(p), arc_slot_width(uv, a, p, b));
                }
            }
        },
        Tool::Polygon => {
            // 1st click → centre; 2nd click → a vertex (sets the circumscribed radius
            // and the orientation).
            if let Some(center) = session.pending.take() {
                let rim = poly_rim(session).unwrap_or(uv);
                commit_polygon(session, center, rim);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Text => {
            // A single click drops the text at the cursor (its baseline start).
            commit_text(session, uv);
            session.dirty = true;
        }
        // Spline points are placed in `sketch_interaction` (it needs the full point list);
        // Pattern / Mirror act on existing geometry rather than placing points.
        Tool::Select | Tool::Dimension | Tool::Spline | Tool::Pattern | Tool::Mirror => {}
    }
}

/// Register one (regular) face per system font family with egui, under a `Name` family
/// equal to the family name, so the font dropdown can render each entry in its own
/// typeface. Done once; reads font files (capped to skip huge CJK files), then rebuilds
/// egui's font atlas a single time.
fn register_system_fonts(ctx: &egui::Context, previews: &mut FontPreviews) {
    let mut defs = egui::FontDefinitions::default(); // keep egui's built-in UI fonts
    for (family, bytes, index) in text::family_preview_data(4 * 1024 * 1024) {
        let mut fd = egui::FontData::from_owned(bytes);
        fd.index = index;
        defs.font_data.insert(family.clone(), std::sync::Arc::new(fd));
        defs.families
            .entry(egui::FontFamily::Name(std::sync::Arc::from(family.as_str())))
            .or_default()
            .push(family.clone());
        previews.families.insert(family);
    }
    ctx.set_fonts(defs);
    previews.done = true;
}

/// An edge's orientation-independent identity: its two extreme endpoints, quantised. Used
/// so distinct edges always stay distinct in the fillet set and only a genuine re-click of
/// the same edge toggles it off.
fn fillet_edge_key(poly: &[[f64; 3]]) -> ([i64; 3], [i64; 3]) {
    let q = |p: [f64; 3]| [(p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64, (p[2] * 1e3).round() as i64];
    let (a, b) = (q(poly[0]), q(poly[poly.len() - 1]));
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Add a body edge (its world-space polyline) to the fillet set, or remove it if the same
/// edge is clicked again. Refreshes the preview.
fn toggle_fillet_edge(ui_state: &mut UiState, chain: &[Vec3]) {
    let poly: Vec<[f64; 3]> = chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect();
    let key = fillet_edge_key(&poly);
    let before = ui_state.fillet_edges.len();
    ui_state.fillet_edges.retain(|e| fillet_edge_key(e) != key);
    if ui_state.fillet_edges.len() == before {
        ui_state.fillet_edges.push(poly); // wasn't present → add it
    }
    ui_state.fillet_shown = None; // recompute the fillet/chamfer preview
    ui_state.chamfer_shown = None;
}

/// Fill in sensible Text-tool defaults the first time it's opened (the system default
/// font, a non-empty string, a visible height).
fn init_text_defaults(session: &mut SketchSession) {
    if !session.text_font_init {
        session.text_font = text::default_family();
        session.text_font_init = true;
    }
    if session.text_string.is_empty() {
        session.text_string = "Text".to_string();
    }
    if session.text_height <= 0.0 {
        session.text_height = 1.0;
    }
}

/// Seed sensible pattern defaults the first time the Pattern tool is opened in a session.
fn init_pattern_defaults(session: &mut SketchSession) {
    if session.pattern_init {
        return;
    }
    session.pattern_init = true;
    session.pat_count1 = 3;
    session.pat_count2 = 1;
    session.pat_spacing1 = 2.0;
    session.pat_spacing2 = 2.0;
    session.pat_circ_count = 6;
    session.pat_circ_angle = 360.0;
    session.pat_fill_spacing = 1.5;
    session.pat_fill_margin = 0.0;
}

/// Bake the current text parameters into outline contours and add a Text entity whose
/// baseline starts at `at`. No-op (with a status hint) if the font yields no outlines.
fn commit_text(session: &mut SketchSession, at: Vec2) {
    let contours = text::glyph_contours(
        &session.text_font,
        session.text_bold,
        session.text_italic,
        &session.text_string,
        session.text_spacing,
    );
    if contours.is_empty() {
        return;
    }
    let origin = session.sketch.add_point(at.x as f64, at.y as f64);
    session.sketch.entities.push(SketchEntity::Text {
        origin,
        contours,
        height: session.text_height.max(0.05) as f64,
        rotation: 0.0,
        mirror: session.text_mirror,
        arc: session.text_arc,
        text: session.text_string.clone(),
        font: session.text_font.clone(),
        bold: session.text_bold,
        italic: session.text_italic,
        spacing: session.text_spacing,
    });
}

/// Re-bake the outline contours of the Text entity at `idx` from its stored parameters
/// (after a font / style / string / spacing edit), keeping its placement transform.
fn rebake_text(session: &mut SketchSession, idx: usize) {
    let params = match session.sketch.entities.get(idx) {
        Some(SketchEntity::Text { text, font, bold, italic, spacing, .. }) => {
            (text.clone(), font.clone(), *bold, *italic, *spacing)
        }
        _ => return,
    };
    let baked = text::glyph_contours(&params.1, params.2, params.3, &params.0, params.4);
    if baked.is_empty() {
        return;
    }
    if let Some(SketchEntity::Text { contours, .. }) = session.sketch.entities.get_mut(idx) {
        *contours = baked;
        session.dirty = true;
    }
}

/// Apply a Text handle drag: `Scale` sets the height from the cursor's distance to the
/// origin; `Rotate` sets the rotation from the cursor's angle about the origin.
fn apply_text_handle(session: &mut SketchSession, handle: TextHandle, cur: Vec2) {
    let idx = match handle {
        TextHandle::Scale(i) | TextHandle::Rotate(i) => i,
    };
    let (origin, height, mirror, arc) = match session.sketch.entities.get(idx) {
        Some(SketchEntity::Text { origin, height, mirror, arc, .. }) => (*origin, *height, *mirror, *arc),
        _ => return,
    };
    let contours = match session.sketch.entities.get(idx) {
        Some(SketchEntity::Text { contours, .. }) => contours.clone(),
        _ => return,
    };
    let Some(op) = session.sketch.points.get(origin) else { return };
    let o = Vec2::new(op.x as f32, op.y as f32);
    // Reference handle vectors at rotation = 0 (the unrotated frame).
    let loops0 = text_contours([o.x as f64, o.y as f64], &contours, height, 0.0, mirror, arc);
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for l in &loops0 {
        for p in l {
            minx = minx.min(p[0]);
            maxx = maxx.max(p[0]);
            miny = miny.min(p[1]);
            maxy = maxy.max(p[1]);
        }
    }
    if minx > maxx {
        return;
    }
    let to_cursor = cur - o;
    match handle {
        TextHandle::Rotate(_) => {
            let base = Vec2::new(((minx + maxx) * 0.5) as f32, maxy as f32);
            let rvec = base + Vec2::new(0.0, (0.6 * height).max(0.1) as f32) - o;
            if to_cursor.length() > 1e-4 && rvec.length() > 1e-4 {
                let rot = to_cursor.y.atan2(to_cursor.x) - rvec.y.atan2(rvec.x);
                if let Some(SketchEntity::Text { rotation, .. }) = session.sketch.entities.get_mut(idx) {
                    *rotation = rot as f64;
                }
            }
        }
        TextHandle::Scale(_) => {
            let svec = Vec2::new(maxx as f32, miny as f32) - o;
            let unit = (svec.length() / height.max(0.05) as f32).max(1e-4);
            let new_h = (to_cursor.length() / unit).max(0.05);
            if let Some(SketchEntity::Text { height: hh, .. }) = session.sketch.entities.get_mut(idx) {
                *hh = new_h as f64;
            }
            session.text_height = new_h;
        }
    }
    session.dirty = true;
}

/// On-canvas handle positions (plane uv) for the Text entity at `idx`: `(scale, rotate,
/// base)`. The scale handle sits at the bounding box's advance-side bottom corner; the
/// rotate handle floats above the box top centre (`base` is on the box edge below it).
fn text_handles(sketch: &Sketch, idx: usize) -> Option<(Vec2, Vec2, Vec2)> {
    let (origin, contours, height, rotation, mirror, arc) = match sketch.entities.get(idx) {
        Some(SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. }) => {
            (*origin, contours, *height, *rotation, *mirror, *arc)
        }
        _ => return None,
    };
    let o = sketch.points.get(origin)?;
    let loops = text_contours([o.x, o.y], contours, height, rotation, mirror, arc);
    let (mut minx, mut maxx, mut miny, mut maxy) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for l in &loops {
        for p in l {
            minx = minx.min(p[0]);
            maxx = maxx.max(p[0]);
            miny = miny.min(p[1]);
            maxy = maxy.max(p[1]);
        }
    }
    if minx > maxx {
        return None;
    }
    let scale = Vec2::new(maxx as f32, miny as f32);
    let base = Vec2::new(((minx + maxx) * 0.5) as f32, maxy as f32);
    let rotate = base + Vec2::new(0.0, (0.6 * height).max(0.1) as f32);
    Some((scale, rotate, base))
}

/// Build a slot entity from end centres `a`,`b`, optional arc bend `mid`, and half-width `r`.
fn commit_slot(session: &mut SketchSession, a: Vec2, b: Vec2, mid: Option<Vec2>, r: f32) {
    let snap = session.snap_dist;
    let pa = get_or_add_point(&mut session.sketch, a, snap);
    let pb = get_or_add_point(&mut session.sketch, b, snap);
    let pmid = mid.map(|m| get_or_add_point(&mut session.sketch, m, snap));
    session.sketch.entities.push(SketchEntity::Slot {
        a: pa,
        b: pb,
        radius: r.max(0.01) as f64,
        construction: session.construction,
        mid: pmid,
    });
    session.dirty = true;
}

/// The polygon's rim/vertex point while dragging: the raw (unsnapped) cursor, snapped
/// only to a genuine sketch point within tolerance. This keeps the size tracking the
/// mouse — the broad rim/quadrant/reference snap cloud (whose tolerance grows with zoom)
/// no longer yanks the vertex to a far target — while still letting a vertex anchor on an
/// existing point.
fn poly_rim(session: &SketchSession) -> Option<Vec2> {
    let raw = session.cursor_raw_uv?;
    if let Some(i) = nearest_point(&session.sketch, raw, session.snap_dist) {
        if let Some(p) = session.sketch.points.get(i) {
            return Some(Vec2::new(p.x as f32, p.y as f32));
        }
    }
    Some(raw)
}

/// Regular polygon inscribed in (circumscribed by) a construction circle: `center` is the
/// middle, `rim` is one vertex (sets the radius and the orientation). Emits the centre
/// point, a dashed construction circle through the vertices, the N solid edges, and a
/// `PointOnCircle` per vertex so dimensioning the circle's radius resizes the whole shape.
fn commit_polygon(session: &mut SketchSession, center: Vec2, rim: Vec2) {
    let n = session.polygon_sides.max(3);
    let r = (rim - center).length().max(0.01);
    let theta0 = (rim - center).y.atan2((rim - center).x);
    // A dedicated (non-merged) centre point: `PointOnCircle` resolves a vertex's radius by
    // finding the circle on its centre point, so concentric polygons MUST NOT share one
    // centre point — else every vertex would read the first (largest) circle's radius and
    // collapse onto it. Each polygon gets its own centre + its own construction circle.
    let cp = session.sketch.add_point(center.x as f64, center.y as f64);
    session.sketch.add_construction_circle(cp, r as f64);
    let mut verts = Vec::with_capacity(n);
    for k in 0..n {
        let ang = theta0 + std::f32::consts::TAU * k as f32 / n as f32;
        let p = center + Vec2::new(ang.cos(), ang.sin()) * r;
        verts.push(session.sketch.add_point(p.x as f64, p.y as f64));
    }
    // The polygon edges are the profile, so they're always solid — only the circumscribed
    // circle is construction. (Don't inherit the sticky Line-tool construction toggle, or
    // the whole polygon becomes a guide and the profile reads as open.)
    for k in 0..n {
        let a = verts[k];
        let b = verts[(k + 1) % n];
        session.sketch.add_line(a, b, false);
    }
    for &v in &verts {
        session.sketch.constraints.push(Constraint::PointOnCircle { p: v, center: cp });
    }
}

/// Width of an arc slot = the cursor's distance from the arc centre line through a,p,b
/// (falls back to the perpendicular distance to the chord if the points are collinear).
fn arc_slot_width(uv: Vec2, a: Vec2, p: Vec2, b: Vec2) -> f32 {
    let d = 2.0 * (a.x * (p.y - b.y) + p.x * (b.y - a.y) + b.x * (a.y - p.y));
    if d.abs() < 1e-6 {
        return perp_dist(uv, a, b);
    }
    let (a2, p2, b2) = (a.length_squared(), p.length_squared(), b.length_squared());
    let cx = (a2 * (p.y - b.y) + p2 * (b.y - a.y) + b2 * (a.y - p.y)) / d;
    let cy = (a2 * (b.x - p.x) + p2 * (a.x - b.x) + b2 * (p.x - a.x)) / d;
    let c = Vec2::new(cx, cy);
    ((uv - c).length() - (a - c).length()).abs()
}

/// Perpendicular distance from `p` to the (infinite) line through `a` and `b`.
fn perp_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len = ab.length();
    if len < 1e-6 {
        return (p - a).length();
    }
    (ab.x * (p.y - a.y) - ab.y * (p.x - a.x)).abs() / len
}

/// Build an axis-aligned rectangle from two opposite corners (4 lines + H/V relations).
/// Returns the four corner point indices, CCW from `c0`.
fn rect_axis_aligned(s: &mut Sketch, c0: Vec2, c1: Vec2, construction: bool) -> [usize; 4] {
    let p0 = s.add_point(c0.x as f64, c0.y as f64);
    let p1 = s.add_point(c1.x as f64, c0.y as f64);
    let p2 = s.add_point(c1.x as f64, c1.y as f64);
    let p3 = s.add_point(c0.x as f64, c1.y as f64);
    s.add_line(p0, p1, construction);
    s.add_line(p1, p2, construction);
    s.add_line(p2, p3, construction);
    s.add_line(p3, p0, construction);
    s.constraints.push(Constraint::Horizontal(p0, p1));
    s.constraints.push(Constraint::Horizontal(p3, p2));
    s.constraints.push(Constraint::Vertical(p1, p2));
    s.constraints.push(Constraint::Vertical(p0, p3));
    [p0, p1, p2, p3]
}

/// Centre rectangle: an axis-aligned rectangle centred at `center` with `corner` as one
/// corner, plus the two diagonals as construction lines (an X) and a pinned centre point.
fn commit_center_rect(session: &mut SketchSession, center: Vec2, corner: Vec2) {
    let opposite = center * 2.0 - corner; // mirror of the corner about the centre
    let con = session.construction;
    let [p0, p1, p2, p3] = rect_axis_aligned(&mut session.sketch, opposite, corner, con);
    let s = &mut session.sketch;
    // X-pattern construction diagonals.
    s.add_line(p0, p2, true);
    s.add_line(p1, p3, true);
    // A centre point pinned to the diagonals' crossing (their shared midpoint).
    let cp = s.add_point(center.x as f64, center.y as f64);
    s.constraints.push(Constraint::Midpoint { mid: cp, a: p0, b: p2 });
}

/// Parallelogram from three points: side A→B is anchored, `c` pulls out the shape; the
/// fourth vertex is `a + (c − b)`. Opposite sides are kept parallel.
fn commit_parallelogram(s: &mut Sketch, a: Vec2, b: Vec2, c: Vec2, construction: bool) {
    let d = a + (c - b);
    let pa = s.add_point(a.x as f64, a.y as f64);
    let pb = s.add_point(b.x as f64, b.y as f64);
    let pc = s.add_point(c.x as f64, c.y as f64);
    let pd = s.add_point(d.x as f64, d.y as f64);
    s.add_line(pa, pb, construction);
    s.add_line(pb, pc, construction);
    s.add_line(pc, pd, construction);
    s.add_line(pd, pa, construction);
    s.constraints.push(Constraint::Parallel(pa, pb, pd, pc)); // AB ∥ DC
    s.constraints.push(Constraint::Parallel(pb, pc, pa, pd)); // BC ∥ AD
}

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut session: ResMut<SketchSession>,
    blocking: Res<UiBlocking>,
) {
    if session.plane.is_none() {
        return;
    }
    // A focused egui text field (e.g. the Text tool's string box) owns the keyboard —
    // don't let letter shortcuts (S/L/C/…) fire while the user is typing.
    if blocking.1 {
        return;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        session.tool = Tool::Select;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        session.tool = Tool::Line;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        session.tool = Tool::Circle;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        session.tool = Tool::Rectangle;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyM) {
        session.tool = Tool::Dimension;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyX) {
        session.construction = !session.construction;
    }
    // Enter finishes an in-progress (open) spline.
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        if session.tool == Tool::Spline && session.spline_pts.len() >= 2 {
            commit_spline(&mut session, false);
        }
    }
    if keys.just_pressed(KeyCode::KeyE) {
        session.op_request = Some(SolidOp::Boss(EXTRUDE_DISTANCE));
    }
    if keys.just_pressed(KeyCode::KeyD) {
        session.op_request = Some(SolidOp::Cut(EXTRUDE_DISTANCE));
    }
    // Delete (or Backspace) removes a selected dimension first, else selected entities.
    if keys.just_pressed(KeyCode::Delete) || keys.just_pressed(KeyCode::Backspace) {
        if let Some(ci) = session.selected_dim.take() {
            if ci < session.sketch.constraints.len() {
                session.sketch.constraints.remove(ci);
            }
            session.dim_edit = None;
            session.dirty = true;
            session.needs_apply = true;
        } else if !session.selected_entities.is_empty() {
            let mut idx = session.selected_entities.clone();
            idx.sort_unstable();
            idx.dedup();
            for &i in idx.iter().rev() {
                if i < session.sketch.entities.len() {
                    session.sketch.entities.remove(i);
                }
            }
            session.sketch.remove_unused_points(); // drop the now-orphan endpoints
            session.selected_entities.clear();
            session.hover_entity = None;
            session.selected_contours.clear(); // region indices shift after a delete
            session.dirty = true;
        }
    }
    if keys.just_pressed(KeyCode::Escape) {
        if !session.spline_pts.is_empty() {
            session.spline_pts.clear(); // 0) cancel the in-progress spline…
        } else if session.pending.is_some() || session.pending_b.is_some() || session.pending_c.is_some() {
            session.pending = None; // 1) cancel the in-progress entity…
            session.pending_b = None;
            session.pending_c = None;
            session.tool = Tool::Select;
        } else if session.tool != Tool::Select {
            session.tool = Tool::Select; // 2) …then drop back to the Select tool…
        } else {
            // 3) already on Select → commit the sketch and leave (handle_exit_sketch).
            session.exit_request = true;
        }
    }
}

/// Ctrl+Z = undo, Ctrl+Shift+Z / Ctrl+Y = redo.
fn history_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<UiState>,
    blocking: Res<UiBlocking>,
) {
    // Don't hijack Ctrl+Z / Ctrl+A etc. from a focused text field.
    if blocking.1 {
        return;
    }
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    if !ctrl {
        return;
    }
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    if keys.just_pressed(KeyCode::KeyZ) {
        if shift {
            ui_state.redo_request = true;
        } else {
            ui_state.undo_request = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyY) {
        ui_state.redo_request = true;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        ui_state.save_request = true;
    }
    if keys.just_pressed(KeyCode::KeyO) {
        ui_state.open_request = true;
    }
}

/// Save the document to a `.hcad` (RON) file, or open one — via a native dialog.
/// The solid isn't stored; opening triggers a regenerate to rebuild it.
fn handle_file_io(
    mut ui_state: ResMut<UiState>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
    mut session: ResMut<SketchSession>,
) {
    if ui_state.save_request {
        ui_state.save_request = false;
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("HCAD part", &["hcad", "ron"])
            .set_file_name("part.hcad")
            .save_file()
        {
            match ron::ser::to_string_pretty(&doc.0, ron::ser::PrettyConfig::default()) {
                Ok(text) => match std::fs::write(&path, text) {
                    Ok(()) => info!("Saved {}", path.display()),
                    Err(e) => warn!("Save failed: {e}"),
                },
                Err(e) => warn!("Serialize failed: {e}"),
            }
        }
    }

    if ui_state.open_request {
        ui_state.open_request = false;
        if let Some(path) = rfd::FileDialog::new().add_filter("HCAD part", &["hcad", "ron"]).pick_file() {
            match std::fs::read_to_string(&path) {
                Ok(text) => match ron::from_str::<Document>(&text) {
                    Ok(loaded) => {
                        doc.0 = loaded;
                        history.undo.clear();
                        history.redo.clear();
                        session.plane = None;
                        session.editing = None;
                        session.selected_contours.clear();
                        session.sketch.clear();
                        ui_state.selected = None;
                        ui_state.regen = true;
                        info!("Opened {}", path.display());
                    }
                    Err(e) => warn!("Could not parse {}: {e}", path.display()),
                },
                Err(e) => warn!("Could not read {}: {e}", path.display()),
            }
        }
    }
}

/// Apply a requested undo/redo. While a sketch is open, undo/redo steps through the
/// per-operation sketch history first (so it reverts the last line / dimension / drag, not
/// the whole sketch feature); only once that's exhausted does it fall back to the
/// document-level (feature timeline) history.
fn apply_history(
    mut history: ResMut<History>,
    mut doc: ResMut<DocRes>,
    mut ui_state: ResMut<UiState>,
    mut session: ResMut<SketchSession>,
) {
    if ui_state.undo_request {
        ui_state.undo_request = false;
        if session.plane.is_some() && !session.undo_sketch.is_empty() {
            let prev = session.undo_sketch.pop().unwrap();
            let cur = std::mem::replace(&mut session.sketch, prev);
            session.redo_sketch.push(cur);
            session.sketch.solve();
            session.dirty = true;
            session.undo_baseline = Some(session.sketch.clone());
            session.undo_fp = sketch_fingerprint(&session.sketch);
            session.selected_entities.clear();
            session.selected_dim = None;
        } else if let Some(prev) = history.undo.pop() {
            history.redo.push(doc.0.clone());
            doc.0 = prev;
            ui_state.regen = true;
            ui_state.selected = None;
        }
    }
    if ui_state.redo_request {
        ui_state.redo_request = false;
        if session.plane.is_some() && !session.redo_sketch.is_empty() {
            let next = session.redo_sketch.pop().unwrap();
            let cur = std::mem::replace(&mut session.sketch, next);
            session.undo_sketch.push(cur);
            session.sketch.solve();
            session.dirty = true;
            session.undo_baseline = Some(session.sketch.clone());
            session.undo_fp = sketch_fingerprint(&session.sketch);
            session.selected_entities.clear();
            session.selected_dim = None;
        } else if let Some(next) = history.redo.pop() {
            history.undo.push(doc.0.clone());
            doc.0 = next;
            ui_state.regen = true;
            ui_state.selected = None;
        }
    }
}

/// Turn the active sketch into a timeline feature, then request a regenerate.
/// The kernel work happens in `do_regenerate` by replaying the whole timeline —
/// so the document, not this op, is the source of truth.
fn do_solid_op(
    mut session: ResMut<SketchSession>,
    part: Res<Part>,
    mut doc: ResMut<DocRes>,
    mut ui_state: ResMut<UiState>,
    mut history: ResMut<History>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
    let Some(op) = session.op_request.take() else { return };
    let Some(ap) = session.plane.clone() else { return };

    let region_count = session.sketch.regions().len();
    if region_count == 0 {
        warn!("Need a closed profile (a loop of lines, or a circle) to extrude.");
        return;
    }
    // The chosen contours, or all of them if none were explicitly selected.
    let regions: Vec<usize> =
        session.selected_contours.iter().copied().filter(|&i| i < region_count).collect();

    // A body exists if there's geometry — either an exact B-rep solid *or* a mesh body
    // (after a fillet/seamless build, `part.solid` is None but `part.mesh` is the body).
    if matches!(op, SolidOp::Cut(_)) && part.solid.is_none() && part.mesh.is_none() {
        warn!("Cut: there is no body yet — extrude a boss first.");
        return;
    }

    history.snapshot(&doc.0);
    let sketch = session.sketch.clone();
    let plane = plane_ref(&ap);
    let kind = match op {
        SolidOp::Boss(d) => FeatureKind::Extrude { sketch, regions, plane, distance: d },
        SolidOp::Cut(d) => FeatureKind::Cut { sketch, regions, plane, distance: d },
    };
    // Editing an existing feature replaces it in place; otherwise append.
    let target = match session.editing {
        Some(i) if i < doc.0.features.len() => {
            doc.0.features[i].kind = kind;
            i
        }
        _ => {
            doc.0.add_feature(kind);
            doc.0.features.len() - 1
        }
    };
    ui_state.regen = true;
    ui_state.selected = Some(target);
    doc.0.rollback = doc.0.features.len(); // roll forward so the result is visible

    session.plane = None;
    session.editing = None;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
    session.selected_contours.clear();
    if let Ok((mut tf, orbit)) = cam_q.single_mut() {
        *tf = camera_transform(orbit);
    }
}

/// (Re)open a feature's sketch for editing — loads it into the session and aligns
/// the camera to its plane.
fn handle_edit_sketch(
    mut ui_state: ResMut<UiState>,
    mut session: ResMut<SketchSession>,
    mut doc: ResMut<DocRes>,
    mut cam_q: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    let Some(i) = ui_state.edit_sketch_request.take() else { return };
    let Some(f) = doc.0.features.get(i) else { return };
    let (sketch, plane, contours) = match &f.kind {
        FeatureKind::Sketch { sketch, plane } => (sketch.clone(), plane.clone(), Vec::new()),
        FeatureKind::Extrude { sketch, plane, regions, .. }
        | FeatureKind::Cut { sketch, plane, regions, .. } => {
            (sketch.clone(), plane.clone(), regions.clone())
        }
        FeatureKind::Plane(_) | FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } => return,
    };
    let ap = active_plane_from_ref(&plane, "Face");
    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
        orbit.radius = orbit.radius.max(6.0);
        look_along(&mut orbit, ap.origin, ap.n);
        *tf = camera_transform(&orbit);
    }
    session.sketch = sketch;
    session.plane = Some(ap);
    session.editing = Some(i);
    session.selected_contours = contours;
    session.selected_entities.clear();
    session.pending = None;
    session.dim_first = None;
    session.drag = None;
    session.cursor_uv = None;
    session.needs_apply = false;
    session.undo_sketch.clear();
    session.redo_sketch.clear();
    session.undo_baseline = Some(session.sketch.clone());
    session.undo_fp = sketch_fingerprint(&session.sketch);
    // Roll back to just before this feature so you sketch on its input geometry
    // (the body without this feature); committing rolls forward again.
    doc.0.rollback = i;
    ui_state.regen = true;
    info!("Editing sketch of feature {i}.");
}

/// Leave sketch mode, committing the sketch to the timeline: update the feature
/// being edited, or add a standalone Sketch feature so it can be returned to.
fn handle_exit_sketch(
    mut session: ResMut<SketchSession>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
    mut ui_state: ResMut<UiState>,
) {
    // Cancel: leave without committing any changes.
    if session.cancel_request {
        session.cancel_request = false;
        session.plane = None;
        session.editing = None;
        session.pending = None;
        session.drag = None;
        session.cursor_uv = None;
        session.selected_contours.clear();
        // Restore the rollback bar (edit rolled it back to the feature).
        doc.0.rollback = doc.0.features.len();
        ui_state.regen = true;
        return;
    }
    if !session.exit_request {
        return;
    }
    session.exit_request = false;
    let Some(ap) = session.plane.clone() else { return };

    // Apply any staged dimension/relation edits before committing the sketch.
    if session.needs_apply {
        session.sketch.solve();
        session.needs_apply = false;
    }

    match session.editing {
        Some(i) if i < doc.0.features.len() => {
            history.snapshot(&doc.0);
            let new_sketch = session.sketch.clone();
            let contours = session.selected_contours.clone();
            match &mut doc.0.features[i].kind {
                FeatureKind::Sketch { sketch, .. } => *sketch = new_sketch,
                FeatureKind::Extrude { sketch, regions: r, .. }
                | FeatureKind::Cut { sketch, regions: r, .. } => {
                    *sketch = new_sketch;
                    *r = contours;
                    ui_state.regen = true;
                }
                FeatureKind::Plane(_) | FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } => {}
            }
            ui_state.selected = Some(i);
        }
        _ => {
            // A brand-new sketch with geometry becomes a standalone Sketch feature.
            if !session.sketch.entities.is_empty() {
                history.snapshot(&doc.0);
                let sketch = session.sketch.clone();
                doc.0.add_feature(FeatureKind::Sketch { sketch, plane: plane_ref(&ap) });
                ui_state.selected = Some(doc.0.features.len() - 1);
            }
        }
    }
    // Roll the bar forward so the committed result is shown (edit rolled it back).
    doc.0.rollback = doc.0.features.len();
    ui_state.regen = true;

    session.plane = None;
    session.editing = None;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
    session.selected_contours.clear();
}

/// Replay the feature timeline (up to the rollback bar) into a solid. This is the
/// heart of M6: editing any feature and re-running this rebuilds everything
/// downstream. Faces are referenced geometrically via each sketch's `PlaneRef`.
/// Rebuild the solid from the timeline (test-facing; discards failure reports).
fn regenerate(doc: &Document) -> Option<KSolid> {
    regenerate_reported(doc).0
}

/// Like [`regenerate`] but also returns a human-readable message per feature that
/// True if any extrude/cut feature's sketch contains outlined text. Such models are
/// built with the mesh kernel — truck's exact booleans choke (and can stack-overflow)
/// on the hundreds of glyph faces a text profile produces.
fn doc_has_text(doc: &Document) -> bool {
    doc.features.iter().any(|f| {
        let sketch = match &f.kind {
            FeatureKind::Extrude { sketch, .. } | FeatureKind::Cut { sketch, .. } => sketch,
            _ => return false,
        };
        sketch.entities.iter().any(|e| matches!(e, SketchEntity::Text { .. }))
    })
}

/// True if the model has a fillet feature — those are mesh-only (truck can't fillet).
fn doc_has_fillet(doc: &Document) -> bool {
    doc.features
        .iter()
        .any(|f| matches!(f.kind, FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. }))
}

/// failed to build, so the UI can tell the user which operation didn't apply.
fn regenerate_reported(doc: &Document) -> (Option<KSolid>, Vec<String>) {
    let mut failures: Vec<String> = Vec::new();
    let mut body: Option<KSolid> = None;
    let end = doc.rollback.min(doc.features.len());
    for feature in &doc.features[..end] {
        match &feature.kind {
            FeatureKind::Plane(_) => {}
            FeatureKind::Sketch { .. } => {} // 2D only — no solid contribution
            FeatureKind::Extrude { sketch, regions, plane, distance } => {
                let all = sketch.regions();
                // A feature built on a face rides on that face: re-resolve its plane
                // to the current body so stacked features build on each other and
                // shift when an upstream feature is edited.
                let resolved = match &body { Some(b) => reproject_plane(plane, b), None => plane.clone() };
                let basis = basis_from_ref(&resolved);
                // Merge adjacent contours into single profiles first (a dumbbell of
                // two circles + connecting band becomes one outline), so each piece
                // extrudes as one solid without a coincident-face boolean.
                let merged = merge_regions(&chosen_regions(&all, regions));
                for r in &merged {
                    let next = match &body {
                        Some(b) => boss_union(b, r, &basis, *distance),
                        None => extrude_solid(&r.outer, &r.holes, &basis, *distance),
                    };
                    if let Some(s) = next {
                        body = Some(s);
                    } else {
                        warn!(
                            "Regen: an extrude contour could not be built. outer[{}]  holes={}  base={}",
                            loop_diag(&r.outer),
                            r.holes.len(),
                            body.is_none()
                        );
                        failures.push("Extrude failed — the kernel could not union this boss (try moving/resizing it, or build it on a clean face).".into());
                    }
                }
            }
            FeatureKind::Cut { sketch, regions, plane, distance } => {
                let Some(b0) = &body else { continue };
                let all = sketch.regions();
                let resolved = reproject_plane(plane, b0);
                let basis = basis_from_ref(&resolved);
                let origin = Vec3::new(resolved.origin[0] as f32, resolved.origin[1] as f32, resolved.origin[2] as f32);
                let n = Vec3::new(resolved.normal[0] as f32, resolved.normal[1] as f32, resolved.normal[2] as f32);
                // Merge adjacent contours into single profiles first (same reason as
                // the boss), then cut each from the current body.
                let merged = merge_regions(&chosen_regions(&all, regions));
                for r in &merged {
                    let Some(b) = &body else { break };
                    // Pick the cut direction from the *current* body, so it stays
                    // correct even after upstream edits move things around.
                    let centroid = mesh_centroid(&tessellate(b, 0.06).mesh);
                    let signed = if (centroid - origin).dot(n) < 0.0 { -*distance } else { *distance };
                    if let Some(s) = cut_op(b, r, &basis, signed) {
                        body = Some(s);
                    } else {
                        warn!("Regen: a cut contour could not be built.");
                        failures.push("Cut failed — the kernel rejected this cut (often a self-touching profile or a coincident wall; try nudging the sketch).".into());
                    }
                }
            }
            // A fillet/chamfer/mirror reshapes the mesh, so a model with one always builds
            // via the mesh path — this exact-kernel path never runs with one present.
            FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } => {
                failures.push("Fillet/chamfer/mirror needs the mesh kernel.".into());
            }
        }
    }
    (body, failures)
}

/// Robust mesh rebuild: replay the whole timeline using the mesh-boolean kernel
/// instead of truck's exact B-rep booleans. truck still does the (reliable) single
/// prism extrude + tessellation for each tool; only the *booleans* are mesh-based, so
/// coincident/coplanar cases (a boss on a cut floor) that the exact kernel rejects
/// still build. Returns the final triangle mesh, or `None` if there's no solid.
///
/// Note: this fallback uses each feature's stored plane (no re-projection against the
/// live body), which is correct for a straight build-up; an *edited* upstream feature
/// may not shift downstream geometry here the way the exact path does.
fn regenerate_mesh(doc: &Document) -> Option<TriMesh> {
    let mut body: Option<TriMesh> = None;
    let end = doc.rollback.min(doc.features.len());
    for feature in &doc.features[..end] {
        match &feature.kind {
            FeatureKind::Plane(_) | FeatureKind::Sketch { .. } => {}
            FeatureKind::Extrude { sketch, regions, plane, distance } => {
                let all = sketch.regions();
                let basis = basis_from_ref(plane);
                for r in &merge_regions(&chosen_regions(&all, regions)) {
                    body = match body.take() {
                        // Boss: dip the prism *substantially* into the body so the join ring
                        // is buried in continuous material — the surface wall then runs
                        // smoothly through it (no seam). A tiny dip leaves sliver triangles
                        // at the join; an exactly-flush union only touches and leaves a full
                        // ring of edges. The dip is bounded by how deep the body sits below
                        // the boss's base plane, so it can't poke out the far side.
                        Some(b) => {
                            let (blo, bhi) = mesh_bbox(&b);
                            let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                            let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
                            // Depth of the body behind the base plane (along −normal).
                            let corners = [
                                Vec3::new(blo.x, blo.y, blo.z), Vec3::new(bhi.x, bhi.y, bhi.z),
                                Vec3::new(blo.x, blo.y, bhi.z), Vec3::new(bhi.x, bhi.y, blo.z),
                                Vec3::new(blo.x, bhi.y, blo.z), Vec3::new(bhi.x, blo.y, bhi.z),
                                Vec3::new(blo.x, bhi.y, bhi.z), Vec3::new(bhi.x, blo.y, blo.z),
                            ];
                            let behind = corners.iter().map(|c| (o - *c).dot(n)).fold(0.0_f32, f32::max);
                            // ~1mm is plenty to bury the join in well-formed triangles; never
                            // dip past the body's far side.
                            let overlap = ((behind * 0.5).min(1.0).max(0.05)).min(behind.max(0.0)) as f64;
                            Some(
                                extrude_tool_mesh(&r.outer, &r.holes, &basis, -overlap, distance + overlap)
                                    .map(|tool| mesh_union(&b, &tool))
                                    .unwrap_or(b),
                            )
                        }
                        // First feature: the prism itself is the body.
                        None => extrude_tool_mesh(&r.outer, &r.holes, &basis, 0.0, *distance),
                    };
                }
            }
            FeatureKind::Cut { sketch, regions, plane, distance } => {
                if body.is_none() {
                    continue;
                }
                let all = sketch.regions();
                let basis = basis_from_ref(plane);
                let origin = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
                let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                for r in &merge_regions(&chosen_regions(&all, regions)) {
                    let Some(cur) = body.take() else { break };
                    // Cut direction from the current body, mirroring the exact path.
                    let signed = if (mesh_centroid(&cur) - origin).dot(n) < 0.0 { -*distance } else { *distance };
                    body = Some(match cut_tool_mesh(&r.outer, &r.holes, &basis, signed) {
                        Some(tool) => mesh_difference(&cur, &tool),
                        None => cur,
                    });
                }
            }
            // Round the body's (picked, or all) edges by the fillet radius.
            FeatureKind::Fillet { radius, edges } => {
                if let Some(b) = body.take() {
                    body = Some(round_mesh(&b, *radius, edges).unwrap_or(b));
                }
            }
            // Flat-bevel the picked edges by the chamfer distance.
            FeatureKind::Chamfer { distance, edges } => {
                if let Some(b) = body.take() {
                    body = Some(chamfer_mesh(&b, *distance, edges).unwrap_or(b));
                }
            }
            // Reflect the body across the plane and union it with the original.
            FeatureKind::Mirror { plane } => {
                if let Some(b) = body.take() {
                    let refl = mirror_mesh(&b, plane.origin, plane.normal);
                    body = Some(mesh_union(&b, &refl));
                }
            }
        }
    }
    body
}

/// Consume a regenerate request: rebuild the solid from the timeline and refresh
/// the rendered meshes.
fn do_regenerate(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ui_state: ResMut<UiState>,
    mut part: ResMut<Part>,
    mut edge_sel: ResMut<EdgeSelection>,
    doc: Res<DocRes>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if !ui_state.regen {
        return;
    }
    ui_state.regen = false;
    // Vertices move when the model rebuilds, so any edge selection is stale.
    edge_sel.clear();

    // Text produces hundreds of tiny glyph faces; truck's recursive B-rep booleans can
    // *stack-overflow* on that (a hard abort `catch_unwind` can't trap), so any model
    // containing text is built with the robust mesh kernel from the start.
    let force_mesh = ui_state.seamless || doc_has_text(&doc.0) || doc_has_fillet(&doc.0);

    // Seamless mode: build the whole model with the mesh kernel (Manifold), which fuses
    // coincident/coplanar faces so adjacent features merge without a seam. The exact
    // path's seams come from truck not merging shared faces; mesh has no such limit.
    if force_mesh {
        let mesh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc.0)))
            .unwrap_or(None);
        for e in &existing {
            commands.entity(e).despawn();
        }
        match mesh {
            Some(m) if !m.positions.is_empty() => {
                let tess = mesh_tessellation(m);
                part.mesh = Some(tess.mesh.clone());
                part.edges = tess.edges.clone();
                part.tangent_edges = tess.tangent_edges.clone();
                spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                part.solid = None; // mesh body has no B-rep handle
                ui_state.last_error = None;
            }
            _ => {
                part.solid = None;
                part.mesh = None;
                part.edges.clear();
                part.tangent_edges.clear();
            }
        }
        return;
    }

    // The whole rebuild runs under a panic guard: a kernel panic deep in a boolean
    // or triangulation leaves the model empty rather than taking the app down.
    let (rebuilt, failures) =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_reported(&doc.0)))
            .unwrap_or_else(|_| {
                warn!("Regenerate panicked — the kernel choked on this geometry; model cleared.");
                (None, vec!["Rebuild crashed in the geometry kernel — the model was cleared. Undo (Ctrl+Z) to recover.".to_string()])
            });

    for e in &existing {
        commands.entity(e).despawn();
    }

    // Clean exact rebuild → use truck's B-rep (exact faces, best quality).
    if failures.is_empty() {
        ui_state.last_error = None;
        match rebuilt {
            Some(solid) => {
                let tess = tessellate(&solid, 0.03);
                part.mesh = Some(tess.mesh.clone());
                part.edges = tess.edges.clone();
                part.tangent_edges = tess.tangent_edges.clone();
                spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                part.solid = Some(solid);
            }
            None => {
                part.solid = None;
                part.mesh = None;
                part.edges.clear();
                part.tangent_edges.clear();
            }
        }
        return;
    }

    // The exact kernel stumbled on a boolean — rebuild the whole model with the robust
    // mesh kernel so the operation still applies (triangulated faces for the result).
    let mesh_body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc.0)))
        .unwrap_or(None);
    match mesh_body {
        Some(mesh) if !mesh.positions.is_empty() => {
            let tess = mesh_tessellation(mesh);
            part.mesh = Some(tess.mesh.clone());
            part.edges = tess.edges.clone();
            part.tangent_edges = tess.tangent_edges.clone();
            spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
            // Keep a B-rep handle (truck's partial body) so "a body exists" logic still
            // holds; the displayed geometry is the full mesh-kernel result.
            part.solid = rebuilt;
            // The mesh kernel succeeded — the geometry built fine, so this isn't an error
            // (only a true failure to generate geometry warrants the banner).
            ui_state.last_error = None;
            info!("Regenerate: used the mesh-boolean fallback for {} operation(s).", failures.len());
        }
        // Even the mesh kernel couldn't do it — keep whatever the exact path produced
        // and report the original failure(s).
        _ => {
            ui_state.last_error = match failures.len() {
                1 => Some(failures.into_iter().next().unwrap()),
                n => Some(format!("{n} operations failed to build. Most recent: {}", failures.last().unwrap())),
            };
            match rebuilt {
                Some(solid) => {
                    let tess = tessellate(&solid, 0.03);
                    part.mesh = Some(tess.mesh.clone());
                    part.edges = tess.edges.clone();
                    part.tangent_edges = tess.tangent_edges.clone();
                    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                    part.solid = Some(solid);
                }
                None => {
                    part.solid = None;
                    part.mesh = None;
                    part.edges.clear();
                    part.tangent_edges.clear();
                }
            }
        }
    }
}

/// Append a confirmed fillet to the timeline and trigger a rebuild.
fn apply_fillet(
    mut ui_state: ResMut<UiState>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
) {
    let Some(radius) = ui_state.fillet_request.take() else { return };
    let edges = std::mem::take(&mut ui_state.fillet_edges);
    history.snapshot(&doc.0);
    doc.0.add_feature(FeatureKind::Fillet { radius, edges });
    doc.0.rollback = doc.0.features.len();
    ui_state.selected = Some(doc.0.features.len() - 1);
    ui_state.regen = true;
}

/// While the Fillet PropertyManager is open, show a live rounded preview of the current
/// body — recomputed only when the radius changes (rounding is expensive).
fn fillet_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ui_state: ResMut<UiState>,
    part: Res<Part>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if ui_state.regen {
        return; // a full rebuild this frame supersedes the preview
    }
    let Some(r) = ui_state.pending_fillet else { return };
    if ui_state.fillet_shown == Some(r) {
        return; // already showing this radius
    }
    let Some(base) = part.mesh.clone() else {
        ui_state.fillet_shown = Some(r);
        return;
    };
    let edges = ui_state.fillet_edges.clone();
    let rounded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| round_mesh(&base, r as f64, &edges)))
        .ok()
        .flatten()
        .unwrap_or(base);
    let tess = mesh_tessellation(rounded);
    for e in &existing {
        commands.entity(e).despawn();
    }
    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
    ui_state.fillet_shown = Some(r);
}

/// Append a confirmed chamfer to the timeline and trigger a rebuild.
fn apply_chamfer(
    mut ui_state: ResMut<UiState>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
) {
    let Some(distance) = ui_state.chamfer_request.take() else { return };
    let edges = std::mem::take(&mut ui_state.fillet_edges);
    history.snapshot(&doc.0);
    doc.0.add_feature(FeatureKind::Chamfer { distance, edges });
    doc.0.rollback = doc.0.features.len();
    ui_state.selected = Some(doc.0.features.len() - 1);
    ui_state.regen = true;
}

/// While the Chamfer PropertyManager is open, show a live beveled preview of the current
/// body — recomputed only when the distance changes (the boolean is expensive).
fn chamfer_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ui_state: ResMut<UiState>,
    part: Res<Part>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if ui_state.regen {
        return; // a full rebuild this frame supersedes the preview
    }
    let Some(d) = ui_state.pending_chamfer else { return };
    if ui_state.chamfer_shown == Some(d) {
        return; // already showing this distance
    }
    let Some(base) = part.mesh.clone() else {
        ui_state.chamfer_shown = Some(d);
        return;
    };
    let edges = ui_state.fillet_edges.clone();
    let beveled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| chamfer_mesh(&base, d as f64, &edges)))
        .ok()
        .flatten()
        .unwrap_or(base);
    let tess = mesh_tessellation(beveled);
    for e in &existing {
        commands.entity(e).despawn();
    }
    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
    ui_state.chamfer_shown = Some(d);
}

/// Append a confirmed mirror to the timeline and trigger a rebuild.
fn apply_mirror_feature(
    mut ui_state: ResMut<UiState>,
    mut doc: ResMut<DocRes>,
    mut history: ResMut<History>,
) {
    let Some(which) = ui_state.mirror_request.take() else { return };
    history.snapshot(&doc.0);
    doc.0.add_feature(FeatureKind::Mirror { plane: standard_plane_ref(which) });
    doc.0.rollback = doc.0.features.len();
    ui_state.selected = Some(doc.0.features.len() - 1);
    ui_state.regen = true;
}

/// While the Mirror PropertyManager is open, show a live preview of the body unioned with
/// its reflection across the chosen plane — recomputed only when the plane changes.
fn mirror_preview(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ui_state: ResMut<UiState>,
    part: Res<Part>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if ui_state.regen {
        return;
    }
    let Some(which) = ui_state.pending_mirror else { return };
    if ui_state.mirror_shown == Some(which) {
        return;
    }
    let Some(base) = part.mesh.clone() else {
        ui_state.mirror_shown = Some(which);
        return;
    };
    let plane = standard_plane_ref(which);
    let mirrored = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let refl = mirror_mesh(&base, plane.origin, plane.normal);
        mesh_union(&base, &refl)
    }))
    .unwrap_or_else(|_| base.clone());
    let tess = mesh_tessellation(mirrored);
    for e in &existing {
        commands.entity(e).despawn();
    }
    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
    ui_state.mirror_shown = Some(which);
}

/// `PlaneBasis` (kernel-side) from a stored `PlaneRef`.
fn basis_from_ref(p: &PlaneRef) -> PlaneBasis {
    PlaneBasis { origin: p.origin, u: p.u, v: p.v, normal: p.normal }
}

/// Re-resolve a face-built feature's plane against the current body: keep its
/// in-plane location and axes, but slide the origin along the normal onto the body's
/// extreme face in that direction (the "top" the sketch sits on). This is what lets
/// stacked features build on each other — when an upstream feature's height changes,
/// downstream features ride up/down with the face instead of being left behind.
///
/// Limitation: it targets the *outermost* coplanar face along the normal, so a
/// feature sketched on a recessed/stepped face of the same orientation can resolve
/// to the wrong one. Robust topological naming (DESIGN.md §4.3) is the eventual fix.
fn reproject_plane(plane: &PlaneRef, body: &KSolid) -> PlaneRef {
    let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
    let u = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
    let v = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
    let mesh = tessellate(body, 0.2).mesh;
    if mesh.indices.len() < 3 {
        return plane.clone();
    }
    let o_n = o.dot(n);
    // The origin projects to (0,0) in the plane's (u,v); find the body face parallel to the
    // sketch plane that lies *under the sketch's footprint* (contains the origin in-plane)
    // and snap the origin onto it along the normal — the face the sketch actually sits on,
    // even after an upstream edit moved it. (The old global-extreme rule sent an angled
    // face's origin off to a far corner.)
    let in_tri = |p: Vec2, a: Vec2, b: Vec2, c: Vec2| {
        let s = |p: Vec2, a: Vec2, b: Vec2| (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        let (d1, d2, d3) = (s(p, a, b), s(p, b, c), s(p, c, a));
        let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(neg && pos)
    };
    let to2d = |p: Vec3| Vec2::new((p - o).dot(u), (p - o).dot(v));
    let mut best: Option<f32> = None;
    for t in mesh.indices.chunks(3) {
        let p = |i: u32| Vec3::from_array(mesh.positions[i as usize]);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let tn = (b - a).cross(c - a).normalize_or_zero();
        if tn.dot(n).abs() < 0.9 {
            continue; // not a face parallel to the sketch plane
        }
        if !in_tri(Vec2::ZERO, to2d(a), to2d(b), to2d(c)) {
            continue; // sketch origin isn't over this face
        }
        let off = a.dot(n);
        if best.map_or(true, |bo| (off - o_n).abs() < (bo - o_n).abs()) {
            best = Some(off); // the parallel face nearest the original origin along n
        }
    }
    match best {
        Some(off) => {
            let shifted = o + n * (off - o_n);
            PlaneRef { origin: [shifted.x as f64, shifted.y as f64, shifted.z as f64], ..plane.clone() }
        }
        None => plane.clone(), // no face under the sketch → leave the plane where it is
    }
}

/// Resolve the selected-contour indices against a sketch's regions. An empty
/// selection means "all regions"; out-of-range indices are skipped (the sketch
/// may have changed since the feature was created).
fn chosen_regions<'a>(all: &'a [hworks_sketch::Region], selected: &[usize]) -> Vec<&'a hworks_sketch::Region> {
    if selected.is_empty() {
        all.iter().collect()
    } else {
        selected.iter().filter_map(|&i| all.get(i)).collect()
    }
}

/// A boss/cut whose face exactly coincides with a body face (e.g. a circle drawn to
/// match an existing arc) defeats truck's boolean. We nudge the profile outward by
/// this much — ~1 micron — so the faces register as distinct. It is far below the
/// tessellation tolerance and any manufacturing precision, so the part stays exact
/// for all practical purposes (and the sketch keeps your exact typed dimension).
const COINCIDENT_NUDGE: f64 = 1.0e-3;
const COINCIDENT_TOL: f64 = 1.0e-4;
/// A robust fallback tolerance/overlap for when the tight, flush settings fail —
/// truck is more forgiving here at the cost of a (still tiny) visible lip.
const ROBUST_TOL: f64 = 0.05;
const ROBUST_OVERLAP: f64 = 0.1;

/// Union a set of sketch regions in 2D into merged outline(s) by cancelling the
/// edges shared between adjacent regions. Adjacent contours collapse into a single
/// profile — so the extrude is one solid and never needs the fragile coincident-
/// face 3D boolean — while disjoint contours stay separate. Falls back to the
/// inputs if a clean merge can't be traced.
fn merge_regions(regions: &[&hworks_sketch::Region]) -> Vec<hworks_sketch::Region> {
    use std::collections::{HashMap, HashSet};
    let key = |p: [f64; 2]| ((p[0] * 1.0e6).round() as i64, (p[1] * 1.0e6).round() as i64);
    let mut ids: HashMap<(i64, i64), usize> = HashMap::new();
    let mut pos: Vec<[f64; 2]> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for r in regions {
        for loop_pts in std::iter::once(&r.outer).chain(r.holes.iter()) {
            let m = loop_pts.len();
            if m < 3 {
                continue;
            }
            let vids: Vec<usize> = loop_pts
                .iter()
                .map(|p| *ids.entry(key(*p)).or_insert_with(|| {
                    pos.push(*p);
                    pos.len() - 1
                }))
                .collect();
            for k in 0..m {
                edges.push((vids[k], vids[(k + 1) % m]));
            }
        }
    }
    // A directed edge whose reverse also appears is internal (between two selected
    // faces) → drop it. The survivors are the union boundary.
    let present: HashSet<(usize, usize)> = edges.iter().copied().collect();
    let mut next: HashMap<usize, usize> = HashMap::new();
    for &(a, b) in &edges {
        if !present.contains(&(b, a)) {
            next.insert(a, b);
        }
    }
    if next.is_empty() {
        return regions.iter().map(|r| (*r).clone()).collect();
    }
    // Trace the boundary edges into loops (assumes degree-2 boundary vertices).
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    let starts: Vec<usize> = next.keys().copied().collect();
    for s in starts {
        if used.contains(&s) {
            continue;
        }
        let mut loop_pts = Vec::new();
        let mut cur = s;
        loop {
            used.insert(cur);
            loop_pts.push(pos[cur]);
            match next.get(&cur) {
                Some(&nx) if nx != s && !used.contains(&nx) => cur = nx,
                _ => break,
            }
            if loop_pts.len() > pos.len() + 1 {
                break;
            }
        }
        if loop_pts.len() >= 3 {
            loops.push(loop_pts);
        }
    }
    if loops.is_empty() {
        return regions.iter().map(|r| (*r).clone()).collect();
    }
    nest_loops(loops)
}

/// Classify a set of closed loops into regions (outer + holes) by even/odd
/// containment — the same nesting rule the sketcher uses.
fn nest_loops(loops: Vec<Vec<[f64; 2]>>) -> Vec<hworks_sketch::Region> {
    let n = loops.len();
    let area = |poly: &[[f64; 2]]| {
        let m = poly.len();
        let mut a = 0.0;
        for i in 0..m {
            let (p, q) = (poly[i], poly[(i + 1) % m]);
            a += p[0] * q[1] - q[0] * p[1];
        }
        (a * 0.5).abs()
    };
    let centroid = |poly: &[[f64; 2]]| {
        let m = poly.len() as f64;
        let (mut x, mut y) = (0.0, 0.0);
        for p in poly {
            x += p[0];
            y += p[1];
        }
        [x / m, y / m]
    };
    let areas: Vec<f64> = loops.iter().map(|l| area(l)).collect();
    let contains = |j: usize, i: usize| {
        j != i && areas[j] > areas[i] && point_in_poly(centroid(&loops[i]), &loops[j])
    };
    let depth: Vec<usize> = (0..n).map(|i| (0..n).filter(|&j| contains(j, i)).count()).collect();
    let mut out = Vec::new();
    for i in 0..n {
        if depth[i] % 2 != 0 {
            continue;
        }
        let holes = (0..n)
            .filter(|&k| depth[k] == depth[i] + 1 && contains(i, k))
            .map(|k| loops[k].clone())
            .collect();
        out.push(hworks_sketch::Region { outer: loops[i].clone(), holes });
    }
    out
}

/// Add a boss (region `r`) to an existing body, trying progressively more robust
/// strategies so a boolean never simply fails: flush+exact, flush+nudge (coincident
/// faces), then the robust overlap/tolerance with and without the nudge. The first
/// (cleanest) one that works wins.
fn boss_union(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    // (radial nudge, overlap, tolerance). The later entries escalate: bigger radial
    // nudges break a boss that is *coincident/concentric* with an existing curved wall
    // (truck's union rejects coincident faces), and bigger overlaps/tolerances absorb
    // awkward planar coincidences. A radial nudge only dodges a coincident wall if the
    // tolerance is *smaller* than the nudge, so every nudged strategy keeps tol ≪ nudge.
    // (These nudges let truck succeed, but on a coincident wall they leave a hairline
    // seam — turn on "Seamless" to rebuild via the mesh kernel and fuse those instead.)
    let strategies = [
        (0.0, BOSS_OVERLAP, COINCIDENT_TOL),
        (COINCIDENT_NUDGE, BOSS_OVERLAP, COINCIDENT_TOL),
        (0.0, ROBUST_OVERLAP, ROBUST_TOL),
        (0.01, ROBUST_OVERLAP, COINCIDENT_TOL),
        (0.05, ROBUST_OVERLAP, COINCIDENT_TOL),
        (0.2, ROBUST_OVERLAP, 1.0e-3),
        (0.5, 0.5, 1.0e-3),
    ];
    let mut extruded_ok = false;
    for (k, &(nudge, overlap, tol)) in strategies.iter().enumerate() {
        let outer = if nudge > 0.0 { inflate_loop(&r.outer, nudge) } else { r.outer.clone() };
        let Some(boss) = extrude_solid_with_overlap(&outer, &r.holes, basis, distance, overlap) else {
            continue;
        };
        extruded_ok = true;
        // Try both operand orders — truck's `or` is order-sensitive on awkward faces.
        if let Some(s) = union_tol(body, &boss, tol).or_else(|| union_tol(&boss, body, tol)) {
            if k > 0 {
                info!("Boss union: used fallback strategy {k} (coincident/awkward faces).");
            }
            return Some(s);
        }
    }
    warn!(
        "Boss could not be built by any strategy — truck rejected this geometry. extrude_ok={extruded_ok}  outer[{}]  holes={}  dist={distance:.3}",
        loop_diag(&r.outer),
        r.holes.len()
    );
    for (i, h) in r.holes.iter().enumerate() {
        warn!("  hole {i}: {}", loop_diag(h));
    }
    // Spatial diagnosis: where does the boss sit relative to the body?
    if let Some(boss) = extrude_solid_with_overlap(&r.outer, &r.holes, basis, distance, BOSS_OVERLAP) {
        let (blo, bhi) = mesh_bbox(&tessellate(&boss, 0.1).mesh);
        let (dlo, dhi) = mesh_bbox(&tessellate(body, 0.1).mesh);
        let overlaps = blo.cmplt(dhi).all() && dlo.cmplt(bhi).all();
        warn!(
            "  boss bbox [{blo:.2?}..{bhi:.2?}]  body bbox [{dlo:.2?}..{dhi:.2?}]  bbox_overlap={overlaps}"
        );
    }
    None
}

/// Cut region `r` from the body — same escalating fallback idea as [`boss_union`].
fn cut_op(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    // (radial nudge, boolean tolerance) — truck's cut boolean is flaky, so sweep a
    // range of tolerances/nudges; the first that produces a solid wins.
    let strategies = [
        (0.0, COINCIDENT_TOL),
        (COINCIDENT_NUDGE, COINCIDENT_TOL),
        (0.0, 1.0e-3),
        (COINCIDENT_NUDGE, 1.0e-3),
        (0.0, 1.0e-2),
        (0.0, ROBUST_TOL),
        (COINCIDENT_NUDGE, ROBUST_TOL),
    ];
    for (k, &(nudge, tol)) in strategies.iter().enumerate() {
        let outer = if nudge > 0.0 { inflate_loop(&r.outer, nudge) } else { r.outer.clone() };
        if let Some(s) = cut_tol(body, &outer, &r.holes, basis, distance, tol) {
            if k > 0 {
                info!("Cut: used fallback strategy {k} (truck's boolean is finicky here).");
            }
            return Some(s);
        }
    }
    warn!(
        "Cut could not be built by any strategy — truck rejected this geometry. outer[{}]  holes={}",
        loop_diag(&r.outer),
        r.holes.len()
    );
    for (i, h) in r.holes.iter().enumerate() {
        warn!("  hole {i}: {}", loop_diag(h));
    }
    None
}

/// A one-line health summary of a profile loop, for diagnosing kernel cut/boss
/// failures: vertex count, signed-ish area, shortest edge, and whether it self-crosses.
fn loop_diag(loop_pts: &[[f64; 2]]) -> String {
    let n = loop_pts.len();
    if n < 3 {
        return format!("pts={n} (degenerate)");
    }
    let mut a = 0.0;
    let mut min_edge = f64::INFINITY;
    for i in 0..n {
        let (p, q) = (loop_pts[i], loop_pts[(i + 1) % n]);
        a += p[0] * q[1] - q[0] * p[1];
        min_edge = min_edge.min(((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt());
    }
    let a = (a * 0.5).abs();
    // Any pair of non-adjacent edges crossing → a self-intersecting (invalid) profile.
    let mut self_int = false;
    'scan: for i in 0..n {
        let (a1, a2) = (loop_pts[i], loop_pts[(i + 1) % n]);
        for j in (i + 1)..n {
            if j == i || (i + 1) % n == j || (j + 1) % n == i {
                continue;
            }
            if segments_cross(a1, a2, loop_pts[j], loop_pts[(j + 1) % n]) {
                self_int = true;
                break 'scan;
            }
        }
    }
    format!("pts={n} area={a:.4} min_edge={min_edge:.5} self_intersecting={self_int}")
}

/// Strict crossing test for two open segments (shared endpoints don't count).
fn segments_cross(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> bool {
    fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }
    let (d1, d2) = (orient(p3, p4, p1), orient(p3, p4, p2));
    let (d3, d4) = (orient(p1, p2, p3), orient(p1, p2, p4));
    (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
}

/// Algebraic (Kåsa) least-squares circle fit through 2D points. Returns the centre
/// and radius, or `None` if the points are collinear/degenerate. Works for an open
/// arc (where the centroid is *not* the centre), so arcs can be detected by fit.
fn fit_circle(pts: &[Vec2]) -> Option<(Vec2, f32)> {
    if pts.len() < 3 {
        return None;
    }
    // Solve, in f64, the normal equations of |x²+y² + D·x + E·y + F|² = 0.
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0);
    for p in pts {
        let (x, y) = (p.x as f64, p.y as f64);
        let z = x * x + y * y;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    let n = pts.len() as f64;
    let m = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let sol = solve3(m, [-sxz, -syz, -sz])?;
    let (cx, cy) = (-sol[0] / 2.0, -sol[1] / 2.0);
    let r2 = cx * cx + cy * cy - sol[2];
    if r2 <= 0.0 {
        return None;
    }
    Some((Vec2::new(cx as f32, cy as f32), r2.sqrt() as f32))
}

/// Perpendicular distance from point `p` to the line through `a`–`b`.
fn point_line_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len = ab.length();
    if len < 1e-6 {
        return p.distance(a);
    }
    ((p - a).perp_dot(ab) / len).abs()
}

/// Split a polyline into maximal circular arcs and straight runs. Each entry is
/// `(start, end, Some((centre, radius)))` for an arc, or `(start, end, None)` for a
/// straight run. Used so a slot's chain (two straight sides + two end arcs, joined
/// tangentially) emits a few key points per part instead of every vertex.
fn segment_chain(pts: &[Vec2]) -> Vec<(usize, usize, Option<(Vec2, f32)>)> {
    let n = pts.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < n {
        // Greedily grow an arc starting at i (needs ≥3 points to fit a circle).
        let mut arc: Option<(Vec2, f32, usize)> = None;
        let mut j = i + 2;
        while j < n {
            if let Some((c, r)) = fit_circle(&pts[i..=j]) {
                let dev = pts[i..=j].iter().map(|p| (p.distance(c) - r).abs()).fold(0.0_f32, f32::max);
                let span = pts[i].distance(pts[j]).max(1e-4);
                // A real arc: tight fit, and radius not absurd vs the span (a near-
                // straight run fits a giant circle — reject that as straight).
                if dev < r * 0.02 && r < span * 50.0 {
                    arc = Some((c, r, j));
                    j += 1;
                    continue;
                }
            }
            break;
        }
        if let Some((c, r, e)) = arc {
            out.push((i, e, Some((c, r))));
            i = e;
        } else {
            // Straight run: extend while the next point stays on the i→end line.
            let mut end = i + 1;
            while end + 1 < n {
                let span = pts[i].distance(pts[end + 1]).max(1e-4);
                if point_line_dist(pts[end + 1], pts[i], pts[end]) < span * 0.01 {
                    end += 1;
                } else {
                    break;
                }
            }
            out.push((i, end, None));
            i = end;
        }
    }
    out
}

/// Determinant of a 3×3 matrix.
fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve a 3×3 linear system by Cramer's rule. `None` if (near-)singular.
fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = det3(m);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut out = [0.0; 3];
    for i in 0..3 {
        let mut mi = m;
        for r in 0..3 {
            mi[r][i] = b[r];
        }
        out[i] = det3(mi) / det;
    }
    Some(out)
}

/// Scale a loop outward about its centroid so every vertex moves out by at least
/// `min_offset`. Used to break exact face coincidences that defeat the kernel.
fn inflate_loop(pts: &[[f64; 2]], min_offset: f64) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n as f64;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n as f64;
    let min_d = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .fold(f64::INFINITY, f64::min);
    if min_d < 1e-6 {
        return pts.to_vec();
    }
    let f = 1.0 + min_offset / min_d;
    pts.iter().map(|p| [cx + (p[0] - cx) * f, cy + (p[1] - cy) * f]).collect()
}

/// Reset the model to an empty part with the three default planes.
fn handle_new_part(
    mut commands: Commands,
    mut ui_state: ResMut<UiState>,
    mut part: ResMut<Part>,
    mut doc: ResMut<DocRes>,
    mut session: ResMut<SketchSession>,
    mut history: ResMut<History>,
    mut edge_sel: ResMut<EdgeSelection>,
    existing: Query<Entity, With<SolidPart>>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
    if !ui_state.new_part {
        return;
    }
    ui_state.new_part = false;
    edge_sel.clear();
    history.snapshot(&doc.0);
    for e in &existing {
        commands.entity(e).despawn();
    }
    part.solid = None;
    part.mesh = None;
    part.edges.clear();
    session.selected_contours.clear();
    session.editing = None;
    doc.0 = Document::with_default_planes();
    session.plane = None;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
    session.sketch.clear();
    ui_state.pending = None;
    ui_state.selected = None;
    if let Ok((mut tf, orbit)) = cam_q.single_mut() {
        *tf = camera_transform(orbit);
    }
    info!("New part — model cleared.");
}

fn spawn_solid(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    tess: Tessellation,
) {
    let mesh = meshes.add(trimesh_to_bevy(tess.mesh));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.66, 0.74),
        metallic: 0.1,
        perceptual_roughness: 0.55,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), SolidPart, Name::new("Body")));
    // Edges are drawn by `draw_body_edges` as a gizmo overlay (no z-fighting).
}

fn trimesh_to_bevy(t: TriMesh) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, t.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, t.normals);
    mesh.insert_indices(Indices::U32(t.indices));
    mesh
}

/// Draw the body's feature edges. Each endpoint is nudged a fraction of the way
/// toward the camera, so a *visible* edge sits just in front of its face (no
/// z-fighting/flicker) while *hidden* edges stay behind their occluding faces
/// (normal depth testing → they're correctly hidden, not see-through).
fn draw_body_edges(
    mut gizmos: Gizmos,
    part: Res<Part>,
    ui_state: Res<UiState>,
    cam_q: Query<&GlobalTransform, With<Camera3d>>,
) {
    let Ok(cam) = cam_q.single() else { return };
    let cam_pos = cam.translation();
    const TOWARD_CAM: f32 = 0.0025; // 0.25% of the way to the camera
    let nudge = |p: Vec3| p + (cam_pos - p) * TOWARD_CAM;

    // Sharp edges (real corners) always draw.
    let col = Color::srgb(0.05, 0.05, 0.07);
    for e in &part.edges {
        gizmos.line(nudge(Vec3::from_array(e[0])), nudge(Vec3::from_array(e[1])), col);
    }
    // Tangent/curvature edges only when the user asks (drawn lighter to read as soft).
    if ui_state.show_tangent_edges {
        let tcol = Color::srgb(0.45, 0.47, 0.52);
        for e in &part.tangent_edges {
            gizmos.line(nudge(Vec3::from_array(e[0])), nudge(Vec3::from_array(e[1])), tcol);
        }
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    blocking: Res<UiBlocking>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut query: Query<(&Camera, &GlobalTransform, &mut Transform, &mut OrbitCamera)>,
) {
    // Orbit/pan/zoom work even while sketching now — only the UI blocks them.
    if blocking.0 {
        return;
    }
    const ORBIT_SENS: f32 = 0.005;
    const ZOOM_SENS: f32 = 0.15;

    let Ok((camera, cam_gt, mut transform, mut cam)) = query.single_mut() else { return };
    let rot = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let right = rot * Vec3::X;
    let up = rot * Vec3::Y;
    let forward = rot * Vec3::NEG_Z; // camera looks down its local -Z (toward focus)
    let mut changed = false;

    // Right-drag: orbit.
    if buttons.pressed(MouseButton::Right) && motion.delta != Vec2::ZERO {
        cam.yaw -= motion.delta.x * ORBIT_SENS;
        cam.pitch = (cam.pitch - motion.delta.y * ORBIT_SENS).clamp(-1.54, 1.54);
        changed = true;
    }

    // Middle-drag: pan (move the focus in the camera's screen plane).
    if buttons.pressed(MouseButton::Middle) && motion.delta != Vec2::ZERO {
        let k = cam.radius * 0.0016;
        cam.focus += (-right * motion.delta.x + up * motion.delta.y) * k;
        changed = true;
    }

    // Scroll: zoom toward the point under the cursor.
    if scroll.delta.y != 0.0 {
        let new_radius = (cam.radius * (1.0 - scroll.delta.y * ZOOM_SENS)).clamp(0.5, 20_000.0);
        if let Some(ray) = windows.single().ok().and_then(|w| cursor_ray(w, camera, cam_gt)) {
            let dir = ray.direction.as_vec3();
            let denom = forward.dot(dir);
            if denom.abs() > 1e-5 {
                // Point under the cursor, in the focal plane through `focus`.
                let t = forward.dot(cam.focus - ray.origin) / denom;
                if t > 0.0 {
                    let hit = ray.origin + dir * t;
                    let blend = 1.0 - new_radius / cam.radius; // >0 zooming in
                    let f = cam.focus;
                    cam.focus = f + (hit - f) * blend;
                }
            }
        }
        cam.radius = new_radius;
        changed = true;
    }

    if changed {
        *transform = camera_transform(&cam);
    }
}

// ---------------------------------------------------------------------------
// Viewport gizmos
// ---------------------------------------------------------------------------

/// Reference planes are visible only when starting a fresh part — no body yet and
/// not currently sketching. Once you're modeling they hide (and become unpickable),
/// removing both the clutter/flicker and the risk of selecting them by accident.
fn update_plane_visibility(
    part: Res<Part>,
    session: Res<SketchSession>,
    mut planes: Query<&mut Visibility, With<RefPlane>>,
) {
    let show = part.solid.is_none() && session.plane.is_none();
    let want = if show { Visibility::Visible } else { Visibility::Hidden };
    for mut vis in &mut planes {
        if *vis != want {
            *vis = want;
        }
    }
}

fn draw_world_axes(mut gizmos: Gizmos, part: Res<Part>) {
    // Only while there's no body — otherwise the axis lines run through the model.
    if part.solid.is_some() {
        return;
    }
    const L: f32 = 5.0;
    gizmos.line(Vec3::ZERO, Vec3::X * L, Color::srgb(1.0, 0.2, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Y * L, Color::srgb(0.2, 1.0, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Z * L, Color::srgb(0.3, 0.5, 1.0));
}

fn draw_sketch(
    mut gizmos: Gizmos,
    mut overlay: Gizmos<OverlayGizmos>,
    mut profile: Gizmos<ProfileGizmos>,
    session: Res<SketchSession>,
    ui_state: Res<UiState>,
    cam_q: Query<&OrbitCamera>,
) {
    let Some(ap) = &session.plane else { return };
    let radius = cam_q.single().map(|c| c.radius).unwrap_or(12.0);

    // Adaptive grid: spacing snaps to a nice 1/2/5×10^k that's ~1/16 of the view,
    // with a bounded number of cells, so it stays usable from millimetres to metres.
    let grid = Color::srgba(0.55, 0.55, 0.62, 0.18);
    let raw = (radius / 16.0).max(1e-4);
    let mag = 10f32.powf(raw.log10().floor());
    let norm = raw / mag;
    let spacing = mag * if norm < 1.5 { 1.0 } else if norm < 3.5 { 2.0 } else if norm < 7.5 { 5.0 } else { 10.0 };
    let cells = 24;
    let ext = spacing * cells as f32;
    for k in -cells..=cells {
        let f = k as f32 * spacing;
        gizmos.line(ap.to_world(Vec2::new(f, -ext)), ap.to_world(Vec2::new(f, ext)), grid);
        gizmos.line(ap.to_world(Vec2::new(-ext, f)), ap.to_world(Vec2::new(ext, f)), grid);
    }

    let solid = Color::srgb(0.95, 0.95, 0.25);
    let construction = Color::srgb(0.9, 0.45, 0.95);
    let circle_col = Color::srgb(0.25, 0.9, 0.95);
    let point_col = Color::srgb(1.0, 0.55, 0.15);
    let preview_col = Color::srgba(1.0, 1.0, 1.0, 0.6);
    let plane_rot = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));
    // Marker/snap-glyph scale tied to the zoom, so points stay visible at any scale.
    let ms = if session.snap_dist > 1e-6 { (session.snap_dist / SNAP).clamp(0.5, 40.0) } else { 1.0 };

    let uv_of = |i: usize| -> Vec2 {
        let p = &session.sketch.points[i];
        Vec2::new(p.x as f32, p.y as f32)
    };

    for e in &session.sketch.entities {
        match e {
            SketchEntity::Line { a, b, construction: is_con, reference } => {
                let (wa, wb) = (ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)));
                if *reference {
                    // Body-edge reference geometry: solid amber so it reads as "locked".
                    gizmos.line(wa, wb, Color::srgb(1.0, 0.65, 0.1));
                } else if *is_con {
                    // Dashed so construction geometry is distinguishable at a glance.
                    dashed_line(&mut gizmos, wa, wb, construction, 0.16, 0.12);
                } else {
                    profile.line(wa, wb, solid);
                }
            }
            SketchEntity::Circle { center, radius, construction: is_con } => {
                let cu = uv_of(*center);
                let r = *radius as f32;
                if *is_con {
                    // Construction circle (e.g. a polygon's circumscribed circle): a dashed
                    // ring so it reads as a guide, not a profile boundary.
                    const SEG: usize = 64;
                    let mut prev = cu + Vec2::new(r, 0.0);
                    for k in 1..=SEG {
                        let a = std::f32::consts::TAU * k as f32 / SEG as f32;
                        let p = cu + Vec2::new(r * a.cos(), r * a.sin());
                        if k % 2 == 0 {
                            gizmos.line(ap.to_world(prev), ap.to_world(p), construction);
                        }
                        prev = p;
                    }
                } else {
                    let iso = Isometry3d::new(ap.to_world(cu), plane_rot);
                    profile.circle(iso, r, circle_col);
                    // Faint diameter leader toward the Ø callout (drawn by the UI).
                    let dcol = Color::srgba(0.55, 0.85, 1.0, 0.5);
                    let d = Vec2::splat(0.707) * r;
                    gizmos.line(ap.to_world(cu - d), ap.to_world(cu + d), dcol);
                }
            }
            SketchEntity::Point { .. } => {}
            SketchEntity::Spline { points, closed, construction: is_con, control } => {
                let pts: Vec<[f64; 2]> = points
                    .iter()
                    .filter_map(|&i| session.sketch.points.get(i))
                    .map(|p| [p.x, p.y])
                    .collect();
                if pts.len() >= 2 {
                    let poly = tessellate_spline(&pts, *closed, *control);
                    let mut seg = |a: Vec2, b: Vec2| {
                        if *is_con {
                            gizmos.line(ap.to_world(a), ap.to_world(b), construction);
                        } else {
                            profile.line(ap.to_world(a), ap.to_world(b), solid);
                        }
                    };
                    for w in poly.windows(2) {
                        seg(Vec2::new(w[0][0] as f32, w[0][1] as f32), Vec2::new(w[1][0] as f32, w[1][1] as f32));
                    }
                    if *closed {
                        let (f, l) = (poly[0], poly[poly.len() - 1]);
                        seg(Vec2::new(l[0] as f32, l[1] as f32), Vec2::new(f[0] as f32, f[1] as f32));
                    }
                }
            }
            SketchEntity::Slot { a, b, radius, construction: is_con, mid } => {
                let pm = mid.and_then(|m| session.sketch.points.get(m)).map(|p| [p.x, p.y]);
                if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                    let poly = match pm {
                        Some(pm) => tessellate_arc_slot([pa.x, pa.y], pm, [pb.x, pb.y], *radius),
                        None => tessellate_slot([pa.x, pa.y], [pb.x, pb.y], *radius),
                    };
                    let n = poly.len();
                    for k in 0..n {
                        let p = Vec2::new(poly[k][0] as f32, poly[k][1] as f32);
                        let q = Vec2::new(poly[(k + 1) % n][0] as f32, poly[(k + 1) % n][1] as f32);
                        if *is_con {
                            gizmos.line(ap.to_world(p), ap.to_world(q), construction);
                        } else {
                            profile.line(ap.to_world(p), ap.to_world(q), solid);
                        }
                    }
                    // Faint centre-to-centre line.
                    gizmos.line(ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)), Color::srgba(0.9, 0.45, 0.95, 0.5));
                }
            }
            SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. } => {
                let o = uv_of(*origin);
                for loop_ in text_contours([o.x as f64, o.y as f64], contours, *height as f64, *rotation, *mirror, *arc) {
                    let n = loop_.len();
                    for k in 0..n {
                        let p = Vec2::new(loop_[k][0] as f32, loop_[k][1] as f32);
                        let q = Vec2::new(loop_[(k + 1) % n][0] as f32, loop_[(k + 1) % n][1] as f32);
                        profile.line(ap.to_world(p), ap.to_world(q), solid);
                    }
                }
            }
        }
    }

    // Text manipulation handles: for every placed Text entity, a scale handle (square,
    // bottom-right corner of its box) and a rotate handle (circle, above the box) so it
    // can be enlarged/rotated on-canvas with the Select tool.
    for (i, e) in session.sketch.entities.iter().enumerate() {
        if let SketchEntity::Text { .. } = e {
            if let Some((sc, rot, base)) = text_handles(&session.sketch, i) {
                let selected = session.selected_entities.contains(&i);
                let hcol = if selected { Color::srgb(0.2, 0.95, 0.4) } else { Color::srgba(0.2, 0.8, 0.95, 0.6) };
                // Rotate handle: a small circle with a stalk from the box top.
                gizmos.line(ap.to_world(base), ap.to_world(rot), hcol);
                gizmos.circle(Isometry3d::new(ap.to_world(rot), plane_rot), 0.12 * ms, hcol);
                // Scale handle: a little square marker.
                draw_marker(&mut gizmos, ap, sc, hcol, ms);
            }
        }
    }

    for p in &session.sketch.points {
        // Locked (body-projected) points are amber; ordinary sketch points use point_col.
        let col = if p.fixed { Color::srgb(1.0, 0.65, 0.1) } else { point_col };
        draw_marker(&mut gizmos, ap, Vec2::new(p.x as f32, p.y as f32), col, ms);
    }

    // Highlight the Selected Contours — outer + holes. Explicitly-picked contours
    // are bright green; if none are picked, every region is shown dim (it's the
    // "all contours" default that an extrude/cut would use).
    let regions = session.sketch.regions();
    if !regions.is_empty() {
        let picked: Vec<usize> =
            session.selected_contours.iter().copied().filter(|&i| i < regions.len()).collect();
        // Dense, zoom-relative scanlines so the region reads as a *solid* translucent fill
        // rather than visible hatch (gizmos can't fill polygons, so we pack the lines).
        let hatch = (radius * 0.0022).max(1e-4);
        let draw_loop = |gizmos: &mut Gizmos, loop_pts: &[[f64; 2]], col: Color| {
            let m = loop_pts.len();
            for k in 0..m {
                let a = Vec2::new(loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                let b = Vec2::new(loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                gizmos.line(ap.to_world(a), ap.to_world(b), col);
            }
        };
        // Every detected region is shown so you can see what's enclosed; explicitly
        // picked contours read brighter (blue), the rest dim green. (Picking some no
        // longer hides the others — that made an enclosed area look un-closed.)
        for (i, r) in regions.iter().enumerate() {
            let sel = picked.contains(&i);
            let (line_col, fill) = if sel {
                (Color::srgb(0.45, 0.85, 1.0), Color::srgba(0.4, 0.8, 1.0, 0.32))
            } else {
                (Color::srgba(0.2, 1.0, 0.45, 0.5), Color::srgba(0.2, 1.0, 0.45, 0.12))
            };
            hatch_region(&mut gizmos, ap, &r.outer, &r.holes, fill, hatch);
            draw_loop(&mut gizmos, &r.outer, line_col);
            for hole in &r.holes {
                draw_loop(&mut gizmos, hole, line_col);
            }
        }
    }

    // Hover highlight (Select tool): the entity under the cursor, if not selected.
    if session.tool == Tool::Select {
        if let Some(h) = session.hover_entity {
            if !session.selected_entities.contains(&h) {
                let hov = Color::srgba(1.0, 0.85, 0.4, 0.7);
                match session.sketch.entities.get(h) {
                    Some(SketchEntity::Line { a, b, .. }) => {
                        gizmos.line(ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)), hov);
                    }
                    Some(SketchEntity::Circle { center, radius, .. }) => {
                        let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                        gizmos.circle(iso, *radius as f32, hov);
                    }
                    _ => {}
                }
            }
        }
    }

    // Drag-over box-select rectangle (Select tool), drawn from the anchor to the cursor.
    if let (Some(start), Some(cur)) = (session.box_select, session.cursor_uv) {
        let col = Color::srgba(0.45, 0.85, 1.0, 0.8);
        let c = [start, Vec2::new(cur.x, start.y), cur, Vec2::new(start.x, cur.y)];
        for k in 0..4 {
            gizmos.line(ap.to_world(c[k]), ap.to_world(c[(k + 1) % 4]), col);
        }
    }

    // Highlight the selected entities (e.g. from a box select) so the selection is visible.
    let selcol = Color::srgb(0.3, 1.0, 0.5);
    for &i in &session.selected_entities {
        match session.sketch.entities.get(i) {
            Some(SketchEntity::Line { a, b, .. }) => {
                profile.line(ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)), selcol);
            }
            Some(SketchEntity::Circle { center, radius, .. }) => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                profile.circle(iso, *radius as f32, selcol);
            }
            _ => {}
        }
    }

    // Body-edge hover highlight while sketching: a single orange line on the edge under
    // the cursor (matching the 3D edge-selection highlight), so you can see what a line
    // will snap to / run along.
    if let Some(es) = session.hover_edge {
        let glow = Color::srgb(1.0, 0.6, 0.1);
        match es {
            EdgeSnap::Line([a, b]) => {
                gizmos.line(ap.to_world(a), ap.to_world(b), glow);
            }
            EdgeSnap::Arc { center, radius, a, b } => {
                // Sample the arc from a→b (the shorter way around the centre).
                let a0 = (a - center).to_angle();
                let mut sweep = (b - center).to_angle() - a0;
                while sweep > std::f32::consts::PI {
                    sweep -= std::f32::consts::TAU;
                }
                while sweep < -std::f32::consts::PI {
                    sweep += std::f32::consts::TAU;
                }
                let steps = 32;
                let mut prev = center + Vec2::from_angle(a0) * radius;
                for s in 1..=steps {
                    let ang = a0 + sweep * (s as f32 / steps as f32);
                    let cur = center + Vec2::from_angle(ang) * radius;
                    gizmos.line(ap.to_world(prev), ap.to_world(cur), glow);
                    prev = cur;
                }
            }
        }
    }

    // Highlight entities selected for a relation (Select tool): a brighter,
    // "thicker" glow (drawn as offset parallel lines) plus endpoint markers.
    let sel_col = Color::srgb(1.0, 0.7, 0.1);
    for &i in &session.selected_entities {
        match session.sketch.entities.get(i) {
            Some(SketchEntity::Line { a, b, .. }) => {
                let (wa, wb) = (ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)));
                let off = ap.n.cross((wb - wa).normalize_or_zero()).normalize_or_zero() * 0.03;
                gizmos.line(wa, wb, sel_col);
                gizmos.line(wa + off, wb + off, sel_col);
                gizmos.line(wa - off, wb - off, sel_col);
                draw_marker(&mut gizmos, ap, uv_of(*a), sel_col, ms);
                draw_marker(&mut gizmos, ap, uv_of(*b), sel_col, ms);
            }
            Some(SketchEntity::Circle { center, radius, .. }) => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                gizmos.circle(iso, *radius as f32, sel_col);
                gizmos.circle(iso, *radius as f32 + 0.03, sel_col);
            }
            _ => {}
        }
    }

    // Dimensions: an offset dimension line + extension lines (offset is draggable).
    let dim_col = Color::srgb(0.55, 0.85, 1.0);
    let pt = |i: usize| session.sketch.points.get(i).copied().map(|p| Vec2::new(p.x as f32, p.y as f32));
    for c in &session.sketch.constraints {
        match c {
            hworks_sketch::Constraint::Distance { a, b, offset, axis, .. } => {
                if let (Some(a2), Some(b2)) = (pt(*a), pt(*b)) {
                    let (p0, p1, _) = distance_dim_geometry(a2, b2, *offset as f32, *axis);
                    gizmos.line(ap.to_world(p0), ap.to_world(p1), dim_col);
                    gizmos.line(ap.to_world(a2), ap.to_world(p0), dim_col);
                    gizmos.line(ap.to_world(b2), ap.to_world(p1), dim_col);
                }
            }
            // Radius/diameter dimension: a leader from centre to rim.
            hworks_sketch::Constraint::Radius { center, value, .. } => {
                if let Some(cu) = pt(*center) {
                    let r = *value as f32;
                    let rim = cu + Vec2::new(r * 0.707, r * 0.707);
                    gizmos.line(ap.to_world(cu), ap.to_world(rim), dim_col);
                }
            }
            // Angle dimension: an arc between the two lines, around their vertex.
            hworks_sketch::Constraint::Angle { a, b, c, d, offset, .. } => {
                if let (Some(a2), Some(b2), Some(c2), Some(d2)) = (pt(*a), pt(*b), pt(*c), pt(*d)) {
                    let (vertex, _) = angle_dim_geometry(a2, b2, c2, d2, *offset as f32);
                    let r = *offset as f32;
                    // Rays away from the vertex along each line, so the arc spans the wedge
                    // *between the two lines* — the short way, regardless of point order.
                    let ray = |p: Vec2, q: Vec2| {
                        let far = if (p - vertex).length() >= (q - vertex).length() { p } else { q };
                        (far - vertex).normalize_or_zero()
                    };
                    let start = ray(a2, b2).to_angle();
                    let mut sweep = ray(c2, d2).to_angle() - start;
                    let tau = std::f32::consts::TAU;
                    while sweep > std::f32::consts::PI {
                        sweep -= tau;
                    }
                    while sweep <= -std::f32::consts::PI {
                        sweep += tau;
                    }
                    let steps = 24;
                    let mut prev = vertex + Vec2::from_angle(start) * r;
                    for s in 1..=steps {
                        let ang = start + sweep * (s as f32 / steps as f32);
                        let cur = vertex + Vec2::from_angle(ang) * r;
                        gizmos.line(ap.to_world(prev), ap.to_world(cur), dim_col);
                        prev = cur;
                    }
                }
            }
            // Point-to-line distance: a perpendicular leader from the point to the edge.
            hworks_sketch::Constraint::PointLineDistance { p, a, b, .. } => {
                if let (Some(p2), Some(a2), Some(b2)) = (pt(*p), pt(*a), pt(*b)) {
                    let (foot, _) = point_line_geometry(p2, a2, b2);
                    gizmos.line(ap.to_world(p2), ap.to_world(foot), dim_col);
                }
            }
            _ => {}
        }
    }

    // Highlight the Dimension tool's first-picked point.
    if let Some(i) = session.dim_first {
        if let Some(p) = session.sketch.points.get(i) {
            let iso = Isometry3d::new(ap.to_world(Vec2::new(p.x as f32, p.y as f32)), plane_rot);
            gizmos.circle(iso, 0.18 * ms, dim_col);
        }
    }

    // Reference snap points from the model's in-plane edges (endpoints +
    // midpoints) — shown persistently so geometry can be aligned to them.
    let ref_col = Color::srgb(0.35, 0.85, 1.0);
    for p in &session.reference_points {
        let iso = Isometry3d::new(ap.to_world(*p), plane_rot);
        gizmos.circle(iso, 0.07 * ms, ref_col);
        draw_marker(&mut gizmos, ap, *p, ref_col, ms);
    }

    // Inference/snap points (line midpoint, circle centre + quadrants) on hover.
    let inf_col = Color::srgb(1.0, 0.9, 0.25);
    for p in &session.inference_points {
        let iso = Isometry3d::new(ap.to_world(*p), plane_rot);
        gizmos.circle(iso, 0.1 * ms, inf_col);
        draw_marker(&mut gizmos, ap, *p, inf_col, ms);
    }

    // Snap indicator: ring the point the cursor would attach to.
    if let Some(cur) = session.cursor_uv {
        if let Some(i) = nearest_point(&session.sketch, cur, session.snap_dist.max(0.01)) {
            let p = &session.sketch.points[i];
            let iso = Isometry3d::new(ap.to_world(Vec2::new(p.x as f32, p.y as f32)), plane_rot);
            gizmos.circle(iso, 0.16 * ms, Color::srgb(0.2, 1.0, 0.45));
        }
    }

    if let (Some(start), Some(cur)) = (session.pending, session.cursor_uv) {
        match session.tool {
            // Midpoint line: preview grows both ways from the first click (the centre).
            Tool::Line if session.line_midpoint => {
                let other = start * 2.0 - cur;
                gizmos.line(ap.to_world(other), ap.to_world(cur), preview_col);
            }
            Tool::Line => gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col),
            // Perimeter circle: diameter from the first rim point to the cursor.
            Tool::Circle if session.circle_perimeter => {
                let center = (start + cur) * 0.5;
                let r = (cur - start).length() * 0.5;
                let iso = Isometry3d::new(ap.to_world(center), plane_rot);
                gizmos.circle(iso, r, preview_col);
            }
            Tool::Circle => {
                let r = snap_radius(start.distance(cur), &session.reference_circles, session.snap_dist.max(SNAP));
                let iso = Isometry3d::new(ap.to_world(start), plane_rot);
                gizmos.circle(iso, r, preview_col);
            }
            Tool::Rectangle => {
                let con_col = Color::srgba(0.9, 0.45, 0.95, 0.7);
                let quad = |gizmos: &mut Gizmos, c: [Vec2; 4]| {
                    for k in 0..4 {
                        gizmos.line(ap.to_world(c[k]), ap.to_world(c[(k + 1) % 4]), preview_col);
                    }
                };
                match session.rect_mode {
                    RectMode::Corner => {
                        quad(&mut gizmos, [start, Vec2::new(cur.x, start.y), cur, Vec2::new(start.x, cur.y)]);
                    }
                    RectMode::Center => {
                        let o = start * 2.0 - cur; // opposite corner
                        let c = [o, Vec2::new(cur.x, o.y), cur, Vec2::new(o.x, cur.y)];
                        quad(&mut gizmos, c);
                        gizmos.line(ap.to_world(c[0]), ap.to_world(c[2]), con_col); // X diagonals
                        gizmos.line(ap.to_world(c[1]), ap.to_world(c[3]), con_col);
                    }
                    RectMode::Parallelogram => {
                        if let Some(b) = session.pending_b {
                            let d = start + (cur - b);
                            quad(&mut gizmos, [start, b, cur, d]);
                            draw_marker(&mut gizmos, ap, b, point_col, ms);
                        } else {
                            gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col);
                        }
                    }
                }
            }
            Tool::Slot => {
                let cl_col = Color::srgba(0.9, 0.45, 0.95, 0.6);
                let outline = |gizmos: &mut Gizmos, poly: &[[f64; 2]]| {
                    let n = poly.len();
                    for k in 0..n {
                        let p = Vec2::new(poly[k][0] as f32, poly[k][1] as f32);
                        let q = Vec2::new(poly[(k + 1) % n][0] as f32, poly[(k + 1) % n][1] as f32);
                        gizmos.line(ap.to_world(p), ap.to_world(q), preview_col);
                    }
                };
                match session.slot_mode {
                    SlotMode::Straight | SlotMode::Centerpoint => {
                        // Centrepoint's first click is the centre, so its line is mirrored.
                        let (a, b) = match (session.slot_mode, session.pending_b) {
                            (SlotMode::Centerpoint, None) => (start * 2.0 - cur, cur),
                            (SlotMode::Centerpoint, Some(end)) => (start * 2.0 - end, end),
                            (_, b) => (start, b.unwrap_or(cur)),
                        };
                        if session.pending_b.is_some() {
                            outline(&mut gizmos, &tessellate_slot([a.x as f64, a.y as f64], [b.x as f64, b.y as f64], perp_dist(cur, a, b).max(0.01) as f64));
                            gizmos.line(ap.to_world(a), ap.to_world(b), cl_col);
                        } else {
                            gizmos.line(ap.to_world(a), ap.to_world(b), preview_col);
                        }
                    }
                    SlotMode::Arc => {
                        let b = session.pending_b;
                        match (b, session.pending_c) {
                            (Some(b), Some(p)) => {
                                let r = arc_slot_width(cur, start, p, b).max(0.01);
                                outline(&mut gizmos, &tessellate_arc_slot([start.x as f64, start.y as f64], [p.x as f64, p.y as f64], [b.x as f64, b.y as f64], r as f64));
                                draw_marker(&mut gizmos, ap, b, point_col, ms);
                                draw_marker(&mut gizmos, ap, p, point_col, ms);
                            }
                            (Some(b), None) => {
                                // Bending: arc centre line through A, cursor, B. tessellate_arc_slot
                                // at r≈0 collapses onto the centreline, tracing the arc itself.
                                let arc = tessellate_arc_slot([start.x as f64, start.y as f64], [cur.x as f64, cur.y as f64], [b.x as f64, b.y as f64], 0.0);
                                let n = arc.len();
                                for k in 0..n {
                                    let p = Vec2::new(arc[k][0] as f32, arc[k][1] as f32);
                                    let q = Vec2::new(arc[(k + 1) % n][0] as f32, arc[(k + 1) % n][1] as f32);
                                    gizmos.line(ap.to_world(p), ap.to_world(q), cl_col);
                                }
                                draw_marker(&mut gizmos, ap, b, point_col, ms);
                            }
                            _ => gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col),
                        }
                    }
                }
            }
            // Polygon: centre is `start`, cursor is a vertex. Preview the N edges and the
            // dashed circumscribed circle.
            Tool::Polygon => {
                let n = session.polygon_sides.max(3);
                let rim = poly_rim(&session).unwrap_or(cur);
                let r = (rim - start).length();
                let theta0 = (rim - start).y.atan2((rim - start).x);
                let vert = |k: usize| {
                    let a = theta0 + std::f32::consts::TAU * k as f32 / n as f32;
                    start + Vec2::new(a.cos(), a.sin()) * r
                };
                for k in 0..n {
                    gizmos.line(ap.to_world(vert(k)), ap.to_world(vert((k + 1) % n)), preview_col);
                }
                // Dashed circumscribed circle.
                const SEG: usize = 64;
                let mut prev = start + Vec2::new(r, 0.0);
                for k in 1..=SEG {
                    let a = std::f32::consts::TAU * k as f32 / SEG as f32;
                    let p = start + Vec2::new(r * a.cos(), r * a.sin());
                    if k % 2 == 0 {
                        gizmos.line(ap.to_world(prev), ap.to_world(p), construction);
                    }
                    prev = p;
                }
            }
            // Text commits on a single click (no rubber-band preview); Pattern / Mirror act
            // on existing geometry rather than rubber-banding a new entity.
            Tool::Select | Tool::Dimension | Tool::Spline | Tool::Text | Tool::Pattern | Tool::Mirror => {}
        }
        draw_marker(&mut gizmos, ap, start, point_col, ms);
    }

    // In-progress spline preview: the curve through the placed points + the cursor.
    if session.tool == Tool::Spline && !session.spline_pts.is_empty() {
        let mut pts: Vec<[f64; 2]> =
            session.spline_pts.iter().map(|p| [p.x as f64, p.y as f64]).collect();
        if let Some(cur) = session.cursor_uv {
            pts.push([cur.x as f64, cur.y as f64]);
        }
        if pts.len() >= 2 {
            let poly = tessellate_spline(&pts, false, session.spline_control);
            for w in poly.windows(2) {
                let a = Vec2::new(w[0][0] as f32, w[0][1] as f32);
                let b = Vec2::new(w[1][0] as f32, w[1][1] as f32);
                gizmos.line(ap.to_world(a), ap.to_world(b), preview_col);
            }
        }
        for p in &session.spline_pts {
            draw_marker(&mut gizmos, ap, *p, point_col, ms);
        }
    }

    // ---- Pattern preview: ghost copies of the selection at the instance transforms ----
    if session.tool == Tool::Pattern {
        // Mark the circular-pattern centre so the chosen revolve point is visible.
        if session.pattern_mode == PatternMode::Circular {
            let c = if session.pat_center_set {
                session.pat_circ_center
            } else {
                selection_centroid(&session.sketch, &pattern_seeds(&session))
            };
            let cw = ap.to_world(c);
            let r = 0.18 * ms;
            let col = Color::srgb(1.0, 0.55, 0.1);
            gizmos.line(ap.to_world(c - Vec2::new(r, 0.0)), ap.to_world(c + Vec2::new(r, 0.0)), col);
            gizmos.line(ap.to_world(c - Vec2::new(0.0, r)), ap.to_world(c + Vec2::new(0.0, r)), col);
            gizmos.circle(Isometry3d::new(cw, plane_rot), r * 0.7, col);
        }
        let seeds = pattern_seeds(&session);
        if let Ok(xfs) = pattern_instances(&session, &seeds) {
            let ghost = Color::srgba(0.35, 0.85, 1.0, 0.7);
            // Cache each seed's outline once, then stamp it at every instance transform.
            let outlines: Vec<Vec<Vec<Vec2>>> =
                seeds.iter().map(|&e| entity_preview_polylines(&session.sketch, e)).collect();
            for xf in &xfs {
                for polylines in &outlines {
                    for poly in polylines {
                        for w in poly.windows(2) {
                            gizmos.line(ap.to_world(xf.apply(w[0])), ap.to_world(xf.apply(w[1])), ghost);
                        }
                    }
                }
            }
        }
    }

    // ---- Mirror preview: ghost reflection of the selection across the axis line ----
    if session.tool == Tool::Mirror {
        if let Some((axis, a, b)) = mirror_axis(&session) {
            // Emphasise the axis line.
            gizmos.line(ap.to_world(a), ap.to_world(b), Color::srgba(0.9, 0.45, 0.95, 0.9));
            let ghost = Color::srgba(0.35, 0.85, 1.0, 0.7);
            for e in mirror_seeds(&session, axis) {
                for poly in entity_preview_polylines(&session.sketch, e) {
                    for w in poly.windows(2) {
                        let (ra, rb) = (reflect_across(w[0], a, b), reflect_across(w[1], a, b));
                        gizmos.line(ap.to_world(ra), ap.to_world(rb), ghost);
                    }
                }
            }
        }
    }

    // ---- Boss/Cut preview: direction arrow + ghost extrusion of the contours ----
    if let Some(op) = &ui_state.pending {
        let regions = session.sketch.regions();
        let picked: Vec<usize> =
            session.selected_contours.iter().copied().filter(|&i| i < regions.len()).collect();
        let indices: Vec<usize> = if picked.is_empty() { (0..regions.len()).collect() } else { picked };
        // A boss goes out along +normal by default; a cut goes in (−normal, into the
        // material). Reverse flips either one.
        let kind_sign = match op.kind {
            OpKind::Boss => 1.0,
            OpKind::Cut => -1.0,
        };
        let nominal = kind_sign * if op.reverse { -1.0 } else { 1.0 };
        let lift = ap.n * (op.depth * nominal);
        let ghost = match op.kind {
            OpKind::Boss => Color::srgba(0.95, 0.85, 0.25, 0.8),
            OpKind::Cut => Color::srgba(1.0, 0.4, 0.35, 0.8),
        };

        // Ghost prism, drawn on the overlay group so it shows THROUGH the model — the
        // far end is the cut-depth indicator (where the cut bottoms out, à la SW).
        // The far loop is drawn brighter so the depth reads clearly.
        let far = match op.kind {
            OpKind::Boss => ghost,
            OpKind::Cut => Color::srgb(1.0, 0.75, 0.2), // bright depth ring for a cut
        };
        for &i in &indices {
            for loop_pts in std::iter::once(&regions[i].outer).chain(regions[i].holes.iter()) {
                let m = loop_pts.len();
                for k in 0..m {
                    let a = Vec2::new(loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                    let b = Vec2::new(loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                    overlay.line(ap.to_world(a) + lift, ap.to_world(b) + lift, far); // far loop (depth)
                    overlay.line(ap.to_world(a), ap.to_world(a) + lift, ghost); // riser
                }
            }
        }
        // Direction arrow from the contours' centroid — on the overlay group so it's
        // always visible. Grab it (within 16px) to drag the depth.
        if let Some(base_uv) = contours_centroid(&session) {
            let c = ap.to_world(base_uv);
            let acol = if session.arrow_drag { Color::srgb(1.0, 1.0, 0.5) } else { Color::srgb(1.0, 0.85, 0.2) };
            // The arrow always points OUT along the face normal (away from the body)
            // so it stays visible and grabbable, even for a cut whose ghost goes in.
            let handle = c + ap.n * op.depth.max(0.5 * ms.max(1.0));
            overlay.arrow(c, handle, acol);
            overlay.sphere(Isometry3d::from_translation(handle), 0.15 * ms.max(1.0), acol);
        }
    }
}

/// Shade a region by scan-line hatching (even-odd, so holes are left empty) — a
/// gizmo-only "fill" that makes a selected contour read as selected.
fn hatch_region(
    gizmos: &mut Gizmos,
    ap: &ActivePlane,
    outer: &[[f64; 2]],
    holes: &[Vec<[f64; 2]>],
    color: Color,
    spacing: f32,
) {
    if outer.len() < 3 || spacing <= 0.0 {
        return;
    }
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in outer {
        ymin = ymin.min(p[1] as f32);
        ymax = ymax.max(p[1] as f32);
    }
    let mut y = ymin + spacing * 0.5;
    let mut guard = 0;
    while y < ymax && guard < 4000 {
        guard += 1;
        let mut xs: Vec<f32> = Vec::new();
        for loop_pts in std::iter::once(outer).chain(holes.iter().map(|h| h.as_slice())) {
            let m = loop_pts.len();
            for k in 0..m {
                let (x1, y1) = (loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                let (x2, y2) = (loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                if (y1 > y) != (y2 > y) {
                    xs.push(x1 + (y - y1) / (y2 - y1) * (x2 - x1));
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        while k + 1 < xs.len() {
            gizmos.line(ap.to_world(Vec2::new(xs[k], y)), ap.to_world(Vec2::new(xs[k + 1], y)), color);
            k += 2;
        }
        y += spacing;
    }
}

fn draw_marker(gizmos: &mut Gizmos, ap: &ActivePlane, uv: Vec2, color: Color, scale: f32) {
    let s = 0.08 * scale;
    gizmos.line(ap.to_world(uv + Vec2::new(-s, 0.0)), ap.to_world(uv + Vec2::new(s, 0.0)), color);
    gizmos.line(ap.to_world(uv + Vec2::new(0.0, -s)), ap.to_world(uv + Vec2::new(0.0, s)), color);
}

/// Draw a dashed segment a→b (used so construction geometry reads as construction).
fn dashed_line(gizmos: &mut Gizmos, a: Vec3, b: Vec3, color: Color, dash: f32, gap: f32) {
    let total = a.distance(b);
    if total < 1e-5 {
        return;
    }
    let dir = (b - a) / total;
    let step = (dash + gap).max(1e-3);
    let mut t = 0.0;
    while t < total {
        let s = a + dir * t;
        let e = a + dir * (t + dash).min(total);
        gizmos.line(s, e, color);
        t += step;
    }
}

// ---------------------------------------------------------------------------
// Model-edge selection (view mode)
//
// `part.edges` is a flat list of straight feature-edge segments (a box has 12; a
// tessellated circle is many short ones). Picking one and walking along it while
// the direction stays smooth turns it into the user-meaningful unit: a straight
// edge stops at sharp corners, while a circle walks all the way round into a loop.
// ---------------------------------------------------------------------------

/// Quantize a world point so coincident edge endpoints match despite float noise.
fn edge_key(p: Vec3) -> (i64, i64, i64) {
    ((p.x * 1.0e4).round() as i64, (p.y * 1.0e4).round() as i64, (p.z * 1.0e4).round() as i64)
}

/// Screen-space distance (logical px) from `cursor` to the projected segment a–b.
fn segment_screen_dist(
    camera: &Camera,
    cam_gt: &GlobalTransform,
    cursor: Vec2,
    a: Vec3,
    b: Vec3,
) -> Option<f32> {
    let pa = camera.world_to_viewport(cam_gt, a).ok()?;
    let pb = camera.world_to_viewport(cam_gt, b).ok()?;
    let ab = pb - pa;
    let t = if ab.length_squared() > 1e-6 {
        ((cursor - pa).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((cursor - (pa + ab * t)).length())
}

/// Centroid (in plane uv) of the contours a pending extrude would use — the base
/// of the direction arrow.
fn contours_centroid(session: &SketchSession) -> Option<Vec2> {
    let regions = session.sketch.regions();
    if regions.is_empty() {
        return None;
    }
    let picked: Vec<usize> =
        session.selected_contours.iter().copied().filter(|&i| i < regions.len()).collect();
    let idxs: Vec<usize> = if picked.is_empty() { (0..regions.len()).collect() } else { picked };
    let (mut sum, mut count) = (Vec2::ZERO, 0.0_f32);
    for i in idxs {
        for p in &regions[i].outer {
            sum += Vec2::new(p[0] as f32, p[1] as f32);
            count += 1.0;
        }
    }
    (count > 0.0).then(|| sum / count)
}

/// Signed parameter `t` along the axis (`base` + `n`·t) of the point closest to the
/// cursor ray — used to read an extrude depth from where the user drags the arrow.
fn closest_t_on_axis(base: Vec3, n: Vec3, ro: Vec3, rd: Vec3) -> f32 {
    let n = n.normalize_or_zero();
    let d = rd.normalize_or_zero();
    let w0 = base - ro;
    let b = n.dot(d);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-4 {
        return 0.0; // axis ~parallel to the view ray
    }
    let t = (b * d.dot(w0) - n.dot(w0)) / denom;
    if t.is_finite() {
        t
    } else {
        0.0
    }
}

/// Drag the boss/cut direction arrow to set the depth live. Returns true while it's
/// actively handling the drag (so the caller skips the normal sketch tools).
fn extrude_arrow_drag(
    session: &mut SketchSession,
    ui_state: &mut UiState,
    window: &Window,
    camera: &Camera,
    cam_gt: &GlobalTransform,
    ray: &Ray3d,
    just_pressed: bool,
    pressed: bool,
    just_released: bool,
) -> bool {
    let (Some(ap), Some(op)) = (session.plane.clone(), ui_state.pending.clone()) else {
        return false;
    };
    let Some(base_uv) = contours_centroid(session) else { return false };
    let base = ap.to_world(base_uv);
    let n = ap.n.normalize_or_zero();
    // The handle points OUT along the face normal — its length must match the DRAWN
    // arrow (which scales with zoom), or the grab target sits where nothing's shown.
    let ms = if session.snap_dist > 1e-6 { (session.snap_dist / SNAP).clamp(0.5, 40.0) } else { 1.0 };
    let tip = base + n * op.depth.max(0.5 * ms.max(1.0));

    if just_pressed {
        if let Some(cursor) = window.cursor_position() {
            // Generous grab: anywhere near the shaft, or within a fat radius of the tip
            // handle, so a short (small-depth) arrow is still easy to catch.
            let near_shaft = segment_screen_dist(camera, cam_gt, cursor, base, tip).is_some_and(|d| d < 22.0);
            let near_tip = camera
                .world_to_viewport(cam_gt, tip)
                .map(|p| p.distance(cursor) < 26.0)
                .unwrap_or(false);
            if near_shaft || near_tip {
                session.arrow_drag = true;
            }
        }
    }
    if session.arrow_drag && pressed {
        // `t` is the distance dragged out along +normal. Set the depth from it (clamped
        // to a sane, finite range); keep the direction on the Reverse checkbox so a
        // little wobble past the face can't silently flip the cut. A non-finite `t`
        // (axis nearly edge-on to the view) is ignored — a NaN here crashes egui.
        let t = closest_t_on_axis(base, n, ray.origin, ray.direction.as_vec3());
        if t.is_finite() {
            let depth = t.clamp(0.1, 10_000.0);
            ui_state.pending = Some(PendingOp { kind: op.kind, depth, reverse: op.reverse });
        }
        if just_released {
            session.arrow_drag = false;
        }
        return true;
    }
    if just_released {
        session.arrow_drag = false;
    }
    false
}

/// Index of the body edge segment nearest the cursor in screen space, within
/// `thresh` pixels.
fn pick_edge(
    edges: &[[[f32; 3]; 2]],
    camera: &Camera,
    cam_gt: &GlobalTransform,
    cursor: Vec2,
    thresh: f32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in edges.iter().enumerate() {
        let (a, b) = (Vec3::from_array(e[0]), Vec3::from_array(e[1]));
        if let Some(d) = segment_screen_dist(camera, cam_gt, cursor, a, b) {
            if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Expand a single picked edge segment into the maximal smooth chain through it.
/// Returns the ordered world vertices and whether the chain closes on itself.
fn edge_chain(edges: &[[[f32; 3]; 2]], seed: usize) -> (Vec<Vec3>, bool) {
    use std::collections::HashMap;
    let mut key_to_id: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut pos: Vec<Vec3> = Vec::new();
    let mut seg: Vec<(usize, usize)> = Vec::with_capacity(edges.len());
    for e in edges {
        let a = vertex_id(Vec3::from_array(e[0]), &mut key_to_id, &mut pos);
        let b = vertex_id(Vec3::from_array(e[1]), &mut key_to_id, &mut pos);
        seg.push((a, b));
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); pos.len()];
    for (si, (a, b)) in seg.iter().enumerate() {
        adj[*a].push(si);
        adj[*b].push(si);
    }
    let dir = |from: usize, to: usize| (pos[to] - pos[from]).normalize_or_zero();

    let (start_a, start_b) = seg[seed];

    // Walk away from `cur` (came in along `incoming`), never re-using `prev_seg`.
    // Stops at the seed vertex (closed loop), a dead end, or a sharp corner.
    let walk = |mut prev_seg: usize, mut cur: usize, mut incoming: Vec3, stop_at: usize| {
        let mut out: Vec<usize> = Vec::new();
        let mut closed = false;
        loop {
            let mut best: Option<(usize, usize, f32)> = None; // (seg, other vertex, dot)
            for &sg in &adj[cur] {
                if sg == prev_seg {
                    continue;
                }
                let (a, b) = seg[sg];
                let other = if a == cur {
                    b
                } else if b == cur {
                    a
                } else {
                    continue;
                };
                let d = incoming.dot(dir(cur, other));
                if best.map_or(true, |(_, _, bd)| d > bd) {
                    best = Some((sg, other, d));
                }
            }
            match best {
                Some((sg, other, d)) if d >= EDGE_CONTINUE_COS => {
                    if other == stop_at {
                        closed = true;
                        break;
                    }
                    out.push(other);
                    incoming = dir(cur, other);
                    prev_seg = sg;
                    cur = other;
                    if out.len() > seg.len() {
                        break; // safety against pathological input
                    }
                }
                _ => break,
            }
        }
        (out, closed)
    };

    let (fwd, closed) = walk(seed, start_b, dir(start_a, start_b), start_a);
    if closed {
        // A loop: seed vertices plus everything walked, no need to walk backward.
        let mut chain = vec![pos[start_a], pos[start_b]];
        chain.extend(fwd.iter().map(|&i| pos[i]));
        return (chain, true);
    }
    let (bwd, _) = walk(seed, start_a, dir(start_b, start_a), start_b);
    let mut chain: Vec<Vec3> = bwd.iter().rev().map(|&i| pos[i]).collect();
    chain.push(pos[start_a]);
    chain.push(pos[start_b]);
    chain.extend(fwd.iter().map(|&i| pos[i]));
    (chain, false)
}

/// Intern a world point, returning a stable per-position vertex id.
fn vertex_id(
    p: Vec3,
    key_to_id: &mut std::collections::HashMap<(i64, i64, i64), usize>,
    pos: &mut Vec<Vec3>,
) -> usize {
    let key = edge_key(p);
    *key_to_id.entry(key).or_insert_with(|| {
        pos.push(p);
        pos.len() - 1
    })
}

/// Key points to flash for a selected chain: a loop's quarter points (which are a
/// circle's top/bottom/left/right), or an open edge's endpoints + midpoint.
fn chain_flash_points(chain: &[Vec3], closed: bool) -> Vec<Vec3> {
    if chain.len() < 2 {
        return chain.to_vec();
    }
    // Cumulative arc length along the polyline (plus the closing segment if a loop).
    let mut verts = chain.to_vec();
    if closed {
        verts.push(chain[0]);
    }
    let mut cum = vec![0.0_f32];
    for w in verts.windows(2) {
        cum.push(cum.last().unwrap() + w[0].distance(w[1]));
    }
    let total = *cum.last().unwrap();
    if total < 1e-5 {
        return vec![chain[0]];
    }
    let at = |target: f32| -> Vec3 {
        let t = target.clamp(0.0, total);
        let k = cum.partition_point(|&c| c < t).max(1);
        let (c0, c1) = (cum[k - 1], cum[k]);
        let f = if c1 > c0 { (t - c0) / (c1 - c0) } else { 0.0 };
        verts[k - 1].lerp(verts[k], f)
    };
    if closed {
        [0.0, 0.25, 0.5, 0.75].iter().map(|f| at(f * total)).collect()
    } else {
        vec![chain[0], at(total * 0.5), *chain.last().unwrap()]
    }
}

/// Count down the edge-selection key-point flash.
fn tick_edge_flash(time: Res<Time>, mut sel: ResMut<EdgeSelection>) {
    if sel.flash > 0.0 {
        sel.flash = (sel.flash - time.delta_secs()).max(0.0);
    }
}

/// Draw the persistently-selected edge/loop, plus its key points while flashing.
fn draw_edge_selection(
    mut gizmos: Gizmos,
    sel: Res<EdgeSelection>,
    session: Res<SketchSession>,
    ui_state: Res<UiState>,
    cam_q: Query<&GlobalTransform, With<Camera3d>>,
) {
    if session.plane.is_some() {
        return; // only in view mode
    }
    let Ok(cam) = cam_q.single() else { return };
    let cam_pos = cam.translation();
    // Nudge toward the camera (like body edges) so the highlight sits in front.
    const TOWARD_CAM: f32 = 0.004;
    let nudge = |p: Vec3| p + (cam_pos - p) * TOWARD_CAM;

    // While the Fillet/Chamfer PM is open, highlight every picked edge (bright yellow).
    if ui_state.pending_fillet.is_some() || ui_state.pending_chamfer.is_some() {
        let fcol = Color::srgb(1.0, 0.95, 0.2);
        for edge in &ui_state.fillet_edges {
            for w in edge.windows(2) {
                let a = Vec3::new(w[0][0] as f32, w[0][1] as f32, w[0][2] as f32);
                let b = Vec3::new(w[1][0] as f32, w[1][1] as f32, w[1][2] as f32);
                gizmos.line(nudge(a), nudge(b), fcol);
            }
        }
    }

    if sel.chain.len() < 2 {
        return;
    }
    let col = Color::srgb(1.0, 0.6, 0.1);
    for w in sel.chain.windows(2) {
        gizmos.line(nudge(w[0]), nudge(w[1]), col);
    }
    if sel.closed {
        gizmos.line(nudge(*sel.chain.last().unwrap()), nudge(sel.chain[0]), col);
    }

    if sel.flash > 0.0 {
        let a = (sel.flash / FLASH_SECS).clamp(0.0, 1.0); // fade out
        let fcol = Color::srgba(0.3, 1.0, 0.5, a);
        // Size the cross-markers by how far the camera is, so they stay visible at
        // any scale (the larger the part / the further back, the bigger they draw).
        let dist = cam_pos.distance(sel.chain[0]).max(1.0);
        let s = (dist * 0.012).clamp(0.08, 50.0);
        for p in &sel.flash_points {
            let c = nudge(*p);
            gizmos.line(c - Vec3::X * s, c + Vec3::X * s, fcol);
            gizmos.line(c - Vec3::Y * s, c + Vec3::Y * s, fcol);
            gizmos.line(c - Vec3::Z * s, c + Vec3::Z * s, fcol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hworks_document::{Document, FeatureKind, PlaneRef};
    use hworks_sketch::Sketch;

    fn rect_sketch(w: f64, h: f64) -> Sketch {
        let mut s = Sketch::default();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(w, 0.0);
        let p2 = s.add_point(w, h);
        let p3 = s.add_point(0.0, h);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        s.add_line(p3, p0, false);
        s
    }

    fn xy() -> PlaneRef {
        PlaneRef { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    fn height(solid: &KSolid) -> f32 {
        let m = tessellate(solid, 0.05).mesh;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &m.positions {
            lo = lo.min(p[2]);
            hi = hi.max(p[2]);
        }
        hi - lo
    }

    #[test]
    fn regenerate_replays_an_extrude_into_a_box() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: xy(), distance: 2.0 });
        let solid = regenerate(&doc).expect("regen should produce a body");
        assert_eq!(tessellate(&solid, 0.05).edges.len(), 12);
    }

    #[test]
    fn adjacent_contours_merge_into_one_extrudable_outline() {
        use hworks_sketch::Sketch;
        let mut s = Sketch::default();
        let ca = s.add_point(0.0, 0.0);
        s.add_circle(ca, 3.0);
        let cb = s.add_point(10.0, 0.0);
        s.add_circle(cb, 3.0);
        let a1 = s.add_point(-1.0, 2.0);
        let a2 = s.add_point(11.0, 2.0);
        s.add_line(a1, a2, false);
        let b1 = s.add_point(-1.0, -2.0);
        let b2 = s.add_point(11.0, -2.0);
        s.add_line(b1, b2, false);
        let regions = s.regions();
        // The three adjacent strips (circle A · band · circle B) share line edges, so
        // they merge into ONE outline — the dumbbell — that extrudes as one solid
        // (no fragile coincident-face union).
        let refs: Vec<&hworks_sketch::Region> = regions.iter().collect();
        let merged = merge_regions(&refs);
        assert_eq!(merged.len(), 1, "adjacent contours should merge into one outline, got {}", merged.len());
        let solid = extrude_solid(&merged[0].outer, &merged[0].holes, &xy_basis(), 2.0);
        assert!(solid.is_some(), "the merged dumbbell outline must extrude");
    }

    #[test]
    fn a_diameter_line_splits_then_merges_back_to_a_disk() {
        use hworks_sketch::Sketch;
        let mut s = Sketch::default();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 3.0);
        let a = s.add_point(-4.0, 0.0);
        let b = s.add_point(4.0, 0.0);
        s.add_line(a, b, false);
        assert_eq!(s.regions().len(), 2, "diameter splits the disk in two");
        let merged = merge_regions(&s.regions().iter().collect::<Vec<_>>());
        assert_eq!(merged.len(), 1, "the two halves merge back into one disk");
    }

    #[test]
    fn disjoint_contours_stay_separate_when_merged() {
        use hworks_sketch::Region;
        let a = Region { outer: rect(0.0, 0.0, 2.0, 2.0), holes: vec![] };
        let b = Region { outer: rect(5.0, 0.0, 7.0, 2.0), holes: vec![] };
        let merged = merge_regions(&[&a, &b]);
        assert_eq!(merged.len(), 2, "disjoint contours must stay two outlines");
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    fn xy_basis() -> hworks_geometry::PlaneBasis {
        hworks_geometry::PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    #[test]
    fn cut_pocket_into_one_of_two_cylinders() {
        use hworks_sketch::Sketch;
        let mut doc = Document::with_default_planes();
        let mut a = Sketch::default();
        let pa = a.add_point(0.0, 0.0);
        a.add_circle(pa, 3.0);
        doc.add_feature(FeatureKind::Extrude { sketch: a, regions: vec![], plane: xy(), distance: 5.0 });
        let mut b = Sketch::default();
        let pb = b.add_point(10.0, 0.0);
        b.add_circle(pb, 3.0);
        doc.add_feature(FeatureKind::Extrude { sketch: b, regions: vec![], plane: xy(), distance: 3.0 });
        let before = tessellate(&regenerate(&doc).unwrap(), 0.05).edges.len();
        // Cut a 1mm hole in the top of cylinder A (z=5), 2mm deep.
        let mut cut = Sketch::default();
        let pc = cut.add_point(0.0, 0.0);
        cut.add_circle(pc, 1.0);
        let top = PlaneRef { origin: [0.0, 0.0, 5.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        doc.add_feature(FeatureKind::Cut { sketch: cut, regions: vec![], plane: top, distance: 2.0 });
        let after_solid = regenerate(&doc).expect("body after cut");
        let after = tessellate(&after_solid, 0.05).edges.len();
        eprintln!("edges before cut {before}, after cut {after}");
        assert!(after > before, "the cut must add pocket edges (before {before}, after {after})");
    }

    #[test]
    fn cut_into_a_dumbbell_does_not_panic() {
        use hworks_sketch::Sketch;
        let mut boss = Sketch::default();
        let ca = boss.add_point(0.0, 0.0);
        boss.add_circle(ca, 3.0);
        let cb = boss.add_point(10.0, 0.0);
        boss.add_circle(cb, 3.0);
        let a1 = boss.add_point(-1.0, 2.0);
        let a2 = boss.add_point(11.0, 2.0);
        boss.add_line(a1, a2, false);
        let b1 = boss.add_point(-1.0, -2.0);
        let b2 = boss.add_point(11.0, -2.0);
        boss.add_line(b1, b2, false);
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: boss, regions: vec![], plane: xy(), distance: 2.0 });
        // Cut a 1mm-radius hole in the middle, from the top face.
        let mut cutsk = Sketch::default();
        let cc = cutsk.add_point(5.0, 0.0);
        cutsk.add_circle(cc, 1.0);
        let top = PlaneRef { origin: [0.0, 0.0, 2.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        doc.add_feature(FeatureKind::Cut { sketch: cutsk, regions: vec![], plane: top, distance: 1.0 });
        let solid = regenerate(&doc); // must not panic
        assert!(solid.is_some(), "dumbbell with a cut should still produce a body");
    }

    #[test]
    fn editing_a_distance_rebuilds_taller() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: xy(), distance: 2.0 });
        let h2 = height(&regenerate(&doc).unwrap());
        if let FeatureKind::Extrude { distance, .. } = &mut doc.features.last_mut().unwrap().kind {
            *distance = 6.0;
        }
        let h6 = height(&regenerate(&doc).unwrap());
        assert!((h2 - 2.0).abs() < 0.1, "h2 was {h2}");
        assert!((h6 - 6.0).abs() < 0.1, "h6 was {h6}");
    }

    #[test]
    fn editing_an_upstream_height_shifts_stacked_features() {
        let mut doc = Document::with_default_planes();
        // Base box 4×4×2 on XY.
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0 });
        // Boss 2×2 sketched on the top face (z=2), 2 tall → stacked total height 4.
        let top = PlaneRef { origin: [0.0, 0.0, 2.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: top, distance: 2.0 });
        let before = height(&regenerate(&doc).unwrap());
        assert!((before - 4.0).abs() < 0.2, "stacked height should be 4, was {before}");
        // Grow the base to 5 tall — the boss must ride up to z=5..7 (total 7), not
        // stay at z=2..4 (which would leave the base poking through it).
        if let FeatureKind::Extrude { distance, .. } = &mut doc.features[3].kind {
            *distance = 5.0;
        }
        let after = height(&regenerate(&doc).unwrap());
        assert!((after - 7.0).abs() < 0.3, "boss should ride up to total 7, was {after}");
    }

    #[test]
    fn rollback_suppresses_downstream_features() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0 });
        assert!(regenerate(&doc).is_some());
        // Roll the bar back before the extrude → no body.
        doc.rollback = doc.features.len() - 1;
        assert!(regenerate(&doc).is_none(), "rolled-back model should have no body");
    }

    #[test]
    fn document_round_trips_through_ron_and_regenerates() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude {
            sketch: rect_sketch(2.0, 2.0),
            regions: vec![0],
            plane: xy(),
            distance: 2.0,
        });
        let text = ron::ser::to_string_pretty(&doc, ron::ser::PrettyConfig::default()).unwrap();
        let reloaded: Document = ron::from_str(&text).expect("RON parses back");
        assert_eq!(reloaded.features.len(), doc.features.len());
        let solid = regenerate(&reloaded).expect("reloaded document regenerates");
        assert_eq!(tessellate(&solid, 0.05).edges.len(), 12);
    }

    #[test]
    fn regenerate_replays_a_cut_through_the_box() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0 });
        // Cut a centred 2x2 pocket from the same plane (body is on the +normal side).
        let mut pocket = Sketch::default();
        let a = pocket.add_point(1.0, 1.0);
        let b = pocket.add_point(3.0, 1.0);
        let c = pocket.add_point(3.0, 3.0);
        let d = pocket.add_point(1.0, 3.0);
        pocket.add_line(a, b, false);
        pocket.add_line(b, c, false);
        pocket.add_line(c, d, false);
        pocket.add_line(d, a, false);
        doc.add_feature(FeatureKind::Cut { sketch: pocket, regions: vec![0], plane: xy(), distance: 2.0 });
        let solid = regenerate(&doc).expect("regen with a cut should produce a body");
        assert!(tessellate(&solid, 0.05).edges.len() > 12, "cut should add edges");
    }

    fn two_disjoint_squares() -> Sketch {
        let mut s = Sketch::default();
        for (x0, y0) in [(0.0_f64, 0.0_f64), (5.0, 0.0)] {
            let p0 = s.add_point(x0, y0);
            let p1 = s.add_point(x0 + 2.0, y0);
            let p2 = s.add_point(x0 + 2.0, y0 + 2.0);
            let p3 = s.add_point(x0, y0 + 2.0);
            s.add_line(p0, p1, false);
            s.add_line(p1, p2, false);
            s.add_line(p2, p3, false);
            s.add_line(p3, p0, false);
        }
        s
    }

    #[test]
    fn extrude_one_of_two_contours_makes_a_single_box() {
        let s = two_disjoint_squares();
        assert_eq!(s.regions().len(), 2, "two separate squares are two regions");
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![0], plane: xy(), distance: 2.0 });
        let edges = tessellate(&regenerate(&doc).unwrap(), 0.05).edges.len();
        assert_eq!(edges, 12, "one selected contour → one box, got {edges}");
    }

    #[test]
    fn extrude_all_contours_builds_both_boxes() {
        let s = two_disjoint_squares();
        let mut doc = Document::with_default_planes();
        // Empty selection ⇒ all contours.
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![], plane: xy(), distance: 2.0 });
        let edges = tessellate(&regenerate(&doc).unwrap(), 0.05).edges.len();
        // Two separate boxes = 24 edges (proves the disjoint union worked).
        assert_eq!(edges, 24, "all contours → two boxes (24 edges), got {edges}");
    }

    #[test]
    fn inflate_loop_grows_every_vertex_past_the_offset() {
        let circle: Vec<[f64; 2]> = (0..32)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 32.0;
                [2.0 * a.cos(), 2.0 * a.sin()]
            })
            .collect();
        let big = inflate_loop(&circle, 0.06);
        for (p, q) in circle.iter().zip(big.iter()) {
            let dp = (p[0].powi(2) + p[1].powi(2)).sqrt();
            let dq = (q[0].powi(2) + q[1].powi(2)).sqrt();
            assert!(dq - dp >= 0.06 - 1e-9, "vertex only moved {}", dq - dp);
        }
    }

    #[test]
    fn boss_union_handles_a_coincident_circle() {
        use hworks_geometry::PlaneBasis;
        use hworks_sketch::Region;
        // Wedge body: a quarter disk (r=2) extruded to height 2 on the XY plane.
        let mut wedge_profile = vec![[0.0_f64, 0.0_f64]];
        for k in 0..=16 {
            let a = std::f64::consts::FRAC_PI_2 * k as f64 / 16.0;
            wedge_profile.push([2.0 * a.cos(), 2.0 * a.sin()]);
        }
        let xy = PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let wedge = extrude_solid(&wedge_profile, &[], &xy, 2.0).expect("wedge");
        // A full circle of the SAME radius drawn on the top face → coincident arc.
        let circle: Vec<[f64; 2]> = (0..64)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / 64.0;
                [2.0 * a.cos(), 2.0 * a.sin()]
            })
            .collect();
        let top = PlaneBasis { origin: [0.0, 0.0, 2.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let region = Region { outer: circle, holes: vec![] };
        // The exact union fails in truck; boss_union must recover via the nudge.
        let solid = boss_union(&wedge, &region, &top, 2.0).expect("coincident cylinder boss should union");
        let edges = tessellate(&solid, 0.05).edges.len();
        assert!(edges > 12, "combined wedge+cylinder should have many edges, got {edges}");
    }

    #[test]
    fn coincident_nudge_is_imperceptible() {
        // The nudge that makes the union succeed must stay far below the 0.03
        // tessellation tolerance, so the part is exact for all practical purposes.
        assert!(COINCIDENT_NUDGE < 0.03 / 10.0, "nudge {COINCIDENT_NUDGE} too large for CAD accuracy");
    }

    #[test]
    fn fit_circle_recovers_centre_and_radius_from_an_arc() {
        // A quarter arc (open) of radius 5 centred at (2, 3) — the case that floods
        // a pie slice. Its centroid is NOT the centre, so this needs a real fit.
        let pts: Vec<Vec2> = (0..=16)
            .map(|k| {
                let a = std::f32::consts::FRAC_PI_2 * k as f32 / 16.0;
                Vec2::new(2.0 + 5.0 * a.cos(), 3.0 + 5.0 * a.sin())
            })
            .collect();
        let (c, r) = fit_circle(&pts).expect("arc fits a circle");
        assert!((c - Vec2::new(2.0, 3.0)).length() < 1e-2, "centre {c:?}");
        assert!((r - 5.0).abs() < 1e-2, "radius {r}");
    }

    #[test]
    fn fit_circle_rejects_a_straight_line() {
        let pts: Vec<Vec2> = (0..5).map(|k| Vec2::new(k as f32, 2.0 * k as f32)).collect();
        assert!(fit_circle(&pts).is_none(), "collinear points have no circle");
    }

    #[test]
    fn edge_chain_single_segment_is_an_open_pair() {
        let edges = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        let (chain, closed) = edge_chain(&edges, 0);
        assert!(!closed);
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn edge_chain_merges_collinear_segments() {
        let edges = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ];
        let (chain, closed) = edge_chain(&edges, 0);
        assert!(!closed);
        assert_eq!(chain.len(), 3, "two collinear segments are one 3-vertex edge");
    }

    #[test]
    fn edge_chain_walks_a_tessellated_circle_into_a_loop() {
        let n = 48usize;
        let mut edges = Vec::new();
        for k in 0..n {
            let a = std::f32::consts::TAU * k as f32 / n as f32;
            let b = std::f32::consts::TAU * (k + 1) as f32 / n as f32;
            edges.push([[a.cos(), a.sin(), 0.0], [b.cos(), b.sin(), 0.0]]);
        }
        let (chain, closed) = edge_chain(&edges, 0);
        assert!(closed, "a finely tessellated circle should close into a loop");
        assert_eq!(chain.len(), n);
    }

    #[test]
    fn edge_chain_stops_at_sharp_square_corners() {
        let edges = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        let (chain, closed) = edge_chain(&edges, 0);
        assert!(!closed, "90° corners are too sharp to chain");
        assert_eq!(chain.len(), 2, "just the one clicked side");
    }

    #[test]
    fn flash_point_counts_match_open_and_closed_chains() {
        let open = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        assert_eq!(chain_flash_points(&open, false).len(), 3, "endpoints + midpoint");
        let loop_pts = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        assert_eq!(chain_flash_points(&loop_pts, true).len(), 4, "loop quarter points");
    }
}
