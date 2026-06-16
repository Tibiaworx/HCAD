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
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::settings::{PowerPreference, RenderCreation, WgpuSettings};
use bevy::render::RenderPlugin;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use hworks_document::{Document, FeatureKind, Plane, PlaneRef};
use hworks_geometry::{
    cut_tol, extrude_solid, extrude_solid_with_overlap, tessellate, union_tol, KSolid, PlaneBasis,
    Tessellation, TriMesh,
};

/// How far a boss reaches back into the body it's built on (flush path). Kept below
/// the 0.03 tessellation tolerance so any overhang lip is sub-facet (invisible),
/// while big enough to keep the union robust paired with the tight tolerance.
const BOSS_OVERLAP: f64 = 0.01;
use hworks_sketch::{point_in_poly, Constraint, Sketch, SketchEntity};

/// Default boss/cut depth used by the keyboard accelerators (the UI lets you edit it).
const EXTRUDE_DISTANCE: f64 = 2.0;
const PLANE_SIZE: f32 = 8.0;
const SNAP: f32 = 0.18;
/// How long (seconds) an edge's key points flash after it's selected.
const FLASH_SECS: f32 = 1.2;
/// Two adjacent edge segments merge into one chain while the turn between them
/// stays under ~60° (dot ≥ this). Sharp model corners (~90°) break the chain, so a
/// box edge stays a single edge while a tessellated circle walks into a full loop.
const EDGE_CONTINUE_COS: f32 = 0.5;
/// Screen-space pixel radius for picking a model edge under the cursor.
const EDGE_PICK_PX: f32 = 9.0;

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
                }),
        )
        .add_plugins(EguiPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .insert_resource(DocRes(Document::with_default_planes()))
        .init_resource::<SketchSession>()
        .init_resource::<Part>()
        .init_resource::<UiState>()
        .init_resource::<UiBlocking>()
        .init_resource::<History>()
        .init_resource::<EdgeSelection>()
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .add_systems(
            Update,
            (
                sketch_interaction,
                handle_keys,
                history_keys,
                apply_history,
                handle_file_io,
                handle_edit_sketch,
                handle_exit_sketch,
                do_solid_op,
                do_regenerate,
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
    Dimension,
}

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Line => "Line",
            Tool::Circle => "Circle",
            Tool::Rectangle => "Rectangle",
            Tool::Dimension => "Dimension",
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
}

/// CommandManager tabs (SolidWorks-style), to declutter the toolbar.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Tab {
    #[default]
    Features,
    Sketch,
}

/// True while egui wants the pointer — suppresses viewport drawing/orbit.
#[derive(Resource, Default)]
struct UiBlocking(bool);

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
    pending: Option<Vec2>,
    /// First point picked by the Dimension tool (point index).
    dim_first: Option<usize>,
    /// Live dimension input while drawing (length for a line, radius for a circle).
    live_buf: f32,
    /// Request keyboard focus on the live-input field (set when a draw starts).
    request_live_focus: bool,
    drag: Option<usize>,
    dirty: bool,
    op_request: Option<SolidOp>,
    cursor_uv: Option<Vec2>,
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
    /// Snap tolerance in world units, scaled to the zoom so snapping feels the same
    /// at any scale (a fixed tolerance is unusable on a large part). Set each frame.
    snap_dist: f32,
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
    doc: Res<DocRes>,
) {
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
) -> Result {
    let ctx = contexts.ctx_mut()?;

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

    // Auto-switch tab on entering/leaving a sketch.
    if in_sketch != ui_state.was_sketching {
        ui_state.active_tab = if in_sketch { Tab::Sketch } else { Tab::Features };
        ui_state.was_sketching = in_sketch;
    }

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
            }
            if ui.selectable_label(ui_state.active_tab == Tab::Sketch, "Sketch").clicked() {
                ui_state.active_tab = Tab::Sketch;
            }
            ui.separator();

            match ui_state.active_tab {
                Tab::Sketch => {
                    ui.add_enabled_ui(in_sketch, |ui| {
                        for (tool, name, tip) in [
                            (Tool::Select, "Select", "Select & drag points — geometry re-solves (S)"),
                            (Tool::Line, "Line", "Draw line segments; endpoints snap to close loops (L)"),
                            (Tool::Circle, "Circle", "Click centre, then radius (C)"),
                            (Tool::Rectangle, "Rectangle", "Click two opposite corners (R)"),
                            (Tool::Dimension, "Dimension", "Click two points to set an exact distance (M)"),
                        ] {
                            if ui.selectable_label(session.tool == tool, name).on_hover_text(tip).clicked() {
                                session.tool = tool;
                                session.pending = None;
                            }
                        }
                        let con = session.construction;
                        if ui.selectable_label(con, "Construction").on_hover_text("Toggle construction geometry (X)").clicked() {
                            session.construction = !con;
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
                }
            }
        });
        ui.add_space(2.0);
    });

    // ---- Left panel: PropertyManager (if configuring) else FeatureManager ----
    egui::SidePanel::left("left_panel").default_width(240.0).show(ctx, |ui| {
        if let Some(mut op) = ui_state.pending.clone() {
            // PropertyManager laid out like SolidWorks' Boss-Extrude.
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
                        egui::DragValue::new(&mut op.depth).speed(0.1).range(0.1..=200.0).suffix(" mm"),
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

            let dims: Vec<usize> = session
                .sketch
                .constraints
                .iter()
                .enumerate()
                .filter_map(|(i, c)| matches!(c, Constraint::Distance { .. }).then_some(i))
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
                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    for (k, i) in dims.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("D{}", k + 1));
                            if let Some(Constraint::Distance { value, .. }) =
                                session.sketch.constraints.get_mut(*i)
                            {
                                if ui
                                    .add(egui::DragValue::new(value).speed(0.05).range(0.01..=10_000.0).suffix(" mm"))
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                    }
                });
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
            let line_circle = lines.len() == 1 && circles.len() == 1;
            ui.label(egui::RichText::new(format!("Relations — {} selected", sel.len())).strong());
            if sel.is_empty() {
                ui.label(
                    egui::RichText::new("Select tool: click 2 lines (or a line + circle).")
                        .italics()
                        .weak(),
                );
            }
            let mut applied: Option<Constraint> = None;
            ui.horizontal_wrapped(|ui| {
                if ui.add_enabled(two_lines, egui::Button::new("Parallel")).clicked() {
                    applied = Some(Constraint::Parallel(lines[0].0, lines[0].1, lines[1].0, lines[1].1));
                }
                if ui.add_enabled(two_lines, egui::Button::new("Perpendicular")).clicked() {
                    applied = Some(Constraint::Perpendicular(lines[0].0, lines[0].1, lines[1].0, lines[1].1));
                }
                if ui.add_enabled(two_lines, egui::Button::new("Equal")).clicked() {
                    applied = Some(Constraint::Equal(lines[0].0, lines[0].1, lines[1].0, lines[1].1));
                }
                if ui.add_enabled(line_circle, egui::Button::new("Tangent")).clicked() {
                    applied = Some(Constraint::Tangent {
                        a: lines[0].0,
                        b: lines[0].1,
                        center: circles[0].0,
                        radius: circles[0].1,
                    });
                }
            });
            if !sel.is_empty() && ui.button("Clear selection").clicked() {
                session.selected_entities.clear();
            }
            if let Some(c) = applied {
                session.sketch.constraints.push(c);
                session.selected_entities.clear();
                session.needs_apply = true;
            }
        } else {
            ui.heading("HCAD Part");

            // Rollback bar: replay only features[..rollback].
            let n = doc.0.features.len();
            if n > 0 {
                let mut rb = doc.0.rollback.min(n);
                ui.horizontal(|ui| {
                    ui.label("Rollback");
                    if ui.add(egui::Slider::new(&mut rb, 0..=n)).changed() {
                        doc.0.rollback = rb;
                        ui_state.regen = true;
                    }
                });
            }
            ui.separator();

            // Hierarchical feature tree. Right-click a node for edit/extrude/delete.
            let rollback = doc.0.rollback;
            let mut action: Option<TreeAction> = None;
            egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
                // Reference planes, tucked into one collapsible group.
                egui::CollapsingHeader::new("Reference Planes").default_open(false).show(ui, |ui| {
                    for (_id, p) in doc.0.planes() {
                        ui.label(egui::RichText::new(&p.name).weak());
                    }
                });

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

                    match &f.kind {
                        FeatureKind::Plane(_) => continue,
                        FeatureKind::Sketch { .. } => {
                            sk += 1;
                            let resp = ui.selectable_label(selected, styled(format!("Sketch{sk}")));
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
                        }
                        FeatureKind::Extrude { distance, .. } | FeatureKind::Cut { distance, .. } => {
                            let (label, child) = match &f.kind {
                                FeatureKind::Extrude { .. } => {
                                    ex += 1;
                                    (format!("Extrude{ex}  (h={distance:.1})"), format!("Sketch of Extrude{ex}"))
                                }
                                _ => {
                                    ct += 1;
                                    (format!("Cut{ct}  (h={distance:.1})"), format!("Sketch of Cut{ct}"))
                                }
                            };
                            let resp = egui::CollapsingHeader::new(styled(label))
                                .id_salt(i)
                                .default_open(false)
                                .show(ui, |ui| {
                                    // The nested sketch has its OWN right-click menu.
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
                            // Feature (header) menu — distinct from the sketch menu above.
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
                        }
                    }
                }
            });

            // Apply the tree context-menu action (after the borrow above ends).
            if let Some(act) = action {
                match act {
                    TreeAction::Select(i) => ui_state.selected = Some(i),
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

            // Inline depth editor for the selected Extrude/Cut — Apply to confirm.
            if let Some(i) = ui_state.selected {
                let depth = doc.0.features.get(i).and_then(|f| match &f.kind {
                    FeatureKind::Extrude { distance, .. } | FeatureKind::Cut { distance, .. } => Some(*distance),
                    _ => None,
                });
                if let Some(depth) = depth {
                    // Sync the editor's value when the selection changes.
                    if ui_state.edit_depth_for != Some(i) {
                        ui_state.edit_depth = depth as f32;
                        ui_state.edit_depth_for = Some(i);
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Edit feature").strong());
                    ui.horizontal(|ui| {
                        ui.label("Depth (mm):");
                        // Allow negatives so a reversed extrude/cut survives editing.
                        ui.add(egui::DragValue::new(&mut ui_state.edit_depth).speed(0.1).range(-200.0..=200.0));
                    });
                    let changed = (ui_state.edit_depth as f64 - depth).abs() > 1e-6;
                    ui.horizontal(|ui| {
                        if ui.add_enabled(changed, egui::Button::new("Apply").fill(egui::Color32::from_rgb(40, 110, 70))).clicked() {
                            history.snapshot(&doc.0);
                            if let Some(f) = doc.0.features.get_mut(i) {
                                match &mut f.kind {
                                    FeatureKind::Extrude { distance, .. }
                                    | FeatureKind::Cut { distance, .. } => *distance = ui_state.edit_depth as f64,
                                    _ => {}
                                }
                            }
                            ui_state.regen = true;
                        }
                        if ui.add_enabled(changed, egui::Button::new("Revert")).clicked() {
                            ui_state.edit_depth = depth as f32;
                        }
                    });
                }
            }
        }
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
    if let (Some(ap), Ok((camera, cam_gt))) = (session.plane.as_ref(), cam_read.single()) {
        for (k, c) in session.sketch.constraints.iter().enumerate() {
            if let Constraint::Distance { a, b, value } = c {
                if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                    let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                    let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                    let dir = (b2 - a2).normalize_or_zero();
                    let perp = Vec2::new(-dir.y, dir.x) * 0.5;
                    let world = ap.to_world((a2 + b2) * 0.5 + perp);
                    // world_to_viewport returns logical pixels = egui points (no /ppp).
                    if let Ok(screen) = camera.world_to_viewport(cam_gt, world) {
                        let pos = egui::pos2(screen.x, screen.y);
                        egui::Area::new(egui::Id::new(("dimlabel", k)))
                            .fixed_pos(pos)
                            .order(egui::Order::Foreground)
                            .show(ctx, |ui| {
                                ui.label(
                                    egui::RichText::new(format!("{value:.1}"))
                                        .color(egui::Color32::from_rgb(150, 215, 255))
                                        .strong(),
                                );
                            });
                    }
                }
            }
        }
    }

    blocking.0 = ctx.wants_pointer_input() || ctx.is_pointer_over_area();
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
            SketchEntity::Circle { center, radius } => {
                ((uv - p(*center)).length() - *radius as f32).abs()
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
        Some(SketchEntity::Circle { center, radius }) => {
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
    points
        .iter()
        .copied()
        .filter(|p| p.distance(uv) <= thresh)
        .min_by(|a, b| a.distance(uv).total_cmp(&b.distance(uv)))
        .unwrap_or(uv)
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
        Some(SketchEntity::Circle { center, radius }) => Some((*center, *radius)),
        _ => None,
    }
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
) {
    if blocking.0 {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_gt, mut cam_tf, mut orbit)) = cam_q.single_mut() else { return };
    let Some(ray) = cursor_ray(window, camera, cam_gt) else { return };

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
        let mut consumed: Vec<Vec3> = Vec::new();
        for (i, e) in part.edges.iter().enumerate() {
            let (a, b) = (Vec3::from_array(e[0]), Vec3::from_array(e[1]));
            if !in_plane(a) || !in_plane(b) {
                continue;
            }
            if a.distance(b) > 0.4 {
                // A straight feature edge → endpoints + midpoint as snap points.
                let (ua, ub) = (to_uv(a), to_uv(b));
                session.reference_points.push(ua);
                session.reference_points.push(ub);
                session.reference_points.push((ua + ub) * 0.5);
            } else if !consumed.iter().any(|c| c.distance(a) < 1e-3) {
                // A short segment may be part of a tessellated circular edge — walk
                // its loop and, if it's circular, expose the centre + radius.
                let (chain, closed) = edge_chain(&part.edges, i);
                if closed && chain.len() >= 8 {
                    let center: Vec3 = chain.iter().copied().sum::<Vec3>() / chain.len() as f32;
                    let radius = chain.iter().map(|p| p.distance(center)).sum::<f32>() / chain.len() as f32;
                    let max_dev =
                        chain.iter().map(|p| (p.distance(center) - radius).abs()).fold(0.0_f32, f32::max);
                    if radius > 1e-3 && max_dev < radius * 0.05 && in_plane(center) {
                        let c_uv = to_uv(center);
                        session.reference_circles.push((c_uv, radius));
                        session.reference_points.push(c_uv); // concentric centre snap
                    }
                    consumed.extend(chain);
                }
            }
        }
        // Drop duplicate points (edges share corner vertices).
        session.reference_points.sort_by(|p, q| {
            p.x.partial_cmp(&q.x).unwrap().then(p.y.partial_cmp(&q.y).unwrap())
        });
        session.reference_points.dedup_by(|p, q| p.distance(*q) < 1e-3);
    }

    // The working cursor snaps to an inference point or a reference point near it.
    if session.plane.is_some() {
        let mut snaps = session.inference_points.clone();
        snaps.extend_from_slice(&session.reference_points);
        session.cursor_uv = active_uv.map(|uv| snap_to_inference(uv, &snaps, snap));
    }

    // Entity under the cursor (Select tool) for hover highlighting.
    session.hover_entity = if session.plane.is_some() && session.tool == Tool::Select {
        active_uv.and_then(|uv| nearest_entity(&session.sketch, uv, snap * 1.5))
    } else {
        None
    };

    let just_pressed = buttons.just_pressed(MouseButton::Left);
    let pressed = buttons.pressed(MouseButton::Left);
    let just_released = buttons.just_released(MouseButton::Left);

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
                info!("Sketching on the {} plane.", ap.name);
                session.plane = Some(ap);
            } else {
                // Clicked empty space (no edge, no face) → deselect.
                edge_sel.clear();
            }
        }
        return;
    }

    // Switching tools drops any in-progress dimension pick / entity selection.
    if session.tool != Tool::Dimension {
        session.dim_first = None;
    }
    if session.tool != Tool::Select {
        session.selected_entities.clear();
    }

    match session.tool {
        Tool::Select => {
            if just_pressed {
                let hit = active_uv.and_then(|uv| nearest_point(&session.sketch, uv, snap));
                session.drag = hit;
                if hit.is_none() {
                    if let Some(uv) = active_uv {
                        if let Some(e) = nearest_entity(&session.sketch, uv, snap * 1.5) {
                            // Click a line/circle to (de)select it for a constraint (keep ≤2).
                            if let Some(pos) = session.selected_entities.iter().position(|&x| x == e) {
                                session.selected_entities.remove(pos);
                            } else {
                                session.selected_entities.push(e);
                                while session.selected_entities.len() > 2 {
                                    session.selected_entities.remove(0);
                                }
                            }
                        } else {
                            // Empty space inside a closed region: toggle it as a
                            // Selected Contour (SolidWorks-style). Clicking truly
                            // empty space just clears the entity selection.
                            session.selected_entities.clear();
                            if let Some(r) = region_at(&session.sketch, uv) {
                                if let Some(pos) = session.selected_contours.iter().position(|&x| x == r) {
                                    session.selected_contours.remove(pos);
                                } else {
                                    session.selected_contours.push(r);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(i) = session.drag {
                if pressed {
                    if let Some(uv) = active_uv {
                        if let Some(p) = session.sketch.points.get_mut(i) {
                            p.x = uv.x as f64;
                            p.y = uv.y as f64;
                        }
                        session.sketch.solve_with_fixed(&[i]);
                    }
                }
                if just_released {
                    session.drag = None;
                }
            }
        }
        Tool::Line | Tool::Circle | Tool::Rectangle if just_pressed => {
            // Use the snapped cursor so endpoints land on midpoints / quadrants / centres.
            if let Some(uv) = session.cursor_uv {
                place_point(&mut session, uv);
            }
        }
        Tool::Dimension if just_pressed => {
            if let Some(uv) = active_uv {
                // Smart pick: a point starts/continues a point-to-point distance;
                // clicking a line dimensions its length; a circle's radius is edited
                // in the panel (all circles are listed there).
                if let Some(p) = nearest_point(&session.sketch, uv, snap * 1.5) {
                    match session.dim_first.take() {
                        Some(first) if first != p => add_distance_dim(&mut session.sketch, first, p),
                        _ => session.dim_first = Some(p),
                    }
                } else if let Some((a, b)) =
                    nearest_entity(&session.sketch, uv, snap * 2.0).and_then(|e| entity_line(&session.sketch, e))
                {
                    add_distance_dim(&mut session.sketch, a, b);
                    session.dim_first = None;
                }
            }
        }
        _ => {}
    }

    if session.drag.is_none() && session.dirty {
        session.sketch.solve();
        session.dirty = false;
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
    session.sketch.constraints.push(Constraint::Distance { a, b, value: length as f64 });
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

/// True if a distance dimension already exists between points `a` and `b`.
fn has_distance(sketch: &Sketch, a: usize, b: usize) -> bool {
    sketch.constraints.iter().any(|c| {
        matches!(c, Constraint::Distance { a: x, b: y, .. } if (*x == a && *y == b) || (*x == b && *y == a))
    })
}

/// Add a distance dimension between two points at their current distance — unless
/// one already exists for that pair (avoids over-driving the geometry).
fn add_distance_dim(sketch: &mut Sketch, a: usize, b: usize) {
    if a == b || has_distance(sketch, a, b) {
        return;
    }
    let (pa, pb) = (sketch.points[a], sketch.points[b]);
    let d = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt();
    sketch.constraints.push(Constraint::Distance { a, b, value: d });
}

fn place_point(session: &mut SketchSession, uv: Vec2) {
    let snap = session.snap_dist;
    match session.tool {
        Tool::Line => {
            if let Some(start) = session.pending.take() {
                let a = get_or_add_point(&mut session.sketch, start, snap);
                let b = get_or_add_point(&mut session.sketch, uv, snap);
                session.sketch.add_line(a, b, session.construction);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Circle => {
            if let Some(center) = session.pending.take() {
                let radius = snap_radius(center.distance(uv), &session.reference_circles, snap);
                let c = get_or_add_point(&mut session.sketch, center, snap);
                session.sketch.add_circle(c, radius as f64);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Rectangle => {
            if let Some(c0) = session.pending.take() {
                let c1 = uv;
                let s = &mut session.sketch;
                let p0 = s.add_point(c0.x as f64, c0.y as f64);
                let p1 = s.add_point(c1.x as f64, c0.y as f64);
                let p2 = s.add_point(c1.x as f64, c1.y as f64);
                let p3 = s.add_point(c0.x as f64, c1.y as f64);
                s.add_line(p0, p1, false);
                s.add_line(p1, p2, false);
                s.add_line(p2, p3, false);
                s.add_line(p3, p0, false);
                s.constraints.push(Constraint::Horizontal(p0, p1));
                s.constraints.push(Constraint::Horizontal(p3, p2));
                s.constraints.push(Constraint::Vertical(p1, p2));
                s.constraints.push(Constraint::Vertical(p0, p3));
                session.dirty = true;
            } else {
                session.pending = Some(uv);
            }
        }
        Tool::Select | Tool::Dimension => {}
    }
}

fn handle_keys(keys: Res<ButtonInput<KeyCode>>, mut session: ResMut<SketchSession>) {
    if session.plane.is_none() {
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
    if keys.just_pressed(KeyCode::KeyE) {
        session.op_request = Some(SolidOp::Boss(EXTRUDE_DISTANCE));
    }
    if keys.just_pressed(KeyCode::KeyD) {
        session.op_request = Some(SolidOp::Cut(EXTRUDE_DISTANCE));
    }
    if keys.just_pressed(KeyCode::Escape) {
        if session.pending.is_some() {
            session.pending = None; // cancel the in-progress entity first
        } else {
            // Commit the sketch to the timeline and leave (handled by handle_exit_sketch).
            session.exit_request = true;
        }
    }
}

/// Ctrl+Z = undo, Ctrl+Shift+Z / Ctrl+Y = redo.
fn history_keys(keys: Res<ButtonInput<KeyCode>>, mut ui_state: ResMut<UiState>) {
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

/// Apply a requested undo/redo by swapping the whole document snapshot.
fn apply_history(mut history: ResMut<History>, mut doc: ResMut<DocRes>, mut ui_state: ResMut<UiState>) {
    if ui_state.undo_request {
        ui_state.undo_request = false;
        if let Some(prev) = history.undo.pop() {
            history.redo.push(doc.0.clone());
            doc.0 = prev;
            ui_state.regen = true;
            ui_state.selected = None;
        }
    }
    if ui_state.redo_request {
        ui_state.redo_request = false;
        if let Some(next) = history.redo.pop() {
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

    if matches!(op, SolidOp::Cut(_)) && part.solid.is_none() {
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
    doc: Res<DocRes>,
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
        FeatureKind::Plane(_) => return,
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
                FeatureKind::Plane(_) => {}
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
fn regenerate(doc: &Document) -> Option<KSolid> {
    let mut body: Option<KSolid> = None;
    let end = doc.rollback.min(doc.features.len());
    for feature in &doc.features[..end] {
        match &feature.kind {
            FeatureKind::Plane(_) => {}
            FeatureKind::Sketch { .. } => {} // 2D only — no solid contribution
            FeatureKind::Extrude { sketch, regions, plane, distance } => {
                let all = sketch.regions();
                let basis = basis_from_ref(plane);
                // Boss each selected contour (empty ⇒ all), unioning into the body.
                for r in chosen_regions(&all, regions) {
                    let next = match &body {
                        Some(b) => boss_union(b, r, &basis, *distance),
                        None => extrude_solid(&r.outer, &r.holes, &basis, *distance),
                    };
                    if let Some(s) = next {
                        body = Some(s);
                    } else {
                        warn!("Regen: an extrude contour could not be built.");
                    }
                }
            }
            FeatureKind::Cut { sketch, regions, plane, distance } => {
                if body.is_none() {
                    continue;
                }
                let all = sketch.regions();
                let basis = basis_from_ref(plane);
                // Cut each selected contour (empty ⇒ all) from the current body.
                for r in chosen_regions(&all, regions) {
                    let Some(b) = &body else { break };
                    // Pick the cut direction from the *current* body, so it stays
                    // correct even after upstream edits move things around.
                    let centroid = mesh_centroid(&tessellate(b, 0.06).mesh);
                    let origin = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
                    let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                    let signed = if (centroid - origin).dot(n) < 0.0 { -*distance } else { *distance };
                    if let Some(s) = cut_op(b, r, &basis, signed) {
                        body = Some(s);
                    } else {
                        warn!("Regen: a cut contour could not be built.");
                    }
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

    for e in &existing {
        commands.entity(e).despawn();
    }
    match regenerate(&doc.0) {
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

/// `PlaneBasis` (kernel-side) from a stored `PlaneRef`.
fn basis_from_ref(p: &PlaneRef) -> PlaneBasis {
    PlaneBasis { origin: p.origin, u: p.u, v: p.v, normal: p.normal }
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

/// Add a boss (region `r`) to an existing body, trying progressively more robust
/// strategies so a boolean never simply fails: flush+exact, flush+nudge (coincident
/// faces), then the robust overlap/tolerance with and without the nudge. The first
/// (cleanest) one that works wins.
fn boss_union(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    // (radial nudge, overlap, tolerance)
    let strategies = [
        (0.0, BOSS_OVERLAP, COINCIDENT_TOL),
        (COINCIDENT_NUDGE, BOSS_OVERLAP, COINCIDENT_TOL),
        (0.0, ROBUST_OVERLAP, ROBUST_TOL),
        (COINCIDENT_NUDGE, ROBUST_OVERLAP, ROBUST_TOL),
    ];
    for (k, &(nudge, overlap, tol)) in strategies.iter().enumerate() {
        let outer = if nudge > 0.0 { inflate_loop(&r.outer, nudge) } else { r.outer.clone() };
        if let Some(s) = extrude_solid_with_overlap(&outer, &r.holes, basis, distance, overlap)
            .and_then(|boss| union_tol(body, &boss, tol))
        {
            if k > 0 {
                info!("Boss union: used fallback strategy {k} (coincident/awkward faces).");
            }
            return Some(s);
        }
    }
    None
}

/// Cut region `r` from the body — same escalating fallback idea as [`boss_union`].
fn cut_op(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64) -> Option<KSolid> {
    let strategies = [(0.0, COINCIDENT_TOL), (COINCIDENT_NUDGE, COINCIDENT_TOL), (0.0, ROBUST_TOL), (COINCIDENT_NUDGE, ROBUST_TOL)];
    for (k, &(nudge, tol)) in strategies.iter().enumerate() {
        let outer = if nudge > 0.0 { inflate_loop(&r.outer, nudge) } else { r.outer.clone() };
        if let Some(s) = cut_tol(body, &outer, &r.holes, basis, distance, tol) {
            if k > 0 {
                info!("Cut: used fallback strategy {k} (coincident/awkward faces).");
            }
            return Some(s);
        }
    }
    None
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

fn draw_sketch(mut gizmos: Gizmos, session: Res<SketchSession>) {
    let Some(ap) = &session.plane else { return };

    let grid = Color::srgba(0.55, 0.55, 0.62, 0.18);
    let half = (PLANE_SIZE * 0.5) as i32;
    for i in -half..=half {
        let f = i as f32;
        gizmos.line(ap.to_world(Vec2::new(f, -PLANE_SIZE * 0.5)), ap.to_world(Vec2::new(f, PLANE_SIZE * 0.5)), grid);
        gizmos.line(ap.to_world(Vec2::new(-PLANE_SIZE * 0.5, f)), ap.to_world(Vec2::new(PLANE_SIZE * 0.5, f)), grid);
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
            SketchEntity::Line { a, b, construction: is_con } => {
                let (wa, wb) = (ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)));
                if *is_con {
                    // Dashed so construction geometry is distinguishable at a glance.
                    dashed_line(&mut gizmos, wa, wb, construction, 0.16, 0.12);
                } else {
                    gizmos.line(wa, wb, solid);
                }
            }
            SketchEntity::Circle { center, radius } => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                gizmos.circle(iso, *radius as f32, circle_col);
            }
            SketchEntity::Point { .. } => {}
        }
    }

    for p in &session.sketch.points {
        draw_marker(&mut gizmos, ap, Vec2::new(p.x as f32, p.y as f32), point_col, ms);
    }

    // Highlight the Selected Contours — outer + holes. Explicitly-picked contours
    // are bright green; if none are picked, every region is shown dim (it's the
    // "all contours" default that an extrude/cut would use).
    let regions = session.sketch.regions();
    if !regions.is_empty() {
        let picked: Vec<usize> =
            session.selected_contours.iter().copied().filter(|&i| i < regions.len()).collect();
        let (indices, region_col): (Vec<usize>, Color) = if picked.is_empty() {
            ((0..regions.len()).collect(), Color::srgba(0.2, 1.0, 0.45, 0.4))
        } else {
            (picked, Color::srgb(0.2, 1.0, 0.45))
        };
        let draw_loop = |gizmos: &mut Gizmos, loop_pts: &[[f64; 2]]| {
            let m = loop_pts.len();
            for k in 0..m {
                let a = Vec2::new(loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                let b = Vec2::new(loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                gizmos.line(ap.to_world(a), ap.to_world(b), region_col);
            }
        };
        for i in indices {
            draw_loop(&mut gizmos, &regions[i].outer);
            for hole in &regions[i].holes {
                draw_loop(&mut gizmos, hole);
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
                    Some(SketchEntity::Circle { center, radius }) => {
                        let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                        gizmos.circle(iso, *radius as f32, hov);
                    }
                    _ => {}
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
            Some(SketchEntity::Circle { center, radius }) => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                gizmos.circle(iso, *radius as f32, sel_col);
                gizmos.circle(iso, *radius as f32 + 0.03, sel_col);
            }
            _ => {}
        }
    }

    // Dimensions (Distance constraints): an offset dimension line + extension lines.
    let dim_col = Color::srgb(0.55, 0.85, 1.0);
    for c in &session.sketch.constraints {
        if let hworks_sketch::Constraint::Distance { a, b, .. } = c {
            if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                let dir = (b2 - a2).normalize_or_zero();
                let perp = Vec2::new(-dir.y, dir.x) * 0.5; // offset the dim line
                gizmos.line(ap.to_world(a2 + perp), ap.to_world(b2 + perp), dim_col);
                gizmos.line(ap.to_world(a2), ap.to_world(a2 + perp), dim_col);
                gizmos.line(ap.to_world(b2), ap.to_world(b2 + perp), dim_col);
            }
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
            Tool::Line => gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col),
            Tool::Circle => {
                let r = snap_radius(start.distance(cur), &session.reference_circles, session.snap_dist.max(SNAP));
                let iso = Isometry3d::new(ap.to_world(start), plane_rot);
                gizmos.circle(iso, r, preview_col);
            }
            Tool::Rectangle => {
                let a = Vec2::new(cur.x, start.y);
                let b = Vec2::new(start.x, cur.y);
                for (p, q) in [(start, a), (a, cur), (cur, b), (b, start)] {
                    gizmos.line(ap.to_world(p), ap.to_world(q), preview_col);
                }
            }
            Tool::Select | Tool::Dimension => {}
        }
        draw_marker(&mut gizmos, ap, start, point_col, ms);
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
    cam_q: Query<&GlobalTransform, With<Camera3d>>,
) {
    if session.plane.is_some() || sel.chain.len() < 2 {
        return; // only in view mode, only with a selection
    }
    let Ok(cam) = cam_q.single() else { return };
    let cam_pos = cam.translation();
    // Nudge toward the camera (like body edges) so the highlight sits in front.
    const TOWARD_CAM: f32 = 0.004;
    let nudge = |p: Vec3| p + (cam_pos - p) * TOWARD_CAM;

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
        // And the result must be real combined geometry, not a degenerate weld.
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
