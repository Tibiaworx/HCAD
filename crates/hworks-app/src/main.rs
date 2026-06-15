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
    cut, extrude_solid, extrude_solid_with_overlap, tessellate, union, KSolid, PlaneBasis,
    Tessellation, TriMesh,
};

/// How far a boss reaches back into the body it's built on, so the union is robust.
const BOSS_OVERLAP: f64 = 0.1;
use hworks_sketch::{point_in_poly, Constraint, Sketch, SketchEntity};

/// Default boss/cut depth used by the keyboard accelerators (the UI lets you edit it).
const EXTRUDE_DISTANCE: f64 = 2.0;
const PLANE_SIZE: f32 = 8.0;
const SNAP: f32 = 0.18;

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "HCAD".into(),
                        // Standard vsync — most compatible present mode on hybrid GPUs.
                        present_mode: bevy::window::PresentMode::Fifo,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    // Render on the integrated GPU, which is wired to the laptop
                    // display. Rendering on the discrete GPU forces a cross-GPU
                    // copy each frame, which flickers on hybrid-graphics laptops.
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        power_preference: PowerPreference::LowPower,
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
}

impl Tool {
    fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Line => "Line",
            Tool::Circle => "Circle",
            Tool::Rectangle => "Rectangle",
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
    drag: Option<usize>,
    dirty: bool,
    op_request: Option<SolidOp>,
    cursor_uv: Option<Vec2>,
    /// Which region of the current sketch is selected for extrude/cut.
    selected_region: Option<usize>,
    /// If editing an existing feature's sketch, its feature index (else a new sketch).
    editing: Option<usize>,
    /// Request to leave sketch mode and commit the sketch to the timeline.
    exit_request: bool,
    /// Request to leave sketch mode and discard the changes (no commit).
    cancel_request: bool,
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
        camera_transform(&cam),
        cam,
        AmbientLight { color: Color::WHITE, brightness: 250.0, ..default() },
        // Disable MSAA: a sample-count mismatch between the 3D pass and the egui
        // overlay pass makes the UI text flicker constantly. Off keeps them in sync.
        Msaa::Off,
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
                if ui.button("Fit").on_hover_text("Zoom to fit").clicked() {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        orbit.focus = Vec3::ZERO;
                        orbit.radius = 14.0;
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
                            ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32 });
                        }
                        if ui.button("Extrude Cut").on_hover_text("Remove material from the sketch (D)").clicked() {
                            if let Some(i) = selected_sketch.filter(|_| !in_sketch) {
                                ui_state.edit_sketch_request = Some(i);
                            }
                            ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32 });
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
            ui.heading(match op.kind {
                OpKind::Boss => "Boss-Extrude",
                OpKind::Cut => "Cut-Extrude",
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Depth (mm):");
                ui.add(egui::DragValue::new(&mut op.depth).speed(0.1).range(0.1..=200.0));
            });
            ui.add_space(8.0);
            let mut keep = true;
            ui.horizontal(|ui| {
                if ui.button("  OK  ").clicked() {
                    session.op_request = Some(match op.kind {
                        OpKind::Boss => SolidOp::Boss(op.depth as f64),
                        OpKind::Cut => SolidOp::Cut(op.depth as f64),
                    });
                    keep = false;
                }
                if ui.button("Cancel").clicked() {
                    keep = false;
                }
            });
            ui_state.pending = if keep { Some(op) } else { None };
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
                        ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32 });
                    }
                    TreeAction::ExtrudeCut(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32 });
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
                        ui.add(egui::DragValue::new(&mut ui_state.edit_depth).speed(0.1).range(0.1..=200.0));
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
                    let sel = session
                        .selected_region
                        .map(|i| (i + 1).to_string())
                        .unwrap_or_else(|| if nreg == 1 { "1".into() } else { "—".into() });
                    ui.label(format!("region {sel}/{nreg}"));
                    if nreg > 1 {
                        ui.label(
                            egui::RichText::new("(Select tool: click a region)")
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
                        orbit.focus = Vec3::ZERO;
                        orbit.radius = 14.0;
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

fn get_or_add_point(sketch: &mut Sketch, uv: Vec2) -> usize {
    nearest_point(sketch, uv, SNAP).unwrap_or_else(|| sketch.add_point(uv.x as f64, uv.y as f64))
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
) {
    if blocking.0 {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_gt, mut cam_tf, mut orbit)) = cam_q.single_mut() else { return };
    let Some(ray) = cursor_ray(window, camera, cam_gt) else { return };

    let active_uv = session.plane.as_ref().and_then(|ap| ray_plane(ap, &ray).map(|(_, uv)| uv));
    if session.plane.is_some() {
        session.cursor_uv = active_uv;
    }

    let just_pressed = buttons.just_pressed(MouseButton::Left);
    let pressed = buttons.pressed(MouseButton::Left);
    let just_released = buttons.just_released(MouseButton::Left);

    if session.plane.is_none() {
        if just_pressed {
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
                orbit.radius = orbit.radius.max(6.0);
                look_along(&mut orbit, ap.origin, ap.n);
                *cam_tf = camera_transform(&orbit);
                session.sketch.clear();
                session.pending = None;
                session.cursor_uv = None;
                session.drag = None;
                session.selected_region = None;
                info!("Sketching on the {} plane.", ap.name);
                session.plane = Some(ap);
            }
        }
        return;
    }

    match session.tool {
        Tool::Select => {
            if just_pressed {
                let hit = active_uv.and_then(|uv| nearest_point(&session.sketch, uv, SNAP));
                session.drag = hit;
                // Clicking empty space inside a region selects that region.
                if hit.is_none() {
                    if let Some(uv) = active_uv {
                        if let Some(r) = region_at(&session.sketch, uv) {
                            session.selected_region = Some(r);
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
            if let Some(uv) = active_uv {
                place_point(&mut session, uv);
            }
        }
        _ => {}
    }

    if session.drag.is_none() && session.dirty {
        session.sketch.solve();
        session.dirty = false;
    }
}

fn place_point(session: &mut SketchSession, uv: Vec2) {
    match session.tool {
        Tool::Line => {
            if let Some(start) = session.pending.take() {
                let a = get_or_add_point(&mut session.sketch, start);
                let b = get_or_add_point(&mut session.sketch, uv);
                session.sketch.add_line(a, b, session.construction);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
            }
        }
        Tool::Circle => {
            if let Some(center) = session.pending.take() {
                let radius = center.distance(uv);
                let c = get_or_add_point(&mut session.sketch, center);
                session.sketch.add_circle(c, radius as f64);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
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
        Tool::Select => {}
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
                        session.selected_region = None;
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

    let regions = session.sketch.regions();
    if regions.is_empty() {
        warn!("Need a closed profile (a loop of lines, or a circle) to extrude.");
        return;
    }
    // Use the selected region, or the only one.
    let idx = session.selected_region.filter(|&i| i < regions.len()).unwrap_or(0);

    if matches!(op, SolidOp::Cut(_)) && part.solid.is_none() {
        warn!("Cut: there is no body yet — extrude a boss first.");
        return;
    }

    history.snapshot(&doc.0);
    let sketch = session.sketch.clone();
    let plane = plane_ref(&ap);
    let kind = match op {
        SolidOp::Boss(d) => FeatureKind::Extrude { sketch, region: idx, plane, distance: d },
        SolidOp::Cut(d) => FeatureKind::Cut { sketch, region: idx, plane, distance: d },
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
    session.selected_region = None;
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
    let (sketch, plane, region) = match &f.kind {
        FeatureKind::Sketch { sketch, plane } => (sketch.clone(), plane.clone(), None),
        FeatureKind::Extrude { sketch, plane, region, .. }
        | FeatureKind::Cut { sketch, plane, region, .. } => {
            (sketch.clone(), plane.clone(), Some(*region))
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
    session.selected_region = region;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
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
        session.selected_region = None;
        return;
    }
    if !session.exit_request {
        return;
    }
    session.exit_request = false;
    let Some(ap) = session.plane.clone() else { return };

    match session.editing {
        Some(i) if i < doc.0.features.len() => {
            history.snapshot(&doc.0);
            let new_sketch = session.sketch.clone();
            let region = session.selected_region.unwrap_or(0);
            match &mut doc.0.features[i].kind {
                FeatureKind::Sketch { sketch, .. } => *sketch = new_sketch,
                FeatureKind::Extrude { sketch, region: r, .. }
                | FeatureKind::Cut { sketch, region: r, .. } => {
                    *sketch = new_sketch;
                    *r = region;
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
    session.selected_region = None;
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
            FeatureKind::Extrude { sketch, region, plane, distance } => {
                let regions = sketch.regions();
                let Some(r) = regions.get(*region).or_else(|| regions.first()) else { continue };
                let basis = basis_from_ref(plane);
                let next = match &body {
                    Some(b) => {
                        extrude_solid_with_overlap(&r.outer, &r.holes, &basis, *distance, BOSS_OVERLAP)
                            .and_then(|boss| union(b, &boss))
                    }
                    None => extrude_solid(&r.outer, &r.holes, &basis, *distance),
                };
                if let Some(s) = next {
                    body = Some(s);
                } else {
                    warn!("Regen: an extrude could not be built.");
                }
            }
            FeatureKind::Cut { sketch, region, plane, distance } => {
                let Some(b) = &body else { continue };
                let regions = sketch.regions();
                let Some(r) = regions.get(*region).or_else(|| regions.first()) else { continue };
                let basis = basis_from_ref(plane);
                // Pick the cut direction from the *current* body, so it stays
                // correct even after upstream edits move things around.
                let centroid = mesh_centroid(&tessellate(b, 0.06).mesh);
                let origin = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
                let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                let signed = if (centroid - origin).dot(n) < 0.0 { -*distance } else { *distance };
                if let Some(s) = cut(b, &r.outer, &r.holes, &basis, signed) {
                    body = Some(s);
                } else {
                    warn!("Regen: a cut could not be built.");
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
    doc: Res<DocRes>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if !ui_state.regen {
        return;
    }
    ui_state.regen = false;

    for e in &existing {
        commands.entity(e).despawn();
    }
    match regenerate(&doc.0) {
        Some(solid) => {
            let tess = tessellate(&solid, 0.03);
            part.mesh = Some(tess.mesh.clone());
            part.edges = tess.edges.clone();
            spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
            part.solid = Some(solid);
        }
        None => {
            part.solid = None;
            part.mesh = None;
            part.edges.clear();
        }
    }
}

/// `PlaneBasis` (kernel-side) from a stored `PlaneRef`.
fn basis_from_ref(p: &PlaneRef) -> PlaneBasis {
    PlaneBasis { origin: p.origin, u: p.u, v: p.v, normal: p.normal }
}

/// Reset the model to an empty part with the three default planes.
fn handle_new_part(
    mut commands: Commands,
    mut ui_state: ResMut<UiState>,
    mut part: ResMut<Part>,
    mut doc: ResMut<DocRes>,
    mut session: ResMut<SketchSession>,
    mut history: ResMut<History>,
    existing: Query<Entity, With<SolidPart>>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
    if !ui_state.new_part {
        return;
    }
    ui_state.new_part = false;
    history.snapshot(&doc.0);
    for e in &existing {
        commands.entity(e).despawn();
    }
    part.solid = None;
    part.mesh = None;
    part.edges.clear();
    session.selected_region = None;
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
    cam_q: Query<&GlobalTransform, With<Camera3d>>,
) {
    let Ok(cam) = cam_q.single() else { return };
    let cam_pos = cam.translation();
    let col = Color::srgb(0.05, 0.05, 0.07);
    const TOWARD_CAM: f32 = 0.0025; // 0.25% of the way to the camera
    for e in &part.edges {
        let a = Vec3::from_array(e[0]);
        let b = Vec3::from_array(e[1]);
        gizmos.line(a + (cam_pos - a) * TOWARD_CAM, b + (cam_pos - b) * TOWARD_CAM, col);
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    blocking: Res<UiBlocking>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    // Orbit/zoom work even while sketching now — only the UI blocks them.
    if blocking.0 {
        return;
    }
    const ORBIT_SENS: f32 = 0.005;
    const ZOOM_SENS: f32 = 0.15;

    for (mut transform, mut cam) in &mut query {
        let mut changed = false;
        if buttons.pressed(MouseButton::Right) && motion.delta != Vec2::ZERO {
            cam.yaw -= motion.delta.x * ORBIT_SENS;
            cam.pitch -= motion.delta.y * ORBIT_SENS;
            cam.pitch = cam.pitch.clamp(-1.54, 1.54);
            changed = true;
        }
        if scroll.delta.y != 0.0 {
            cam.radius = (cam.radius * (1.0 - scroll.delta.y * ZOOM_SENS)).clamp(2.0, 100.0);
            changed = true;
        }
        if changed {
            *transform = camera_transform(&cam);
        }
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

    let uv_of = |i: usize| -> Vec2 {
        let p = &session.sketch.points[i];
        Vec2::new(p.x as f32, p.y as f32)
    };

    for e in &session.sketch.entities {
        match e {
            SketchEntity::Line { a, b, construction: is_con } => {
                let col = if *is_con { construction } else { solid };
                gizmos.line(ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)), col);
            }
            SketchEntity::Circle { center, radius } => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                gizmos.circle(iso, *radius as f32, circle_col);
            }
            SketchEntity::Point { .. } => {}
        }
    }

    for p in &session.sketch.points {
        draw_marker(&mut gizmos, ap, Vec2::new(p.x as f32, p.y as f32), point_col);
    }

    // Highlight the selected region (or the only one) — outer + holes in green.
    let regions = session.sketch.regions();
    if !regions.is_empty() {
        let idx = session
            .selected_region
            .filter(|&i| i < regions.len())
            .or((regions.len() == 1).then_some(0));
        if let Some(i) = idx {
            let region_col = Color::srgb(0.2, 1.0, 0.45);
            let draw_loop = |gizmos: &mut Gizmos, loop_pts: &[[f64; 2]]| {
                let m = loop_pts.len();
                for k in 0..m {
                    let a = Vec2::new(loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                    let b = Vec2::new(loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                    gizmos.line(ap.to_world(a), ap.to_world(b), region_col);
                }
            };
            draw_loop(&mut gizmos, &regions[i].outer);
            for hole in &regions[i].holes {
                draw_loop(&mut gizmos, hole);
            }
        }
    }

    // Snap indicator: ring the point the cursor would attach to.
    if let Some(cur) = session.cursor_uv {
        if let Some(i) = nearest_point(&session.sketch, cur, SNAP) {
            let p = &session.sketch.points[i];
            let iso = Isometry3d::new(ap.to_world(Vec2::new(p.x as f32, p.y as f32)), plane_rot);
            gizmos.circle(iso, 0.16, Color::srgb(0.2, 1.0, 0.45));
        }
    }

    if let (Some(start), Some(cur)) = (session.pending, session.cursor_uv) {
        match session.tool {
            Tool::Line => gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col),
            Tool::Circle => {
                let iso = Isometry3d::new(ap.to_world(start), plane_rot);
                gizmos.circle(iso, start.distance(cur), preview_col);
            }
            Tool::Rectangle => {
                let a = Vec2::new(cur.x, start.y);
                let b = Vec2::new(start.x, cur.y);
                for (p, q) in [(start, a), (a, cur), (cur, b), (b, start)] {
                    gizmos.line(ap.to_world(p), ap.to_world(q), preview_col);
                }
            }
            Tool::Select => {}
        }
        draw_marker(&mut gizmos, ap, start, point_col);
    }
}

fn draw_marker(gizmos: &mut Gizmos, ap: &ActivePlane, uv: Vec2, color: Color) {
    const S: f32 = 0.08;
    gizmos.line(ap.to_world(uv + Vec2::new(-S, 0.0)), ap.to_world(uv + Vec2::new(S, 0.0)), color);
    gizmos.line(ap.to_world(uv + Vec2::new(0.0, -S)), ap.to_world(uv + Vec2::new(0.0, S)), color);
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), region: 0, plane: xy(), distance: 2.0 });
        let solid = regenerate(&doc).expect("regen should produce a body");
        assert_eq!(tessellate(&solid, 0.05).edges.len(), 12);
    }

    #[test]
    fn editing_a_distance_rebuilds_taller() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), region: 0, plane: xy(), distance: 2.0 });
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), region: 0, plane: xy(), distance: 2.0 });
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
            region: 0,
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), region: 0, plane: xy(), distance: 2.0 });
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
        doc.add_feature(FeatureKind::Cut { sketch: pocket, region: 0, plane: xy(), distance: 2.0 });
        let solid = regenerate(&doc).expect("regen with a cut should produce a body");
        assert!(tessellate(&solid, 0.05).edges.len() > 12, "cut should add edges");
    }
}
