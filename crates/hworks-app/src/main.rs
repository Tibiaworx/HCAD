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
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use hworks_document::{Document, FeatureKind, Plane};
use hworks_geometry::{cut, extrude_solid, tessellate, union, KSolid, PlaneBasis, Tessellation, TriMesh};
use hworks_sketch::{Constraint, Sketch, SketchEntity};

/// Default boss/cut depth used by the keyboard accelerators (the UI lets you edit it).
const EXTRUDE_DISTANCE: f64 = 2.0;
const PLANE_SIZE: f32 = 8.0;
const SNAP: f32 = 0.18;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HCAD".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .insert_resource(DocRes(Document::with_default_planes()))
        .init_resource::<SketchSession>()
        .init_resource::<Part>()
        .init_resource::<UiState>()
        .init_resource::<UiBlocking>()
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_system)
        .add_systems(
            Update,
            (
                sketch_interaction,
                handle_keys,
                do_solid_op,
                orbit_camera,
                draw_world_axes,
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

/// PropertyManager state for a boss/cut being configured in the UI.
#[derive(Clone)]
struct PendingOp {
    kind: OpKind,
    depth: f32,
}

#[derive(Resource, Default)]
struct UiState {
    pending: Option<PendingOp>,
}

/// True while egui wants the pointer — suppresses viewport drawing/orbit.
#[derive(Resource, Default)]
struct UiBlocking(bool);

#[derive(Resource, Default)]
struct Part {
    solid: Option<KSolid>,
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
    sketch: Sketch,
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
            ..default()
        });
        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(material),
            Transform { translation: ap.origin, rotation, ..default() },
            Name::new(plane.name.clone()),
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
    ));

    println!("HCAD ready — mouse-driven UI. Click a reference plane to start sketching.");
}

fn camera_transform(cam: &OrbitCamera) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let translation = cam.focus + rotation * Vec3::new(0.0, 0.0, cam.radius);
    Transform { translation, rotation, ..default() }
}

// ---------------------------------------------------------------------------
// egui shell
// ---------------------------------------------------------------------------

fn ui_system(
    mut contexts: EguiContexts,
    mut session: ResMut<SketchSession>,
    mut ui_state: ResMut<UiState>,
    mut blocking: ResMut<UiBlocking>,
    doc: Res<DocRes>,
    mut cam_q: Query<(&mut Transform, &mut OrbitCamera)>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let in_sketch = session.plane.is_some();
    let has_profile = session.sketch.closed_loop().is_some();

    // ---- Top toolbar (CommandManager) ----
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("HCAD").strong().size(16.0));
            ui.separator();

            if in_sketch {
                for (tool, name) in [
                    (Tool::Select, "Select"),
                    (Tool::Line, "Line"),
                    (Tool::Circle, "Circle"),
                    (Tool::Rectangle, "Rectangle"),
                ] {
                    if ui.selectable_label(session.tool == tool, name).clicked() {
                        session.tool = tool;
                        session.pending = None;
                    }
                }
                let con = session.construction;
                if ui.selectable_label(con, "Construction").clicked() {
                    session.construction = !con;
                }
                ui.separator();
                ui.add_enabled_ui(has_profile, |ui| {
                    if ui.button("Extrude Boss").clicked() {
                        ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32 });
                    }
                    if ui.button("Extrude Cut").clicked() {
                        ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32 });
                    }
                });
                ui.separator();
                if ui.button("Exit Sketch").clicked() {
                    session.plane = None;
                    session.pending = None;
                    session.drag = None;
                    session.cursor_uv = None;
                    if let Ok((mut tf, orbit)) = cam_q.single_mut() {
                        *tf = camera_transform(&orbit);
                    }
                }
            } else {
                ui.label("Click a reference plane in the viewport to start a sketch.");
            }

            // Right-aligned view controls.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for (name, yaw, pitch) in [
                    ("Iso", 0.8_f32, -0.55_f32),
                    ("Right", 1.5708, 0.0),
                    ("Top", 0.0, -1.553),
                    ("Front", 0.0, 0.0),
                ] {
                    if ui.button(name).clicked() {
                        if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                            orbit.yaw = yaw;
                            orbit.pitch = pitch;
                            *tf = camera_transform(&orbit);
                        }
                    }
                }
                if ui.button("Fit").clicked() {
                    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                        orbit.focus = Vec3::ZERO;
                        orbit.radius = 14.0;
                        *tf = camera_transform(&orbit);
                    }
                }
            });
        });
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
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                for line in doc.0.tree_labels() {
                    ui.label(egui::RichText::new(line).monospace());
                }
            });
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
            ui.label(format!("{} features", doc.0.features.len()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label("mm");
            });
        });
    });

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

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

fn sketch_interaction(
    buttons: Res<ButtonInput<MouseButton>>,
    blocking: Res<UiBlocking>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cam_q: Query<(&Camera, &GlobalTransform, &mut Transform, &OrbitCamera)>,
    doc: Res<DocRes>,
    mut session: ResMut<SketchSession>,
) {
    if blocking.0 {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Ok((camera, cam_gt, mut cam_tf, orbit)) = cam_q.single_mut() else { return };
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
            if let Some((_, ap)) = best {
                let dist = orbit.radius.max(6.0);
                let eye = ap.origin + ap.n * dist;
                *cam_tf = Transform::from_translation(eye).looking_at(ap.origin, ap.v);
                session.sketch.clear();
                session.pending = None;
                session.cursor_uv = None;
                session.drag = None;
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

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut session: ResMut<SketchSession>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
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
            session.pending = None;
        } else {
            session.plane = None;
            session.cursor_uv = None;
            session.drag = None;
            if let Ok((mut tf, orbit)) = cam_q.single_mut() {
                *tf = camera_transform(orbit);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn do_solid_op(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut session: ResMut<SketchSession>,
    mut part: ResMut<Part>,
    mut doc: ResMut<DocRes>,
    existing: Query<Entity, With<SolidPart>>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
    let Some(op) = session.op_request.take() else { return };
    let Some(ap) = session.plane.clone() else { return };

    let Some(loop_idx) = session.sketch.closed_loop() else {
        warn!("Need a single closed profile (a closed loop of lines) for this operation.");
        return;
    };
    let profile: Vec<[f64; 2]> = loop_idx
        .iter()
        .map(|&i| {
            let p = &session.sketch.points[i];
            [p.x, p.y]
        })
        .collect();
    let basis = PlaneBasis {
        origin: [ap.origin.x as f64, ap.origin.y as f64, ap.origin.z as f64],
        u: [ap.u.x as f64, ap.u.y as f64, ap.u.z as f64],
        v: [ap.v.x as f64, ap.v.y as f64, ap.v.z as f64],
        normal: [ap.n.x as f64, ap.n.y as f64, ap.n.z as f64],
    };

    let new_body: Option<KSolid> = match op {
        SolidOp::Boss(d) => match extrude_solid(&profile, &basis, d) {
            Some(solid) => match &part.solid {
                Some(existing) => union(existing, &solid),
                None => Some(solid),
            },
            None => None,
        },
        SolidOp::Cut(d) => match &part.solid {
            Some(existing) => cut(existing, &profile, &basis, d),
            None => {
                warn!("Cut: there is no body yet — extrude a boss first.");
                return;
            }
        },
    };

    let Some(body) = new_body else {
        warn!("The kernel could not complete that operation.");
        return;
    };

    doc.0.add_feature(FeatureKind::Sketch(session.sketch.clone()));
    doc.0.add_feature(match op {
        SolidOp::Boss(d) => FeatureKind::Extrude { distance: d },
        SolidOp::Cut(d) => FeatureKind::Cut { distance: d },
    });

    part.solid = Some(body);
    for e in &existing {
        commands.entity(e).despawn();
    }
    let tess = tessellate(part.solid.as_ref().unwrap(), 0.03);
    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);

    session.plane = None;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
    if let Ok((mut tf, orbit)) = cam_q.single_mut() {
        *tf = camera_transform(orbit);
    }
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

    let edge_mesh = meshes.add(edges_to_bevy(&tess.edges));
    let edge_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.04, 0.04, 0.06),
        unlit: true,
        ..default()
    });
    commands.spawn((Mesh3d(edge_mesh), MeshMaterial3d(edge_material), SolidPart, Name::new("BodyEdges")));
}

fn trimesh_to_bevy(t: TriMesh) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, t.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, t.normals);
    mesh.insert_indices(Indices::U32(t.indices));
    mesh
}

fn edges_to_bevy(edges: &[[[f32; 3]; 2]]) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        positions.push(e[0]);
        positions.push(e[1]);
    }
    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    blocking: Res<UiBlocking>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    session: Res<SketchSession>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    if session.plane.is_some() || blocking.0 {
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

fn draw_world_axes(mut gizmos: Gizmos) {
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
