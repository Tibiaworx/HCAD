//! HCAD — Layer 4: the Bevy viewport.
//!
//! Milestone **M2**: constraints. The sketch is no longer raw points — it's a
//! constraint system the solver keeps satisfied. New in M2:
//!  - **Rectangle tool** built from 4 lines + horizontal/vertical constraints.
//!  - **Construction lines** (toggle), and endpoint **snapping** (shared points
//!    = implicit coincidence) so geometry connects.
//!  - **Select/drag**: grab a point and the rest of the sketch re-solves live.
//!
//! Controls:
//!   View mode:   right-drag = orbit · scroll = zoom · left-click a plane = sketch
//!   Sketch mode: S = select/drag · L = line · C = circle · R = rectangle
//!                X = toggle construction · left-click = place · Esc = cancel / exit

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::window::PrimaryWindow;
use hworks_document::{Document, FeatureKind, Plane};
use hworks_geometry::{cut, extrude_solid, tessellate, union, KSolid, PlaneBasis, Tessellation, TriMesh};
use hworks_sketch::{Constraint, Sketch, SketchEntity};

/// How far a boss/cut pushes the profile along the plane normal (fixed default).
const EXTRUDE_DISTANCE: f64 = 2.0;

/// Edge length of the square reference-plane quads (world units).
const PLANE_SIZE: f32 = 8.0;
/// How close (world units) a click must be to an existing point to snap/grab it.
const SNAP: f32 = 0.18;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HCAD — M4: Cut + Feature Tree".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .insert_resource(DocRes(Document::with_default_planes()))
        .init_resource::<SketchSession>()
        .init_resource::<Part>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                sketch_interaction,
                handle_keys,
                do_solid_op,
                orbit_camera,
                update_hud,
                update_tree,
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
            Tool::Select => "Select/drag",
            Tool::Line => "Line",
            Tool::Circle => "Circle",
            Tool::Rectangle => "Rectangle",
        }
    }
}

/// A reference plane resolved into world-space vectors, ready for ray math.
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
    /// First placed point of the in-progress entity (plane-local uv).
    pending: Option<Vec2>,
    /// Point index currently being dragged (Select tool).
    drag: Option<usize>,
    /// Re-solve requested after a structural change.
    dirty: bool,
    /// A solid operation was requested (E boss / D cut); consumed by `do_solid_op`.
    op_request: Option<SolidOp>,
    cursor_uv: Option<Vec2>,
    sketch: Sketch,
}

/// A requested boolean solid operation from the active sketch.
#[derive(Clone, Copy)]
enum SolidOp {
    Boss,
    Cut,
}

/// The current accumulated 3D body (a truck B-rep behind the kernel seam).
#[derive(Resource, Default)]
struct Part {
    solid: Option<KSolid>,
}

/// Marks entities that make up generated 3D solids (so they can be cleared later).
#[derive(Component)]
struct SolidPart;

/// Marks the feature-tree panel text.
#[derive(Component)]
struct TreeText;

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
struct HudText;

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
        Color::srgba(0.85, 0.25, 0.25, 0.18),
        Color::srgba(0.25, 0.75, 0.30, 0.18),
        Color::srgba(0.25, 0.45, 0.90, 0.18),
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

    // On-screen heads-up text.
    commands.spawn((
        Text::new(""),
        TextFont { font_size: 15.0, ..default() },
        TextColor(Color::srgb(0.88, 0.90, 0.96)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(10.0),
            ..default()
        },
        HudText,
    ));

    // Feature-tree panel (left side).
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(58.0),
                left: Val::Px(8.0),
                width: Val::Px(250.0),
                padding: UiRect::all(Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.45)),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Feature tree"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgb(0.85, 0.88, 0.95)),
                TreeText,
            ));
        });

    println!("HCAD M4 ready. Draw a closed profile on a plane, then E to boss-extrude or D to cut.");
}

fn camera_transform(cam: &OrbitCamera) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let translation = cam.focus + rotation * Vec3::new(0.0, 0.0, cam.radius);
    Transform { translation, rotation, ..default() }
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

/// Nearest existing point to `uv` within `thresh`, if any.
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

/// Reuse a nearby point (implicit coincidence) or create a new one.
fn get_or_add_point(sketch: &mut Sketch, uv: Vec2) -> usize {
    nearest_point(sketch, uv, SNAP).unwrap_or_else(|| sketch.add_point(uv.x as f64, uv.y as f64))
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

fn sketch_interaction(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cam_q: Query<(&Camera, &GlobalTransform, &mut Transform, &OrbitCamera)>,
    doc: Res<DocRes>,
    mut session: ResMut<SketchSession>,
) {
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

    // ---- View mode: pick a plane to sketch on. ----
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
                info!("Entered sketch mode on the {} plane.", ap.name);
                session.plane = Some(ap);
            }
        }
        return;
    }

    // ---- Sketch mode. ----
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

    // Re-solve after a structural change (never while actively dragging).
    if session.drag.is_none() && session.dirty {
        session.sketch.solve();
        session.dirty = false;
    }
}

/// Handle a click for the active drawing tool.
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
                // Corners: p0 = c0, p1 = (c1.x, c0.y), p2 = c1, p3 = (c0.x, c1.y).
                let p0 = s.add_point(c0.x as f64, c0.y as f64);
                let p1 = s.add_point(c1.x as f64, c0.y as f64);
                let p2 = s.add_point(c1.x as f64, c1.y as f64);
                let p3 = s.add_point(c0.x as f64, c1.y as f64);
                s.add_line(p0, p1, false);
                s.add_line(p1, p2, false);
                s.add_line(p2, p3, false);
                s.add_line(p3, p0, false);
                // Make it a parametric rectangle: bottom/top horizontal, sides vertical.
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
        session.op_request = Some(SolidOp::Boss);
    }
    if keys.just_pressed(KeyCode::KeyD) {
        session.op_request = Some(SolidOp::Cut);
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
            info!("Left sketch mode.");
        }
    }
}

/// Consume a solid-op request (boss/cut): turn the sketch's closed loop into a
/// truck solid, union/subtract it against the current body, record the feature,
/// re-tessellate, and drop back to the 3D view.
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
    let dist = EXTRUDE_DISTANCE;

    // Compute the new body.
    let new_body: Option<KSolid> = match op {
        SolidOp::Boss => match extrude_solid(&profile, &basis, dist) {
            Some(solid) => match &part.solid {
                Some(existing) => union(existing, &solid),
                None => Some(solid),
            },
            None => None,
        },
        SolidOp::Cut => match &part.solid {
            Some(existing) => cut(existing, &profile, &basis, dist),
            None => {
                warn!("Cut: there is no body yet — extrude a boss first.");
                return;
            }
        },
    };

    let Some(body) = new_body else {
        warn!("The kernel could not complete that {} operation.", match op {
            SolidOp::Boss => "boss",
            SolidOp::Cut => "cut",
        });
        return;
    };

    // Record the feature(s) in the timeline.
    doc.0.add_feature(FeatureKind::Sketch(session.sketch.clone()));
    doc.0.add_feature(match op {
        SolidOp::Boss => FeatureKind::Extrude { distance: dist },
        SolidOp::Cut => FeatureKind::Cut { distance: dist },
    });

    // Replace the rendered solid.
    part.solid = Some(body);
    for e in &existing {
        commands.entity(e).despawn();
    }
    let tess = tessellate(part.solid.as_ref().unwrap(), 0.03);
    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);

    info!("Applied {} operation; {} features in tree.", match op {
        SolidOp::Boss => "boss",
        SolidOp::Cut => "cut",
    }, doc.0.features.len());

    // Drop back to the orbit view to inspect the body.
    session.plane = None;
    session.pending = None;
    session.drag = None;
    session.cursor_uv = None;
    if let Ok((mut tf, orbit)) = cam_q.single_mut() {
        *tf = camera_transform(orbit);
    }
}

/// Spawn the shaded solid mesh + a black wireframe overlay for a tessellation.
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

/// Convert our kernel [`TriMesh`] into a Bevy render mesh.
fn trimesh_to_bevy(t: TriMesh) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, t.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, t.normals);
    mesh.insert_indices(Indices::U32(t.indices));
    mesh
}

/// Build a `LineList` mesh from wireframe edge segments.
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
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    session: Res<SketchSession>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
    if session.plane.is_some() {
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

fn update_hud(session: Res<SketchSession>, mut q: Query<&mut Text, With<HudText>>) {
    let Ok(mut text) = q.single_mut() else { return };
    let s = match &session.plane {
        None => "View — right-drag orbit · scroll zoom · left-click a reference plane to sketch".to_string(),
        Some(ap) => {
            let con = if session.construction { " · CONSTRUCTION on" } else { "" };
            format!(
                "Sketch on {} — tool: {}{}\nS select · L line · C circle · R rect · X constr · E boss · D cut · Esc cancel/exit",
                ap.name,
                session.tool.label(),
                con,
            )
        }
    };
    *text = Text::new(s);
}

/// Rebuild the feature-tree panel text from the document timeline.
fn update_tree(doc: Res<DocRes>, mut q: Query<&mut Text, With<TreeText>>) {
    let Ok(mut text) = q.single_mut() else { return };
    let mut s = String::from("Feature tree\n");
    for line in doc.0.tree_labels() {
        s.push('\n');
        s.push_str(&line);
    }
    *text = Text::new(s);
}

// ---------------------------------------------------------------------------
// Rendering (gizmos)
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

    // Live preview of the in-progress entity.
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
