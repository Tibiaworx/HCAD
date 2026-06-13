//! HCAD — Layer 4: the Bevy viewport.
//!
//! Milestone **M1**: pick a reference plane and sketch on it.
//!  - Left-click a reference plane → the camera snaps face-on and enters sketch mode.
//!  - Draw raw (unconstrained) geometry on the plane: lines and circles.
//!  - Constraints/dimensions and the solver arrive at M2.
//!
//! Controls:
//!   View mode:   right-drag = orbit · scroll = zoom · left-click a plane = sketch on it
//!   Sketch mode: L = line tool · C = circle tool · left-click = place points
//!                Esc = cancel current draw, or (again) leave sketch mode
//!
//! Picking and the draw cursor are done with manual ray/plane intersection so the
//! drawing surface is the *infinite* plane, not just the visible quad. The active
//! [`Sketch`] (in `hworks-sketch`) is the data; the gizmos are throwaway rendering.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use hworks_document::{Document, Plane};
use hworks_sketch::{Sketch, SketchEntity};

/// Edge length of the square reference-plane quads (world units).
const PLANE_SIZE: f32 = 8.0;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HCAD — M1: Sketch on a Plane".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .insert_resource(DocRes(Document::with_default_planes()))
        .init_resource::<SketchSession>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                sketch_interaction,
                handle_keys,
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

/// The document (feature tree) held as a Bevy resource. Source of truth.
#[derive(Resource)]
struct DocRes(Document);

/// Which draw tool is active while sketching.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Tool {
    #[default]
    Line,
    Circle,
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

    /// Convert plane-local (u, v) coordinates to a world position.
    fn to_world(&self, uv: Vec2) -> Vec3 {
        self.origin + self.u * uv.x + self.v * uv.y
    }
}

/// The active sketching session. When `plane` is `None` we're in view mode.
#[derive(Resource, Default)]
struct SketchSession {
    plane: Option<ActivePlane>,
    tool: Tool,
    /// First placed point of the in-progress line/circle, if any (plane-local uv).
    pending: Option<Vec2>,
    /// Where the cursor currently projects onto the active plane (for rubber-banding).
    cursor_uv: Option<Vec2>,
    sketch: Sketch,
}

/// Orbit-camera state: spherical position (`yaw`, `pitch`, `radius`) about a `focus`.
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
        Color::srgba(0.85, 0.25, 0.25, 0.18), // Front (XY)
        Color::srgba(0.25, 0.75, 0.30, 0.18), // Top   (XZ)
        Color::srgba(0.25, 0.45, 0.90, 0.18), // Right (YZ)
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

    println!("HCAD M1 ready.");
    println!("  View:   right-drag = orbit, scroll = zoom, left-click a plane to sketch on it.");
    println!("  Sketch: L = line, C = circle, left-click = place points, Esc = cancel / exit.");
}

fn camera_transform(cam: &OrbitCamera) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let translation = cam.focus + rotation * Vec3::new(0.0, 0.0, cam.radius);
    Transform { translation, rotation, ..default() }
}

// ---------------------------------------------------------------------------
// Ray / plane math
// ---------------------------------------------------------------------------

/// Intersect a world-space ray with a plane. Returns `(t, uv)` where `t` is the
/// ray parameter and `uv` is the hit point in the plane's local coordinates.
fn ray_plane(ap: &ActivePlane, ray: &Ray3d) -> Option<(f32, Vec2)> {
    let dir = ray.direction.as_vec3();
    let denom = ap.n.dot(dir);
    if denom.abs() < 1e-6 {
        return None; // ray parallel to plane
    }
    let t = (ap.origin - ray.origin).dot(ap.n) / denom;
    if t <= 0.0 {
        return None; // plane is behind the camera
    }
    let hit = ray.origin + dir * t;
    let d = hit - ap.origin;
    Some((t, Vec2::new(d.dot(ap.u), d.dot(ap.v))))
}

/// Build the world-space ray under the cursor, if available.
fn cursor_ray(
    window: &Window,
    camera: &Camera,
    cam_transform: &GlobalTransform,
) -> Option<Ray3d> {
    let cursor = window.cursor_position()?;
    camera.viewport_to_world(cam_transform, cursor).ok()
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

/// View-mode plane picking + sketch-mode drawing, all via the cursor ray.
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

    // Track where the cursor sits on the active plane (for rubber-band preview).
    let active_uv = session.plane.as_ref().and_then(|ap| ray_plane(ap, &ray).map(|(_, uv)| uv));
    if session.plane.is_some() {
        session.cursor_uv = active_uv;
    }

    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    if session.plane.is_none() {
        // --- View mode: pick the closest reference plane under the cursor. ---
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
            // Snap the camera face-on: look down the plane normal, up = plane v.
            let dist = orbit.radius.max(6.0);
            let eye = ap.origin + ap.n * dist;
            *cam_tf = Transform::from_translation(eye).looking_at(ap.origin, ap.v);

            session.sketch.clear();
            session.pending = None;
            session.cursor_uv = None;
            info!("Entered sketch mode on the {} plane.", ap.name);
            session.plane = Some(ap);
        }
    } else if let Some(uv) = active_uv {
        // --- Sketch mode: place points for the active tool. ---
        match session.tool {
            Tool::Line => {
                if let Some(start) = session.pending.take() {
                    let a = session.sketch.add_point(start.x as f64, start.y as f64);
                    let b = session.sketch.add_point(uv.x as f64, uv.y as f64);
                    session.sketch.add_line(a, b, false);
                } else {
                    session.pending = Some(uv);
                }
            }
            Tool::Circle => {
                if let Some(center) = session.pending.take() {
                    let radius = center.distance(uv);
                    let c = session.sketch.add_point(center.x as f64, center.y as f64);
                    session.sketch.add_circle(c, radius as f64);
                } else {
                    session.pending = Some(uv);
                }
            }
        }
    }
}

/// Tool selection and exit, active only while sketching.
fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut session: ResMut<SketchSession>,
    mut cam_q: Query<(&mut Transform, &OrbitCamera)>,
) {
    if session.plane.is_none() {
        return;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        session.tool = Tool::Line;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        session.tool = Tool::Circle;
        session.pending = None;
    }
    if keys.just_pressed(KeyCode::Escape) {
        if session.pending.is_some() {
            session.pending = None; // cancel the in-progress entity first
        } else {
            // Leave sketch mode and restore the prior orbit view.
            session.plane = None;
            session.cursor_uv = None;
            if let Ok((mut tf, orbit)) = cam_q.single_mut() {
                *tf = camera_transform(orbit);
            }
            info!("Left sketch mode.");
        }
    }
}

/// Right-drag orbit + scroll zoom. Disabled while sketching (camera is locked face-on).
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

// ---------------------------------------------------------------------------
// Rendering (gizmos)
// ---------------------------------------------------------------------------

fn draw_world_axes(mut gizmos: Gizmos) {
    const L: f32 = 5.0;
    gizmos.line(Vec3::ZERO, Vec3::X * L, Color::srgb(1.0, 0.2, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Y * L, Color::srgb(0.2, 1.0, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Z * L, Color::srgb(0.3, 0.5, 1.0));
}

/// Draw the active sketch: a reference grid, committed entities, and a live preview.
fn draw_sketch(mut gizmos: Gizmos, session: Res<SketchSession>) {
    let Some(ap) = &session.plane else { return };

    // Grid + plane outline for drawing reference.
    let grid = Color::srgba(0.55, 0.55, 0.62, 0.18);
    let half = (PLANE_SIZE * 0.5) as i32;
    for i in -half..=half {
        let f = i as f32;
        gizmos.line(ap.to_world(Vec2::new(f, -PLANE_SIZE * 0.5)), ap.to_world(Vec2::new(f, PLANE_SIZE * 0.5)), grid);
        gizmos.line(ap.to_world(Vec2::new(-PLANE_SIZE * 0.5, f)), ap.to_world(Vec2::new(PLANE_SIZE * 0.5, f)), grid);
    }

    let line_col = Color::srgb(0.95, 0.95, 0.25);
    let circle_col = Color::srgb(0.25, 0.9, 0.95);
    let point_col = Color::srgb(1.0, 0.55, 0.15);
    let preview_col = Color::srgba(1.0, 1.0, 1.0, 0.6);
    let plane_rot = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));

    let uv_of = |i: usize| -> Vec2 {
        let p = &session.sketch.points[i];
        Vec2::new(p.x as f32, p.y as f32)
    };

    // Committed entities.
    for e in &session.sketch.entities {
        match e {
            SketchEntity::Line { a, b, .. } => {
                gizmos.line(ap.to_world(uv_of(*a)), ap.to_world(uv_of(*b)), line_col);
            }
            SketchEntity::Circle { center, radius } => {
                let iso = Isometry3d::new(ap.to_world(uv_of(*center)), plane_rot);
                gizmos.circle(iso, *radius as f32, circle_col);
            }
            SketchEntity::Point { .. } => {}
        }
    }

    // Vertex markers.
    for p in &session.sketch.points {
        draw_marker(&mut gizmos, ap, Vec2::new(p.x as f32, p.y as f32), point_col);
    }

    // Live preview of the in-progress entity.
    if let (Some(start), Some(cur)) = (session.pending, session.cursor_uv) {
        match session.tool {
            Tool::Line => {
                gizmos.line(ap.to_world(start), ap.to_world(cur), preview_col);
            }
            Tool::Circle => {
                let iso = Isometry3d::new(ap.to_world(start), plane_rot);
                gizmos.circle(iso, start.distance(cur), preview_col);
            }
        }
        draw_marker(&mut gizmos, ap, start, point_col);
    }
}

/// A small in-plane cross marking a sketch point.
fn draw_marker(gizmos: &mut Gizmos, ap: &ActivePlane, uv: Vec2, color: Color) {
    const S: f32 = 0.08;
    gizmos.line(ap.to_world(uv + Vec2::new(-S, 0.0)), ap.to_world(uv + Vec2::new(S, 0.0)), color);
    gizmos.line(ap.to_world(uv + Vec2::new(0.0, -S)), ap.to_world(uv + Vec2::new(0.0, S)), color);
}
