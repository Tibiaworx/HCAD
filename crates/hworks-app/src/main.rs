//! HCAD — Layer 4: the Bevy viewport.
//!
//! Milestone **M0**: open a window, render the three standard reference planes
//! from a fresh [`Document`], and let the user orbit/zoom around them.
//!
//! Controls:  hold **right mouse** to orbit · **scroll** to zoom.
//!
//! The planes are rendered *from* the document (the source of truth). The Bevy
//! meshes are throwaway render artifacts — see `DESIGN.md` §2.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use hworks_document::Document;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HCAD — M0: Reference Planes".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.10, 0.11, 0.13)))
        .add_systems(Startup, setup)
        .add_systems(Update, (orbit_camera, draw_world_axes))
        .run();
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // --- The model: a fresh document with the three reference planes. ---
    let doc = Document::with_default_planes();

    // A square mesh shared by all planes (the Rectangle mesh lies in local XY).
    const SIZE: f32 = 8.0;
    let plane_mesh = meshes.add(Rectangle::new(SIZE, SIZE));

    // Distinct, mostly-transparent colors so overlapping planes stay readable.
    let colors = [
        Color::srgba(0.85, 0.25, 0.25, 0.22), // Front (XY) — reddish
        Color::srgba(0.25, 0.75, 0.30, 0.22), // Top   (XZ) — greenish
        Color::srgba(0.25, 0.45, 0.90, 0.22), // Right (YZ) — bluish
    ];

    for (i, (_id, plane)) in doc.planes().enumerate() {
        let u = Vec3::from_array(plane.u);
        let v = Vec3::from_array(plane.v);
        let n = u.cross(v); // plane normal
        // Map the mesh's local axes (X→u, Y→v, Z→normal) into world space.
        let rotation = Quat::from_mat3(&Mat3::from_cols(u, v, n));

        let material = materials.add(StandardMaterial {
            base_color: colors[i % colors.len()],
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,     // visible from both sides
            double_sided: true,  // lit correctly from both sides
            ..default()
        });

        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(material),
            Transform {
                translation: Vec3::from_array(plane.origin),
                rotation,
                ..default()
            },
            Name::new(plane.name.clone()),
        ));
    }

    // --- Lighting ---
    commands.spawn((
        DirectionalLight { illuminance: 6_000.0, shadows_enabled: false, ..default() },
        Transform::from_xyz(6.0, 10.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // --- Camera ---
    let cam = OrbitCamera::default();
    commands.spawn((
        Camera3d::default(),
        camera_transform(&cam),
        cam,
        // In Bevy 0.18 AmbientLight is a per-camera component, not a global resource.
        AmbientLight { color: Color::WHITE, brightness: 250.0, ..default() },
    ));

    println!("HCAD M0 ready — {} reference planes:", doc.planes().count());
    for (_id, p) in doc.planes() {
        println!("  • {}", p.name);
    }
    println!("Controls: right-mouse = orbit, scroll = zoom.");
}

/// Build a Transform that places the camera at its spherical position, looking
/// at the focus point (the camera looks down its local -Z).
fn camera_transform(cam: &OrbitCamera) -> Transform {
    let rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let translation = cam.focus + rotation * Vec3::new(0.0, 0.0, cam.radius);
    Transform { translation, rotation, ..default() }
}

/// Right-drag to orbit, scroll to zoom.
fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut query: Query<(&mut Transform, &mut OrbitCamera)>,
) {
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

/// Draw the world origin axes (X red, Y green, Z blue) as a spatial reference.
fn draw_world_axes(mut gizmos: Gizmos) {
    const L: f32 = 5.0;
    gizmos.line(Vec3::ZERO, Vec3::X * L, Color::srgb(1.0, 0.2, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Y * L, Color::srgb(0.2, 1.0, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Z * L, Color::srgb(0.3, 0.5, 1.0));
}
