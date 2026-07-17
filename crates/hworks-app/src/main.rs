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
use hworks_document::{Document, FeatureId, FeatureKind, LoftProfile, Plane, PlaneOffset, PlaneRef};
use hworks_geometry::{
    bevel_mesh_and_edges, bevel_mesh_selected, chamfer_mesh, cut_tol, cut_tol_arcs, cut_tool_mesh, difference, extrude_solid, extrude_solid_arcs,
    extrude_solid_with_overlap, extrude_solid_with_overlap_arcs,
    export_step, export_stl, extrude_tool_mesh, loft_mesh, mesh_difference, mesh_tessellation, mesh_to_solid, mesh_union, mirror_mesh, revolve_solid_arcs, revolve_tool_mesh, round_mesh,
    solid_renderable, take_fallback_count, tessellate, threaded_hole, union, union_tol, KSolid, PlaneBasis, Tessellation, TriMesh,
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
    point_in_poly, tessellate_arc, tessellate_arc_slot, tessellate_slot, tessellate_spline, text_contours,
    Constraint, DimAxis, Sketch, SketchEntity,
};

mod text;

/// The HCAD wordmark logo, embedded so it ships in the binary (About dialog).
const LOGO_PNG: &[u8] = include_bytes!("../assets/logo.png");
/// The square gear mark — the wordmark reads poorly at taskbar size, so the OS icon uses just the
/// gear (the logo's "C" glyph on a white square).
const GEAR_PNG: &[u8] = include_bytes!("../assets/gear.png");

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
        // Seamless on by default — build with the robust mesh kernel so shared/coincident
        // walls fuse without a seam. Toggle off in the toolbar for exact B-rep faces.
        .insert_resource({
            // Restore persisted preferences (units, camera, mouse scheme, seamless, edges).
            let s = load_settings();
            UiState {
                seamless: s.seamless,
                unit: s.unit,
                show_tangent_edges: s.show_tangent_edges,
                perspective: s.perspective,
                mouse_scheme: s.mouse_scheme,
                plane_size: PLANE_SIZE,
                ..Default::default()
            }
        })
        .init_resource::<UiBlocking>()
        .init_resource::<FontPreviews>()
        .init_resource::<History>()
        .init_resource::<EdgeSelection>()
        .add_systems(Startup, (setup, open_cli_file))
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
                    apply_thread,
                    do_regenerate,
                    apply_section,
                    set_window_icon,
                    update_projection,
                    fillet_preview,
                    chamfer_preview,
                    mirror_preview,
                    thread_ghost,
                ),
                (
                    handle_new_part,
                    highlight_face,
                    hover_body_edge,
                    sync_ref_planes,
                    scale_ref_planes,
                    sync_ref_images,
                    update_ref_images,
                    update_window_title,
                    update_plane_visibility,
                    update_body_transparency,
                    draw_selected_plane,
                    orbit_camera,
                    animate_camera,
                    draw_world_axes,
                    draw_measure,
                    draw_body_edges,
                    draw_feature_previews,
                    (draw_selected_feature, draw_sketch, persist_settings, draw_section_gizmo),
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
    /// 3-point arc: start, end, then a point the arc passes through.
    Arc,
    Rectangle,
    Slot,
    Polygon,
    Spline,
    Text,
    Dimension,
    Pattern,
    Mirror,
    /// Trim-to-closest: click a line segment to delete it back to the nearest intersections.
    Trim,
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

/// Trim-tool variant.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum TrimMode {
    /// Click a piece → delete it back to the nearest intersections.
    #[default]
    Closest,
    /// Drag a stroke → trim every entity the stroke crosses (paint-to-delete).
    Power,
    /// Pick two lines → trim/extend both so they meet at a clean corner.
    Corner,
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
            Tool::Arc => "Arc",
            Tool::Rectangle => "Rectangle",
            Tool::Slot => "Slot",
            Tool::Polygon => "Polygon",
            Tool::Text => "Text",
            Tool::Spline => "Spline",
            Tool::Dimension => "Dimension",
            Tool::Pattern => "Pattern",
            Tool::Mirror => "Mirror",
            Tool::Trim => "Trim",
        }
    }
}

/// A requested boolean solid operation, carrying its depth.
#[derive(Clone, Copy)]
enum SolidOp {
    /// Boss extrude: `(distance, back, thin, thin_side)` — `back` is the Direction-2 distance
    /// (0 = single direction); `thin` > 0 is a thin-feature wall thickness (0 = solid), `thin_side`
    /// 0=outward/1=inward/2=mid.
    Boss(f64, f64, f64, u8),
    Cut(f64, f64, f64, u8),
    /// Revolve the profile around the picked axis line by this many radians (adds material).
    Revolve(f64),
    /// Revolve, but subtract the swept solid from the body (a lathe groove/bore).
    RevolveCut(f64),
}

#[derive(Clone, Copy, PartialEq)]
enum OpKind {
    Boss,
    Cut,
    Revolve,
    RevolveCut,
}

/// Hide/show eye toggle for a feature-tree row, using the project's SVG icons
/// (icons/visable.svg / invisable.svg, embedded in the binary). Returns true on
/// click. Only offered on visual-only features — planes, sketches, reference
/// images; hiding a solid feature in a boolean chain wouldn't mean anything.
fn eye_button(ui: &mut egui::Ui, hidden: bool) -> bool {
    let src = if hidden {
        egui::include_image!("../../../icons/invisable.svg")
    } else {
        egui::include_image!("../../../icons/visable.svg")
    };
    ui.add(egui::Button::image(egui::Image::new(src).fit_to_exact_size(egui::vec2(20.0, 11.0))).frame(false))
        .on_hover_text(if hidden { "Show" } else { "Hide" })
        .clicked()
}

/// Mouse view-control preset: how orbit and pan map onto the mouse.
#[derive(Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum MouseScheme {
    /// HCAD native: right-drag orbits, middle-drag pans.
    #[default]
    Hcad,
    /// Blender-style: middle-drag orbits, Shift+middle pans.
    Blender,
    /// SolidWorks-style: middle-drag orbits, Ctrl+middle pans.
    SolidWorks,
}

/// User preferences persisted across sessions — `%APPDATA%\HCAD\settings.ron`. Every field has a
/// serde default so adding a preference later still loads old files.
#[derive(Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    unit: Unit,
    #[serde(default = "default_true")]
    seamless: bool,
    #[serde(default)]
    show_tangent_edges: bool,
    #[serde(default)]
    perspective: bool,
    #[serde(default)]
    mouse_scheme: MouseScheme,
}

fn default_true() -> bool {
    true
}

/// `%APPDATA%\HCAD\settings.ron` (falls back to the exe's directory if APPDATA is unset).
fn settings_path() -> std::path::PathBuf {
    std::env::var("APPDATA")
        .map(|d| std::path::PathBuf::from(d).join("HCAD"))
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("settings.ron")
}

fn load_settings() -> Settings {
    std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|t| ron::from_str(&t).ok())
        .unwrap_or_else(|| Settings { seamless: true, ..Default::default() })
}

impl Settings {
    fn from_ui(ui: &UiState) -> Self {
        Settings {
            unit: ui.unit,
            seamless: ui.seamless,
            show_tangent_edges: ui.show_tangent_edges,
            perspective: ui.perspective,
            mouse_scheme: ui.mouse_scheme,
        }
    }
}

/// Persist preferences whenever one changes (compared against the last write — cheap).
fn persist_settings(ui_state: Res<UiState>, mut last: Local<Option<Settings>>) {
    let cur = Settings::from_ui(&ui_state);
    if last.as_ref() == Some(&cur) {
        return;
    }
    let path = settings_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = ron::ser::to_string_pretty(&cur, ron::ser::PrettyConfig::default()) {
        if let Err(e) = std::fs::write(&path, text) {
            warn!("Could not save settings to {}: {e}", path.display());
        }
    }
    *last = Some(cur);
}

/// An action chosen from a feature-tree right-click menu (applied after the
/// tree's immutable borrow ends).
#[derive(Clone, Copy)]
enum TreeAction {
    Select(usize),
    Edit(usize),
    /// Reopen a sketch that holds a Text entity, straight into the Text tool with it selected.
    EditText(usize),
    /// Reopen a Fillet/Chamfer/Mirror/Thread/Loft in its PropertyManager with the stored
    /// parameters loaded; OK updates the feature in place.
    EditPm(usize),
    ExtrudeBoss(usize),
    ExtrudeCut(usize),
    Delete(usize),
    EditImage(usize),
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

/// Display units. The model is always stored in millimetres; this only affects what's shown and
/// typed (1 in = 25.4 mm).
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
enum Unit {
    #[default]
    Mm,
    Inch,
}

impl Unit {
    /// mm × factor → display value.
    fn factor(self) -> f32 {
        match self {
            Unit::Mm => 1.0,
            Unit::Inch => 1.0 / 25.4,
        }
    }
    fn suffix(self) -> &'static str {
        match self {
            Unit::Mm => " mm",
            Unit::Inch => " in",
        }
    }
    fn short(self) -> &'static str {
        match self {
            Unit::Mm => "mm",
            Unit::Inch => "in",
        }
    }
    /// Format a millimetre length in this unit (mm → 2 dp, inch → 4 dp).
    fn fmt(self, mm: f32) -> String {
        match self {
            Unit::Mm => format!("{:.2} mm", mm),
            Unit::Inch => format!("{:.4} in", mm / 25.4),
        }
    }
}

/// Format a millimetre length for a compact label (no suffix): mm → 2 dp, inch → 4 dp.
fn fmt_len_bare(mm: f32, unit: Unit) -> String {
    match unit {
        Unit::Mm => format!("{mm:.2}"),
        Unit::Inch => format!("{:.4}", mm / 25.4),
    }
}

/// A length `DragValue` that stores millimetres but displays/edits in the current unit.
fn unit_drag(ui: &mut egui::Ui, mm: &mut f32, unit: Unit, speed: f64, lo: f32, hi: f32) -> egui::Response {
    let f = unit.factor();
    let mut disp = *mm * f;
    let r = ui.add(egui::DragValue::new(&mut disp).speed(speed * f as f64).range((lo * f)..=(hi * f)).suffix(unit.suffix()));
    if r.changed() {
        *mm = disp / f;
    }
    r
}

/// PropertyManager state for a boss/cut being configured in the UI.
#[derive(Clone)]
struct PendingOp {
    kind: OpKind,
    depth: f32,
    /// Direction 1 "reverse" toggle — extrude/cut against the sketch normal.
    reverse: bool,
    /// Direction 2: when enabled, the prism also extends `depth2` the *opposite* way.
    dir2: bool,
    depth2: f32,
    /// Thin feature: when enabled, sweep a wall of thickness `thin` (mm) instead of a solid.
    /// `thin_side`: 0 = outward, 1 = inward, 2 = mid-plane.
    thin: bool,
    thin_mm: f32,
    thin_side: u8,
}

/// PropertyManager state for creating a reference (construction) plane offset from a base — a datum
/// plane *or* a picked body face — the groundwork for lofts (each profile on its own plane). The
/// base is stored as a full plane so it works for either source; `base_name` is for display.
#[derive(Clone)]
struct PlaneSpec {
    base: ActivePlane,
    base_name: String,
    offset: f32,
    flip: bool,
    /// When editing an existing plane: the feature index to replace in place (else append a new one).
    edit_target: Option<usize>,
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
    /// Save As (always prompt) and Export (STL / STEP) requests.
    save_as_request: bool,
    export_stl_request: bool,
    export_step_request: bool,
    /// The file this part is bound to — Save writes here directly (no dialog) once it's set.
    current_file: Option<std::path::PathBuf>,
    /// Transient status messages (text + remaining seconds) shown as fading toasts, bottom-right.
    toasts: Vec<(String, f32)>,
    /// Whether the About window is open.
    show_about: bool,
    /// Displacement (logical px, +right/+down) of the VISIBLE viewport centre from the window
    /// centre — panels/toolbars overlay the 3D view. Measured from egui's rects each frame;
    /// used by "Normal To" framing so geometry centres in what the user actually sees.
    view_center_offset: (f32, f32),
    /// Perspective camera toggle. Default OFF: CAD works in ORTHOGRAPHIC projection so a
    /// "Normal To" view is measurably true — perspective foreshortens off-centre geometry
    /// (parallax), making circles read elliptical and edges skew.
    perspective: bool,
    /// Mouse view-control preset (Tools → Mouse controls).
    mouse_scheme: MouseScheme,
    /// Section view (SolidWorks-style): visually cut the DISPLAYED body with a plane.
    /// Display-only: the document and regeneration always use the full body.
    section: Option<SectionSpec>,
    /// The section parameters currently baked into the displayed mesh (None = full body shown).
    section_shown: Option<SectionSpec>,
    /// Timeline index of the Fillet/Chamfer being EDITED via its PM (tree double-click / Edit).
    /// While set, the doc is rolled back to just before it (so the preview builds on the
    /// pre-bevel body) and OK updates the feature in place instead of appending a new one.
    editing_feature: Option<usize>,
    /// Set with `edit_sketch_request` to reopen a Text feature straight into the Text tool with the
    /// text entity selected and its parameters loaded into the PM. Consumed by `handle_edit_sketch`.
    edit_as_text: bool,
    /// Request to (re)open a feature's sketch for editing.
    edit_sketch_request: Option<usize>,
    /// Datum plane (by order: 0=Front, 1=Top, 2=Right) currently selected in the tree — shown
    /// highlighted in the viewport, SolidWorks-style. `None` ⇒ no datum plane selected.
    selected_plane: Option<usize>,
    /// Request to start a *fresh* sketch on datum plane N (by order). Consumed by handle_edit_sketch.
    sketch_plane_request: Option<usize>,
    /// Reference-plane creation PropertyManager (offset plane), when open.
    plane_spec: Option<PlaneSpec>,
    /// Display size (edge length) of the reference-plane quads/outlines — adjustable so planes can
    /// be made large enough to see/use on a big part. Defaults to `PLANE_SIZE`.
    plane_size: f32,
    /// Loft PropertyManager: ordered `(sketch feature index, chosen region)` pairs — click sketches
    /// in the tree to add them, click a contour in the viewport to choose its region. `None` ⇒ not lofting.
    loft_spec: Option<Vec<(usize, usize)>>,
    /// True when the open Loft PM is a *cut* (subtract the lofted solid) rather than a boss.
    loft_cut: bool,
    /// Display unit (mm / inch) for all readouts and length inputs.
    unit: Unit,
    /// Measure tool: active flag + the up-to-two picked world points (a 3rd click restarts).
    measuring: bool,
    measure_pts: Vec<Vec3>,
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
    /// The edge loop under the cursor while picking fillet/chamfer edges (world-space polyline +
    /// closed flag), previewed so you can see what a click would grab. Recomputed each frame.
    hover_edge_loop: Option<(Vec<Vec3>, bool)>,
    /// Chamfer tool: the bevel distance the PM is configuring (mirrors the fillet state).
    pending_chamfer: Option<f32>,
    chamfer_shown: Option<f32>,
    chamfer_request: Option<f64>,
    /// Mirror feature: the chosen mirror plane while the PM is open (0=Front, 1=Top,
    /// 2=Right), the plane currently previewed, and a confirmed mirror to append.
    pending_mirror: Option<u8>,
    mirror_shown: Option<u8>,
    mirror_request: Option<u8>,
    /// Hole Genie (threads): the spec being configured, and a confirmed thread to append.
    pending_thread: Option<ThreadSpec>,
    thread_request: Option<ThreadSpec>,
    /// The snapped placement point under the cursor while the Hole Genie PM is open
    /// (`(origin, axis)`), for the hover marker and to anchor on click.
    thread_hover: Option<(Vec3, Vec3)>,
    /// Request to insert a reference image: open a file dialog and place it on the
    /// currently-selected plane (or Front if none). Consumed by the file-IO system.
    insert_image_request: bool,
    /// Request to start a fresh sketch on an arbitrary stored plane (e.g. a reference image's plane,
    /// so you can trace it). Consumed by `handle_edit_sketch`.
    sketch_on_ref: Option<PlaneRef>,
    /// Feature index of the reference image whose PropertyManager is open (`None` ⇒ closed).
    image_edit: Option<usize>,
    /// Lock the image's width:height to its source pixel aspect ratio when editing one dimension.
    image_lock_aspect: bool,
    /// Click-to-calibrate state: the up-to-two picked uv points (in the image plane, mm at the
    /// current scale) and the entered real distance. `Some` ⇒ calibration mode is active.
    image_calib: Option<ImageCalib>,
}

/// Two-point scale calibration for a reference image: pick two points on the picture, type the
/// real distance between them, and the image is scaled so they match.
#[derive(Clone, Default)]
struct ImageCalib {
    /// Picked uv points on the image plane (mm at the current scale); collects up to two.
    pts: Vec<Vec2>,
    /// The real-world distance the two points should be apart (mm), typed by the user.
    target: f32,
    /// Live cursor position on the image plane (uv), for the rubber-band line while picking.
    cursor: Option<Vec2>,
}

/// A standard thread: display name, major diameter (mm), coarse pitch (mm).
/// Thread size tables: (name, major Ø mm, standard pitches mm — COARSE first, then fine(s)).
/// Metric pitches are the ISO coarse/fine series; imperial entries are the UNC/UNF (and UNEF)
/// pitches converted from TPI (pitch = 25.4 / TPI) — shown to the user as TPI.
const METRIC_THREADS: &[(&str, f32, &[f32])] = &[
    ("M2", 2.0, &[0.4, 0.25]),
    ("M2.5", 2.5, &[0.45, 0.35]),
    ("M3", 3.0, &[0.5, 0.35]),
    ("M4", 4.0, &[0.7, 0.5]),
    ("M5", 5.0, &[0.8, 0.5]),
    ("M6", 6.0, &[1.0, 0.75]),
    ("M8", 8.0, &[1.25, 1.0, 0.75]),
    ("M10", 10.0, &[1.5, 1.25, 1.0]),
    ("M12", 12.0, &[1.75, 1.5, 1.25]),
    ("M14", 14.0, &[2.0, 1.5]),
    ("M16", 16.0, &[2.0, 1.5]),
    ("M20", 20.0, &[2.5, 2.0, 1.5]),
];
const IMPERIAL_THREADS: &[(&str, f32, &[f32])] = &[
    ("#4", 2.845, &[0.635, 0.529]),      // 40 UNC, 48 UNF
    ("#6", 3.505, &[0.794, 0.635]),      // 32 UNC, 40 UNF
    ("#8", 4.166, &[0.794, 0.706]),      // 32 UNC, 36 UNF
    ("#10", 4.826, &[1.058, 0.794]),     // 24 UNC, 32 UNF
    ("1/4\"", 6.35, &[1.27, 0.907]),     // 20 UNC, 28 UNF
    ("5/16\"", 7.938, &[1.411, 1.058]),  // 18 UNC, 24 UNF
    ("3/8\"", 9.525, &[1.588, 1.058]),   // 16 UNC, 24 UNF
    ("7/16\"", 11.112, &[1.814, 1.27]),  // 14 UNC, 20 UNF
    ("1/2\"", 12.7, &[1.954, 1.27]),     // 13 UNC, 20 UNF
    ("5/8\"", 15.875, &[2.309, 1.411]),  // 11 UNC, 18 UNF
];

/// Display label for a standard pitch: metric shows mm + coarse/fine, imperial shows TPI + series.
fn pitch_label(metric: bool, k: usize, p: f32) -> String {
    if metric {
        format!("{p:.2} mm ({})", if k == 0 { "coarse" } else { "fine" })
    } else {
        let tpi = (25.4 / p).round() as i32;
        let series = match k {
            0 => "UNC",
            1 => "UNF",
            _ => "UNEF",
        };
        format!("{tpi} TPI ({series})")
    }
}

/// Hole Genie thread state (a threaded hole or an external thread).
#[derive(Clone, PartialEq)]
struct ThreadSpec {
    placed: bool,    // has the user clicked a face to anchor it?
    origin: Vec3,    // hole centre on the face
    axis: Vec3,      // outward face normal (thread runs −axis into the body)
    metric: bool,    // metric vs imperial size table
    size: usize,     // index into the table
    pitch: f32,
    depth: f32,
    internal: bool,  // tap a hole (true) vs thread an existing boss (false)
    rh: bool,        // right-handed
}

impl Default for ThreadSpec {
    fn default() -> Self {
        ThreadSpec {
            placed: false,
            origin: Vec3::ZERO,
            axis: Vec3::Z,
            metric: true,
            size: 5, // M6
            pitch: 1.0,
            depth: 6.0,
            internal: true,
            rh: true,
        }
    }
}

impl ThreadSpec {
    fn table(&self) -> &'static [(&'static str, f32, &'static [f32])] {
        if self.metric { METRIC_THREADS } else { IMPERIAL_THREADS }
    }
    fn major_d(&self) -> f32 {
        self.table().get(self.size).map(|t| t.1).unwrap_or(6.0)
    }
}

/// A `PlaneRef` for one of the three standard reference planes (0=Front, 1=Top, 2=Right),
/// matching `Document::with_default_planes`.
/// Section-view cutting plane: a standard datum plane, slid along its normal and tiltable
/// about its two in-plane axes. Display-only — never touches the document.
#[derive(Clone, Copy, PartialEq)]
struct SectionSpec {
    /// Base plane: 0 = Front (XY), 1 = Top (XZ), 2 = Right (YZ).
    which: u8,
    /// Offset along the (rotated) normal, mm.
    offset: f32,
    /// Keep the other side instead.
    flip: bool,
    /// Tilt about the plane's local u axis, degrees.
    rot_u: f32,
    /// Tilt about the plane's local v axis, degrees.
    rot_v: f32,
}

impl SectionSpec {
    fn new(which: u8) -> Self {
        Self { which, offset: 0.0, flip: false, rot_u: 0.0, rot_v: 0.0 }
    }
}

/// The section plane's world axes (u, v, normal) with its tilt applied.
/// Rotation order: about the base u first, then about the base v — matching how the two
/// PM angles compose, and what the rotation-handle drags accumulate into.
fn section_axes(spec: &SectionSpec) -> (Vec3, Vec3, Vec3) {
    let pr = standard_plane_ref(spec.which);
    let u0 = Vec3::new(pr.u[0] as f32, pr.u[1] as f32, pr.u[2] as f32);
    let v0 = Vec3::new(pr.v[0] as f32, pr.v[1] as f32, pr.v[2] as f32);
    let n0 = Vec3::new(pr.normal[0] as f32, pr.normal[1] as f32, pr.normal[2] as f32);
    let q = Quat::from_axis_angle(v0, spec.rot_v.to_radians()) * Quat::from_axis_angle(u0, spec.rot_u.to_radians());
    (q * u0, q * v0, q * n0)
}

fn standard_plane_ref(which: u8) -> PlaneRef {
    match which {
        1 => PlaneRef { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0], datum: true }, // Top (XZ)
        2 => PlaneRef { origin: [0.0; 3], u: [0.0, 0.0, -1.0], v: [0.0, 1.0, 0.0], normal: [1.0, 0.0, 0.0], datum: true }, // Right (YZ)
        _ => PlaneRef { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0], datum: true }, // Front (XY)
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
    /// Fillet/chamfer boundary seams (from the bevel engine, clipped to the final surface).
    /// A round body's rim keeps these as its only edges, so they draw like real edges and are
    /// always selectable — unlike `tangent_edges`, which on the exact path is every facet line.
    seam_edges: Vec<[[f32; 3]; 2]>,
    /// Full-body seam set stashed while a SECTION VIEW is active (the displayed `seam_edges` are
    /// filtered to the kept side; moving the plane back must be able to restore them).
    seam_backup: Vec<[[f32; 3]; 2]>,
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
    /// True for a fixed datum plane (Front/Top/Right); false for a body-face sketch plane. Carried
    /// into the feature's `PlaneRef` so regeneration knows not to reproject a datum sketch.
    datum: bool,
}

impl ActivePlane {
    fn from_doc(p: &Plane) -> Self {
        let u = Vec3::from_array(p.u);
        let v = Vec3::from_array(p.v);
        Self { name: p.name.clone(), origin: Vec3::from_array(p.origin), u, v, n: u.cross(v), datum: true }
    }
    /// Build from a stored `PlaneRef` (a feature's recorded plane), normalising u/v to a clean basis.
    fn from_ref(p: &PlaneRef) -> Self {
        let u = Vec3::new(p.u[0] as f32, p.u[1] as f32, p.u[2] as f32).normalize_or_zero();
        let v = Vec3::new(p.v[0] as f32, p.v[1] as f32, p.v[2] as f32).normalize_or_zero();
        let origin = Vec3::new(p.origin[0] as f32, p.origin[1] as f32, p.origin[2] as f32);
        Self { name: String::new(), origin, u, v, n: u.cross(v), datum: p.datum }
    }
    fn to_world(&self, uv: Vec2) -> Vec3 {
        self.origin + self.u * uv.x + self.v * uv.y
    }
}

/// Marker on a spawned reference-image quad, tagged with the document feature it mirrors so the
/// sync system can match/despawn it and the update system can refresh its transform/opacity.
#[derive(Component)]
struct RefImageEnt {
    id: FeatureId,
}

#[derive(Resource, Default)]
struct SketchSession {
    plane: Option<ActivePlane>,
    tool: Tool,
    construction: bool,
    /// Cached constraint-status report, keyed by the sketch fingerprint so it's
    /// recomputed only when the sketch actually changes. Drives the
    /// fully/under/over-defined status line and the point coloring.
    dof_cache: Option<(u64, hworks_sketch::DofReport)>,
    /// Cached `sketch.regions()` result, keyed by the sketch fingerprint. The
    /// region builder re-runs the whole planar arrangement, and several per-frame
    /// paths (region fills, hit tests, panel counts) need the same answer — this
    /// makes all but the first call after an edit free. Interior mutability so
    /// read-only systems can refresh it via [`SketchSession::cached_regions`].
    regions_cache: std::sync::Mutex<Option<(u64, std::sync::Arc<Vec<hworks_sketch::Region>>)>>,
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
    /// Trim-tool variant (closest / power / corner).
    trim_mode: TrimMode,
    /// Power-trim: the cursor position last frame (the running stroke).
    power_prev: Option<Vec2>,
    /// Power-trim: the cursor path of the current drag, drawn as a trail.
    power_path: Vec<Vec2>,
    /// Corner-trim: the first line picked (waiting for the second).
    trim_first: Option<usize>,
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
    /// Like `dim_line` but for a slot whose width box is open: clicking a line / edge next
    /// converts it into a distance from that line to the slot's centre line.
    dim_slot: Option<usize>,
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
    /// The sketch line chosen as the **revolve axis** (entity index), while the Revolve
    /// PropertyManager is open. Click a line in the sketch to set it.
    revolve_axis: Option<usize>,
    /// Sketch entities selected (with the Select tool) for applying a constraint.
    selected_entities: Vec<usize>,
    /// Snap/inference points for the entity currently under the cursor (line
    /// midpoint, circle centre + quadrants). Recomputed each frame on hover.
    inference_points: Vec<Vec2>,
    /// User toggle to disable the snap/inference points.
    hide_inference: bool,
    /// Live inference guide lines (uv → uv) to draw dotted this frame — SolidWorks-style
    /// alignment/extension/tangent hints showing why the cursor snapped where it did.
    inference_guides: Vec<(Vec2, Vec2)>,
    /// Sketch point the cursor is currently vertically / horizontally aligned with (inference).
    /// On placing the point these become captured Vertical/Horizontal relations so the alignment
    /// survives later edits. `start_infer` snapshots them at the first click of a two-click tool
    /// so the *start* endpoint also captures its alignment.
    infer_v: Option<usize>,
    infer_h: Option<usize>,
    start_infer: (Option<usize>, Option<usize>),
    /// Relation badges to show next to the cursor this frame (SolidWorks-style hints for why the
    /// cursor snapped: horizontal, vertical, coincident, on-edge, collinear, tangent).
    infer_badges: Vec<InferBadge>,
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
    /// True while dragging the *Direction 2* arrow (sets `depth2`).
    arrow_drag2: bool,
    /// Active section-gizmo rotation drag: (which angle 0 = rot_u / 1 = rot_v, the frozen
    /// world rotation axis, last frame's "clock hand" direction from the gizmo centre).
    section_rot: Option<(u8, Vec3, Vec3)>,
    /// Where on the arrow the section offset drag was grabbed: `offset - t(cursor)` at grab
    /// time, added back during the drag so the plane doesn't jump to the click point.
    section_grab: f32,
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

impl SketchSession {
    /// The active sketch's regions, computed at most once per sketch edit: the
    /// result is cached keyed by the sketch fingerprint, so every per-frame
    /// caller after the first gets the shared `Arc` back for free. Always
    /// consistent with the current sketch (a stale cache recomputes inline).
    fn cached_regions(&self) -> std::sync::Arc<Vec<hworks_sketch::Region>> {
        let fp = sketch_fingerprint(&self.sketch);
        let mut guard = self.regions_cache.lock().unwrap();
        if let Some((cached_fp, regions)) = &*guard {
            if *cached_fp == fp {
                return regions.clone();
            }
        }
        let regions = std::sync::Arc::new(self.sketch.regions());
        *guard = Some((fp, regions.clone()));
        regions
    }
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
    /// When set, the camera eases toward this target (a "snap to view" button) instead of jumping.
    /// Cleared once reached, or the moment the user orbits/pans/zooms by hand.
    anim: Option<CamTarget>,
}

/// A camera pose to glide toward, used by the animated view transitions.
#[derive(Clone, Copy)]
struct CamTarget {
    focus: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { focus: Vec3::ZERO, radius: 12.0, yaw: 0.8, pitch: -0.55, anim: None }
    }
}

impl OrbitCamera {
    /// Begin gliding to a new pose. Yaw/pitch use the shortest angular path, resolved each frame.
    fn animate_to(&mut self, focus: Vec3, radius: f32, yaw: f32, pitch: f32) {
        self.anim = Some(CamTarget { focus, radius, yaw, pitch });
    }
    /// Glide to a new orientation, keeping the current focus and zoom (Front/Top/Right/Iso).
    fn animate_view(&mut self, yaw: f32, pitch: f32) {
        self.animate_to(self.focus, self.radius, yaw, pitch);
    }
}

/// Ease the orbit camera toward its `anim` target (smooth "snap to view"). Exponential smoothing
/// gives a fast-in/slow-out glide that settles in ~0.2s; angles take the shortest way round. Runs
/// right after `orbit_camera`, which clears `anim` on any manual orbit/pan/zoom so a drag wins.
fn animate_camera(time: Res<Time>, mut query: Query<(&mut Transform, &mut OrbitCamera)>) {
    let Ok((mut transform, mut cam)) = query.single_mut() else { return };
    let Some(t) = cam.anim else { return };
    let dt = time.delta_secs();
    // 1 - e^(-dt/τ): frame-rate independent. τ≈0.07 → ~95% there in 0.2s.
    let k = 1.0 - (-dt / 0.07).exp();
    let wrap = |a: f32| (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    let dyaw = wrap(t.yaw - cam.yaw);
    let dpitch = wrap(t.pitch - cam.pitch);
    let dfocus = t.focus - cam.focus;
    let dradius = t.radius - cam.radius;
    // Close enough → snap exactly and stop animating.
    if dyaw.abs() < 1e-3 && dpitch.abs() < 1e-3 && dfocus.length() < 1e-3 && dradius.abs() < 1e-3 {
        cam.focus = t.focus;
        cam.radius = t.radius;
        cam.yaw = t.yaw;
        cam.pitch = t.pitch;
        cam.anim = None;
    } else {
        cam.yaw += dyaw * k;
        cam.pitch += dpitch * k;
        cam.focus += dfocus * k;
        cam.radius += dradius * k;
    }
    *transform = camera_transform(&cam);
}

#[derive(Component)]
struct SolidPart;

/// A reference plane's quad. Visible at part-start, while a datum plane is selected in the tree,
/// or hidden otherwise (see `update_plane_visibility`) — SolidWorks-style.
#[derive(Component)]
struct RefPlane;

/// The datum plane's order (0=Front, 1=Top, 2=Right) — lets the visibility system show just the
/// one selected in the tree.
#[derive(Component)]
struct RefPlaneIdx(usize);

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
) {
    // Overlay gizmos draw in front of the solid so the extrude preview/arrow and the
    // cut-depth indicator are visible through the model.
    gizmo_store.config_mut::<OverlayGizmos>().0.depth_bias = -1.0;
    // Drawn sketch lines are a touch thicker than the grid/markers for visibility, and get a
    // forward depth bias so an active-sketch line drawn ON a body edge renders in FRONT of that
    // black edge (which itself is nudged toward the camera). Without this, a line "run down an
    // edge" snaps into place and then vanishes under the coincident edge.
    {
        let cfg = &mut gizmo_store.config_mut::<ProfileGizmos>().0;
        cfg.line.width = 3.2;
        cfg.depth_bias = -0.4;
    }

    // Reference-plane quads (Front/Top/Right, plus any the user creates later) are spawned by
    // `sync_ref_planes` from the document — keeping one source of truth so New Part / added planes
    // stay in sync.

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

/// Decode the embedded logo and trim its white margins to the wordmark, returning (rgba, w, h) —
/// a tight image for the About dialog (the raw asset is a square with generous whitespace).
fn logo_cropped_rgba() -> Option<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(LOGO_PNG).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            // A pixel is "ink" if it's opaque and not near-white.
            if p[3] > 16 && (p[0] < 220 || p[1] < 220 || p[2] < 220) {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let pad = 14u32;
    let (x0, y0) = (x0.saturating_sub(pad), y0.saturating_sub(pad));
    let (x1, y1) = ((x1 + pad).min(w - 1), (y1 + pad).min(h - 1));
    let (cw, ch) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
    let mut out = Vec::with_capacity(cw * ch * 4);
    for y in y0..=y1 {
        for x in x0..=x1 {
            out.extend_from_slice(&img.get_pixel(x, y).0);
        }
    }
    Some((out, cw, ch))
}

/// Set the OS window/taskbar icon from the embedded HCAD logo (winit — Bevy has no Window.icon
/// field). Runs in Update with a run-once guard: `WinitWindows` isn't populated until the winit
/// event loop has created the window, which is after `Startup`, so it's `Option` + retried.
fn set_window_icon(
    mut done: Local<bool>,
    windows: Option<NonSend<bevy::winit::WinitWindows>>,
    primary: Query<Entity, With<PrimaryWindow>>,
) {
    if *done {
        return;
    }
    let Some(windows) = windows else { return };
    let Ok(entity) = primary.single() else { return };
    let Some(win) = windows.get_window(entity) else { return };
    let Ok(img) = image::load_from_memory(GEAR_PNG) else { return };
    // 256² is plenty for the OS to downscale to 16/32/48; keeps the icon payload small.
    let rgba = img.resize_exact(256, 256, image::imageops::FilterType::Lanczos3).into_rgba8();
    let (w, h) = rgba.dimensions();
    if let Ok(icon) = winit::window::Icon::from_rgba(rgba.into_raw(), w, h) {
        win.set_window_icon(Some(icon));
    }
    *done = true;
}

/// Keep the camera's projection in sync with the zoom and the Perspective toggle. Orthographic is
/// the default (CAD-true views, no parallax): the view height tracks the orbit radius through the
/// same half-FOV formula the perspective camera uses, so toggling projections holds the framing
/// and zoom-to-cursor keeps working (it scales `radius`, which scales the ortho window).
fn update_projection(ui_state: Res<UiState>, mut q: Query<(&OrbitCamera, &mut Projection)>) {
    const VFOV: f32 = std::f32::consts::PI / 4.0;
    for (cam, mut proj) in &mut q {
        if ui_state.perspective {
            if !matches!(*proj, Projection::Perspective(_)) {
                *proj = Projection::from(PerspectiveProjection { near: 0.02, far: 100_000.0, ..default() });
            }
        } else {
            let h = 2.0 * cam.radius.max(0.01) * (VFOV * 0.5).tan();
            *proj = Projection::Orthographic(OrthographicProjection {
                scaling_mode: bevy::camera::ScalingMode::FixedVertical { viewport_height: h },
                near: -100_000.0,
                far: 100_000.0,
                ..OrthographicProjection::default_3d()
            });
        }
    }
}

/// Open a `.hcad` passed on the command line (double-clicking an associated file hands us its
/// path as the first argument). Loads it exactly like File → Open, so the window title, Save
/// binding, and regeneration all behave as if opened from the dialog.
fn open_cli_file(mut doc: ResMut<DocRes>, mut ui_state: ResMut<UiState>) {
    let Some(arg) = std::env::args().nth(1) else { return };
    let path = std::path::PathBuf::from(&arg);
    if !path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("hcad") || e.eq_ignore_ascii_case("ron")) {
        return;
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => match ron::from_str::<Document>(&text) {
            Ok(loaded) => {
                doc.0 = loaded;
                ui_state.current_file = Some(path.clone());
                ui_state.regen = true;
                info!("Opened {} (command line)", path.display());
            }
            Err(e) => {
                warn!("Could not parse {}: {e}", path.display());
                ui_state.last_error = Some(format!("Couldn't open {} — it isn't a valid HCAD part.", path.display()));
            }
        },
        Err(e) => {
            warn!("Could not read {}: {e}", path.display());
            ui_state.last_error = Some(format!("Couldn't read {}: {e}", path.display()));
        }
    }
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

/// Nudge the orbit pivot so a "Normal To" frame lands the geometry in the middle of the *visible*
/// viewport, not the middle of the window. The left panel and the toolbar strip both overlay the
/// 3D view, so the visible centre sits right of AND below the window centre — `offset` is that
/// displacement in logical px (measured from egui's real panel rects each frame, so it tracks a
/// resized panel and any UI change; the old hardcoded panel-width guess drifted top-left).
/// Call *after* `look_along` (needs the final yaw/pitch/radius). `win_h` is the logical window height.
fn recenter_for_panel(cam: &mut OrbitCamera, win_h: f32, offset: (f32, f32)) {
    const VFOV: f32 = std::f32::consts::PI / 4.0; // Bevy PerspectiveProjection default vertical fov
    if win_h <= 1.0 {
        return;
    }
    let world_per_px = 2.0 * cam.radius * (VFOV * 0.5).tan() / win_h;
    let rot = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    let right = rot * Vec3::X;
    let up = rot * Vec3::Y;
    // Aim so the geometry projects onto the VISIBLE centre: offset.0 px right, offset.1 px down.
    cam.focus = cam.focus - right * (offset.0 * world_per_px) + up * (offset.1 * world_per_px);
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

/// A menu button with a **drawn** down-arrow (our font has no ▼ glyph, so a text arrow renders as a
/// box). Trailing space reserves room; the triangle is painted over the button's right edge.
fn flyout_menu<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<Option<R>> {
    let inner = ui.menu_button(format!("{label}    "), add);
    let rect = inner.response.rect;
    let c = egui::pos2(rect.right() - 10.0, rect.center().y);
    let col = if inner.response.hovered() { ui.visuals().strong_text_color() } else { ui.visuals().text_color() };
    ui.painter().add(egui::Shape::convex_polygon(
        vec![
            egui::pos2(c.x - 4.0, c.y - 2.5),
            egui::pos2(c.x + 4.0, c.y - 2.5),
            egui::pos2(c.x, c.y + 3.0),
        ],
        col,
        egui::Stroke::NONE,
    ));
    inner
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
    time: Res<Time>,
    mut logo_tex: Local<Option<egui::TextureHandle>>,
    mut image_loaders: Local<bool>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    let unit = ui_state.unit; // display unit for this frame's readouts/labels

    // While a 3D gizmo arrow is being dragged, drop egui keyboard focus: a focused DragValue
    // (e.g. the PM's depth field) keeps its own text buffer and COMMITS the stale value when
    // focus later drops — snapping an arrow-dragged depth back to where it started.
    if session.arrow_drag || session.arrow_drag2 || session.section_rot.is_some() {
        ctx.memory_mut(|m| {
            if let Some(f) = m.focused() {
                m.surrender_focus(f);
            }
        });
    }

    // Register egui's image loaders once (SVG support for the hide/show eye icons).
    if !*image_loaders {
        egui_extras::install_image_loaders(ctx);
        *image_loaders = true;
    }

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
    let has_profile = !session.cached_regions().is_empty();
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

    // ---- Top toolbar: a classic menu bar row, then quick actions + view controls, then the
    // CommandManager tabs. (The menu bar is kept on its own strip, SolidWorks-style.)
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.add_space(2.0);
        // Row 0: a compact menu-bar strip — tight spacing + small text, old-school, but using the
        // program's own panel background so it blends with the rest of the toolbar (no inset tint).
        egui::Frame::new()
            .inner_margin(egui::Margin { left: 4, right: 4, top: 1, bottom: 1 })
            .show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            // Classic application menu bar: items sit close together, small text, no button frame.
            ui.spacing_mut().item_spacing.x = 1.0;
            ui.spacing_mut().button_padding = egui::vec2(5.0, 1.0);
            ui.style_mut().text_styles.insert(
                egui::TextStyle::Button,
                egui::FontId::new(12.5, egui::FontFamily::Proportional),
            );
            ui.visuals_mut().widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
            ui.visuals_mut().widgets.inactive.bg_stroke = egui::Stroke::NONE;
            // Menus — only real, wired actions (no dead placeholders).
            ui.menu_button("File", |ui| {
                if ui.button("New Part").clicked() { ui_state.new_part = true; ui.close(); }
                if ui.button("Open…").clicked() { ui_state.open_request = true; ui.close(); }
                ui.separator();
                if ui.button("Save").clicked() { ui_state.save_request = true; ui.close(); }
                if ui.button("Save As…").clicked() { ui_state.save_as_request = true; ui.close(); }
                ui.separator();
                ui.menu_button("Export", |ui| {
                    if ui.button("STL… (mesh)").on_hover_text("Triangle mesh for 3D printing — works for any body").clicked() {
                        ui_state.export_stl_request = true;
                        ui.close();
                    }
                    if ui.button("STEP… (B-rep)").on_hover_text("Exact CAD interchange — needs the exact kernel (turn off Seamless; no loft/fillet)").clicked() {
                        ui_state.export_step_request = true;
                        ui.close();
                    }
                });
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(!history.undo.is_empty(), egui::Button::new("Undo")).clicked() { ui_state.undo_request = true; ui.close(); }
                if ui.add_enabled(!history.redo.is_empty(), egui::Button::new("Redo")).clicked() { ui_state.redo_request = true; ui.close(); }
            });
            ui.menu_button("View", |ui| {
                ui.checkbox(&mut ui_state.show_tangent_edges, "Tangent edges");
                if ui.checkbox(&mut ui_state.seamless, "Seamless").changed() { ui_state.regen = true; }
                ui.checkbox(&mut ui_state.perspective, "Perspective")
                    .on_hover_text("Perspective camera (default is orthographic — CAD-true views with no parallax)");
                let mut sec = ui_state.section.is_some();
                if ui
                    .checkbox(&mut sec, "Section view")
                    .on_hover_text("Cut the displayed body with a plane to see inside (display only — the model is untouched)")
                    .changed()
                {
                    ui_state.section = sec.then_some(SectionSpec::new(0));
                }
            });
            ui.menu_button("Insert", |ui| {
                if ui
                    .button("Sketch Picture…")
                    .on_hover_text("Place a reference image on the selected plane (or Front) to trace over — then sketch on the same plane")
                    .clicked()
                {
                    ui_state.insert_image_request = true;
                    ui.close();
                }
            });
            ui.menu_button("Tools", |ui| {
                if ui.selectable_label(ui_state.measuring, "Measure").on_hover_text("Click two points on the body to measure the distance (Esc to stop)").clicked() {
                    ui_state.measuring = !ui_state.measuring;
                    ui_state.measure_pts.clear();
                    ui.close();
                }
                ui.menu_button("Units", |ui| {
                    if ui.selectable_label(ui_state.unit == Unit::Mm, "Millimetres (mm)").clicked() { ui_state.unit = Unit::Mm; ui.close(); }
                    if ui.selectable_label(ui_state.unit == Unit::Inch, "Inches (in)").clicked() { ui_state.unit = Unit::Inch; ui.close(); }
                });
                ui.menu_button("Mouse controls", |ui| {
                    let mut pick = |ui: &mut egui::Ui, scheme: MouseScheme, label: &str, tip: &str| {
                        if ui.selectable_label(ui_state.mouse_scheme == scheme, label).on_hover_text(tip).clicked() {
                            ui_state.mouse_scheme = scheme;
                            ui.close();
                        }
                    };
                    pick(ui, MouseScheme::Hcad, "HCAD (default)", "Right-drag orbits, middle-drag pans, scroll zooms to the cursor");
                    pick(ui, MouseScheme::Blender, "Blender", "Middle-drag orbits, Shift+middle pans, scroll zooms");
                    pick(ui, MouseScheme::SolidWorks, "SolidWorks", "Middle-drag orbits, Ctrl+middle pans, scroll zooms");
                });
            });
            ui.menu_button("Help", |ui| {
                if ui.button("About HCAD").clicked() {
                    ui_state.show_about = true;
                    ui.close();
                }
            });
        });
        }); // close the menu-bar frame
        ui.separator();

        // Row 1: quick actions + view controls.
        ui.horizontal_wrapped(|ui| {
            if ui.button("New").on_hover_text("Clear the model and start over").clicked() {
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
                if ui
                    .selectable_label(ui_state.section.is_some(), "Section")
                    .on_hover_text("Section view: cut the part with a plane (drag the arrow to slide the cut)")
                    .clicked()
                {
                    if ui_state.section.is_some() {
                        ui_state.section = None;
                    } else {
                        ui_state.section = Some(SectionSpec::new(0));
                    }
                }
                ui.separator();
                if ui.button("Fit").on_hover_text("Zoom to fit the part").clicked() {
                    if let Ok((_tf, mut orbit)) = cam_q.single_mut() {
                        let (focus, radius) = fit_view(&part);
                        let (yaw, pitch) = (orbit.yaw, orbit.pitch);
                        orbit.animate_to(focus, radius, yaw, pitch);
                    }
                }
                for (name, yaw, pitch) in [
                    ("Iso", 0.8_f32, -0.55_f32),
                    ("Right", 1.5708, 0.0),
                    ("Top", 0.0, -1.553),
                    ("Front", 0.0, 0.0),
                ] {
                    if ui.button(name).on_hover_text(format!("{name} view")).clicked() {
                        if let Ok((_tf, mut orbit)) = cam_q.single_mut() {
                            orbit.animate_view(yaw, pitch);
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
                        // Arc tool: 3-point arc (start, end, point-on-arc).
                        if ui
                            .selectable_label(session.tool == Tool::Arc, "Arc")
                            .on_hover_text("3-point arc: click start, end, then a point the arc passes through")
                            .clicked()
                        {
                            session.tool = Tool::Arc;
                            session.pending = None;
                            session.pending_b = None;
                        }
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
                        // Trim tool + a ▾ dropdown: closest / power / corner.
                        if ui
                            .selectable_label(session.tool == Tool::Trim, "Trim")
                            .on_hover_text("Trim entities — pick a mode from the ▾ (closest / power / corner)")
                            .clicked()
                        {
                            session.tool = Tool::Trim;
                            session.pending = None;
                            session.spline_pts.clear();
                        }
                        egui::Popup::menu(&dropdown_arrow(ui, "Trim modes")).show(|ui| {
                            let m = session.trim_mode;
                            if ui.selectable_label(m == TrimMode::Closest, "Trim to closest").on_hover_text("Click a piece → delete it back to the nearest intersections").clicked() {
                                session.tool = Tool::Trim;
                                session.trim_mode = TrimMode::Closest;
                            }
                            if ui.selectable_label(m == TrimMode::Power, "Power Trim").on_hover_text("Drag a stroke across entities → trim everything it crosses").clicked() {
                                session.tool = Tool::Trim;
                                session.trim_mode = TrimMode::Power;
                            }
                            if ui.selectable_label(m == TrimMode::Corner, "Corner").on_hover_text("Click two lines → trim/extend both to meet at a corner").clicked() {
                                session.tool = Tool::Trim;
                                session.trim_mode = TrimMode::Corner;
                            }
                        });
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
                    }
                    // (The "click a plane to sketch" prompt now lives in the status bar.)
                }
                Tab::Features => {
                    // SolidWorks-style flyouts: the ~dozen feature commands grouped by what they do
                    // (add material / remove material / bevel / reference), so the strip is a handful of
                    // labelled dropdowns instead of a wall of same-weight buttons.
                    let sketch_count = doc.0.features.iter().filter(|f| matches!(f.kind, FeatureKind::Sketch { .. })).count();
                    let has_mesh = part.mesh.is_some();
                    let can_loft = sketch_count >= 2 && !in_sketch;
                    let start_op = |ui_state: &mut UiState, kind: OpKind, depth: f32| {
                        if let Some(i) = selected_sketch.filter(|_| !in_sketch) {
                            ui_state.edit_sketch_request = Some(i);
                        }
                        ui_state.pending = Some(PendingOp { kind, depth, reverse: false, dir2: false, depth2: 10.0 , thin: false, thin_mm: 2.0, thin_side: 0 });
                    };
                    // Add material.
                    flyout_menu(ui, "Boss", |ui| {
                        if ui.add_enabled(can_extrude, egui::Button::new("Extrude Boss")).on_hover_text("Add material from the sketch (E)").clicked() {
                            start_op(&mut ui_state, OpKind::Boss, EXTRUDE_DISTANCE as f32);
                            ui.close();
                        }
                        if ui.add_enabled(can_extrude, egui::Button::new("Revolve")).on_hover_text("Revolve the profile around a picked axis line (adds material)").clicked() {
                            start_op(&mut ui_state, OpKind::Revolve, 360.0);
                            ui.close();
                        }
                        if ui.add_enabled(can_loft, egui::Button::new("Loft")).on_hover_text("Skin a solid between 2+ sketch profiles — click the sketches in the tree in order").clicked() {
                            ui_state.loft_spec = Some(Vec::new());
                            ui_state.loft_cut = false;
                            ui.close();
                        }
                    });
                    // Remove material.
                    flyout_menu(ui, "Cut", |ui| {
                        if ui.add_enabled(can_extrude, egui::Button::new("Extrude Cut")).on_hover_text("Remove material from the sketch (D)").clicked() {
                            start_op(&mut ui_state, OpKind::Cut, EXTRUDE_DISTANCE as f32);
                            ui.close();
                        }
                        if ui.add_enabled(can_extrude, egui::Button::new("Revolve Cut")).on_hover_text("Revolve the profile around a picked axis line and subtract it (a lathe groove/bore)").clicked() {
                            start_op(&mut ui_state, OpKind::RevolveCut, 360.0);
                            ui.close();
                        }
                        if ui.add_enabled(can_loft && has_mesh, egui::Button::new("Loft Cut")).on_hover_text("Subtract a solid lofted between 2+ profiles from the body (a tapered pocket/bore)").clicked() {
                            ui_state.loft_spec = Some(Vec::new());
                            ui_state.loft_cut = true;
                            ui.close();
                        }
                        if ui.add_enabled(has_mesh, egui::Button::new("Hole Genie")).on_hover_text("Threaded holes: click a face (or a sketch point) to place, pick a size & pitch — taps a hole or threads a boss").clicked() {
                            ui_state.pending_thread = Some(ThreadSpec::default());
                            ui_state.pending_fillet = None;
                            ui_state.pending_chamfer = None;
                            ui_state.pending_mirror = None;
                            ui.close();
                        }
                    });
                    // Bevel / pattern the body.
                    flyout_menu(ui, "Fillet", |ui| {
                        // Seed the edge set from a pre-selected edge (click an edge, then the tool).
                        let seed = |ui_state: &mut UiState| {
                            ui_state.fillet_edges.clear();
                            if edge_sel.chain.len() >= 2 {
                                ui_state.fillet_edges.push(edge_sel.chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect());
                            }
                        };
                        let bevel_ok = has_mesh && !in_sketch;
                        if ui.add_enabled(bevel_ok, egui::Button::new("Fillet")).on_hover_text("Round picked edges by a radius — click edges on the body").clicked() {
                            ui_state.pending_fillet = Some(0.2);
                            ui_state.fillet_shown = None;
                            ui_state.pending_chamfer = None;
                            seed(&mut ui_state);
                            ui.close();
                        }
                        if ui.add_enabled(bevel_ok, egui::Button::new("Chamfer")).on_hover_text("Flat-bevel picked edges by a distance — click edges on the body").clicked() {
                            ui_state.pending_chamfer = Some(0.2);
                            ui_state.chamfer_shown = None;
                            ui_state.pending_fillet = None;
                            seed(&mut ui_state);
                            ui.close();
                        }
                        if ui.add_enabled(bevel_ok, egui::Button::new("Mirror")).on_hover_text("Reflect the whole body across a plane and union it (a symmetric part)").clicked() {
                            ui_state.pending_mirror = Some(0);
                            ui_state.mirror_shown = None;
                            ui_state.pending_fillet = None;
                            ui_state.pending_chamfer = None;
                            ui.close();
                        }
                    });
                    ui.separator();
                    // Reference geometry — always available (you can build a plane with no body yet).
                    if ui.button("Plane").on_hover_text("Create a reference plane offset from a plane or a picked face — then sketch on it (e.g. stacked loft profiles)").clicked() {
                        if let Some((_, p)) = doc.0.planes().next() {
                            ui_state.plane_spec = Some(PlaneSpec { base: ActivePlane::from_doc(p), base_name: p.name.clone(), offset: 10.0, flip: false, edit_target: None });
                        }
                    }
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
                        if ui.small_button("✖").on_hover_text("Remove this edge").clicked() {
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
                    "Click an edge on the body to pick it — Ctrl+click grabs a whole edge loop."
                } else {
                    "Click more edges to add (Ctrl+click for a loop), or a picked edge again to remove."
                })
                .weak()
                .small(),
            );
            ui.separator();
            if commit && n_edges == 0 {
                // Nothing picked → don't apply (would round every edge). Keep the panel open and
                // wait for a selection.
                ui_state.last_error = Some("Select one or more edges to round, then click OK.".into());
                ui_state.pending_fillet = Some(r.max(0.01));
            } else if commit {
                ui_state.fillet_request = Some(r as f64);
                ui_state.pending_fillet = None;
                ui_state.fillet_shown = None;
            } else if cancel {
                ui_state.pending_fillet = None;
                ui_state.fillet_shown = None;
                ui_state.fillet_edges.clear();
                // Cancelling an EDIT restores the timeline (the doc was rolled back to preview
                // against the pre-fillet body) — the original fillet reappears untouched.
                if ui_state.editing_feature.take().is_some() {
                    doc.0.rollback = doc.0.features.len();
                }
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
                        if ui.small_button("✖").on_hover_text("Remove this edge").clicked() {
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
            ui.label(egui::RichText::new("Click edges on the body to bevel them — Ctrl+click grabs a whole edge loop.").weak().small());
            ui.separator();
            let has_edges = !ui_state.fillet_edges.is_empty();
            if commit && !has_edges {
                // Nothing picked → don't apply (would chamfer every edge). Keep the panel open.
                ui_state.last_error = Some("Select one or more edges to bevel, then click OK.".into());
                ui_state.pending_chamfer = Some(d.max(0.01));
            } else if commit {
                ui_state.chamfer_request = Some(d as f64);
                ui_state.pending_chamfer = None;
                ui_state.chamfer_shown = None;
            } else if cancel {
                ui_state.pending_chamfer = None;
                ui_state.chamfer_shown = None;
                ui_state.fillet_edges.clear();
                // Cancelling an EDIT restores the timeline (see the fillet cancel above).
                if ui_state.editing_feature.take().is_some() {
                    doc.0.rollback = doc.0.features.len();
                }
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
                // Cancelling an EDIT restores the timeline (rolled back for the preview).
                if ui_state.editing_feature.take().is_some() {
                    doc.0.rollback = doc.0.features.len();
                }
                ui_state.regen = true;
            } else {
                ui_state.pending_mirror = Some(which);
            }
        }
        if let Some(mut spec) = ui_state.pending_thread.clone() {
            ui.heading("Hole Genie");
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
            // Mode.
            ui.horizontal(|ui| {
                if ui.radio(spec.internal, "Tap hole").clicked() {
                    spec.internal = true;
                }
                if ui.radio(!spec.internal, "Thread boss").clicked() {
                    spec.internal = false;
                }
            });
            // Placement status.
            if spec.placed {
                ui.label(egui::RichText::new("Location: placed ✓").color(egui::Color32::from_rgb(90, 200, 120)));
            } else {
                ui.label(egui::RichText::new("Click a face on the body to place it.").color(egui::Color32::from_rgb(230, 170, 60)));
            }
            ui.separator();
            // Standard table (metric / imperial).
            ui.horizontal(|ui| {
                if ui.radio(spec.metric, "Metric").clicked() && !spec.metric {
                    spec.metric = true;
                    spec.size = spec.size.min(METRIC_THREADS.len() - 1);
                    spec.pitch = spec.table()[spec.size].2[0];
                }
                if ui.radio(!spec.metric, "Imperial").clicked() && spec.metric {
                    spec.metric = false;
                    spec.size = spec.size.min(IMPERIAL_THREADS.len() - 1);
                    spec.pitch = spec.table()[spec.size].2[0];
                }
            });
            spec.size = spec.size.min(spec.table().len() - 1);
            let cur = spec.table()[spec.size].0;
            egui::ComboBox::from_label("Size").selected_text(cur).show_ui(ui, |ui| {
                for (i, (name, _, pitches)) in spec.table().iter().enumerate() {
                    if ui.selectable_label(spec.size == i, *name).clicked() {
                        spec.size = i;
                        spec.pitch = pitches[0]; // coarse default
                    }
                }
            });
            // Standard pitches for the chosen size (coarse/fine; imperial shown as TPI), with a
            // custom field for anything off the chart.
            let pitches = spec.table()[spec.size].2;
            let sel_label = pitches
                .iter()
                .position(|&p| (p - spec.pitch).abs() < 5.0e-3)
                .map(|k| pitch_label(spec.metric, k, pitches[k]))
                .unwrap_or_else(|| format!("Custom {:.2} mm", spec.pitch));
            egui::ComboBox::from_label("Pitch").selected_text(sel_label).show_ui(ui, |ui| {
                for (k, &p) in pitches.iter().enumerate() {
                    if ui.selectable_label((p - spec.pitch).abs() < 5.0e-3, pitch_label(spec.metric, k, p)).clicked() {
                        spec.pitch = p;
                    }
                }
            });
            egui::Grid::new("thread_params").num_columns(2).show(ui, |ui| {
                ui.label("Custom pitch");
                ui.add(egui::DragValue::new(&mut spec.pitch).range(0.1..=10.0).speed(0.01).suffix(" mm"));
                ui.end_row();
                ui.label("Depth");
                ui.add(egui::DragValue::new(&mut spec.depth).range(0.5..=1000.0).speed(0.1).suffix(" mm"));
                ui.end_row();
            });
            // Quick depth presets from the rule-of-thumb engagement lengths.
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Depth:").weak().small());
                for (lbl, m) in [("1.5×d", 1.5_f32), ("2×d", 2.0), ("3×d", 3.0)] {
                    if ui.small_button(lbl).on_hover_text(format!("Set depth to {m}× the major diameter")).clicked() {
                        spec.depth = spec.major_d() * m;
                    }
                }
            });
            if ui.checkbox(&mut spec.rh, "Right-handed").changed() {
            }
            ui.label(egui::RichText::new(format!("Major Ø {:.2} mm", spec.major_d())).weak().small());
            if spec.internal {
                // The pilot hole to drill before tapping (metric rule d − p; a close
                // approximation for unified threads too).
                ui.label(
                    egui::RichText::new(format!("Tap drill Ø {:.2} mm", spec.major_d() - spec.pitch)).weak().small(),
                );
            } else {
                ui.label(
                    egui::RichText::new(format!("Boss to thread should be Ø {:.2} mm", spec.major_d())).weak().small(),
                );
            }
            ui.separator();
            if commit {
                if spec.placed {
                    ui_state.thread_request = Some(spec);
                    ui_state.pending_thread = None;
                } else {
                    ui_state.last_error = Some("Hole Genie: click a face to place the thread first.".into());
                    ui_state.pending_thread = Some(spec);
                }
            } else if cancel {
                ui_state.pending_thread = None;
                // Cancelling an EDIT restores the timeline (rolled back for the ghost preview).
                if ui_state.editing_feature.take().is_some() {
                    doc.0.rollback = doc.0.features.len();
                }
                ui_state.regen = true;
            } else {
                ui_state.pending_thread = Some(spec);
            }
        }
        if let Some(mut spec) = ui_state.section {
            // Section view controls: pick the cutting plane, slide it, tilt it, flip the kept side.
            ui.heading("Section View");
            ui.horizontal(|ui| {
                if ui.button("Done").clicked() {
                    ui_state.section = None;
                }
                ui.checkbox(&mut spec.flip, "Flip side");
            });
            for (k, name) in [(0u8, "Front (XY)"), (1, "Top (XZ)"), (2, "Right (YZ)")] {
                if ui.radio(spec.which == k, name).clicked() {
                    spec.which = k;
                    spec.rot_u = 0.0;
                    spec.rot_v = 0.0;
                }
            }
            ui.horizontal(|ui| {
                ui.label("Offset");
                ui.add(egui::DragValue::new(&mut spec.offset).speed(0.2).suffix(" mm"));
            });
            ui.horizontal(|ui| {
                ui.label("Rotate");
                ui.add(egui::DragValue::new(&mut spec.rot_u).speed(0.5).suffix("°").range(-180.0..=180.0))
                    .on_hover_text("Tilt about the plane's horizontal axis (or drag the gizmo's top/bottom handles)");
                ui.add(egui::DragValue::new(&mut spec.rot_v).speed(0.5).suffix("°").range(-180.0..=180.0))
                    .on_hover_text("Tilt about the plane's vertical axis (or drag the gizmo's side handles)");
                if (spec.rot_u != 0.0 || spec.rot_v != 0.0) && ui.small_button("Reset").clicked() {
                    spec.rot_u = 0.0;
                    spec.rot_v = 0.0;
                }
            });
            ui.label(egui::RichText::new("Display only — modelling always uses the full body.").weak().small());
            ui.separator();
            if ui_state.section.is_some() {
                ui_state.section = Some(spec);
            }
        }
        if let Some(mut spec) = ui_state.plane_spec.clone() {
            // Reference-plane PropertyManager: a plane parallel to a base (a datum plane or a picked
            // body face), offset along its normal. Live preview + drag arrow in the viewport.
            ui.heading("Reference Plane");
            let datums: Vec<(String, ActivePlane)> = doc.0.planes().map(|(_, p)| (p.name.clone(), ActivePlane::from_doc(p))).collect();
            let mut keep = true;
            let mut create = false;
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70))).clicked() {
                    create = true;
                }
                if ui.add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55))).clicked() {
                    keep = false;
                }
            });
            ui.separator();
            ui.label("Reference (offset from)");
            egui::ComboBox::from_id_salt("plane_base").selected_text(&spec.base_name).show_ui(ui, |ui| {
                for (n, ap) in &datums {
                    if ui.selectable_label(spec.base_name == *n, n).clicked() {
                        spec.base = ap.clone();
                        spec.base_name = n.clone();
                    }
                }
            });
            ui.label(egui::RichText::new("…or click a body face in the viewport.").italics().weak().small());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Distance");
                unit_drag(ui, &mut spec.offset, ui_state.unit, 0.5, 0.0, 100_000.0);
            });
            ui.checkbox(&mut spec.flip, "Flip to the other side");
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Plane size");
                let mut sz = ui_state.plane_size.max(1.0);
                if ui.add(egui::DragValue::new(&mut sz).speed(0.5).range(1.0..=100_000.0).suffix(" mm")).on_hover_text("Display size of every reference plane — make it larger to see/use on a big part").changed() {
                    ui_state.plane_size = sz;
                }
            });
            ui.label(egui::RichText::new("Drag the arrow in the viewport to set the offset, or\ntype it above. Double-click the plane in the tree to sketch.").weak().small());

            if create {
                let ap = &spec.base;
                let nrm = ap.n.normalize_or_zero();
                let off = if spec.flip { -spec.offset } else { spec.offset };
                let origin = ap.origin + nrm * off;
                // Editing keeps the existing name; a new plane is the next Plane#.
                let name = match spec.edit_target.and_then(|fi| doc.0.features.get(fi)) {
                    Some(f) => match &f.kind {
                        FeatureKind::Plane(p) => p.name.clone(),
                        _ => format!("Plane{}", doc.0.planes().count().saturating_sub(2)),
                    },
                    None => format!("Plane{}", doc.0.planes().count().saturating_sub(2)),
                };
                let plane = Plane {
                    name,
                    origin: [origin.x, origin.y, origin.z],
                    u: [ap.u.x, ap.u.y, ap.u.z],
                    v: [ap.v.x, ap.v.y, ap.v.z],
                    offset: Some(PlaneOffset {
                        base_origin: [ap.origin.x, ap.origin.y, ap.origin.z],
                        base_u: [ap.u.x, ap.u.y, ap.u.z],
                        base_v: [ap.v.x, ap.v.y, ap.v.z],
                        base_name: spec.base_name.clone(),
                        distance: spec.offset,
                        flip: spec.flip,
                    }),
                };
                match spec.edit_target {
                    Some(fi) if fi < doc.0.features.len() => doc.0.features[fi].kind = FeatureKind::Plane(plane),
                    _ => {
                        doc.0.add_feature(FeatureKind::Plane(plane));
                    }
                }
                ui_state.regen = true; // a moved plane shifts any sketch/feature built on it
                keep = false;
            }
            ui_state.plane_spec = if keep { Some(spec) } else { None };
        }
        // Reference-image PropertyManager: opacity, size (with aspect lock), position, rotation,
        // mirror, and the two-point scale calibration.
        if let Some(idx) = ui_state.image_edit {
            let cur = doc.0.features.get(idx).and_then(|f| match &f.kind {
                FeatureKind::RefImage { px_w, px_h, center, rot, width, height, opacity, flip_h, flip_v, .. } => {
                    Some((*px_w, *px_h, *center, *rot, *width, *height, *opacity, *flip_h, *flip_v))
                }
                _ => None,
            });
            if let Some((px_w, px_h, mut center, mut rot, mut width, mut height, mut opacity, mut flip_h, mut flip_v)) = cur {
                ui.heading("Reference Image");
                let mut close = false;
                let mut delete = false;
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(egui::RichText::new("✔  Done").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70))).clicked() {
                        close = true;
                    }
                    if ui.add(egui::Button::new(egui::RichText::new("🗑  Delete").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55))).clicked() {
                        delete = true;
                    }
                });
                ui.separator();
                if ui
                    .add(egui::Button::new(egui::RichText::new("✎  Sketch on this plane").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(45, 95, 150)))
                    .on_hover_text("Open a sketch on the picture's plane and trace over it")
                    .clicked()
                {
                    if let Some(FeatureKind::RefImage { plane, .. }) = doc.0.features.get(idx).map(|f| &f.kind) {
                        ui_state.sketch_on_ref = Some(plane.clone());
                    }
                    close = true;
                }
                ui.separator();
                ui.label(format!("Source: {px_w}×{px_h} px"));
                ui.add_space(4.0);
                // Opacity.
                ui.horizontal(|ui| {
                    ui.label("Opacity");
                    ui.add(egui::Slider::new(&mut opacity, 0.05..=1.0).fixed_decimals(2));
                });
                ui.separator();
                // Size — width/height, optionally locked to the source pixel aspect ratio.
                let aspect = px_h.max(1) as f64 / px_w.max(1) as f64; // h / w
                ui.checkbox(&mut ui_state.image_lock_aspect, "Lock aspect ratio");
                let lock = ui_state.image_lock_aspect;
                ui.horizontal(|ui| {
                    ui.label("Width ");
                    let mut w = width as f32;
                    if unit_drag(ui, &mut w, ui_state.unit, 0.5, 0.1, 100_000.0).changed() {
                        width = w as f64;
                        if lock {
                            height = width * aspect;
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Height");
                    let mut h = height as f32;
                    if unit_drag(ui, &mut h, ui_state.unit, 0.5, 0.1, 100_000.0).changed() {
                        height = h as f64;
                        if lock {
                            width = height / aspect;
                        }
                    }
                });
                ui.separator();
                // Position (centre on the plane) + rotation.
                ui.horizontal(|ui| {
                    ui.label("Pos U ");
                    let mut u = center[0] as f32;
                    if unit_drag(ui, &mut u, ui_state.unit, 0.5, -100_000.0, 100_000.0).changed() {
                        center[0] = u as f64;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Pos V ");
                    let mut v = center[1] as f32;
                    if unit_drag(ui, &mut v, ui_state.unit, 0.5, -100_000.0, 100_000.0).changed() {
                        center[1] = v as f64;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Rotate");
                    let mut deg = rot.to_degrees() as f32;
                    if ui.add(egui::DragValue::new(&mut deg).speed(1.0).range(-360.0..=360.0).suffix("°")).changed() {
                        rot = (deg as f64).to_radians();
                    }
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut flip_h, "Flip H");
                    ui.checkbox(&mut flip_v, "Flip V");
                });
                ui.separator();
                // Two-point scale calibration.
                match ui_state.image_calib.clone() {
                    None => {
                        if ui.button("📐  Calibrate scale…").on_hover_text("Click two points on the picture, then type the real distance between them").clicked() {
                            ui_state.image_calib = Some(ImageCalib::default());
                        }
                    }
                    Some(mut cal) => {
                        ui.label(egui::RichText::new("Calibrate: click two points on the picture.").strong());
                        ui.label(egui::RichText::new(format!("Picked {}/2 points.", cal.pts.len())).weak().small());
                        let mut apply = false;
                        if cal.pts.len() == 2 {
                            let cur_d = (cal.pts[0] - cal.pts[1]).length();
                            ui.horizontal(|ui| {
                                ui.label("Real distance");
                                unit_drag(ui, &mut cal.target, ui_state.unit, 0.5, 0.01, 1_000_000.0);
                            });
                            ui.label(egui::RichText::new(format!("Currently {} on the picture.", fmt_len_bare(cur_d, ui_state.unit))).weak().small());
                            ui.horizontal(|ui| {
                                if ui.add_enabled(cal.target > 0.0 && cur_d > 1e-4, egui::Button::new("Apply scale")).clicked() {
                                    let k = (cal.target / cur_d) as f64;
                                    width *= k;
                                    height *= k;
                                    apply = true;
                                }
                                if ui.button("Reset points").clicked() {
                                    cal.pts.clear();
                                }
                            });
                        }
                        if ui.button("Cancel calibration").clicked() || apply {
                            ui_state.image_calib = None;
                        } else {
                            ui_state.image_calib = Some(cal);
                        }
                    }
                }
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Sketch on the same plane to trace over the picture.").italics().weak().small());

                // Write the edited values back into the feature.
                if let Some(FeatureKind::RefImage { center: c, rot: r, width: w, height: h, opacity: o, flip_h: fh, flip_v: fv, .. }) =
                    doc.0.features.get_mut(idx).map(|f| &mut f.kind)
                {
                    *c = center;
                    *r = rot;
                    *w = width;
                    *h = height;
                    *o = opacity;
                    *fh = flip_h;
                    *fv = flip_v;
                }
                if delete {
                    history.snapshot(&doc.0);
                    if idx < doc.0.features.len() {
                        doc.0.features.remove(idx);
                        if doc.0.rollback > doc.0.features.len() {
                            doc.0.rollback = doc.0.features.len();
                        }
                    }
                    ui_state.image_edit = None;
                    ui_state.image_calib = None;
                } else if close {
                    ui_state.image_edit = None;
                    ui_state.image_calib = None;
                }
            } else {
                ui_state.image_edit = None;
                ui_state.image_calib = None;
            }
        }
        if let Some(profiles) = ui_state.loft_spec.clone() {
            ui.heading(if ui_state.loft_cut { "Loft Cut" } else { "Loft" });
            let mut keep = true;
            let mut create = false;
            ui.horizontal(|ui| {
                if ui.add_enabled(profiles.len() >= 2, egui::Button::new(egui::RichText::new("✔  OK").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(40, 140, 70))).clicked() {
                    create = true;
                }
                if ui.add(egui::Button::new(egui::RichText::new("✖  Cancel").color(egui::Color32::WHITE)).fill(egui::Color32::from_rgb(170, 55, 55))).clicked() {
                    keep = false;
                }
            });
            ui.separator();
            ui.label("Profiles (skinned in this order):");
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_width(190.0);
                if profiles.is_empty() {
                    ui.colored_label(egui::Color32::from_rgb(230, 170, 90), "Click sketches in the tree to add them.");
                } else {
                    // Sketch ordinal + which contour, for a friendly label.
                    for (n, &(fi, region)) in profiles.iter().enumerate() {
                        let sk_ord = doc.0.features[..fi.min(doc.0.features.len())].iter().filter(|f| matches!(f.kind, FeatureKind::Sketch { .. })).count() + 1;
                        let nreg = match doc.0.features.get(fi).map(|f| &f.kind) {
                            Some(FeatureKind::Sketch { sketch, .. }) => sketch.regions().len(),
                            _ => 0,
                        };
                        let contour = if nreg > 1 { format!("  ·  contour {}/{nreg}", region + 1) } else { String::new() };
                        ui.colored_label(egui::Color32::from_rgb(150, 180, 255), format!("{}.  Sketch{sk_ord}{contour}", n + 1));
                    }
                }
            });
            if !profiles.is_empty() && ui.button("Clear profiles").clicked() {
                ui_state.loft_spec = Some(Vec::new());
            }
            ui.label(egui::RichText::new("Pick ≥2 closed profiles (click sketches in the tree).\nClick a contour in the viewport to choose its region.").weak().small());

            if create {
                let built: Vec<LoftProfile> = profiles
                    .iter()
                    .filter_map(|&(fi, region)| match doc.0.features.get(fi).map(|f| &f.kind) {
                        Some(FeatureKind::Sketch { sketch, plane }) => Some(LoftProfile { sketch: sketch.clone(), plane: plane.clone(), region }),
                        _ => None,
                    })
                    .collect();
                if built.len() >= 2 {
                    history.snapshot(&doc.0);
                    // Editing an existing loft (tree → Edit): update it in place.
                    if let Some(i) = ui_state.editing_feature.take() {
                        if let Some(f) = doc.0.features.get_mut(i) {
                            f.kind = FeatureKind::Loft { profiles: built, cut: ui_state.loft_cut };
                            doc.0.rollback = doc.0.features.len();
                            ui_state.selected = Some(i);
                        }
                    } else {
                        doc.0.add_feature(FeatureKind::Loft { profiles: built, cut: ui_state.loft_cut });
                    }
                    ui_state.regen = true;
                }
                keep = false;
            }
            // Leaving the PM without committing while EDITING a loft → restore the timeline
            // (it was rolled back so profiles previewed against the pre-loft body).
            if !keep && ui_state.editing_feature.take().is_some() {
                doc.0.rollback = doc.0.features.len();
                ui_state.regen = true;
            }
            ui_state.loft_spec = if keep { Some(profiles) } else { None };
        }
        // A datum/construction plane selected in the tree: quick size control + a Sketch button.
        if ui_state.plane_spec.is_none() && ui_state.pending.is_none() && session.plane.is_none() {
            if let Some(order) = ui_state.selected_plane {
                let name = doc.0.planes().nth(order).map(|(_, p)| p.name.clone()).unwrap_or_default();
                ui.heading(format!("Plane: {name}"));
                ui.horizontal(|ui| {
                    ui.label("Size");
                    let mut sz = ui_state.plane_size.max(1.0);
                    if ui.add(egui::DragValue::new(&mut sz).speed(0.5).range(1.0..=100_000.0).suffix(" mm")).on_hover_text("Display size of the reference planes").changed() {
                        ui_state.plane_size = sz;
                    }
                });
                if ui.button("Sketch on this plane").clicked() {
                    ui_state.sketch_plane_request = Some(order);
                }
                ui.separator();
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
                    OpKind::Revolve => "Revolve",
                    OpKind::RevolveCut => "Revolve Cut",
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
                    session.revolve_axis = None;
                }
            });
            ui.separator();

            // From — the sketch plane (only option for now).
            egui::CollapsingHeader::new("From").default_open(true).show(ui, |ui| {
                ui.add_enabled(false, egui::Button::new("Sketch Plane             ▼"));
            });

            let is_rev = matches!(op.kind, OpKind::Revolve | OpKind::RevolveCut);
            // Revolve: an Axis box — click a sketch line to set the revolve axis.
            if is_rev {
                egui::CollapsingHeader::new("Axis").default_open(true).show(ui, |ui| {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_min_width(190.0);
                        match session.revolve_axis {
                            Some(e) => {
                                ui.horizontal(|ui| {
                                    ui.colored_label(egui::Color32::from_rgb(120, 200, 255), format!("Axis: Line {e}"));
                                    if ui.small_button("Clear").clicked() {
                                        session.revolve_axis = None;
                                    }
                                });
                            }
                            None => {
                                ui.colored_label(egui::Color32::from_rgb(230, 170, 90), "Click a line in the sketch to set the axis.");
                            }
                        }
                    });
                });
            }

            // Direction 1 (boss/cut: depth) or Angle (revolve).
            egui::CollapsingHeader::new(if is_rev { "Angle" } else { "Direction 1" }).default_open(true).show(ui, |ui| {
                if !is_rev {
                    ui.add_enabled(false, egui::Button::new("Blind                    ▼"))
                        .on_disabled_hover_text("End condition (only Blind for now)");
                }
                ui.checkbox(&mut op.reverse, if is_rev { "Reverse (spin other way)" } else { "Reverse direction" });
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(if is_rev { "A1" } else { "D1" }).strong());
                    if is_rev {
                        ui.add(egui::DragValue::new(&mut op.depth).speed(1.0).range(1.0..=360.0).suffix("°"));
                    } else {
                        unit_drag(ui, &mut op.depth, ui_state.unit, 0.1, 0.1, 10_000.0);
                    }
                });
            });

            // Direction 2 — extend the boss/cut the opposite way too. (Not for revolve.)
            if !is_rev {
                egui::CollapsingHeader::new("Direction 2").default_open(op.dir2).show(ui, |ui| {
                    ui.checkbox(&mut op.dir2, "Extend the other direction");
                    if op.dir2 {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("D2").strong());
                            unit_drag(ui, &mut op.depth2, ui_state.unit, 0.1, 0.1, 10_000.0);
                        });
                    }
                });
            }
            // Thin Feature — sweep a wall of a chosen thickness instead of filling the profile
            // (a pipe/box shell). Builds with the mesh kernel (forces Seamless on).
            egui::CollapsingHeader::new("Thin Feature").default_open(op.thin).show(ui, |ui| {
                ui.checkbox(&mut op.thin, "Thin feature (wall)");
                if op.thin {
                    ui.horizontal(|ui| {
                        ui.label("Thickness");
                        unit_drag(ui, &mut op.thin_mm, ui_state.unit, 0.1, 0.01, 10_000.0);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Side");
                        egui::ComboBox::from_id_salt("thin_side")
                            .selected_text(match op.thin_side { 1 => "Inward", 2 => "Mid-plane", _ => "Outward" })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut op.thin_side, 0, "Outward");
                                ui.selectable_value(&mut op.thin_side, 1, "Inward");
                                ui.selectable_value(&mut op.thin_side, 2, "Mid-plane");
                            });
                    });
                    ui.label(egui::RichText::new("Wall follows the profile; the region isn't filled.").weak().small());
                }
            });

            // Selected Contours — the closed regions this op will use (empty = all).
            egui::CollapsingHeader::new("Selected Contours").default_open(true).show(ui, |ui| {
                let nreg = session.cached_regions().len();
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
                // Reverse flips the sweep direction (signed distance); Direction 2 adds `back`.
                let d = if op.reverse { -(op.depth as f64) } else { op.depth as f64 };
                let back = if op.dir2 { op.depth2.max(0.0) as f64 } else { 0.0 };
                let thin = if op.thin { op.thin_mm.max(0.001) as f64 } else { 0.0 };
                session.op_request = Some(match op.kind {
                    OpKind::Boss => SolidOp::Boss(d, back, thin, op.thin_side),
                    OpKind::Cut => SolidOp::Cut(d, back, thin, op.thin_side),
                    // For revolve, `depth` carries the angle in degrees.
                    OpKind::Revolve => SolidOp::Revolve((d).to_radians()),
                    OpKind::RevolveCut => SolidOp::RevolveCut((d).to_radians()),
                });
                keep = false;
            }
            ui_state.pending = if keep { Some(op) } else { None };
        } else if in_sketch {
            // Sketch panel: edit dimensions / relations, then Apply to re-solve.
            ui.heading("Sketch");

            // Constraint status, SolidWorks-style: green = fully defined, blue =
            // degrees of freedom remain, red = conflicting relations.
            if let Some((_, rep)) = &session.dof_cache {
                let (txt, col) = if rep.over_defined {
                    let n = rep.conflicting.len();
                    (format!("Over defined — {n} conflicting relation{} (shown red)", if n == 1 { "" } else { "s" }),
                     egui::Color32::from_rgb(235, 90, 90))
                } else if rep.dof == 0 {
                    ("Fully defined".to_string(), egui::Color32::from_rgb(110, 220, 130))
                } else {
                    (format!("Under defined — {} DOF", rep.dof), egui::Color32::from_rgb(110, 170, 255))
                };
                ui.label(egui::RichText::new(txt).color(col).strong())
                    .on_hover_text("Blue points can still move. Anchor the sketch to the origin point and dimension it to fully define it.");
            }

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
                        // All circle measurements are DIAMETER (Ø) — matching the viewport
                        // callouts, dimensions, and the panel's circle fields. (`live_buf`
                        // stays a radius internally; only the display converts.)
                        ui.label(egui::RichText::new("Circle diameter").strong());
                        ui.horizontal(|ui| {
                            let mut dia = session.live_buf * 2.0;
                            let resp = ui.add(
                                egui::DragValue::new(&mut dia).speed(0.1).range(0.02..=20_000.0).prefix("Ø ").suffix(" mm"),
                            );
                            if resp.changed() {
                                session.live_buf = dia * 0.5;
                            }
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
            // Rescue an over-defined sketch: strip redundant/legacy auto-relations while keeping
            // dimensions, structural joins, and one positional pin per point. Old builds captured
            // alignment guides as relations and doubled circle+arc pins — this heals such files.
            ui.horizontal(|ui| {
                if ui
                    .small_button("Clean relations")
                    .on_hover_text(
                        "Remove redundant auto-relations (duplicate circle/arc pins, cross-point \
                         align captures from older versions). Keeps dimensions, coincident joins, \
                         and each line's own horizontal/vertical.",
                    )
                    .clicked()
                {
                    let removed = clean_redundant_relations(&mut session.sketch);
                    if removed > 0 {
                        session.dirty = true;
                        session.needs_apply = true;
                    }
                    info!("Clean relations: removed {removed} redundant constraint(s).");
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
                            | Constraint::SlotWidth { .. }
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
                                | Some(Constraint::PointLineDistance { value, .. })
                                | Some(Constraint::SlotWidth { value, .. }) => {
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
                            // Shown as DIAMETER (Ø) to match the viewport callouts — the panel
                            // showing R while the canvas showed Ø read like two different sizes.
                            let mut dia = *radius * 2.0;
                            if ui
                                .add(egui::DragValue::new(&mut dia).speed(0.1).range(0.02..=20_000.0).prefix("Ø ").suffix(" mm"))
                                .changed()
                            {
                                *radius = dia * 0.5;
                                changed = true;
                            }
                        }
                    });
                }
                if changed {
                    session.needs_apply = true;
                    session.dirty = true; // re-solve so dependent geometry follows the new size
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
                if let Some(ci) = add_point_line_distance(&mut session.sketch, line_entities[0], line_entities[1]) {
                    session.selected_entities.clear();
                    open_dim_edit(&mut session, ci, None);
                    session.needs_apply = true;
                }
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
                // Root node shows the saved part's name (file stem), or "Part" for an unsaved doc.
                let part_name = ui_state
                    .current_file
                    .as_ref()
                    .and_then(|p| p.file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "Part".to_string());
                ui.label(egui::RichText::new(format!("⬡ {part_name}")).strong().size(15.0))
                    .on_hover_text(
                        ui_state
                            .current_file
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "Unsaved part".to_string()),
                    );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let shown = rollback.saturating_sub(nplanes);
                    let total = doc.0.features.len().saturating_sub(nplanes);
                    ui.label(egui::RichText::new(format!("{shown} / {total}")).weak().small())
                        .on_hover_text("Active features (drag the blue rollback bar to change)");
                });
            });
            ui.separator();

            let mut action: Option<TreeAction> = None;
            // Deferred hide/show toggle (the feature loop holds a borrow of the list).
            let mut toggle_hidden: Option<usize> = None;
            // Rows of solid features, with their on-screen rects, for the rollback bar.
            let mut feat_rows: Vec<(usize, egui::Rect)> = Vec::new();

            egui::ScrollArea::vertical().max_height(340.0).show(ui, |ui| {
                // Datum planes + origin, shown flat at the top like SolidWorks. Click a plane to
                // select/show it; double-click (or right-click → Sketch) to sketch on it — works
                // even with a body present, so you can cut/revolve through the part's centre.
                // Snapshot (order, name, feature index, offset) so the borrow ends before we mutate.
                let plane_rows: Vec<(usize, String, Option<usize>, Option<PlaneOffset>, bool)> = doc
                    .0
                    .planes()
                    .enumerate()
                    .map(|(order, (id, p))| {
                        let fi = doc.0.features.iter().position(|f| f.id == *id);
                        let hidden = fi.map_or(false, |i| doc.0.features[i].hidden);
                        (order, p.name.clone(), fi, p.offset.clone(), hidden)
                    })
                    .collect();
                for (order, name, feat_idx, offset, hidden) in plane_rows {
                    let sel = ui_state.selected_plane == Some(order);
                    let resp = ui.horizontal(|ui| {
                        // Eye toggle first, then the row label (hidden rows read dimmer).
                        if let Some(fi) = feat_idx {
                            if eye_button(ui, hidden) {
                                doc.0.features[fi].hidden = !hidden;
                            }
                        }
                        let mut rt = egui::RichText::new(format!("▱  {name} Plane")).weak();
                        if hidden {
                            rt = rt.color(egui::Color32::from_gray(90));
                        }
                        ui.selectable_label(sel, rt)
                    })
                    .inner;
                    if resp.clicked() {
                        ui_state.selected_plane = if sel { None } else { Some(order) };
                        ui_state.selected = None;
                    }
                    if resp.double_clicked() {
                        ui_state.selected_plane = Some(order);
                        ui_state.sketch_plane_request = Some(order);
                    }
                    resp.context_menu(|ui| {
                        if ui.button(if hidden { "Show" } else { "Hide" }).clicked() {
                            if let Some(fi) = feat_idx {
                                doc.0.features[fi].hidden = !hidden;
                            }
                            ui.close();
                        }
                        if ui.button("Sketch on plane").clicked() {
                            ui_state.selected_plane = Some(order);
                            ui_state.sketch_plane_request = Some(order);
                            ui.close();
                        }
                        // Only user-created offset planes carry construction info to edit.
                        if let (Some(off), Some(fi)) = (&offset, feat_idx) {
                            if ui.button("Edit plane").clicked() {
                                let base = ActivePlane {
                                    name: off.base_name.clone(),
                                    origin: Vec3::from_array(off.base_origin),
                                    u: Vec3::from_array(off.base_u),
                                    v: Vec3::from_array(off.base_v),
                                    n: Vec3::from_array(off.base_u).cross(Vec3::from_array(off.base_v)),
                                    datum: true,
                                };
                                ui_state.plane_spec = Some(PlaneSpec { base, base_name: off.base_name.clone(), offset: off.distance, flip: off.flip, edit_target: Some(fi) });
                                ui.close();
                            }
                        }
                    });
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
                        FeatureKind::Sketch { sketch, .. } => {
                            sk += 1;
                            // A sketch made of text reads as a "Text" feature in the tree (SolidWorks
                            // shows Sketch Text as its own node); double-click / Edit reopens it in the
                            // Text tool. Any other sketch stays "Sketch{n}".
                            let text_str = sketch.entities.iter().find_map(|e| match e {
                                SketchEntity::Text { text, .. } => Some(text.clone()),
                                _ => None,
                            });
                            let label = if let Some(t) = &text_str {
                                let first_line = t.lines().next().unwrap_or("");
                                let short: String = first_line.chars().take(18).collect();
                                let ell = if first_line.chars().count() > 18 || t.contains('\n') { "…" } else { "" };
                                format!("[text]   \"{short}{ell}\"")
                            } else {
                                format!("[sketch] Sketch{sk}")
                            };
                            // While the Loft PM is open, a sketch row is highlighted if already a
                            // profile; clicking adds it (or removes it) from the loft's ordered list.
                            let in_loft = ui_state.loft_spec.as_ref().is_some_and(|v| v.iter().any(|(x, _)| *x == i));
                            let resp = ui
                                .horizontal(|ui| {
                                    if eye_button(ui, f.hidden) {
                                        toggle_hidden = Some(i);
                                    }
                                    ui.selectable_label(selected || in_loft, styled(label))
                                })
                                .inner;
                            if resp.clicked() {
                                if let Some(v) = ui_state.loft_spec.as_mut() {
                                    if let Some(pos) = v.iter().position(|(x, _)| *x == i) {
                                        v.remove(pos);
                                    } else {
                                        v.push((i, 0)); // default to the first region; pick a contour in the viewport
                                    }
                                } else {
                                    ui_state.selected = Some(i);
                                }
                            }
                            if resp.double_clicked() {
                                action = Some(if text_str.is_some() { TreeAction::EditText(i) } else { TreeAction::Edit(i) });
                            }
                            resp.context_menu(|ui| {
                                if text_str.is_some() {
                                    if ui.button("Edit text").clicked() {
                                        action = Some(TreeAction::EditText(i));
                                        ui.close();
                                    }
                                } else if ui.button("Edit sketch").clicked() {
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
                            // A text profile makes this an extruded/cut *Text* feature — surface that in
                            // the label and let the child row (or double-click) reopen it in the Text tool.
                            let text_str = match &f.kind {
                                FeatureKind::Extrude { sketch, .. } | FeatureKind::Cut { sketch, .. } => {
                                    sketch.entities.iter().find_map(|e| match e {
                                        SketchEntity::Text { text, .. } => Some(text.clone()),
                                        _ => None,
                                    })
                                }
                                _ => None,
                            };
                            let short = text_str.as_ref().map(|t| {
                                let line = t.lines().next().unwrap_or("");
                                let s: String = line.chars().take(14).collect();
                                let ell = if line.chars().count() > 14 || t.contains('\n') { "…" } else { "" };
                                format!("\"{s}{ell}\"")
                            });
                            let (label, child) = match &f.kind {
                                FeatureKind::Extrude { .. } => {
                                    ex += 1;
                                    let l = match &short {
                                        Some(s) => format!("Text-Extrude{ex}  {s}"),
                                        None => format!("Boss-Extrude{ex}  (h {distance:.1})"),
                                    };
                                    (l, if short.is_some() { format!("Text of Extrude{ex}") } else { format!("Sketch of Extrude{ex}") })
                                }
                                _ => {
                                    ct += 1;
                                    let l = match &short {
                                        Some(s) => format!("Text-Cut{ct}  {s}"),
                                        None => format!("Cut-Extrude{ct}  (h {distance:.1})"),
                                    };
                                    (l, if short.is_some() { format!("Text of Cut{ct}") } else { format!("Sketch of Cut{ct}") })
                                }
                            };
                            let is_text = text_str.is_some();
                            let edit_action = if is_text { TreeAction::EditText(i) } else { TreeAction::Edit(i) };
                            let edit_label = if is_text { "Edit text" } else { "Edit sketch" };
                            let resp = egui::CollapsingHeader::new(styled(label))
                                .id_salt(i)
                                .default_open(false)
                                .show(ui, |ui| {
                                    let child_resp =
                                        ui.selectable_label(false, egui::RichText::new(child).weak());
                                    if child_resp.double_clicked() {
                                        action = Some(edit_action);
                                    }
                                    child_resp.context_menu(|ui| {
                                        if ui.button(edit_label).clicked() {
                                            action = Some(edit_action);
                                            ui.close();
                                        }
                                    });
                                });
                            if resp.header_response.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.header_response.double_clicked() && is_text {
                                action = Some(edit_action);
                            }
                            resp.header_response.context_menu(|ui| {
                                if is_text && ui.button("Edit text").clicked() {
                                    action = Some(edit_action);
                                    ui.close();
                                }
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
                        FeatureKind::Revolve { angle, cut, .. } => {
                            let (label, child) = if *cut {
                                ct += 1;
                                (format!("RevCut{ct}  ({:.0}°)", angle.to_degrees()), format!("Sketch of RevCut{ct}"))
                            } else {
                                ex += 1;
                                (format!("Revolve{ex}  ({:.0}°)", angle.to_degrees()), format!("Sketch of Revolve{ex}"))
                            };
                            let resp = egui::CollapsingHeader::new(styled(format!("{label}")))
                                .id_salt(i)
                                .default_open(false)
                                .show(ui, |ui| {
                                    let child_resp = ui.selectable_label(false, egui::RichText::new(child).weak());
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
                                if ui.button("Edit feature (angle/axis)").clicked() {
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
                            let resp = ui.selectable_label(selected, styled(format!("Fillet  (r {radius:.2}, {scope})")));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.double_clicked() {
                                action = Some(TreeAction::EditPm(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit fillet").clicked() {
                                    action = Some(TreeAction::EditPm(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Chamfer { distance, edges } => {
                            let resp = ui.selectable_label(selected, styled(format!("Chamfer  (d {distance:.2}, {} edge(s))", edges.len())));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.double_clicked() {
                                action = Some(TreeAction::EditPm(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit chamfer").clicked() {
                                    action = Some(TreeAction::EditPm(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Mirror { .. } => {
                            let resp = ui.selectable_label(selected, styled("Mirror".to_string()));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.double_clicked() {
                                action = Some(TreeAction::EditPm(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit mirror").clicked() {
                                    action = Some(TreeAction::EditPm(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Loft { profiles, cut } => {
                            let label = if *cut {
                                ct += 1;
                                format!("LoftCut{ct}  ({} profiles)", profiles.len())
                            } else {
                                ex += 1;
                                format!("Loft{ex}  ({} profiles)", profiles.len())
                            };
                            let resp = ui.selectable_label(selected, styled(label));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.double_clicked() {
                                action = Some(TreeAction::EditPm(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit loft").clicked() {
                                    action = Some(TreeAction::EditPm(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::Thread { major_d, pitch, internal, .. } => {
                            let kind = if *internal { "tap" } else { "ext" };
                            let resp = ui.selectable_label(selected, styled(format!("Thread {kind}  M{major_d:.1}×{pitch:.2}")));
                            if resp.clicked() {
                                ui_state.selected = Some(i);
                            }
                            if resp.double_clicked() {
                                action = Some(TreeAction::EditPm(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button("Edit thread").clicked() {
                                    action = Some(TreeAction::EditPm(i));
                                    ui.close();
                                }
                                if ui.button("Delete feature").clicked() {
                                    action = Some(TreeAction::Delete(i));
                                    ui.close();
                                }
                            });
                            resp.rect
                        }
                        FeatureKind::RefImage { width, height, .. } => {
                            let resp = ui
                                .horizontal(|ui| {
                                    if eye_button(ui, f.hidden) {
                                        toggle_hidden = Some(i);
                                    }
                                    ui.selectable_label(selected, styled(format!("Picture  ({width:.0}×{height:.0})")))
                                })
                                .inner;
                            if resp.clicked() {
                                action = Some(TreeAction::EditImage(i));
                            }
                            resp.context_menu(|ui| {
                                if ui.button(if f.hidden { "Show picture" } else { "Hide picture" }).clicked() {
                                    toggle_hidden = Some(i);
                                    ui.close();
                                }
                                if ui.button("Edit picture").clicked() {
                                    action = Some(TreeAction::EditImage(i));
                                    ui.close();
                                }
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
                if let Some(i) = toggle_hidden {
                    doc.0.features[i].hidden = !doc.0.features[i].hidden;
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
                        // (kind, PM value, reverse). Boss/Cut carry a distance; a revolve carries
                        // its angle in degrees (negative stored angle ⇒ the "spin other way" flag).
                        // (kind, PM depth, reverse, Direction-2 distance).
                        let op = doc.0.features.get(i).and_then(|f| match &f.kind {
                            FeatureKind::Extrude { distance, back, thin, thin_side, .. } => Some((OpKind::Boss, (distance.abs() as f32).max(0.1), *distance < 0.0, *back as f32, *thin, *thin_side)),
                            FeatureKind::Cut { distance, back, thin, thin_side, .. } => Some((OpKind::Cut, (distance.abs() as f32).max(0.1), *distance < 0.0, *back as f32, *thin, *thin_side)),
                            FeatureKind::Revolve { angle, cut, .. } => Some((
                                if *cut { OpKind::RevolveCut } else { OpKind::Revolve },
                                (angle.abs().to_degrees() as f32).clamp(1.0, 360.0),
                                *angle < 0.0,
                                0.0,
                                0.0,
                                0,
                            )),
                            _ => None,
                        });
                        if let Some((kind, depth, reverse, back, thin, thin_side)) = op {
                            ui_state.edit_sketch_request = Some(i);
                            ui_state.pending = Some(PendingOp { kind, depth, reverse, dir2: back > 0.0, depth2: back.max(0.1), thin: thin > 0.0, thin_mm: if thin > 0.0 { thin as f32 } else { 2.0 }, thin_side });
                        } else {
                            ui_state.selected = Some(i);
                        }
                    }
                    TreeAction::Edit(i) => ui_state.edit_sketch_request = Some(i),
                    TreeAction::EditText(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.edit_as_text = true;
                    }
                    TreeAction::EditPm(i) => {
                        // Reopen the Fillet/Chamfer PM with the stored size and picked edges. The
                        // doc rolls back to just before the feature so the preview (and any edge
                        // re-picking) works against the PRE-bevel body; OK updates it in place.
                        if let Some(f) = doc.0.features.get(i) {
                            match &f.kind {
                                FeatureKind::Fillet { radius, edges } => {
                                    ui_state.pending_fillet = Some(*radius as f32);
                                    ui_state.pending_chamfer = None;
                                    ui_state.fillet_edges = edges.clone();
                                    ui_state.fillet_shown = None;
                                    ui_state.editing_feature = Some(i);
                                    doc.0.rollback = i;
                                    ui_state.regen = true;
                                }
                                FeatureKind::Chamfer { distance, edges } => {
                                    ui_state.pending_chamfer = Some(*distance as f32);
                                    ui_state.pending_fillet = None;
                                    ui_state.fillet_edges = edges.clone();
                                    ui_state.chamfer_shown = None;
                                    ui_state.editing_feature = Some(i);
                                    doc.0.rollback = i;
                                    ui_state.regen = true;
                                }
                                FeatureKind::Mirror { plane } => {
                                    // Recover which datum plane from the stored normal.
                                    let n = plane.normal;
                                    let which = if n[1].abs() > 0.9 { 1 } else if n[0].abs() > 0.9 { 2 } else { 0 };
                                    ui_state.pending_mirror = Some(which);
                                    ui_state.mirror_shown = None;
                                    ui_state.editing_feature = Some(i);
                                    doc.0.rollback = i;
                                    ui_state.regen = true;
                                }
                                FeatureKind::Thread { origin, axis, major_d, pitch, depth, internal, rh } => {
                                    // Rebuild the spec: keep the anchored placement, find the size
                                    // table entry nearest the stored major diameter.
                                    let mut spec = ThreadSpec {
                                        placed: true,
                                        origin: Vec3::new(origin[0] as f32, origin[1] as f32, origin[2] as f32),
                                        axis: Vec3::new(axis[0] as f32, axis[1] as f32, axis[2] as f32),
                                        pitch: *pitch as f32,
                                        depth: *depth as f32,
                                        internal: *internal,
                                        rh: *rh,
                                        ..Default::default()
                                    };
                                    let (mut best, mut metric, mut size) = (f32::MAX, true, 3usize);
                                    for (m, table) in [(true, METRIC_THREADS), (false, IMPERIAL_THREADS)] {
                                        for (k, t) in table.iter().enumerate() {
                                            let d = (t.1 - *major_d as f32).abs();
                                            if d < best {
                                                best = d;
                                                metric = m;
                                                size = k;
                                            }
                                        }
                                    }
                                    spec.metric = metric;
                                    spec.size = size;
                                    ui_state.pending_thread = Some(spec);
                                    ui_state.editing_feature = Some(i);
                                    doc.0.rollback = i;
                                    ui_state.regen = true;
                                }
                                FeatureKind::Loft { profiles, cut } => {
                                    // Map each stored profile back to its timeline sketch by
                                    // fingerprint; unmatched ones are re-picked by the user.
                                    let spec: Vec<(usize, usize)> = profiles
                                        .iter()
                                        .filter_map(|p| {
                                            let fp = sketch_fingerprint(&p.sketch);
                                            doc.0.features.iter().position(|f| matches!(&f.kind, FeatureKind::Sketch { sketch, .. } if sketch_fingerprint(sketch) == fp)).map(|fi| (fi, p.region))
                                        })
                                        .collect();
                                    ui_state.loft_cut = *cut;
                                    ui_state.loft_spec = Some(spec);
                                    ui_state.editing_feature = Some(i);
                                    doc.0.rollback = i;
                                    ui_state.regen = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    TreeAction::ExtrudeBoss(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.pending = Some(PendingOp { kind: OpKind::Boss, depth: EXTRUDE_DISTANCE as f32, reverse: false, dir2: false, depth2: 10.0 , thin: false, thin_mm: 2.0, thin_side: 0 });
                    }
                    TreeAction::ExtrudeCut(i) => {
                        ui_state.edit_sketch_request = Some(i);
                        ui_state.pending = Some(PendingOp { kind: OpKind::Cut, depth: EXTRUDE_DISTANCE as f32, reverse: false, dir2: false, depth2: 10.0 , thin: false, thin_mm: 2.0, thin_side: 0 });
                    }
                    TreeAction::Delete(i) => {
                        if i < doc.0.features.len() {
                            history.snapshot(&doc.0);
                            doc.0.features.remove(i);
                            if doc.0.rollback > doc.0.features.len() {
                                doc.0.rollback = doc.0.features.len();
                            }
                            ui_state.selected = None;
                            ui_state.image_edit = None;
                            ui_state.regen = true;
                        }
                    }
                    TreeAction::EditImage(i) => {
                        ui_state.image_edit = Some(i);
                        ui_state.image_lock_aspect = true;
                        ui_state.image_calib = None;
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
                Some(uv) => {
                    let f = ui_state.unit.factor();
                    ui.label(format!("x {:.3}  y {:.3}", uv.x * f, uv.y * f))
                }
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
                let nreg = session.cached_regions().len();
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
            // Context hint (moved out of the toolbar): what to do next when nothing is actionable yet.
            if !in_sketch {
                let hint = match ui_state.active_tab {
                    Tab::Sketch => Some("Click a plane or face to start a sketch."),
                    Tab::Features if !can_extrude => Some("Select a sketch, or draw a closed profile, to make a feature."),
                    _ => None,
                };
                if let Some(h) = hint {
                    ui.separator();
                    ui.label(egui::RichText::new(h).italics().weak());
                }
            }
            // Measure readout (when two points are picked).
            if ui_state.measure_pts.len() == 2 {
                let d = ui_state.measure_pts[0].distance(ui_state.measure_pts[1]);
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(255, 220, 120), format!("📏 {}", ui_state.unit.fmt(d)));
            } else if ui_state.measuring {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(230, 170, 60), "Measure: click two points");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.selectable_label(false, ui_state.unit.short()).on_hover_text("Switch units (mm / in)").clicked() {
                    ui_state.unit = if ui_state.unit == Unit::Mm { Unit::Inch } else { Unit::Mm };
                }
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
                    if let (Some(ap), Ok((_tf, mut orbit))) =
                        (session.plane.clone(), cam_q.single_mut())
                    {
                        // Reuse look_along/recenter to derive the target pose, then restore the
                        // current pose and glide to it (so the transition animates).
                        let cur = (orbit.yaw, orbit.pitch, orbit.focus, orbit.radius);
                        look_along(&mut orbit, ap.origin, ap.n);
                        recenter_for_panel(&mut orbit, ctx.screen_rect().height(), ui_state.view_center_offset);
                        let tgt = (orbit.focus, orbit.radius, orbit.yaw, orbit.pitch);
                        (orbit.yaw, orbit.pitch, orbit.focus, orbit.radius) = cur;
                        orbit.animate_to(tgt.0, tgt.1, tgt.2, tgt.3);
                    }
                }
                ViewAction::Normal(n) => {
                    if let Ok((_tf, mut orbit)) = cam_q.single_mut() {
                        let cur = (orbit.yaw, orbit.pitch, orbit.focus, orbit.radius);
                        look_along(&mut orbit, Vec3::ZERO, n);
                        let tgt = (orbit.focus, orbit.radius, orbit.yaw, orbit.pitch);
                        (orbit.yaw, orbit.pitch, orbit.focus, orbit.radius) = cur;
                        orbit.animate_to(tgt.0, tgt.1, tgt.2, tgt.3);
                    }
                }
                ViewAction::Iso => {
                    if let Ok((_tf, mut orbit)) = cam_q.single_mut() {
                        let radius = orbit.radius;
                        orbit.animate_to(Vec3::ZERO, radius, 0.8, -0.55);
                    }
                }
                ViewAction::Fit => {
                    if let Ok((_tf, mut orbit)) = cam_q.single_mut() {
                        let (focus, radius) = fit_view(&part);
                        let (yaw, pitch) = (orbit.yaw, orbit.pitch);
                        orbit.animate_to(focus, radius, yaw, pitch);
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

    // Sketch inference badges: small yellow relation chips beside the cursor (SolidWorks-style).
    if !session.infer_badges.is_empty() {
        if let (Some(ap), Some(cur), Ok((camera, cam_gt))) = (session.plane.clone(), session.cursor_uv, cam_read.single()) {
            if let Ok(sp) = camera.world_to_viewport(cam_gt, ap.to_world(cur)) {
                let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("infer_badges")));
                let mut x = sp.x + 14.0;
                let y = sp.y - 22.0;
                for b in &session.infer_badges {
                    let rect = egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(16.0, 16.0));
                    painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(245, 210, 70));
                    painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 95, 20)), egui::StrokeKind::Inside);
                    painter.text(rect.center(), egui::Align2::CENTER_CENTER, b.symbol(), egui::FontId::proportional(12.0), egui::Color32::from_gray(20));
                    x += 19.0;
                }
            }
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
                        act(label_at(ctx, egui::Id::new(("dimlabel", k)), ap.to_world(lab), fmt_len_bare(*value as f32, unit), on), k, &mut dim_action);
                    }
                }
                Constraint::Radius { center, value, diameter } => {
                    dimensioned_centers.push(*center);
                    if let Some(c) = session.sketch.points.get(*center) {
                        let cu = Vec2::new(c.x as f32, c.y as f32);
                        let r = *value as f32;
                        let edge = cu + Vec2::new(r * 0.707, r * 0.707);
                        let text = if *diameter { format!("Ø{}", fmt_len_bare(*value as f32 * 2.0, unit)) } else { format!("R{}", fmt_len_bare(*value as f32, unit)) };
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
                        act(label_at(ctx, egui::Id::new(("pldlabel", k)), ap.to_world(lab), fmt_len_bare(*value as f32, unit), on), k, &mut dim_action);
                    }
                }
                Constraint::SlotWidth { a, b, value, .. } => {
                    if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                        let a2 = Vec2::new(pa.x as f32, pa.y as f32);
                        let b2 = Vec2::new(pb.x as f32, pb.y as f32);
                        let (_, _, lab) = slot_width_geometry(a2, b2, (*value * 0.5) as f32);
                        act(label_at(ctx, egui::Id::new(("slotlabel", k)), ap.to_world(lab), fmt_len_bare(*value as f32, unit), on), k, &mut dim_action);
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
                    let resp = label_at(ctx, egui::Id::new(("dialabel", k)), ap.to_world(edge), format!("Ø{}", fmt_len_bare(*radius as f32 * 2.0, unit)), false);
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
                                    let f = unit.factor();
                                    let mut disp = buf * f;
                                    let resp = ui.add_sized(
                                        egui::vec2(78.0, ui.spacing().interact_size.y),
                                        egui::DragValue::new(&mut disp).speed(0.1 * f as f64).range((0.001 * f)..=(1_000_000.0 * f)).max_decimals(if unit == Unit::Inch { 4 } else { 2 }).suffix(unit.suffix()),
                                    );
                                    buf = disp / f;
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
            // Slot width edits like a plain mm distance (no axis/diameter toggle).
            Some(Constraint::PointLineDistance { .. }) | Some(Constraint::SlotWidth { .. }) => Some(DimKind::PointLine),
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
            Some(Constraint::SlotWidth { a, b, value, .. }) => match (pt(*a), pt(*b)) {
                (Some(a2), Some(b2)) => Some(slot_width_geometry(a2, b2, (*value * 0.5) as f32).2),
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
                                    Some(Constraint::SlotWidth { value, .. }) => *value = v,
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
    // World-axis labels (X/Y/Z) at the axis tips while starting a part (no body, not sketching),
    // matching the coloured gizmo axis lines so the reference-plane orientation is readable.
    if part.solid.is_none() && part.mesh.is_none() && session.plane.is_none() {
        if let Ok((camera, cam_gt)) = cam_read.single() {
            // Only draw a tip label if it lands inside the 3D viewport — otherwise the projected tip
            // can climb into the toolbar/side panels and bleed over them (the "Y popping through").
            let view = ctx.available_rect();
            for (world, label, color) in [
                (Vec3::X * 5.2, "X", egui::Color32::from_rgb(255, 80, 80)),
                (Vec3::Y * 5.2, "Y", egui::Color32::from_rgb(90, 230, 90)),
                (Vec3::Z * 5.2, "Z", egui::Color32::from_rgb(110, 150, 255)),
            ] {
                if let Ok(p) = camera.world_to_viewport(cam_gt, world) {
                    let tip = egui::pos2(p.x, p.y);
                    if !view.contains(tip) {
                        continue;
                    }
                    egui::Area::new(egui::Id::new(("axislabel", label)))
                        .order(egui::Order::Middle)
                        .fixed_pos(egui::pos2(p.x - 5.0, p.y - 9.0))
                        .show(ctx, |ui| {
                            ui.label(egui::RichText::new(label).size(16.0).strong().color(color));
                        });
                }
            }
        }
    }

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
                            if ui.small_button("✖").on_hover_text("Dismiss").clicked() {
                                dismiss = true;
                            }
                        });
                    });
            });
        if dismiss {
            ui_state.last_error = None;
        }
    }

    // ---- Navigation triad: a small clickable axis gizmo, top-right of the viewport. It mirrors the
    // camera orientation; clicking an axis tip glides to that standard view (SolidWorks/Fusion-style).
    if let Ok((_, orbit)) = cam_q.single() {
        let (yaw, pitch) = (orbit.yaw, orbit.pitch);
        let rot = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        let right = rot * Vec3::X;
        let up = rot * Vec3::Y;
        let fwd = rot * Vec3::NEG_Z; // into the scene
        // Each world axis → a labelled tip. Project onto screen (x=right·d, y=−up·d); depth = fwd·d.
        let axes: [(Vec3, &str, egui::Color32); 6] = [
            (Vec3::X, "X", egui::Color32::from_rgb(230, 90, 90)),
            (Vec3::NEG_X, "-X", egui::Color32::from_rgb(150, 70, 70)),
            (Vec3::Y, "Y", egui::Color32::from_rgb(110, 210, 110)),
            (Vec3::NEG_Y, "-Y", egui::Color32::from_rgb(70, 140, 70)),
            (Vec3::Z, "Z", egui::Color32::from_rgb(110, 150, 240)),
            (Vec3::NEG_Z, "-Z", egui::Color32::from_rgb(70, 95, 160)),
        ];
        let sz = 88.0;
        let vr = ctx.available_rect();
        let mut nav_target: Option<Vec3> = None;
        egui::Area::new(egui::Id::new("nav_triad"))
            .fixed_pos(egui::pos2(vr.right() - sz - 12.0, vr.top() + 12.0))
            .order(egui::Order::Middle)
            .show(ctx, |ui| {
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(sz, sz), egui::Sense::click());
                let c = rect.center();
                let r = sz * 0.34;
                let painter = ui.painter();
                let ptr = resp.hover_pos();
                // Draw back-to-front so near tips overlap far ones.
                let mut order: Vec<usize> = (0..6).collect();
                order.sort_by(|&a, &b| fwd.dot(axes[b].0).partial_cmp(&fwd.dot(axes[a].0)).unwrap());
                let proj = |d: Vec3| egui::vec2(d.dot(right), -d.dot(up));
                for &i in &order {
                    let (axis, label, col) = axes[i];
                    let tip = c + proj(axis) * r;
                    let positive = !label.starts_with('-');
                    let hovered = ptr.map_or(false, |p| p.distance(tip) < 12.0);
                    // Axis stick (only for the positive tips, so the gizmo reads as a 3-arm triad).
                    if positive {
                        painter.line_segment([c, tip], egui::Stroke::new(2.0, col));
                    }
                    let rad = if positive { 9.0 } else { 6.0 };
                    let fill = if hovered { egui::Color32::WHITE } else if positive { col } else { egui::Color32::from_gray(70) };
                    painter.circle_filled(tip, rad, fill);
                    painter.circle_stroke(tip, rad, egui::Stroke::new(1.0, col));
                    if positive {
                        painter.text(tip, egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(11.0), egui::Color32::from_gray(20));
                    }
                    if hovered && resp.clicked() {
                        nav_target = Some(axis);
                    }
                }
                resp.on_hover_text("Click an axis to snap to that view");
            });
        if let Some(axis) = nav_target {
            if let Ok((_, mut orbit)) = cam_q.single_mut() {
                let n = axis.normalize();
                let (ty, tp) = (n.x.atan2(n.z), (-n.y).asin().clamp(-1.54, 1.54));
                orbit.animate_view(ty, tp);
            }
        }
    }

    // About window.
    if ui_state.show_about {
        // Lazily upload the (cropped) logo into an egui texture the first time About is shown.
        let tex = logo_tex.get_or_insert_with(|| {
            let (rgba, w, h) = logo_cropped_rgba().unwrap_or((vec![0, 0, 0, 0], 1, 1));
            ctx.load_texture(
                "hcad_logo",
                egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                egui::TextureOptions::LINEAR,
            )
        });
        let logo_id = tex.id();
        let logo_size = tex.size();
        let mut open = true;
        egui::Window::new("About HCAD")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .open(&mut open)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                // The wordmark logo on a white plate so the black ink reads on the dark theme.
                let aspect = logo_size[0].max(1) as f32 / logo_size[1].max(1) as f32;
                let lw = 260.0_f32.min(ui.available_width());
                egui::Frame::new()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .corner_radius(6.0)
                    .show(ui, |ui| {
                        ui.add(egui::Image::new((logo_id, egui::vec2(lw, lw / aspect))));
                    });
                ui.add_space(6.0);
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.add_space(6.0);
                ui.label("A parametric solid modeler — sketch, extrude, revolve, loft,");
                ui.label("fillet/chamfer, and mesh booleans, with STL/STEP export.");
                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Kernels:");
                    ui.label(egui::RichText::new("truck (exact B-rep) + Manifold (mesh)").weak());
                });
                ui.add_space(4.0);
            });
        ui_state.show_about = open;
    }

    // Toasts: transient status chips, bottom-right, fading out over their last second.
    if !ui_state.toasts.is_empty() {
        let dt = time.delta_secs();
        for (_, ttl) in ui_state.toasts.iter_mut() {
            *ttl -= dt;
        }
        ui_state.toasts.retain(|(_, ttl)| *ttl > 0.0);
        let toasts = ui_state.toasts.clone();
        for (i, (text, ttl)) in toasts.iter().enumerate() {
            let a = ttl.clamp(0.0, 1.0); // fade over the final second
            egui::Area::new(egui::Id::new(("toast", i)))
                .order(egui::Order::Foreground)
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -16.0 - i as f32 * 40.0))
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgba_unmultiplied(40, 44, 52, (235.0 * a) as u8))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(90, 150, 90, (200.0 * a) as u8)))
                        .inner_margin(egui::Margin::symmetric(12, 8))
                        .corner_radius(6.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(text)
                                    .color(egui::Color32::from_rgba_unmultiplied(235, 245, 235, (255.0 * a) as u8)),
                            );
                        });
                });
        }
        ctx.request_repaint(); // keep animating the fade
    }

    // Where the visible 3D area's centre sits relative to the window centre (all panels are
    // declared by now, so available_rect is the true remaining viewport).
    {
        let vr = ctx.available_rect();
        let sr = ctx.screen_rect();
        let d = vr.center() - sr.center();
        ui_state.view_center_offset = (d.x, d.y);
    }

    blocking.0 = ctx.wants_pointer_input() || ctx.is_pointer_over_area();
    blocking.1 = ctx.wants_keyboard_input();

    // Tool-aware cursor over the 3D viewport (only when the pointer isn't over an egui panel/widget,
    // so egui's own resize/text cursors still win there). A crosshair for placing geometry or
    // measuring; a pointing hand when a body edge loop is ready to pick.
    if !ctx.is_pointer_over_area() {
        let cursor = if in_sketch && session.tool != Tool::Select {
            Some(egui::CursorIcon::Crosshair)
        } else if ui_state.measuring {
            Some(egui::CursorIcon::Crosshair)
        } else if ui_state.hover_edge_loop.is_some() {
            Some(egui::CursorIcon::PointingHand)
        } else {
            None
        };
        if let Some(c) = cursor {
            ctx.set_cursor_icon(c);
        }
    }
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

/// Lock an active inference alignment into a real relation when its point is placed: a shared-X
/// alignment (`v`) becomes `Vertical(p, anchor)`, a shared-Y alignment (`h`) becomes
/// `Horizontal(p, anchor)`, so the "even with that point" alignment survives later edits — exactly
/// like clicking an inference in SolidWorks. Skips self / duplicate / stale-index anchors.
/// SolidWorks-style inferencing: when the cursor lines up with existing geometry, nudge it onto
/// the alignment and report dotted guide segments (uv→uv) explaining the snap. Priority, highest
/// first: horizontal/vertical alignment with an existing point (the dotted "even with" line you
/// see in SolidWorks) → collinear extension of a nearby line → tangent off the in-progress line's
/// start when that start sits on a circle/arc. `start` is the rubber-band's anchor (if any).
/// Result of inferencing: the (possibly snapped) cursor, dotted guides to draw, and — for relation
/// capture — the sketch point this cursor is now vertically / horizontally aligned with (if any).
struct Inference {
    uv: Vec2,
    guides: Vec<(Vec2, Vec2)>,
    /// Sketch point sharing the cursor's X (a Vertical relation can be captured against it).
    v_anchor: Option<usize>,
    /// Sketch point sharing the cursor's Y (a Horizontal relation can be captured against it).
    h_anchor: Option<usize>,
    /// A non-alignment relation the cursor snapped onto (collinear/tangent), for the cursor badge.
    kind: Option<InferBadge>,
}

/// A relation hint shown as a small badge next to the sketch cursor.
#[derive(Clone, Copy, PartialEq)]
enum InferBadge {
    Horizontal,
    Vertical,
    Coincident,
    OnEdge,
    Collinear,
    Tangent,
}

impl InferBadge {
    /// The single-glyph symbol drawn in the badge (kept to font-safe ASCII).
    fn symbol(self) -> &'static str {
        match self {
            InferBadge::Horizontal => "—",
            InferBadge::Vertical => "|",
            InferBadge::Coincident => "•",
            InferBadge::OnEdge => "/",
            InferBadge::Collinear => "L",
            InferBadge::Tangent => "T",
        }
    }
}

fn infer_cursor(session: &SketchSession, cur: Vec2, start: Option<Vec2>, tol: f32) -> Inference {
    let mut out = cur;
    let mut guides: Vec<(Vec2, Vec2)> = Vec::new();
    let none = |uv, guides| Inference { uv, guides, v_anchor: None, h_anchor: None, kind: None };
    let with_kind = |uv, guides, kind| Inference { uv, guides, v_anchor: None, h_anchor: None, kind: Some(kind) };

    // Anchor points to align to: every sketch point (capturable, carries its index) + body
    // reference points + the origin (centre) — the latter two align/snap but can't be constrained.
    let mut anchors: Vec<(Vec2, Option<usize>)> =
        session.sketch.points.iter().enumerate().map(|(i, p)| (Vec2::new(p.x as f32, p.y as f32), Some(i))).collect();
    anchors.extend(session.reference_points.iter().map(|p| (*p, None)));
    anchors.push((Vec2::ZERO, None));

    // --- Horizontal / vertical alignment with an anchor point ---
    // Pick the closest anchor sharing the cursor's X (vertical guide) and the closest sharing its
    // Y (horizontal guide); they can both fire, snapping the cursor onto their crossing.
    let mut vx: Option<(Vec2, Option<usize>)> = None; // anchor with matching X → vertical guide
    let mut hy: Option<(Vec2, Option<usize>)> = None; // anchor with matching Y → horizontal guide
    for &(a, idx) in &anchors {
        if (a.x - cur.x).abs() <= tol && (a.y - cur.y).abs() > tol && vx.map_or(true, |(b, _): (Vec2, _)| (a.x - cur.x).abs() < (b.x - cur.x).abs()) {
            vx = Some((a, idx));
        }
        if (a.y - cur.y).abs() <= tol && (a.x - cur.x).abs() > tol && hy.map_or(true, |(b, _): (Vec2, _)| (a.y - cur.y).abs() < (b.y - cur.y).abs()) {
            hy = Some((a, idx));
        }
    }
    if let Some((a, _)) = vx {
        out.x = a.x;
    }
    if let Some((a, _)) = hy {
        out.y = a.y;
    }
    if vx.is_some() || hy.is_some() {
        if let Some((a, _)) = vx {
            guides.push((a, Vec2::new(out.x, a.y))); // a → straight up/down to the cursor row
            guides.push((Vec2::new(out.x, a.y), out));
        }
        if let Some((a, _)) = hy {
            guides.push((a, out));
        }
        return Inference { uv: out, guides, v_anchor: vx.and_then(|(_, i)| i), h_anchor: hy.and_then(|(_, i)| i), kind: None };
    }

    // --- Collinear: snap onto the infinite extension of a nearby (non-reference) line ---
    // Only past the segment's ends — the span itself is already a snap target elsewhere.
    let mut best: Option<(f32, Vec2, Vec2)> = None; // (perp dist, projected pt, near endpoint)
    for e in &session.sketch.entities {
        if let SketchEntity::Line { a, b, reference: false, construction: false, .. } = e {
            let (pa, pb) = (pt2(&session.sketch, *a), pt2(&session.sketch, *b));
            let ab = pb - pa;
            let len = ab.length();
            if len < 1e-5 {
                continue;
            }
            let dir = ab / len;
            let t = (cur - pa).dot(dir);
            if t > -tol && t < len + tol {
                continue; // within (or right at) the span — not an extension
            }
            let proj = pa + dir * t;
            let d = cur.distance(proj);
            if d <= tol && best.map_or(true, |(bd, _, _)| d < bd) {
                let near = if t < 0.0 { pa } else { pb };
                best = Some((d, proj, near));
            }
        }
    }
    if let Some((_, proj, near)) = best {
        return with_kind(proj, vec![(near, proj)], InferBadge::Collinear);
    }

    // --- Tangent: if the rubber-band started on a circle/arc rim, snap the line tangent there ---
    if let Some(s) = start {
        let on_rim = |c: Vec2, r: f32| (s.distance(c) - r).abs() <= tol && s.distance(c) > 1e-4;
        let mut center_r: Option<(Vec2, f32)> = None;
        for e in &session.sketch.entities {
            match e {
                SketchEntity::Circle { center, radius, construction: false } => {
                    let c = pt2(&session.sketch, *center);
                    if on_rim(c, *radius as f32) {
                        center_r = Some((c, *radius as f32));
                    }
                }
                SketchEntity::Arc { center, .. } => {
                    let c = pt2(&session.sketch, *center);
                    let r = (s - c).length();
                    if on_rim(c, r) {
                        center_r = Some((c, r));
                    }
                }
                _ => {}
            }
        }
        if let Some((c, _)) = center_r {
            let radial = (s - c).normalize_or_zero();
            if radial != Vec2::ZERO {
                let td = Vec2::new(-radial.y, radial.x); // tangent = perp to the radius at s
                let along = (cur - s).dot(td);
                let proj = s + td * along;
                if cur.distance(proj) <= tol {
                    return with_kind(proj, vec![(s, proj)], InferBadge::Tangent);
                }
            }
        }
    }

    none(out, guides)
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

/// Points that exist ONLY as endpoints of projected reference lines — internal plumbing, not user
/// geometry. They sit within a whisker of the real projected corner point but are NOT it (edge
/// tessellation vs corner projection differ by microns): welding a drawn endpoint onto one, then
/// relating it to the true corner, created a Coincident between two fixed points that can never
/// meet — an infeasible system the solver mangled ("the triangle snapped out of existence").
fn ref_only_points(sketch: &Sketch) -> std::collections::HashSet<usize> {
    use std::collections::HashSet;
    let mut in_ref: HashSet<usize> = HashSet::new();
    let mut in_real: HashSet<usize> = HashSet::new();
    for (i, e) in sketch.entities.iter().enumerate() {
        if let SketchEntity::Line { a, b, reference: true, .. } = e {
            in_ref.insert(*a);
            in_ref.insert(*b);
        } else {
            for p in entity_points(sketch, i) {
                in_real.insert(p);
            }
        }
    }
    in_ref.retain(|p| !in_real.contains(p));
    in_ref
}

/// Like `get_or_add_point`, but if the position coincides with a body-projected
/// reference snap point (a corner/centre), the new point is *locked* there — so the
/// endpoint stays constrained to that 3D feature through later solves.
fn get_or_add_point_ref(session: &mut SketchSession, uv: Vec2, snap: f32) -> usize {
    // Weld to the nearest point that ISN'T reference-line plumbing (see ref_only_points).
    let skip = ref_only_points(&session.sketch);
    let mut best: Option<(usize, f32)> = None;
    for (i, p) in session.sketch.points.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        let d = Vec2::new(p.x as f32, p.y as f32).distance(uv);
        if d <= snap && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    if let Some((i, _)) = best {
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

/// Place a **circle's centre** point. Like `get_or_add_point_ref`, but it never welds onto a point
/// that is *already another circle's centre*: two circles sharing a centre point also share every
/// centre-keyed constraint (Radius/Diameter/Concentric) and collapse to one radius in the solver —
/// which is exactly why "a circle inside a circle" vanished. Instead give the new circle its own
/// centre at the same spot and a Coincident (concentric) constraint, SolidWorks-style. Reusing a
/// non-circle point (the origin anchor, a line endpoint) is still fine — those don't carry radii.
fn add_circle_center(session: &mut SketchSession, uv: Vec2, snap: f32) -> usize {
    if let Some(existing) = nearest_point(&session.sketch, uv, snap) {
        let is_circle_center = session
            .sketch
            .entities
            .iter()
            .any(|e| matches!(e, SketchEntity::Circle { center, .. } if *center == existing));
        if is_circle_center {
            let p = session.sketch.points[existing];
            let new = session.sketch.add_point(p.x, p.y);
            session.sketch.constraints.push(Constraint::Coincident(new, existing));
            return new;
        }
    }
    get_or_add_point_ref(session, uv, snap)
}

/// Strip redundant / legacy auto-relations from a sketch, returning how many were removed.
/// Heals files saved by older builds that (a) captured the dotted alignment guides as permanent
/// cross-point Horizontal/Vertical relations and (b) double-pinned snapped endpoints with BOTH a
/// parametric PointOnCircle and an absolute-coordinate PointOnArc of the same rim — the arc pin
/// freezes the old position/radius and fights every later edit (the "collapse / wonky" solves).
/// Kept: dimensions, coincident joins, midpoints, one positional pin per point, and each line's
/// OWN horizontal/vertical (the pair being that line's endpoints).
fn clean_redundant_relations(sketch: &mut Sketch) -> usize {
    use std::collections::HashSet;
    let before = sketch.constraints.len();
    // Points already pinned to a sketch circle (parametric — follows edits).
    let circle_pinned: HashSet<usize> = sketch
        .constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::PointOnCircle { p, .. } => Some(*p),
            _ => None,
        })
        .collect();
    // Points pinned onto a (projected) line: an exact H/V on the same line conflicts with the
    // pin's micron-scale tilt — their only common solution is a single point, so the solver
    // collapses the line onto it (or shoots it off along the extrapolation).
    let line_pinned: HashSet<usize> = sketch
        .constraints
        .iter()
        .filter_map(|c| match c {
            Constraint::PointOnLine { p, .. } => Some(*p),
            _ => None,
        })
        .collect();
    // Endpoint pairs of actual sketch lines — their own H/V relations are the wanted ones.
    let line_pairs: HashSet<(usize, usize)> = sketch
        .entities
        .iter()
        .filter_map(|e| match e {
            SketchEntity::Line { a, b, .. } => Some(((*a).min(*b), (*a).max(*b))),
            _ => None,
        })
        .collect();
    let mut seen: HashSet<String> = HashSet::new();
    sketch.constraints.retain(|c| {
        // Exact duplicates of any kind: keep the first occurrence only.
        if !seen.insert(format!("{c:?}")) {
            return false;
        }
        match c {
            // Stale absolute-arc pin duplicating a parametric circle pin on the same point.
            Constraint::PointOnArc { p, .. } if circle_pinned.contains(p) => false,
            // Cross-point align captures (guides are snap-only now); a line's own H/V stays —
            // unless an endpoint is pinned onto a projected line (the pin + exact axis conflict).
            Constraint::Horizontal(a, b) | Constraint::Vertical(a, b) => {
                line_pairs.contains(&((*a).min(*b), (*a).max(*b)))
                    && !line_pinned.contains(a)
                    && !line_pinned.contains(b)
            }
            _ => true,
        }
    });
    before - sketch.constraints.len()
}

/// If point `p` sits on a sketch circle's rim (e.g. a line endpoint just snapped to it),
/// record a point-on-circle relation so the endpoint follows later radius/centre edits.
/// `tol` should be a hair under the snap distance so only genuine rim landings qualify.
fn maybe_add_point_on_circle(sketch: &mut Sketch, p: usize, tol: f32) {
    // A fixed (body-projected) point can't move — relating it does nothing at best, and at worst
    // creates an unsatisfiable constraint (two pinned points that don't quite coincide).
    if sketch.points.get(p).map_or(true, |q| q.fixed) {
        return;
    }
    // One curve constraint per point: if `p` is already pinned to a line/arc/circle, don't stack
    // another (a snapped endpoint used to collect PointOnCircle + PointOnArc for the same rim —
    // 2 curve equations + an axis relation on a 2-DOF point = over-defined, solver goes wonky).
    let pinned = sketch.constraints.iter().any(|c| match c {
        Constraint::PointOnLine { p: q, .. }
        | Constraint::PointOnArc { p: q, .. }
        | Constraint::PointOnCircle { p: q, .. } => *q == p,
        _ => false,
    });
    if pinned {
        return;
    }
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
    // Never relate a fixed point (see maybe_add_point_on_circle) — a Coincident between two
    // pinned-but-not-identical points is permanently infeasible and poisons every later solve.
    if sketch.points.get(p).map_or(true, |q| q.fixed) {
        return;
    }
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
    // One curve constraint per point (see maybe_add_point_on_circle): a sketch-circle pin and its
    // body-edge (arc) twin describe the same rim — adding both over-defines the endpoint.
    let already = |c: &Constraint| match c {
        Constraint::PointOnLine { p: q, .. }
        | Constraint::PointOnArc { p: q, .. }
        | Constraint::PointOnCircle { p: q, .. } => *q == p,
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
            SketchEntity::Arc { center, a, b, ccw, .. } => {
                match (sketch.points.get(*center), sketch.points.get(*a), sketch.points.get(*b)) {
                    (Some(c), Some(pa), Some(pb)) => {
                        let poly = tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw);
                        let mut dmin = f32::MAX;
                        for w in poly.windows(2) {
                            let a2 = Vec2::new(w[0][0] as f32, w[0][1] as f32);
                            let b2 = Vec2::new(w[1][0] as f32, w[1][1] as f32);
                            let ab = b2 - a2;
                            let t = if ab.length_squared() > 1e-9 { ((uv - a2).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                            dmin = dmin.min((uv - (a2 + ab * t)).length());
                        }
                        dmin
                    }
                    _ => continue,
                }
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

/// The (centre-line endpoints, half-width) of a slot entity, if `i` is a slot.
fn entity_slot(sketch: &Sketch, i: usize) -> Option<(usize, usize, f64)> {
    match sketch.entities.get(i) {
        Some(SketchEntity::Slot { a, b, radius, .. }) => Some((*a, *b, *radius)),
        _ => None,
    }
}

/// The two side points (across the width) and the label anchor of a slot-width dimension:
/// the perpendicular through the centre line's midpoint, ±`half` to each side.
fn slot_width_geometry(a2: Vec2, b2: Vec2, half: f32) -> (Vec2, Vec2, Vec2) {
    let center = (a2 + b2) * 0.5;
    let mut dir = (b2 - a2).normalize_or_zero();
    if dir == Vec2::ZERO {
        dir = Vec2::X;
    }
    let perp = Vec2::new(-dir.y, dir.x);
    (center + perp * half, center - perp * half, center)
}

/// Add a slot-width dimension (or return the existing one) driving the slot's half-width.
fn add_slot_width_dim(sketch: &mut Sketch, slot_entity: usize) -> Option<usize> {
    let (a, b, radius) = entity_slot(sketch, slot_entity)?;
    if let Some(i) = sketch.constraints.iter().position(|c| {
        matches!(c, Constraint::SlotWidth { a: x, b: y, .. } if (*x == a && *y == b) || (*x == b && *y == a))
    }) {
        return Some(i);
    }
    let value = (radius * 2.0).max(0.001);
    sketch.constraints.push(Constraint::SlotWidth { a, b, value, offset: 0.0 });
    Some(sketch.constraints.len() - 1)
}

/// The point indices an entity is built on (a line's ends, a circle's centre).
fn entity_points(sketch: &Sketch, i: usize) -> Vec<usize> {
    match sketch.entities.get(i) {
        Some(SketchEntity::Line { a, b, .. }) => vec![*a, *b],
        Some(SketchEntity::Circle { center, .. }) => vec![*center],
        Some(SketchEntity::Arc { center, a, b, .. }) => vec![*center, *a, *b],
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
        Constraint::SlotWidth { a, b, .. } => vec![*a, *b],
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
        Constraint::SlotWidth { value, .. } => format!("Slot width  {value:.2}"),
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
    // Entity PARAMETERS matter too: a circle/slot radius lives on the entity, not in a point or
    // a constraint — editing it in the panel changed nothing this hash saw, so the cached
    // regions (the green contour fill) froze at the old size.
    for e in &s.entities {
        let (tag, val) = match e {
            SketchEntity::Point { .. } => (1u64, 0.0),
            SketchEntity::Line { construction, reference, .. } => (2 | (*construction as u64) << 4 | (*reference as u64) << 5, 0.0),
            SketchEntity::Circle { radius, construction, .. } => (3 | (*construction as u64) << 4, *radius),
            SketchEntity::Arc { ccw, construction, .. } => (4 | (*ccw as u64) << 4 | (*construction as u64) << 5, 0.0),
            SketchEntity::Slot { radius, construction, .. } => (5 | (*construction as u64) << 4, *radius),
            SketchEntity::Spline { closed, construction, .. } => (6 | (*closed as u64) << 4 | (*construction as u64) << 5, 0.0),
            SketchEntity::Text { height, .. } => (7, *height),
        };
        mix(tag);
        mix((val * 1.0e4).round() as i64 as u64);
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

/// Index of the region (in a precomputed `regions()` list) whose interior — inside
/// the outer loop, outside any hole — contains `uv`. Takes the list rather than the
/// sketch so callers can use the session's cached regions.
fn region_at(regions: &[hworks_sketch::Region], uv: Vec2) -> Option<usize> {
    let p = [uv.x as f64, uv.y as f64];
    regions
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

    // Area-weighted centroid of all triangles coplanar with the hit → face origin.
    // Weighting by triangle area gives the *true* centroid of the face region, so a
    // curved-edge face (e.g. a cylinder's circular top) centres exactly on its axis
    // regardless of how it was triangulated. An unweighted mean of triangle
    // centroids is biased by the tessellation (a fan packs more small triangles to
    // one side), which drifted a stacked concentric boss off-axis by ~1% of radius.
    let plane_d = n.dot(hit);
    let mut sum = Vec3::ZERO;
    let mut total_area = 0.0_f32;
    for tri in mesh.indices.chunks(3) {
        let a = Vec3::from_array(pos[tri[0] as usize]);
        let b = Vec3::from_array(pos[tri[1] as usize]);
        let c = Vec3::from_array(pos[tri[2] as usize]);
        let cross = (b - a).cross(c - a);
        let mut tn = cross.normalize_or_zero();
        if tn.dot(n) < 0.0 {
            tn = -tn;
        }
        let centroid = (a + b + c) / 3.0;
        if tn.dot(n) > 0.99 && (n.dot(centroid) - plane_d).abs() < 0.01 {
            let area = cross.length() * 0.5;
            sum += centroid * area;
            total_area += area;
        }
    }
    let mut origin = if total_area > 1e-12 { sum / total_area } else { hit };
    origin -= n * (n.dot(origin) - plane_d); // snap exactly onto the plane

    // In-plane axes from the normal.
    let seed = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let u = (seed - n * seed.dot(n)).normalize();
    let v = n.cross(u).normalize();

    Some((best_t, ActivePlane { name: "Face".into(), origin, u, v, n, datum: false }))
}

/// An `ActivePlane` (app-side) from a stored `PlaneRef` (document-side).
fn active_plane_from_ref(p: &PlaneRef, name: &str) -> ActivePlane {
    let f = |a: [f64; 3]| Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32);
    ActivePlane { name: name.into(), origin: f(p.origin), u: f(p.u), v: f(p.v), n: f(p.normal), datum: p.datum }
}

/// `PlaneRef` (document-side) from an active plane (app-side).
fn plane_ref(ap: &ActivePlane) -> PlaneRef {
    PlaneRef {
        origin: [ap.origin.x as f64, ap.origin.y as f64, ap.origin.z as f64],
        u: [ap.u.x as f64, ap.u.y as f64, ap.u.z as f64],
        v: [ap.v.x as f64, ap.v.y as f64, ap.v.z as f64],
        normal: [ap.n.x as f64, ap.n.y as f64, ap.n.z as f64],
        datum: ap.datum,
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

/// While the Fillet/Chamfer PropertyManager is open, preview the edge loop under the cursor so the
/// user sees exactly what a click will grab (SolidWorks-style pre-highlight). Stored in `ui_state`
/// and drawn by `draw_edge_selection`. Runs regardless of the egui pointer-block — `pick_edge`
/// returns nothing unless the cursor is actually on a body edge, so it never fights the panel.
fn hover_body_edge(
    windows: Query<&Window, With<PrimaryWindow>>,
    cam_q: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    part: Res<Part>,
    session: Res<SketchSession>,
    keys: Res<ButtonInput<KeyCode>>,
    mut ui_state: ResMut<UiState>,
) {
    let picking = ui_state.pending_fillet.is_some() || ui_state.pending_chamfer.is_some();
    if !picking || session.plane.is_some() {
        if ui_state.hover_edge_loop.is_some() {
            ui_state.hover_edge_loop = None;
        }
        return;
    }
    // Match the click exactly: plain hover previews the single edge; Ctrl previews the loop.
    let loop_snap = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let hit = windows
        .single()
        .ok()
        .zip(cam_q.single().ok())
        .and_then(|(w, (cam, gt))| w.cursor_position().map(|c| (c, cam, gt)))
        // Same pools as the click: sharp edges, then a round body's tangent fillet seams.
        .and_then(|(cursor, cam, gt)| pick_edge_loop_any(&part, cam, gt, cursor, EDGE_PICK_PX, loop_snap));
    ui_state.hover_edge_loop = hit;
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

fn sketch_interaction(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
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
    // Keep the active sketch anchored and its status fresh. The origin anchor (a
    // fixed point at 0,0, SolidWorks-style) is what makes "fully defined" reachable:
    // relative constraints alone always leave rigid-body freedom. The DOF report is
    // recomputed only when the sketch fingerprint changes.
    if session.plane.is_some() {
        session.sketch.ensure_origin();
        // Skip the (eigen-decomposition) status refresh mid-drag — positions churn
        // every frame there; the report refreshes on release.
        if session.drag.is_none() {
            let fp = sketch_fingerprint(&session.sketch);
            if session.dof_cache.as_ref().map_or(true, |(f, _)| *f != fp) {
                session.dof_cache = Some((fp, session.sketch.dof_report()));
            }
        }
    }

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
            // A plain click picks the SINGLE edge under the cursor (the smooth chain,
            // stopping at sharp corners) — Ctrl+click loop-snaps to the whole closed
            // planar loop (a face's perimeter in one click, for rounding a full rim).
            // Both pools are pickable here — sharp corners AND the tangent seams a
            // previous fillet left on a round body (so a rounded edge can round again).
            let loop_snap = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
            if let Some((chain, closed)) = pick_edge_loop_any(&part, camera, cam_gt, cursor, EDGE_PICK_PX, loop_snap) {
                toggle_fillet_edge(&mut ui_state, &chain, closed);
                edge_sel.set(chain, closed);
            }
        }
        return;
    }

    // Measure tool (view mode): click body feature points (vertex / edge midpoint, else the surface
    // hit); two points give a distance (shown in the status bar). A third click starts a new pair.
    if ui_state.measuring && session.plane.is_none() && !blocking.0 {
        if buttons.just_pressed(MouseButton::Left) {
            if let Some(cursor) = window.cursor_position() {
                if let Some(p) = nearest_measure_point(&part, camera, cam_gt, cursor) {
                    if ui_state.measure_pts.len() >= 2 {
                        ui_state.measure_pts.clear();
                    }
                    ui_state.measure_pts.push(p);
                }
            }
            return;
        }
    }

    // Reference-image calibration: while the "Calibrate scale" tool is armed, the user DRAGS a line
    // across a known feature of the picture (press = start, drag = rubber-band, release = end). A
    // plain two-click also works (press near the previous point restarts; a real drag/second point
    // sets the span). The PM reads the two points back to compute the scale factor.
    if ui_state.image_calib.is_some() && ui_state.image_edit.is_some() && !blocking.0 {
        let plane = ui_state
            .image_edit
            .and_then(|i| doc.0.features.get(i))
            .and_then(|f| match &f.kind {
                FeatureKind::RefImage { plane, .. } => Some(plane.clone()),
                _ => None,
            });
        if let Some(plane) = plane {
            let ap = ActivePlane::from_ref(&plane);
            let uv = ray_plane(&ap, &ray).map(|(_, uv)| uv);
            if let Some(cal) = ui_state.image_calib.as_mut() {
                cal.cursor = uv;
                if let Some(uv) = uv {
                    // Min separation to count as the second point (so a click-in-place doesn't
                    // collapse the span). Supports BOTH gestures: drag (press→release apart) and
                    // two clicks (first click leaves one point, a second far click completes it).
                    let eps = (orbit.radius * 0.02).max(0.5);
                    let far = cal.pts.first().map(|p| (*p - uv).length() > eps).unwrap_or(false);
                    if buttons.just_pressed(MouseButton::Left) {
                        if cal.pts.len() == 1 && far {
                            cal.pts.push(uv); // second click of a two-click measurement
                        } else {
                            cal.pts = vec![uv]; // start a fresh measurement
                        }
                    } else if buttons.just_released(MouseButton::Left) && cal.pts.len() == 1 && far {
                        cal.pts.push(uv); // end of a drag
                    }
                }
            }
        }
        if buttons.just_pressed(MouseButton::Left) || buttons.just_released(MouseButton::Left) {
            return;
        }
    }

    // Hole Genie: while its PM is open, hover the body and snap to the nearest feature point
    // (vertex / edge midpoint / circle centre) so the hole drops onto a precise location
    // without fiddly aiming. A click anchors it there (hit = centre, face normal = axis).
    ui_state.thread_hover = None;
    if ui_state.pending_thread.is_some() && !blocking.0 {
        if let Some(cursor) = window.cursor_position() {
            // While a sketch is open, prefer snapping to one of its points (projected to 3D) — so
            // you can drill exactly where you placed a point. The hole axis is the sketch-plane
            // normal (the outward face normal when sketching on a face).
            let sketch_hit = session.plane.as_ref().and_then(|ap| {
                let mut best: Option<(f32, Vec3)> = None;
                for p in &session.sketch.points {
                    let w = ap.to_world(Vec2::new(p.x as f32, p.y as f32));
                    if let Ok(s) = camera.world_to_viewport(cam_gt, w) {
                        let d = s.distance(cursor);
                        if d < 16.0 && best.map_or(true, |(bd, _)| d < bd) {
                            best = Some((d, w));
                        }
                    }
                }
                best.map(|(_, w)| (w, ap.n))
            });
            // Otherwise snap to the nearest body feature point under the cursor.
            let face_hit = part.mesh.as_ref().and_then(|mesh| {
                pick_face_point(mesh, camera, cam_gt, cursor).map(|(hit, normal)| (snap_place_point(&part, camera, cam_gt, cursor, hit), normal))
            });
            if let Some((origin, normal)) = sketch_hit.or(face_hit) {
                ui_state.thread_hover = Some((origin, normal));
                if buttons.just_pressed(MouseButton::Left) {
                    if let Some(spec) = ui_state.pending_thread.clone() {
                        let mut s = spec;
                        s.origin = origin;
                        s.axis = normal;
                        s.placed = true;
                        ui_state.pending_thread = Some(s);
                    }
                }
            }
        }
        if buttons.just_pressed(MouseButton::Left) {
            return; // consume the placement click
        }
    }

    if blocking.0 {
        return;
    }

    let active_uv = session.plane.as_ref().and_then(|ap| ray_plane(ap, &ray).map(|(_, uv)| uv));

    // Snap tolerance scaled to the zoom, so the grab radius stays ~constant in *screen* space at
    // any scale. The floor is tiny (not 0) so it keeps shrinking as you zoom way in — otherwise a
    // fixed floor balloons on screen at deep zoom and over-snaps; it only guards radius → 0.
    let snap = (orbit.radius * (SNAP / 12.0)).clamp(5.0e-4, 200.0);
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
        // Exact circles from the timeline's source sketch circles in this plane — used to
        // replace each tessellation-fit (centre, radius) with the true value so concentric
        // bosses snap exactly. A fit circle is refined to the nearest matching exact one.
        let exact = exact_plane_circles(&doc.0, &ap);
        let refine = |c: Vec2, r: f32| refine_circle(&exact, c, r);
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
                let (c, r) = refine(c, r); // use the exact source radius/centre when known
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
                            let (c, r) = refine(*c, *r); // exact source radius/centre if known
                            session.reference_circles.push((c, r));
                            session.reference_points.push(c);
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
        // EXCEPT reference-line plumbing endpoints: they shadow the true projected
        // corner by microns, and welding onto one poisons the constraint system.
        let skip = ref_only_points(&session.sketch);
        for (i, p) in session.sketch.points.iter().enumerate() {
            if !skip.contains(&i) {
                snaps.push(Vec2::new(p.x as f32, p.y as f32));
            }
        }
        // The sketch origin (0,0) — on a datum plane this is the part's centre. Snapping to it
        // lets you put a circle's centre or a revolve axis exactly on centreline, so a revolve
        // is concentric with the body instead of slightly eccentric (which tears the boolean).
        snaps.push(Vec2::ZERO);
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
            // Strong targets win over the square snap so connections aren't pulled off. Two
            // signals: the cursor ALREADY snapped to something this frame (it moved off the raw
            // position — a point, corner, line span, rim), OR it's sitting ON a hovered body edge.
            // The second matters when TRACING ALONG an edge: the cursor is already on the edge, so
            // the snap doesn't move it — a movement-only test misses that and the 90° square snap
            // yanks the line off a slanted edge (the triangle-edge bug).
            let near = |p: Vec2, t: f32| p.distance(cur) <= t;
            let strong = snap * 0.6;
            let snapped_to_target = session.cursor_raw_uv.map_or(false, |raw| raw.distance(cur) > 1e-4)
                || session.cursor_edge.is_some()
                || session.hover_edge.is_some_and(|es| edge_snap_point(es, cur).distance(cur) <= strong);
            let on_strong = snapped_to_target
                || nearest_point(&session.sketch, cur, strong).is_some()
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
    // Inferencing (SolidWorks-style): once the cursor is otherwise snapped, see if it lines up
    // with existing geometry (horizontal/vertical with a point, collinear with a line, tangent off
    // a circle the rubber-band started on) and nudge it onto that alignment, drawing dotted hints.
    // Yields to a genuine coincident point snap so connections aren't pulled off.
    session.inference_guides.clear();
    session.infer_v = None;
    session.infer_h = None;
    session.infer_badges.clear();
    let drawing_tool = matches!(
        session.tool,
        Tool::Line | Tool::Circle | Tool::Arc | Tool::Rectangle | Tool::Slot | Tool::Polygon | Tool::Spline
    );
    if drawing_tool && !session.hide_inference {
        if let Some(cur) = session.cursor_uv {
            // A cursor that ALREADY snapped to a real target this frame (point, corner, body edge,
            // line span, circle rim — the cursor moved off the raw position) must not be re-nudged
            // by the alignment guides: yanking a corner-snapped endpoint onto some point's X/Y
            // alignment is how a line aimed at a corner ended up rotated 90° off. Sitting ON a
            // hovered edge counts too — tracing along an edge doesn't move the cursor, so the
            // movement test alone misses it.
            let snapped_to_target = session.cursor_raw_uv.map_or(false, |raw| raw.distance(cur) > 1e-4)
                || session.cursor_edge.is_some()
                || session.hover_edge.is_some_and(|es| edge_snap_point(es, cur).distance(cur) <= snap * 0.6);
            let coincident = nearest_point(&session.sketch, cur, snap * 0.5).is_some()
                || session.reference_points.iter().any(|r| r.distance(cur) <= snap * 0.5);
            if snapped_to_target && !coincident {
                // Keep the snap; no guides, no alignment nudge.
            } else if coincident {
                session.infer_badges.push(InferBadge::Coincident);
            } else {
                let inf = infer_cursor(&session, cur, session.pending, snap * 0.8);
                session.cursor_uv = Some(inf.uv);
                session.inference_guides = inf.guides;
                session.infer_v = inf.v_anchor;
                session.infer_h = inf.h_anchor;
                if let Some(k) = inf.kind {
                    session.infer_badges.push(k);
                }
                if session.infer_h.is_some() {
                    session.infer_badges.push(InferBadge::Horizontal);
                }
                if session.infer_v.is_some() {
                    session.infer_badges.push(InferBadge::Vertical);
                }
            }
        }
        // Snapped onto a body edge → point-on-edge relation (shown unless already coincident).
        if session.cursor_edge.is_some() && !session.infer_badges.contains(&InferBadge::Coincident) {
            session.infer_badges.push(InferBadge::OnEdge);
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
            // The Modify box is open on a line's length; clicking a *second* line (or a body
            // edge) turns it into a line-to-line dimension. Pick the nearest sketch line
            // under the cursor (preferring a real sketch line over a reference line); if
            // there's none, fall back to a body edge under the cursor and project it.
            let first_line = session.dim_line;
            let first_slot = session.dim_slot;
            let mut second_line: Option<usize> = None;
            let mut second_slot: Option<usize> = None;
            if let Some(uv) = active_uv {
                // Nearest sketch line (preferring a real line over a reference line).
                let mut best: Option<(usize, bool, f32)> = None;
                for (i, e) in session.sketch.entities.iter().enumerate() {
                    if Some(i) == first_line {
                        continue;
                    }
                    if let SketchEntity::Line { a, b, reference, .. } = e {
                        if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                            let va = Vec2::new(pa.x as f32, pa.y as f32);
                            let vb = Vec2::new(pb.x as f32, pb.y as f32);
                            let d = closest_on_segment(uv, va, vb).distance(uv);
                            if d <= snap * 2.0 {
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
                second_line = best.map(|(i, _, _)| i);
                // A slot under the cursor (its centre line can be dimensioned to a line).
                if let Some(e) = nearest_entity(&session.sketch, uv, snap * 2.0) {
                    if Some(e) != first_slot && entity_slot(&session.sketch, e).is_some() {
                        second_slot = Some(e);
                    }
                }
                // Otherwise a body edge → project it as a locked reference line.
                if second_line.is_none() && second_slot.is_none() {
                    if let (Some(cursor), Some(ap)) = (window.cursor_position(), session.plane.clone()) {
                        if let Some(ei) = pick_edge(&part.edges, camera, cam_gt, cursor, 10.0) {
                            let seg = part.edges[ei];
                            let to_uv = |w: Vec3| {
                                let d = w - ap.origin;
                                Vec2::new(d.dot(ap.u), d.dot(ap.v))
                            };
                            let a2 = to_uv(Vec3::from_array(seg[0]));
                            let b2 = to_uv(Vec3::from_array(seg[1]));
                            second_line = add_or_get_reference_line(&mut session, a2, b2);
                        }
                    }
                }
            }
            // Decide the new dimension: slot↔line distance, or line↔line distance/angle.
            enum Act {
                SlotLine(usize, usize), // (slot entity, line entity)
                LineLine(usize, usize),
            }
            let act = if let (Some(sl), Some(ln)) = (first_slot, second_line) {
                Some(Act::SlotLine(sl, ln))
            } else if let (Some(ln), Some(sl)) = (first_line, second_slot) {
                Some(Act::SlotLine(sl, ln))
            } else if let (Some(l1), Some(l2)) = (first_line, second_line) {
                (l2 != l1).then_some(Act::LineLine(l1, l2))
            } else {
                None
            };
            if let Some(act) = act {
                // Drop the length/width dim we just made; replace it with the pair dim.
                if ci + 1 == session.sketch.constraints.len() {
                    session.sketch.constraints.pop();
                }
                let new_ci = match act {
                    Act::SlotLine(sl, ln) => add_slot_line_distance(&mut session.sketch, sl, ln),
                    Act::LineLine(l1, l2) => {
                        if lines_parallel(&session.sketch, l1, l2) {
                            add_point_line_distance(&mut session.sketch, l1, l2)
                        } else {
                            add_angle_dim(&mut session.sketch, l1, l2)
                        }
                    }
                };
                if let Some(ci2) = new_ci {
                    session.dim_edit = Some(ci2);
                    session.dim_line = None;
                    session.dim_slot = None;
                    session.dim_buf = match session.sketch.constraints.get(ci2) {
                        Some(Constraint::Angle { value, .. }) => value.to_degrees(),
                        Some(Constraint::PointLineDistance { value, .. }) => *value,
                        _ => 0.0,
                    };
                    session.dim_edit_focus = true;
                    return;
                }
            }
            session.dim_edit = None;
            session.dim_line = None;
            session.dim_slot = None;
        }
        return;
    }

    // While a boss/cut is being configured, grabbing its direction arrow and dragging
    // sets the depth live (which shows in the panel and the feature tree on commit). NOT for a
    // revolve: there's no arrow (the sweep is rotational), and the handler would otherwise hijack
    // a viewport click — meant for picking the axis line — and drag `op.depth` (the angle) to a
    // tiny value, so the revolve came out as a thin wedge instead of the full turn.
    let revolving = matches!(ui_state.pending.as_ref().map(|o| o.kind), Some(OpKind::Revolve | OpKind::RevolveCut));
    if ui_state.pending.is_some() && session.plane.is_some() && !revolving {
        // Direction-2 arrow first (it lives on the opposite side), then the main arrow.
        if extrude_dir2_arrow_drag(&mut session, &mut ui_state, window, camera, cam_gt, &ray, just_pressed, pressed, just_released) {
            return;
        }
        if extrude_arrow_drag(&mut session, &mut ui_state, window, camera, cam_gt, &ray, just_pressed, pressed, just_released) {
            return;
        }
    }

    // Loft contour pick: while the Loft PM is open, a viewport click inside a profile's contour
    // chooses that region for the profile (nearest plane hit wins).
    if ui_state.loft_spec.is_some() && session.plane.is_none() {
        if just_pressed {
            let mut best: Option<(f32, usize, usize)> = None; // (depth, profile index, region index)
            if let Some(profiles) = ui_state.loft_spec.clone() {
                for (pi, (fi, _)) in profiles.iter().enumerate() {
                    if let Some(FeatureKind::Sketch { sketch, plane }) = doc.0.features.get(*fi).map(|f| &f.kind) {
                        let ap = active_plane_from_ref(plane, "");
                        if let Some((t, uv)) = ray_plane(&ap, &ray) {
                            for (ri, r) in sketch.regions().iter().enumerate() {
                                if point_in_poly([uv.x as f64, uv.y as f64], &r.outer) && best.map_or(true, |(bt, _, _)| t < bt) {
                                    best = Some((t, pi, ri));
                                }
                            }
                        }
                    }
                }
            }
            if let (Some((_, pi, ri)), Some(v)) = (best, ui_state.loft_spec.as_mut()) {
                v[pi].1 = ri;
            }
        }
        return; // swallow viewport clicks while lofting (orbit still works on right-drag)
    }

    // Section-view gizmo: grab the offset arrow or a rotation handle in plain view mode.
    if ui_state.section.is_some()
        && session.plane.is_none()
        && section_arrow_drag(&mut session, &mut ui_state, &part, window, camera, cam_gt, &ray, just_pressed, pressed, just_released)
    {
        return;
    }

    // Reference-plane creation: drag the offset arrow, or click a face / datum plane to set the
    // base to offset from. (Swallows viewport interaction while the Plane PM is open.)
    if ui_state.plane_spec.is_some() && session.plane.is_none() {
        if plane_arrow_drag(&mut session, &mut ui_state, window, camera, cam_gt, &ray, just_pressed, pressed, just_released) {
            return;
        }
        if just_pressed {
            let mut best: Option<(f32, ActivePlane, String)> = None;
            // Hidden planes are invisible, so they must not be clickable either.
            for (_id, p, _) in doc.0.planes_vis().filter(|(_, _, h)| !h) {
                let ap = ActivePlane::from_doc(p);
                if let Some((t, uv)) = ray_plane(&ap, &ray) {
                    let half = ui_state.plane_size.max(1.0) * 0.5;
                    if uv.x.abs() <= half && uv.y.abs() <= half && best.as_ref().map_or(true, |(bt, _, _)| t < *bt) {
                        best = Some((t, ap, p.name.clone()));
                    }
                }
            }
            if let Some(mesh) = &part.mesh {
                if let Some((t, ap)) = pick_face(mesh, &ray) {
                    if best.as_ref().map_or(true, |(bt, _, _)| t < *bt) {
                        best = Some((t, ap, "Face".to_string()));
                    }
                }
            }
            if let (Some((_, ap, name)), Some(spec)) = (best, ui_state.plane_spec.as_mut()) {
                spec.base = ap;
                spec.base_name = name;
            }
        }
        return;
    }

    if session.plane.is_none() {
        if just_pressed {
            // A click near a body edge selects that edge/loop (and flashes its key
            // points) instead of starting a sketch. Edges are thin, faces are wide,
            // so clicking the open part of a face still enters sketch mode below.
            if !part.edges.is_empty() || !part.seam_edges.is_empty() || !part.tangent_edges.is_empty() {
                if let Some(cursor) = window.cursor_position() {
                    // View-mode edge selection keeps the loop-snap (grabbing a whole rim in
                    // one click is what you want when inspecting / measuring a feature).
                    if let Some((chain, closed)) = pick_edge_loop_any(&part, camera, cam_gt, cursor, EDGE_PICK_PX, true) {
                        edge_sel.set(chain, closed);
                        return;
                    }
                    // Tangent curvature lines are selectable too, but only while they're drawn
                    // ("Tangent edges" on) — an invisible line must never steal a sketch-on-face click.
                    if ui_state.show_tangent_edges {
                        if let Some(si) = pick_edge(&part.tangent_edges, camera, cam_gt, cursor, EDGE_PICK_PX) {
                            let (chain, closed) = edge_loop(&part.tangent_edges, si);
                            if chain.len() >= 2 {
                                edge_sel.set(chain, closed);
                                return;
                            }
                        }
                    }
                }
            }

            let mut best: Option<(f32, ActivePlane)> = None;
            // Reference planes — only while starting the part (they're hidden once
            // a body exists, so you sketch on faces from then on).
            if part.solid.is_none() {
                // Hidden planes are invisible, so they must not be clickable either.
                for (_id, p, _) in doc.0.planes_vis().filter(|(_, _, h)| !h) {
                    let ap = ActivePlane::from_doc(p);
                    if let Some((t, uv)) = ray_plane(&ap, &ray) {
                        let half = ui_state.plane_size.max(1.0) * 0.5;
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
                recenter_for_panel(&mut orbit, window.height(), ui_state.view_center_offset);
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
        session.dim_slot = None;
    }
    if session.tool != Tool::Spline && !session.spline_pts.is_empty() {
        session.spline_pts.clear(); // leaving the spline tool drops the in-progress curve
    }
    if session.tool != Tool::Trim {
        session.trim_first = None; // drop a half-finished corner / power stroke on tool change
        session.power_prev = None;
        session.power_path.clear();
    }
    if session.tool != Tool::Rectangle && session.tool != Tool::Slot && session.tool != Tool::Arc {
        session.pending_b = None; // the second anchor belongs to the parallelogram / slot / arc
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
                        let region = region_at(&session.cached_regions(), uv);
                        let near_entity = nearest_entity(&session.sketch, uv, snap * 1.5);
                        // While a boss/cut PropertyManager is open the click is
                        // unambiguously a *contour* pick, so a region always wins over a
                        // nearby edge — otherwise thin arrangement faces (bounded by
                        // close-together edges) are almost impossible to select.
                        let choosing_contours =
                            matches!(ui_state.pending.as_ref().map(|o| o.kind), Some(OpKind::Boss | OpKind::Cut));
                        let entity_pick = if choosing_contours {
                            region.is_none().then(|| on_entity.or(near_entity)).flatten()
                        } else {
                            on_entity.or(region.is_none().then_some(()).and(near_entity))
                        };
                        if let Some(e) = entity_pick {
                            let picking_axis = matches!(ui_state.pending.as_ref().map(|o| o.kind), Some(OpKind::Revolve | OpKind::RevolveCut))
                                && matches!(session.sketch.entities.get(e), Some(SketchEntity::Line { .. }));
                            if picking_axis {
                                // Revolve PM open + a line clicked → that line is the axis.
                                session.revolve_axis = Some(e);
                            } else if let Some(pos) = session.selected_entities.iter().position(|&x| x == e) {
                                // (de)select the entity for a constraint. Many can be selected
                                // (e.g. Equal across several lines); pairwise relations just
                                // stay disabled unless exactly two are chosen.
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
        Tool::Line | Tool::Circle | Tool::Arc | Tool::Rectangle | Tool::Slot | Tool::Polygon | Tool::Text if just_pressed => {
            // Use the snapped cursor so endpoints land on midpoints / quadrants / centres.
            if let Some(uv) = session.cursor_uv {
                place_point(&mut session, uv);
            }
        }
        Tool::Trim => {
            let thresh = snap * 1.5;
            match session.trim_mode {
                // Click a piece → delete it back to the nearest intersections.
                TrimMode::Closest => {
                    if just_pressed {
                        if let Some(uv) = active_uv {
                            apply_trim(&mut session, uv, thresh);
                        }
                    }
                }
                // Drag a stroke → trim every entity the cursor path crosses.
                TrimMode::Power => {
                    if just_pressed {
                        session.power_prev = active_uv;
                        session.power_path.clear();
                        session.power_path.extend(active_uv);
                    } else if pressed {
                        if let (Some(prev), Some(cur)) = (session.power_prev, active_uv) {
                            if prev.distance(cur) > 1e-4 {
                                power_trim_stroke(&mut session, prev, cur, thresh);
                                session.power_prev = Some(cur);
                                session.power_path.push(cur);
                            }
                        }
                    }
                    if just_released {
                        session.power_prev = None;
                        session.power_path.clear();
                    }
                }
                // Pick two lines → trim/extend both to meet at a clean corner.
                TrimMode::Corner => {
                    if just_pressed {
                        if let Some(uv) = active_uv {
                            match (session.trim_first, nearest_line(&session.sketch, uv, thresh)) {
                                (None, Some(li)) => session.trim_first = Some(li),
                                (Some(first), Some(li)) if li != first => {
                                    corner_trim(&mut session, first, li);
                                    session.trim_first = None;
                                }
                                (_, None) => session.trim_first = None, // clicked empty → reset
                                _ => {}
                            }
                        }
                    }
                }
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
                    let mut slot_ctx = None;
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
                        } else if entity_slot(&session.sketch, e).is_some() {
                            // A slot is one entity — clicking it dimensions its width; but
                            // remember it so a follow-up click on a line/edge instead makes
                            // a distance from that line to the slot's centre line.
                            slot_ctx = Some(e);
                            add_slot_width_dim(&mut session.sketch, e)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(ci) = created {
                        open_dim_edit(&mut session, ci, line_ctx);
                        session.dim_slot = slot_ctx;
                    }
                }
            }
        }
        _ => {}
    }

    if session.drag.is_none() && session.dirty {
        session.sketch.solve();
        session.dirty = false;
        // Anomaly trap (silent in normal use): if the most recent line COLLAPSED in the solve,
        // dump the full sketch state to run.log so the failure is replayable from the field —
        // this is how the tilted-edge-pin vs exact-Vertical conflict was found.
        if let Some(SketchEntity::Line { a, b, .. }) = session
            .sketch
            .entities
            .iter()
            .rev()
            .find(|e| matches!(e, SketchEntity::Line { reference: false, .. }))
        {
            if let (Some(pa), Some(pb)) = (session.sketch.points.get(*a), session.sketch.points.get(*b)) {
                let len = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt();
                if len < 1.0e-3 {
                    warn!("sketch anomaly: last line collapsed to len={len:.6} — dumping state");
                    for (i, p) in session.sketch.points.iter().enumerate() {
                        warn!("  P{i}: ({:.6},{:.6}) fixed={}", p.x, p.y, p.fixed);
                    }
                    for (i, e) in session.sketch.entities.iter().enumerate() {
                        warn!("  E{i}: {e:?}");
                    }
                    for (i, c) in session.sketch.constraints.iter().enumerate() {
                        warn!("  C{i}: {c:?}");
                    }
                }
            }
        }
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
            let regions = session.cached_regions();
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

/// Find the sketch line entity collinear with a stored revolve axis (point + direction) — used to
/// re-select the axis when reopening a Revolve feature for editing.
fn find_axis_line(sketch: &Sketch, axis_pt: [f64; 2], axis_dir: [f64; 2]) -> Option<usize> {
    let ap = Vec2::new(axis_pt[0] as f32, axis_pt[1] as f32);
    let ad = Vec2::new(axis_dir[0] as f32, axis_dir[1] as f32).normalize_or_zero();
    if ad == Vec2::ZERO {
        return None;
    }
    for (i, e) in sketch.entities.iter().enumerate() {
        if let SketchEntity::Line { a, b, .. } = e {
            let (pa, pb) = (pt2(sketch, *a), pt2(sketch, *b));
            let d = (pb - pa).normalize_or_zero();
            if d == Vec2::ZERO || d.perp_dot(ad).abs() > 1.0e-3 {
                continue; // not parallel to the stored axis
            }
            // Does the stored axis point lie on this line's infinite extension?
            let off = ap - pa;
            if (off - d * off.dot(d)).length() < 1.0e-3 {
                return Some(i);
            }
        }
    }
    None
}

/// The revolve axis: the line the user picked (`session.revolve_axis`, any sketch line) as a uv
/// point and direction. `None` if nothing is picked or it's degenerate.
fn revolve_axis(session: &SketchSession) -> Option<([f64; 2], [f64; 2])> {
    let e = session.revolve_axis?;
    if let Some(SketchEntity::Line { a, b, .. }) = session.sketch.entities.get(e) {
        let (pa, pb) = (session.sketch.points[*a], session.sketch.points[*b]);
        let dir = [pb.x - pa.x, pb.y - pa.y];
        return (dir[0].hypot(dir[1]) > 1e-6).then_some(([pa.x, pa.y], dir));
    }
    None
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
            Constraint::SlotWidth { a, b, value, .. } => match (pt(*a), pt(*b)) {
                (Some(a2), Some(b2)) => Some(slot_width_geometry(a2, b2, (*value * 0.5) as f32).2),
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
        Some(Constraint::SlotWidth { value, .. }) => *value,
        _ => return,
    };
    session.dim_edit = Some(ci);
    session.dim_buf = buf;
    session.dim_edit_focus = true;
    session.dim_line = line;
}

/// Replace a freshly-placed line-length dimension (`ci`) with an angle dimension
/// between lines `l1` and `l2`. Returns the new constraint index.
/// Add a perpendicular distance dimension between two line entities (e.g. the two parallel
/// sides of a slot, or a line and a body-edge reference line). A reference line is used as
/// the base; an endpoint of the other line is the driven point. Returns the constraint idx.
fn add_point_line_distance(sketch: &mut Sketch, l1: usize, l2: usize) -> Option<usize> {
    let _ = entity_line(sketch, l1)?;
    let _ = entity_line(sketch, l2)?;
    let is_ref = |i: usize| matches!(sketch.entities.get(i), Some(SketchEntity::Line { reference: true, .. }));
    // Prefer a reference (body-edge) line as the fixed base.
    let (base, other) = if is_ref(l2) && !is_ref(l1) { (l2, l1) } else { (l1, l2) };
    let (ba, bb) = entity_line(sketch, base)?;
    let (pp, _) = entity_line(sketch, other)?;
    if let Some(i) = sketch.constraints.iter().position(|c| {
        matches!(c, Constraint::PointLineDistance { p, a, b, .. } if *p == pp && *a == ba && *b == bb)
    }) {
        return Some(i);
    }
    let v = |i: usize| {
        let q = sketch.points[i];
        Vec2::new(q.x as f32, q.y as f32)
    };
    let (foot, _) = point_line_geometry(v(pp), v(ba), v(bb));
    let value = (v(pp) - foot).length().max(0.001) as f64;
    sketch.constraints.push(Constraint::PointLineDistance { p: pp, a: ba, b: bb, value, offset: 0.0 });
    Some(sketch.constraints.len() - 1)
}

/// Add a distance dimension between a slot and a line/edge: the perpendicular distance from
/// the slot's centre line (its nearer endpoint) to the line. Editing it slides the slot.
fn add_slot_line_distance(sketch: &mut Sketch, slot_entity: usize, line_entity: usize) -> Option<usize> {
    let (sa, sb, _) = entity_slot(sketch, slot_entity)?;
    let (la, lb) = entity_line(sketch, line_entity)?;
    let v = |i: usize| {
        let q = sketch.points[i];
        Vec2::new(q.x as f32, q.y as f32)
    };
    let perp = |p: usize| {
        let (foot, _) = point_line_geometry(v(p), v(la), v(lb));
        (v(p) - foot).length()
    };
    // Drive whichever centre-line endpoint is nearer the line (so the slot slides square).
    let pp = if perp(sa) <= perp(sb) { sa } else { sb };
    if let Some(i) = sketch.constraints.iter().position(|c| {
        matches!(c, Constraint::PointLineDistance { p, a, b, .. } if *p == pp && *a == la && *b == lb)
    }) {
        return Some(i);
    }
    let value = perp(pp).max(0.001) as f64;
    sketch.constraints.push(Constraint::PointLineDistance { p: pp, a: la, b: lb, value, offset: 0.0 });
    Some(sketch.constraints.len() - 1)
}

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

/// Whether two line entities are (near) parallel — used to decide between a perpendicular
/// distance dimension (parallel) and an angle dimension (not).
fn lines_parallel(sketch: &Sketch, l1: usize, l2: usize) -> bool {
    let dir = |l: usize| {
        entity_line(sketch, l).and_then(|(a, b)| {
            let va = Vec2::new(sketch.points[a].x as f32, sketch.points[a].y as f32);
            let vb = Vec2::new(sketch.points[b].x as f32, sketch.points[b].y as f32);
            (vb - va).try_normalize()
        })
    };
    match (dir(l1), dir(l2)) {
        (Some(d1), Some(d2)) => (d1.x * d2.y - d1.y * d2.x).abs() < 0.08, // ~4.5°
        _ => false,
    }
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

// ---- Trim Entities (trim to closest) ----

fn pt2(sketch: &Sketch, i: usize) -> Vec2 {
    Vec2::new(sketch.points[i].x as f32, sketch.points[i].y as f32)
}

/// Parameter `t` ∈ (0,1) along segment a→b where it actually crosses segment c→d (within both),
/// or `None`.
fn seg_seg_t(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Option<f32> {
    let (r, s) = (b - a, d - c);
    let denom = r.x * s.y - r.y * s.x;
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denom;
    let u = ((c.x - a.x) * r.y - (c.y - a.y) * r.x) / denom;
    (t > 1e-3 && t < 1.0 - 1e-3 && u >= -1e-3 && u <= 1.0 + 1e-3).then_some(t)
}

/// Parameters along segment a→b where it crosses circle (`center`, `r`), within the segment.
fn seg_circle_t(a: Vec2, b: Vec2, center: Vec2, r: f32) -> Vec<f32> {
    let d = b - a;
    let aa = d.dot(d);
    if aa < 1e-9 {
        return vec![];
    }
    let f = a - center;
    let bb = 2.0 * f.dot(d);
    let cc = f.dot(f) - r * r;
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return vec![];
    }
    let sq = disc.sqrt();
    [(-bb - sq) / (2.0 * aa), (-bb + sq) / (2.0 * aa)]
        .into_iter()
        .filter(|&t| t > 1e-3 && t < 1.0 - 1e-3)
        .collect()
}

/// Tessellated polyline of any sketch entity, for trim intersection tests. Closed shapes
/// (circle, slot, closed spline) include the wrap segment back to the start. Reference lines,
/// text and bare points return empty (they don't bound a trim).
fn entity_polyline_app(sketch: &Sketch, ei: usize) -> Vec<Vec2> {
    let v2 = |p: &[f64; 2]| Vec2::new(p[0] as f32, p[1] as f32);
    match &sketch.entities[ei] {
        SketchEntity::Line { a, b, reference: false, .. } => vec![pt2(sketch, *a), pt2(sketch, *b)],
        SketchEntity::Circle { center, radius, .. } => {
            let (c, r) = (pt2(sketch, *center), *radius as f32);
            (0..=96).map(|k| { let a = std::f32::consts::TAU * k as f32 / 96.0; c + Vec2::new(a.cos(), a.sin()) * r }).collect()
        }
        SketchEntity::Arc { center, a, b, ccw, .. } => match (sketch.points.get(*center), sketch.points.get(*a), sketch.points.get(*b)) {
            (Some(c), Some(pa), Some(pb)) => tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw).iter().map(v2).collect(),
            _ => vec![],
        },
        SketchEntity::Spline { points, closed, control, .. } => {
            let pts: Vec<[f64; 2]> = points.iter().filter_map(|&i| sketch.points.get(i)).map(|p| [p.x, p.y]).collect();
            if pts.len() < 2 {
                return vec![];
            }
            let mut poly: Vec<Vec2> = tessellate_spline(&pts, *closed, *control).iter().map(v2).collect();
            if *closed && poly.len() >= 2 {
                poly.push(poly[0]);
            }
            poly
        }
        SketchEntity::Slot { a, b, radius, mid, .. } => match (sketch.points.get(*a), sketch.points.get(*b)) {
            (Some(pa), Some(pb)) => {
                let pm = mid.and_then(|m| sketch.points.get(m)).map(|p| [p.x, p.y]);
                let mut poly: Vec<Vec2> = match pm {
                    Some(pm) => tessellate_arc_slot([pa.x, pa.y], pm, [pb.x, pb.y], *radius),
                    None => tessellate_slot([pa.x, pa.y], [pb.x, pb.y], *radius),
                }
                .iter()
                .map(v2)
                .collect();
                if poly.len() >= 2 {
                    poly.push(poly[0]); // close the stadium loop
                }
                poly
            }
            _ => vec![],
        },
        _ => vec![],
    }
}

/// Param `t∈(0,1)` values along segment `a→b` where it crosses entity `ei`'s polyline (any type).
fn seg_entity_ts(sketch: &Sketch, a: Vec2, b: Vec2, ei: usize) -> Vec<f32> {
    entity_polyline_app(sketch, ei)
        .windows(2)
        .filter_map(|w| seg_seg_t(a, b, w[0], w[1]))
        .collect()
}

/// The nearest (non-reference) line to `uv` within `thresh`, by perpendicular segment distance.
fn nearest_line(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in sketch.entities.iter().enumerate() {
        if let SketchEntity::Line { a, b, reference: false, .. } = e {
            let (pa, pb) = (pt2(sketch, *a), pt2(sketch, *b));
            let ab = pb - pa;
            let l2 = ab.dot(ab);
            let t = if l2 > 1e-9 { ((uv - pa).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = uv.distance(pa + ab * t);
            if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// For the line nearest `uv`, the sub-segment (params `t_lo`..`t_hi` along it) that "trim to
/// closest" would delete — bracketed by the nearest intersections with other entities on each
/// side of the click. Used for both the red hover preview and the actual trim.
fn trim_bracket(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<(usize, f32, f32)> {
    let li = nearest_line(sketch, uv, thresh)?;
    let (ia, ib) = entity_line(sketch, li)?;
    let (a, b) = (pt2(sketch, ia), pt2(sketch, ib));
    let ab = b - a;
    let l2 = ab.dot(ab);
    if l2 < 1e-9 {
        return None;
    }
    let t_click = ((uv - a).dot(ab) / l2).clamp(0.0, 1.0);
    let mut ts: Vec<f32> = Vec::new();
    for (j, e) in sketch.entities.iter().enumerate() {
        if j == li {
            continue;
        }
        match e {
            SketchEntity::Line { a: c, b: d, .. } => {
                if let Some(t) = seg_seg_t(a, b, pt2(sketch, *c), pt2(sketch, *d)) {
                    ts.push(t);
                }
            }
            SketchEntity::Circle { center, radius, .. } => {
                ts.extend(seg_circle_t(a, b, pt2(sketch, *center), *radius as f32));
            }
            // Arcs, splines and slots bound a trim via their tessellated outline.
            SketchEntity::Arc { .. } | SketchEntity::Spline { .. } | SketchEntity::Slot { .. } => {
                ts.extend(seg_entity_ts(sketch, a, b, j));
            }
            _ => {}
        }
    }
    let t_lo = ts.iter().copied().filter(|&t| t < t_click - 1e-4).fold(0.0_f32, f32::max);
    let t_hi = ts.iter().copied().filter(|&t| t > t_click + 1e-4).fold(1.0_f32, f32::min);
    Some((li, t_lo, t_hi))
}

/// Intersection points of two circles (0, 1, or 2).
fn circle_circle_pts(c1: Vec2, r1: f32, c2: Vec2, r2: f32) -> Vec<Vec2> {
    let d = c1.distance(c2);
    if d < 1e-6 || d > r1 + r2 + 1e-4 || d < (r1 - r2).abs() - 1e-4 {
        return vec![];
    }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h2 = r1 * r1 - a * a;
    let mid = c1 + (c2 - c1) * (a / d);
    if h2 <= 1e-9 {
        return vec![mid];
    }
    let h = h2.sqrt();
    let perp = Vec2::new(-(c2.y - c1.y), c2.x - c1.x) / d;
    vec![mid + perp * h, mid - perp * h]
}

/// The angles (about the circle's centre) where every *other* entity crosses circle `ci`'s rim.
fn circle_cut_angles(sketch: &Sketch, ci: usize) -> Vec<f32> {
    match &sketch.entities[ci] {
        SketchEntity::Circle { center, radius, .. } => cut_angles_on_circle(sketch, pt2(sketch, *center), *radius as f32, ci),
        _ => vec![],
    }
}

/// Crossing angles of every entity except `skip` against the circle (`center`, `r`). Shared by
/// circle trimming and arc trimming (an arc rides on its own circle).
fn cut_angles_on_circle(sketch: &Sketch, center: Vec2, r: f32, skip: usize) -> Vec<f32> {
    let ci = skip;
    let ang = |p: Vec2| (p.y - center.y).atan2(p.x - center.x).rem_euclid(std::f32::consts::TAU);
    let on_tol = (r * 5.0e-3).max(1.0e-3); // a point "on the rim" (an endpoint snapped to it)
    let on_rim = |p: Vec2| ((p - center).length() - r).abs() <= on_tol;
    let mut out = Vec::new();
    for (j, e) in sketch.entities.iter().enumerate() {
        if j == ci {
            continue;
        }
        match e {
            SketchEntity::Line { a, b, reference: false, .. } => {
                let (pa, pb) = (pt2(sketch, *a), pt2(sketch, *b));
                for t in seg_circle_t(pa, pb, center, r) {
                    out.push(ang(pa + (pb - pa) * t));
                }
                // An endpoint lying on the rim (e.g. a chord whose end was trimmed to the circle)
                // is itself a cut — seg_circle_t excludes segment ends, so add them here.
                for p in [pa, pb] {
                    if on_rim(p) {
                        out.push(ang(p));
                    }
                }
            }
            SketchEntity::Circle { center: c2, radius: r2, .. } => {
                for p in circle_circle_pts(center, r, pt2(sketch, *c2), *r2 as f32) {
                    out.push(ang(p));
                }
            }
            SketchEntity::Arc { center: c2, a, b, ccw, .. } => {
                if let (Some(cc), Some(pa), Some(pb)) = (sketch.points.get(*c2), sketch.points.get(*a), sketch.points.get(*b)) {
                    let poly = tessellate_arc([cc.x, cc.y], [pa.x, pa.y], [pb.x, pb.y], *ccw);
                    for w in poly.windows(2) {
                        let (w0, w1) = (Vec2::new(w[0][0] as f32, w[0][1] as f32), Vec2::new(w[1][0] as f32, w[1][1] as f32));
                        for t in seg_circle_t(w0, w1, center, r) {
                            out.push(ang(w0 + (w1 - w0) * t));
                        }
                    }
                    for p in [pt2(sketch, *a), pt2(sketch, *b)] {
                        if on_rim(p) {
                            out.push(ang(p));
                        }
                    }
                }
            }
            // Splines and slots cross the rim along their tessellated outline.
            SketchEntity::Spline { .. } | SketchEntity::Slot { .. } => {
                for w in entity_polyline_app(sketch, j).windows(2) {
                    for t in seg_circle_t(w[0], w[1], center, r) {
                        out.push(ang(w[0] + (w[1] - w[0]) * t));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Nearest circle (by rim distance) within `thresh`.
fn nearest_circle(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in sketch.entities.iter().enumerate() {
        if let SketchEntity::Circle { center, radius, .. } = e {
            let d = ((uv - pt2(sketch, *center)).length() - *radius as f32).abs();
            if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// The removed angular interval `(lo, hi)` (CCW, containing the click) for trimming circle `ci`,
/// or `None` if it has fewer than two cuts (then the whole circle would be deleted).
fn circle_trim_interval(sketch: &Sketch, ci: usize, uv: Vec2) -> Option<(f32, f32)> {
    let center = match &sketch.entities[ci] {
        SketchEntity::Circle { center, .. } => pt2(sketch, *center),
        _ => return None,
    };
    let mut angs = circle_cut_angles(sketch, ci);
    if angs.len() < 2 {
        return None;
    }
    angs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    angs.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
    let tau = std::f32::consts::TAU;
    let click = (uv.y - center.y).atan2(uv.x - center.x).rem_euclid(tau);
    // Find the consecutive pair (CCW) bracketing the click.
    for i in 0..angs.len() {
        let lo = angs[i];
        let hi = if i + 1 < angs.len() { angs[i + 1] } else { angs[0] + tau };
        let c = if click < lo { click + tau } else { click };
        if c >= lo && c <= hi {
            return Some((lo, hi)); // hi may exceed τ (the wrap interval); cos/sin handle it
        }
    }
    None
}

/// Perform a trim-to-closest at `uv`: pick the nearest line *or* circle and drop the piece the
/// click is on, leaving the survivor(s) (a trimmed line, or a circle → arc). Returns true if
/// anything changed.
fn apply_trim(session: &mut SketchSession, uv: Vec2, thresh: f32) -> bool {
    let line = nearest_line(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
    let circ = nearest_circle(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
    let arc = nearest_arc(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
    let spline = nearest_spline(&session.sketch, uv, thresh);
    // Prefer whichever rim/segment the cursor is closest to.
    let pick = [line, circ, arc, spline].into_iter().flatten().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let Some((ei, _)) = pick else { return false };

    if let SketchEntity::Spline { .. } = session.sketch.entities[ei] {
        return trim_spline(session, ei, uv);
    }

    if let SketchEntity::Arc { .. } = session.sketch.entities[ei] {
        // Arc → trim to the nearest cuts, leaving the surviving sub-arc(s).
        let Some((center_i, center, r, theta_start, span)) = arc_geom(&session.sketch, ei) else { return false };
        let construction = matches!(&session.sketch.entities[ei], SketchEntity::Arc { construction: true, .. });
        let ccw = span > 0.0;
        let snap = session.snap_dist;
        let at_u = |u: f32| {
            let th = theta_start + span * u;
            center + Vec2::new(th.cos(), th.sin()) * r
        };
        if let Some((u_lo, u_hi)) = arc_trim_bracket(&session.sketch, ei, uv) {
            session.sketch.entities.remove(ei);
            if u_lo > 1e-3 {
                let p0 = get_or_add_point(&mut session.sketch, at_u(0.0), snap);
                let p1 = get_or_add_point(&mut session.sketch, at_u(u_lo), snap);
                session.sketch.add_arc(center_i, p0, p1, ccw, construction);
            }
            if u_hi < 1.0 - 1e-3 {
                let p0 = get_or_add_point(&mut session.sketch, at_u(u_hi), snap);
                let p1 = get_or_add_point(&mut session.sketch, at_u(1.0), snap);
                session.sketch.add_arc(center_i, p0, p1, ccw, construction);
            }
        } else {
            session.sketch.entities.remove(ei);
        }
        session.sketch.remove_unused_points();
        session.dirty = true;
        return true;
    }

    if let SketchEntity::Circle { .. } = session.sketch.entities[ei] {
        // Circle → arc (keep the complement of the clicked interval), or delete if <2 cuts.
        let center = match &session.sketch.entities[ei] { SketchEntity::Circle { center, .. } => *center, _ => return false };
        let r = match &session.sketch.entities[ei] { SketchEntity::Circle { radius, .. } => *radius, _ => return false };
        let construction = matches!(&session.sketch.entities[ei], SketchEntity::Circle { construction: true, .. });
        let snap = session.snap_dist;
        match circle_trim_interval(&session.sketch, ei, uv) {
            Some((lo, hi)) => {
                let c = pt2(&session.sketch, center);
                let at = |ang: f32| c + Vec2::new(ang.cos(), ang.sin()) * r as f32;
                session.sketch.entities.remove(ei);
                let p_hi = get_or_add_point(&mut session.sketch, at(hi), snap);
                let p_lo = get_or_add_point(&mut session.sketch, at(lo), snap);
                // Kept arc = from hi CCW round to lo (the part NOT under the click).
                session.sketch.add_arc(center, p_hi, p_lo, true, construction);
            }
            None => {
                session.sketch.entities.remove(ei); // no bracketing cuts → remove the whole circle
            }
        }
        session.sketch.remove_unused_points();
        session.dirty = true;
        return true;
    }

    // Line trim (the original behaviour).
    let Some((_, t_lo, t_hi)) = trim_bracket(&session.sketch, uv, thresh) else { return false };
    let Some((ia, ib)) = entity_line(&session.sketch, ei) else { return false };
    let (a, b) = (pt2(&session.sketch, ia), pt2(&session.sketch, ib));
    let construction = matches!(&session.sketch.entities[ei], SketchEntity::Line { construction: true, .. });
    let snap = session.snap_dist;
    session.sketch.entities.remove(ei);
    if t_lo > 1e-3 {
        let p = get_or_add_point(&mut session.sketch, a + (b - a) * t_lo, snap);
        session.sketch.add_line(ia, p, construction);
    }
    if t_hi < 1.0 - 1e-3 {
        let p = get_or_add_point(&mut session.sketch, a + (b - a) * t_hi, snap);
        session.sketch.add_line(p, ib, construction);
    }
    session.sketch.remove_unused_points();
    session.dirty = true;
    true
}

/// Distance from `uv` to entity `i` (line segment or circle rim); `MAX` for others.
fn dist_to_entity(sketch: &Sketch, i: usize, uv: Vec2) -> f32 {
    match &sketch.entities[i] {
        SketchEntity::Line { a, b, .. } => {
            let (pa, pb) = (pt2(sketch, *a), pt2(sketch, *b));
            let ab = pb - pa;
            let t = if ab.length_squared() > 1e-9 { ((uv - pa).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
            uv.distance(pa + ab * t)
        }
        SketchEntity::Circle { center, radius, .. } => ((uv - pt2(sketch, *center)).length() - *radius as f32).abs(),
        SketchEntity::Arc { center, a, b, ccw, .. } => {
            match (sketch.points.get(*center), sketch.points.get(*a), sketch.points.get(*b)) {
                (Some(c), Some(pa), Some(pb)) => {
                    let poly = tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw);
                    let mut dmin = f32::MAX;
                    for w in poly.windows(2) {
                        let a2 = Vec2::new(w[0][0] as f32, w[0][1] as f32);
                        let b2 = Vec2::new(w[1][0] as f32, w[1][1] as f32);
                        let ab = b2 - a2;
                        let t = if ab.length_squared() > 1e-9 { ((uv - a2).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
                        dmin = dmin.min((uv - (a2 + ab * t)).length());
                    }
                    dmin
                }
                _ => f32::MAX,
            }
        }
        _ => f32::MAX,
    }
}

/// (centre point index, centre, radius, θ_start, signed span) of arc entity `ai`.
fn arc_geom(sketch: &Sketch, ai: usize) -> Option<(usize, Vec2, f32, f32, f32)> {
    if let SketchEntity::Arc { center, a, b, ccw, .. } = &sketch.entities[ai] {
        let c = pt2(sketch, *center);
        let pa = pt2(sketch, *a);
        let pb = pt2(sketch, *b);
        let r = (pa - c).length();
        if r < 1e-6 {
            return None;
        }
        let tau = std::f32::consts::TAU;
        let ta = (pa.y - c.y).atan2(pa.x - c.x);
        let tb = (pb.y - c.y).atan2(pb.x - c.x);
        let mut span = if *ccw { (tb - ta).rem_euclid(tau) } else { -((ta - tb).rem_euclid(tau)) };
        if span.abs() < 1e-6 {
            span = if *ccw { tau } else { -tau };
        }
        Some((*center, c, r, ta, span))
    } else {
        None
    }
}

/// Param u∈(0,1) along an arc (θ_start, signed `span`) at angle φ, if it lies on the arc interior.
fn arc_param(theta_start: f32, span: f32, phi: f32) -> Option<f32> {
    let tau = std::f32::consts::TAU;
    let along = if span >= 0.0 { (phi - theta_start).rem_euclid(tau) } else { (theta_start - phi).rem_euclid(tau) };
    let u = along / span.abs();
    (u > 1e-3 && u < 1.0 - 1e-3).then_some(u)
}

/// Nearest arc (by distance to its rim) within `thresh`.
fn nearest_arc(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in sketch.entities.iter().enumerate() {
        if matches!(e, SketchEntity::Arc { .. }) {
            let d = dist_to_entity(sketch, i, uv);
            if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
    }
    best.map(|(i, _)| i)
}

/// Nearest spline (by distance to its tessellated outline) within `thresh` → (index, dist).
fn nearest_spline(sketch: &Sketch, uv: Vec2, thresh: f32) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, e) in sketch.entities.iter().enumerate() {
        if !matches!(e, SketchEntity::Spline { .. }) {
            continue;
        }
        let poly = entity_polyline_app(sketch, i);
        let mut d = f32::MAX;
        for w in poly.windows(2) {
            let ab = w[1] - w[0];
            let l2 = ab.dot(ab);
            let t = if l2 > 1e-9 { ((uv - w[0]).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
            d = d.min(uv.distance(w[0] + ab * t));
        }
        if d <= thresh && best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best
}

/// Bracket (u_lo, u_hi) of arc `ai` to remove at `uv` — the piece between the nearest cuts on
/// each side of the click. u_lo=0 / u_hi=1 mean "up to the arc's own end".
fn arc_trim_bracket(sketch: &Sketch, ai: usize, uv: Vec2) -> Option<(f32, f32)> {
    let (_, center, r, theta_start, span) = arc_geom(sketch, ai)?;
    let mut us: Vec<f32> = cut_angles_on_circle(sketch, center, r, ai)
        .into_iter()
        .filter_map(|phi| arc_param(theta_start, span, phi))
        .collect();
    let click = arc_param(theta_start, span, (uv.y - center.y).atan2(uv.x - center.x))?;
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let u_lo = us.iter().copied().filter(|&u| u < click - 1e-4).fold(0.0_f32, f32::max);
    let u_hi = us.iter().copied().filter(|&u| u > click + 1e-4).fold(1.0_f32, f32::min);
    Some((u_lo, u_hi))
}

/// The first point at which the stroke segment `prev`→`cur` crosses any (non-reference) entity —
/// used by Power Trim to find what the dragged cursor is cutting.
fn stroke_crosses(sketch: &Sketch, prev: Vec2, cur: Vec2) -> Option<Vec2> {
    for (j, e) in sketch.entities.iter().enumerate() {
        match e {
            SketchEntity::Line { reference: true, .. } | SketchEntity::Point { .. } | SketchEntity::Text { .. } => {}
            // Circles need the exact rim crossing; everything else rides its tessellated outline.
            SketchEntity::Circle { center, radius, .. } => {
                if let Some(t) = seg_circle_t(prev, cur, pt2(sketch, *center), *radius as f32).into_iter().next() {
                    return Some(prev + (cur - prev) * t);
                }
            }
            _ => {
                if let Some(t) = seg_entity_ts(sketch, prev, cur, j).into_iter().next() {
                    return Some(prev + (cur - prev) * t);
                }
            }
        }
    }
    None
}

/// Power Trim one stroke step: if the cursor's path `prev`→`cur` crossed an entity, trim that
/// entity at the crossing (trim-to-closest there).
fn power_trim_stroke(session: &mut SketchSession, prev: Vec2, cur: Vec2, thresh: f32) {
    if let Some(p) = stroke_crosses(&session.sketch, prev, cur) {
        apply_trim(session, p, thresh.max(0.05));
    }
}

/// Corner Trim: trim/extend lines `li1` and `li2` so they meet at their intersection. Each line's
/// endpoint nearest the intersection is moved onto a shared point there (forming a clean corner).
fn corner_trim(session: &mut SketchSession, li1: usize, li2: usize) -> bool {
    let (Some((a0, a1)), Some((b0, b1))) = (entity_line(&session.sketch, li1), entity_line(&session.sketch, li2)) else {
        return false;
    };
    let (pa0, pa1, pb0, pb1) = (pt2(&session.sketch, a0), pt2(&session.sketch, a1), pt2(&session.sketch, b0), pt2(&session.sketch, b1));
    let Some(x) = line_intersection(pa0, pa1, pb0, pb1) else { return false }; // parallel → no corner
    let snap = session.snap_dist;
    let xi = get_or_add_point(&mut session.sketch, x, snap);
    // Re-point each line's endpoint nearest the intersection onto the shared corner point.
    let near1 = if pa0.distance(x) <= pa1.distance(x) { a0 } else { a1 };
    let near2 = if pb0.distance(x) <= pb1.distance(x) { b0 } else { b1 };
    for (li, near) in [(li1, near1), (li2, near2)] {
        if let SketchEntity::Line { a, b, .. } = &mut session.sketch.entities[li] {
            if *a == near {
                *a = xi;
            } else if *b == near {
                *b = xi;
            }
        }
    }
    session.sketch.remove_unused_points();
    session.dirty = true;
    true
}

/// Trim a spline `ei` at `uv`: drop the sub-curve between the nearest crossings on either side of
/// the click, rebuilding the survivor(s) as splines through the original through-points plus the
/// cut endpoints. An open spline leaves up to two pieces; a closed spline opens into the single
/// complementary arc. (Slots aren't trimmed as targets yet.)
fn trim_spline(session: &mut SketchSession, ei: usize, uv: Vec2) -> bool {
    const STEPS: usize = 16; // must match tessellate_spline's samples-per-segment
    let (pt_idx, closed, control, construction) = match &session.sketch.entities[ei] {
        SketchEntity::Spline { points, closed, control, construction } => (points.clone(), *closed, *control, *construction),
        _ => return false,
    };
    if pt_idx.len() < 3 {
        return false;
    }
    let through: Vec<Vec2> = pt_idx.iter().map(|&i| pt2(&session.sketch, i)).collect();
    let poly = entity_polyline_app(&session.sketch, ei); // open ⇒ (n-1)*STEPS+1, closed ⇒ n*STEPS+1
    if poly.len() < 2 {
        return false;
    }
    let span = (poly.len() - 1) as f32; // total parameter length; through-point i sits at i*STEPS
    let step = STEPS as f32;
    // Position along the poly is "segment index + t".
    let project = |p: Vec2| -> f32 {
        let (mut pos, mut best) = (0.0_f32, f32::MAX);
        for (k, w) in poly.windows(2).enumerate() {
            let ab = w[1] - w[0];
            let l2 = ab.dot(ab);
            let t = if l2 > 1e-9 { ((p - w[0]).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = p.distance(w[0] + ab * t);
            if d < best {
                best = d;
                pos = k as f32 + t;
            }
        }
        pos
    };
    let click_pos = project(uv);
    // Every crossing of this spline's outline with another entity → (pos, point).
    let mut cuts: Vec<(f32, Vec2)> = Vec::new();
    for j in 0..session.sketch.entities.len() {
        if j == ei {
            continue;
        }
        for w in entity_polyline_app(&session.sketch, j).windows(2) {
            for (k, ws) in poly.windows(2).enumerate() {
                if let Some(t) = seg_seg_t(ws[0], ws[1], w[0], w[1]) {
                    cuts.push((k as f32 + t, ws[0] + (ws[1] - ws[0]) * t));
                }
            }
        }
    }
    let snap = session.snap_dist;
    let push_spline = |session: &mut SketchSession, pts: &[Vec2]| {
        if pts.len() < 2 {
            return;
        }
        let idx: Vec<usize> = pts.iter().map(|p| get_or_add_point(&mut session.sketch, *p, snap)).collect();
        session.sketch.entities.push(SketchEntity::Spline { points: idx, closed: false, construction, control });
    };

    if !closed {
        // Open spline: bracket the click between the nearest crossings (or the spline's own ends).
        let lo = cuts.iter().filter(|(p, _)| *p < click_pos - 1e-3).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).copied();
        let hi = cuts.iter().filter(|(p, _)| *p > click_pos + 1e-3).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).copied();
        if lo.is_none() && hi.is_none() {
            return false; // no bounding crossings → nothing to trim
        }
        let lo_pos = lo.map(|(p, _)| p).unwrap_or(0.0);
        let hi_pos = hi.map(|(p, _)| p).unwrap_or(span);
        let mut left: Vec<Vec2> = through.iter().enumerate().filter(|(i, _)| (*i as f32) * step < lo_pos - 0.5).map(|(_, &p)| p).collect();
        if let Some((_, cp)) = lo {
            left.push(cp);
        }
        let mut right: Vec<Vec2> = Vec::new();
        if let Some((_, cp)) = hi {
            right.push(cp);
        }
        right.extend(through.iter().enumerate().filter(|(i, _)| (*i as f32) * step > hi_pos + 0.5).map(|(_, &p)| p));
        session.sketch.entities.remove(ei);
        push_spline(session, &left);
        push_spline(session, &right);
    } else {
        // Closed spline: need two crossings to bound a removable arc. Bracket the click
        // *circularly*; the survivor is the complementary arc opened into a single spline.
        if cuts.len() < 2 {
            return false;
        }
        let mut sorted = cuts.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (hi_pos, hi_pt) = sorted.iter().find(|(p, _)| *p > click_pos).copied().unwrap_or((sorted[0].0 + span, sorted[0].1));
        let (lo_pos, lo_pt) = sorted.iter().rev().find(|(p, _)| *p < click_pos).copied().unwrap_or((sorted.last().unwrap().0 - span, sorted.last().unwrap().1));
        // Survivor runs from hi_pos up to lo_pos+span (the arc NOT under the click). Collect the
        // through-points whose position (or its +span image) lands inside that open window.
        let mut surv: Vec<(f32, Vec2)> = Vec::new();
        for (i, &p) in through.iter().enumerate() {
            let base = i as f32 * step;
            for img in [base, base + span] {
                if img > hi_pos + 0.5 && img < lo_pos + span - 0.5 {
                    surv.push((img, p));
                }
            }
        }
        surv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut pts = vec![hi_pt];
        pts.extend(surv.iter().map(|(_, p)| *p));
        pts.push(lo_pt);
        session.sketch.entities.remove(ei);
        push_spline(session, &pts);
    }
    session.sketch.remove_unused_points();
    session.dirty = true;
    true
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

/// Circumcentre of three points, or `None` if (nearly) collinear.
fn circumcenter(a: Vec2, b: Vec2, c: Vec2) -> Option<Vec2> {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-6 {
        return None;
    }
    let (a2, b2, c2) = (a.length_squared(), b.length_squared(), c.length_squared());
    Some(Vec2::new(
        (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d,
        (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d,
    ))
}

/// Commit a 3-point arc from `start` to `end` passing through `through`. Falls back to a line if
/// the three points are collinear.
fn commit_arc(session: &mut SketchSession, start: Vec2, end: Vec2, through: Vec2) {
    let snap = session.snap_dist;
    let Some(center) = circumcenter(start, end, through) else {
        let a = get_or_add_point_ref(session, start, snap);
        let b = get_or_add_point_ref(session, end, snap);
        session.sketch.add_line(a, b, session.construction);
        session.construction = false;
        session.dirty = true;
        return;
    };
    let tau = std::f32::consts::TAU;
    let ang = |p: Vec2| (p.y - center.y).atan2(p.x - center.x).rem_euclid(tau);
    let (ta, tb, tt) = (ang(start), ang(end), ang(through));
    // CCW if `through` lies on the CCW sweep from start to end.
    let ccw = (tt - ta).rem_euclid(tau) < (tb - ta).rem_euclid(tau);
    let ci = get_or_add_point(&mut session.sketch, center, snap);
    let a = get_or_add_point_ref(session, start, snap);
    let b = get_or_add_point_ref(session, end, snap);
    session.sketch.add_arc(ci, a, b, ccw, session.construction);
    session.construction = false;
    session.dirty = true;
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
                // The dotted "even with another point" guides are inference only — they snap the
                // cursor onto the alignment but do NOT create a relation (SolidWorks-style). Auto-
                // capturing them piled up Horizontal/Vertical constraints between arbitrary point
                // pairs and quickly over-defined the sketch (relations collapsing / going wonky).
                // A line's OWN axis-alignment and perpendicular joints are still captured below —
                // those are the standard, wanted auto-relations.
                session.start_infer = (None, None);
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
                // Persist the square/perpendicular relation the 90° snap implied, so resizing
                // keeps the sketch square — but ONLY when neither endpoint is pinned to a
                // curve/edge. A projected edge is never *exactly* axis-aligned (tessellation
                // tilts it by microns), so PointOnLine + an exact Vertical intersect at a single
                // point: the solver slides the endpoint there — collapsing the line onto the
                // corner or shooting it past the body. The edge pin alone defines the direction.
                let pinned = |sk: &Sketch, p: usize| {
                    sk.constraints.iter().any(|c| match c {
                        Constraint::PointOnLine { p: q, .. }
                        | Constraint::PointOnArc { p: q, .. }
                        | Constraint::PointOnCircle { p: q, .. } => *q == p,
                        _ => false,
                    })
                };
                if !pinned(&session.sketch, a) && !pinned(&session.sketch, b) {
                    add_square_relations(&mut session.sketch, a, b);
                }
                session.dirty = true;
                // A construction line is a one-shot: revert to the regular line tool after
                // drawing one (re-pick "Construction Line" for another).
                session.construction = false;
            } else {
                session.pending = Some(uv);
                session.pending_edge = session.cursor_edge; // remember the start's edge
                session.start_infer = (session.infer_v, session.infer_h); // capture start's alignment
                session.request_live_focus = true;
            }
        }
        Tool::Circle if session.circle_perimeter => {
            // Perimeter circle: the two clicks are opposite ends of a diameter, so the
            // centre is their midpoint and the radius is half the distance.
            if let Some(p1) = session.pending.take() {
                let center = (p1 + uv) * 0.5;
                let radius = ((uv - p1).length() * 0.5).max(0.01);
                let c = add_circle_center(session, center, snap);
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
                let c = add_circle_center(session, center, snap);
                session.sketch.add_circle(c, radius as f64);
                session.dirty = true;
            } else {
                session.pending = Some(uv);
                session.request_live_focus = true;
            }
        }
        Tool::Arc => {
            // 3-point arc: 1) start, 2) end, 3) a point the arc passes through → commit.
            if session.pending.is_none() {
                session.pending = Some(uv);
                session.request_live_focus = true;
            } else if session.pending_b.is_none() {
                session.pending_b = Some(uv);
            } else {
                let start = session.pending.take().unwrap();
                let end = session.pending_b.take().unwrap();
                commit_arc(session, start, end, uv);
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
        Tool::Select | Tool::Dimension | Tool::Spline | Tool::Pattern | Tool::Mirror | Tool::Trim => {}
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
fn toggle_fillet_edge(ui_state: &mut UiState, chain: &[Vec3], closed: bool) {
    // A closed loop is stored with its first point REPEATED, so the closing segment is explicit.
    // Without it the bevel never sees the loop's last edge: one rim edge stays square (a notch)
    // and its seam detours around the corner (the "chevron sticking out"), and the missing edge
    // makes the mesh surgery fail → coarse CSG fallback. The duplicated point is safe for the
    // engine (`rim_pick_from_tessellation_closes_the_loop` pins this).
    let mut poly: Vec<[f64; 3]> = chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect();
    if closed && poly.len() >= 3 {
        poly.push(poly[0]);
    }
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
    mut ui_state: ResMut<UiState>,
    blocking: Res<UiBlocking>,
) {
    // Escape leaves the measure tool (works in view mode, where the sketch shortcuts below don't run).
    if keys.just_pressed(KeyCode::Escape) && ui_state.measuring && !blocking.1 {
        ui_state.measuring = false;
        ui_state.measure_pts.clear();
    }
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
        session.op_request = Some(SolidOp::Boss(EXTRUDE_DISTANCE, 0.0, 0.0, 0));
    }
    if keys.just_pressed(KeyCode::KeyD) {
        session.op_request = Some(SolidOp::Cut(EXTRUDE_DISTANCE, 0.0, 0.0, 0));
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
            // 0) Finish the in-progress spline at the current segment (open) rather than
            // discarding it — ≥2 points make a real spline whose endpoints stay snappable.
            if session.spline_pts.len() >= 2 {
                commit_spline(&mut session, false);
            } else {
                session.spline_pts.clear(); // a lone point: nothing to keep
            }
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
    part: Res<Part>,
) {
    // Force a `.hcad` extension on a chosen path (so Save As without a typed extension still saves a part).
    let with_hcad = |mut p: std::path::PathBuf| {
        if p.extension().is_none() {
            p.set_extension("hcad");
        }
        p
    };
    let write_doc = |doc: &Document, path: &std::path::Path| -> bool {
        match ron::ser::to_string_pretty(doc, ron::ser::PrettyConfig::default()) {
            Ok(text) => match std::fs::write(path, text) {
                Ok(()) => {
                    info!("Saved {}", path.display());
                    true
                }
                Err(e) => {
                    warn!("Save failed: {e}");
                    false
                }
            },
            Err(e) => {
                warn!("Serialize failed: {e}");
                false
            }
        }
    };

    // Save: write to the bound file directly; if there isn't one yet, fall through to Save As.
    if ui_state.save_request {
        ui_state.save_request = false;
        match ui_state.current_file.clone() {
            Some(path) => {
                if write_doc(&doc.0, &path) {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("part").to_string();
                    ui_state.toasts.push((format!("Saved {name}"), 2.5));
                } else {
                    ui_state.last_error = Some("Save failed — see the log.".into());
                }
            }
            None => ui_state.save_as_request = true,
        }
    }
    // Save As: always prompt; bind the chosen path so later Saves are dialog-free.
    if ui_state.save_as_request {
        ui_state.save_as_request = false;
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("HCAD part", &["hcad"])
            .set_file_name(ui_state.current_file.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("part.hcad"))
            .save_file()
        {
            let path = with_hcad(path);
            if write_doc(&doc.0, &path) {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("part").to_string();
                ui_state.current_file = Some(path);
                ui_state.toasts.push((format!("Saved {name}"), 2.5));
            } else {
                ui_state.last_error = Some("Save failed — see the log.".into());
            }
        }
    }

    // Export STL — the mesh body (works for any part).
    if ui_state.export_stl_request {
        ui_state.export_stl_request = false;
        match &part.mesh {
            Some(mesh) if !mesh.positions.is_empty() => {
                // Default the export name to the part's saved name (part.hcad → part.stl).
                let stem = ui_state.current_file.as_ref().and_then(|p| p.file_stem()).and_then(|s| s.to_str()).unwrap_or("part");
                if let Some(mut path) = rfd::FileDialog::new().add_filter("STL mesh", &["stl"]).set_file_name(format!("{stem}.stl")).save_file() {
                    if path.extension().is_none() {
                        path.set_extension("stl");
                    }
                    match std::fs::write(&path, export_stl(mesh)) {
                        Ok(()) => {
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("part.stl").to_string();
                            ui_state.toasts.push((format!("Exported {name}"), 2.5));
                            info!("Exported STL {}", path.display());
                        }
                        Err(e) => {
                            warn!("STL export failed: {e}");
                            ui_state.last_error = Some(format!("STL export failed: {e}"));
                        }
                    }
                }
            }
            _ => ui_state.last_error = Some("Nothing to export — build a body first.".into()),
        }
    }

    // Export STEP — the exact B-rep when available; otherwise a faceted reconstruction from the
    // mesh (so a loft / fillet / Seamless body still exports, just faceted rather than smooth).
    if ui_state.export_step_request {
        ui_state.export_step_request = false;
        let faceted = part.solid.is_none();
        let solid = part.solid.clone().or_else(|| part.mesh.as_ref().and_then(|m| mesh_to_solid(m)));
        match solid.as_ref().and_then(export_step) {
            Some(step) => {
                let stem = ui_state.current_file.as_ref().and_then(|p| p.file_stem()).and_then(|s| s.to_str()).unwrap_or("part");
                if let Some(mut path) = rfd::FileDialog::new().add_filter("STEP", &["step", "stp"]).set_file_name(format!("{stem}.step")).save_file() {
                    if path.extension().is_none() {
                        path.set_extension("step");
                    }
                    match std::fs::write(&path, step) {
                        Ok(()) => {
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("part.step").to_string();
                            ui_state.toasts.push((format!("Exported {name}"), 2.5));
                            info!("Exported STEP {} ({})", path.display(), if faceted { "faceted from mesh" } else { "exact B-rep" });
                            if faceted {
                                ui_state.last_error = Some("Exported a FACETED STEP (this body has no exact B-rep — built with the mesh kernel). Geometry is correct but flat-faced; for smooth surfaces, build with Seamless off and no loft/fillet.".into());
                            }
                        }
                        Err(e) => {
                            warn!("STEP export failed: {e}");
                            ui_state.last_error = Some(format!("STEP export failed: {e}"));
                        }
                    }
                }
            }
            None => ui_state.last_error = Some("STEP export failed — no exportable body (build a part first).".into()),
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
                        ui_state.current_file = Some(path.clone());
                        ui_state.regen = true;
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("part").to_string();
                        ui_state.toasts.push((format!("Opened {name}"), 2.5));
                        info!("Opened {}", path.display());
                    }
                    Err(e) => {
                        warn!("Could not parse {}: {e}", path.display());
                        ui_state.last_error = Some(format!("Couldn't open {} — it isn't a valid HCAD part.", path.display()));
                    }
                },
                Err(e) => {
                    warn!("Could not read {}: {e}", path.display());
                    ui_state.last_error = Some(format!("Couldn't read {}: {e}", path.display()));
                }
            }
        }
    }

    // Insert a reference image: pick a PNG/JPG, embed it (base64), and pin it to the selected plane
    // (or Front if none). It's a non-geometry underlay, so no regen — the sync system spawns the quad.
    if ui_state.insert_image_request {
        ui_state.insert_image_request = false;
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "bmp", "gif", "webp"])
            .pick_file()
        {
            match std::fs::read(&path) {
                Ok(bytes) => match image::load_from_memory(&bytes) {
                    Ok(img) => {
                        use base64::Engine;
                        let (px_w, px_h) = (img.width().max(1), img.height().max(1));
                        let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
                        // Base the image on the selected datum plane, else Front.
                        let which = ui_state.selected_plane.unwrap_or(0) as u8;
                        let plane = standard_plane_ref(which);
                        // Default size: 100 mm on the longer side, centred on the plane origin.
                        let aspect = px_h as f64 / px_w as f64;
                        let (width, height) = if px_w >= px_h { (100.0, 100.0 * aspect) } else { (100.0 / aspect, 100.0) };
                        history.snapshot(&doc.0);
                        let id = doc.0.add_feature(FeatureKind::RefImage {
                            plane,
                            data,
                            px_w,
                            px_h,
                            center: [0.0, 0.0],
                            rot: 0.0,
                            width,
                            height,
                            opacity: 0.6,
                            flip_h: false,
                            flip_v: false,
                        });
                        // Open its PropertyManager (find its index).
                        ui_state.image_edit = doc.0.features.iter().position(|f| f.id.0 == id.0);
                        ui_state.image_lock_aspect = true;
                        info!("Inserted reference image {} ({px_w}×{px_h})", path.display());
                    }
                    Err(e) => {
                        warn!("Could not decode image {}: {e}", path.display());
                        ui_state.last_error = Some(format!("Could not read that image: {e}"));
                    }
                },
                Err(e) => {
                    warn!("Could not read image {}: {e}", path.display());
                    ui_state.last_error = Some(format!("Could not read the file: {e}"));
                }
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

    let region_count = session.cached_regions().len();
    if region_count == 0 {
        warn!("Need a closed profile (a loop of lines, or a circle) to extrude.");
        return;
    }
    // The chosen contours, or all of them if none were explicitly selected.
    let regions: Vec<usize> =
        session.selected_contours.iter().copied().filter(|&i| i < region_count).collect();

    // A body exists if there's geometry — either an exact B-rep solid *or* a mesh body
    // (after a fillet/seamless build, `part.solid` is None but `part.mesh` is the body).
    if matches!(op, SolidOp::Cut(..) | SolidOp::RevolveCut(_)) && part.solid.is_none() && part.mesh.is_none() {
        warn!("Cut: there is no body yet — extrude a boss first.");
        return;
    }
    // Revolve needs an axis — the line the user clicked in the sketch.
    let axis = revolve_axis(&session);
    if matches!(op, SolidOp::Revolve(_) | SolidOp::RevolveCut(_)) && axis.is_none() {
        warn!("Revolve: click a line in the sketch to use as the axis.");
        ui_state.last_error = Some("Revolve needs an axis — click a line in the sketch to select it.".into());
        return;
    }

    history.snapshot(&doc.0);
    let sketch = session.sketch.clone();
    let plane = plane_ref(&ap);
    let kind = match op {
        SolidOp::Boss(d, back, thin, thin_side) => FeatureKind::Extrude { sketch, regions, plane, distance: d, back, thin, thin_side },
        SolidOp::Cut(d, back, thin, thin_side) => FeatureKind::Cut { sketch, regions, plane, distance: d, back, thin, thin_side },
        SolidOp::Revolve(angle) => {
            let (axis_pt, axis_dir) = axis.unwrap();
            FeatureKind::Revolve { sketch, regions, plane, axis_pt, axis_dir, angle, cut: false }
        }
        SolidOp::RevolveCut(angle) => {
            let (axis_pt, axis_dir) = axis.unwrap();
            FeatureKind::Revolve { sketch, regions, plane, axis_pt, axis_dir, angle, cut: true }
        }
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
    session.revolve_axis = None;
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
    part: Res<Part>,
    mut cam_q: Query<(&mut Transform, &mut OrbitCamera)>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let win_h = windows.single().map(|w| w.height()).unwrap_or(0.0);
    // Start a fresh sketch on an arbitrary stored plane (a reference image's plane) so you can
    // trace over the picture. Same setup as the datum-plane path, but from a recorded PlaneRef.
    if let Some(plane) = ui_state.sketch_on_ref.take() {
        let ap = active_plane_from_ref(&plane, "Picture");
        if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
            let (center, radius) = fit_view(&part);
            orbit.radius = (radius * 2.1).max(6.0);
            look_along(&mut orbit, center, ap.n);
            recenter_for_panel(&mut orbit, win_h, ui_state.view_center_offset);
            *tf = camera_transform(&orbit);
        }
        session.sketch.clear();
        session.editing = None;
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
        info!("Sketching on a reference-image plane.");
        session.plane = Some(ap);
        ui_state.selected_plane = None;
        ui_state.image_edit = None;
        ui_state.image_calib = None;
        return;
    }

    // Start a fresh sketch on a datum plane picked from the tree (works with a body present).
    if let Some(order) = ui_state.sketch_plane_request.take() {
        if let Some((_, p)) = doc.0.planes().nth(order) {
            let ap = ActivePlane::from_doc(p);
            if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
                // Frame the body (centre + fit radius) so it sits centred in the viewport — aiming
                // at the plane origin alone would put the part's base at centre, not the part. The
                // extra 1.4× zooms out a little so the whole part fits with padding (room to draw
                // a profile beside it).
                let (center, radius) = fit_view(&part);
                orbit.radius = (radius * 2.1).max(6.0);
                look_along(&mut orbit, center, ap.n);
                recenter_for_panel(&mut orbit, win_h, ui_state.view_center_offset);
                *tf = camera_transform(&orbit);
            }
            session.sketch.clear();
            session.editing = None;
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
            ui_state.selected_plane = None; // it's now the sketch plane, not a standalone selection
        }
        return;
    }

    let Some(i) = ui_state.edit_sketch_request.take() else { return };
    let Some(f) = doc.0.features.get(i) else { return };
    let (sketch, plane, contours) = match &f.kind {
        FeatureKind::Sketch { sketch, plane } => (sketch.clone(), plane.clone(), Vec::new()),
        FeatureKind::Extrude { sketch, plane, regions, .. }
        | FeatureKind::Cut { sketch, plane, regions, .. }
        | FeatureKind::Revolve { sketch, plane, regions, .. } => {
            (sketch.clone(), plane.clone(), regions.clone())
        }
        FeatureKind::Plane(_) | FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } | FeatureKind::Thread { .. } | FeatureKind::Loft { .. } | FeatureKind::RefImage { .. } => return,
    };
    // A revolve also remembers its axis (point + direction) — re-select the matching line so the
    // PropertyManager's Axis box is filled and the preview shows when reopening it.
    let axis = match &f.kind {
        FeatureKind::Revolve { axis_pt, axis_dir, .. } => Some((*axis_pt, *axis_dir)),
        _ => None,
    };
    let ap = active_plane_from_ref(&plane, "Face");
    if let Ok((mut tf, mut orbit)) = cam_q.single_mut() {
        // Centre the body (not the plane origin) in the viewport, looking down the sketch normal,
        // with a little padding (1.4×) so the whole part fits with room to draw.
        let (center, radius) = fit_view(&part);
        orbit.radius = (radius * 2.1).max(6.0);
        look_along(&mut orbit, center, ap.n);
        recenter_for_panel(&mut orbit, win_h, ui_state.view_center_offset);
        *tf = camera_transform(&orbit);
    }
    session.sketch = sketch;
    session.plane = Some(ap);
    session.editing = Some(i);
    session.selected_contours = contours;
    session.revolve_axis = axis.and_then(|(pt, dir)| find_axis_line(&session.sketch, pt, dir));
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
    // Reopened as a Text feature: jump straight into the Text tool with the text selected and its
    // parameters loaded into the PM, so editing feels like editing a dedicated feature.
    if std::mem::take(&mut ui_state.edit_as_text) {
        let found = session.sketch.entities.iter().enumerate().find_map(|(ei, e)| match e {
            SketchEntity::Text { text, font, height, arc, mirror, bold, italic, spacing, .. } => Some((
                ei,
                text.clone(),
                font.clone(),
                *height as f32,
                *arc,
                *mirror,
                *bold,
                *italic,
                *spacing,
            )),
            _ => None,
        });
        if let Some((ei, text, font, height, arc, mirror, bold, italic, spacing)) = found {
            session.text_string = text;
            session.text_font = font;
            session.text_height = height.max(0.05);
            session.text_arc = arc;
            session.text_mirror = mirror;
            session.text_bold = bold;
            session.text_italic = italic;
            session.text_spacing = spacing;
            session.text_font_init = true; // values come from the entity — don't reset to defaults
            session.tool = Tool::Text;
            session.selected_entities = vec![ei];
        }
    }
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
                | FeatureKind::Cut { sketch, regions: r, .. }
                | FeatureKind::Revolve { sketch, regions: r, .. } => {
                    *sketch = new_sketch;
                    *r = contours;
                    ui_state.regen = true;
                }
                FeatureKind::Plane(_) | FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } | FeatureKind::Thread { .. } | FeatureKind::Loft { .. } | FeatureKind::RefImage { .. } => {}
            }
            ui_state.selected = Some(i);
        }
        _ => {
            // A brand-new sketch with geometry becomes a standalone Sketch feature
            // (the ever-present origin anchor alone doesn't count as geometry).
            if session.sketch.has_geometry() {
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
        .any(|f| matches!(f.kind, FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } | FeatureKind::Thread { .. }))
}

/// True if any extrude/cut is a **thin feature** (wall thickness > 0). The exact B-rep path
/// ignores `thin`, so a doc with one must regenerate through the mesh kernel.
fn doc_has_thin(doc: &Document) -> bool {
    doc.features.iter().any(|f| matches!(&f.kind,
        FeatureKind::Extrude { thin, .. } | FeatureKind::Cut { thin, .. } if *thin > 0.0))
}

/// failed to build, so the UI can tell the user which operation didn't apply.
fn regenerate_reported(doc: &Document) -> (Option<KSolid>, Vec<String>) {
    let mut failures: Vec<String> = Vec::new();
    let mut body: Option<KSolid> = None;
    let end = doc.rollback.min(doc.features.len());
    // Exact-arc (NURBS) profiles are only safe for the LAST solid feature: truck's
    // booleans panic against a NURBS-faced *base*, so any body that will receive
    // further features must stay faceted. A NURBS *tool* against a faceted base is
    // the proven-good direction, and the strategy ladders still guard the attempt.
    let is_solid_feature = |k: &FeatureKind| {
        matches!(k, FeatureKind::Extrude { .. } | FeatureKind::Cut { .. } | FeatureKind::Revolve { .. })
    };
    let last_solid = doc.features[..end].iter().rposition(|f| is_solid_feature(&f.kind));
    // Strip the exact-arc annotations unless this is the last solid feature with a
    // single profile (multiple profiles boolean against each other in sequence, so
    // all but the final result would become a NURBS base).
    let gate_arcs = |merged: &mut Vec<hworks_sketch::Region>, fi: usize| {
        if Some(fi) != last_solid || merged.len() != 1 {
            for r in merged.iter_mut() {
                r.outer_arcs.clear();
                r.hole_arcs.clear();
            }
        }
    };
    for (fi, feature) in doc.features[..end].iter().enumerate() {
        match &feature.kind {
            FeatureKind::Plane(_) => {}
            FeatureKind::Sketch { .. } => {} // 2D only — no solid contribution
            FeatureKind::RefImage { .. } => {} // a visual underlay — no solid contribution
            FeatureKind::Extrude { sketch, regions, plane, distance, back, .. } => {
                let all = sketch.regions();
                // A feature built on a face rides on that face: re-resolve its plane
                // to the current body so stacked features build on each other and
                // shift when an upstream feature is edited.
                let resolved = match &body { Some(b) => reproject_plane(plane, b), None => plane.clone() };
                let basis = basis_from_ref(&resolved);
                // Merge adjacent contours into single profiles first (a dumbbell of
                // two circles + connecting band becomes one outline), so each piece
                // extrudes as one solid without a coincident-face boolean.
                let mut merged = merge_regions(&chosen_regions(&all, regions));
                gate_arcs(&mut merged, fi);
                for r in &merged {
                    let next = match &body {
                        Some(b) => boss_union(b, r, &basis, *distance, *back),
                        // First feature: Direction 2 extends the prism the other way.
                        None if *back > 0.0 => extrude_solid_with_overlap_arcs(
                            &r.outer, &r.holes, &kernel_spans(&r.outer_arcs), &kernel_hole_spans(r), &basis, *distance, *back,
                        ),
                        None => extrude_solid_arcs(
                            &r.outer, &r.holes, &kernel_spans(&r.outer_arcs), &kernel_hole_spans(r), &basis, *distance,
                        ),
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
            FeatureKind::Cut { sketch, regions, plane, distance, back, .. } => {
                let Some(b0) = &body else { continue };
                let all = sketch.regions();
                let resolved = reproject_plane(plane, b0);
                let basis = basis_from_ref(&resolved);
                let origin = Vec3::new(resolved.origin[0] as f32, resolved.origin[1] as f32, resolved.origin[2] as f32);
                let n = Vec3::new(resolved.normal[0] as f32, resolved.normal[1] as f32, resolved.normal[2] as f32);
                // Merge adjacent contours into single profiles first (same reason as
                // the boss), then cut each from the current body.
                let mut merged = merge_regions(&chosen_regions(&all, regions));
                gate_arcs(&mut merged, fi);
                for r in &merged {
                    let Some(b) = &body else { break };
                    // Pick the cut direction from the *current* body, so it stays
                    // correct even after upstream edits move things around.
                    let centroid = mesh_centroid(&tessellate(b, 0.06).mesh);
                    let signed = if (centroid - origin).dot(n) < 0.0 { -*distance } else { *distance };
                    if let Some(s) = cut_op(b, r, &basis, signed, *back) {
                        body = Some(s);
                    } else {
                        warn!("Regen: a cut contour could not be built.");
                        failures.push("Cut failed — the kernel rejected this cut (often a self-touching profile or a coincident wall; try nudging the sketch).".into());
                    }
                }
            }
            FeatureKind::Revolve { sketch, regions, plane, axis_pt, axis_dir, angle, cut } => {
                let all = sketch.regions();
                let resolved = match &body { Some(b) => reproject_plane(plane, b), None => plane.clone() };
                let basis = basis_from_ref(&resolved);
                let mut merged = merge_regions(&chosen_regions(&all, regions));
                gate_arcs(&mut merged, fi);
                for r in &merged {
                    let solid = revolve_solid_arcs(
                        &r.outer, &r.holes, &kernel_spans(&r.outer_arcs), &kernel_hole_spans(r),
                        &basis, *axis_pt, *axis_dir, *angle,
                    );
                    let next = match (&body, solid) {
                        // Boss adds the swept solid; cut subtracts it from the body.
                        // An untessellatable NURBS boolean result counts as failure
                        // (the mesh path then retries this feature).
                        (Some(b), Some(s)) => {
                            (if *cut { difference(b, &s) } else { union(b, &s) }).filter(solid_renderable)
                        }
                        (None, Some(s)) => (!*cut).then_some(s), // first feature: a boss *is* the body
                        (_, None) => None,
                    };
                    if let Some(s) = next {
                        body = Some(s);
                    } else if *cut {
                        failures.push("Revolve cut failed — the kernel rejected the subtraction (the mesh path will retry).".into());
                    } else {
                        failures.push("Revolve failed — the profile may straddle the axis, or the union was rejected (the mesh path will retry).".into());
                    }
                }
            }
            // These reshape the mesh, so a model with one always builds via the mesh path —
            // this exact-kernel path never runs with one present.
            FeatureKind::Fillet { .. } | FeatureKind::Chamfer { .. } | FeatureKind::Mirror { .. } | FeatureKind::Thread { .. } | FeatureKind::Loft { .. } => {
                failures.push("Fillet/chamfer/mirror/thread/loft needs the mesh kernel.".into());
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
/// Closest distance from point `p` to triangle `abc` (Ericson, *Real-Time Collision Detection* §5.1.5).
fn point_tri_dist(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let sub = |u: [f32; 3], v: [f32; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f32; 3], v: [f32; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let dist = |u: [f32; 3], v: [f32; 3]| dot(sub(u, v), sub(u, v)).sqrt();
    let add = |u: [f32; 3], v: [f32; 3], s: f32| [u[0] + v[0] * s, u[1] + v[1] * s, u[2] + v[2] * s];
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return dist(p, a);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return dist(p, b);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return dist(p, add(a, ab, d1 / (d1 - d3)));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return dist(p, c);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return dist(p, add(a, ac, d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return dist(p, add(b, sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let proj = add(add(a, ab, vb * denom), ac, vc * denom);
    dist(p, proj)
}

/// Keep only the (bevel) edges that still lie on the final surface. A fillet/chamfer emits its
/// tangent/hard edges for selection, but a *later* cut can remove the material those edges sat on,
/// leaving them floating in the void (the "lines sticking out" artifact, plus breaks where a chord
/// straddles the boundary). The robust test is the edge **midpoint's distance to the nearest mesh
/// triangle**: a normal fillet chord's midpoint sits a hair off the curved surface, while a void
/// chord's midpoint is millimetres into open space. Triangles are spatial-hashed by bounding box.
fn clip_edges_to_mesh(edges: &[([[f32; 3]; 2], [f32; 3])], mesh: &TriMesh, rel: f32) -> Vec<[[f32; 3]; 2]> {
    use std::collections::HashMap;
    if mesh.positions.is_empty() || edges.is_empty() || mesh.indices.len() < 3 {
        return Vec::new();
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &mesh.positions {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
    let tol = (diag * rel).max(1.0e-4);
    // Hash grid coarse enough that big flat triangles don't explode into millions of cells.
    let cell = (diag * 0.03).max(tol);
    let gq = |p: [f32; 3]| ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64, (p[2] / cell).floor() as i64);
    let pos = |i: u32| mesh.positions[i as usize];
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (ti, t) in mesh.indices.chunks_exact(3).enumerate() {
        let (a, b, c) = (pos(t[0]), pos(t[1]), pos(t[2]));
        let (mut tl, mut th) = (a, a);
        for v in [b, c] {
            for k in 0..3 {
                tl[k] = tl[k].min(v[k]);
                th[k] = th[k].max(v[k]);
            }
        }
        let (gl, gh) = (gq(tl), gq(th));
        for gx in gl.0..=gh.0 {
            for gy in gl.1..=gh.1 {
                for gz in gl.2..=gh.2 {
                    grid.entry((gx, gy, gz)).or_default().push(ti);
                }
            }
        }
    }
    // A point is SUPPORTED when a nearby triangle both touches it (within tol) AND faces the way
    // the seam's own flat face faced at emission. Distance alone is not enough: a later cut whose
    // wall/floor happens to graze the old seam line (a slot bottom coincident with a fillet-base
    // ring, a bore wall under a crossing chord) keeps stale seams alive — but those surfaces face
    // a completely different way, so the normal test kills them.
    let supported = |p: [f32; 3], n_seam: [f32; 3]| -> bool {
        let (gx, gy, gz) = gq(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(ts) = grid.get(&(gx + dx, gy + dy, gz + dz)) {
                        for &ti in ts {
                            let t = &mesh.indices[ti * 3..ti * 3 + 3];
                            if point_tri_dist(p, pos(t[0]), pos(t[1]), pos(t[2])) <= tol {
                                let tn = mesh.normals[t[0] as usize]; // flat normal, same on all 3
                                let dot = tn[0] * n_seam[0] + tn[1] * n_seam[1] + tn[2] * n_seam[2];
                                if dot > 0.7 {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    };
    // Classify each bevel chord by which of {endpoint A, endpoint B, midpoint} are supported:
    //   • legit chord:      midpoint supported                       → keep
    //   • straddling (break): exactly ONE end supported              → keep (reaches the cut edge)
    //   • void/grazing span: nothing (or only a foreign face) nearby → drop
    edges
        .iter()
        .filter(|(e, n)| {
            let mid = [(e[0][0] + e[1][0]) * 0.5, (e[0][1] + e[1][1]) * 0.5, (e[0][2] + e[1][2]) * 0.5];
            let (a_on, b_on) = (supported(e[0], *n), supported(e[1], *n));
            supported(mid, *n) || (a_on != b_on)
        })
        .map(|(e, _)| *e)
        .collect()
}

/// Rebuild the mesh-kernel body, plus the selectable feature edges emitted by the *last*
/// bevel (fillet/chamfer) if it's still the final body operation — those tangent/hard edges
/// are otherwise invisible to angle-based edge extraction, so we plumb them out explicitly.
fn regenerate_mesh(doc: &Document) -> Option<(TriMesh, Vec<([[f32; 3]; 2], [f32; 3])>)> {
    let mut body: Option<TriMesh> = None;
    let mut bevel_edges: Vec<([[f32; 3]; 2], [f32; 3])> = Vec::new();
    let end = doc.rollback.min(doc.features.len());
    for feature in &doc.features[..end] {
        match &feature.kind {
            FeatureKind::Plane(_) | FeatureKind::Sketch { .. } | FeatureKind::RefImage { .. } => {}
            FeatureKind::Extrude { sketch, regions, plane, distance, back, thin, thin_side } => {
                // A boss adds material elsewhere; it doesn't invalidate edges from an earlier
                // fillet/chamfer, so keep them (don't clear here).
                let all = sketch.regions();
                let regs = merge_regions(&chosen_regions(&all, regions));
                // Reproject onto the live body's matching face (like the exact path), so a boss
                // stacks on the *current* top rather than the stale stored plane — testing under
                // the footprint centroid so it can't snap onto an unrelated feature's top.
                let samples = sketch_footprint_samples(plane, &regs);
                let plane = match &body {
                    Some(b) => reproject_plane_on_mesh(plane, b, &samples),
                    None => plane.clone(),
                };
                let basis = basis_from_ref(&plane);
                // A thin feature sweeps a wall of thickness `thin` instead of the filled region.
                let make_prism = |outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>], start: f64, length: f64| {
                    if *thin > 0.0 {
                        thin_wall_mesh(outer, holes, &basis, start, length, *thin, *thin_side)
                    } else {
                        extrude_tool_mesh(outer, holes, &basis, start, length)
                    }
                };
                for r in &regs {
                    body = match body.take() {
                        // Boss: dip the prism *substantially* into the body so the join ring
                        // is buried in continuous material — the surface wall then runs
                        // smoothly through it (no seam). A tiny dip leaves sliver triangles
                        // at the join; an exactly-flush union only touches and leaves a full
                        // ring of edges. The dip is bounded by how deep the body sits below
                        // the boss's base plane, so it can't poke out the far side.
                        Some(b) => {
                            let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                            // Bury the JOIN — the end of the prism that meets existing material —
                            // so the union has no coplanar seam. WHICH side of the base plane has
                            // material varies (behind for a boss stacked on a top face; past the
                            // plane for one on a thin feature's wall; NEITHER for a boss bridging
                            // a slot), and a global bbox reads it wrong on thin bodies — the old
                            // bbox estimate grew this boss by a full 1.0 out its exposed tip. So
                            // PROBE locally at the footprint centroid, just off the plane on the
                            // join side, and dip only as deep as there is actual material; flush
                            // (no dip) when the join side is open air — the mesh kernel unions
                            // touching solids fine (that's the Seamless premise).
                            let cen = sketch_footprint_world(&plane, r.outer.iter());
                            let dip_dir = if *distance >= 0.0 { -n } else { n };
                            let mut overlap = 0.0_f64;
                            for d in [1.0_f32, 0.5, 0.25, 0.1] {
                                if point_inside_mesh(&b, cen + dip_dir * d) {
                                    overlap = d as f64 * 0.9; // stay inside what we probed
                                    break;
                                }
                            }
                            // Direction 2 (`back`) extends the prism the opposite way — the same
                            // side as the body dip — so burying the join by at least `back` gives
                            // the both-directions extrude. Unlike the auto-dip it's NOT clamped to
                            // the body depth: the user asked for that length, even through the back.
                            let dip = overlap.max(*back);
                            // Bury the join-side end; the exposed tip stays EXACTLY at the asked
                            // depth: a normal (+) boss dips below the plane; a reversed (−) boss
                            // keeps its tip at `distance` and dips past the plane into the body.
                            let (start, length) = if *distance >= 0.0 {
                                (-dip, distance + dip)
                            } else {
                                (*distance, -distance + dip)
                            };
                            Some(
                                make_prism(&r.outer, &r.holes, start, length)
                                    .map(|tool| mesh_union(&b, &tool))
                                    .unwrap_or(b),
                            )
                        }
                        // First feature: the prism itself is the body. Direction 2 extends it
                        // the opposite way from the base plane.
                        None => {
                            let (start, length) = if *distance >= 0.0 {
                                (-*back, distance + back)
                            } else {
                                (distance - back, -distance + back)
                            };
                            make_prism(&r.outer, &r.holes, start, length)
                        }
                    };
                }
            }
            FeatureKind::Cut { sketch, regions, plane, distance, back, thin, thin_side } => {
                // A cut elsewhere doesn't invalidate an earlier fillet/chamfer's edges, so keep
                // them (don't clear) — they accumulate through the timeline like the boss path.
                let Some(cur0) = &body else { continue };
                let all = sketch.regions();
                // Reproject onto the live body's top face (like the exact path) so the cut
                // starts at the real surface — otherwise a chamfered body (forced onto this
                // mesh path) cuts from the stale stored plane and comes out shallow. Test under
                // the footprint centroid so it lands on the face the sketch actually sits on.
                let cut_regs = merge_regions(&chosen_regions(&all, regions));
                let ref_world = sketch_footprint_world(plane, cut_regs.iter().flat_map(|r| r.outer.iter()));
                let samples = sketch_footprint_samples(plane, &cut_regs);
                let plane = reproject_plane_on_mesh(plane, cur0, &samples);
                let basis = basis_from_ref(&plane);
                let origin = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
                let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
                // Cut INTO the material. Decide the direction *locally* — probe just under the cut
                // point on each side and see which is solid. The old global-centroid test failed
                // when a tall feature elsewhere (a boss) pulled the centroid past the cut plane,
                // flipping the cut to face open air (a shallow nick).
                let surf = ref_world - n * (ref_world - origin).dot(n); // footprint point on the plane
                let eps = 0.02_f32.max(*distance as f32 * 0.01);
                let neg_in = point_inside_mesh(cur0, surf - n * eps);
                let pos_in = point_inside_mesh(cur0, surf + n * eps);
                let into = if neg_in && !pos_in {
                    -1.0
                } else if pos_in && !neg_in {
                    1.0
                } else if (mesh_centroid(cur0) - origin).dot(n) < 0.0 {
                    -1.0 // ambiguous → fall back to the centroid heuristic
                } else {
                    1.0
                };
                for r in &cut_regs {
                    let Some(cur) = body.take() else { break };
                    let signed = into * *distance;
                    // Thin cut: subtract a wall of thickness `thin` (over the same span cut_tool_mesh
                    // would sweep, including its through-cut overshoot) instead of the filled region.
                    let tool = if *thin > 0.0 {
                        let depth = signed.abs();
                        let e = 0.05 + depth * 0.02;
                        let bk = back.max(0.0);
                        let (start, length) = if signed >= 0.0 { (-(e + bk), depth + 2.0 * e + bk) } else { (-(depth + e), depth + 2.0 * e + bk) };
                        thin_wall_mesh(&r.outer, &r.holes, &basis, start, length, *thin, *thin_side)
                    } else {
                        cut_tool_mesh(&r.outer, &r.holes, &basis, signed, *back)
                    };
                    body = Some(match tool {
                        Some(tool) => mesh_difference(&cur, &tool),
                        None => cur,
                    });
                }
            }
            FeatureKind::Revolve { sketch, regions, plane, axis_pt, axis_dir, angle, cut } => {
                bevel_edges.clear();
                let all = sketch.regions();
                let regs = merge_regions(&chosen_regions(&all, regions));
                let samples = sketch_footprint_samples(plane, &regs);
                let resolved = match &body {
                    Some(b) => reproject_plane_on_mesh(plane, b, &samples),
                    None => plane.clone(),
                };
                let basis = basis_from_ref(&resolved);
                for r in &regs {
                    if let Some(tool) = revolve_tool_mesh(&r.outer, &r.holes, &basis, *axis_pt, *axis_dir, *angle) {
                        body = Some(match body.take() {
                            // Boss unions the swept solid; cut subtracts it (no-op if no body yet).
                            Some(b) => if *cut { mesh_difference(&b, &tool) } else { mesh_union(&b, &tool) },
                            None => if *cut { continue } else { tool },
                        });
                    }
                }
            }
            FeatureKind::Loft { profiles, cut } => {
                bevel_edges.clear();
                let mut secs: Vec<(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)> = profiles.iter().filter_map(loft_profile_loops).collect();
                // A loft CUT whose end profile sits on a body face leaves the cap coincident with
                // that face (a zero-thickness skin survives the subtraction). Extend the lofted
                // solid a little past each end along the loft direction so the caps break cleanly
                // through the surface — exactly like an extrude cut overshoots.
                if *cut {
                    extend_loft_caps(&mut secs);
                }
                if let Some(m) = loft_mesh(&secs) {
                    body = match body.take() {
                        // Cut subtracts the lofted solid (a tapered pocket); boss unions it.
                        Some(b) => Some(if *cut { mesh_difference(&b, &m) } else { mesh_union(&b, &m) }),
                        // First feature: a boss *is* the body; a cut with nothing to cut is a no-op.
                        None => (!*cut).then_some(m),
                    };
                }
            }
            // Round the body's (picked, or all) edges by the fillet radius. We try the
            // mesh-surgery bevel first (no CSG → clean corners); it self-checks watertightness
            // and returns None on cases it can't resolve, so we fall back to the CSG round.
            FeatureKind::Fillet { radius, edges } => {
                if let Some(b) = body.take() {
                    let seg = ((*radius * 6.0).round() as usize).clamp(3, 12);
                    // One topology pass gives both the surgery mesh and the tangent edges. The
                    // edges are emitted whether the surgery succeeded or fell back to CSG (they
                    // sit at the same contact lines), so the rounded edges stay selectable. Stacked
                    // fillets accumulate (a cylinder's top + bottom).
                    let (beveled, fe) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bevel_mesh_and_edges(&b, *radius, seg, edges)
                    }))
                    .unwrap_or((None, Vec::new()));
                    bevel_edges.extend(fe);
                    body = Some(beveled.unwrap_or_else(|| round_mesh(&b, *radius, edges).unwrap_or(b)));
                }
            }
            // Flat-bevel the picked (or all) edges by the chamfer distance — same engine with a
            // single (flat) profile segment; CSG chamfer fallback.
            FeatureKind::Chamfer { distance, edges } => {
                if let Some(b) = body.take() {
                    let (beveled, fe) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        bevel_mesh_and_edges(&b, *distance, 1, edges)
                    }))
                    .unwrap_or((None, Vec::new()));
                    bevel_edges.extend(fe);
                    body = Some(beveled.unwrap_or_else(|| chamfer_mesh(&b, *distance, edges).unwrap_or(b)));
                }
            }
            // Reflect the body across the plane and union it with the original.
            FeatureKind::Mirror { plane } => {
                bevel_edges.clear(); // body changes → any prior bevel edges are stale
                if let Some(b) = body.take() {
                    let refl = mirror_mesh(&b, plane.origin, plane.normal);
                    body = Some(mesh_union(&b, &refl));
                }
            }
            // Tap a threaded hole / thread a boss.
            FeatureKind::Thread { origin, axis, major_d, pitch, depth, internal, rh } => {
                bevel_edges.clear();
                if let Some(b) = body.take() {
                    body = Some(threaded_hole(&b, *origin, *axis, *major_d, *pitch, *depth, *internal, *rh).unwrap_or(b));
                }
            }
        }
    }
    body.map(|m| (m, bevel_edges))
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
    // The rebuild replaces the displayed body with the FULL model — an active section view
    // re-applies afterwards (apply_section keys off this reset).
    ui_state.section_shown = None;
    // Vertices move when the model rebuilds, so any edge selection is stale.
    edge_sel.clear();

    // Text produces hundreds of tiny glyph faces; truck's recursive B-rep booleans can
    // *stack-overflow* on that (a hard abort `catch_unwind` can't trap), so any model
    // containing text is built with the robust mesh kernel from the start.
    let has_loft = doc.0.features.iter().any(|f| matches!(f.kind, FeatureKind::Loft { .. }));
    let force_mesh = ui_state.seamless || doc_has_text(&doc.0) || doc_has_fillet(&doc.0) || has_loft || doc_has_thin(&doc.0);

    // Seamless mode: build the whole model with the mesh kernel (Manifold), which fuses
    // coincident/coplanar faces so adjacent features merge without a seam. The exact
    // path's seams come from truck not merging shared faces; mesh has no such limit.
    if force_mesh {
        let mesh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc.0)))
            .unwrap_or(None);
        let fallbacks = take_fallback_count(); // booleans Manifold couldn't do (→ lossy BSP)
        for e in &existing {
            commands.entity(e).despawn();
        }
        match mesh {
            Some((m, bevel_edges)) if !m.positions.is_empty() => {
                let tess = mesh_tessellation(m);
                // The face-boundary detector already reports every real (sharp) edge, including a
                // chamfer's flat-face boundaries. The bevel's own edges cover the *tangent* boundary
                // of a fillet (a rounded edge, which the face detector merges away). They go in the
                // dedicated SEAM set: drawn like real edges (a filleted rim keeps its boundary ring)
                // and always selectable, without dragging in the exact path's facet lines. Clip to
                // the final surface first so a later cut doesn't leave them floating in the void.
                part.seam_edges = clip_edges_to_mesh(&bevel_edges, &tess.mesh, 0.01);
                part.mesh = Some(tess.mesh.clone());
                part.edges = tess.edges.clone();
                part.tangent_edges = tess.tangent_edges.clone();
                spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                part.solid = None; // mesh body has no B-rep handle
                ui_state.last_error = (fallbacks > 0).then(|| {
                    warn!("{fallbacks} boolean(s) fell back to the lossy BSP CSG — surface may be torn.");
                    format!("⚠ {fallbacks} boolean(s) used the lossy fallback (Manifold rejected them) — the surface may be torn. Likely an exact tangent/coincident face: check the revolve axis is centred and the profile isn't flush with the boss wall.")
                });
            }
            _ => {
                part.solid = None;
                part.mesh = None;
                part.edges.clear();
                part.tangent_edges.clear();
                part.seam_edges.clear();
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
                part.seam_edges.clear(); // exact path has no bevels (fillet forces the mesh path)
                spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                part.solid = Some(solid);
            }
            None => {
                part.solid = None;
                part.mesh = None;
                part.edges.clear();
                part.tangent_edges.clear();
                part.seam_edges.clear();
            }
        }
        return;
    }

    // The exact kernel stumbled on a boolean — rebuild the whole model with the robust
    // mesh kernel so the operation still applies (triangulated faces for the result).
    let mesh_body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc.0)))
        .unwrap_or(None);
    match mesh_body {
        Some((mesh, bevel_edges)) if !mesh.positions.is_empty() => {
            let tess = mesh_tessellation(mesh);
            // Fillet boundary seams → the dedicated seam set (drawn like real edges, selectable).
            part.seam_edges = clip_edges_to_mesh(&bevel_edges, &tess.mesh, 0.01);
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
                    part.seam_edges.clear();
                    spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
                    part.solid = Some(solid);
                }
                None => {
                    part.solid = None;
                    part.mesh = None;
                    part.edges.clear();
                    part.tangent_edges.clear();
                    part.seam_edges.clear();
                }
            }
        }
    }
}

/// Apply the display-only SECTION VIEW: cut the (full) tessellated body with a half-space box on
/// the discarded side of the chosen plane and show the capped result — a true cross-section, not
/// a hollow shader clip. Re-runs when the parameters change or a regen refreshed the body;
/// switching it off restores the full display via a normal regen. `part.mesh` is never touched,
/// so regeneration, face reprojection, and the document stay on the full body.
fn apply_section(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut ui_state: ResMut<UiState>,
    mut part: ResMut<Part>,
    existing: Query<Entity, With<SolidPart>>,
) {
    if ui_state.regen {
        return; // let the pending rebuild land first; it resets `section_shown`
    }
    if ui_state.section == ui_state.section_shown {
        return;
    }
    match ui_state.section {
        None => {
            // Section switched off → a normal regen restores the full display and edge pools.
            ui_state.section_shown = None;
            ui_state.regen = true;
        }
        Some(spec) => {
            let Some(full) = part.mesh.clone() else {
                ui_state.section_shown = ui_state.section;
                return;
            };
            if full.positions.is_empty() {
                ui_state.section_shown = ui_state.section;
                return;
            }
            // First application since the last regen: stash the full seam set for re-filtering.
            if ui_state.section_shown.is_none() {
                part.seam_backup = part.seam_edges.clone();
            }
            let (lo, hi) = mesh_bbox(&full);
            let l = (((hi - lo).length()) as f64 * 1.5).max(10.0);
            let (u, v, n) = section_axes(&spec);
            let d3 = |w: Vec3| [w.x as f64, w.y as f64, w.z as f64];
            let pr = PlaneRef { origin: [0.0; 3], u: d3(u), v: d3(v), normal: d3(n), datum: true };
            let basis = basis_from_ref(&pr);
            let sq = [[-l, -l], [l, -l], [l, l], [-l, l]];
            // The half-space box covers the DISCARDED side: past the offset along +normal, or
            // before it when flipped.
            let (start, len) = if spec.flip { (spec.offset as f64 - 2.0 * l, 2.0 * l) } else { (spec.offset as f64, 2.0 * l) };
            let Some(tool) = extrude_tool_mesh(&sq, &[], &basis, start, len) else { return };
            let cut = mesh_difference(&full, &tool);
            let _ = take_fallback_count(); // a display-only cut must never raise the torn-surface banner
            let tess = mesh_tessellation(cut);
            for e in &existing {
                commands.entity(e).despawn();
            }
            // Displayed edges follow the sectioned view (including the fresh section outline);
            // seams re-filter from the stashed full set so sliding the plane back restores them.
            part.edges = tess.edges.clone();
            part.tangent_edges = tess.tangent_edges.clone();
            let side = if spec.flip { -1.0 } else { 1.0 };
            let backup = part.seam_backup.clone();
            part.seam_edges = backup
                .into_iter()
                .filter(|e| {
                    let mid = Vec3::new((e[0][0] + e[1][0]) * 0.5, (e[0][1] + e[1][1]) * 0.5, (e[0][2] + e[1][2]) * 0.5);
                    (mid.dot(n) - spec.offset) * side <= 0.0
                })
                .collect();
            spawn_solid(&mut commands, &mut meshes, &mut materials, tess);
            ui_state.section_shown = ui_state.section;
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
    // Guard the round-everything footgun: an empty selection (e.g. cleared by a sketch edit)
    // used to silently round every edge. Require at least one picked edge.
    if edges.is_empty() {
        ui_state.last_error = Some("Select one or more edges to fillet (click them on the body).".into());
        return;
    }
    history.snapshot(&doc.0);
    // Editing an existing fillet (tree → Edit): update it in place instead of appending.
    if let Some(i) = ui_state.editing_feature.take() {
        if let Some(f) = doc.0.features.get_mut(i) {
            f.kind = FeatureKind::Fillet { radius, edges };
            doc.0.rollback = doc.0.features.len();
            ui_state.selected = Some(i);
            ui_state.regen = true;
            return;
        }
    }
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
    if edges.is_empty() {
        // Nothing picked yet — don't preview (empty = "all edges" would round the whole body).
        // Restore the real body and wait for the user to click edges.
        ui_state.regen = true;
        ui_state.fillet_shown = Some(r);
        return;
    }
    let seg = ((r * 6.0).round() as usize).clamp(3, 12);
    let rounded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Mesh bevel first (clean corners); CSG round on anything it can't resolve.
        bevel_mesh_selected(&base, r as f64, seg, &edges).or_else(|| round_mesh(&base, r as f64, &edges))
    }))
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
    if edges.is_empty() {
        ui_state.last_error = Some("Select one or more edges to chamfer (click them on the body).".into());
        return;
    }
    history.snapshot(&doc.0);
    // Editing an existing chamfer (tree → Edit): update it in place instead of appending.
    if let Some(i) = ui_state.editing_feature.take() {
        if let Some(f) = doc.0.features.get_mut(i) {
            f.kind = FeatureKind::Chamfer { distance, edges };
            doc.0.rollback = doc.0.features.len();
            ui_state.selected = Some(i);
            ui_state.regen = true;
            return;
        }
    }
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
    if edges.is_empty() {
        // Nothing picked yet — don't preview (empty = "all edges" would bevel the whole body).
        ui_state.regen = true;
        ui_state.chamfer_shown = Some(d);
        return;
    }
    let beveled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Mesh bevel with a flat (1-segment) profile first; CSG chamfer fallback.
        bevel_mesh_selected(&base, d as f64, 1, &edges).or_else(|| chamfer_mesh(&base, d as f64, &edges))
    }))
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
    // Editing an existing mirror (tree → Edit): update it in place instead of appending.
    if let Some(i) = ui_state.editing_feature.take() {
        if let Some(f) = doc.0.features.get_mut(i) {
            f.kind = FeatureKind::Mirror { plane: standard_plane_ref(which) };
            doc.0.rollback = doc.0.features.len();
            ui_state.selected = Some(i);
            ui_state.regen = true;
            return;
        }
    }
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

/// Build the `FeatureKind::Thread` from a placed spec.
fn thread_feature(spec: &ThreadSpec) -> FeatureKind {
    FeatureKind::Thread {
        origin: [spec.origin.x as f64, spec.origin.y as f64, spec.origin.z as f64],
        axis: [spec.axis.x as f64, spec.axis.y as f64, spec.axis.z as f64],
        major_d: spec.major_d() as f64,
        pitch: spec.pitch as f64,
        depth: spec.depth as f64,
        internal: spec.internal,
        rh: spec.rh,
    }
}

/// Append a confirmed thread to the timeline and trigger a rebuild.
fn apply_thread(mut ui_state: ResMut<UiState>, mut doc: ResMut<DocRes>, mut history: ResMut<History>) {
    let Some(spec) = ui_state.thread_request.take() else { return };
    history.snapshot(&doc.0);
    // Editing an existing thread (tree → Edit): update it in place instead of appending.
    if let Some(i) = ui_state.editing_feature.take() {
        if let Some(f) = doc.0.features.get_mut(i) {
            f.kind = thread_feature(&spec);
            doc.0.rollback = doc.0.features.len();
            ui_state.selected = Some(i);
            ui_state.regen = true;
            return;
        }
    }
    doc.0.add_feature(thread_feature(&spec));
    doc.0.rollback = doc.0.features.len();
    ui_state.selected = Some(doc.0.features.len() - 1);
    ui_state.regen = true;
}

/// While the Hole Genie PM is open, draw a see-through ghost of the hole/thread on the
/// overlay group (so it shows *through* the model, like the cut-extrude preview): a hover
/// snap marker before it's placed, then the bore rings + helix going into the body.
fn thread_ghost(mut overlay: Gizmos<OverlayGizmos>, ui_state: Res<UiState>) {
    let Some(spec) = &ui_state.pending_thread else { return };
    // Hover marker at the snapped placement point (before it's anchored).
    if !spec.placed {
        if let Some((o, a)) = ui_state.thread_hover {
            let r = (spec.major_d() * 0.5).max(0.3);
            ring(&mut overlay, o, a.normalize_or_zero(), r, Color::srgb(0.2, 1.0, 0.45));
            let u = a.normalize_or_zero().any_orthonormal_vector() * (r * 0.4);
            overlay.line(o - u, o + u, Color::srgb(0.2, 1.0, 0.45));
            let w = a.normalize_or_zero().cross(u.normalize_or_zero()) * (r * 0.4);
            overlay.line(o - w, o + w, Color::srgb(0.2, 1.0, 0.45));
        }
        return;
    }
    let a = spec.axis.normalize_or_zero();
    let r = (spec.major_d() * 0.5).max(0.1);
    let depth = spec.depth.max(0.1);
    let o = spec.origin;
    let bore = Color::srgba(1.0, 0.4, 0.35, 0.9); // reddish, like a cut
    let depth_col = Color::srgb(1.0, 0.75, 0.2); // bright depth ring
    // Top and bottom (depth) rings.
    ring(&mut overlay, o, a, r, bore);
    ring(&mut overlay, o - a * depth, a, r, depth_col);
    // Four risers down the bore.
    let u = a.any_orthonormal_vector();
    let v = a.cross(u);
    for k in 0..4 {
        let ang = std::f32::consts::TAU * k as f32 / 4.0;
        let p = o + (u * ang.cos() + v * ang.sin()) * r;
        overlay.line(p, p - a * depth, bore);
    }
    // Helical thread line, suggesting the pitch.
    let pitch = spec.pitch.max(0.1);
    let turns = (depth / pitch).clamp(0.5, 200.0);
    let n = (turns * 24.0).ceil() as usize;
    let sign = if spec.rh { 1.0 } else { -1.0 };
    let mut prev = o + u * r;
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let ang = sign * std::f32::consts::TAU * turns * t;
        let p = o - a * (depth * t) + (u * ang.cos() + v * ang.sin()) * r;
        overlay.line(prev, p, Color::srgba(1.0, 0.6, 0.2, 0.8));
        prev = p;
    }
    overlay.line(o, o - a * depth, depth_col); // axis
}

/// Draw a ring (circle) of `radius` centred at `c` in the plane with normal `n`.
fn ring(overlay: &mut Gizmos<OverlayGizmos>, c: Vec3, n: Vec3, radius: f32, color: Color) {
    let n = n.normalize_or_zero();
    let u = n.any_orthonormal_vector();
    let v = n.cross(u);
    const SEG: usize = 48;
    let mut prev = c + u * radius;
    for k in 1..=SEG {
        let a = std::f32::consts::TAU * k as f32 / SEG as f32;
        let p = c + (u * a.cos() + v * a.sin()) * radius;
        overlay.line(prev, p, color);
        prev = p;
    }
}

/// `PlaneBasis` (kernel-side) from a stored `PlaneRef`.
fn basis_from_ref(p: &PlaneRef) -> PlaneBasis {
    PlaneBasis { origin: p.origin, u: p.u, v: p.v, normal: p.normal }
}

/// A loft profile's outer boundary + hole loops lifted into 3D world space (its sketch region
/// through the profile's plane). `None` if the sketch has no usable closed region.
fn loft_profile_loops(p: &LoftProfile) -> Option<(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)> {
    let regions = p.sketch.regions();
    let r = regions.get(p.region).or_else(|| regions.first())?;
    let (o, u, v) = (p.plane.origin, p.plane.u, p.plane.v);
    let to3 = |uv: &[f64; 2]| [o[0] + u[0] * uv[0] + v[0] * uv[1], o[1] + u[1] * uv[0] + v[1] * uv[1], o[2] + u[2] * uv[0] + v[2] * uv[1]];
    let outer = r.outer.iter().map(to3).collect();
    let holes = r.holes.iter().map(|h| h.iter().map(to3).collect()).collect();
    Some((outer, holes))
}

/// Extend a loft's end caps outward along the loft direction by a small overshoot, so a loft *cut*
/// whose end profile lies on a body face breaks cleanly through it instead of leaving a coincident
/// zero-thickness skin. Prepends a copy of the first section shifted off the start, and appends a
/// copy of the last section shifted off the end. No-op for fewer than two sections.
fn extend_loft_caps(secs: &mut Vec<(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)>) {
    if secs.len() < 2 {
        return;
    }
    let centroid = |s: &(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>)| {
        let o = &s.0;
        let n = o.len().max(1) as f64;
        let mut c = [0.0; 3];
        for p in o {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        [c[0] / n, c[1] / n, c[2] / n]
    };
    let shift = |s: &(Vec<[f64; 3]>, Vec<Vec<[f64; 3]>>), d: [f64; 3]| {
        let t = |loop_: &[[f64; 3]]| loop_.iter().map(|p| [p[0] + d[0], p[1] + d[1], p[2] + d[2]]).collect::<Vec<_>>();
        (t(&s.0), s.1.iter().map(|h| t(h)).collect::<Vec<_>>())
    };
    let unit = |a: [f64; 3], b: [f64; 3]| {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if l < 1e-9 { [0.0, 0.0, 1.0] } else { [d[0] / l, d[1] / l, d[2] / l] }
    };
    // Overshoot scaled to the loft's own span (so it clears a surface without distorting the shape).
    let n = secs.len();
    let span: f64 = (0..n - 1)
        .map(|i| {
            let (a, b) = (centroid(&secs[i]), centroid(&secs[i + 1]));
            unit_len(a, b)
        })
        .sum();
    let over = (span * 0.05).clamp(0.5, 3.0);
    // Start: push the first section away from the second (out of the body at an open end).
    let d0 = unit(centroid(&secs[1]), centroid(&secs[0]));
    let start = shift(&secs[0], [d0[0] * over, d0[1] * over, d0[2] * over]);
    // End: push the last section away from the second-to-last.
    let dn = unit(centroid(&secs[n - 2]), centroid(&secs[n - 1]));
    let end = shift(&secs[n - 1], [dn[0] * over, dn[1] * over, dn[2] * over]);
    secs.insert(0, start);
    secs.push(end);
}

/// Euclidean distance between two 3D points.
fn unit_len(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
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
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
    reproject_plane_on_mesh(plane, &tessellate(body, 0.2).mesh, &[o])
}

/// Möller–Trumbore ray/triangle; positive hit distance along `dir`, else `None`.
fn ray_tri_hit(o: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let (e1, e2) = (b - a, c - a);
    let pv = dir.cross(e2);
    let det = e1.dot(pv);
    if det.abs() < 1e-9 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = o - a;
    let u = tv.dot(pv) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qv = tv.cross(e1);
    let v = dir.dot(qv) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(qv) * inv;
    (t > 1e-5).then_some(t)
}

/// True if `p` is inside the closed mesh — odd number of ray crossings along an arbitrary ray.
fn point_inside_mesh(mesh: &TriMesh, p: Vec3) -> bool {
    let dir = Vec3::new(0.573_257, 0.577_350, 0.581_443).normalize();
    let mut crossings = 0u32;
    for t in mesh.indices.chunks(3) {
        let a = Vec3::from_array(mesh.positions[t[0] as usize]);
        let b = Vec3::from_array(mesh.positions[t[1] as usize]);
        let c = Vec3::from_array(mesh.positions[t[2] as usize]);
        if ray_tri_hit(p, dir, a, b, c).is_some() {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

/// World-space centroid of a sketch footprint (its 2D outer points lifted through the plane).
/// Falls back to the plane origin when there are no points. Used to reproject under the face the
/// sketch actually sits on, not under the arbitrary plane origin.
fn sketch_footprint_world<'a>(plane: &PlaneRef, pts: impl Iterator<Item = &'a [f64; 2]>) -> Vec3 {
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
    let u = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
    let v = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
    let (mut cx, mut cy, mut n) = (0.0f64, 0.0f64, 0u32);
    for p in pts {
        cx += p[0];
        cy += p[1];
        n += 1;
    }
    if n == 0 {
        return o;
    }
    o + u * (cx / n as f64) as f32 + v * (cy / n as f64) as f32
}

/// Footprint sample points in world space for face re-projection: each region's boundary points PLUS
/// an interior grid (points inside the region, outside its holes). The interior grid is essential for
/// a big overhanging profile — e.g. a large boss circle whose *perimeter* hangs over empty space and
/// whose *centre* sits over a hole: the real face lies in the annulus between, which only interior
/// samples reach. Falls back to the plane origin if empty.
fn sketch_footprint_samples(plane: &PlaneRef, regs: &[hworks_sketch::Region]) -> Vec<Vec3> {
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
    let u = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
    let v = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
    let w = |p: [f64; 2]| o + u * p[0] as f32 + v * p[1] as f32;
    let mut out = Vec::new();
    for r in regs {
        if r.outer.is_empty() {
            continue;
        }
        for p in &r.outer {
            out.push(w(*p));
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &r.outer {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        const N: usize = 10;
        for i in 0..N {
            for j in 0..N {
                let x = lo[0] + (hi[0] - lo[0]) * (i as f64 + 0.5) / N as f64;
                let y = lo[1] + (hi[1] - lo[1]) * (j as f64 + 0.5) / N as f64;
                if point_in_poly([x, y], &r.outer) && !r.holes.iter().any(|h| point_in_poly([x, y], h)) {
                    out.push(w([x, y]));
                }
            }
        }
    }
    if out.is_empty() {
        out.push(o);
    }
    out
}

/// Replace a tessellation-fitted circle `(c, r)` with the exact source circle it matches,
/// if any. A match must agree in **centre AND radius** — concentric features (a boss with
/// a bore) produce several exact circles at the same centre, so matching by centre alone
/// could hand a hole reference the boss's radius. Ties break to the closest combined fit.
fn refine_circle(exact: &[(Vec2, f32)], c: Vec2, r: f32) -> (Vec2, f32) {
    let tol = (r * 0.05).max(0.1);
    exact
        .iter()
        .copied()
        .filter(|(ec, er)| ec.distance(c) <= tol && (er - r).abs() <= tol)
        .min_by(|a, b| {
            (a.0.distance(c) + (a.1 - r).abs()).total_cmp(&(b.0.distance(c) + (b.1 - r).abs()))
        })
        .unwrap_or((c, r))
}

/// Exact `(centre_uv, radius)` of every body circular edge lying in the sketch plane `ap`,
/// read from the *source* sketch entities in the timeline (circles, arcs, and slot end
/// caps). The B-rep tessellates circles to polygons, so a tessellation fit only
/// approximates the radius (~sagitta); this recovers the true value so a concentric boss
/// snaps exactly and joins without a micro-step.
fn exact_plane_circles(doc: &Document, ap: &ActivePlane) -> Vec<(Vec2, f32)> {
    let mut out = Vec::new();
    let n_unit = ap.n.normalize_or_zero();
    let end = doc.rollback.min(doc.features.len());
    for f in &doc.features[..end] {
        let (sketch, plane, dist, back) = match &f.kind {
            FeatureKind::Extrude { sketch, plane, distance, back, .. } => (sketch, plane, *distance as f32, *back as f32),
            FeatureKind::Cut { sketch, plane, distance, back, .. } => (sketch, plane, *distance as f32, *back as f32),
            _ => continue,
        };
        let fo = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
        let fu = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
        let fv = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
        let fnormal = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32).normalize_or_zero();
        // The circle's plane must be parallel to the sketch plane to project to a circle.
        if fnormal.dot(n_unit).abs() < 0.999 {
            continue;
        }
        // Prism cap planes: both sweep ends, and the Direction-2 (`back`) ends when set.
        let offs = [0.0_f32, dist, -dist, -back, dist + back, -dist - back];
        // Exact circle sources in the sketch: circle entities, arc entities (radius
        // |centre→a|), and a slot's two semicircular end caps.
        let mut push = |center_uv: Vec2, radius: f32| {
            if radius <= 1e-4 {
                return;
            }
            let cbase = fo + fu * center_uv.x + fv * center_uv.y;
            for off in offs {
                let c3 = cbase + fnormal * off;
                if (c3 - ap.origin).dot(ap.n).abs() < 0.02 {
                    let d = c3 - ap.origin;
                    let uv = Vec2::new(d.dot(ap.u), d.dot(ap.v));
                    if !out.iter().any(|(c, r): &(Vec2, f32)| c.distance(uv) < 1e-4 && (r - radius).abs() < 1e-4) {
                        out.push((uv, radius));
                    }
                }
            }
        };
        let pt = |i: usize| sketch.points.get(i).map(|p| Vec2::new(p.x as f32, p.y as f32));
        for ent in &sketch.entities {
            match ent {
                SketchEntity::Circle { center, radius, construction: false } => {
                    if let Some(c) = pt(*center) {
                        push(c, *radius as f32);
                    }
                }
                SketchEntity::Arc { center, a, construction: false, .. } => {
                    if let (Some(c), Some(pa)) = (pt(*center), pt(*a)) {
                        push(c, c.distance(pa));
                    }
                }
                SketchEntity::Slot { a, b, radius, construction: false, .. } => {
                    for &i in [a, b] {
                        if let Some(c) = pt(i) {
                            push(c, *radius as f32);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Same as [`reproject_plane`] but driven by a triangle mesh directly — so the mesh-kernel
/// path (forced by a fillet/chamfer/mirror) can reproject a feature's plane onto the live
/// body too, instead of using the stale stored plane (which made cuts land shallow).
fn reproject_plane_on_mesh(plane: &PlaneRef, mesh: &TriMesh, samples: &[Vec3]) -> PlaneRef {
    // A datum plane (Front/Top/Right) is fixed in space — never snap it onto a body face, or a
    // centre-plane sketch (e.g. a revolve-cut profile through the middle) gets shoved onto a cap.
    if plane.datum {
        return plane.clone();
    }
    let n = Vec3::new(plane.normal[0] as f32, plane.normal[1] as f32, plane.normal[2] as f32);
    let u = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
    let v = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
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
    // Snap the plane onto the body face it sits on. Consider only faces:
    //  • **same-facing** (`tn·n > 0.9`, not `abs`) — a sketch rides a face whose outward normal points
    //    the sketch's way; the back face would let a stacked boss drop onto the base's *bottom*.
    //  • **under the footprint** — any of the sampled profile points projects onto the face. Sampling
    //    the whole footprint (not just its centroid) is what fixes a big boss whose *centre* sits over
    //    a hole: the real face is only under the perimeter, so a centroid-only test teleported it onto
    //    a far floor at the bottom of the hole.
    // Among those, take the one whose offset is nearest the stored plane (a no-op on a faithful replay,
    // and it tracks the face if an upstream edit slid it along the normal).
    let samples2d: Vec<Vec2> = samples.iter().map(|s| to2d(*s)).collect();
    let mut best: Option<f32> = None;
    for t in mesh.indices.chunks(3) {
        let p = |i: u32| Vec3::from_array(mesh.positions[i as usize]);
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let tn = (b - a).cross(c - a).normalize_or_zero();
        if tn.dot(n) < 0.9 {
            continue;
        }
        let (ta, tb, tc) = (to2d(a), to2d(b), to2d(c));
        if !samples2d.iter().any(|&s| in_tri(s, ta, tb, tc)) {
            continue; // no footprint sample is over this face
        }
        let off = a.dot(n);
        if best.map_or(true, |bo| (off - o_n).abs() < (bo - o_n).abs()) {
            best = Some(off);
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

/// Convert a region's sketch-layer exact-arc annotations into the kernel's type
/// (the two crates don't depend on each other, so the span structs are twins).
fn kernel_spans(spans: &[hworks_sketch::ArcSpan]) -> Vec<hworks_geometry::ArcSpan> {
    spans
        .iter()
        .map(|s| hworks_geometry::ArcSpan {
            first_edge: s.first_edge,
            count: s.count,
            center: s.center,
            radius: s.radius,
        })
        .collect()
}

/// The per-hole arc annotations of a region, in the kernel's type.
fn kernel_hole_spans(r: &hworks_sketch::Region) -> Vec<Vec<hworks_geometry::ArcSpan>> {
    r.hole_arcs.iter().map(|h| kernel_spans(h)).collect()
}

/// Resolve the selected-contour indices against a sketch's regions. An empty
/// selection means "all regions" — but excludes `nested` regions (holes re-exposed
/// as selectable disks), which would double-cover the region that owns them as a
/// hole. Explicit selections are honoured verbatim (you *can* pick a nested disk).
/// Out-of-range indices are skipped (the sketch may have changed since creation).
fn chosen_regions<'a>(all: &'a [hworks_sketch::Region], selected: &[usize]) -> Vec<&'a hworks_sketch::Region> {
    if selected.is_empty() {
        all.iter().filter(|r| !r.nested).collect()
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
    // Nothing cancels ⇒ no two regions are adjacent ⇒ nothing to merge. Return
    // the inputs as-is (keeping their exact-arc annotations, which re-tracing
    // the loops below would discard).
    if !edges.iter().any(|&(a, b)| present.contains(&(b, a))) {
        return regions.iter().map(|r| (*r).clone()).collect();
    }
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
        // Merged outlines re-trace the loops, so per-edge arc annotations no
        // longer apply — leave them empty (the kernel then uses line edges).
        out.push(hworks_sketch::Region { outer: loops[i].clone(), holes, ..Default::default() });
    }
    out
}

/// Add a boss (region `r`) to an existing body, trying progressively more robust
/// strategies so a boolean never simply fails: flush+exact, flush+nudge (coincident
/// faces), then the robust overlap/tolerance with and without the nudge. The first
/// (cleanest) one that works wins.
fn boss_union(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64, back: f64) -> Option<KSolid> {
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
    let (outer_arcs, hole_arcs) = (kernel_spans(&r.outer_arcs), kernel_hole_spans(r));
    let mut extruded_ok = false;
    for (k, &(nudge, overlap, tol)) in strategies.iter().enumerate() {
        // Direction 2 (`back`) extends the prism the opposite way; it shares the same side
        // as the union overlap, so burying by at least `back` gives the both-directions boss.
        // A nudged (inflated) profile no longer lies on its source circles, so only the
        // un-nudged strategies use the exact-arc annotations.
        let boss = if nudge > 0.0 {
            extrude_solid_with_overlap(&inflate_loop(&r.outer, nudge), &r.holes, basis, distance, overlap.max(back))
        } else {
            extrude_solid_with_overlap_arcs(&r.outer, &r.holes, &outer_arcs, &hole_arcs, basis, distance, overlap.max(back))
        };
        let Some(boss) = boss else {
            continue;
        };
        extruded_ok = true;
        // Try both operand orders — truck's `or` is order-sensitive on awkward faces.
        if let Some(s) = union_tol(body, &boss, tol).or_else(|| union_tol(&boss, body, tol)) {
            // A NURBS (exact-arc) boolean can "succeed" but be untessellatable —
            // treat that as a failure so the next strategy (faceted) gets a shot.
            if !solid_renderable(&s) {
                info!("Boss union: strategy {k} result won't tessellate — trying the next.");
                continue;
            }
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
fn cut_op(body: &KSolid, r: &hworks_sketch::Region, basis: &PlaneBasis, distance: f64, back: f64) -> Option<KSolid> {
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
    let (outer_arcs, hole_arcs) = (kernel_spans(&r.outer_arcs), kernel_hole_spans(r));
    for (k, &(nudge, tol)) in strategies.iter().enumerate() {
        // As in `boss_union`: a nudged profile is off its source circles, so the
        // exact-arc annotations only apply to the un-nudged strategies.
        let s = if nudge > 0.0 {
            cut_tol(body, &inflate_loop(&r.outer, nudge), &r.holes, basis, distance, back, tol)
        } else {
            cut_tol_arcs(body, &r.outer, &r.holes, &outer_arcs, &hole_arcs, basis, distance, back, tol)
        };
        if let Some(s) = s {
            // The body may carry NURBS (exact-arc) faces, and a boolean against
            // them can "succeed" untessellatable — count that as a miss.
            if !solid_renderable(&s) {
                info!("Cut: strategy {k} result won't tessellate — trying the next.");
                continue;
            }
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

/// Miter-offset a closed 2D loop by `d`: shift each edge perpendicular by `d`, join at the
/// intersections of consecutive offset edges. `d > 0` grows the loop's enclosed area (consistent for
/// CW and CCW winding via the signed area); `d < 0` shrinks it. Powers thin-feature wall loops.
fn offset_loop(pts: &[[f64; 2]], d: f64) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 3 || d == 0.0 {
        return pts.to_vec();
    }
    let mut area = 0.0;
    for i in 0..n {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        area += a[0] * b[1] - b[0] * a[1];
    }
    let s = if area >= 0.0 { 1.0 } else { -1.0 };
    let normal = |i: usize| -> [f64; 2] {
        let (a, b) = (pts[i], pts[(i + 1) % n]);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 { [0.0, 0.0] } else { [s * dy / len, -s * dx / len] }
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ep = (i + n - 1) % n;
        let (np, nc) = (normal(ep), normal(i));
        let p1 = [pts[ep][0] + d * np[0], pts[ep][1] + d * np[1]];
        let d1 = [pts[i][0] - pts[ep][0], pts[i][1] - pts[ep][1]];
        let p2 = [pts[i][0] + d * nc[0], pts[i][1] + d * nc[1]];
        let d2 = [pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1]];
        let denom = d1[0] * d2[1] - d1[1] * d2[0];
        if denom.abs() < 1e-12 {
            out.push([pts[i][0] + d * nc[0], pts[i][1] + d * nc[1]]); // collinear → plain edge shift
        } else {
            let t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom;
            out.push([p1[0] + d1[0] * t, p1[1] + d1[1] * t]);
        }
    }
    out
}

/// Grow (`+d`) or shrink (`-d`) a region-with-holes: the outer loop offsets by `d`; holes offset the
/// opposite way so the *solid* dilates/erodes uniformly (growing the region shrinks its holes).
fn offset_region(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>], d: f64) -> (Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>) {
    (offset_loop(outer, d), holes.iter().map(|h| offset_loop(h, -d)).collect())
}

/// A **thin-feature** wall prism for one region over the span `[start, start+length]`: a wall of
/// thickness `thin` following the profile instead of the filled region — a pipe/box shell. `side`:
/// 0 = outward, 1 = inward, 2 = mid-plane. Built as extrude(grown region) − extrude(shrunk region),
/// which handles convex, concave, and holed profiles uniformly. `None` if the wall is degenerate.
fn thin_wall_mesh(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>], basis: &PlaneBasis, start: f64, length: f64, thin: f64, side: u8) -> Option<TriMesh> {
    if thin <= 0.0 {
        return None;
    }
    let (grow, shrink) = match side {
        1 => (0.0, thin),             // inward: outer stays, inner wall shrinks in
        2 => (thin * 0.5, thin * 0.5), // mid-plane: split the thickness
        _ => (thin, 0.0),             // outward: outer wall grows out
    };
    let (bo, bh) = offset_region(outer, holes, grow);
    let (so, sh) = offset_region(outer, holes, -shrink);
    let big = extrude_tool_mesh(&bo, &bh, basis, start, length)?;
    let small = extrude_tool_mesh(&so, &sh, basis, start, length)?;
    Some(mesh_difference(&big, &small))
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
    // Reference-plane quads resync via `sync_ref_planes` once the doc resets to default planes.
    part.solid = None;
    part.mesh = None;
    // Clear EVERY edge pool — leaving tangent/seam edges behind ghosted the old part's
    // fillet rings over the fresh empty scene (they draw from Part, not the despawned mesh).
    part.edges.clear();
    part.tangent_edges.clear();
    part.seam_edges.clear();
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
    ui_state.current_file = None; // a fresh part isn't bound to the old file
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
    // Fillet/chamfer boundary seams draw like real edges — a filleted rim keeps its visible,
    // selectable boundary ring (on a round body they're often its ONLY edges).
    for e in &part.seam_edges {
        gizmos.line(nudge(Vec3::from_array(e[0])), nudge(Vec3::from_array(e[1])), col);
    }
    // Tangent/curvature edges only when the user asks (drawn lighter to read as soft) —
    // on the exact path this set is every facet line of a curved wall, so it stays opt-in.
    if ui_state.show_tangent_edges {
        let tcol = Color::srgb(0.45, 0.47, 0.52);
        for e in &part.tangent_edges {
            gizmos.line(nudge(Vec3::from_array(e[0])), nudge(Vec3::from_array(e[1])), tcol);
        }
    }
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    blocking: Res<UiBlocking>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    ui_state: Res<UiState>,
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

    // Which gesture is which button depends on the chosen control scheme.
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let middle = buttons.pressed(MouseButton::Middle);
    let (orbiting, panning) = match ui_state.mouse_scheme {
        // HCAD (default): right-drag orbits, middle-drag pans.
        MouseScheme::Hcad => (buttons.pressed(MouseButton::Right), middle),
        // Blender: middle-drag orbits, Shift+middle pans.
        MouseScheme::Blender => (middle && !shift, middle && shift),
        // SolidWorks: middle-drag orbits, Ctrl+middle pans.
        MouseScheme::SolidWorks => (middle && !ctrl, middle && ctrl),
    };

    if orbiting && motion.delta != Vec2::ZERO {
        cam.yaw -= motion.delta.x * ORBIT_SENS;
        // No pole stop: the camera rotation comes straight from Euler angles, so pitch can roll
        // continuously over the top (turntable-style). Wrap into ±π so it never grows unbounded.
        let tau = std::f32::consts::TAU;
        cam.pitch = (cam.pitch - motion.delta.y * ORBIT_SENS + std::f32::consts::PI).rem_euclid(tau) - std::f32::consts::PI;
        changed = true;
    }

    // Pan: move the focus in the camera's screen plane.
    if panning && motion.delta != Vec2::ZERO {
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
        cam.anim = None; // a hand movement cancels any in-flight view animation
        *transform = camera_transform(&cam);
    }
}

// ---------------------------------------------------------------------------
// Viewport gizmos
// ---------------------------------------------------------------------------

/// Reference-plane visibility, SolidWorks-style:
///   * a plane hidden via its tree eye toggle → always hidden (the toggle wins);
///   * while actively sketching → all hidden (the sketch grid stands in);
///   * a datum plane selected in the tree → show just that one (so you can pick it to sketch on,
///     even with a body present — needed for e.g. a revolve cut through the centre);
///   * no body yet and nothing selected → show all three (so the first sketch is easy to start);
///   * otherwise (a body exists, nothing selected) → all hidden, to avoid clutter.
fn update_plane_visibility(
    part: Res<Part>,
    session: Res<SketchSession>,
    ui_state: Res<UiState>,
    doc: Res<DocRes>,
    mut planes: Query<(&mut Visibility, &RefPlaneIdx)>,
) {
    let sketching = session.plane.is_some();
    let show_all = part.solid.is_none() && ui_state.selected_plane.is_none();
    let hidden_by_user: Vec<bool> = doc.0.planes_vis().map(|(_, _, h)| h).collect();
    for (mut vis, idx) in &mut planes {
        let user_hidden = hidden_by_user.get(idx.0).copied().unwrap_or(false);
        let want = if user_hidden || sketching {
            Visibility::Hidden
        } else if ui_state.selected_plane == Some(idx.0) || show_all {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != want {
            *vis = want;
        }
    }
}

/// Keep the reference-plane quads in sync with the document: the default Front/Top/Right plus any
/// the user creates (an offset construction plane). On a mismatch (a plane added, or New Part /
/// Open changing the set) it respawns the whole set, so user-created planes show in the viewport
/// and are pickable/sketchable exactly like the defaults.
fn sync_ref_planes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    doc: Res<DocRes>,
    existing: Query<(Entity, &RefPlaneIdx)>,
) {
    let have = existing.iter().count();
    let want = doc.0.planes().count();
    if have == want {
        return;
    }
    for (e, _) in &existing {
        commands.entity(e).despawn();
    }
    let plane_mesh = meshes.add(Rectangle::new(PLANE_SIZE, PLANE_SIZE));
    for (i, (_id, plane)) in doc.0.planes().enumerate() {
        let ap = ActivePlane::from_doc(plane);
        let rotation = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));
        // Front/Top/Right keep their red/green/blue tint; user-created planes are amber.
        let base_color = [
            Color::srgba(0.85, 0.25, 0.25, 0.16),
            Color::srgba(0.25, 0.75, 0.30, 0.16),
            Color::srgba(0.25, 0.45, 0.90, 0.16),
        ]
        .get(i)
        .copied()
        .unwrap_or(Color::srgba(0.85, 0.75, 0.30, 0.16));
        let material = materials.add(StandardMaterial {
            base_color,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            double_sided: true,
            depth_bias: i as f32,
            ..default()
        });
        commands.spawn((
            Mesh3d(plane_mesh.clone()),
            MeshMaterial3d(material),
            Transform { translation: ap.origin, rotation, ..default() },
            Name::new(plane.name.clone()),
            RefPlane,
            RefPlaneIdx(i),
        ));
    }
}

/// Keep the window title showing the bound file name (or "Untitled") — a quick at-a-glance of
/// what's open and whether it's been saved to a file yet.
fn update_window_title(ui_state: Res<UiState>, mut windows: Query<&mut Window>) {
    if let Ok(mut w) = windows.single_mut() {
        let name = ui_state.current_file.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("Untitled");
        let want = format!("HCAD — {name}");
        if w.title != want {
            w.title = want;
        }
    }
}

/// Scale the reference-plane quads to the user's chosen display size (they're spawned at the base
/// `PLANE_SIZE`), so planes can be enlarged to suit a big part without rebuilding the mesh.
fn scale_ref_planes(ui_state: Res<UiState>, mut q: Query<&mut Transform, With<RefPlane>>) {
    let s = (ui_state.plane_size.max(1.0) / PLANE_SIZE).max(0.01);
    for mut t in &mut q {
        if (t.scale.x - s).abs() > 1.0e-4 {
            t.scale = Vec3::splat(s);
        }
    }
}

/// Decode a base64 PNG/JPG into a Bevy RGBA8 texture. `None` if the data is unreadable.
fn decode_image_texture(b64: &str) -> Option<Image> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    let dynimg = image::load_from_memory(&bytes).ok()?;
    let rgba = dynimg.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some(Image::new(
        bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        bevy::render::render_resource::TextureDimension::D2,
        rgba.into_raw(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::RENDER_WORLD,
    ))
}

/// Keep the reference-image quads in sync with the document: one textured quad per `RefImage`
/// feature. On a mismatch (image inserted/deleted, New Part / Open) it respawns the whole set,
/// decoding each texture once. Per-frame transform/opacity tweaks are handled by `update_ref_images`
/// — this only fires when the *set of images* changes, so re-decoding is rare.
fn sync_ref_images(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    doc: Res<DocRes>,
    existing: Query<(Entity, &RefImageEnt)>,
) {
    use std::collections::HashSet;
    let want: HashSet<u64> = doc
        .0
        .features
        .iter()
        .filter(|f| matches!(f.kind, FeatureKind::RefImage { .. }))
        .map(|f| f.id.0)
        .collect();
    let have: HashSet<u64> = existing.iter().map(|(_, r)| r.id.0).collect();
    if want == have {
        return;
    }
    // Despawn quads whose feature is gone; remember which ids already have a quad.
    for (e, r) in &existing {
        if !want.contains(&r.id.0) {
            commands.entity(e).despawn();
        }
    }
    // A unit quad (1×1 in its local XY) — sized/oriented per-image via the Transform.
    let quad = meshes.add(Rectangle::new(1.0, 1.0));
    for f in &doc.0.features {
        let FeatureKind::RefImage { plane, data, opacity, .. } = &f.kind else { continue };
        if have.contains(&f.id.0) {
            continue; // already spawned
        }
        let Some(tex) = decode_image_texture(data) else {
            warn!("Reference image: could not decode the embedded picture data.");
            continue;
        };
        let handle = images.add(tex);
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 1.0, 1.0, *opacity),
            base_color_texture: Some(handle),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            cull_mode: None,
            double_sided: true,
            depth_bias: -1.0, // sit just behind sketch lines so you trace on top
            ..default()
        });
        let ap = ActivePlane::from_ref(plane);
        commands.spawn((
            Mesh3d(quad.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(ap.origin), // real transform set by update_ref_images
            Name::new("Reference Image"),
            RefImageEnt { id: f.id },
        ));
    }
}

/// Cheaply refresh each reference-image quad every frame from its feature: position (centre on the
/// plane), in-plane rotation, size (width × height via Transform scale), mirror (negative scale),
/// and opacity (material alpha). No texture re-decode — that only happens in `sync_ref_images`.
fn update_ref_images(
    doc: Res<DocRes>,
    session: Res<SketchSession>,
    ui_state: Res<UiState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(&RefImageEnt, &mut Transform, &MeshMaterial3d<StandardMaterial>, &mut Visibility)>,
) {
    // SolidWorks-style: a sketch picture is only visible while you're editing a sketch on its plane
    // (so you can trace it) — or while its own PropertyManager is open (to place/calibrate it).
    // Once you finish the sketch and move on, it hides; reopen that sketch and it returns.
    let editing_plane = session.plane.clone();
    for (r, mut tf, math, mut vis) in &mut q {
        let Some(f) = doc.0.features.iter().find(|f| f.id.0 == r.id.0) else { continue };
        let FeatureKind::RefImage { plane, center, rot, width, height, opacity, flip_h, flip_v, .. } = &f.kind else {
            continue;
        };
        let ap = ActivePlane::from_ref(plane);
        // Centre of the image on the plane.
        tf.translation = ap.origin + ap.u * center[0] as f32 + ap.v * center[1] as f32;
        // Orient to the plane (u→local X, v→local Y, n→local Z), then spin in-plane by `rot`.
        let basis = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));
        tf.rotation = basis * Quat::from_rotation_z(*rot as f32);
        // Size via scale; negative scale mirrors (double_sided keeps it lit either way).
        let sx = width.max(0.001) as f32 * if *flip_h { -1.0 } else { 1.0 };
        let sy = height.max(0.001) as f32 * if *flip_v { -1.0 } else { 1.0 };
        tf.scale = Vec3::new(sx, sy, 1.0);
        if let Some(m) = materials.get_mut(&math.0) {
            let a = opacity.clamp(0.0, 1.0);
            if m.base_color.alpha() != a {
                m.base_color.set_alpha(a);
            }
        }
        // Show only when relevant: sketching on this image's plane, or its PM is open —
        // and never when the user hid it via the tree's eye toggle (the toggle wins,
        // except while the PM is open to place/calibrate it).
        let pm_open = ui_state.image_edit.and_then(|i| doc.0.features.get(i)).map(|f| f.id.0) == Some(r.id.0);
        let sketching_here = editing_plane.as_ref().is_some_and(|sp| planes_coincident(sp, &ap));
        let want = if pm_open || (sketching_here && !f.hidden) { Visibility::Visible } else { Visibility::Hidden };
        if *vis != want {
            *vis = want;
        }
    }
}

/// True when two planes are the same infinite plane (parallel normals, ~coincident): used to decide
/// whether the sketch being edited lies on a reference image's plane.
fn planes_coincident(a: &ActivePlane, b: &ActivePlane) -> bool {
    let (na, nb) = (a.n.normalize_or_zero(), b.n.normalize_or_zero());
    if na.cross(nb).length() > 1.0e-3 {
        return false; // not parallel
    }
    (a.origin - b.origin).dot(na).abs() < 1.0e-3 // coincident along the normal
}

/// See-through mechanic: fade the body to translucent only when you actually need to see *into* it
/// — sketching on a datum/construction plane (Front/Top/Right/offset, which pass through the body)
/// or configuring a cut (extrude/revolve cut). Sketching on a face, or a plain boss, stays opaque
/// (the see-through is just clutter there). Restores to opaque otherwise.
fn update_body_transparency(
    session: Res<SketchSession>,
    ui_state: Res<UiState>,
    bodies: Query<&MeshMaterial3d<StandardMaterial>, With<SolidPart>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let on_datum = session.plane.as_ref().is_some_and(|p| p.datum);
    let cutting = matches!(ui_state.pending.as_ref().map(|o| o.kind), Some(OpKind::Cut | OpKind::RevolveCut));
    let see_through = on_datum || cutting;
    let want = if see_through { 0.28 } else { 1.0 };
    for mat in &bodies {
        if let Some(m) = materials.get_mut(&mat.0) {
            if (m.base_color.alpha() - want).abs() > 1e-3 {
                m.base_color = m.base_color.with_alpha(want);
                m.alpha_mode = if see_through { AlphaMode::Blend } else { AlphaMode::Opaque };
            }
        }
    }
}

/// Draw a bright bordered outline (and a diagonal) for the datum plane selected in the tree, so it
/// reads as a real plane to sketch on — not just the faint translucent quad. Skipped while
/// sketching (the grid stands in then).
fn draw_selected_plane(
    mut gizmos: Gizmos,
    ui_state: Res<UiState>,
    session: Res<SketchSession>,
    doc: Res<DocRes>,
) {
    if session.plane.is_some() {
        return;
    }
    let h = ui_state.plane_size.max(1.0) * 0.5;
    // Outline a plane (centre + in-plane axes u,v) as a rectangle with diagonals.
    let outline = |gizmos: &mut Gizmos, center: Vec3, u: Vec3, v: Vec3, col: Color, faint: Color| {
        let c = [center - u * h - v * h, center + u * h - v * h, center + u * h + v * h, center - u * h + v * h];
        for k in 0..4 {
            gizmos.line(c[k], c[(k + 1) % 4], col);
        }
        gizmos.line(c[0], c[2], faint);
        gizmos.line(c[1], c[3], faint);
    };

    // Construction-plane creation preview: the offset plane outline + a draggable normal arrow.
    if let Some(spec) = &ui_state.plane_spec {
        let ap = &spec.base;
        let n = ap.n.normalize_or_zero();
        let signed = if spec.flip { -spec.offset } else { spec.offset };
        let center = ap.origin + n * signed;
        outline(&mut gizmos, center, ap.u, ap.v, Color::srgba(1.0, 0.8, 0.3, 0.95), Color::srgba(1.0, 0.8, 0.3, 0.3));
        // Normal arrow from the base origin out to the new plane.
        let arrow = Color::srgb(1.0, 0.6, 0.2);
        gizmos.line(ap.origin, center, arrow);
        let dir = (center - ap.origin).normalize_or_zero();
        if dir != Vec3::ZERO {
            let head = (signed.abs() * 0.18).clamp(0.3, 2.0);
            for s in [ap.u, -ap.u, ap.v, -ap.v] {
                gizmos.line(center, center - dir * head + s.normalize_or_zero() * (head * 0.5), arrow);
            }
        }
    }

    let Some(order) = ui_state.selected_plane else { return };
    let Some((_, p)) = doc.0.planes().nth(order) else { return };
    let ap = ActivePlane::from_doc(p);
    outline(&mut gizmos, ap.origin, ap.u, ap.v, Color::srgba(0.45, 0.8, 1.0, 0.9), Color::srgba(0.45, 0.8, 1.0, 0.35));
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

/// Draw a stored sketch's profile geometry in world space — a read-only preview (a selected sketch
/// in the tree, or the profiles picked for a loft). `pick` highlights one region's outer loop and
/// faintly hatches it, so you can see which contour is chosen.
fn draw_stored_sketch(gizmos: &mut Gizmos, sketch: &Sketch, plane: &PlaneRef, color: Color, pick: Option<usize>) {
    let o = Vec3::new(plane.origin[0] as f32, plane.origin[1] as f32, plane.origin[2] as f32);
    let u = Vec3::new(plane.u[0] as f32, plane.u[1] as f32, plane.u[2] as f32);
    let v = Vec3::new(plane.v[0] as f32, plane.v[1] as f32, plane.v[2] as f32);
    let w = |x: f32, y: f32| o + u * x + v * y;
    let p2 = |i: usize| -> (f32, f32) {
        let p = &sketch.points[i];
        (p.x as f32, p.y as f32)
    };
    let poly_line = |gizmos: &mut Gizmos, poly: &[[f64; 2]], close: bool, col: Color| {
        let m = poly.len();
        let last = if close { m } else { m.saturating_sub(1) };
        for k in 0..last {
            let a = poly[k];
            let b = poly[(k + 1) % m];
            gizmos.line(w(a[0] as f32, a[1] as f32), w(b[0] as f32, b[1] as f32), col);
        }
    };
    for e in &sketch.entities {
        match e {
            SketchEntity::Line { a, b, construction: false, reference: false } => {
                let (ax, ay) = p2(*a);
                let (bx, by) = p2(*b);
                gizmos.line(w(ax, ay), w(bx, by), color);
            }
            SketchEntity::Circle { center, radius, construction: false } => {
                let (cx, cy) = p2(*center);
                let r = *radius as f32;
                let pts: Vec<[f64; 2]> = (0..48).map(|k| { let a = std::f32::consts::TAU * k as f32 / 48.0; [(cx + r * a.cos()) as f64, (cy + r * a.sin()) as f64] }).collect();
                poly_line(gizmos, &pts, true, color);
            }
            SketchEntity::Arc { center, a, b, ccw, construction: false } => {
                if let (Some(c), Some(pa), Some(pb)) = (sketch.points.get(*center), sketch.points.get(*a), sketch.points.get(*b)) {
                    poly_line(gizmos, &tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw), false, color);
                }
            }
            SketchEntity::Spline { points, closed, control, construction: false } => {
                let pts: Vec<[f64; 2]> = points.iter().filter_map(|&i| sketch.points.get(i)).map(|p| [p.x, p.y]).collect();
                if pts.len() >= 2 {
                    poly_line(gizmos, &tessellate_spline(&pts, *closed, *control), *closed, color);
                }
            }
            SketchEntity::Slot { a, b, radius, construction: false, mid } => {
                if let (Some(pa), Some(pb)) = (sketch.points.get(*a), sketch.points.get(*b)) {
                    let pm = mid.and_then(|m| sketch.points.get(m)).map(|p| [p.x, p.y]);
                    let poly = match pm {
                        Some(pm) => tessellate_arc_slot([pa.x, pa.y], pm, [pb.x, pb.y], *radius),
                        None => tessellate_slot([pa.x, pa.y], [pb.x, pb.y], *radius),
                    };
                    poly_line(gizmos, &poly, true, color);
                }
            }
            _ => {}
        }
    }
    if let Some(ri) = pick {
        if let Some(r) = sketch.regions().get(ri) {
            poly_line(gizmos, &r.outer, true, Color::srgb(0.3, 1.0, 0.5));
            for hole in &r.holes {
                poly_line(gizmos, hole, true, Color::srgb(0.3, 1.0, 0.5));
            }
        }
    }
}

/// Preview the geometry of the selected sketch (or, while lofting, every picked profile with its
/// chosen contour highlighted) so you can see what's on a sketch without opening it.
fn draw_feature_previews(
    mut gizmos: Gizmos,
    mut overlay: Gizmos<OverlayGizmos>,
    ui_state: Res<UiState>,
    doc: Res<DocRes>,
    session: Res<SketchSession>,
) {
    // Reference-image calibration markers: the picked points (crosses), the rubber-band to the cursor
    // (after the first pick), and the line between the two points. Drawn BEFORE the sketch-mode early
    // return so it shows even while a sketch is open on the picture's plane (you can calibrate then
    // trace). On the overlay group so the measuring line is clearly visible over the picture.
    if let (Some(cal), Some(idx)) = (&ui_state.image_calib, ui_state.image_edit) {
        if let Some(FeatureKind::RefImage { plane, .. }) = doc.0.features.get(idx).map(|f| &f.kind) {
            let ap = ActivePlane::from_ref(plane);
            let col = Color::srgb(1.0, 0.85, 0.2);
            let cross = |overlay: &mut Gizmos<OverlayGizmos>, p: Vec2| {
                let c = ap.to_world(p);
                let s = 2.0;
                overlay.line(c - ap.u * s, c + ap.u * s, col);
                overlay.line(c - ap.v * s, c + ap.v * s, col);
            };
            for p in &cal.pts {
                cross(&mut overlay, *p);
            }
            match (cal.pts.len(), cal.cursor) {
                // After the first pick, rubber-band from it to the cursor so you can see the span.
                (1, Some(cur)) => {
                    cross(&mut overlay, cur);
                    overlay.line(ap.to_world(cal.pts[0]), ap.to_world(cur), col);
                }
                (2, _) => overlay.line(ap.to_world(cal.pts[0]), ap.to_world(cal.pts[1]), col),
                _ => {}
            }
        }
    }
    if session.plane.is_some() {
        return; // the live sketch already draws itself
    }
    if let Some(profiles) = &ui_state.loft_spec {
        let palette = [Color::srgb(0.95, 0.85, 0.25), Color::srgb(0.25, 0.9, 0.95), Color::srgb(0.9, 0.5, 0.95), Color::srgb(0.4, 0.95, 0.5)];
        for (n, &(fi, region)) in profiles.iter().enumerate() {
            if let Some(FeatureKind::Sketch { sketch, plane }) = doc.0.features.get(fi).map(|f| &f.kind) {
                draw_stored_sketch(&mut gizmos, sketch, plane, palette[n % palette.len()], Some(region));
            }
        }
        return;
    }
    if let Some(i) = ui_state.selected {
        if let Some(f) = doc.0.features.get(i) {
            if let (FeatureKind::Sketch { sketch, plane }, false) = (&f.kind, f.hidden) {
                draw_stored_sketch(&mut gizmos, sketch, plane, Color::srgb(0.95, 0.9, 0.3), None);
            }
        }
    }
}

/// Outline the *selected* feature's sketch profile in the viewport (view mode), so picking a
/// feature in the tree gives geometric feedback — not just a highlighted row. Uses the shared
/// selection accent so tree, sketch, and this all read as "selected".
fn draw_selected_feature(
    mut gizmos: Gizmos,
    doc: Res<DocRes>,
    ui_state: Res<UiState>,
    session: Res<SketchSession>,
    cam_q: Query<&GlobalTransform, With<Camera3d>>,
) {
    if session.plane.is_some() {
        return; // only in view mode (while sketching, the live sketch already shows)
    }
    let Some(i) = ui_state.selected else { return };
    let Some(f) = doc.0.features.get(i) else { return };
    if f.hidden {
        return; // eye-toggled off — no profile outline either
    }
    let (sketch, plane) = match &f.kind {
        FeatureKind::Sketch { sketch, plane }
        | FeatureKind::Extrude { sketch, plane, .. }
        | FeatureKind::Cut { sketch, plane, .. }
        | FeatureKind::Revolve { sketch, plane, .. } => (sketch, plane),
        _ => return,
    };
    let ap = active_plane_from_ref(plane, "sel");
    let col = Color::srgb(1.0, 0.7, 0.1); // shared selection accent (matches sketch selection)
    let cam_pos = cam_q.single().map(|g| g.translation()).unwrap_or(ap.origin + ap.n * 10.0);
    let nudge = |p: Vec3| p + (cam_pos - p) * 0.004; // sit just in front, no z-fight
    let w = |p: &[f64; 2]| ap.to_world(Vec2::new(p[0] as f32, p[1] as f32));
    for reg in sketch.regions() {
        for loop_pts in std::iter::once(&reg.outer).chain(reg.holes.iter()) {
            for seg in loop_pts.windows(2) {
                gizmos.line(nudge(w(&seg[0])), nudge(w(&seg[1])), col);
            }
            if let (Some(first), Some(last)) = (loop_pts.first(), loop_pts.last()) {
                gizmos.line(nudge(w(last)), nudge(w(first)), col); // close the loop
            }
        }
    }
    // Text has no closed regions — outline its baked glyph contours directly.
    for e in &sketch.entities {
        if let SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. } = e {
            let o = sketch.points.get(*origin).map(|p| [p.x, p.y]).unwrap_or([0.0, 0.0]);
            for loop_ in text_contours(o, contours, *height, *rotation, *mirror, *arc) {
                let n = loop_.len();
                for k in 0..n {
                    gizmos.line(nudge(w(&loop_[k])), nudge(w(&loop_[(k + 1) % n])), col);
                }
            }
        }
    }
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

    // Lift the whole sketch a hair off its plane toward the camera ONLY when sketching on a body
    // face: there the geometry is coplanar with that face and z-fights with it (flicker/vanish).
    // A datum/construction plane has no coplanar face to fight, so it isn't lifted — the lift
    // scales with zoom and would otherwise show as a small gap between the sketch and an extrusion
    // drawn from the true plane. (Picking always uses the true plane.)
    let lifted;
    let ap = {
        let eye = cam_q.single().map(|c| camera_transform(c).translation).unwrap_or(ap.origin + ap.n);
        let sign = if (eye - ap.origin).dot(ap.n) >= 0.0 { 1.0 } else { -1.0 };
        let lift = if ap.datum { Vec3::ZERO } else { ap.n * (radius.max(1e-3) * 2.0e-3 * sign) };
        lifted = ActivePlane { name: ap.name.clone(), origin: ap.origin + lift, u: ap.u, v: ap.v, n: ap.n, datum: ap.datum };
        &lifted
    };

    // Adaptive grid: spacing snaps to a nice 1/2/5×10^k that's ~1/16 of the view,
    // with a bounded number of cells, so it stays usable from millimetres to metres. Each line is
    // drawn cell-by-cell and faded by radial distance from the origin, so the grid dissolves into a
    // soft disc instead of ending at a hard square edge.
    let base_a = 0.24_f32;
    let raw = (radius / 16.0).max(1e-4);
    let mag = 10f32.powf(raw.log10().floor());
    let norm = raw / mag;
    let spacing = mag * if norm < 1.5 { 1.0 } else if norm < 3.5 { 2.0 } else if norm < 7.5 { 5.0 } else { 10.0 };
    let cells = 24;
    let ext = spacing * cells as f32;
    let fade = |d: f32| base_a * (1.0 - d / ext).clamp(0.0, 1.0);
    for k in -cells..=cells {
        let f = k as f32 * spacing;
        for j in -cells..cells {
            let g0 = j as f32 * spacing;
            let g1 = g0 + spacing;
            let mid = g0 + spacing * 0.5;
            let d = (f * f + mid * mid).sqrt();
            let a = fade(d);
            if a < 0.012 {
                continue;
            }
            let col = Color::srgba(0.55, 0.55, 0.62, a);
            // vertical line x = f, and horizontal line y = f (same radial distance profile)
            gizmos.line(ap.to_world(Vec2::new(f, g0)), ap.to_world(Vec2::new(f, g1)), col);
            gizmos.line(ap.to_world(Vec2::new(g0, f)), ap.to_world(Vec2::new(g1, f)), col);
        }
    }

    let solid = Color::srgb(0.95, 0.95, 0.25);
    let construction = Color::srgb(0.9, 0.45, 0.95);
    let circle_col = Color::srgb(0.25, 0.9, 0.95);
    let point_col = Color::srgb(1.0, 0.55, 0.15);
    let preview_col = Color::srgb(1.0, 0.95, 0.45); // opaque so the rubber-band reads over a picture
    let plane_rot = Quat::from_mat3(&Mat3::from_cols(ap.u, ap.v, ap.n));
    // Marker/snap-glyph scale tied to the zoom so points stay a ~constant *screen* size. `snap_dist`
    // already tracks the zoom (∝ camera radius), so `ms` ∝ radius keeps markers screen-constant. The
    // clamp only guards the extremes — a floor of 0.5 kicked in at a moderate zoom-in and made the
    // markers balloon on screen (very visible when picking a circular-pattern centre), so keep it low.
    let ms = if session.snap_dist > 1e-6 { (session.snap_dist / SNAP).clamp(0.03, 400.0) } else { 1.0 };

    let uv_of = |i: usize| -> Vec2 {
        let p = &session.sketch.points[i];
        Vec2::new(p.x as f32, p.y as f32)
    };

    // Centre axes through the origin (0,0): full-length guide lines (X=0 vertical green, Y=0
    // horizontal red) so a centreline / symmetric profile is easy to line up to the part centre —
    // the cursor inference snaps onto these, and a bright crosshair marks the exact origin.
    {
        let vcol = Color::srgba(0.4, 1.0, 0.45, 0.55); // x = 0 (vertical centre line)
        let hcol = Color::srgba(1.0, 0.4, 0.4, 0.55); // y = 0 (horizontal centre line)
        gizmos.line(ap.to_world(Vec2::new(0.0, -ext)), ap.to_world(Vec2::new(0.0, ext)), vcol);
        gizmos.line(ap.to_world(Vec2::new(-ext, 0.0)), ap.to_world(Vec2::new(ext, 0.0)), hcol);
        let os = 0.3 * ms;
        gizmos.line(ap.to_world(Vec2::new(-os, 0.0)), ap.to_world(Vec2::new(os, 0.0)), Color::srgb(1.0, 0.4, 0.4));
        gizmos.line(ap.to_world(Vec2::new(0.0, -os)), ap.to_world(Vec2::new(0.0, os)), Color::srgb(0.4, 1.0, 0.45));
    }

    // The active sketch plane's outline — a faint bordered rectangle around the sketch, so when
    // you sketch on an offset/construction plane it reads as floating off the nearby geometry
    // (not lying on it). Sized to enclose the sketch content, with a margin.
    {
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in &session.sketch.points {
            let v = Vec2::new(p.x as f32, p.y as f32);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let (cen, half) = if lo.x.is_finite() {
            ((lo + hi) * 0.5, ((hi - lo).max_element() * 0.65).max(PLANE_SIZE * 0.5))
        } else {
            (Vec2::ZERO, PLANE_SIZE * 0.5)
        };
        let border = Color::srgba(0.5, 0.72, 1.0, 0.3);
        let bc = [
            cen + Vec2::new(-half, -half),
            cen + Vec2::new(half, -half),
            cen + Vec2::new(half, half),
            cen + Vec2::new(-half, half),
        ];
        for k in 0..4 {
            gizmos.line(ap.to_world(bc[k]), ap.to_world(bc[(k + 1) % 4]), border);
        }
    }

    // Inference guides: dotted hints showing why the cursor snapped (alignment / extension /
    // tangent). Faint amber dashes, drawn under the geometry.
    let infer_col = Color::srgba(1.0, 0.82, 0.3, 0.75);
    let dash = (radius * 0.012).clamp(0.02, 1.0);
    for (a, b) in &session.inference_guides {
        dashed_line(&mut gizmos, ap.to_world(*a), ap.to_world(*b), infer_col, dash, dash * 0.8);
    }

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
            SketchEntity::Arc { center, a, b, ccw, construction: is_con } => {
                if let (Some(c), Some(pa), Some(pb)) =
                    (session.sketch.points.get(*center), session.sketch.points.get(*a), session.sketch.points.get(*b))
                {
                    let poly = tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw);
                    for w in poly.windows(2) {
                        let (wa, wb) = (
                            ap.to_world(Vec2::new(w[0][0] as f32, w[0][1] as f32)),
                            ap.to_world(Vec2::new(w[1][0] as f32, w[1][1] as f32)),
                        );
                        if *is_con {
                            gizmos.line(wa, wb, construction);
                        } else {
                            profile.line(wa, wb, circle_col);
                        }
                    }
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

    // Constraint-status point coloring (SolidWorks-style): a point that can still
    // move draws blue; a fully defined one draws in the normal point colour.
    // Locked (body-projected) points stay amber. The cached report is used only
    // while it matches the current sketch (it can lag an edit by a frame).
    let free_of = session
        .dof_cache
        .as_ref()
        .filter(|(fp, rep)| {
            *fp == sketch_fingerprint(&session.sketch) && rep.free_points.len() == session.sketch.points.len()
        })
        .map(|(_, rep)| &rep.free_points);
    let under_col = Color::srgb(0.35, 0.65, 1.0);
    for (i, p) in session.sketch.points.iter().enumerate() {
        let col = if p.fixed {
            Color::srgb(1.0, 0.65, 0.1)
        } else if free_of.is_some_and(|f| f[i]) {
            under_col
        } else {
            point_col
        };
        draw_marker(&mut gizmos, ap, Vec2::new(p.x as f32, p.y as f32), col, ms);
    }

    // Highlight the Selected Contours — outer + holes. Explicitly-picked contours
    // are bright green; if none are picked, every region is shown dim (it's the
    // "all contours" default that an extrude/cut would use).
    let regions = session.cached_regions();
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
        // Every detected region is shown so you can see what's enclosed; explicitly picked
        // contours read in the shared SELECTION ACCENT (orange — same as selected edges and
        // features) with a strong fill, the rest dim green. (Picking some no longer hides the
        // others — that made an enclosed area look un-closed.)
        for (i, r) in regions.iter().enumerate() {
            let sel = picked.contains(&i);
            let (line_col, fill) = if sel {
                (Color::srgb(1.0, 0.7, 0.1), Color::srgba(1.0, 0.65, 0.15, 0.4))
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
    // A dimension that participates in an over-defined *conflict* draws red
    // (SolidWorks-style), so the user can see exactly which relations clash.
    let base_dim_col = Color::srgb(0.55, 0.85, 1.0);
    let conflict_col = Color::srgb(1.0, 0.3, 0.3);
    let conflicting: &[usize] = session
        .dof_cache
        .as_ref()
        .filter(|(fp, _)| *fp == sketch_fingerprint(&session.sketch))
        .map(|(_, rep)| rep.conflicting.as_slice())
        .unwrap_or(&[]);
    let pt = |i: usize| session.sketch.points.get(i).copied().map(|p| Vec2::new(p.x as f32, p.y as f32));
    for (ci, c) in session.sketch.constraints.iter().enumerate() {
        let dim_col = if conflicting.contains(&ci) { conflict_col } else { base_dim_col };
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
            // Slot width: a witness line straight across the slot (side to side).
            hworks_sketch::Constraint::SlotWidth { a, b, value, .. } => {
                if let (Some(a2), Some(b2)) = (pt(*a), pt(*b)) {
                    let (s1, s2, _) = slot_width_geometry(a2, b2, (*value * 0.5) as f32);
                    gizmos.line(ap.to_world(s1), ap.to_world(s2), dim_col);
                }
            }
            _ => {}
        }
    }

    // Highlight the Dimension tool's first-picked point.
    if let Some(i) = session.dim_first {
        if let Some(p) = session.sketch.points.get(i) {
            let iso = Isometry3d::new(ap.to_world(Vec2::new(p.x as f32, p.y as f32)), plane_rot);
            gizmos.circle(iso, 0.18 * ms, base_dim_col);
        }
    }
    // Indicator that the Dimension tool has a first line picked (for a line-to-line / angle
    // dimension): glow it and ring its endpoints, so it's clear what the next click pairs to.
    if session.tool == Tool::Dimension {
        if let Some((a, b)) = session.dim_line.and_then(|li| entity_line(&session.sketch, li)) {
            if let (Some(pa), Some(pb)) = (session.sketch.points.get(a), session.sketch.points.get(b)) {
                let (va, vb) = (Vec2::new(pa.x as f32, pa.y as f32), Vec2::new(pb.x as f32, pb.y as f32));
                let (wa, wb) = (ap.to_world(va), ap.to_world(vb));
                let off = ap.n.cross((wb - wa).normalize_or_zero()).normalize_or_zero() * 0.03;
                let hl = Color::srgb(1.0, 0.7, 0.1);
                gizmos.line(wa + off, wb + off, hl);
                gizmos.line(wa - off, wb - off, hl);
                for v in [va, vb] {
                    gizmos.circle(Isometry3d::new(ap.to_world(v), plane_rot), 0.12 * ms, hl);
                }
            }
        }
        // Same indicator for a first-picked slot: glow its centre line + ring its ends.
        if let Some((a, b, _)) = session.dim_slot.and_then(|si| entity_slot(&session.sketch, si)) {
            if let (Some(pa), Some(pb)) = (session.sketch.points.get(a), session.sketch.points.get(b)) {
                let (va, vb) = (Vec2::new(pa.x as f32, pa.y as f32), Vec2::new(pb.x as f32, pb.y as f32));
                let hl = Color::srgb(1.0, 0.7, 0.1);
                gizmos.line(ap.to_world(va), ap.to_world(vb), hl);
                for v in [va, vb] {
                    gizmos.circle(Isometry3d::new(ap.to_world(v), plane_rot), 0.12 * ms, hl);
                }
            }
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
        // Draw the rubber-band on the OVERLAY group (depth-biased, drawn on top) so it stays clearly
        // visible over a reference picture / the body — the default group washes out against an image.
        match session.tool {
            // Midpoint line: preview grows both ways from the first click (the centre).
            Tool::Line if session.line_midpoint => {
                let other = start * 2.0 - cur;
                overlay.line(ap.to_world(other), ap.to_world(cur), preview_col);
            }
            Tool::Line => overlay.line(ap.to_world(start), ap.to_world(cur), preview_col),
            // Perimeter circle: diameter from the first rim point to the cursor.
            Tool::Circle if session.circle_perimeter => {
                let center = (start + cur) * 0.5;
                let r = (cur - start).length() * 0.5;
                let iso = Isometry3d::new(ap.to_world(center), plane_rot);
                overlay.circle(iso, r, preview_col);
            }
            Tool::Circle => {
                let r = snap_radius(start.distance(cur), &session.reference_circles, session.snap_dist.max(SNAP));
                let iso = Isometry3d::new(ap.to_world(start), plane_rot);
                overlay.circle(iso, r, preview_col);
            }
            Tool::Rectangle => {
                let con_col = Color::srgba(0.9, 0.45, 0.95, 0.7);
                let quad = |g: &mut Gizmos<OverlayGizmos>, c: [Vec2; 4]| {
                    for k in 0..4 {
                        g.line(ap.to_world(c[k]), ap.to_world(c[(k + 1) % 4]), preview_col);
                    }
                };
                match session.rect_mode {
                    RectMode::Corner => {
                        quad(&mut overlay, [start, Vec2::new(cur.x, start.y), cur, Vec2::new(start.x, cur.y)]);
                    }
                    RectMode::Center => {
                        let o = start * 2.0 - cur; // opposite corner
                        let c = [o, Vec2::new(cur.x, o.y), cur, Vec2::new(o.x, cur.y)];
                        quad(&mut overlay, c);
                        overlay.line(ap.to_world(c[0]), ap.to_world(c[2]), con_col); // X diagonals
                        overlay.line(ap.to_world(c[1]), ap.to_world(c[3]), con_col);
                    }
                    RectMode::Parallelogram => {
                        if let Some(b) = session.pending_b {
                            let d = start + (cur - b);
                            quad(&mut overlay, [start, b, cur, d]);
                            draw_marker(&mut gizmos, ap, b, point_col, ms);
                        } else {
                            overlay.line(ap.to_world(start), ap.to_world(cur), preview_col);
                        }
                    }
                }
            }
            Tool::Slot => {
                let cl_col = Color::srgba(0.9, 0.45, 0.95, 0.6);
                let outline = |g: &mut Gizmos<OverlayGizmos>, poly: &[[f64; 2]]| {
                    let n = poly.len();
                    for k in 0..n {
                        let p = Vec2::new(poly[k][0] as f32, poly[k][1] as f32);
                        let q = Vec2::new(poly[(k + 1) % n][0] as f32, poly[(k + 1) % n][1] as f32);
                        g.line(ap.to_world(p), ap.to_world(q), preview_col);
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
                            outline(&mut overlay, &tessellate_slot([a.x as f64, a.y as f64], [b.x as f64, b.y as f64], perp_dist(cur, a, b).max(0.01) as f64));
                            overlay.line(ap.to_world(a), ap.to_world(b), cl_col);
                        } else {
                            overlay.line(ap.to_world(a), ap.to_world(b), preview_col);
                        }
                    }
                    SlotMode::Arc => {
                        let b = session.pending_b;
                        match (b, session.pending_c) {
                            (Some(b), Some(p)) => {
                                let r = arc_slot_width(cur, start, p, b).max(0.01);
                                outline(&mut overlay, &tessellate_arc_slot([start.x as f64, start.y as f64], [p.x as f64, p.y as f64], [b.x as f64, b.y as f64], r as f64));
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
                                    overlay.line(ap.to_world(p), ap.to_world(q), cl_col);
                                }
                                draw_marker(&mut gizmos, ap, b, point_col, ms);
                            }
                            _ => overlay.line(ap.to_world(start), ap.to_world(cur), preview_col),
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
                    overlay.line(ap.to_world(vert(k)), ap.to_world(vert((k + 1) % n)), preview_col);
                }
                // Dashed circumscribed circle.
                const SEG: usize = 64;
                let mut prev = start + Vec2::new(r, 0.0);
                for k in 1..=SEG {
                    let a = std::f32::consts::TAU * k as f32 / SEG as f32;
                    let p = start + Vec2::new(r * a.cos(), r * a.sin());
                    if k % 2 == 0 {
                        overlay.line(ap.to_world(prev), ap.to_world(p), construction);
                    }
                    prev = p;
                }
            }
            // 3-point arc: 1st click sets start; preview the chord, then the arc through cursor.
            Tool::Arc => match session.pending_b {
                None => overlay.line(ap.to_world(start), ap.to_world(cur), preview_col),
                Some(end) => {
                    if let Some(center) = circumcenter(start, end, cur) {
                        let tau = std::f32::consts::TAU;
                        let ang = |p: Vec2| (p.y - center.y).atan2(p.x - center.x).rem_euclid(tau);
                        let ccw = (ang(cur) - ang(start)).rem_euclid(tau) < (ang(end) - ang(start)).rem_euclid(tau);
                        let poly = tessellate_arc([center.x as f64, center.y as f64], [start.x as f64, start.y as f64], [end.x as f64, end.y as f64], ccw);
                        for w in poly.windows(2) {
                            overlay.line(
                                ap.to_world(Vec2::new(w[0][0] as f32, w[0][1] as f32)),
                                ap.to_world(Vec2::new(w[1][0] as f32, w[1][1] as f32)),
                                preview_col,
                            );
                        }
                    } else {
                        overlay.line(ap.to_world(start), ap.to_world(end), preview_col);
                    }
                    draw_marker(&mut gizmos, ap, end, point_col, ms);
                }
            },
            // Text commits on a single click (no rubber-band preview); Pattern / Mirror act
            // on existing geometry rather than rubber-banding a new entity.
            Tool::Select | Tool::Dimension | Tool::Spline | Tool::Text | Tool::Pattern | Tool::Mirror | Tool::Trim => {}
        }
        draw_marker(&mut gizmos, ap, start, point_col, ms);
    }

    // Power Trim: draw the cursor stroke so far (where it's cutting), plus the live segment.
    if session.tool == Tool::Trim && session.trim_mode == TrimMode::Power && !session.power_path.is_empty() {
        let trail = Color::srgb(1.0, 0.45, 0.2);
        for w in session.power_path.windows(2) {
            gizmos.line(ap.to_world(w[0]), ap.to_world(w[1]), trail);
        }
        if let (Some(&last), Some(cur)) = (session.power_path.last(), session.cursor_uv) {
            gizmos.line(ap.to_world(last), ap.to_world(cur), trail);
        }
    }
    // Corner mode: highlight the first picked line while waiting for the second.
    if session.tool == Tool::Trim && session.trim_mode == TrimMode::Corner {
        if let Some((a, b)) = session.trim_first.and_then(|li| entity_line(&session.sketch, li)) {
            gizmos.line(
                ap.to_world(pt2(&session.sketch, a)),
                ap.to_world(pt2(&session.sketch, b)),
                Color::srgb(0.3, 0.9, 1.0),
            );
        }
    }
    // Trim/Power hover preview: the piece that a click (or stroke crossing) would delete, in red.
    if session.tool == Tool::Trim && session.trim_mode != TrimMode::Corner {
        if let Some(uv) = session.cursor_uv {
            let red = Color::srgb(1.0, 0.3, 0.3);
            let thresh = ms * 0.04 + 0.25;
            let line = nearest_line(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
            let circ = nearest_circle(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
            let arc = nearest_arc(&session.sketch, uv, thresh).map(|i| (i, dist_to_entity(&session.sketch, i, uv)));
            if let Some((ei, _)) = [line, circ, arc].into_iter().flatten().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                match &session.sketch.entities[ei] {
                    SketchEntity::Arc { .. } => {
                        if let (Some((_, center, r, theta_start, span)), Some((u_lo, u_hi))) =
                            (arc_geom(&session.sketch, ei), arc_trim_bracket(&session.sketch, ei, uv))
                        {
                            let n = (((u_hi - u_lo) * span.abs() / std::f32::consts::TAU * 64.0).ceil() as usize).max(2);
                            for k in 0..n {
                                let th0 = theta_start + span * (u_lo + (u_hi - u_lo) * (k as f32 / n as f32));
                                let th1 = theta_start + span * (u_lo + (u_hi - u_lo) * ((k + 1) as f32 / n as f32));
                                gizmos.line(
                                    ap.to_world(center + Vec2::new(th0.cos(), th0.sin()) * r),
                                    ap.to_world(center + Vec2::new(th1.cos(), th1.sin()) * r),
                                    red,
                                );
                            }
                        }
                    }
                    SketchEntity::Circle { center, radius, .. } => {
                        let (c, r) = (pt2(&session.sketch, *center), *radius as f32);
                        if let Some((lo, hi)) = circle_trim_interval(&session.sketch, ei, uv) {
                            // The removed arc lo→hi (the side the cursor is on).
                            let span = hi - lo;
                            let n = ((span.abs() / std::f32::consts::TAU * 64.0).ceil() as usize).max(2);
                            for k in 0..n {
                                let a0 = lo + span * (k as f32 / n as f32);
                                let a1 = lo + span * ((k + 1) as f32 / n as f32);
                                gizmos.line(
                                    ap.to_world(c + Vec2::new(a0.cos(), a0.sin()) * r),
                                    ap.to_world(c + Vec2::new(a1.cos(), a1.sin()) * r),
                                    red,
                                );
                            }
                        }
                    }
                    _ => {
                        if let Some((li, t_lo, t_hi)) = trim_bracket(&session.sketch, uv, thresh) {
                            if let Some((ia, ib)) = entity_line(&session.sketch, li) {
                                let (a, b) = (pt2(&session.sketch, ia), pt2(&session.sketch, ib));
                                gizmos.line(ap.to_world(a + (b - a) * t_lo), ap.to_world(a + (b - a) * t_hi), red);
                            }
                        }
                    }
                }
            }
        }
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
        let regions = session.cached_regions();
        let picked: Vec<usize> =
            session.selected_contours.iter().copied().filter(|&i| i < regions.len()).collect();
        // "All contours" default skips nested disks (they'd double-draw over the
        // region that owns them as a hole) — same rule the regen applies.
        let indices: Vec<usize> = if picked.is_empty() {
            regions.iter().enumerate().filter(|(_, r)| !r.nested).map(|(i, _)| i).collect()
        } else {
            picked
        };
        // A boss goes out along +normal by default; a cut goes in (−normal, into the
        // material). Reverse flips either one.
        if matches!(op.kind, OpKind::Revolve | OpKind::RevolveCut) {
            // Revolve preview: highlight the axis line, then ghost the profile swept
            // around it — rotated profile copies (the last one bright, where the sweep
            // ends) plus the circular paths of sampled profile vertices. A cut reads red.
            let cut = matches!(op.kind, OpKind::RevolveCut);
            let Some((p, d)) = revolve_axis(&session) else { return };
            let a2 = Vec2::new(p[0] as f32, p[1] as f32);
            let dir2d = Vec2::new(d[0] as f32, d[1] as f32).normalize_or_zero();
            let half = (ms.max(2.0)) * 1.5;
            let axis_col = if cut { Color::srgba(1.0, 0.45, 0.4, 0.9) } else { Color::srgba(0.4, 0.85, 0.95, 0.9) };
            overlay.line(ap.to_world(a2 - dir2d * half), ap.to_world(a2 + dir2d * half), axis_col);

            // World-space axis (point + unit direction) and Rodrigues rotation about it.
            let ao = ap.to_world(a2);
            let k = (ap.u * dir2d.x + ap.v * dir2d.y).normalize_or_zero();
            if k == Vec3::ZERO {
                return;
            }
            let rot = |w: Vec3, th: f32| {
                let dl = w - ao;
                ao + dl * th.cos() + k.cross(dl) * th.sin() + k * k.dot(dl) * (1.0 - th.cos())
            };
            let angle = op.depth.clamp(1.0, 360.0).to_radians() * if op.reverse { -1.0 } else { 1.0 };
            let ghost = if cut { Color::srgba(1.0, 0.4, 0.35, 0.6) } else { Color::srgba(0.95, 0.85, 0.25, 0.6) };
            let far = if cut { Color::srgb(1.0, 0.75, 0.2) } else { Color::srgba(0.95, 0.85, 0.25, 0.95) };
            // One profile copy every ~30° of sweep; the vertex paths step every ~15°.
            let copies = ((angle.abs().to_degrees() / 30.0).ceil() as usize).clamp(1, 12);
            let path_segs = ((angle.abs().to_degrees() / 15.0).ceil() as usize).clamp(2, 24);
            for &i in &indices {
                for loop_pts in std::iter::once(&regions[i].outer).chain(regions[i].holes.iter()) {
                    let m = loop_pts.len();
                    if m < 2 {
                        continue;
                    }
                    let wpt = |j: usize| {
                        let q = loop_pts[j % m];
                        ap.to_world(Vec2::new(q[0] as f32, q[1] as f32))
                    };
                    // Rotated profile copies; the final one (bright) is where the sweep ends.
                    for s in 1..=copies {
                        let th = angle * s as f32 / copies as f32;
                        let col = if s == copies { far } else { ghost };
                        for j in 0..m {
                            overlay.line(rot(wpt(j), th), rot(wpt(j + 1), th), col);
                        }
                    }
                    // Circular paths of sampled profile vertices, 0 → angle.
                    for j in (0..m).step_by((m / 12).max(1)) {
                        let w0 = wpt(j);
                        let mut prev = w0;
                        for s in 1..=path_segs {
                            let cur = rot(w0, angle * s as f32 / path_segs as f32);
                            overlay.line(prev, cur, ghost);
                            prev = cur;
                        }
                    }
                }
            }
            return;
        }
        let kind_sign = match op.kind {
            OpKind::Boss => 1.0,
            OpKind::Cut => -1.0,
            OpKind::Revolve | OpKind::RevolveCut => 1.0, // unreachable (revolve returns above)
        };
        let nominal = kind_sign * if op.reverse { -1.0 } else { 1.0 };
        let lift = ap.n * (op.depth * nominal);
        let ghost = match op.kind {
            OpKind::Boss => Color::srgba(0.95, 0.85, 0.25, 0.8),
            OpKind::Cut => Color::srgba(1.0, 0.4, 0.35, 0.8),
            OpKind::Revolve | OpKind::RevolveCut => Color::srgba(0.4, 0.85, 0.95, 0.8),
        };

        // Ghost prism, drawn on the overlay group so it shows THROUGH the model — the
        // far end is the cut-depth indicator (where the cut bottoms out, à la SW).
        // The far loop is drawn brighter so the depth reads clearly.
        let far = match op.kind {
            OpKind::Boss => ghost,
            OpKind::Cut => Color::srgb(1.0, 0.75, 0.2), // bright depth ring for a cut
            OpKind::Revolve | OpKind::RevolveCut => ghost, // unreachable (revolve returns above)
        };
        // Direction 2 extends the prism the opposite way by `depth2`.
        let lift2 = if op.dir2 { ap.n * (-op.depth2 * nominal) } else { Vec3::ZERO };
        for &i in &indices {
            for loop_pts in std::iter::once(&regions[i].outer).chain(regions[i].holes.iter()) {
                let m = loop_pts.len();
                for k in 0..m {
                    let a = Vec2::new(loop_pts[k][0] as f32, loop_pts[k][1] as f32);
                    let b = Vec2::new(loop_pts[(k + 1) % m][0] as f32, loop_pts[(k + 1) % m][1] as f32);
                    overlay.line(ap.to_world(a) + lift, ap.to_world(b) + lift, far); // far loop (depth)
                    overlay.line(ap.to_world(a), ap.to_world(a) + lift, ghost); // riser
                    if op.dir2 {
                        overlay.line(ap.to_world(a) + lift2, ap.to_world(b) + lift2, far); // Direction-2 far loop
                        overlay.line(ap.to_world(a), ap.to_world(a) + lift2, ghost); // Direction-2 riser
                    }
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
            // Direction-2 arrow: points the OPPOSITE way along the normal, draggable to set depth2.
            if op.dir2 {
                let acol2 = if session.arrow_drag2 { Color::srgb(1.0, 1.0, 0.5) } else { Color::srgb(0.5, 0.8, 1.0) };
                let handle2 = c - ap.n * op.depth2.max(0.5 * ms.max(1.0));
                overlay.arrow(c, handle2, acol2);
                overlay.sphere(Isometry3d::from_translation(handle2), 0.15 * ms.max(1.0), acol2);
            }
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
    let regions = session.cached_regions();
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
/// Drag the reference-plane creation arrow (along the base normal) to set the offset live — the
/// plane analogue of [`extrude_arrow_drag`]. The drag sign sets the `flip` flag, so a wobble past
/// the base flips sides cleanly. Returns true while dragging (so the caller swallows the click).
/// Geometry of the on-screen section-plane gizmo: (rect centre at the current offset, plane
/// u axis, v axis, normal, arrow tip, rect half-extent). Shared by the drawing and the drag
/// hit-test so the grab targets always match what's on screen.
fn section_gizmo_geom(part: &Part, spec: &SectionSpec) -> Option<(Vec3, Vec3, Vec3, Vec3, Vec3, f32)> {
    let mesh = part.mesh.as_ref()?;
    if mesh.positions.is_empty() {
        return None;
    }
    let (lo, hi) = mesh_bbox(mesh);
    let c = (lo + hi) * 0.5;
    let (u, v, n) = section_axes(spec);
    let base = c - n * (c.dot(n) - spec.offset); // bbox centre projected onto the section plane
    let diag = (hi - lo).length().max(1.0);
    let dir = if spec.flip { -n } else { n }; // points toward the DISCARDED side (the view direction)
    let tip = base + dir * (diag * 0.28);
    Some((base, u, v, n, tip, diag * 0.55))
}

/// The four rotation handles on the gizmo rectangle's edge midpoints. Top/bottom (±v) tilt
/// about the u axis (`rot_u`), left/right (±u) tilt about the v axis (`rot_v`).
/// Returns (world position, which angle it drives: 0 = rot_u, 1 = rot_v).
fn section_rot_handles(base: Vec3, u: Vec3, v: Vec3, half: f32) -> [(Vec3, u8); 4] {
    [
        (base + v * half, 0),
        (base - v * half, 0),
        (base + u * half, 1),
        (base - u * half, 1),
    ]
}

/// Where the mouse ray pierces the plane through `base` perpendicular to `axis`, as a unit
/// direction from `base` (the rotation-drag "clock hand"). None if the ray runs parallel.
fn section_rot_dir(base: Vec3, axis: Vec3, ray: &Ray3d) -> Option<Vec3> {
    let rd = ray.direction.as_vec3();
    let denom = rd.dot(axis);
    if denom.abs() < 1e-5 {
        return None;
    }
    let t = (base - ray.origin).dot(axis) / denom;
    if !t.is_finite() || t <= 0.0 {
        return None;
    }
    let w = ray.origin + rd * t - base;
    let w = w - axis * w.dot(axis);
    (w.length() > 1e-4).then(|| w.normalize())
}

/// Drag the section gizmo (view mode): the normal arrow slides the cut along the plane's
/// normal; the four edge-midpoint handles rotate the plane about its in-plane axes.
#[allow(clippy::too_many_arguments)]
fn section_arrow_drag(
    session: &mut SketchSession,
    ui_state: &mut UiState,
    part: &Part,
    window: &Window,
    camera: &Camera,
    cam_gt: &GlobalTransform,
    ray: &Ray3d,
    just_pressed: bool,
    pressed: bool,
    just_released: bool,
) -> bool {
    let Some(spec) = ui_state.section else { return false };
    let Some((base, u, v, n, tip, half)) = section_gizmo_geom(part, &spec) else { return false };
    if just_pressed && !session.arrow_drag && session.section_rot.is_none() {
        if let Some(cursor) = window.cursor_position() {
            // Rotation handles win over the arrow when both are in reach.
            let mut grabbed = false;
            for (pos, idx) in section_rot_handles(base, u, v, half) {
                let near = camera.world_to_viewport(cam_gt, pos).map(|p| p.distance(cursor) < 18.0).unwrap_or(false);
                if near {
                    // The world axis this angle actually rotates about (see `section_axes`):
                    // rot_u turns about the final u; rot_v about the UNROTATED base v.
                    let axis = if idx == 0 {
                        u
                    } else {
                        let pr = standard_plane_ref(spec.which);
                        Vec3::new(pr.v[0] as f32, pr.v[1] as f32, pr.v[2] as f32)
                    };
                    if let Some(dir) = section_rot_dir(base, axis, ray) {
                        session.section_rot = Some((idx, axis, dir));
                        grabbed = true;
                    }
                    break;
                }
            }
            if !grabbed && session.section_rot.is_none() {
                let near_shaft = segment_screen_dist(camera, cam_gt, cursor, base, tip).is_some_and(|d| d < 22.0);
                let near_tip = camera.world_to_viewport(cam_gt, tip).map(|p| p.distance(cursor) < 26.0).unwrap_or(false);
                if near_shaft || near_tip {
                    session.arrow_drag = true;
                    // Anchor the grab: remember how far along the axis the click landed from
                    // the current offset, so the plane follows the mouse instead of jumping.
                    let base0 = base - n * spec.offset;
                    let t0 = closest_t_on_axis(base0, n, ray.origin, ray.direction.as_vec3());
                    session.section_grab = if t0.is_finite() { spec.offset - t0 } else { 0.0 };
                }
            }
        }
    }
    if let Some((idx, axis, last)) = session.section_rot {
        if pressed {
            if let Some(dir) = section_rot_dir(base, axis, ray) {
                let delta = last.cross(dir).dot(axis).atan2(last.dot(dir).clamp(-1.0, 1.0)).to_degrees();
                let mut s = spec;
                let a = if idx == 0 { &mut s.rot_u } else { &mut s.rot_v };
                *a += delta;
                if *a > 180.0 {
                    *a -= 360.0;
                } else if *a < -180.0 {
                    *a += 360.0;
                }
                ui_state.section = Some(s);
                session.section_rot = Some((idx, axis, dir));
            }
            if just_released {
                session.section_rot = None;
            }
            return true;
        }
        session.section_rot = None;
    }
    if session.arrow_drag && pressed {
        // The axis through the plane's zero position: t along `n` IS the offset.
        let base0 = base - n * spec.offset;
        let t = closest_t_on_axis(base0, n, ray.origin, ray.direction.as_vec3());
        if t.is_finite() {
            ui_state.section = Some(SectionSpec { offset: t + session.section_grab, ..spec });
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

/// Draw the section-plane gizmo: the cutting rectangle, a direction arrow (drag to slide the
/// cut) and four edge-midpoint handles (drag to tilt the plane) — SolidWorks-style.
fn draw_section_gizmo(mut overlay: Gizmos<OverlayGizmos>, ui_state: Res<UiState>, part: Res<Part>, session: Res<SketchSession>) {
    let Some(spec) = ui_state.section else { return };
    let Some((base, ua, va, n, tip, half)) = section_gizmo_geom(&part, &spec) else { return };
    let u = ua * half;
    let v = va * half;
    let col = Color::srgba(1.0, 0.6, 0.15, 0.9);
    let faint = Color::srgba(1.0, 0.6, 0.15, 0.25);
    // The cutting rectangle + corner-to-corner cross so the plane reads as a surface.
    let corners = [base - u - v, base + u - v, base + u + v, base - u + v];
    for k in 0..4 {
        overlay.line(corners[k], corners[(k + 1) % 4], col);
    }
    overlay.line(corners[0], corners[2], faint);
    overlay.line(corners[1], corners[3], faint);
    // The offset arrow (highlighted while dragging).
    let acol = if session.arrow_drag { Color::srgb(1.0, 1.0, 0.5) } else { col };
    overlay.line(base, tip, acol);
    let d = (tip - base).normalize_or_zero();
    let side = d.any_orthonormal_vector() * (half * 0.06);
    let back = tip - d * (half * 0.12);
    overlay.line(tip, back + side, acol);
    overlay.line(tip, back - side, acol);
    let side2 = d.cross(side.normalize_or_zero()) * (half * 0.06);
    overlay.line(tip, back + side2, acol);
    overlay.line(tip, back - side2, acol);
    // Rotation handles: small diamonds on the edge midpoints, drawn in each handle's drag
    // plane so they read as "turns this way". Highlighted while that angle is being dragged.
    let r = half * 0.045;
    for (pos, idx) in section_rot_handles(base, ua, va, half) {
        let active = session.section_rot.is_some_and(|(i, _, _)| i == idx);
        let hcol = if active { Color::srgb(1.0, 1.0, 0.5) } else { col };
        // Diamond spanning the rotation direction (normal-ward) and the edge direction.
        let along = if idx == 0 { va } else { ua };
        let pts = [pos + n * r, pos + along * r, pos - n * r, pos - along * r];
        for k in 0..4 {
            overlay.line(pts[k], pts[(k + 1) % 4], hcol);
        }
    }
}

fn plane_arrow_drag(
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
    let Some(spec) = ui_state.plane_spec.clone() else { return false };
    let base = spec.base.origin;
    let n = spec.base.n.normalize_or_zero();
    if n == Vec3::ZERO {
        return false;
    }
    let signed = if spec.flip { -spec.offset } else { spec.offset };
    let tip = base + n * signed;
    if just_pressed {
        if let Some(cursor) = window.cursor_position() {
            let near_shaft = segment_screen_dist(camera, cam_gt, cursor, base, tip).is_some_and(|d| d < 22.0);
            let near_tip = camera.world_to_viewport(cam_gt, tip).map(|p| p.distance(cursor) < 26.0).unwrap_or(false);
            if near_shaft || near_tip {
                session.arrow_drag = true;
            }
        }
    }
    if session.arrow_drag && pressed {
        let t = closest_t_on_axis(base, n, ray.origin, ray.direction.as_vec3());
        if t.is_finite() {
            if let Some(s) = ui_state.plane_spec.as_mut() {
                s.flip = t < 0.0;
                s.offset = t.abs().clamp(0.0, 100_000.0);
            }
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
        // `t` is the signed distance dragged along +normal. Drag past the back of the sketch and
        // the direction flips (sign → Reverse), like SolidWorks; a small deadzone near zero keeps
        // a wobble from flickering the side. A non-finite `t` (axis edge-on) is ignored (NaN
        // crashes egui).
        let t = closest_t_on_axis(base, n, ray.origin, ray.direction.as_vec3());
        if t.is_finite() {
            let mut p = op.clone();
            p.depth = t.abs().clamp(0.1, 10_000.0);
            if t.abs() > 0.05 {
                p.reverse = t < 0.0;
            }
            ui_state.pending = Some(p);
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

/// Drag the *Direction 2* arrow (the one pointing the opposite way along the normal) to set
/// `depth2`. Mirrors [`extrude_arrow_drag`] but on the −normal side, and never touches `reverse`
/// (Direction 2 is always relative to Direction 1).
fn extrude_dir2_arrow_drag(
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
    if !op.dir2 {
        return false;
    }
    let Some(base_uv) = contours_centroid(session) else { return false };
    let base = ap.to_world(base_uv);
    let n = -ap.n.normalize_or_zero(); // Direction 2 points the opposite way
    let ms = if session.snap_dist > 1e-6 { (session.snap_dist / SNAP).clamp(0.5, 40.0) } else { 1.0 };
    let tip = base + n * op.depth2.max(0.5 * ms.max(1.0));

    if just_pressed {
        if let Some(cursor) = window.cursor_position() {
            let near_shaft = segment_screen_dist(camera, cam_gt, cursor, base, tip).is_some_and(|d| d < 22.0);
            let near_tip = camera
                .world_to_viewport(cam_gt, tip)
                .map(|p| p.distance(cursor) < 26.0)
                .unwrap_or(false);
            if near_shaft || near_tip {
                session.arrow_drag2 = true;
            }
        }
    }
    if session.arrow_drag2 && pressed {
        // Distance dragged along the −normal axis sets depth2 (magnitude only).
        let t = closest_t_on_axis(base, n, ray.origin, ray.direction.as_vec3());
        if t.is_finite() {
            let mut p = op.clone();
            p.depth2 = t.abs().clamp(0.1, 10_000.0);
            ui_state.pending = Some(p);
        }
        if just_released {
            session.arrow_drag2 = false;
        }
        return true;
    }
    if just_released {
        session.arrow_drag2 = false;
    }
    false
}

/// Index of the body edge segment nearest the cursor in screen space, within
/// `thresh` pixels.
/// Snap a raw face-hit to the nearest body feature point in screen space (within ~14 px):
/// an edge endpoint, an edge midpoint, or a circular-edge centre — so a hole drops precisely
/// onto a vertex / hole centre. Falls back to the raw `hit` when nothing is close.
fn snap_place_point(part: &Part, camera: &Camera, cam_gt: &GlobalTransform, cursor: Vec2, hit: Vec3) -> Vec3 {
    let mut cands: Vec<Vec3> = Vec::new();
    for e in &part.edges {
        let a = Vec3::from_array(e[0]);
        let b = Vec3::from_array(e[1]);
        cands.push(a);
        cands.push(b);
        cands.push((a + b) * 0.5);
    }
    // Circular-edge centres (concentric holes/bosses): the centroid of a closed edge loop.
    let mut seen = vec![false; part.edges.len()];
    for i in 0..part.edges.len() {
        if seen[i] {
            continue;
        }
        let (chain, closed) = edge_chain(&part.edges, i);
        for (j, e) in part.edges.iter().enumerate() {
            if chain.iter().any(|p| p.distance(Vec3::from_array(e[0])) < 1e-4) {
                seen[j] = true; // don't re-walk this loop
            }
        }
        if closed && chain.len() >= 8 {
            cands.push(chain.iter().fold(Vec3::ZERO, |s, p| s + *p) / chain.len() as f32);
        }
    }
    let mut best: Option<(Vec3, f32)> = None;
    for c in cands {
        if let Ok(s) = camera.world_to_viewport(cam_gt, c) {
            let d = s.distance(cursor);
            if d <= 14.0 && best.map_or(true, |(_, bd)| d < bd) {
                best = Some((c, d));
            }
        }
    }
    best.map(|(c, _)| c).unwrap_or(hit)
}

/// Raycast the cursor onto the body mesh: the nearest triangle hit, returning the world hit
/// point and that face's normal (oriented toward the camera). Used to place a threaded hole.
/// Nearest measure target under the cursor: a body edge endpoint or midpoint within ~14 px
/// (screen space), else the raw surface hit. So a measurement snaps to vertices/midpoints.
fn nearest_measure_point(part: &Part, camera: &Camera, cam_gt: &GlobalTransform, cursor: Vec2) -> Option<Vec3> {
    let mut best: Option<(f32, Vec3)> = None;
    for e in &part.edges {
        let (a, b) = (Vec3::from_array(e[0]), Vec3::from_array(e[1]));
        for p in [a, b, (a + b) * 0.5] {
            if let Ok(s) = camera.world_to_viewport(cam_gt, p) {
                let d = s.distance(cursor);
                if d < 14.0 && best.map_or(true, |(bd, _)| d < bd) {
                    best = Some((d, p));
                }
            }
        }
    }
    if let Some((_, p)) = best {
        return Some(p);
    }
    part.mesh.as_ref().and_then(|m| pick_face_point(m, camera, cam_gt, cursor).map(|(hit, _)| hit))
}

/// Draw the measure tool's picked points (small 3D crosses) and the line between two of them.
fn draw_measure(mut gizmos: Gizmos, ui_state: Res<UiState>, cam_q: Query<&OrbitCamera>) {
    if ui_state.measure_pts.is_empty() {
        return;
    }
    let r = cam_q.single().map(|c| c.radius).unwrap_or(12.0);
    let s = (r * 0.012).max(0.05);
    let col = Color::srgb(1.0, 0.85, 0.3);
    for &p in &ui_state.measure_pts {
        gizmos.line(p - Vec3::X * s, p + Vec3::X * s, col);
        gizmos.line(p - Vec3::Y * s, p + Vec3::Y * s, col);
        gizmos.line(p - Vec3::Z * s, p + Vec3::Z * s, col);
    }
    if ui_state.measure_pts.len() == 2 {
        gizmos.line(ui_state.measure_pts[0], ui_state.measure_pts[1], col);
    }
}

fn pick_face_point(mesh: &TriMesh, camera: &Camera, cam_gt: &GlobalTransform, cursor: Vec2) -> Option<(Vec3, Vec3)> {
    let ray = camera.viewport_to_world(cam_gt, cursor).ok()?;
    let (orig, dir) = (ray.origin, *ray.direction);
    let mut best: Option<(f32, Vec3)> = None; // (t, normal)
    for tri in mesh.indices.chunks_exact(3) {
        let a = Vec3::from_array(mesh.positions[tri[0] as usize]);
        let b = Vec3::from_array(mesh.positions[tri[1] as usize]);
        let c = Vec3::from_array(mesh.positions[tri[2] as usize]);
        if let Some(t) = ray_triangle(orig, dir, a, b, c) {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, (b - a).cross(c - a).normalize_or_zero()));
            }
        }
    }
    best.map(|(t, mut n)| {
        if n.dot(dir) > 0.0 {
            n = -n; // face toward the camera (outward)
        }
        (orig + dir * t, n)
    })
}

/// Pick an edge loop for fillet/chamfer from BOTH selectable pools: sharp edges first (they win
/// any tie), then the fillet-boundary seams. The seam pool is what a fillet leaves behind on a
/// round body — its boundary meets the walls smoothly, so it's no longer a sharp edge, and without
/// this fallback a rounded rim could never be picked again for a second fillet/chamfer. The loop
/// is walked inside whichever pool the hit came from.
/// Pick the body edge under the cursor from either edge pool (sharp edges first, then a
/// round body's tangent fillet seams). `loop_snap` chooses the expansion: `true` grabs the
/// smallest closed planar loop through the clicked edge (a face's whole perimeter in one
/// click); `false` grabs just the single edge (the maximal smooth chain, stopping at sharp
/// corners) — SolidWorks-style single-edge picking.
fn pick_edge_loop_any(
    part: &Part,
    camera: &Camera,
    cam_gt: &GlobalTransform,
    cursor: Vec2,
    thresh: f32,
    loop_snap: bool,
) -> Option<(Vec<Vec3>, bool)> {
    let expand = if loop_snap { edge_loop } else { edge_chain };
    if let Some(si) = pick_edge(&part.edges, camera, cam_gt, cursor, thresh) {
        let (chain, closed) = expand(&part.edges, si);
        if chain.len() >= 2 {
            return Some((chain, closed));
        }
    }
    let si = pick_edge(&part.seam_edges, camera, cam_gt, cursor, thresh)?;
    let (chain, closed) = expand(&part.seam_edges, si);
    (chain.len() >= 2).then_some((chain, closed))
}

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

/// Like [`edge_chain`], but when the tangent walk is *open* (it stopped at sharp corners), try
/// to snap to the smallest **closed, planar** edge loop through the clicked edge — e.g. a box
/// face's whole perimeter from one click. Selecting the full loop also hands the bevel a
/// complete, corner-closed set, so it rounds cleanly instead of falling back at a lone edge.
/// Falls back to the open chain when no planar loop exists (a truly isolated edge).
fn edge_loop(edges: &[[[f32; 3]; 2]], seed: usize) -> (Vec<Vec3>, bool) {
    use std::collections::HashMap;
    let (chain, closed) = edge_chain(edges, seed);
    if closed {
        return (chain, true);
    }
    // Rebuild the welded segment graph.
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
    let (sa, sb) = seg[seed];
    // BFS the shortest path sa→sb that avoids the seed segment — that path plus the seed is the
    // smallest cycle through the clicked edge.
    let mut prev = vec![usize::MAX; pos.len()];
    let mut seen = vec![false; pos.len()];
    let mut q = std::collections::VecDeque::new();
    seen[sa] = true;
    q.push_back(sa);
    while let Some(v) = q.pop_front() {
        if v == sb {
            break;
        }
        for &sg in &adj[v] {
            if sg == seed {
                continue;
            }
            let (a, b) = seg[sg];
            let w = if a == v { b } else { a };
            if !seen[w] {
                seen[w] = true;
                prev[w] = v;
                q.push_back(w);
            }
        }
    }
    if !seen[sb] {
        return (chain, false); // no cycle → lone edge
    }
    // Reconstruct the loop vertices sa..sb.
    let mut loop_ids = vec![sb];
    let mut v = sb;
    while v != sa {
        v = prev[v];
        if v == usize::MAX {
            return (chain, false);
        }
        loop_ids.push(v);
    }
    let pts: Vec<Vec3> = loop_ids.iter().map(|&i| pos[i]).collect();
    // Accept only a reasonably planar loop (so we grab a flat face perimeter, not a path that
    // wanders over the body). Newell normal, then max out-of-plane distance.
    let c = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
    let mut nrm = Vec3::ZERO;
    for i in 0..pts.len() {
        let (p, qn) = (pts[i] - c, pts[(i + 1) % pts.len()] - c);
        nrm += p.cross(qn);
    }
    let nrm = nrm.normalize_or_zero();
    let span = pts.iter().map(|p| (*p - c).length()).fold(0.0_f32, f32::max).max(1e-3);
    let flat = nrm != Vec3::ZERO && pts.iter().all(|p| (*p - c).dot(nrm).abs() < 0.02 * span);
    if flat {
        (pts, true)
    } else {
        (chain, false)
    }
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
        // Pre-highlight the loop under the cursor (soft cyan) so a click's target is obvious.
        if let Some((chain, closed)) = &ui_state.hover_edge_loop {
            let hcol = Color::srgb(0.4, 0.85, 1.0);
            for w in chain.windows(2) {
                gizmos.line(nudge(w[0]), nudge(w[1]), hcol);
            }
            if *closed && chain.len() >= 2 {
                gizmos.line(nudge(*chain.last().unwrap()), nudge(chain[0]), hcol);
            }
        }
        let fcol = Color::srgb(1.0, 0.95, 0.2);
        let v3 = |p: &[f64; 3]| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32);
        for edge in &ui_state.fillet_edges {
            for w in edge.windows(2) {
                gizmos.line(nudge(v3(&w[0])), nudge(v3(&w[1])), fcol);
            }
            // The polyline is stored open. If it's actually a closed loop (its two ends are about one
            // segment apart, not a full span), draw the closing segment so a filleted rim reads as an
            // unbroken circle — without feeding the bevel a duplicate seam point (which breaks it).
            if edge.len() >= 3 {
                let (first, last) = (edge[0], edge[edge.len() - 1]);
                let seg0 = (v3(&edge[1]) - v3(&edge[0])).length();
                if (v3(&last) - v3(&first)).length() <= seg0 * 2.0 {
                    gizmos.line(nudge(v3(&last)), nudge(v3(&first)), fcol);
                }
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

    /// On-demand manifold check of a saved part: load the `.hcad`, regenerate the mesh body, and
    /// For a face-built Extrude/Cut, show where face re-projection moves its plane vs the stored one.
    ///   HCAD_FILE="…\part.hcad" cargo test -p hworks-app diag_face_reprojection -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diag_face_reprojection() {
        let path = std::env::var("HCAD_FILE").expect("set HCAD_FILE");
        let doc: Document = ron::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for i in 0..doc.features.len() {
            let (sketch, plane, regions, kind) = match &doc.features[i].kind {
                FeatureKind::Extrude { sketch, plane, regions, .. } => (sketch, plane, regions, "boss"),
                FeatureKind::Cut { sketch, plane, regions, .. } => (sketch, plane, regions, "cut"),
                _ => continue,
            };
            if plane.datum {
                continue;
            }
            // Body from all features strictly before i.
            let mut d = doc.clone();
            d.rollback = i;
            let body = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&d))).ok().flatten().map(|(m, _)| m);
            let Some(body) = body.filter(|m| !m.positions.is_empty()) else {
                eprintln!("feat {i} ({kind}): no body before it");
                continue;
            };
            let all = sketch.regions();
            let regs = merge_regions(&chosen_regions(&all, regions));
            if regs.is_empty() {
                continue;
            }
            let refw = sketch_footprint_world(plane, regs.iter().flat_map(|r| r.outer.iter()));
            let samples = sketch_footprint_samples(plane, &regs);
            let rp = reproject_plane_on_mesh(plane, &body, &samples);
            let d3 = |a: [f64; 3], b: [f64; 3]| (((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)) as f64).sqrt();
            let moved = d3(plane.origin, rp.origin);
            eprintln!(
                "feat {i} ({kind}): stored={:.2?} reproj={:.2?} moved={moved:.3}  ref_world=({:.2},{:.2},{:.2})",
                plane.origin, rp.origin, refw.x, refw.y, refw.z
            );
        }
    }

    /// A thin-feature extrude of a square profile must produce a hollow box shell whose volume is the
    /// wall-annulus area × height — not the filled prism. Validates `thin_wall_mesh` (grow/shrink offset
    /// + boolean difference) end-to-end via the mesh's signed volume.
    #[test]
    fn thin_feature_makes_a_hollow_shell() {
        // 20×20 square, centred, on the XY plane; extrude up 10 as a 2mm-thick outward wall.
        let s = 10.0;
        let outer = vec![[-s, -s], [s, -s], [s, s], [-s, s]];
        let basis = PlaneBasis {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 1.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let thick = 2.0;
        let height = 10.0;
        let mesh = thin_wall_mesh(&outer, &[], &basis, 0.0, height, thick, 0).expect("thin wall");
        // Signed volume of the closed triangle soup (divergence theorem).
        let vol = {
            let p = &mesh.positions;
            let mut v = 0.0f64;
            for t in mesh.indices.chunks(3) {
                let a = p[t[0] as usize];
                let b = p[t[1] as usize];
                let c = p[t[2] as usize];
                let (a, b, c) = ([a[0] as f64, a[1] as f64, a[2] as f64], [b[0] as f64, b[1] as f64, b[2] as f64], [c[0] as f64, c[1] as f64, c[2] as f64]);
                v += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1])) / 6.0;
            }
            v.abs()
        };
        // Outer 24×24 (grew 2 outward), inner 20×20 → annulus 576−400=176, × height 10 = 1760.
        let outer_area = (2.0 * (s + thick)).powi(2);
        let inner_area = (2.0 * s).powi(2);
        let expected = (outer_area - inner_area) * height;
        assert!((vol - expected).abs() < expected * 0.02, "thin-wall volume {vol:.1} vs expected {expected:.1}");
        // Sanity: a hollow shell is far less than the filled prism (24×24×10 = 5760).
        assert!(vol < outer_area * height * 0.7, "shell should be hollow, got {vol:.1}");
    }

    /// A rim picked the way the APP picks it (from the tessellated body, chained by `edge_loop`)
    /// must fillet completely: every rim edge selected, mesh surgery (no CSG fallback), and the
    /// seam edges forming complete degree-2 rings on the ideal circles — no notch, no chevron, no
    /// break. Covers BOTH the new store format (closed loop = first point repeated) and the heal
    /// for old documents (open-stored loop, closing pair matched by the wrap in `edge_is_picked`).
    #[test]
    fn rim_pick_from_tessellation_closes_the_loop() {
        let basis = PlaneBasis { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let (rad, h, fr) = (12.0_f64, 9.2_f64, 1.0_f64);
        let n = 48;
        let circ: Vec<[f64; 2]> = (0..n).map(|i| { let a = std::f64::consts::TAU * i as f64 / n as f64; [rad * a.cos(), rad * a.sin()] }).collect();
        let cyl = extrude_tool_mesh(&circ, &[], &basis, 0.0, h).expect("cylinder");
        let tess0 = mesh_tessellation(cyl.clone());
        let rim_seed = tess0
            .edges
            .iter()
            .position(|e| (e[0][2] - h as f32).abs() < 1e-3 && (e[1][2] - h as f32).abs() < 1e-3)
            .expect("find a top-rim segment");
        let (chain, closed) = edge_loop(&tess0.edges, rim_seed);
        assert!(closed && chain.len() >= n - 2, "rim should chain into a closed loop");
        let open: Vec<[f64; 3]> = chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect();
        let mut shut = open.clone();
        shut.push(shut[0]); // what toggle_fillet_edge now stores for a closed pick
        let seg = ((fr * 6.0).round() as usize).clamp(3, 12);
        for (label, picked) in [("open (old doc, wrap-healed)", vec![open]), ("closed (new store)", vec![shut])] {
            let (beveled, fe) = bevel_mesh_and_edges(&cyl, fr, seg, &picked);
            // A rim on a tessellated curved wall must DECLINE the mesh surgery (its per-facet
            // strips crease into a striped/notched surface) and take the smooth CSG round instead.
            assert!(beveled.is_none(), "{label}: curved-wall rim must decline surgery");
            assert_eq!(fe.len(), 2 * n, "{label}: expected two complete {n}-segment seam rings, got {}", fe.len());
            let body = round_mesh(&cyl, fr, &picked).expect("CSG round");
            let tess = mesh_tessellation(body);
            // The CSG body must be SMOOTH: sharp edges ≈ the untouched bottom rim plus a few CSG
            // tessellation creases (~2n). The failure mode this guards is the creased/striped
            // surgery output, which shows up as ~20n sharp edges — an order of magnitude apart.
            assert!(
                tess.edges.len() <= n * 3,
                "{label}: body should be smooth (~{n} bottom-rim edges), got {} sharp edges",
                tess.edges.len()
            );
            let kept = clip_edges_to_mesh(&fe, &tess.mesh, 0.01);
            assert_eq!(kept.len(), fe.len(), "{label}: no seam segment should be clipped off-surface");
            // Every seam vertex has degree 2 (two closed rings — no dangling break, no junction).
            use std::collections::HashMap;
            let q = |p: [f32; 3]| ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64);
            let mut deg: HashMap<(i64, i64, i64), usize> = HashMap::new();
            for s in &kept {
                *deg.entry(q(s[0])).or_default() += 1;
                *deg.entry(q(s[1])).or_default() += 1;
            }
            assert!(deg.values().all(|&d| d == 2), "{label}: seam rings must be closed (all degree 2)");
            // And every vertex sits on one of the two ideal circles (no chevron poking out).
            let worst = kept
                .iter()
                .flat_map(|s| s.iter())
                .map(|p| {
                    let rxy = (p[0] as f64).hypot(p[1] as f64);
                    let cap = ((rxy - (rad - fr)).powi(2) + (p[2] as f64 - h).powi(2)).sqrt();
                    let wall = ((rxy - rad).powi(2) + (p[2] as f64 - (h - fr)).powi(2)).sqrt();
                    cap.min(wall)
                })
                .fold(0.0_f64, f64::max);
            assert!(worst < 0.05, "{label}: seam deviates {worst:.3} from the ideal rings");
        }
    }

    /// Replay of the logged "line down the edge shoots off / collapses": the projected reference
    /// edge is tilted by microns (tessellation), so PointOnLine + an exact Vertical share exactly
    /// ONE solution — the solver slid the endpoint there. The cleanup must drop the axis relation
    /// wherever an endpoint is line-pinned, after which the solve keeps the line's length.
    #[test]
    fn line_pinned_to_tilted_edge_survives_after_cleanup() {
        let mut s = Sketch::default();
        let _o = s.add_fixed_point(0.0, 0.0);
        let p1 = s.add_point(4.542434, -1.400048); // bottom corner (free, pinned to edge)
        let r2 = s.add_fixed_point(4.542158, 2.399952); // projected edge — tilted 276 µm over its run
        let r3 = s.add_fixed_point(4.542434, -1.400048);
        let p4 = s.add_fixed_point(4.542434, -1.399122); // old corner weld nearby
        let p5 = s.add_point(4.542434, -0.741); // upper endpoint of the drawn line
        s.add_reference_line(r2, r3);
        s.add_line(p1, p4, false);
        s.add_line(p1, p5, false); // the "line down the edge"
        s.constraints.push(Constraint::PointOnLine { p: p1, a: r2, b: r3 });
        s.constraints.push(Constraint::Vertical(p1, p4)); // exact axis vs tilted pin → conflict
        s.constraints.push(Constraint::Vertical(p1, p5));
        s.constraints.push(Constraint::PointOnLine { p: p5, a: r2, b: r3 });
        let removed = clean_redundant_relations(&mut s);
        assert!(removed >= 2, "cleanup should drop the axis relations on line-pinned endpoints (removed {removed})");
        s.solve();
        let (a, b) = (s.points[p1], s.points[p5]);
        let len = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
        assert!(a.x.is_finite() && b.y.is_finite(), "NaN after solve");
        assert!(len > 0.5, "line still collapsed/shot off: p1=({:.4},{:.4}) p5=({:.4},{:.4}) len={len:.4}", a.x, a.y, b.x, b.y);
    }

    /// Replay of the logged triangle collapse: the closing line's corner click sat nearest a
    /// REFERENCE-LINE endpoint (projected edge plumbing, fixed) rather than the projected corner
    /// point (also fixed, microns away). Welding onto the plumbing point then relating it to the
    /// corner created Coincident(fixed, fixed) — infeasible — and the solve dragged the triangle
    /// flat. The weld must skip reference-line-only points and pick the real corner.
    #[test]
    fn endpoint_weld_skips_reference_line_plumbing() {
        let mut s = SketchSession::default();
        s.snap_dist = 0.09;
        // The projected corner point (what earlier clicks created)…
        let corner = s.sketch.add_fixed_point(4.542434, -1.3936961);
        // …and a projected reference edge whose lower endpoint shadows it by ~6 microns-scale.
        let r4 = s.sketch.add_fixed_point(4.542158, 2.399952);
        let r5 = s.sketch.add_fixed_point(4.542434, -1.400048);
        s.sketch.add_reference_line(r4, r5);
        // A line already uses the corner (so it's "real" geometry, not plumbing).
        let p2 = s.sketch.add_point(5.42283, -0.5843668);
        s.sketch.add_line(corner, p2, false);
        // The user's closing click, nearer the plumbing endpoint than the corner:
        let w = get_or_add_point_ref(&mut s, Vec2::new(4.542, -1.4024), 0.09);
        assert_eq!(w, corner, "weld must pick the projected corner, not the reference-line endpoint (got point {w})");
        // And the relation helpers must never touch fixed points at all.
        let before = s.sketch.constraints.len();
        maybe_add_point_on_sketch_line(&mut s.sketch, r5, 0.05);
        maybe_add_point_on_circle(&mut s.sketch, r5, 0.05);
        assert_eq!(s.sketch.constraints.len(), before, "no relations may be attached to fixed points");
    }

    /// `clean_redundant_relations` heals the circincirc.hcad pattern: an endpoint double-pinned
    /// (parametric PointOnCircle + stale absolute PointOnArc) plus a cross-point Horizontal from
    /// the old guide capture. It keeps the circle pin and the line's OWN vertical, drops the rest.
    #[test]
    fn clean_relations_drops_arc_twin_and_cross_point_aligns() {
        let mut s = Sketch::default();
        let o = s.add_fixed_point(0.0, 0.0);
        let a = s.add_point(0.0, -3.9); // line start
        let b = s.add_point(0.0, -3.0); // endpoint snapped onto the circle rim
        let stray = s.add_fixed_point(2.0, -3.0); // unrelated point b once "aligned" with
        s.add_circle(o, 3.0);
        s.add_line(a, b, false);
        s.constraints.push(Constraint::Vertical(a, b)); // the line's own — keep
        s.constraints.push(Constraint::PointOnCircle { p: b, center: o }); // keep
        s.constraints.push(Constraint::PointOnArc { p: b, cx: 0.0, cy: 0.0, radius: 3.0 }); // stale twin — drop
        s.constraints.push(Constraint::Horizontal(b, stray)); // old guide capture — drop
        s.constraints.push(Constraint::Vertical(a, b)); // exact duplicate — drop
        let removed = clean_redundant_relations(&mut s);
        assert_eq!(removed, 3, "expected 3 removals, got {removed}: {:?}", s.constraints);
        assert!(s.constraints.iter().any(|c| matches!(c, Constraint::Vertical(x, y) if (*x == a && *y == b) || (*x == b && *y == a))));
        assert!(s.constraints.iter().any(|c| matches!(c, Constraint::PointOnCircle { p, .. } if *p == b)));
        assert!(!s.constraints.iter().any(|c| matches!(c, Constraint::PointOnArc { .. } | Constraint::Horizontal(..))));
    }

    /// Drawing a second circle concentric-ish with the first (a "circle inside a circle") must
    /// create a SECOND circle entity with its own centre point — not silently reuse the first
    /// circle's centre (which makes them share centre-keyed constraints and collapse to one).
    #[test]
    fn concentric_circle_gets_its_own_centre_point() {
        let mut s = SketchSession::default();
        s.snap_dist = SNAP;
        s.tool = Tool::Circle;
        // Circle 1: centre near origin, radius ~8.5 (two clicks).
        place_point(&mut s, Vec2::new(0.0, 0.0));
        place_point(&mut s, Vec2::new(8.5, 0.0));
        // Circle 2: centre a hair off the same origin (within snap → would weld), radius ~3.
        place_point(&mut s, Vec2::new(0.02, 0.0));
        place_point(&mut s, Vec2::new(3.0, 0.0));
        let circles: Vec<(usize, f64)> = s
            .sketch
            .entities
            .iter()
            .filter_map(|e| match e {
                SketchEntity::Circle { center, radius, .. } => Some((*center, *radius)),
                _ => None,
            })
            .collect();
        assert_eq!(circles.len(), 2, "both circles must exist, got {circles:?}");
        assert_ne!(circles[0].0, circles[1].0, "the two circles must have DISTINCT centre points (not welded)");
        assert!((circles[0].1 - 8.5).abs() < 0.2 && (circles[1].1 - 3.0).abs() < 0.2, "radii preserved: {circles:?}");
    }

    /// A slot cut through a filleted rim must CLIP the fillet's seam rings at the cut: no seam
    /// segment may float across the void (the "edge lines extend past the cut" bug).
    #[test]
    fn cut_through_fillet_clips_the_seam_rings() {
        let basis = PlaneBasis { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let (rad, h, fr) = (6.0_f64, 9.0_f64, 1.0_f64);
        let n = 128;
        let circ: Vec<[f64; 2]> = (0..n).map(|i| { let a = std::f64::consts::TAU * i as f64 / n as f64; [rad * a.cos(), rad * a.sin()] }).collect();
        let cyl = extrude_tool_mesh(&circ, &[], &basis, 0.0, h).expect("cylinder");
        let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], h]).collect();
        rim.push(rim[0]);
        let picked = vec![rim];
        // Fillet the rim exactly like regen: surgery (declines on curved walls) → CSG round.
        let (beveled, fe) = bevel_mesh_and_edges(&cyl, fr, ((fr * 6.0) as usize).clamp(3, 12), &picked);
        let body = beveled.unwrap_or_else(|| round_mesh(&cyl, fr, &picked).expect("csg round"));
        // Slot cut straight through the top, like the user's Cut-Extrude2: a 4-wide rectangle
        // across the whole diameter, 5.2 deep from the top.
        let slot = extrude_tool_mesh(
            &[[-rad - 1.0, -2.0], [rad + 1.0, -2.0], [rad + 1.0, 2.0], [-rad - 1.0, 2.0]],
            &[],
            &basis,
            h - 5.2,
            h + 1.0 - (h - 5.2),
        )
        .expect("slot tool");
        let cut = mesh_difference(&body, &slot);
        let tess = mesh_tessellation(cut);
        let kept = clip_edges_to_mesh(&fe, &tess.mesh, 0.01);
        // No kept segment's midpoint may sit inside the slot mouth (|y| < 2 − margin, above the
        // slot floor): those spans float across the void.
        let floaters = kept
            .iter()
            .filter(|s| {
                let my = (s[0][1] + s[1][1]) * 0.5;
                let mz = (s[0][2] + s[1][2]) * 0.5;
                (my.abs() as f64) < 1.7 && (mz as f64) > h - 5.2 + 0.3
            })
            .count();
        assert_eq!(floaters, 0, "{floaters} seam segment(s) float across the slot void (of {} kept)", kept.len());
        // Sanity: plenty of the ring survives outside the cut.
        assert!(kept.len() > n, "most of the seam rings should survive, got {}", kept.len());
    }

    /// What does the LAST feature actually add? Regen with and without it and diff the bboxes.
    ///   HCAD_FILE="…\part.hcad" cargo test -p hworks-app --release diag_last_feature_delta -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diag_last_feature_delta() {
        let path = std::env::var("HCAD_FILE").expect("set HCAD_FILE");
        let doc: Document = ron::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (full, _) = regenerate_mesh(&doc).expect("full regen");
        let mut prev_doc = doc.clone();
        prev_doc.rollback = prev_doc.features.len().saturating_sub(1);
        let (prev, _) = regenerate_mesh(&prev_doc).expect("prev regen");
        let bbox = |m: &TriMesh| {
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for p in &m.positions {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        };
        let (plo, phi) = bbox(&prev);
        let (flo, fhi) = bbox(&full);
        eprintln!("prev bbox: lo=({:.3},{:.3},{:.3}) hi=({:.3},{:.3},{:.3})", plo[0], plo[1], plo[2], phi[0], phi[1], phi[2]);
        eprintln!("full bbox: lo=({:.3},{:.3},{:.3}) hi=({:.3},{:.3},{:.3})", flo[0], flo[1], flo[2], fhi[0], fhi[1], fhi[2]);
        // The material the last feature added, isolated: full − prev.
        let added = mesh_difference(&full, &prev);
        if !added.positions.is_empty() {
            let (alo, ahi) = bbox(&added);
            eprintln!(
                "added material bbox: lo=({:.3},{:.3},{:.3}) hi=({:.3},{:.3},{:.3}) spans=({:.3},{:.3},{:.3})",
                alo[0], alo[1], alo[2], ahi[0], ahi[1], ahi[2],
                ahi[0] - alo[0], ahi[1] - alo[1], ahi[2] - alo[2]
            );
        } else {
            eprintln!("added material: none (difference empty)");
        }
        if let Some(f) = doc.features.last() {
            eprintln!("last feature: {:?}", match &f.kind {
                FeatureKind::Extrude { distance, back, plane, .. } => format!("Extrude d={distance} back={back} origin={:?} normal={:?}", plane.origin, plane.normal),
                other => format!("{other:?}").chars().take(80).collect::<String>(),
            });
        }
    }

    /// Reproduce the filleted-cylinder seam artifacts (break + chevron poking out of the rim):
    ///   cargo test -p hworks-app --release diag_cylinder_fillet_seams -- --ignored --nocapture
    /// Mirrors the app's regen exactly: 48-gon extrude → bevel (surgery or CSG fallback) →
    /// mesh_tessellation → clip_edges_to_mesh, then audits the surviving seam chains.
    #[test]
    #[ignore]
    fn diag_cylinder_fillet_seams() {
        // With HCAD_FILE set: replay the REAL document through regenerate_mesh and audit its body +
        // seams (watertightness, edge counts, seam degrees) instead of the synthetic cylinder.
        if let Ok(path) = std::env::var("HCAD_FILE") {
            let doc: Document = ron::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let (mesh, bevel_edges) = regenerate_mesh(&doc).expect("regen");
            eprintln!("body: {} verts / {} tris, bevel_edges={}", mesh.positions.len(), mesh.indices.len() / 3, bevel_edges.len());
            let tess = mesh_tessellation(mesh);
            eprintln!("tess: sharp={} tangent={}", tess.edges.len(), tess.tangent_edges.len());
            let kept = clip_edges_to_mesh(&bevel_edges, &tess.mesh, 0.01);
            use std::collections::HashMap;
            let q = |p: [f32; 3]| ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64);
            let mut deg: HashMap<(i64, i64, i64), usize> = HashMap::new();
            for s in &kept {
                *deg.entry(q(s[0])).or_default() += 1;
                *deg.entry(q(s[1])).or_default() += 1;
            }
            let dangles = deg.values().filter(|&&d| d == 1).count();
            let junctions = deg.values().filter(|&&d| d > 2).count();
            eprintln!("seams: kept={} dropped={} dangling={dangles} junctions={junctions}", kept.len(), bevel_edges.len() - kept.len());
            // Component analysis of the SHARP set: sizes and where the small ones sit.
            {
                use std::collections::HashMap as HM;
                let q2 = |p: [f32; 3]| ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64);
                let mut ids: HM<(i64, i64, i64), usize> = HM::new();
                let segs: Vec<(usize, usize, f32)> = tess.edges.iter().map(|e| {
                    let n = ids.len();
                    let a = *ids.entry(q2(e[0])).or_insert(n);
                    let n = ids.len();
                    let b = *ids.entry(q2(e[1])).or_insert(n);
                    let d = ((e[0][0]-e[1][0]).powi(2)+(e[0][1]-e[1][1]).powi(2)+(e[0][2]-e[1][2]).powi(2)).sqrt();
                    (a, b, d)
                }).collect();
                let mut uf: Vec<usize> = (0..ids.len()).collect();
                fn f(uf: &mut [usize], mut x: usize) -> usize { while uf[x] != x { uf[x] = uf[uf[x]]; x = uf[x]; } x }
                for &(a, b, _) in &segs { let (ra, rb) = (f(&mut uf, a), f(&mut uf, b)); if ra != rb { uf[ra] = rb; } }
                let mut comp: HM<usize, (usize, f32, [f32;3])> = HM::new();
                for (si, &(a, _, d)) in segs.iter().enumerate() {
                    let root = f(&mut uf, a);
                    let e = comp.entry(root).or_insert((0, 0.0, tess.edges[si][0]));
                    e.0 += 1; e.1 += d;
                }
                let mut list: Vec<(usize, f32, [f32;3])> = comp.values().copied().collect();
                list.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
                eprintln!("sharp components: {} total", list.len());
                for (n, len, at) in list.iter().take(12) {
                    eprintln!("  comp: segs={n} len={len:.3} at=({:.2},{:.2},{:.2})", at[0], at[1], at[2]);
                }
            }
            // True floaters among the kept: midpoint far (>3× clip tol) from EVERY mesh triangle.
            let m = &tess.mesh;
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for p in &m.positions {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
            let tol = (diag * 0.01).max(1.0e-4);
            let dist_to_mesh = |p: [f32; 3]| -> f32 {
                let mut best = f32::MAX;
                for t in m.indices.chunks_exact(3) {
                    let d = point_tri_dist(p, m.positions[t[0] as usize], m.positions[t[1] as usize], m.positions[t[2] as usize]);
                    best = best.min(d);
                }
                best
            };
            let mut floaters = 0;
            for s in &kept {
                let mid = [(s[0][0] + s[1][0]) * 0.5, (s[0][1] + s[1][1]) * 0.5, (s[0][2] + s[1][2]) * 0.5];
                let d = dist_to_mesh(mid);
                if d > tol * 3.0 {
                    floaters += 1;
                    if floaters <= 8 {
                        eprintln!("  floater: mid=({:.2},{:.2},{:.2}) dist={d:.3} (tol {tol:.3})", mid[0], mid[1], mid[2]);
                    }
                }
            }
            eprintln!("floaters among kept: {floaters}");
            return;
        }
        let basis = PlaneBasis { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] };
        let (rad, h, fr) = (
            std::env::var("DIAG_R").ok().and_then(|v| v.parse().ok()).unwrap_or(25.0_f64),
            std::env::var("DIAG_H").ok().and_then(|v| v.parse().ok()).unwrap_or(5.7_f64),
            std::env::var("DIAG_FR").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0_f64),
        );
        let n: usize = std::env::var("DIAG_N").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
        let circ: Vec<[f64; 2]> = (0..n).map(|i| { let a = std::f64::consts::TAU * i as f64 / n as f64; [rad * a.cos(), rad * a.sin()] }).collect();
        let cyl = extrude_tool_mesh(&circ, &[], &basis, 0.0, h).expect("cylinder");
        // Pick the rim the way the APP does: from the tessellated body's sharp edges, chained by
        // edge_loop — not the synthetic profile polygon. The tessellation roundtrips the mesh
        // through Manifold (rewelds/reorders), so this chain is what a real click yields.
        let tess0 = mesh_tessellation(cyl.clone());
        let rim_seed = tess0
            .edges
            .iter()
            .position(|e| (e[0][2] - h as f32).abs() < 1e-3 && (e[1][2] - h as f32).abs() < 1e-3)
            .expect("find a top-rim segment");
        let (chain, closed) = edge_loop(&tess0.edges, rim_seed);
        eprintln!("picked rim: {} pts, closed={closed}", chain.len());
        let picked: Vec<Vec<[f64; 3]>> = vec![chain.iter().map(|p| [p.x as f64, p.y as f64, p.z as f64]).collect()];
        let seg = ((fr * 6.0).round() as usize).clamp(3, 12);
        for round in 0..10 {
            let (beveled, fe) = bevel_mesh_and_edges(&cyl, fr, seg, &picked);
            let path = if round >= 5 || beveled.is_none() { "CSG-fallback" } else { "surgery" };
            // Rounds 5+ force the CSG fallback to audit the seams against ITS surface too.
            let body = if round >= 5 {
                round_mesh(&cyl, fr, &picked).expect("csg round")
            } else {
                beveled.unwrap_or_else(|| round_mesh(&cyl, fr, &picked).expect("csg round"))
            };
            let tess = mesh_tessellation(body);
            let kept = clip_edges_to_mesh(&fe, &tess.mesh, 0.01);
            // Audit the surviving seam network: vertex degrees (a clean pair of rings = all degree
            // 2), and each vertex's deviation from the two ideal circles (cap ring |xy| = R − r at
            // z = h; wall ring |xy| = R at z = h − r).
            use std::collections::HashMap;
            let q = |p: [f32; 3]| ((p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64);
            let mut deg: HashMap<(i64, i64, i64), usize> = HashMap::new();
            for s in &kept {
                *deg.entry(q(s[0])).or_default() += 1;
                *deg.entry(q(s[1])).or_default() += 1;
            }
            let dangles = deg.values().filter(|&&d| d == 1).count();
            let junctions = deg.values().filter(|&&d| d > 2).count();
            let worst = kept
                .iter()
                .flat_map(|s| s.iter())
                .map(|p| {
                    let rxy = (p[0] as f64).hypot(p[1] as f64);
                    let cap = ((rxy - (rad - fr)).powi(2) + (p[2] as f64 - h).powi(2)).sqrt();
                    let wall = ((rxy - rad).powi(2) + (p[2] as f64 - (h - fr)).powi(2)).sqrt();
                    cap.min(wall)
                })
                .fold(0.0_f64, f64::max);
            eprintln!(
                "round {round}: path={path} emitted={} kept={} dropped={} dangling-ends={dangles} junctions={junctions} worst-deviation={worst:.3} body-sharp={} body-tris={}",
                fe.len(),
                kept.len(),
                fe.len() - kept.len(),
                tess.edges.len(),
                tess.mesh.indices.len() / 3,
            );
        }
    }

    /// report watertightness + edge topology. Ignored by default — run with the path:
    ///   cargo test -p hworks-app check_hcad_manifold -- --ignored --nocapture
    #[test]
    #[ignore]
    fn parse_saved_files() {
        // Parse-only check of every .hcad under the saved-files dir — a schema regression here is
        // exactly what makes "Open" silently fail (doc unchanged, no file bound → Save shows Save As).
        //   HCAD_DIR="…\saved files" cargo test -p hworks-app parse_saved_files -- --ignored --nocapture
        let dir = std::env::var("HCAD_DIR").expect("set HCAD_DIR");
        let mut fails = 0;
        for entry in std::fs::read_dir(&dir).expect("read dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("hcad") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            match ron::from_str::<Document>(&text) {
                Ok(_) => eprintln!("ok:   {}", path.file_name().unwrap().to_string_lossy()),
                Err(e) => {
                    fails += 1;
                    eprintln!("FAIL: {} — {e}", path.file_name().unwrap().to_string_lossy());
                }
            }
        }
        assert_eq!(fails, 0, "{fails} saved file(s) failed to parse with the current schema");
    }

    #[test]
    #[ignore]
    fn check_hcad_manifold() {
        // Path from $HCAD_FILE, else the default sample. e.g.:
        //   HCAD_FILE="C:\path\to\part.hcad" cargo test -p hworks-app check_hcad_manifold -- --ignored --nocapture
        let path = std::env::var("HCAD_FILE").unwrap_or_else(|_| r"C:\Users\BIG2\Desktop\DEV for BIG\HCAD\saved files\lofttest.hcad".to_string());
        eprintln!("checking: {path}");
        let text = std::fs::read_to_string(&path).expect("read .hcad");
        let doc: Document = ron::from_str(&text).expect("parse RON");
        let mesh = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc))).ok().flatten().map(|(m, _)| m);
        eprintln!("BSP fallbacks during regen: {}", hworks_geometry::take_fallback_count());
        let Some(m) = mesh.filter(|m| !m.positions.is_empty()) else {
            eprintln!("RESULT: no mesh body produced (regen failed)");
            return;
        };
        // Weld by position, then count how many times each undirected edge is used.
        use std::collections::HashMap;
        let key = |p: [f32; 3]| ((p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64, (p[2] * 1e3).round() as i64);
        let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut remap = vec![0u32; m.positions.len()];
        for (i, p) in m.positions.iter().enumerate() {
            let n = map.len() as u32;
            remap[i] = *map.entry(key(*p)).or_insert(n);
        }
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        for t in m.indices.chunks_exact(3) {
            let (a, b, c) = (remap[t[0] as usize], remap[t[1] as usize], remap[t[2] as usize]);
            for (x, y) in [(a, b), (b, c), (c, a)] {
                let k = if x < y { (x, y) } else { (y, x) };
                *edges.entry(k).or_insert(0) += 1;
            }
        }
        let boundary = edges.values().filter(|&&v| v == 1).count();
        let nonman = edges.values().filter(|&&v| v > 2).count();
        let watertight = boundary == 0 && nonman == 0;
        eprintln!(
            "RESULT: manifold(Manifold-ingestible)={}  watertight={}  verts={}  tris={}  boundary_edges={}  nonmanifold_edges={}",
            hworks_geometry::is_manifold(&m),
            watertight,
            map.len(),
            m.indices.len() / 3,
            boundary,
            nonman,
        );
    }

    /// Detailed feature-edge diagnostics for a real model — characterises the strays so the edge
    /// detector can be tuned. Run:
    ///   HCAD_FILE="C:\path\to\part.hcad" cargo test -p hworks-app analyze_hcad_edges -- --ignored --nocapture
    #[test]
    #[ignore]
    fn analyze_hcad_edges() {
        use std::collections::HashMap;
        let path = std::env::var("HCAD_FILE").unwrap_or_else(|_| r"C:\Users\BIG2\Desktop\DEV for BIG\HCAD\saved files\lofttest.hcad".to_string());
        eprintln!("analyzing: {path}");
        let text = std::fs::read_to_string(&path).expect("read .hcad");
        let doc: Document = ron::from_str(&text).expect("parse RON");
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| regenerate_mesh(&doc)))
            .ok().flatten().filter(|(m, _)| !m.positions.is_empty());
        let Some((m, bevel_edges)) = res else { eprintln!("no mesh"); return };

        // Bbox-relative weld (mirror feature_edges: 2e-4 of the diagonal).
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &m.positions { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
        let diag = ((hi[0]-lo[0]).powi(2)+(hi[1]-lo[1]).powi(2)+(hi[2]-lo[2]).powi(2)).sqrt();
        let cell = (diag * 2.0e-4).max(1e-6);
        let scale = 1.0 / cell;
        let q = |p: [f32; 3]| ((p[0]*scale).round() as i64, (p[1]*scale).round() as i64, (p[2]*scale).round() as i64);
        let mut canon: HashMap<(i64,i64,i64), usize> = HashMap::new();
        let mut cpos: Vec<[f32;3]> = Vec::new();
        let mut vid = vec![0usize; m.positions.len()];
        for (i, p) in m.positions.iter().enumerate() {
            vid[i] = *canon.entry(q(*p)).or_insert_with(|| { cpos.push(*p); cpos.len()-1 });
        }
        // edge -> incident face normals
        let mut emap: HashMap<(usize,usize), Vec<[f32;3]>> = HashMap::new();
        for t in m.indices.chunks_exact(3) {
            let n = m.normals[t[0] as usize];
            let (a,b,c) = (vid[t[0] as usize], vid[t[1] as usize], vid[t[2] as usize]);
            for (i,j) in [(a,b),(b,c),(c,a)] { let k = if i<j {(i,j)} else {(j,i)}; emap.entry(k).or_default().push(n); }
        }
        // dihedral histogram (manifold edges only) + counts at candidate thresholds
        let mut buckets = [0u32; 19]; // 0-10,10-20,...,180
        let (mut boundary, mut nonman) = (0u32, 0u32);
        let mut deg_at = |deg_deg: f64| -> usize {
            emap.iter().filter(|(_, ns)| ns.len()==2).filter(|(_, ns)| {
                let d = (ns[0][0]*ns[1][0]+ns[0][1]*ns[1][1]+ns[0][2]*ns[1][2]).clamp(-1.0,1.0);
                (d as f64).acos().to_degrees() >= deg_deg
            }).count()
        };
        for (_, ns) in &emap {
            match ns.len() {
                1 => boundary += 1,
                2 => {
                    let d = (ns[0][0]*ns[1][0]+ns[0][1]*ns[1][1]+ns[0][2]*ns[1][2]).clamp(-1.0,1.0);
                    let ang = (d as f64).acos().to_degrees();
                    buckets[((ang/10.0) as usize).min(18)] += 1;
                }
                _ => nonman += 1,
            }
        }
        eprintln!("diag={diag:.2} verts={} tris={} boundary={boundary} nonmanifold={nonman}", cpos.len(), m.indices.len()/3);
        eprintln!("dihedral histogram (deg : count):");
        for (i, &c) in buckets.iter().enumerate() { if c>0 { eprintln!("  {:>3}-{:<3}: {c}", i*10, i*10+10); } }
        for th in [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0] {
            eprintln!("  sharp edges >= {th:>4}°: {}", deg_at(th));
        }
        // Degree distribution of the 30° sharp set (mirrors feature_edges: count==2 by dihedral,
        // plus non-manifold count>=3 kept if any incident pair is a real corner).
        let max_dihedral = |ns: &Vec<[f32;3]>| -> f64 {
            let mut md = 1.0f32;
            for a in 0..ns.len() { for b in (a+1)..ns.len() {
                md = md.min((ns[a][0]*ns[b][0]+ns[a][1]*ns[b][1]+ns[a][2]*ns[b][2]).clamp(-1.0,1.0));
            }}
            (md as f64).acos().to_degrees()
        };
        let sharp: Vec<(usize,usize)> = emap.iter().filter(|(_, ns)| {
            (ns.len()==2 || ns.len()>=3) && max_dihedral(ns) >= 30.0
        }).map(|(&k,_)| k).collect();
        let mut vdeg: HashMap<usize,u32> = HashMap::new();
        for &(a,b) in &sharp { *vdeg.entry(a).or_default()+=1; *vdeg.entry(b).or_default()+=1; }
        let (mut d1, mut d2, mut d3) = (0u32,0u32,0u32);
        for (_, &d) in &vdeg { match d { 1 => d1+=1, 2 => d2+=1, _ => d3+=1 } }
        let embedded = sharp.iter().filter(|&&(a,b)| vdeg[&a]>=2 && vdeg[&b]>=2).count();
        eprintln!("30° set: {} edges | vertices deg1={d1} deg2={d2} deg3+={d3} | embedded(both ends deg>=2)={embedded}", sharp.len());

        // Bevel (fillet/chamfer) edges: how many sit OFF the final surface (left floating in a void
        // by a later cut)? Test each endpoint's distance to the nearest welded mesh vertex.
        let cells = (scale).max(1e-6);
        let mut grid: HashMap<(i64,i64,i64), Vec<usize>> = HashMap::new();
        let gq = |p:[f32;3]| ((p[0]*cells/8.0).floor() as i64,(p[1]*cells/8.0).floor() as i64,(p[2]*cells/8.0).floor() as i64);
        for (i,p) in cpos.iter().enumerate() { grid.entry(gq(*p)).or_default().push(i); }
        let near_vert = |p:[f32;3]| -> f32 {
            let (gx,gy,gz) = gq(p);
            let mut best = f32::MAX;
            for dx in -1..=1 { for dy in -1..=1 { for dz in -1..=1 {
                if let Some(v) = grid.get(&(gx+dx,gy+dy,gz+dz)) {
                    for &i in v { let q=cpos[i]; let d=((p[0]-q[0]).powi(2)+(p[1]-q[1]).powi(2)+(p[2]-q[2]).powi(2)).sqrt(); best=best.min(d); }
                }
            }}}
            best
        };
        let tol = diag * 0.01;
        let mid = |e: &[[f32;3];2]| [(e[0][0]+e[1][0])*0.5,(e[0][1]+e[1][1])*0.5,(e[0][2]+e[1][2])*0.5];
        let (mut off_either, mut off_mid, mut off_all3) = (0,0,0);
        let mut lens: Vec<f32> = Vec::new();
        for (e, _n) in &bevel_edges {
            let (da, db, dm) = (near_vert(e[0]), near_vert(e[1]), near_vert(mid(e)));
            if da > tol || db > tol { off_either += 1; }
            if dm > tol { off_mid += 1; }
            if da > tol && db > tol && dm > tol { off_all3 += 1; }
            let d=[e[1][0]-e[0][0],e[1][1]-e[0][1],e[1][2]-e[0][2]];
            lens.push((d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt());
        }
        lens.sort_by(|a,b| a.partial_cmp(b).unwrap());
        eprintln!("bevel_edges: {} total | tol={:.2}mm | off(either endpoint)={off_either} off(midpoint-to-vertex)={off_mid} off(all3)={off_all3}", bevel_edges.len(), tol);
        eprintln!("  bevel seg length: min={:.3} median={:.3} max={:.3}", lens.first().copied().unwrap_or(0.0), lens.get(lens.len()/2).copied().unwrap_or(0.0), lens.last().copied().unwrap_or(0.0));
        // Length histogram of bevel edges whose midpoint is on-surface (before any length cap) — is
        // there a clean gap between the legit cluster and the void-spanner outliers?
        {
            // reuse the real clip's surface test by calling it without the cap is not exposed; approximate
            // with midpoint-to-vertex unavailable for surface — instead show all bevel-edge lengths bucketed.
            let mut hist = [0u32; 12];
            for &l in &lens { hist[((l/0.25) as usize).min(11)] += 1; }
            eprint!("  bevel length hist (0.25mm bins): ");
            for (i,&c) in hist.iter().enumerate() { if c>0 { eprint!("[{:.2}:{}] ", i as f32*0.25, c); } }
            eprintln!();
        }
        let kept = clip_edges_to_mesh(&bevel_edges, &m, 0.01);
        let mut klens: Vec<f32> = kept.iter().map(|e| { let d=[e[1][0]-e[0][0],e[1][1]-e[0][1],e[1][2]-e[0][2]]; (d[0]*d[0]+d[1]*d[1]+d[2]*d[2]).sqrt() }).collect();
        klens.sort_by(|a,b| a.partial_cmp(b).unwrap());
        let kmed = klens.get(klens.len()/2).copied().unwrap_or(0.0);
        let over = |t: f32| klens.iter().filter(|&&l| l > t).count();
        eprintln!("  midpoint-to-SURFACE clip: kept {}/{} | dropped {}", kept.len(), bevel_edges.len(), bevel_edges.len()-kept.len());
        eprintln!("  kept seg lengths: median={kmed:.3} | >1.5mm={} >2mm={} >2.5mm={} >3mm={} | >3*med({:.2})={}", over(1.5), over(2.0), over(2.5), over(3.0), kmed*3.0, over(kmed*3.0));

        // FACE-BASED detector (FreeCAD-style) vs the angle detector: report edge count + degree
        // distribution (deg1 dangles = gaps/strays). Clean = deg1==0, deg2 dominant.
        if let Some((fsharp, ftan)) = hworks_geometry::feature_edges_by_face(&m, 20.0, 8.0) {
            let mut fdeg: HashMap<(i64, i64, i64), u32> = HashMap::new();
            for e in &fsharp {
                let kf = |p: [f32; 3]| ((p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64, (p[2] * 1e3).round() as i64);
                *fdeg.entry(kf(e[0])).or_default() += 1;
                *fdeg.entry(kf(e[1])).or_default() += 1;
            }
            let (mut f1, mut f2, mut f3) = (0, 0, 0);
            for (_, &d) in &fdeg { match d { 1 => f1 += 1, 2 => f2 += 1, _ => f3 += 1 } }
            eprintln!("FACE-DETECTOR: sharp={} tangent={} | vertices deg1={f1} deg2={f2} deg3+={f3}", fsharp.len(), ftan.len());
        } else {
            eprintln!("FACE-DETECTOR: (mesh could not be ingested)");
        }
    }

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
        PlaneRef { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0], datum: false }
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
    fn refine_circle_disambiguates_concentric_features() {
        // A boss (r=25) with a concentric bore (r=15): two exact circles at the SAME
        // centre. A slightly-off fit of the bore must refine to the bore's radius,
        // never the boss's — centre-only matching got this wrong.
        let exact = vec![(Vec2::ZERO, 25.0_f32), (Vec2::ZERO, 15.0_f32)];
        let (c, r) = refine_circle(&exact, Vec2::new(0.02, -0.01), 15.03);
        assert_eq!(r, 15.0, "bore fit must take the bore radius");
        assert_eq!(c, Vec2::ZERO);
        let (_, r) = refine_circle(&exact, Vec2::new(0.01, 0.0), 24.98);
        assert_eq!(r, 25.0, "boss fit must take the boss radius");
        // A fit that matches nothing stays untouched.
        let (c, r) = refine_circle(&exact, Vec2::new(9.0, 0.0), 5.0);
        assert_eq!((c, r), (Vec2::new(9.0, 0.0), 5.0));
    }

    #[test]
    fn exact_plane_circles_reads_true_radii_from_the_timeline() {
        // A cylinder (r=50) extruded 50 up from the Top plane. On a sketch plane at
        // its top cap, the exact reference circle must be (0,0) with radius exactly 50.
        let mut doc = Document::with_default_planes();
        let mut s = Sketch::default();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 50.0);
        let top = PlaneRef {
            origin: [0.0, 0.0, 0.0],
            u: [1.0, 0.0, 0.0],
            v: [0.0, 0.0, -1.0],
            normal: [0.0, 1.0, 0.0],
            datum: true,
        };
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![], plane: top, distance: 50.0, back: 0.0, thin: 0.0, thin_side: 0 });
        let cap = ActivePlane {
            name: "Face".into(),
            origin: Vec3::new(0.0, 50.0, 0.0),
            u: Vec3::X,
            v: Vec3::NEG_Z,
            n: Vec3::Y,
            datum: false,
        };
        let circles = exact_plane_circles(&doc, &cap);
        assert_eq!(circles.len(), 1, "one exact circle on the cap plane: {circles:?}");
        assert!(circles[0].0.length() < 1e-4, "centred on the axis");
        assert_eq!(circles[0].1, 50.0, "radius is the exact sketch value");
        // A fitted circle with tessellation error snaps to it exactly.
        let (rc, rr) = refine_circle(&circles, Vec2::new(0.03, -0.02), 49.97);
        assert_eq!((rc, rr), (Vec2::ZERO, 50.0));
    }

    #[test]
    fn pick_face_centroid_is_unbiased_by_tessellation() {
        // A radius-50 disk at y=20 in the XZ plane, tessellated as a fan from ONE
        // rim vertex — a deliberately lopsided triangulation (tiny slivers by the
        // apex, big triangles opposite). The face origin must still land on the
        // true centre (0,20,0): an unweighted mean of triangle centroids drifts
        // toward the apex (the stacked-cylinder bug), the area-weighted one doesn't.
        const R: f32 = 50.0;
        const Y: f32 = 20.0;
        const N: usize = 64;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let apex = [R, Y, 0.0]; // rim vertex at angle 0
        positions.push(apex);
        for k in 0..=N {
            let a = std::f32::consts::TAU * k as f32 / N as f32;
            positions.push([R * a.cos(), Y, R * a.sin()]);
        }
        // Fan triangles wound so the normal points +Y (up).
        for k in 1..N {
            indices.extend([0u32, (k + 1) as u32, k as u32]);
        }
        let mesh = TriMesh { positions, normals: vec![[0.0, 1.0, 0.0]; N + 2], indices };
        // Ray straight down at the centre from above.
        let ray = Ray3d { origin: Vec3::new(0.0, 100.0, 0.0), direction: Dir3::NEG_Y };
        let (_, ap) = pick_face(&mesh, &ray).expect("face picked");
        assert!(ap.origin.distance(Vec3::new(0.0, Y, 0.0)) < 0.05, "face origin drifted to {:?}", ap.origin);
    }

    #[test]
    fn regenerate_replays_an_extrude_into_a_box() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        let a = Region { outer: rect(0.0, 0.0, 2.0, 2.0), holes: vec![], ..Default::default() };
        let b = Region { outer: rect(5.0, 0.0, 7.0, 2.0), holes: vec![], ..Default::default() };
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
        doc.add_feature(FeatureKind::Extrude { sketch: a, regions: vec![], plane: xy(), distance: 5.0, back: 0.0, thin: 0.0, thin_side: 0 });
        let mut b = Sketch::default();
        let pb = b.add_point(10.0, 0.0);
        b.add_circle(pb, 3.0);
        doc.add_feature(FeatureKind::Extrude { sketch: b, regions: vec![], plane: xy(), distance: 3.0, back: 0.0, thin: 0.0, thin_side: 0 });
        let before = tessellate(&regenerate(&doc).unwrap(), 0.05).edges.len();
        // Cut a 1mm hole in the top of cylinder A (z=5), 2mm deep.
        let mut cut = Sketch::default();
        let pc = cut.add_point(0.0, 0.0);
        cut.add_circle(pc, 1.0);
        let top = PlaneRef { origin: [0.0, 0.0, 5.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0], datum: false };
        doc.add_feature(FeatureKind::Cut { sketch: cut, regions: vec![], plane: top, distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        doc.add_feature(FeatureKind::Extrude { sketch: boss, regions: vec![], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
        // Cut a 1mm-radius hole in the middle, from the top face.
        let mut cutsk = Sketch::default();
        let cc = cutsk.add_point(5.0, 0.0);
        cutsk.add_circle(cc, 1.0);
        let top = PlaneRef { origin: [0.0, 0.0, 2.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0], datum: false };
        doc.add_feature(FeatureKind::Cut { sketch: cutsk, regions: vec![], plane: top, distance: 1.0, back: 0.0, thin: 0.0, thin_side: 0 });
        let solid = regenerate(&doc); // must not panic
        assert!(solid.is_some(), "dumbbell with a cut should still produce a body");
    }

    #[test]
    fn editing_a_distance_rebuilds_taller() {
        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
        // Boss 2×2 sketched on the top face (z=2), 2 tall → stacked total height 4.
        let top = PlaneRef { origin: [0.0, 0.0, 2.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0], datum: false };
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(2.0, 2.0), regions: vec![0], plane: top, distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
            back: 0.0,
            thin: 0.0,
            thin_side: 0,
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
        doc.add_feature(FeatureKind::Extrude { sketch: rect_sketch(4.0, 4.0), regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        doc.add_feature(FeatureKind::Cut { sketch: pocket, regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![0], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
        let edges = tessellate(&regenerate(&doc).unwrap(), 0.05).edges.len();
        assert_eq!(edges, 12, "one selected contour → one box, got {edges}");
    }

    #[test]
    fn extrude_all_contours_builds_both_boxes() {
        let s = two_disjoint_squares();
        let mut doc = Document::with_default_planes();
        // Empty selection ⇒ all contours.
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![], plane: xy(), distance: 2.0, back: 0.0, thin: 0.0, thin_side: 0 });
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
        let region = Region { outer: circle, holes: vec![], ..Default::default() };
        // The exact union fails in truck; boss_union must recover via the nudge.
        let solid = boss_union(&wedge, &region, &top, 2.0, 0.0).expect("coincident cylinder boss should union");
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
    fn edge_loop_closes_a_sharp_square_but_chains_an_open_run() {
        let square = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        let (chain, closed) = edge_loop(&square, 0);
        assert!(closed, "edge_loop snaps the whole planar square perimeter");
        assert_eq!(chain.len(), 4, "all four sides");

        // A lone edge with no closing loop stays a single open segment.
        let lone = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]];
        let (c2, closed2) = edge_loop(&lone, 0);
        assert!(!closed2);
        assert_eq!(c2.len(), 2);

        // A non-planar 4-cycle must NOT be snapped (it's not a flat rim).
        let skew = vec![
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ];
        let (_, closed3) = edge_loop(&skew, 0);
        assert!(!closed3, "non-planar cycles are not loop-snapped");
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

    #[test]
    fn top_rim_fillet_through_the_real_regen_pipeline_stays_in_bounds() {
        // saved files/fillererror2.hcad, replayed through the actual document/regen path (not a
        // hand-built mesh) — a box extruded from a datum-plane sketch, then its top-face rim
        // filleted. Every vertex of the regenerated body must stay within the box's own
        // pre-fillet bounding box: a fillet only removes material, it can't add any.
        let (x0, x1) = (0.0_f64, 40.284645080566406);
        let (z0, z1) = (0.0_f64, 38.863014221191406);
        let dist = 26.026290893554688_f64;
        // Matches the saved file's plane exactly: u=+X, v=-Z, normal=+Y (extrudes up the world
        // Y axis, world z = -py) — NOT the xy() helper (which extrudes along Z).
        let top = PlaneRef { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0], datum: true };
        let mut s = Sketch::default();
        let p0 = s.add_point(x0, -z0);
        let p1 = s.add_point(x1, -z0);
        let p2 = s.add_point(x1, -z1);
        let p3 = s.add_point(x0, -z1);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        s.add_line(p3, p0, false);

        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![], plane: top, distance: dist, back: 0.0, thin: 0.0, thin_side: 0 });

        let radius = 7.78000020980835;
        let top_rim = vec![vec![
            [x0, dist, z1], [x1, dist, z1], [x1, dist, z0], [x0, dist, z0],
        ]];
        doc.add_feature(FeatureKind::Fillet { radius, edges: top_rim });

        // regenerate() is the EXACT-kernel path, which explicitly skips Fillet ("needs the mesh
        // kernel") — a document containing one actually renders through regenerate_mesh(), so
        // that's what must be tested here, or the fillet is silently never applied.
        let (mesh, _tangent_edges) = regenerate_mesh(&doc).expect("mesh regen with a top-rim fillet should produce a body");
        assert!(!mesh.positions.is_empty(), "fillet must not empty out the body");
        assert!(hworks_geometry::is_manifold(&mesh), "beveled body must stay a closed, 2-manifold surface (a failed corner patch leaves a hole)");

        let tol = 1.0e-3_f32;
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &mesh.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        eprintln!("bbox lo={lo:?} hi={hi:?}  (box was x[{x0},{x1}] y[0,{dist}] z[{z0},{z1}])");
        assert!(lo[0] >= x0 as f32 - tol && hi[0] <= x1 as f32 + tol, "x out of [{x0},{x1}]: {lo:?}..{hi:?}");
        assert!(lo[1] >= -tol && hi[1] <= dist as f32 + tol, "y out of [0,{dist}]: {lo:?}..{hi:?}");
        assert!(lo[2] >= z0 as f32 - tol && hi[2] <= z1 as f32 + tol, "z out of [{z0},{z1}] — the block bug: {lo:?}..{hi:?}");
    }

    /// The fillererror3.hcad box replayed through the real regen path, filleted with the
    /// given picked chains, then checked: manifold, in-bounds, and — the wing/bowtie class
    /// of bug — no vertex still at any of the box's ORIGINAL top corners (a correct fillet
    /// of any top edge removes the corners it touches; overlapping membrane/fan geometry
    /// left them in place).
    fn fillererror3_box_filleted(radius: f64, edges: Vec<Vec<[f64; 3]>>, gone_corners: &[[f64; 3]]) {
        let (x0, x1) = (0.0_f64, 5.708608627319336);
        let (z0, z1) = (0.0_f64, 3.6159682273864746);
        let dist = 2.903108596801758_f64;
        let top = PlaneRef { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 0.0, -1.0], normal: [0.0, 1.0, 0.0], datum: true };
        let mut s = Sketch::default();
        let p0 = s.add_point(x0, -z0);
        let p1 = s.add_point(x1, -z0);
        let p2 = s.add_point(x1, -z1);
        let p3 = s.add_point(x0, -z1);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        s.add_line(p3, p0, false);

        let mut doc = Document::with_default_planes();
        doc.add_feature(FeatureKind::Extrude { sketch: s, regions: vec![], plane: top, distance: dist, back: 0.0, thin: 0.0, thin_side: 0 });
        doc.add_feature(FeatureKind::Fillet { radius, edges });

        let (mesh, _tangent_edges) = regenerate_mesh(&doc).expect("mesh regen with a fillet should produce a body");
        assert!(!mesh.positions.is_empty(), "fillet must not empty out the body");
        assert!(hworks_geometry::is_manifold(&mesh), "beveled body must stay a closed, 2-manifold surface");

        let tol = 1.0e-3_f32;
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for p in &mesh.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        eprintln!("bbox lo={lo:?} hi={hi:?}  (box was x[{x0},{x1}] y[0,{dist}] z[{z0},{z1}])");
        assert!(lo[0] >= x0 as f32 - tol && hi[0] <= x1 as f32 + tol, "x out of [{x0},{x1}]: {lo:?}..{hi:?}");
        assert!(lo[1] >= -tol && hi[1] <= dist as f32 + tol, "y out of [0,{dist}]: {lo:?}..{hi:?}");
        assert!(lo[2] >= z0 as f32 - tol && hi[2] <= z1 as f32 + tol, "z out of [{z0},{z1}]: {lo:?}..{hi:?}");
        for c in gone_corners {
            let stale = mesh.positions.iter().any(|p| {
                (p[0] as f64 - c[0]).abs() < 1e-5 && (p[1] as f64 - c[1]).abs() < 1e-5 && (p[2] as f64 - c[2]).abs() < 1e-5
            });
            assert!(!stale, "original corner {c:?} still present — leftover membrane/fan geometry (the wing artifact)");
        }
    }

    #[test]
    fn single_edge_fillet_through_the_real_regen_pipeline() {
        // fillererror3's box with ONE top edge filleted (the terminal-splice case): both of
        // that edge's end corners must be notched away.
        let (x1, dist) = (5.708608627319336, 2.903108596801758);
        fillererror3_box_filleted(
            1.3064,
            vec![vec![[0.0, dist, 0.0], [x1, dist, 0.0]]],
            &[[0.0, dist, 0.0], [x1, dist, 0.0]],
        );
    }

    #[test]
    fn top_rim_loop_fillet_through_the_real_regen_pipeline() {
        // The EXACT committed feature from saved files/fillererror3.hcad: the whole top rim
        // as one closed picked chain (first point repeated), r≈0.95. Every corner is a WELD
        // (two rounded edges + one sharp vertical edge), which used to leave "wing" fins —
        // strips ending on mismatched per-edge arcs, papered over by a fan. All four original
        // top corners must be gone from the mesh.
        let (x1, z1, dist) = (5.708608627319336, 3.6159682273864746, 2.903108596801758);
        fillererror3_box_filleted(
            0.949999988079071,
            vec![vec![
                [x1, dist, z1],
                [x1, dist, 0.0],
                [0.0, dist, 0.0],
                [0.0, dist, z1],
                [x1, dist, z1],
            ]],
            &[[x1, dist, z1], [x1, dist, 0.0], [0.0, dist, 0.0], [0.0, dist, z1]],
        );
    }
}
