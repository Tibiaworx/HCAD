//! Vector tool icons, drawn with egui primitives.
//!
//! Line art rather than bitmaps: the icons pick up the current text colour (so they follow
//! enabled/disabled/selected state and any theme change), stay crisp at any DPI or zoom, and
//! add no binary assets to the repo. Each icon is a list of [`Prim`]s in a 0..1 unit box with
//! **y pointing down** (screen convention), scaled into whatever rect the button gives it.
//!
//! `prims()` is the single source of truth for the artwork: [`paint`] renders it through egui,
//! and the ignored `dump_icon_sheet` test writes the same data out for offline preview.

use bevy_egui::egui;

/// Every icon the toolbars can draw.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Icon {
    // --- sketch tools ---
    Select,
    Line,
    Circle,
    Arc,
    Rectangle,
    Slot,
    Polygon,
    Spline,
    Text,
    Dimension,
    Pattern,
    Mirror,
    Trim,
    // --- sketch tool variants (dropdowns) ---
    ConstructionLine,
    MidpointLine,
    CenterLine,
    PerimeterCircle,
    CenterRectangle,
    Parallelogram,
    SplineThrough,
    SplineControl,
    CenterpointSlot,
    ArcSlot,
    FillPattern,
    PowerTrim,
    TrimCorner,
    // --- features: add material ---
    Boss,
    Revolve,
    Loft,
    Sweep,
    // --- features: remove material ---
    Cut,
    RevolveCut,
    LoftCut,
    SweepCut,
    Hole,
    // --- features: modify ---
    Fillet,
    Chamfer,
    Shell,
    LinearPattern,
    CircularPattern,
    MirrorFeature,
    // --- reference geometry ---
    Plane,
    Sketch,
}

/// A drawing primitive in the 0..1 unit box (y down).
#[derive(Clone, Debug)]
pub enum Prim {
    /// Polyline through the points; `true` closes it back to the start.
    Path(Vec<[f32; 2]>, bool),
    /// A dashed polyline — construction geometry, or material being removed.
    Dash(Vec<[f32; 2]>),
    /// Filled polygon: arrow heads and the cursor body.
    Fill(Vec<[f32; 2]>),
    /// Circle outline: centre, radius.
    Circle([f32; 2], f32),
    /// A small filled dot — sketch points and pick handles.
    Dot([f32; 2]),
    /// Arc: centre, radius, start and end angle in DEGREES (y-down, so 90 points down).
    Arc([f32; 2], f32, f32, f32),
}

/// An arrow head (a small filled triangle) at `tip`, pointing along `dir`.
fn head(tip: [f32; 2], dir: [f32; 2]) -> Prim {
    let l = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt().max(1e-6);
    let (dx, dy) = (dir[0] / l, dir[1] / l);
    let (px, py) = (-dy, dx); // perpendicular
    let (b, w) = (0.20, 0.085); // length back from the tip, half-width
    Prim::Fill(vec![
        tip,
        [tip[0] - dx * b + px * w, tip[1] - dy * b + py * w],
        [tip[0] - dx * b - px * w, tip[1] - dy * b - py * w],
    ])
}

/// A regular polygon centred at `c` — used for the polygon tool and pattern seeds.
fn ngon(c: [f32; 2], r: f32, n: usize, start_deg: f32) -> Prim {
    let pts = (0..n)
        .map(|i| {
            let a = (start_deg + 360.0 * i as f32 / n as f32).to_radians();
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect();
    Prim::Path(pts, true)
}

/// An axis-aligned rectangle path.
fn boxp(x0: f32, y0: f32, x1: f32, y1: f32) -> Prim {
    Prim::Path(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]], true)
}

/// A cubic Bézier sampled into a polyline.
fn bezier(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> Vec<[f32; 2]> {
    (0..=16)
        .map(|i| {
            let t = i as f32 / 16.0;
            let m = 1.0 - t;
            let (a, b, c, d) = (m * m * m, 3.0 * m * m * t, 3.0 * m * t * t, t * t * t);
            [
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            ]
        })
        .collect()
}

/// Every icon with its name — the one list the bounds test and the preview dump share, so a
/// new icon can't silently escape either.
#[cfg(test)]
pub const ALL: &[(&str, Icon)] = &[
    ("Select", Icon::Select), ("Line", Icon::Line), ("Circle", Icon::Circle), ("Arc", Icon::Arc),
    ("Rectangle", Icon::Rectangle), ("Slot", Icon::Slot), ("Polygon", Icon::Polygon), ("Spline", Icon::Spline),
    ("Text", Icon::Text), ("Dimension", Icon::Dimension), ("Pattern", Icon::Pattern), ("Mirror", Icon::Mirror),
    ("Trim", Icon::Trim),
    ("ConstructionLine", Icon::ConstructionLine), ("MidpointLine", Icon::MidpointLine), ("CenterLine", Icon::CenterLine),
    ("PerimeterCircle", Icon::PerimeterCircle), ("CenterRectangle", Icon::CenterRectangle),
    ("Parallelogram", Icon::Parallelogram), ("SplineThrough", Icon::SplineThrough), ("SplineControl", Icon::SplineControl),
    ("CenterpointSlot", Icon::CenterpointSlot), ("ArcSlot", Icon::ArcSlot), ("FillPattern", Icon::FillPattern),
    ("PowerTrim", Icon::PowerTrim), ("TrimCorner", Icon::TrimCorner),
    ("Boss", Icon::Boss), ("Revolve", Icon::Revolve), ("Loft", Icon::Loft), ("Sweep", Icon::Sweep),
    ("Cut", Icon::Cut), ("RevolveCut", Icon::RevolveCut), ("LoftCut", Icon::LoftCut), ("SweepCut", Icon::SweepCut),
    ("Hole", Icon::Hole), ("Fillet", Icon::Fillet), ("Chamfer", Icon::Chamfer), ("Shell", Icon::Shell),
    ("LinearPattern", Icon::LinearPattern), ("CircularPattern", Icon::CircularPattern),
    ("MirrorFeature", Icon::MirrorFeature), ("Plane", Icon::Plane), ("Sketch", Icon::Sketch),
];

/// The artwork for one icon, in the 0..1 unit box.
pub fn prims(icon: Icon) -> Vec<Prim> {
    use Icon::*;
    match icon {
        // ---------------- sketch tools ----------------
        Select => vec![Prim::Fill(vec![
            [0.30, 0.10],
            [0.30, 0.84],
            [0.45, 0.68],
            [0.56, 0.90],
            [0.68, 0.84],
            [0.57, 0.63],
            [0.76, 0.60],
        ])],
        Line => vec![Prim::Path(vec![[0.18, 0.82], [0.82, 0.18]], false), Prim::Dot([0.18, 0.82]), Prim::Dot([0.82, 0.18])],
        Circle => vec![Prim::Circle([0.5, 0.5], 0.34), Prim::Dot([0.5, 0.5])],
        // An upward-bulging arc with its two endpoints.
        Arc => vec![Prim::Arc([0.5, 0.74], 0.42, 180.0, 360.0), Prim::Dot([0.08, 0.74]), Prim::Dot([0.92, 0.74])],
        Rectangle => vec![boxp(0.16, 0.24, 0.84, 0.76), Prim::Dot([0.16, 0.24]), Prim::Dot([0.84, 0.76])],
        // Stadium: two half-round ends joined by straight flanks.
        Slot => vec![
            Prim::Arc([0.33, 0.5], 0.2, 90.0, 270.0),
            Prim::Arc([0.67, 0.5], 0.2, 270.0, 450.0),
            Prim::Path(vec![[0.33, 0.30], [0.67, 0.30]], false),
            Prim::Path(vec![[0.33, 0.70], [0.67, 0.70]], false),
        ],
        Polygon => vec![ngon([0.5, 0.5], 0.36, 6, -90.0), Prim::Dot([0.5, 0.5])],
        Spline => {
            let mut v = vec![Prim::Path(bezier([0.12, 0.74], [0.34, 0.10], [0.66, 0.90], [0.88, 0.26]), false)];
            v.push(Prim::Dot([0.12, 0.74]));
            v.push(Prim::Dot([0.88, 0.26]));
            v
        }
        // A capital "A".
        Text => vec![
            Prim::Path(vec![[0.24, 0.84], [0.5, 0.16], [0.76, 0.84]], false),
            Prim::Path(vec![[0.35, 0.60], [0.65, 0.60]], false),
        ],
        // Witness lines either side of an arrowed dimension line.
        Dimension => vec![
            Prim::Path(vec![[0.16, 0.20], [0.16, 0.84]], false),
            Prim::Path(vec![[0.84, 0.20], [0.84, 0.84]], false),
            Prim::Path(vec![[0.16, 0.52], [0.84, 0.52]], false),
            head([0.16, 0.52], [-1.0, 0.0]),
            head([0.84, 0.52], [1.0, 0.0]),
        ],
        // A seed square repeated across a grid.
        Pattern => vec![
            boxp(0.10, 0.10, 0.36, 0.36),
            boxp(0.10, 0.64, 0.36, 0.90),
            boxp(0.64, 0.10, 0.90, 0.36),
            boxp(0.64, 0.64, 0.90, 0.90),
        ],
        // Two shapes either side of a dashed mirror axis.
        Mirror | MirrorFeature => vec![
            Prim::Path(vec![[0.10, 0.82], [0.36, 0.82], [0.36, 0.24]], true),
            Prim::Path(vec![[0.90, 0.82], [0.64, 0.82], [0.64, 0.24]], true),
            Prim::Dash(vec![[0.5, 0.08], [0.5, 0.92]]),
        ],
        // A crossing where one branch has been trimmed away (dashed).
        Trim => vec![
            Prim::Path(vec![[0.10, 0.18], [0.90, 0.82]], false),
            Prim::Path(vec![[0.10, 0.82], [0.50, 0.50]], false),
            Prim::Dash(vec![[0.50, 0.50], [0.90, 0.18]]),
            Prim::Dot([0.50, 0.50]),
        ],

        // ---------------- sketch tool variants ----------------
        // Construction geometry is dashed, matching how it draws in the sketch.
        ConstructionLine => vec![Prim::Dash(vec![[0.18, 0.82], [0.82, 0.18]]), Prim::Dot([0.18, 0.82]), Prim::Dot([0.82, 0.18])],
        // Grows symmetrically from the middle: centre mark, arrows heading both ways.
        MidpointLine => vec![
            Prim::Path(vec![[0.22, 0.78], [0.78, 0.22]], false),
            Prim::Dot([0.5, 0.5]),
            head([0.88, 0.12], [1.0, -1.0]),
            head([0.12, 0.88], [-1.0, 1.0]),
        ],
        // The two combined: a dashed line that grows from its centre.
        CenterLine => vec![
            Prim::Dash(vec![[0.22, 0.78], [0.78, 0.22]]),
            Prim::Dot([0.5, 0.5]),
            head([0.88, 0.12], [1.0, -1.0]),
            head([0.12, 0.88], [-1.0, 1.0]),
        ],
        // Defined by points ON the rim rather than from the centre.
        PerimeterCircle => {
            let mut v = vec![Prim::Circle([0.5, 0.5], 0.34)];
            for deg in [-90.0f32, 30.0, 150.0] {
                let a = deg.to_radians();
                v.push(Prim::Dot([0.5 + 0.34 * a.cos(), 0.5 + 0.34 * a.sin()]));
            }
            v
        }
        // Centre out, with the X construction diagonals the tool actually adds.
        CenterRectangle => vec![
            boxp(0.16, 0.24, 0.84, 0.76),
            Prim::Dash(vec![[0.16, 0.24], [0.84, 0.76]]),
            Prim::Dash(vec![[0.84, 0.24], [0.16, 0.76]]),
            Prim::Dot([0.5, 0.5]),
        ],
        Parallelogram => vec![
            Prim::Path(vec![[0.30, 0.24], [0.94, 0.24], [0.70, 0.76], [0.06, 0.76]], true),
            Prim::Dot([0.30, 0.24]),
            Prim::Dot([0.94, 0.24]),
        ],
        // Interpolating: the points sit ON the curve.
        SplineThrough => vec![
            Prim::Path(bezier([0.12, 0.74], [0.34, 0.10], [0.66, 0.90], [0.88, 0.26]), false),
            Prim::Dot([0.12, 0.74]),
            Prim::Dot([0.5, 0.5]),
            Prim::Dot([0.88, 0.26]),
        ],
        // Approximating: a dashed control polygon with the handles off the curve.
        SplineControl => vec![
            Prim::Path(bezier([0.12, 0.74], [0.34, 0.10], [0.66, 0.90], [0.88, 0.26]), false),
            Prim::Dash(vec![[0.12, 0.74], [0.34, 0.10], [0.66, 0.90], [0.88, 0.26]]),
            Prim::Dot([0.34, 0.10]),
            Prim::Dot([0.66, 0.90]),
        ],
        // A slot placed from its centre: centre mark on the axis between the end radii.
        CenterpointSlot => vec![
            Prim::Arc([0.33, 0.5], 0.2, 90.0, 270.0),
            Prim::Arc([0.67, 0.5], 0.2, 270.0, 450.0),
            Prim::Path(vec![[0.33, 0.30], [0.67, 0.30]], false),
            Prim::Path(vec![[0.33, 0.70], [0.67, 0.70]], false),
            Prim::Dash(vec![[0.33, 0.5], [0.67, 0.5]]),
            Prim::Dot([0.5, 0.5]),
        ],
        // The same stadium bent along an arc.
        ArcSlot => vec![
            Prim::Arc([0.5, 0.88], 0.46, 205.0, 335.0),
            Prim::Arc([0.5, 0.88], 0.24, 205.0, 335.0),
            Prim::Path(vec![[0.083, 0.685], [0.283, 0.779]], false),
            Prim::Path(vec![[0.917, 0.685], [0.717, 0.779]], false),
        ],
        // Copies tiled to fill a closed region.
        FillPattern => {
            let mut v = vec![Prim::Path(vec![[0.10, 0.20], [0.90, 0.12], [0.88, 0.84], [0.14, 0.90]], true)];
            for y in [0.38f32, 0.66] {
                for x in [0.28f32, 0.50, 0.72] {
                    v.push(Prim::Dot([x, y]));
                }
            }
            v
        }
        // A stroke dragged across everything it crosses.
        PowerTrim => vec![
            Prim::Path(vec![[0.26, 0.12], [0.26, 0.88]], false),
            Prim::Path(vec![[0.52, 0.12], [0.52, 0.88]], false),
            Prim::Path(vec![[0.78, 0.12], [0.78, 0.88]], false),
            Prim::Dash(bezier([0.08, 0.74], [0.36, 0.86], [0.62, 0.20], [0.94, 0.32])),
        ],
        // Two lines extended (dashed) until they meet at a corner.
        TrimCorner => vec![
            Prim::Path(vec![[0.20, 0.12], [0.20, 0.56]], false),
            Prim::Dash(vec![[0.20, 0.56], [0.20, 0.80]]),
            Prim::Path(vec![[0.88, 0.80], [0.44, 0.80]], false),
            Prim::Dash(vec![[0.44, 0.80], [0.20, 0.80]]),
            Prim::Dot([0.20, 0.80]),
        ],

        // ---------------- features: add material ----------------
        // A profile with material pulled out of it.
        Boss => vec![boxp(0.18, 0.62, 0.82, 0.90), Prim::Path(vec![[0.5, 0.58], [0.5, 0.20]], false), head([0.5, 0.10], [0.0, -1.0])],
        // Profile + axis, with a sweep arc showing the turn.
        Revolve => vec![
            Prim::Dash(vec![[0.16, 0.06], [0.16, 0.94]]),
            boxp(0.30, 0.46, 0.58, 0.88),
            Prim::Arc([0.16, 0.46], 0.42, -78.0, 0.0),
            head([0.20, 0.06], [-0.34, -0.94]),
        ],
        // Two profiles skinned together.
        Loft => vec![
            Prim::Path(vec![[0.32, 0.18], [0.68, 0.18]], false),
            Prim::Path(vec![[0.12, 0.84], [0.88, 0.84]], false),
            Prim::Path(vec![[0.32, 0.18], [0.12, 0.84]], false),
            Prim::Path(vec![[0.68, 0.18], [0.88, 0.84]], false),
            Prim::Dash(vec![[0.5, 0.18], [0.5, 0.84]]),
        ],
        // A profile carried along a curved path.
        Sweep => vec![
            Prim::Dash(bezier([0.14, 0.80], [0.34, 0.16], [0.66, 0.84], [0.88, 0.22])),
            Prim::Circle([0.14, 0.80], 0.13),
            Prim::Circle([0.88, 0.22], 0.13),
        ],

        // ---------------- features: remove material ----------------
        // Same grammar as the boss icons, but the arrow drives INTO the profile and the
        // removed material is dashed.
        Cut => vec![boxp(0.18, 0.62, 0.82, 0.90), Prim::Path(vec![[0.5, 0.08], [0.5, 0.46]], false), head([0.5, 0.58], [0.0, 1.0])],
        RevolveCut => vec![
            Prim::Dash(vec![[0.16, 0.06], [0.16, 0.94]]),
            Prim::Dash(vec![[0.30, 0.46], [0.58, 0.46], [0.58, 0.88], [0.30, 0.88], [0.30, 0.46]]),
            Prim::Arc([0.16, 0.46], 0.42, -78.0, 0.0),
            head([0.20, 0.06], [-0.34, -0.94]),
        ],
        LoftCut => vec![
            Prim::Dash(vec![[0.32, 0.18], [0.68, 0.18]]),
            Prim::Dash(vec![[0.12, 0.84], [0.88, 0.84]]),
            Prim::Dash(vec![[0.32, 0.18], [0.12, 0.84]]),
            Prim::Dash(vec![[0.68, 0.18], [0.88, 0.84]]),
            Prim::Path(vec![[0.5, 0.28], [0.5, 0.52]], false),
            head([0.5, 0.66], [0.0, 1.0]),
        ],
        SweepCut => vec![
            Prim::Dash(bezier([0.14, 0.80], [0.34, 0.16], [0.66, 0.84], [0.88, 0.22])),
            Prim::Circle([0.14, 0.80], 0.13),
            head([0.88, 0.22], [0.7, -0.7]),
        ],
        // A tapped hole: bore circle with thread ticks.
        Hole => {
            // Thread crest as a dashed circle around the solid bore, plus the centre mark.
            let ring: Vec<[f32; 2]> = (0..=40)
                .map(|i| {
                    let a = (i as f32 * 9.0f32).to_radians();
                    [0.5 + 0.36 * a.cos(), 0.5 + 0.36 * a.sin()]
                })
                .collect();
            vec![Prim::Circle([0.5, 0.5], 0.24), Prim::Dash(ring), Prim::Dot([0.5, 0.5])]
        }

        // ---------------- features: modify ----------------
        // An L corner, rounded.
        Fillet => vec![
            Prim::Path(vec![[0.16, 0.92], [0.16, 0.46]], false),
            Prim::Arc([0.46, 0.46], 0.30, 180.0, 270.0),
            Prim::Path(vec![[0.46, 0.16], [0.92, 0.16]], false),
            Prim::Dash(vec![[0.16, 0.16], [0.16, 0.46]]),
            Prim::Dash(vec![[0.16, 0.16], [0.46, 0.16]]),
        ],
        // The same corner, cut flat.
        Chamfer => vec![
            Prim::Path(vec![[0.16, 0.92], [0.16, 0.50], [0.50, 0.16], [0.92, 0.16]], false),
            Prim::Dash(vec![[0.16, 0.16], [0.16, 0.50]]),
            Prim::Dash(vec![[0.16, 0.16], [0.50, 0.16]]),
        ],
        // A box hollowed out — outer wall plus the inner void, open at the top.
        Shell => vec![
            Prim::Path(vec![[0.12, 0.16], [0.12, 0.88], [0.88, 0.88], [0.88, 0.16]], false),
            Prim::Path(vec![[0.30, 0.16], [0.30, 0.70], [0.70, 0.70], [0.70, 0.16]], false),
        ],
        // Seed square stepped along a direction arrow.
        LinearPattern => vec![
            boxp(0.08, 0.14, 0.34, 0.40),
            boxp(0.40, 0.14, 0.66, 0.40),
            boxp(0.72, 0.14, 0.98, 0.40),
            Prim::Path(vec![[0.08, 0.72], [0.80, 0.72]], false),
            head([0.92, 0.72], [1.0, 0.0]),
        ],
        // Seed squares stepped around a centre.
        CircularPattern => {
            // Seeds sitting ON a dashed pattern circle — the circular twin of the linear
            // pattern's row of seeds along an arrow.
            let ring: Vec<[f32; 2]> = (0..=40)
                .map(|i| {
                    let a = (i as f32 * 9.0f32).to_radians();
                    [0.5 + 0.36 * a.cos(), 0.5 + 0.36 * a.sin()]
                })
                .collect();
            let mut v = vec![Prim::Dash(ring), Prim::Dot([0.5, 0.5])];
            for deg in [-90.0f32, 30.0, 150.0] {
                let a = deg.to_radians();
                let (cx, cy) = (0.5 + 0.36 * a.cos(), 0.5 + 0.36 * a.sin());
                v.push(boxp(cx - 0.11, cy - 0.11, cx + 0.11, cy + 0.11));
            }
            v
        }

        // ---------------- reference geometry ----------------
        // A plane in perspective.
        Plane => vec![Prim::Path(vec![[0.06, 0.66], [0.58, 0.30], [0.94, 0.36], [0.42, 0.72]], true), Prim::Dot([0.5, 0.51])],
        // A profile on a plane.
        Sketch => vec![
            Prim::Dash(vec![[0.06, 0.70], [0.58, 0.40], [0.94, 0.46], [0.42, 0.76], [0.06, 0.70]]),
            Prim::Path(vec![[0.30, 0.56], [0.56, 0.42], [0.76, 0.52], [0.50, 0.66]], true),
            Prim::Dot([0.30, 0.56]),
            Prim::Dot([0.76, 0.52]),
        ],
    }
}

/// Draw `icon` into `rect`, in `color`. The artwork is defined in a square unit box, so a
/// non-square rect gets the icon centred in its largest fitting square.
pub fn paint(painter: &egui::Painter, rect: egui::Rect, icon: Icon, color: egui::Color32) {
    let side = rect.width().min(rect.height());
    let o = rect.center() - egui::vec2(side * 0.5, side * 0.5);
    let w = (side * 0.09).clamp(1.0, 2.0); // stroke scales with the icon, but stays hairline-crisp
    let stroke = egui::Stroke::new(w, color);
    let p = |q: &[f32; 2]| egui::pos2(o.x + q[0] * side, o.y + q[1] * side);
    let poly = |pts: &[[f32; 2]], closed: bool| {
        let mut v: Vec<egui::Pos2> = pts.iter().map(p).collect();
        if closed {
            if let Some(&f) = v.first() {
                v.push(f);
            }
        }
        v
    };
    for prim in prims(icon) {
        match prim {
            Prim::Path(pts, closed) => {
                painter.add(egui::Shape::line(poly(&pts, closed), stroke));
            }
            Prim::Dash(pts) => {
                painter.add(egui::Shape::dashed_line(&poly(&pts, false), stroke, side * 0.1, side * 0.07));
            }
            Prim::Fill(pts) => {
                painter.add(egui::Shape::convex_polygon(poly(&pts, false), color, egui::Stroke::NONE));
            }
            Prim::Circle(c, r) => {
                painter.circle_stroke(p(&c), r * side, stroke);
            }
            Prim::Dot(c) => {
                painter.circle_filled(p(&c), (side * 0.075).max(1.2), color);
            }
            Prim::Arc(c, r, a0, a1) => {
                let n = 24;
                let pts: Vec<egui::Pos2> = (0..=n)
                    .map(|i| {
                        let a = (a0 + (a1 - a0) * i as f32 / n as f32).to_radians();
                        p(&[c[0] + r * a.cos(), c[1] + r * a.sin()])
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every icon must actually draw something, and stay inside the unit box (a stray
    /// coordinate would clip against the neighbouring button).
    #[test]
    fn icons_are_non_empty_and_in_bounds() {
        for &(name, icon) in ALL {
            let ps = prims(icon);
            assert!(!ps.is_empty(), "{name} draws nothing");
            let mut check = |q: [f32; 2]| {
                assert!(
                    (-0.02..=1.02).contains(&q[0]) && (-0.02..=1.02).contains(&q[1]),
                    "{name} has a point outside the unit box: {q:?}"
                );
            };
            for prim in ps {
                match prim {
                    Prim::Path(v, _) | Prim::Dash(v) | Prim::Fill(v) => v.into_iter().for_each(&mut check),
                    Prim::Dot(c) => check(c),
                    Prim::Circle(c, r) => {
                        check([c[0] - r, c[1] - r]);
                        check([c[0] + r, c[1] + r]);
                    }
                    // Sample the arc the way `paint` does — a partial sweep's extent is not
                    // its full circle's bounding box.
                    Prim::Arc(c, r, a0, a1) => {
                        for i in 0..=24 {
                            let a = (a0 + (a1 - a0) * i as f32 / 24.0).to_radians();
                            check([c[0] + r * a.cos(), c[1] + r * a.sin()]);
                        }
                    }
                }
            }
        }
    }

    /// Dump every icon as plain text so the artwork can be rasterised and eyeballed offline.
    /// `cargo test -p hworks-app --release dump_icon_sheet -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn dump_icon_sheet() {
        let mut out = String::new();
        for &(name, icon) in ALL {
            out.push_str(&format!("ICON {name}
"));
            for prim in prims(icon) {
                let pts = |v: &Vec<[f32; 2]>| v.iter().map(|q| format!("{},{}", q[0], q[1])).collect::<Vec<_>>().join(" ");
                match prim {
                    Prim::Path(v, c) => out.push_str(&format!("path {} {}
", if c { 1 } else { 0 }, pts(&v))),
                    Prim::Dash(v) => out.push_str(&format!("dash {}
", pts(&v))),
                    Prim::Fill(v) => out.push_str(&format!("fill {}
", pts(&v))),
                    Prim::Circle(c, r) => out.push_str(&format!("circle {} {} {r}
", c[0], c[1])),
                    Prim::Dot(c) => out.push_str(&format!("dot {} {}
", c[0], c[1])),
                    Prim::Arc(c, r, a0, a1) => out.push_str(&format!("arc {} {} {r} {a0} {a1}
", c[0], c[1])),
                }
            }
        }
        let path = std::env::var("ICON_SHEET").unwrap_or_else(|_| "icons.txt".into());
        std::fs::write(&path, out).unwrap();
        println!("wrote {path}");
    }
}
