//! `hworks-sketch` — Layer 2: the 2D sketcher and constraint solver.
//!
//! A sketch is a 2D problem in a plane's local UV coordinates: a set of points
//! (the solver's unknowns), entities built on those points (lines, circles,
//! rectangles, construction geometry), and constraints/dimensions that the
//! solver satisfies all at once. See `DESIGN.md` §5.
//!
//! At milestone **M0** this is a stub describing the data model. Free drawing
//! arrives at **M1**, the Newton/least-squares solver at **M2**.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// A 2D point in plane-local coordinates — every entity endpoint is one of these,
/// and each is an unknown the constraint solver positions. A `fixed` point is locked:
/// the solver never moves it (used for geometry projected from the 3D body — corners,
/// centres, and edges — so a sketch can be referenced/constrained to the solid).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub fixed: bool,
}

/// A drawable sketch entity. A rectangle/square is four `Line`s plus constraints;
/// a "construction line with midpoint" is a construction `Line` plus a midpoint
/// `Point` tied to it by a `Midpoint` constraint.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SketchEntity {
    /// A straight edge. `construction` lines guide but don't form profiles;
    /// `reference` lines are projected from the 3D body (locked, also non-profile).
    Line {
        a: usize,
        b: usize,
        construction: bool,
        #[serde(default)]
        reference: bool,
    },
    Circle {
        center: usize,
        radius: f64,
        /// A construction circle guides/sizes (e.g. a polygon's circumscribed circle)
        /// but isn't a profile boundary, so it doesn't form a fillable region.
        #[serde(default)]
        construction: bool,
    },
    Point { at: usize },
    /// A smooth spline through (or guided by) its `points`. `control == false` ⇒ the
    /// curve passes *through* the points (interpolating Catmull-Rom); `control == true`
    /// ⇒ the points are a B-spline control hull the curve only approaches. `closed`
    /// wraps it into a loop. `construction` splines guide but don't form profiles.
    Spline {
        points: Vec<usize>,
        #[serde(default)]
        closed: bool,
        #[serde(default)]
        construction: bool,
        #[serde(default)]
        control: bool,
    },
    /// A slot whose two rounded ends are centred at points `a` and `b`, with half-width
    /// `radius`. If `mid` is `Some`, the centre line bends through that point into an arc
    /// (a curved slot); otherwise it's a straight stadium.
    Slot {
        a: usize,
        b: usize,
        radius: f64,
        #[serde(default)]
        construction: bool,
        #[serde(default)]
        mid: Option<usize>,
    },
    /// Outlined text. The glyph outlines are *baked* once (in the app, from a system
    /// font) into `contours` — closed loops in normalized EM space (baseline at y=0, cap
    /// height ≈ 1, x advancing right; counters/holes included). They're transformed by
    /// `origin` (a sketch point = baseline start), `height`, `rotation`, optional `mirror`
    /// and `arc` (text-on-arc radius) to produce the actual profile. The remaining fields
    /// are the parameters kept so the UI can re-bake on a font/style/text edit.
    Text {
        origin: usize,
        contours: Vec<Vec<[f64; 2]>>,
        height: f64,
        #[serde(default)]
        rotation: f64,
        #[serde(default)]
        mirror: bool,
        #[serde(default)]
        arc: f64,
        text: String,
        font: String,
        #[serde(default)]
        bold: bool,
        #[serde(default)]
        italic: bool,
        #[serde(default)]
        spacing: f64,
    },
}

/// Transform baked text `contours` (normalized EM space) into sketch-plane space using
/// the baseline `origin`, `height` scale, `rotation` (rad), `mirror` (flip X), and `arc`
/// (text-on-arc radius; 0 = straight). Returns closed loops ready to fill/extrude.
pub fn text_contours(
    origin: [f64; 2],
    contours: &[Vec<[f64; 2]>],
    height: f64,
    rotation: f64,
    mirror: bool,
    arc: f64,
) -> Vec<Vec<[f64; 2]>> {
    let (sr, cr) = rotation.sin_cos();
    let map = |p: [f64; 2]| -> [f64; 2] {
        // 1) scale to height (normalized cap height ≈ 1).
        let mut x = p[0] * height;
        let y = p[1] * height;
        if mirror {
            x = -x;
        }
        // 2) optional arc warp: bend the baseline onto a circle of radius |arc|. Positive
        //    arc curves the text upward (centre above), negative downward.
        let (mut lx, mut ly) = (x, y);
        if arc.abs() > 1e-9 {
            let r = arc;
            let a = x / r; // arc angle from the baseline start
            lx = (r - y) * a.sin();
            ly = r - (r - y) * a.cos();
        }
        // 3) rotate about the origin, then translate.
        [
            origin[0] + lx * cr - ly * sr,
            origin[1] + lx * sr + ly * cr,
        ]
    };
    contours
        .iter()
        .map(|c| {
            let mut out: Vec<[f64; 2]> = c.iter().map(|&p| map(p)).collect();
            // Mirroring flips winding; reverse so fill orientation stays consistent.
            if mirror {
                out.reverse();
            }
            out
        })
        .collect()
}

/// Geometric and dimensional relations the solver drives the geometry to satisfy.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Constraint {
    Coincident(usize, usize),
    Horizontal(usize, usize),
    Vertical(usize, usize),
    Midpoint { mid: usize, a: usize, b: usize },
    /// Driving distance between points a and b. `offset` is the (display-only)
    /// perpendicular offset of the dimension line from the geometry, in plane uv.
    /// `axis` chooses whether `value` measures the true (aligned) distance or only
    /// the horizontal / vertical gap (SolidWorks-style projected dimensions).
    Distance { a: usize, b: usize, value: f64, offset: f64, #[serde(default)] axis: DimAxis },
    /// Line (a,b) parallel to line (c,d).
    Parallel(usize, usize, usize, usize),
    /// Line (a,b) perpendicular to line (c,d).
    Perpendicular(usize, usize, usize, usize),
    /// Line (a,b) has the same length as line (c,d).
    Equal(usize, usize, usize, usize),
    /// Line (a,b) tangent to a circle centred at `center` with the given radius.
    Tangent { a: usize, b: usize, center: usize, radius: f64 },
    /// The circles centred at points `a` and `b` have equal radius (a drives b).
    /// Radius isn't a solver variable, so this is enforced after each solve.
    EqualRadius { a: usize, b: usize },
    /// Driving radius dimension: the circle centred at `center` has radius `value`.
    /// Enforced after the solve (radius isn't a point variable). `diameter` is a
    /// display choice — when true the dimension reads as Ø (2·value).
    Radius { center: usize, value: f64, #[serde(default)] diameter: bool },
    /// Driving angle between directed lines (a→b) and (c→d). `value` is in radians;
    /// `offset` is the (display-only) radius of the angle arc from the vertex.
    Angle { a: usize, b: usize, c: usize, d: usize, value: f64, offset: f64 },
    /// Driving perpendicular distance from point `p` to the line through (a,b). Used to
    /// dimension a sketch line off a body edge (the edge is a locked reference line).
    PointLineDistance { p: usize, a: usize, b: usize, value: f64, offset: f64 },
    /// Point `p` lies on the rim of the circle centred at `center`. The radius is read
    /// from the circle entity at solve time, so the point follows radius edits.
    PointOnCircle { p: usize, center: usize },
    /// Point `p` lies on the (infinite) line through (a,b) — perpendicular distance is
    /// zero. Used to snap a sketch point/line onto a body edge (a locked reference line);
    /// two of these on one drawn line make it collinear with the edge.
    PointOnLine { p: usize, a: usize, b: usize },
    /// Point `p` lies on a body arc/circle of `radius` centred at (`cx`,`cy`). The centre
    /// and radius are baked (projected reference geometry), so this snaps a sketch point
    /// onto a rounded body edge without needing a sketch circle entity.
    PointOnArc { p: usize, cx: f64, cy: f64, radius: f64 },
}

/// Which span a [`Constraint::Distance`] measures: the true point-to-point distance
/// (`Aligned`) or only the horizontal / vertical component (projected dimensions).
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DimAxis {
    #[default]
    Aligned,
    Horizontal,
    Vertical,
}

/// A closed area of a sketch, ready to extrude: one outer boundary loop plus any
/// inner loops that are *holes* in it (e.g. a rectangle with a circular hole).
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Region {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
}

impl Region {
    /// A representative interior point (used for hit-testing which region the
    /// user clicked). The outer centroid, nudged off any hole it lands in.
    pub fn interior_point(&self) -> [f64; 2] {
        centroid(&self.outer)
    }
}

/// A 2D sketch bound to a plane (or, from M5, a planar face of a solid).
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Sketch {
    pub points: Vec<Point2>,
    pub entities: Vec<SketchEntity>,
    pub constraints: Vec<Constraint>,
}

impl Sketch {
    /// Add a free point and return its index (the solver's future unknown).
    pub fn add_point(&mut self, x: f64, y: f64) -> usize {
        self.points.push(Point2 { x, y, fixed: false });
        self.points.len() - 1
    }

    /// Add a locked point (projected from the 3D body) the solver won't move.
    pub fn add_fixed_point(&mut self, x: f64, y: f64) -> usize {
        self.points.push(Point2 { x, y, fixed: true });
        self.points.len() - 1
    }

    /// Add a line between two existing points.
    pub fn add_line(&mut self, a: usize, b: usize, construction: bool) {
        self.entities.push(SketchEntity::Line { a, b, construction, reference: false });
    }

    /// Add a reference line projected from a 3D body edge (locked, non-profile).
    /// Returns the new entity's index so the caller can select it for a constraint.
    pub fn add_reference_line(&mut self, a: usize, b: usize) -> usize {
        self.entities.push(SketchEntity::Line { a, b, construction: false, reference: true });
        self.entities.len() - 1
    }

    /// Add a circle from a center point and radius.
    pub fn add_circle(&mut self, center: usize, radius: f64) {
        self.entities.push(SketchEntity::Circle { center, radius, construction: false });
    }

    /// Add a construction circle (guides/sizes geometry but forms no profile region).
    /// Returns its entity index so callers can constrain points onto its rim.
    pub fn add_construction_circle(&mut self, center: usize, radius: f64) -> usize {
        self.entities.push(SketchEntity::Circle { center, radius, construction: true });
        self.entities.len() - 1
    }

    /// Remove all geometry, leaving an empty sketch (used when (re)entering a plane).
    pub fn clear(&mut self) {
        self.points.clear();
        self.entities.clear();
        self.constraints.clear();
    }

    /// Drop points no entity references (e.g. the stray endpoints left after a line
    /// is deleted), remapping indices. Constraints that touched a removed point are
    /// dropped too. Call after deleting entities.
    pub fn remove_unused_points(&mut self) {
        let n = self.points.len();
        let mut used = vec![false; n];
        for e in &self.entities {
            for p in entity_point_indices(e) {
                if p < n {
                    used[p] = true;
                }
            }
        }
        // A constraint on a vanished point is meaningless → drop it.
        self.constraints.retain(|c| constraint_point_indices(c).iter().all(|&p| p < n && used[p]));

        let mut remap = vec![0usize; n];
        let mut keep = Vec::with_capacity(n);
        for i in 0..n {
            if used[i] {
                remap[i] = keep.len();
                keep.push(self.points[i]);
            }
        }
        self.points = keep;
        for e in &mut self.entities {
            remap_entity(e, &remap);
        }
        for c in &mut self.constraints {
            remap_constraint(c, &remap);
        }
    }

    /// If the non-construction lines form a single closed loop, return the point
    /// indices in order around it. Otherwise `None`. This is the profile the
    /// kernel extrudes (M3). Requires every involved point to have degree 2 and
    /// the whole thing to be one cycle.
    pub fn closed_loop(&self) -> Option<Vec<usize>> {
        use std::collections::HashMap;
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut edges = 0usize;
        for e in &self.entities {
            if let SketchEntity::Line { a, b, construction: false, reference: false } = e {
                adj.entry(*a).or_default().push(*b);
                adj.entry(*b).or_default().push(*a);
                edges += 1;
            }
        }
        if edges < 3 || adj.values().any(|nbrs| nbrs.len() != 2) {
            return None;
        }
        let start = *adj.keys().min()?;
        let mut order = vec![start];
        let mut prev = start;
        let mut cur = adj[&start][0];
        while cur != start {
            order.push(cur);
            let nbrs = &adj[&cur];
            let next = if nbrs[0] == prev { nbrs[1] } else { nbrs[0] };
            prev = cur;
            cur = next;
            if order.len() > adj.len() {
                return None; // ran away — not a clean single cycle
            }
        }
        if order.len() == adj.len() {
            Some(order)
        } else {
            None // a cycle, but it didn't cover every involved point
        }
    }

    /// The closed outer profile as ordered 2D points, ready for the kernel to
    /// extrude — whichever way the sketch closes:
    ///   - a single closed loop of (non-construction) lines, or
    ///   - a single circle (tessellated into a polygon).
    /// Returns `None` if the sketch has no closed region.
    pub fn outer_profile(&self) -> Option<Vec<[f64; 2]>> {
        // Prefer a closed loop of lines (rectangles, polygons).
        if let Some(idx) = self.closed_loop() {
            return Some(idx.iter().map(|&i| [self.points[i].x, self.points[i].y]).collect());
        }
        // Otherwise, a lone circle becomes a polygonal profile.
        let mut found = None;
        let mut circles = 0;
        for e in &self.entities {
            if let SketchEntity::Circle { center, radius, .. } = e {
                found = Some((*center, *radius));
                circles += 1;
            }
        }
        if circles == 1 {
            let (c, r) = found?;
            let center = self.points.get(c)?;
            const SEGMENTS: usize = 64;
            let pts = (0..SEGMENTS)
                .map(|k| {
                    let a = std::f64::consts::TAU * (k as f64) / (SEGMENTS as f64);
                    [center.x + r * a.cos(), center.y + r * a.sin()]
                })
                .collect();
            return Some(pts);
        }
        None
    }

    /// Every closed contour in the sketch as an ordered point loop: each closed
    /// loop of non-construction lines, plus each circle (tessellated).
    pub fn contours(&self) -> Vec<Vec<[f64; 2]>> {
        use std::collections::{HashMap, HashSet};
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for e in &self.entities {
            if let SketchEntity::Line { a, b, construction: false, reference: false } = e {
                adj.entry(*a).or_default().push(*b);
                adj.entry(*b).or_default().push(*a);
            }
        }

        let mut contours = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut starts: Vec<usize> = adj.keys().copied().collect();
        starts.sort_unstable();
        for start in starts {
            if visited.contains(&start) {
                continue;
            }
            // Gather this connected component.
            let mut comp = Vec::new();
            let mut stack = vec![start];
            let mut seen = HashSet::new();
            while let Some(v) = stack.pop() {
                if !seen.insert(v) {
                    continue;
                }
                comp.push(v);
                for &w in &adj[&v] {
                    if !seen.contains(&w) {
                        stack.push(w);
                    }
                }
            }
            for &v in &comp {
                visited.insert(v);
            }
            // A simple cycle needs every vertex to have degree 2.
            if !comp.iter().all(|v| adj[v].len() == 2) {
                continue;
            }
            let mut order = vec![start];
            let (mut prev, mut cur) = (start, adj[&start][0]);
            let mut ok = true;
            while cur != start {
                order.push(cur);
                let nbrs = &adj[&cur];
                let next = if nbrs[0] == prev { nbrs[1] } else { nbrs[0] };
                prev = cur;
                cur = next;
                if order.len() > comp.len() {
                    ok = false;
                    break;
                }
            }
            if ok && order.len() == comp.len() {
                contours.push(order.iter().map(|&i| [self.points[i].x, self.points[i].y]).collect());
            }
        }

        for e in &self.entities {
            if let SketchEntity::Circle { center, radius, .. } = e {
                if let Some(c) = self.points.get(*center) {
                    const SEG: usize = 48;
                    contours.push(
                        (0..SEG)
                            .map(|k| {
                                let a = std::f64::consts::TAU * (k as f64) / (SEG as f64);
                                [c.x + radius * a.cos(), c.y + radius * a.sin()]
                            })
                            .collect(),
                    );
                }
            }
        }
        contours
    }

    /// The selectable closed regions of the sketch — its "contours".
    ///
    /// Built from the **planar arrangement** of all non-construction geometry: every
    /// curve is tessellated to straight segments, split at every intersection, and
    /// the bounded minimal faces are traced. So a circle cut by two lines yields the
    /// pie slice *and* the remainder as separate, individually-selectable regions.
    /// Disconnected inner loops are then nested as holes by even/odd containment
    /// (a rectangle with a separate circle inside becomes one region with a hole).
    pub fn regions(&self) -> Vec<Region> {
        let contours = self.arrangement_faces();
        let n = contours.len();
        let areas: Vec<f64> = contours.iter().map(|c| area(c)).collect();
        // `j` contains `i` if it's bigger and `i`'s centroid falls inside it.
        let contains = |j: usize, i: usize| {
            j != i && areas[j] > areas[i] && point_in_poly(centroid(&contours[i]), &contours[j])
        };
        let depth: Vec<usize> =
            (0..n).map(|i| (0..n).filter(|&j| contains(j, i)).count()).collect();

        let mut regions = Vec::new();
        for i in 0..n {
            if depth[i] % 2 != 0 {
                continue; // odd nesting depth ⇒ this face is a hole, not a solid
            }
            let holes = (0..n)
                .filter(|&k| depth[k] == depth[i] + 1 && contains(i, k))
                .map(|k| contours[k].clone())
                .collect();
            regions.push(Region { outer: contours[i].clone(), holes });
        }
        regions
    }

    /// Tessellate all non-construction geometry to straight segments, split them at
    /// every intersection, and trace the bounded minimal faces of the resulting
    /// planar arrangement. Each face is a simple polygon wound counter-clockwise.
    fn arrangement_faces(&self) -> Vec<Vec<[f64; 2]>> {
        let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
        for e in &self.entities {
            match e {
                SketchEntity::Line { a, b, construction: false, reference: false } => {
                    if let (Some(pa), Some(pb)) = (self.points.get(*a), self.points.get(*b)) {
                        segs.push(([pa.x, pa.y], [pb.x, pb.y]));
                    }
                }
                SketchEntity::Circle { center, radius, construction: false } => {
                    if let Some(c) = self.points.get(*center) {
                        const SEG: usize = 64;
                        let mut prev = [c.x + radius, c.y];
                        for k in 1..=SEG {
                            let ang = std::f64::consts::TAU * k as f64 / SEG as f64;
                            let p = [c.x + radius * ang.cos(), c.y + radius * ang.sin()];
                            segs.push((prev, p));
                            prev = p;
                        }
                    }
                }
                SketchEntity::Spline { points, closed, construction: false, control } => {
                    let pts: Vec<[f64; 2]> =
                        points.iter().filter_map(|&i| self.points.get(i)).map(|p| [p.x, p.y]).collect();
                    if pts.len() >= 2 {
                        let poly = tessellate_spline(&pts, *closed, *control);
                        for w in poly.windows(2) {
                            segs.push((w[0], w[1]));
                        }
                        if *closed && poly.len() >= 2 {
                            segs.push((poly[poly.len() - 1], poly[0]));
                        }
                    }
                }
                SketchEntity::Slot { a, b, radius, construction: false, mid } => {
                    let pm = mid.and_then(|m| self.points.get(m)).map(|p| [p.x, p.y]);
                    if let (Some(pa), Some(pb)) = (self.points.get(*a), self.points.get(*b)) {
                        let poly = match pm {
                            Some(pm) => tessellate_arc_slot([pa.x, pa.y], pm, [pb.x, pb.y], *radius),
                            None => tessellate_slot([pa.x, pa.y], [pb.x, pb.y], *radius),
                        };
                        for w in poly.windows(2) {
                            segs.push((w[0], w[1]));
                        }
                        if poly.len() >= 2 {
                            segs.push((poly[poly.len() - 1], poly[0]));
                        }
                    }
                }
                SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. } => {
                    if let Some(o) = self.points.get(*origin) {
                        for loop_ in text_contours([o.x, o.y], contours, *height, *rotation, *mirror, *arc) {
                            for w in loop_.windows(2) {
                                segs.push((w[0], w[1]));
                            }
                            if loop_.len() >= 2 {
                                segs.push((loop_[loop_.len() - 1], loop_[0]));
                            }
                        }
                    }
                }
                SketchEntity::Line { .. }
                | SketchEntity::Circle { .. }
                | SketchEntity::Point { .. }
                | SketchEntity::Spline { .. }
                | SketchEntity::Slot { .. } => {}
            }
        }
        if segs.len() < 3 {
            return Vec::new();
        }
        trace_minimal_faces(&split_at_intersections(&segs))
    }
}

/// The closed outline of a slot whose centre line is `cl` (≥2 points), half-width `r`:
/// the two offset sides joined by semicircular caps at the ends. Wound as one loop.
/// Works for any centre line, so a straight and an arc slot share this code.
fn slot_outline(cl: &[[f64; 2]], r: f64) -> Vec<[f64; 2]> {
    let n = cl.len();
    if n < 2 || r <= 0.0 {
        return Vec::new();
    }
    let pi = std::f64::consts::PI;
    let tangent = |i: usize| -> [f64; 2] {
        let (p, q) = if i == 0 {
            (cl[0], cl[1])
        } else if i == n - 1 {
            (cl[n - 2], cl[n - 1])
        } else {
            (cl[i - 1], cl[i + 1])
        };
        let d = [q[0] - p[0], q[1] - p[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if len > 1e-9 { [d[0] / len, d[1] / len] } else { [1.0, 0.0] }
    };
    let perp = |t: [f64; 2]| [-t[1], t[0]];
    const CAP: usize = 16;
    let mut out = Vec::new();
    // Outer side (centre + perp·r), forward.
    for i in 0..n {
        let pp = perp(tangent(i));
        out.push([cl[i][0] + pp[0] * r, cl[i][1] + pp[1] * r]);
    }
    // Cap at the end (sweeps past cl[n-1] in the tangent direction).
    let pe = perp(tangent(n - 1));
    let a0 = pe[1].atan2(pe[0]);
    for k in 1..CAP {
        let t = a0 - pi * (k as f64 / CAP as f64);
        out.push([cl[n - 1][0] + r * t.cos(), cl[n - 1][1] + r * t.sin()]);
    }
    // Inner side (centre − perp·r), backward.
    for i in (0..n).rev() {
        let pp = perp(tangent(i));
        out.push([cl[i][0] - pp[0] * r, cl[i][1] - pp[1] * r]);
    }
    // Cap at the start.
    let ps = perp(tangent(0));
    let a1 = ps[1].atan2(ps[0]) + pi;
    for k in 1..CAP {
        let t = a1 - pi * (k as f64 / CAP as f64);
        out.push([cl[0][0] + r * t.cos(), cl[0][1] + r * t.sin()]);
    }
    out
}

/// Outline of a straight slot (stadium) centred on segment `a`→`b`, half-width `r`.
pub fn tessellate_slot(a: [f64; 2], b: [f64; 2], r: f64) -> Vec<[f64; 2]> {
    slot_outline(&[a, b], r)
}

/// Outline of an arc slot whose centre line is the circular arc through `a`, `p`, `b`
/// (falling back to a straight slot if those three points are collinear).
pub fn tessellate_arc_slot(a: [f64; 2], p: [f64; 2], b: [f64; 2], r: f64) -> Vec<[f64; 2]> {
    slot_outline(&arc_through(a, p, b, 40), r)
}

/// Tessellate the circular arc through `a`, `p`, `b` (in order) into `steps`+1 points.
/// Returns `[a, b]` if the three points are (nearly) collinear.
fn arc_through(a: [f64; 2], p: [f64; 2], b: [f64; 2], steps: usize) -> Vec<[f64; 2]> {
    let d = 2.0 * (a[0] * (p[1] - b[1]) + p[0] * (b[1] - a[1]) + b[0] * (a[1] - p[1]));
    if d.abs() < 1e-9 {
        return vec![a, b];
    }
    let (a2, p2, b2) = (a[0] * a[0] + a[1] * a[1], p[0] * p[0] + p[1] * p[1], b[0] * b[0] + b[1] * b[1]);
    let cx = (a2 * (p[1] - b[1]) + p2 * (b[1] - a[1]) + b2 * (a[1] - p[1])) / d;
    let cy = (a2 * (b[0] - p[0]) + p2 * (a[0] - b[0]) + b2 * (p[0] - a[0])) / d;
    let rad = ((a[0] - cx).powi(2) + (a[1] - cy).powi(2)).sqrt();
    let ang = |q: [f64; 2]| (q[1] - cy).atan2(q[0] - cx);
    let (ta, tp, tb) = (ang(a), ang(p), ang(b));
    let tau = std::f64::consts::TAU;
    let norm = |x: f64| {
        let mut y = x % tau;
        if y < 0.0 {
            y += tau;
        }
        y
    };
    // Sweep CCW if P lies on the CCW arc from A to B, else CW (the long way round).
    let sweep = if norm(tp - ta) <= norm(tb - ta) { norm(tb - ta) } else { norm(tb - ta) - tau };
    (0..=steps)
        .map(|k| {
            let t = ta + sweep * (k as f64 / steps as f64);
            [cx + rad * t.cos(), cy + rad * t.sin()]
        })
        .collect()
}

/// Tessellate a spline into a polyline. `control == false` ⇒ interpolating Catmull-Rom
/// (passes through the points); `control == true` ⇒ approximating uniform cubic B-spline
/// (open ends are clamped to the first/last control point). `closed` wraps it into a loop.
pub fn tessellate_spline(pts: &[[f64; 2]], closed: bool, control: bool) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec(); // 0/1/2 points ⇒ point or straight segment
    }
    const STEPS: usize = 16;
    let lerp4 = |w: [f64; 4], q: [[f64; 2]; 4]| {
        [
            w[0] * q[0][0] + w[1] * q[1][0] + w[2] * q[2][0] + w[3] * q[3][0],
            w[0] * q[0][1] + w[1] * q[1][1] + w[2] * q[2][1] + w[3] * q[3][1],
        ]
    };
    let mut out = Vec::new();
    if control {
        // Build the control sequence: closed wraps; open clamps the ends (so the curve
        // actually touches the first and last control points).
        let cps: Vec<[f64; 2]> = if closed {
            pts.to_vec()
        } else {
            let mut v = vec![pts[0], pts[0]];
            v.extend_from_slice(pts);
            v.push(pts[n - 1]);
            v.push(pts[n - 1]);
            v
        };
        let m = cps.len();
        let segs = if closed { m } else { m - 3 };
        for s in 0..segs {
            let q = [cps[s % m], cps[(s + 1) % m], cps[(s + 2) % m], cps[(s + 3) % m]];
            for t in 0..STEPS {
                let u = t as f64 / STEPS as f64;
                let (u2, u3) = (u * u, u * u * u);
                let w = [
                    (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
                    (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
                    (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
                    u3 / 6.0,
                ];
                out.push(lerp4(w, q));
            }
        }
    } else {
        // Catmull-Rom through the points.
        let get = |i: isize| -> [f64; 2] {
            if closed {
                pts[(((i % n as isize) + n as isize) % n as isize) as usize]
            } else {
                pts[i.clamp(0, n as isize - 1) as usize]
            }
        };
        let segs = if closed { n } else { n - 1 };
        for s in 0..segs {
            let q = [get(s as isize - 1), get(s as isize), get(s as isize + 1), get(s as isize + 2)];
            for t in 0..STEPS {
                let u = t as f64 / STEPS as f64;
                let (u2, u3) = (u * u, u * u * u);
                let w = [
                    0.5 * (-u + 2.0 * u2 - u3),
                    0.5 * (2.0 - 5.0 * u2 + 3.0 * u3),
                    0.5 * (u + 4.0 * u2 - 3.0 * u3),
                    0.5 * (-u2 + u3),
                ];
                out.push(lerp4(w, q));
            }
        }
        if !closed {
            out.push(pts[n - 1]);
        }
    }
    out
}

/// Absolute area of a 2D polygon.
fn area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    (a * 0.5).abs()
}

/// Average of a loop's points.
fn centroid(poly: &[[f64; 2]]) -> [f64; 2] {
    if poly.is_empty() {
        return [0.0, 0.0];
    }
    let (mut x, mut y) = (0.0, 0.0);
    for p in poly {
        x += p[0];
        y += p[1];
    }
    [x / poly.len() as f64, y / poly.len() as f64]
}

/// Signed area of a 2D polygon (positive ⇒ counter-clockwise winding).
fn signed_area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

/// Linear interpolation between two 2D points.
fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Distance between two 2D points.
fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// If segment p1→p2 meets segment p3→p4 at a single point, return the parameter
/// t ∈ [0,1] of that point along p1→p2. Parallel/collinear segments give `None`.
fn intersect_param(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> Option<f64> {
    let (rx, ry) = (p2[0] - p1[0], p2[1] - p1[1]);
    let (sx, sy) = (p4[0] - p3[0], p4[1] - p3[1]);
    let denom = rx * sy - ry * sx;
    if denom.abs() < 1e-12 {
        return None; // parallel or collinear
    }
    let (qx, qy) = (p3[0] - p1[0], p3[1] - p1[1]);
    let t = (qx * sy - qy * sx) / denom;
    let u = (qx * ry - qy * rx) / denom;
    let eps = 1e-9;
    ((-eps..=1.0 + eps).contains(&t) && (-eps..=1.0 + eps).contains(&u)).then(|| t.clamp(0.0, 1.0))
}

/// Split every segment at all its intersections with the others, so the result is
/// a planar set of segments that only meet at shared endpoints.
fn split_at_intersections(segs: &[([f64; 2], [f64; 2])]) -> Vec<([f64; 2], [f64; 2])> {
    let mut out = Vec::new();
    for (i, (a, b)) in segs.iter().enumerate() {
        let mut ts = vec![0.0, 1.0];
        for (j, (c, d)) in segs.iter().enumerate() {
            if i != j {
                if let Some(t) = intersect_param(*a, *b, *c, *d) {
                    ts.push(t);
                }
            }
        }
        ts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut prev = ts[0];
        for &t in &ts[1..] {
            if t - prev > 1e-7 {
                let (p, q) = (lerp2(*a, *b, prev), lerp2(*a, *b, t));
                if dist2(p, q) > 1e-7 {
                    out.push((p, q));
                }
                prev = t;
            }
        }
    }
    out
}

/// Trace the bounded minimal faces of a planar arrangement of (already split)
/// segments. Each face is returned as a simple CCW polygon; the unbounded face is
/// discarded. This is what lets the user pick the individual closed areas formed
/// by intersecting sketch geometry.
fn trace_minimal_faces(segs: &[([f64; 2], [f64; 2])]) -> Vec<Vec<[f64; 2]>> {
    use std::collections::{HashMap, HashSet};
    let key = |p: [f64; 2]| ((p[0] * 1.0e6).round() as i64, (p[1] * 1.0e6).round() as i64);
    let mut ids: HashMap<(i64, i64), usize> = HashMap::new();
    let mut pos: Vec<[f64; 2]> = Vec::new();

    // Build directed half-edges (both directions of each undirected edge).
    let (mut he_from, mut he_to): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for (a, b) in segs {
        let va = *ids.entry(key(*a)).or_insert_with(|| {
            pos.push(*a);
            pos.len() - 1
        });
        let vb = *ids.entry(key(*b)).or_insert_with(|| {
            pos.push(*b);
            pos.len() - 1
        });
        if va == vb {
            continue;
        }
        for (u, v) in [(va, vb), (vb, va)] {
            if seen.insert((u, v)) {
                he_from.push(u);
                he_to.push(v);
            }
        }
    }
    let nhe = he_from.len();
    if nhe == 0 {
        return Vec::new();
    }

    let angle = |h: usize| {
        let (f, t) = (pos[he_from[h]], pos[he_to[h]]);
        (t[1] - f[1]).atan2(t[0] - f[0])
    };
    // Outgoing half-edges per vertex, sorted by angle; plus a twin lookup.
    let mut out_he: Vec<Vec<usize>> = vec![Vec::new(); pos.len()];
    let mut twin: HashMap<(usize, usize), usize> = HashMap::new();
    for h in 0..nhe {
        out_he[he_from[h]].push(h);
        twin.insert((he_from[h], he_to[h]), h);
    }
    for v in 0..pos.len() {
        out_he[v].sort_by(|&x, &y| angle(x).partial_cmp(&angle(y)).unwrap());
    }
    // next(h): arriving at h's target, take the outgoing edge immediately clockwise
    // of the twin — which traces each bounded face counter-clockwise.
    let next_of = |h: usize| -> usize {
        let v = he_to[h];
        let tw = twin[&(v, he_from[h])];
        let ring = &out_he[v];
        let idx = ring.iter().position(|&x| x == tw).unwrap();
        ring[(idx + ring.len() - 1) % ring.len()]
    };

    let mut visited = vec![false; nhe];
    let mut faces = Vec::new();
    for start in 0..nhe {
        if visited[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut h = start;
        loop {
            visited[h] = true;
            cycle.push(pos[he_from[h]]);
            h = next_of(h);
            if h == start || cycle.len() > nhe + 1 {
                break;
            }
        }
        if cycle.len() >= 3 && signed_area(&cycle) > 1e-9 {
            faces.push(cycle); // bounded (CCW) face; the outer face is CW → dropped
        }
    }
    faces
}

/// Point indices an entity references.
fn entity_point_indices(e: &SketchEntity) -> Vec<usize> {
    match e {
        SketchEntity::Line { a, b, .. } => vec![*a, *b],
        SketchEntity::Circle { center, .. } => vec![*center],
        SketchEntity::Point { at } => vec![*at],
        SketchEntity::Spline { points, .. } => points.clone(),
        SketchEntity::Slot { a, b, mid, .. } => {
            let mut v = vec![*a, *b];
            v.extend(mid.iter().copied());
            v
        }
        SketchEntity::Text { origin, .. } => vec![*origin],
    }
}

/// Point indices a constraint references.
fn constraint_point_indices(c: &Constraint) -> Vec<usize> {
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

/// Remap an entity's point indices through `m`.
fn remap_entity(e: &mut SketchEntity, m: &[usize]) {
    match e {
        SketchEntity::Line { a, b, .. } => {
            *a = m[*a];
            *b = m[*b];
        }
        SketchEntity::Circle { center, .. } => *center = m[*center],
        SketchEntity::Point { at } => *at = m[*at],
        SketchEntity::Spline { points, .. } => {
            for p in points.iter_mut() {
                *p = m[*p];
            }
        }
        SketchEntity::Slot { a, b, mid, .. } => {
            *a = m[*a];
            *b = m[*b];
            if let Some(p) = mid {
                *p = m[*p];
            }
        }
        SketchEntity::Text { origin, .. } => *origin = m[*origin],
    }
}

/// Remap a constraint's point indices through `m`.
fn remap_constraint(c: &mut Constraint, m: &[usize]) {
    match c {
        Constraint::Coincident(a, b)
        | Constraint::Horizontal(a, b)
        | Constraint::Vertical(a, b)
        | Constraint::Distance { a, b, .. }
        | Constraint::EqualRadius { a, b } => {
            *a = m[*a];
            *b = m[*b];
        }
        Constraint::Midpoint { mid, a, b } => {
            *mid = m[*mid];
            *a = m[*a];
            *b = m[*b];
        }
        Constraint::Parallel(a, b, c, d)
        | Constraint::Perpendicular(a, b, c, d)
        | Constraint::Equal(a, b, c, d) => {
            *a = m[*a];
            *b = m[*b];
            *c = m[*c];
            *d = m[*d];
        }
        Constraint::Tangent { a, b, center, .. } => {
            *a = m[*a];
            *b = m[*b];
            *center = m[*center];
        }
        Constraint::Radius { center, .. } => *center = m[*center],
        Constraint::Angle { a, b, c, d, .. } => {
            *a = m[*a];
            *b = m[*b];
            *c = m[*c];
            *d = m[*d];
        }
        Constraint::PointLineDistance { p, a, b, .. } => {
            *p = m[*p];
            *a = m[*a];
            *b = m[*b];
        }
        Constraint::PointOnCircle { p, center } => {
            *p = m[*p];
            *center = m[*center];
        }
        Constraint::PointOnLine { p, a, b } => {
            *p = m[*p];
            *a = m[*a];
            *b = m[*b];
        }
        Constraint::PointOnArc { p, .. } => *p = m[*p],
    }
}

/// Even-odd ray-cast point-in-polygon test.
pub fn point_in_poly(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if ((yi > p[1]) != (yj > p[1])) && (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// Constraint solver (M2)
//
// All point coordinates are unknowns x ∈ ℝ²ᴺ. Each constraint is a residual
// fᵢ(x) = 0. We drive ‖f‖ → 0 with damped Gauss-Newton (Levenberg-Marquardt),
// seeded from the current positions so the solution stays near where the user
// drew. Under-constrained DOF are harmless: the damping keeps unconstrained
// points near where they already are (which is exactly what dragging wants).
// See DESIGN.md §5.
// ---------------------------------------------------------------------------

impl Sketch {
    /// Solve all constraints, moving points to satisfy them.
    pub fn solve(&mut self) {
        self.solve_with_fixed(&[]);
    }

    /// Solve constraints while holding the listed points fixed (used for dragging:
    /// the dragged point is pinned to the cursor and everything else follows).
    pub fn solve_with_fixed(&mut self, fixed_points: &[usize]) {
        let n = self.points.len();
        let m = self.residual_len();
        if n == 0 || m == 0 {
            self.apply_equal_radius(); // radius relations don't need a point solve
            return;
        }
        let nvars = 2 * n;

        // Free-variable indices (everything except the pinned points' x/y). Points
        // flagged `fixed` (projected from the 3D body) are always pinned, on top of
        // any caller-supplied pins (e.g. the point being dragged).
        let mut is_fixed = vec![false; nvars];
        for (i, p) in self.points.iter().enumerate() {
            if p.fixed {
                is_fixed[2 * i] = true;
                is_fixed[2 * i + 1] = true;
            }
        }
        for &pi in fixed_points {
            if pi < n {
                is_fixed[2 * pi] = true;
                is_fixed[2 * pi + 1] = true;
            }
        }
        let free: Vec<usize> = (0..nvars).filter(|i| !is_fixed[*i]).collect();
        if free.is_empty() {
            return;
        }

        // Seed from current positions.
        let mut x = DVector::<f64>::zeros(nvars);
        for (i, p) in self.points.iter().enumerate() {
            x[2 * i] = p.x;
            x[2 * i + 1] = p.y;
        }

        let mut lambda = 1e-3_f64;
        for _ in 0..100 {
            let r = self.residuals(&x);
            let cost = r.norm_squared();
            if r.norm() < 1e-10 {
                break;
            }
            let jac = self.jacobian(&x, m);
            let jf = jac.select_columns(free.iter());
            let jtj = jf.transpose() * &jf;
            let jtr = jf.transpose() * &r;

            // Levenberg-Marquardt: grow lambda until a step reduces the cost.
            let mut improved = false;
            for _ in 0..24 {
                let mut a = jtj.clone();
                for i in 0..free.len() {
                    a[(i, i)] += lambda;
                }
                if let Some(chol) = a.cholesky() {
                    let dxf = chol.solve(&(-&jtr));
                    let mut x_new = x.clone();
                    for (k, &vi) in free.iter().enumerate() {
                        x_new[vi] += dxf[k];
                    }
                    if self.residuals(&x_new).norm_squared() < cost {
                        x = x_new;
                        lambda = (lambda * 0.5).max(1e-9);
                        improved = true;
                        break;
                    }
                }
                lambda *= 3.0;
                if lambda > 1e10 {
                    break;
                }
            }
            if !improved {
                break; // converged or stuck
            }
        }

        // Only accept the result if it's all finite — an over/under-constrained solve
        // can diverge to NaN/∞, and a NaN coordinate poisons everything downstream
        // (dimension values, radii) and even crashes egui's number widgets.
        if x.iter().all(|v| v.is_finite()) {
            for (i, p) in self.points.iter_mut().enumerate() {
                p.x = x[2 * i];
                p.y = x[2 * i + 1];
            }
        }

        self.apply_equal_radius();
    }

    /// Enforce radius relations after the point solve (radius isn't a solver
    /// variable): `Radius` sets a circle's radius directly; `EqualRadius` makes
    /// circle `b` adopt circle `a`'s radius.
    fn apply_equal_radius(&mut self) {
        // Radius dimensions first (they may drive an EqualRadius source).
        let radii: Vec<(usize, f64)> = self
            .constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::Radius { center, value, .. } => Some((*center, *value)),
                _ => None,
            })
            .collect();
        for (center_pt, value) in radii {
            for e in self.entities.iter_mut() {
                if let SketchEntity::Circle { center, radius, .. } = e {
                    if *center == center_pt {
                        *radius = value.max(1e-4);
                    }
                }
            }
        }
        let pairs: Vec<(usize, usize)> = self
            .constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::EqualRadius { a, b } => Some((*a, *b)),
                _ => None,
            })
            .collect();
        for (a, b) in pairs {
            let ra = self.entities.iter().find_map(|e| match e {
                SketchEntity::Circle { center, radius, .. } if *center == a => Some(*radius),
                _ => None,
            });
            if let Some(ra) = ra {
                for e in self.entities.iter_mut() {
                    if let SketchEntity::Circle { center, radius, .. } = e {
                        if *center == b {
                            *radius = ra;
                        }
                    }
                }
            }
        }
    }

    /// Total number of scalar residual equations across all constraints.
    fn residual_len(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| match c {
                Constraint::Coincident(..) => 2,
                Constraint::Midpoint { .. } => 2,
                Constraint::Horizontal(..)
                | Constraint::Vertical(..)
                | Constraint::Distance { .. }
                | Constraint::Parallel(..)
                | Constraint::Perpendicular(..)
                | Constraint::Equal(..)
                | Constraint::Tangent { .. }
                | Constraint::Angle { .. }
                | Constraint::PointLineDistance { .. }
                | Constraint::PointOnCircle { .. }
                | Constraint::PointOnLine { .. }
                | Constraint::PointOnArc { .. } => 1,
                // Enforced after the solve (radius isn't a point variable).
                Constraint::EqualRadius { .. } | Constraint::Radius { .. } => 0,
            })
            .sum()
    }

    /// Evaluate the residual vector f(x).
    fn residuals(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut r = DVector::zeros(self.residual_len());
        let mut k = 0;
        for c in &self.constraints {
            match c {
                Constraint::Coincident(a, b) => {
                    let (a, b) = (*a, *b);
                    r[k] = x[2 * a] - x[2 * b];
                    r[k + 1] = x[2 * a + 1] - x[2 * b + 1];
                    k += 2;
                }
                Constraint::Horizontal(a, b) => {
                    r[k] = x[2 * *a + 1] - x[2 * *b + 1];
                    k += 1;
                }
                Constraint::Vertical(a, b) => {
                    r[k] = x[2 * *a] - x[2 * *b];
                    k += 1;
                }
                Constraint::Distance { a, b, value, axis, .. } => {
                    let (a, b) = (*a, *b);
                    let dx = x[2 * a] - x[2 * b];
                    let dy = x[2 * a + 1] - x[2 * b + 1];
                    let measured = match axis {
                        DimAxis::Aligned => (dx * dx + dy * dy).sqrt(),
                        DimAxis::Horizontal => dx.abs(),
                        DimAxis::Vertical => dy.abs(),
                    };
                    r[k] = measured - *value;
                    k += 1;
                }
                Constraint::Midpoint { mid, a, b } => {
                    let (mid, a, b) = (*mid, *a, *b);
                    r[k] = x[2 * mid] - 0.5 * (x[2 * a] + x[2 * b]);
                    r[k + 1] = x[2 * mid + 1] - 0.5 * (x[2 * a + 1] + x[2 * b + 1]);
                    k += 2;
                }
                Constraint::Parallel(a, b, c, d) => {
                    let (d1x, d1y) = (x[2 * *b] - x[2 * *a], x[2 * *b + 1] - x[2 * *a + 1]);
                    let (d2x, d2y) = (x[2 * *d] - x[2 * *c], x[2 * *d + 1] - x[2 * *c + 1]);
                    r[k] = d1x * d2y - d1y * d2x; // cross product → 0 when parallel
                    k += 1;
                }
                Constraint::Perpendicular(a, b, c, d) => {
                    let (d1x, d1y) = (x[2 * *b] - x[2 * *a], x[2 * *b + 1] - x[2 * *a + 1]);
                    let (d2x, d2y) = (x[2 * *d] - x[2 * *c], x[2 * *d + 1] - x[2 * *c + 1]);
                    r[k] = d1x * d2x + d1y * d2y; // dot product → 0 when perpendicular
                    k += 1;
                }
                Constraint::Equal(a, b, c, d) => {
                    let l1 = ((x[2 * *b] - x[2 * *a]).powi(2) + (x[2 * *b + 1] - x[2 * *a + 1]).powi(2)).sqrt();
                    let l2 = ((x[2 * *d] - x[2 * *c]).powi(2) + (x[2 * *d + 1] - x[2 * *c + 1]).powi(2)).sqrt();
                    r[k] = l1 - l2;
                    k += 1;
                }
                Constraint::Tangent { a, b, center, radius } => {
                    let (a, b, c) = (*a, *b, *center);
                    let (dx, dy) = (x[2 * b] - x[2 * a], x[2 * b + 1] - x[2 * a + 1]);
                    let len = (dx * dx + dy * dy).sqrt();
                    // Perpendicular distance from the circle centre to the line.
                    let cross = dx * (x[2 * c + 1] - x[2 * a + 1]) - dy * (x[2 * c] - x[2 * a]);
                    let dist = if len > 1e-9 { cross.abs() / len } else { 0.0 };
                    r[k] = dist - *radius;
                    k += 1;
                }
                Constraint::Angle { a, b, c, d, value, .. } => {
                    // Signed angle from line (a→b) to line (c→d) should equal `value`.
                    let (v1x, v1y) = (x[2 * *b] - x[2 * *a], x[2 * *b + 1] - x[2 * *a + 1]);
                    let (v2x, v2y) = (x[2 * *d] - x[2 * *c], x[2 * *d + 1] - x[2 * *c + 1]);
                    let cross = v1x * v2y - v1y * v2x;
                    let dot = v1x * v2x + v1y * v2y;
                    let ang = cross.atan2(dot);
                    let diff = ang - *value;
                    r[k] = diff.sin().atan2(diff.cos()); // wrap to (-π, π]
                    k += 1;
                }
                Constraint::PointLineDistance { p, a, b, value, .. } => {
                    // Perpendicular distance from point p to the line (a,b) → value.
                    let (px, py) = (x[2 * *p], x[2 * *p + 1]);
                    let (dx, dy) = (x[2 * *b] - x[2 * *a], x[2 * *b + 1] - x[2 * *a + 1]);
                    let len = (dx * dx + dy * dy).sqrt();
                    let cross = dx * (py - x[2 * *a + 1]) - dy * (px - x[2 * *a]);
                    let dist = if len > 1e-9 { cross.abs() / len } else { 0.0 };
                    r[k] = dist - *value;
                    k += 1;
                }
                Constraint::PointOnLine { p, a, b } => {
                    // Perpendicular distance from point p to the line (a,b) → zero.
                    let (px, py) = (x[2 * *p], x[2 * *p + 1]);
                    let (dx, dy) = (x[2 * *b] - x[2 * *a], x[2 * *b + 1] - x[2 * *a + 1]);
                    let len = (dx * dx + dy * dy).sqrt();
                    let cross = dx * (py - x[2 * *a + 1]) - dy * (px - x[2 * *a]);
                    r[k] = if len > 1e-9 { cross / len } else { 0.0 };
                    k += 1;
                }
                Constraint::PointOnArc { p, cx, cy, radius } => {
                    // Distance from p to the baked arc centre equals its radius.
                    let (ddx, ddy) = (x[2 * *p] - *cx, x[2 * *p + 1] - *cy);
                    r[k] = (ddx * ddx + ddy * ddy).sqrt() - *radius;
                    k += 1;
                }
                Constraint::PointOnCircle { p, center } => {
                    // Distance from p to the centre equals the circle's current radius.
                    let dx = x[2 * *p] - x[2 * *center];
                    let dy = x[2 * *p + 1] - x[2 * *center + 1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    // Radius is a parameter (read from the entity), not a point variable.
                    let radius = self
                        .entities
                        .iter()
                        .find_map(|e| match e {
                            SketchEntity::Circle { center: cc, radius, .. } if cc == center => Some(*radius),
                            _ => None,
                        })
                        .unwrap_or(dist); // circle gone → no-op residual
                    r[k] = dist - radius;
                    k += 1;
                }
                // Not point residuals — enforced separately after the solve.
                Constraint::EqualRadius { .. } | Constraint::Radius { .. } => {}
            }
        }
        r
    }

    /// Finite-difference Jacobian (m residuals × 2N variables). Constraints here
    /// are simple, so numeric differentiation is accurate and fast enough.
    fn jacobian(&self, x: &DVector<f64>, m: usize) -> DMatrix<f64> {
        let nvars = x.len();
        let eps = 1e-7;
        let r0 = self.residuals(x);
        let mut jac = DMatrix::zeros(m, nvars);
        let mut xp = x.clone();
        for j in 0..nvars {
            let old = xp[j];
            xp[j] = old + eps;
            let r1 = self.residuals(&xp);
            xp[j] = old;
            for i in 0..m {
                jac[(i, j)] = (r1[i] - r0[i]) / eps;
            }
        }
        jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_constraint_pulls_points_apart() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(1.0, 0.0);
        s.constraints.push(Constraint::Distance { a, b, value: 5.0, offset: 0.5, axis: DimAxis::Aligned });
        s.solve();
        let d = ((s.points[a].x - s.points[b].x).powi(2)
            + (s.points[a].y - s.points[b].y).powi(2))
        .sqrt();
        assert!((d - 5.0).abs() < 1e-6, "distance was {d}");
    }

    #[test]
    fn fixed_points_stay_put_through_solve() {
        // A locked (body-projected) point must not move even under a distance pull.
        let mut s = Sketch::default();
        let a = s.add_fixed_point(0.0, 0.0);
        let b = s.add_point(2.0, 0.0);
        s.constraints.push(Constraint::Distance { a, b, value: 10.0, offset: 0.5, axis: DimAxis::Aligned });
        s.solve();
        assert!((s.points[a].x).abs() < 1e-9 && (s.points[a].y).abs() < 1e-9, "fixed point moved");
        let d = ((s.points[a].x - s.points[b].x).powi(2) + (s.points[a].y - s.points[b].y).powi(2)).sqrt();
        assert!((d - 10.0).abs() < 1e-4, "distance not met: {d}");
    }

    #[test]
    fn point_on_circle_follows_the_rim() {
        // A point off the rim is pulled onto it; growing the radius keeps it on.
        let mut s = Sketch::default();
        let center = s.add_fixed_point(0.0, 0.0);
        s.add_circle(center, 5.0);
        let p = s.add_point(4.0, 0.0); // inside the rim
        s.constraints.push(Constraint::PointOnCircle { p, center });
        s.solve();
        let d = ((s.points[p].x).powi(2) + (s.points[p].y).powi(2)).sqrt();
        assert!((d - 5.0).abs() < 1e-3, "point not on rim: {d}");
        // Grow the circle and re-solve — the point should ride out to the new radius.
        if let Some(SketchEntity::Circle { radius, .. }) = s.entities.get_mut(0) {
            *radius = 8.0;
        }
        s.solve();
        let d2 = ((s.points[p].x).powi(2) + (s.points[p].y).powi(2)).sqrt();
        assert!((d2 - 8.0).abs() < 1e-3, "point didn't follow radius: {d2}");
    }

    #[test]
    fn slot_outline_is_a_stadium() {
        // Horizontal slot: centres (0,0)–(10,0), half-width 2. The outline should span
        // x∈[-2,12], y∈[-2,2], and every point sits exactly 2 from the centre line.
        let poly = tessellate_slot([0.0, 0.0], [10.0, 0.0], 2.0);
        assert!(poly.len() > 8, "slot outline too coarse");
        let (mut xmin, mut xmax, mut ymax) = (f64::MAX, f64::MIN, f64::MIN);
        for p in &poly {
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymax = ymax.max(p[1].abs());
            // Distance to the segment [0,10]×{0}: clamp x to [0,10], then dist.
            let cx = p[0].clamp(0.0, 10.0);
            let d = ((p[0] - cx).powi(2) + p[1].powi(2)).sqrt();
            assert!((d - 2.0).abs() < 1e-6, "slot point {p:?} not on the r=2 boundary (d={d})");
        }
        assert!((xmin + 2.0).abs() < 1e-6 && (xmax - 12.0).abs() < 1e-6, "slot x-extent wrong");
        assert!((ymax - 2.0).abs() < 1e-6, "slot y-extent wrong");
    }

    #[test]
    fn catmull_rom_spline_passes_through_its_points() {
        let pts = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 3.0]];
        let poly = tessellate_spline(&pts, false, false);
        // Every input point must appear (closely) on the interpolating curve.
        for p in &pts {
            let near = poly.iter().any(|q| (q[0] - p[0]).abs() < 1e-6 && (q[1] - p[1]).abs() < 1e-6);
            assert!(near, "through-points spline missed {p:?}");
        }
    }

    #[test]
    fn bspline_stays_inside_its_control_hull() {
        // A control-point spline should NOT pass through the interior control points.
        let pts = [[0.0, 0.0], [1.0, 5.0], [2.0, 0.0]];
        let poly = tessellate_spline(&pts, false, true);
        let peak = poly.iter().map(|q| q[1]).fold(0.0_f64, f64::max);
        assert!(peak < 5.0, "control-point curve overshot the hull (peak {peak})");
    }

    #[test]
    fn point_on_line_snaps_onto_the_edge() {
        // A fixed reference edge along y=0; a free point off it is pulled onto the line.
        let mut s = Sketch::default();
        let a = s.add_fixed_point(0.0, 0.0);
        let b = s.add_fixed_point(10.0, 0.0);
        let p = s.add_point(3.0, 4.0); // 4 above the edge
        s.constraints.push(Constraint::PointOnLine { p, a, b });
        s.solve();
        assert!((s.points[p].y).abs() < 1e-4, "point not on edge: y={}", s.points[p].y);
    }

    #[test]
    fn point_line_distance_drives_a_gap() {
        // A horizontal reference edge along y=0; drive a free point to 5 above it.
        let mut s = Sketch::default();
        let a = s.add_fixed_point(0.0, 0.0);
        let b = s.add_fixed_point(4.0, 0.0);
        let p = s.add_point(2.0, 1.0);
        s.constraints.push(Constraint::PointLineDistance { p, a, b, value: 5.0, offset: 0.0 });
        s.solve();
        let dy = (s.points[p].y).abs();
        assert!((dy - 5.0).abs() < 1e-3, "point-line gap was {dy}");
    }

    #[test]
    fn horizontal_distance_ignores_vertical_gap() {
        // A projected (horizontal) dimension drives only the x-extent.
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(1.0, 4.0);
        s.constraints.push(Constraint::Distance { a, b, value: 10.0, offset: 0.5, axis: DimAxis::Horizontal });
        s.solve();
        let dx = (s.points[a].x - s.points[b].x).abs();
        assert!((dx - 10.0).abs() < 1e-4, "horizontal extent was {dx}");
    }

    #[test]
    fn angle_constraint_opens_two_lines() {
        // Two lines sharing a vertex; drive the angle between them to 90°.
        let mut s = Sketch::default();
        let o = s.add_point(0.0, 0.0);
        let b = s.add_point(2.0, 0.0); // line 1: o→b along +x
        let c = s.add_point(0.0, 0.0); // line 2 starts near the vertex
        let d = s.add_point(2.0, 0.3); // nearly along +x → solver opens it up
        s.constraints.push(Constraint::Coincident(o, c));
        s.constraints.push(Constraint::Angle {
            a: o,
            b,
            c,
            d,
            value: std::f64::consts::FRAC_PI_2,
            offset: 1.0,
        });
        s.solve();
        let v1 = (s.points[b].x - s.points[o].x, s.points[b].y - s.points[o].y);
        let v2 = (s.points[d].x - s.points[c].x, s.points[d].y - s.points[c].y);
        let ang = (v1.0 * v2.1 - v1.1 * v2.0).atan2(v1.0 * v2.0 + v1.1 * v2.1);
        assert!((ang.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-2, "angle was {}", ang.to_degrees());
    }

    #[test]
    fn horizontal_constraint_levels_a_line() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(3.0, 2.0);
        s.constraints.push(Constraint::Horizontal(a, b));
        s.solve();
        assert!((s.points[a].y - s.points[b].y).abs() < 1e-6);
    }

    #[test]
    fn closed_loop_found_for_a_rectangle_and_not_for_an_open_path() {
        let mut s = Sketch::default();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(2.0, 0.0);
        let p2 = s.add_point(2.0, 1.0);
        let p3 = s.add_point(0.0, 1.0);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        // Open path so far: three of four edges — no loop yet.
        assert!(s.closed_loop().is_none());
        // Close it.
        s.add_line(p3, p0, false);
        let loop_idx = s.closed_loop().expect("rectangle should be a closed loop");
        assert_eq!(loop_idx.len(), 4);
        // A construction line must not create a loop on its own.
        let mut c = Sketch::default();
        let a = c.add_point(0.0, 0.0);
        let b = c.add_point(1.0, 0.0);
        let d = c.add_point(1.0, 1.0);
        c.add_line(a, b, true);
        c.add_line(b, d, true);
        c.add_line(d, a, true);
        assert!(c.closed_loop().is_none(), "construction-only loop must be ignored");
    }

    #[test]
    fn rectangle_tool_geometry_yields_a_closed_profile() {
        // Mirror exactly how the app's Rectangle tool builds geometry.
        let mut s = Sketch::default();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(3.0, 0.0);
        let p2 = s.add_point(3.0, 2.0);
        let p3 = s.add_point(0.0, 2.0);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        s.add_line(p3, p0, false);
        let prof = s.outer_profile().expect("rectangle is a closed profile");
        assert_eq!(prof.len(), 4);
    }

    #[test]
    fn a_lone_circle_yields_a_closed_profile() {
        let mut s = Sketch::default();
        let c = s.add_point(1.0, 1.0);
        s.add_circle(c, 2.0);
        // closed_loop sees no lines, but outer_profile tessellates the circle.
        assert!(s.closed_loop().is_none());
        let prof = s.outer_profile().expect("circle is a closed profile");
        assert_eq!(prof.len(), 64);
        // Every profile point sits ~2.0 from the centre.
        for p in &prof {
            let d = ((p[0] - 1.0).powi(2) + (p[1] - 1.0).powi(2)).sqrt();
            assert!((d - 2.0).abs() < 1e-9, "radius was {d}");
        }
    }

    fn add_rect(s: &mut Sketch, x0: f64, y0: f64, x1: f64, y1: f64) {
        let p0 = s.add_point(x0, y0);
        let p1 = s.add_point(x1, y0);
        let p2 = s.add_point(x1, y1);
        let p3 = s.add_point(x0, y1);
        s.add_line(p0, p1, false);
        s.add_line(p1, p2, false);
        s.add_line(p2, p3, false);
        s.add_line(p3, p0, false);
    }

    #[test]
    fn perpendicular_makes_two_lines_meet_at_a_right_angle() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(2.0, 0.0); // line 1 along +x
        let c = s.add_point(0.0, 0.0);
        let d = s.add_point(1.0, 0.5); // line 2 at a shallow angle
        s.constraints.push(Constraint::Perpendicular(a, b, c, d));
        s.solve();
        let (d1x, d1y) = (s.points[b].x - s.points[a].x, s.points[b].y - s.points[a].y);
        let (d2x, d2y) = (s.points[d].x - s.points[c].x, s.points[d].y - s.points[c].y);
        assert!((d1x * d2x + d1y * d2y).abs() < 1e-6, "dot should be ~0");
    }

    #[test]
    fn equal_makes_two_lines_the_same_length() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(4.0, 0.0); // length 4
        let c = s.add_point(0.0, 2.0);
        let d = s.add_point(1.0, 2.0); // length 1
        s.constraints.push(Constraint::Equal(a, b, c, d));
        s.solve();
        let l1 = ((s.points[b].x - s.points[a].x).powi(2) + (s.points[b].y - s.points[a].y).powi(2)).sqrt();
        let l2 = ((s.points[d].x - s.points[c].x).powi(2) + (s.points[d].y - s.points[c].y).powi(2)).sqrt();
        assert!((l1 - l2).abs() < 1e-6, "lengths {l1} vs {l2}");
    }

    #[test]
    fn tangent_pulls_a_line_to_touch_a_circle() {
        let mut s = Sketch::default();
        let center = s.add_point(0.0, 0.0);
        // A horizontal line at y = 3, should drop to y = 2 to be tangent to r=2.
        let a = s.add_point(-5.0, 3.0);
        let b = s.add_point(5.0, 3.0);
        s.constraints.push(Constraint::Horizontal(a, b));
        s.constraints.push(Constraint::Tangent { a, b, center, radius: 2.0 });
        s.solve();
        let (dx, dy) = (s.points[b].x - s.points[a].x, s.points[b].y - s.points[a].y);
        let len = (dx * dx + dy * dy).sqrt();
        let cross = dx * (s.points[center].y - s.points[a].y) - dy * (s.points[center].x - s.points[a].x);
        assert!((cross.abs() / len - 2.0).abs() < 1e-5, "distance to centre should be the radius");
    }

    #[test]
    fn remove_unused_points_drops_orphan_endpoints() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(5.0, 0.0);
        s.add_line(a, b, false);
        let c = s.add_point(2.0, 2.0); // a circle that shares no points with the line
        s.add_circle(c, 1.0);
        // Delete the line (leaving its two endpoints orphaned), then clean up.
        s.entities.remove(0);
        s.remove_unused_points();
        assert_eq!(s.points.len(), 1, "only the circle centre should remain");
        // The circle still resolves to a valid centre.
        assert!(matches!(s.entities[0], SketchEntity::Circle { center: 0, .. }));
    }

    #[test]
    fn equal_radius_makes_two_circles_match() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        s.add_circle(a, 2.0);
        let b = s.add_point(10.0, 0.0);
        s.add_circle(b, 5.0);
        s.constraints.push(Constraint::EqualRadius { a, b });
        s.solve();
        let rb = s
            .entities
            .iter()
            .find_map(|e| match e {
                SketchEntity::Circle { center, radius, .. } if *center == b => Some(*radius),
                _ => None,
            })
            .unwrap();
        assert!((rb - 2.0).abs() < 1e-9, "circle b should match circle a's radius, was {rb}");
    }

    #[test]
    fn nested_loops_become_a_region_with_a_hole() {
        let mut s = Sketch::default();
        add_rect(&mut s, 0.0, 0.0, 10.0, 10.0); // outer
        add_rect(&mut s, 3.0, 3.0, 7.0, 7.0); // inner → hole
        let regions = s.regions();
        assert_eq!(regions.len(), 1, "one region");
        assert_eq!(regions[0].holes.len(), 1, "with one hole");
    }

    #[test]
    fn two_separate_rectangles_are_two_regions() {
        let mut s = Sketch::default();
        add_rect(&mut s, 0.0, 0.0, 2.0, 2.0);
        add_rect(&mut s, 5.0, 0.0, 7.0, 2.0);
        let regions = s.regions();
        assert_eq!(regions.len(), 2);
        assert!(regions.iter().all(|r| r.holes.is_empty()));
    }

    #[test]
    fn a_diagonal_splits_a_rectangle_into_two_triangles() {
        let mut s = Sketch::default();
        add_rect(&mut s, 0.0, 0.0, 4.0, 4.0);
        let a = s.add_point(0.0, 0.0); // coincides with a corner → same arrangement vertex
        let b = s.add_point(4.0, 4.0);
        s.add_line(a, b, false);
        let regions = s.regions();
        assert_eq!(regions.len(), 2, "a diagonal cuts the square into two regions");
        assert!(regions.iter().all(|r| r.holes.is_empty()));
        // Each triangle is half the 16-unit square.
        for r in &regions {
            assert!((area(&r.outer) - 8.0).abs() < 1e-6, "triangle area {}", area(&r.outer));
        }
    }

    #[test]
    fn a_chord_splits_a_circle_into_two_regions() {
        let mut s = Sketch::default();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 2.0);
        let a = s.add_point(-3.0, 0.5); // horizontal chord crossing the disk
        let b = s.add_point(3.0, 0.5);
        s.add_line(a, b, false);
        assert_eq!(s.regions().len(), 2, "a chord cuts the disk into two areas");
    }

    #[test]
    fn two_circles_linked_by_lines_expose_the_middle_region() {
        let mut s = Sketch::default();
        let ca = s.add_point(0.0, 0.0);
        s.add_circle(ca, 3.0);
        let cb = s.add_point(10.0, 0.0);
        s.add_circle(cb, 3.0);
        // Two lines crossing both circles (secants) near the top and bottom.
        let a1 = s.add_point(-1.0, 2.0);
        let a2 = s.add_point(11.0, 2.0);
        s.add_line(a1, a2, false);
        let b1 = s.add_point(-1.0, -2.0);
        let b2 = s.add_point(11.0, -2.0);
        s.add_line(b1, b2, false);
        let regions = s.regions();
        let inside = |rs: &[Region], p: [f64; 2]| {
            rs.iter().position(|r| point_in_poly(p, &r.outer) && !r.holes.iter().any(|h| point_in_poly(p, h)))
        };
        eprintln!("SECANT: region count = {}", regions.len());
        eprintln!("  middle (5,0) -> {:?}, left circle (0,0) -> {:?}", inside(&regions, [5.0, 0.0]), inside(&regions, [0.0, 0.0]));

        // Tangent/external connectors: lines just touch the circles (endpoints on
        // the rim), so the circles stay whole — the case from the screenshot.
        let mut t = Sketch::default();
        let ta = t.add_point(0.0, 0.0);
        t.add_circle(ta, 3.0);
        let tb = t.add_point(10.0, 0.0);
        t.add_circle(tb, 3.0);
        let p1 = t.add_point(0.0, 3.0);
        let p2 = t.add_point(10.0, 3.0);
        t.add_line(p1, p2, false);
        let p3 = t.add_point(0.0, -3.0);
        let p4 = t.add_point(10.0, -3.0);
        t.add_line(p3, p4, false);
        let tr = t.regions();
        assert!(inside(&regions, [5.0, 0.0]).is_some(), "secant middle band should be a region");
        assert!(inside(&tr, [5.0, 0.0]).is_some(), "tangent middle band should be a region");
        assert!(inside(&tr, [0.0, 0.0]).is_some(), "each circle is still a region");
    }

    #[test]
    fn two_radii_carve_a_quarter_pie_from_a_circle() {
        let mut s = Sketch::default();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 2.0);
        // Two radii from the centre out past the rim → a quarter pie + the rest.
        let center = s.add_point(0.0, 0.0);
        let rx = s.add_point(3.0, 0.0);
        let ry = s.add_point(0.0, 3.0);
        s.add_line(center, rx, false);
        s.add_line(center, ry, false);
        let regions = s.regions();
        assert_eq!(regions.len(), 2, "two radii split the disk into a pie slice and the rest");
        let smallest = regions.iter().map(|r| area(&r.outer)).fold(f64::INFINITY, f64::min);
        let quarter = std::f64::consts::PI * 2.0 * 2.0 / 4.0; // πr²/4 with r=2
        assert!((smallest - quarter).abs() < 0.3, "pie area {smallest}, expected ~{quarter}");
    }

    #[test]
    fn dragging_a_corner_keeps_a_rectangle_square() {
        // p0-p1 bottom (H), p1-p2 right (V), p2-p3 top (H), p3-p0 left (V).
        let mut s = Sketch::default();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(2.0, 0.0);
        let p2 = s.add_point(2.0, 1.0);
        let p3 = s.add_point(0.0, 1.0);
        s.constraints.push(Constraint::Horizontal(p0, p1));
        s.constraints.push(Constraint::Horizontal(p3, p2));
        s.constraints.push(Constraint::Vertical(p1, p2));
        s.constraints.push(Constraint::Vertical(p0, p3));
        // Drag p2 to (4, 3); it stays pinned, the rest must keep right angles.
        s.points[p2] = Point2 { x: 4.0, y: 3.0, fixed: false };
        s.solve_with_fixed(&[p2]);
        assert!((s.points[p1].x - s.points[p2].x).abs() < 1e-6, "right edge not vertical");
        assert!((s.points[p3].y - s.points[p2].y).abs() < 1e-6, "top edge not horizontal");
        assert!((s.points[p0].y - s.points[p1].y).abs() < 1e-6, "bottom edge not horizontal");
        assert!((s.points[p0].x - s.points[p3].x).abs() < 1e-6, "left edge not vertical");
    }
}
