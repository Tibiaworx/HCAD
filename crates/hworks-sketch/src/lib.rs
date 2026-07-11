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
    /// A circular arc centred at `center`, sweeping from endpoint `a` to endpoint `b` (both on
    /// the rim, radius = |center→a|). `ccw` chooses the sweep direction so major/minor is
    /// unambiguous. Endpoints are real sketch points, so lines snap to them to close a profile.
    Arc {
        center: usize,
        a: usize,
        b: usize,
        #[serde(default)]
        ccw: bool,
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
    /// Driving width dimension for the slot whose centre line runs `a`→`b`: the distance
    /// across its parallel sides equals `value` (so its half-width = value/2). Enforced
    /// after the solve (the slot's radius isn't a point variable). `offset` is the display
    /// offset of the dimension line.
    SlotWidth { a: usize, b: usize, value: f64, offset: f64 },
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

/// A run of consecutive boundary edges that lies exactly on a circle: edges
/// `first_edge .. first_edge+count` (wrapping) of the loop's polyline, sampled
/// from the circle centred at `center` with `radius`. Carried alongside the
/// polyline so the geometry kernel can rebuild the run as a **true circular
/// arc** (a cylindrical face after sweeping) instead of `count` line facets.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct ArcSpan {
    pub first_edge: usize,
    pub count: usize,
    pub center: [f64; 2],
    pub radius: f64,
}

/// A closed area of a sketch, ready to extrude: one outer boundary loop plus any
/// inner loops that are *holes* in it (e.g. a rectangle with a circular hole).
///
/// Loops are polylines (`outer` / `holes`) — the universal form every consumer
/// (hit-testing, rendering, mesh booleans) understands — with [`ArcSpan`]
/// annotations recording which edge runs lie on exact circles, so the exact
/// B-rep path can rebuild those as true arcs.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Region {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
    /// Exact-arc runs of `outer`'s edges (may be empty: all straight lines).
    #[serde(default)]
    pub outer_arcs: Vec<ArcSpan>,
    /// Exact-arc runs per hole, parallel to `holes` (empty ⇒ no arc info).
    #[serde(default)]
    pub hole_arcs: Vec<Vec<ArcSpan>>,
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
    /// Ensure the sketch has its **origin anchor**: a fixed point at (0,0) with a
    /// `Point` entity, SolidWorks-style. Every constraint except the anchor is
    /// *relative* (a fully dimensioned rectangle can still slide and spin as a
    /// rigid body), so snapping/dimensioning to this point is what lets a sketch
    /// on a datum plane become fully defined. Idempotent; returns the point index.
    pub fn ensure_origin(&mut self) -> usize {
        if let Some(i) = self.origin_point() {
            return i;
        }
        let at = self.add_fixed_point(0.0, 0.0);
        self.entities.push(SketchEntity::Point { at });
        at
    }

    /// The origin anchor's point index, if the sketch has one: a `Point` entity
    /// whose point is fixed at exactly (0,0).
    pub fn origin_point(&self) -> Option<usize> {
        self.entities.iter().find_map(|e| match e {
            SketchEntity::Point { at } => {
                let p = self.points.get(*at)?;
                (p.fixed && p.x == 0.0 && p.y == 0.0).then_some(*at)
            }
            _ => None,
        })
    }

    /// True if the sketch holds any user geometry beyond the origin anchor —
    /// the "is there anything worth keeping?" test.
    pub fn has_geometry(&self) -> bool {
        let origin = self.origin_point();
        self.entities.iter().any(|e| match e {
            SketchEntity::Point { at } => Some(*at) != origin,
            _ => true,
        })
    }

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

    /// Add an arc about `center` from endpoint `a` to `b` (sweep `ccw`). Returns its index.
    pub fn add_arc(&mut self, center: usize, a: usize, b: usize, ccw: bool, construction: bool) -> usize {
        self.entities.push(SketchEntity::Arc { center, a, b, ccw, construction });
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
    /// Edge runs that came from a circle/arc entity are annotated as [`ArcSpan`]s
    /// so the exact kernel can rebuild them as true arcs.
    pub fn regions(&self) -> Vec<Region> {
        let faces = self.arrangement_faces();
        let n = faces.len();
        let contours: Vec<&Vec<[f64; 2]>> = faces.iter().map(|f| &f.0).collect();
        let spans: Vec<Vec<ArcSpan>> = faces.iter().map(|f| arc_spans(&f.1)).collect();
        let areas: Vec<f64> = contours.iter().map(|c| area(c)).collect();
        // `j` contains `i` if it's bigger and `i`'s centroid falls inside it.
        let contains = |j: usize, i: usize| {
            j != i && areas[j] > areas[i] && point_in_poly(centroid(contours[i]), contours[j])
        };
        let depth: Vec<usize> =
            (0..n).map(|i| (0..n).filter(|&j| contains(j, i)).count()).collect();

        let mut regions = Vec::new();
        for i in 0..n {
            if depth[i] % 2 != 0 {
                continue; // odd nesting depth ⇒ this face is a hole, not a solid
            }
            let hole_ids: Vec<usize> =
                (0..n).filter(|&k| depth[k] == depth[i] + 1 && contains(i, k)).collect();
            regions.push(Region {
                outer: contours[i].clone(),
                holes: hole_ids.iter().map(|&k| contours[k].clone()).collect(),
                outer_arcs: spans[i].clone(),
                hole_arcs: hole_ids.iter().map(|&k| spans[k].clone()).collect(),
            });
        }
        regions
    }

    /// Tessellate all non-construction geometry to straight segments, split them at
    /// every intersection, and trace the bounded minimal faces of the resulting
    /// planar arrangement. Each face is a simple polygon wound counter-clockwise,
    /// returned with a per-edge tag saying which circle/arc entity (if any) that
    /// edge was sampled from — the raw material for [`ArcSpan`] annotations.
    fn arrangement_faces(&self) -> Vec<(Vec<[f64; 2]>, Vec<Option<CurveTag>>)> {
        let mut segs: Vec<TagSeg> = Vec::new();
        for (ei, e) in self.entities.iter().enumerate() {
            match e {
                SketchEntity::Line { a, b, construction: false, reference: false } => {
                    if let (Some(pa), Some(pb)) = (self.points.get(*a), self.points.get(*b)) {
                        segs.push(([pa.x, pa.y], [pb.x, pb.y], None));
                    }
                }
                SketchEntity::Circle { center, radius, construction: false } => {
                    if let Some(c) = self.points.get(*center) {
                        let tag = Some(CurveTag { id: ei, center: [c.x, c.y], radius: *radius });
                        // Finer than the rendering tessellation: a circular profile becomes a
                        // prism/revolve whose facets must meet cleanly at a boolean intersection
                        // seam — too coarse and the seam shows tiny facet-mismatch gaps.
                        const SEG: usize = 128;
                        let tau = std::f64::consts::TAU;
                        let step = tau / SEG as f64;
                        // Any sketch point sitting on this rim (a line/arc endpoint snapped
                        // to it) must become a vertex of the tessellation — otherwise the
                        // connecting edge dangles against a chord and the area never closes.
                        // Using the point's *exact* coordinates makes the arrangement share
                        // the node, so the enclosed region traces cleanly and can extrude.
                        let on_tol = (radius * 5.0e-3).max(1.0e-4);
                        let mut samples: Vec<(f64, [f64; 2])> = Vec::new();
                        for (pi, p) in self.points.iter().enumerate() {
                            if pi == *center {
                                continue;
                            }
                            let (dx, dy) = (p.x - c.x, p.y - c.y);
                            let d = (dx * dx + dy * dy).sqrt();
                            if d > 1.0e-9 && (d - radius).abs() <= on_tol {
                                samples.push((dy.atan2(dx).rem_euclid(tau), [p.x, p.y]));
                            }
                        }
                        // Fill the rest of the circle with even samples, skipping any that
                        // land almost on a connection vertex (avoids zero-length slivers).
                        for k in 0..SEG {
                            let a = step * k as f64;
                            let near = samples.iter().any(|(ta, _)| {
                                let d = (a - ta).abs();
                                d.min(tau - d) < step * 0.4
                            });
                            if !near {
                                samples.push((a, [c.x + radius * a.cos(), c.y + radius * a.sin()]));
                            }
                        }
                        samples.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                        let ring: Vec<[f64; 2]> = samples.iter().map(|(_, p)| *p).collect();
                        for w in ring.windows(2) {
                            segs.push((w[0], w[1], tag));
                        }
                        if ring.len() >= 2 {
                            segs.push((ring[ring.len() - 1], ring[0], tag));
                        }
                    }
                }
                SketchEntity::Spline { points, closed, construction: false, control } => {
                    let pts: Vec<[f64; 2]> =
                        points.iter().filter_map(|&i| self.points.get(i)).map(|p| [p.x, p.y]).collect();
                    if pts.len() >= 2 {
                        let poly = tessellate_spline(&pts, *closed, *control);
                        for w in poly.windows(2) {
                            segs.push((w[0], w[1], None));
                        }
                        if *closed && poly.len() >= 2 {
                            segs.push((poly[poly.len() - 1], poly[0], None));
                        }
                    }
                }
                SketchEntity::Arc { center, a, b, ccw, construction: false } => {
                    if let (Some(c), Some(pa), Some(pb)) =
                        (self.points.get(*center), self.points.get(*a), self.points.get(*b))
                    {
                        let radius = ((pa.x - c.x).powi(2) + (pa.y - c.y).powi(2)).sqrt();
                        let tag = Some(CurveTag { id: ei, center: [c.x, c.y], radius });
                        let poly = tessellate_arc([c.x, c.y], [pa.x, pa.y], [pb.x, pb.y], *ccw);
                        for w in poly.windows(2) {
                            segs.push((w[0], w[1], tag)); // open arc — no closing chord
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
                            segs.push((w[0], w[1], None));
                        }
                        if poly.len() >= 2 {
                            segs.push((poly[poly.len() - 1], poly[0], None));
                        }
                    }
                }
                SketchEntity::Text { origin, contours, height, rotation, mirror, arc, .. } => {
                    if let Some(o) = self.points.get(*origin) {
                        for loop_ in text_contours([o.x, o.y], contours, *height, *rotation, *mirror, *arc) {
                            for w in loop_.windows(2) {
                                segs.push((w[0], w[1], None));
                            }
                            if loop_.len() >= 2 {
                                segs.push((loop_[loop_.len() - 1], loop_[0], None));
                            }
                        }
                    }
                }
                SketchEntity::Line { .. }
                | SketchEntity::Circle { .. }
                | SketchEntity::Arc { .. }
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

/// Tessellate a circular arc about `center` from endpoint `a` to endpoint `b`, sweeping `ccw`
/// (else CW). The polyline starts exactly at `a` and ends exactly at `b` so connecting edges
/// share those nodes.
pub fn tessellate_arc(center: [f64; 2], a: [f64; 2], b: [f64; 2], ccw: bool) -> Vec<[f64; 2]> {
    let r = ((a[0] - center[0]).powi(2) + (a[1] - center[1]).powi(2)).sqrt();
    if r < 1e-9 {
        return vec![a, b];
    }
    let tau = std::f64::consts::TAU;
    let ta = (a[1] - center[1]).atan2(a[0] - center[0]);
    let tb = (b[1] - center[1]).atan2(b[0] - center[0]);
    let mut span = if ccw { (tb - ta).rem_euclid(tau) } else { -((ta - tb).rem_euclid(tau)) };
    if span.abs() < 1e-9 {
        span = if ccw { tau } else { -tau }; // coincident endpoints ⇒ full circle
    }
    let n = ((span.abs() / tau * 64.0).ceil() as usize).max(2);
    let mut out = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let t = ta + span * (k as f64 / n as f64);
        out.push([center[0] + r * t.cos(), center[1] + r * t.sin()]);
    }
    out[0] = a;
    *out.last_mut().unwrap() = b;
    out
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

/// Which exact curve an arrangement segment was sampled from: the tessellating
/// entity's index plus the circle it lies on. Segments split at intersections
/// inherit their parent's tag, so a face edge always knows its source curve.
#[derive(Debug, Clone, Copy)]
struct CurveTag {
    id: usize,
    center: [f64; 2],
    radius: f64,
}

/// An arrangement segment: endpoints plus the source-curve tag (None ⇒ straight).
type TagSeg = ([f64; 2], [f64; 2], Option<CurveTag>);

/// Group a face loop's per-edge tags into maximal same-curve runs — the
/// [`ArcSpan`]s the kernel rebuilds as true arcs. Runs may wrap the loop seam;
/// a loop that is entirely one curve (a full circle) becomes a single span
/// covering every edge.
fn arc_spans(tags: &[Option<CurveTag>]) -> Vec<ArcSpan> {
    let n = tags.len();
    if n == 0 {
        return Vec::new();
    }
    let same = |x: &Option<CurveTag>, y: &Option<CurveTag>| match (x, y) {
        (Some(a), Some(b)) => a.id == b.id,
        _ => false,
    };
    // An edge whose tag differs from its predecessor starts a run. No boundary
    // at all ⇒ the whole loop is one curve.
    let start = (0..n).find(|&i| !same(&tags[(i + n - 1) % n], &tags[i]));
    let Some(start) = start else {
        return match tags[0] {
            Some(t) => vec![ArcSpan { first_edge: 0, count: n, center: t.center, radius: t.radius }],
            None => Vec::new(),
        };
    };
    let mut spans = Vec::new();
    let mut i = 0usize;
    while i < n {
        let e = (start + i) % n;
        match tags[e] {
            Some(t) => {
                let mut count = 1;
                while i + count < n && same(&tags[e], &tags[(start + i + count) % n]) {
                    count += 1;
                }
                spans.push(ArcSpan { first_edge: e, count, center: t.center, radius: t.radius });
                i += count;
            }
            None => i += 1,
        }
    }
    spans
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
/// a planar set of segments that only meet at shared endpoints. Split pieces
/// inherit the parent segment's source-curve tag.
fn split_at_intersections(segs: &[TagSeg]) -> Vec<TagSeg> {
    let mut out = Vec::new();
    for (i, (a, b, tag)) in segs.iter().enumerate() {
        let mut ts = vec![0.0, 1.0];
        for (j, (c, d, _)) in segs.iter().enumerate() {
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
                    out.push((p, q, *tag));
                }
                prev = t;
            }
        }
    }
    out
}

/// Trace the bounded minimal faces of a planar arrangement of (already split)
/// segments. Each face is returned as a simple CCW polygon plus a per-edge
/// source-curve tag (edge `i` runs vertex `i` → `i+1`); the unbounded face is
/// discarded. This is what lets the user pick the individual closed areas formed
/// by intersecting sketch geometry.
fn trace_minimal_faces(segs: &[TagSeg]) -> Vec<(Vec<[f64; 2]>, Vec<Option<CurveTag>>)> {
    use std::collections::{HashMap, HashSet};
    let key = |p: [f64; 2]| ((p[0] * 1.0e6).round() as i64, (p[1] * 1.0e6).round() as i64);
    let mut ids: HashMap<(i64, i64), usize> = HashMap::new();
    let mut pos: Vec<[f64; 2]> = Vec::new();

    // Build directed half-edges (both directions of each undirected edge).
    let (mut he_from, mut he_to): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut he_tag: Vec<Option<CurveTag>> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    for (a, b, tag) in segs {
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
                he_tag.push(*tag);
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
        let mut tags = Vec::new();
        let mut h = start;
        loop {
            visited[h] = true;
            cycle.push(pos[he_from[h]]);
            tags.push(he_tag[h]); // tag of the edge leaving this vertex
            h = next_of(h);
            if h == start || cycle.len() > nhe + 1 {
                break;
            }
        }
        if cycle.len() >= 3 && signed_area(&cycle) > 1e-9 {
            faces.push((cycle, tags)); // bounded (CCW) face; the outer face is CW → dropped
        }
    }
    faces
}

/// Point indices an entity references.
fn entity_point_indices(e: &SketchEntity) -> Vec<usize> {
    match e {
        SketchEntity::Line { a, b, .. } => vec![*a, *b],
        SketchEntity::Circle { center, .. } => vec![*center],
        SketchEntity::Arc { center, a, b, .. } => vec![*center, *a, *b],
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
        Constraint::SlotWidth { a, b, .. } => vec![*a, *b],
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
        SketchEntity::Arc { center, a, b, .. } => {
            *center = m[*center];
            *a = m[*a];
            *b = m[*b];
        }
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
        Constraint::SlotWidth { a, b, .. } => {
            *a = m[*a];
            *b = m[*b];
        }
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
// Point coordinates and circle radii are the unknowns x ∈ ℝ²ᴺ⁺ᴷ. Each
// constraint contributes residuals fᵢ(x) = 0 with hand-written analytic
// derivatives (each residual touches at most a handful of variables, so this
// is far cheaper than finite differencing). We drive ‖f‖ → 0 with damped
// Gauss-Newton (Levenberg-Marquardt), seeded from the current positions so
// the solution stays near where the user drew. Radii solve in two stages —
// pinned first, freed only if the constraints can't be met by moving points —
// so circles keep their size unless a resize is genuinely required (e.g. a
// tangent line whose endpoints are locked). Under-constrained DOF are
// harmless: the damping keeps unconstrained points near where they already
// are (which is exactly what dragging wants). See DESIGN.md §5.
// ---------------------------------------------------------------------------

/// Constraint state of a sketch, for SolidWorks-style status feedback.
#[derive(Debug, Clone, Default)]
pub struct DofReport {
    /// Degrees of freedom remaining (0 ⇒ fully defined). Each unconstrained
    /// point contributes 2; an undimensioned circle radius contributes 1.
    pub dof: usize,
    /// Constraints conflict — the solver cannot satisfy them all at once.
    pub over_defined: bool,
    /// Per-point: true ⇒ the point can still move (draw it "under-defined").
    pub free_points: Vec<bool>,
}

/// Layout of the solver's variable vector: point coordinates first
/// (x[2i], x[2i+1]), then one radius variable per circle entity — radii are
/// real unknowns, so tangency and driving-radius dimensions can co-solve with
/// positions instead of being patched in afterwards.
struct VarLayout {
    npoints: usize,
    /// (circle entity index, centre point index), in entity order; the k-th
    /// entry's radius lives at variable `2*npoints + k`.
    circles: Vec<(usize, usize)>,
}

impl VarLayout {
    fn new(sketch: &Sketch) -> Self {
        let circles = sketch
            .entities
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                SketchEntity::Circle { center, .. } => Some((i, *center)),
                _ => None,
            })
            .collect();
        VarLayout { npoints: sketch.points.len(), circles }
    }

    fn nvars(&self) -> usize {
        2 * self.npoints + self.circles.len()
    }

    /// Radius variable index for the circle centred at point `center` (the
    /// first such circle, matching how constraints reference circles), if any.
    fn radius_var(&self, center: usize) -> Option<usize> {
        self.circles.iter().position(|&(_, c)| c == center).map(|k| 2 * self.npoints + k)
    }
}

/// Write one residual row: its value plus the sparse gradient entries.
/// Gradients accumulate, so a variable a row touches twice (e.g. a line
/// constrained against its own endpoint) sums correctly.
fn put_row(
    r: &mut DVector<f64>,
    jac: &mut Option<&mut DMatrix<f64>>,
    k: &mut usize,
    val: f64,
    grads: &[(usize, f64)],
) {
    r[*k] = val;
    if let Some(j) = jac.as_deref_mut() {
        for &(c, g) in grads {
            j[(*k, c)] += g;
        }
    }
    *k += 1;
}

impl Sketch {
    /// Solve all constraints, moving points to satisfy them.
    pub fn solve(&mut self) {
        self.solve_with_fixed(&[]);
    }

    /// Solve constraints while holding the listed points fixed (used for dragging:
    /// the dragged point is pinned to the cursor and everything else follows).
    ///
    /// Two stages: first with every circle radius **pinned** (points move, sizes
    /// don't — the everyday case), then, only if the constraints can't be met that
    /// way, again with the radii **free** — so tangency/radius relations resize a
    /// circle exactly when that's the only way to satisfy them.
    pub fn solve_with_fixed(&mut self, fixed_points: &[usize]) {
        let layout = VarLayout::new(self);
        let n = self.points.len();
        let m = self.residual_len(&layout);
        if layout.nvars() == 0 || m == 0 {
            self.apply_equal_radius(); // radius relations don't need a point solve
            return;
        }

        // Pinned point variables: points flagged `fixed` (projected from the 3D
        // body) plus any caller-supplied pins (e.g. the point being dragged).
        let mut pinned = vec![false; layout.nvars()];
        for (i, p) in self.points.iter().enumerate() {
            if p.fixed {
                pinned[2 * i] = true;
                pinned[2 * i + 1] = true;
            }
        }
        for &pi in fixed_points {
            if pi < n {
                pinned[2 * pi] = true;
                pinned[2 * pi + 1] = true;
            }
        }

        // Seed from current positions and radii.
        let mut x = DVector::<f64>::zeros(layout.nvars());
        for (i, p) in self.points.iter().enumerate() {
            x[2 * i] = p.x;
            x[2 * i + 1] = p.y;
        }
        for (k, &(ei, _)) in layout.circles.iter().enumerate() {
            if let Some(SketchEntity::Circle { radius, .. }) = self.entities.get(ei) {
                x[2 * n + k] = *radius;
            }
        }

        // Stage 1: radii pinned — only point variables move.
        let point_free: Vec<usize> = (0..2 * n).filter(|&i| !pinned[i]).collect();
        if !point_free.is_empty() {
            self.lm_minimize(&layout, &mut x, &point_free, m);
        }
        // Stage 2: if moving points alone couldn't satisfy everything, free the
        // radii too (least-change resize).
        if !layout.circles.is_empty() && self.eval(&layout, &x, &mut None).norm() > 1e-7 {
            let free: Vec<usize> = (0..layout.nvars()).filter(|&i| !pinned[i]).collect();
            if !free.is_empty() {
                self.lm_minimize(&layout, &mut x, &free, m);
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
            for (k, &(ei, _)) in layout.circles.iter().enumerate() {
                if let Some(SketchEntity::Circle { radius, .. }) = self.entities.get_mut(ei) {
                    *radius = x[2 * n + k].abs().max(1e-4);
                }
            }
        }

        self.apply_equal_radius();
    }

    /// Damped Gauss-Newton (Levenberg-Marquardt) over the `free` variable subset,
    /// updating `x` in place.
    fn lm_minimize(&self, layout: &VarLayout, x: &mut DVector<f64>, free: &[usize], m: usize) {
        let mut lambda = 1e-3_f64;
        for _ in 0..100 {
            let mut jac = DMatrix::<f64>::zeros(m, layout.nvars());
            let r = self.eval(layout, x, &mut Some(&mut jac));
            let cost = r.norm_squared();
            if r.norm() < 1e-10 {
                break;
            }
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
                    if self.eval(layout, &x_new, &mut None).norm_squared() < cost {
                        *x = x_new;
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
    }

    /// Analyze the sketch's constraint state for SolidWorks-style feedback:
    /// how many degrees of freedom remain, whether constraints conflict, and
    /// which points can still move. Call on a *solved* sketch.
    ///
    /// The remaining freedom is the null space of the constraint Jacobian at
    /// the current solution: rank comes from the eigenvalues of JᵀJ, and a
    /// point is "free" (draw it blue) if some null-space direction moves it.
    pub fn dof_report(&self) -> DofReport {
        let layout = VarLayout::new(self);
        let n = self.points.len();
        let nvars = layout.nvars();
        let free_of = |var_can_move: &[bool]| -> Vec<bool> {
            (0..n).map(|i| var_can_move[2 * i] || var_can_move[2 * i + 1]).collect()
        };
        if nvars == 0 {
            return DofReport::default();
        }
        // Free variables: everything except body-projected (fixed) points.
        let mut movable = vec![true; nvars];
        for (i, p) in self.points.iter().enumerate() {
            if p.fixed {
                movable[2 * i] = false;
                movable[2 * i + 1] = false;
            }
        }
        let free: Vec<usize> = (0..nvars).filter(|&i| movable[i]).collect();
        if free.is_empty() {
            return DofReport { dof: 0, over_defined: false, free_points: vec![false; n] };
        }
        let m = self.residual_len(&layout);
        if m == 0 {
            // Nothing constrains anything — every movable variable is a DOF.
            return DofReport { dof: free.len(), over_defined: false, free_points: free_of(&movable) };
        }

        let mut x = DVector::<f64>::zeros(nvars);
        for (i, p) in self.points.iter().enumerate() {
            x[2 * i] = p.x;
            x[2 * i + 1] = p.y;
        }
        for (k, &(ei, _)) in layout.circles.iter().enumerate() {
            if let Some(SketchEntity::Circle { radius, .. }) = self.entities.get(ei) {
                x[2 * n + k] = *radius;
            }
        }
        let mut jac = DMatrix::<f64>::zeros(m, nvars);
        let r = self.eval(&layout, &x, &mut Some(&mut jac));
        let jf = jac.select_columns(free.iter());
        // Null space of J from the near-zero eigenvalues of JᵀJ (symmetric,
        // always full decomposition — unlike a compact SVD it can't hide null
        // directions when the system is wide).
        let eig = (jf.transpose() * &jf).symmetric_eigen();
        let emax = eig.eigenvalues.iter().cloned().fold(0.0_f64, f64::max);
        let tol = (emax * 1.0e-9).max(1.0e-12);
        let mut var_can_move = vec![false; nvars];
        let mut dof = 0usize;
        for (k, &ev) in eig.eigenvalues.iter().enumerate() {
            if ev.abs() < tol {
                dof += 1;
                let v = eig.eigenvectors.column(k);
                for (j, &col) in free.iter().enumerate() {
                    if v[j].abs() > 1.0e-6 {
                        var_can_move[col] = true;
                    }
                }
            }
        }
        // Conflicting (over-defined) constraints: the solved sketch still can't
        // reach zero residual.
        DofReport { dof, over_defined: r.norm() > 1.0e-5, free_points: free_of(&var_can_move) }
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
        // Slot-width dimensions: drive the slot's half-width (radius = width/2).
        let widths: Vec<(usize, usize, f64)> = self
            .constraints
            .iter()
            .filter_map(|c| match c {
                Constraint::SlotWidth { a, b, value, .. } => Some((*a, *b, *value)),
                _ => None,
            })
            .collect();
        for (sa, sb, value) in widths {
            for e in self.entities.iter_mut() {
                if let SketchEntity::Slot { a, b, radius, .. } = e {
                    if (*a == sa && *b == sb) || (*a == sb && *b == sa) {
                        *radius = (value * 0.5).max(1e-4);
                    }
                }
            }
        }
    }

    /// Total number of scalar residual equations: all constraints plus one
    /// implicit arc-consistency equation per arc entity.
    fn residual_len(&self, layout: &VarLayout) -> usize {
        let from_constraints: usize = self
            .constraints
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
                // A driving radius solves only if its circle (radius variable) exists.
                Constraint::Radius { center, .. } => layout.radius_var(*center).map_or(0, |_| 1),
                // Enforced after the solve: EqualRadius keeps its "a drives b"
                // semantics; slot width isn't a solver variable.
                Constraint::EqualRadius { .. } | Constraint::SlotWidth { .. } => 0,
            })
            .sum();
        let arcs = self.entities.iter().filter(|e| matches!(e, SketchEntity::Arc { .. })).count();
        from_constraints + arcs
    }

    /// Evaluate the residual vector f(x) and, when `jac` is given, fill its
    /// analytic Jacobian ∂f/∂x alongside. Row order matches [`residual_len`]:
    /// constraints first, then one arc-consistency row per arc entity.
    fn eval(&self, layout: &VarLayout, x: &DVector<f64>, jac: &mut Option<&mut DMatrix<f64>>) -> DVector<f64> {
        let mut r = DVector::zeros(self.residual_len(layout));
        let mut k = 0usize;
        let px = |i: usize| x[2 * i];
        let py = |i: usize| x[2 * i + 1];
        // Sign of a value, treating exactly-zero as positive so an |·| residual
        // still has a usable descent direction right at the kink.
        let sgn = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };

        for c in &self.constraints {
            match c {
                Constraint::Coincident(a, b) => {
                    let (a, b) = (*a, *b);
                    put_row(&mut r, jac, &mut k, px(a) - px(b), &[(2 * a, 1.0), (2 * b, -1.0)]);
                    put_row(&mut r, jac, &mut k, py(a) - py(b), &[(2 * a + 1, 1.0), (2 * b + 1, -1.0)]);
                }
                Constraint::Horizontal(a, b) => {
                    let (a, b) = (*a, *b);
                    put_row(&mut r, jac, &mut k, py(a) - py(b), &[(2 * a + 1, 1.0), (2 * b + 1, -1.0)]);
                }
                Constraint::Vertical(a, b) => {
                    let (a, b) = (*a, *b);
                    put_row(&mut r, jac, &mut k, px(a) - px(b), &[(2 * a, 1.0), (2 * b, -1.0)]);
                }
                Constraint::Distance { a, b, value, axis, .. } => {
                    let (a, b) = (*a, *b);
                    let (dx, dy) = (px(a) - px(b), py(a) - py(b));
                    match axis {
                        DimAxis::Aligned => {
                            let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                            put_row(&mut r, jac, &mut k, len - *value, &[
                                (2 * a, dx / len),
                                (2 * a + 1, dy / len),
                                (2 * b, -dx / len),
                                (2 * b + 1, -dy / len),
                            ]);
                        }
                        DimAxis::Horizontal => {
                            let s = sgn(dx);
                            put_row(&mut r, jac, &mut k, dx.abs() - *value, &[(2 * a, s), (2 * b, -s)]);
                        }
                        DimAxis::Vertical => {
                            let s = sgn(dy);
                            put_row(&mut r, jac, &mut k, dy.abs() - *value, &[(2 * a + 1, s), (2 * b + 1, -s)]);
                        }
                    }
                }
                Constraint::Midpoint { mid, a, b } => {
                    let (mid, a, b) = (*mid, *a, *b);
                    put_row(&mut r, jac, &mut k, px(mid) - 0.5 * (px(a) + px(b)), &[
                        (2 * mid, 1.0),
                        (2 * a, -0.5),
                        (2 * b, -0.5),
                    ]);
                    put_row(&mut r, jac, &mut k, py(mid) - 0.5 * (py(a) + py(b)), &[
                        (2 * mid + 1, 1.0),
                        (2 * a + 1, -0.5),
                        (2 * b + 1, -0.5),
                    ]);
                }
                Constraint::Parallel(a, b, c, d) => {
                    let (a, b, c, d) = (*a, *b, *c, *d);
                    let (d1x, d1y) = (px(b) - px(a), py(b) - py(a));
                    let (d2x, d2y) = (px(d) - px(c), py(d) - py(c));
                    // cross product → 0 when parallel
                    put_row(&mut r, jac, &mut k, d1x * d2y - d1y * d2x, &[
                        (2 * a, -d2y),
                        (2 * a + 1, d2x),
                        (2 * b, d2y),
                        (2 * b + 1, -d2x),
                        (2 * c, d1y),
                        (2 * c + 1, -d1x),
                        (2 * d, -d1y),
                        (2 * d + 1, d1x),
                    ]);
                }
                Constraint::Perpendicular(a, b, c, d) => {
                    let (a, b, c, d) = (*a, *b, *c, *d);
                    let (d1x, d1y) = (px(b) - px(a), py(b) - py(a));
                    let (d2x, d2y) = (px(d) - px(c), py(d) - py(c));
                    // dot product → 0 when perpendicular
                    put_row(&mut r, jac, &mut k, d1x * d2x + d1y * d2y, &[
                        (2 * a, -d2x),
                        (2 * a + 1, -d2y),
                        (2 * b, d2x),
                        (2 * b + 1, d2y),
                        (2 * c, -d1x),
                        (2 * c + 1, -d1y),
                        (2 * d, d1x),
                        (2 * d + 1, d1y),
                    ]);
                }
                Constraint::Equal(a, b, c, d) => {
                    let (a, b, c, d) = (*a, *b, *c, *d);
                    let (d1x, d1y) = (px(b) - px(a), py(b) - py(a));
                    let (d2x, d2y) = (px(d) - px(c), py(d) - py(c));
                    let l1 = (d1x * d1x + d1y * d1y).sqrt().max(1e-12);
                    let l2 = (d2x * d2x + d2y * d2y).sqrt().max(1e-12);
                    put_row(&mut r, jac, &mut k, l1 - l2, &[
                        (2 * a, -d1x / l1),
                        (2 * a + 1, -d1y / l1),
                        (2 * b, d1x / l1),
                        (2 * b + 1, d1y / l1),
                        (2 * c, d2x / l2),
                        (2 * c + 1, d2y / l2),
                        (2 * d, -d2x / l2),
                        (2 * d + 1, -d2y / l2),
                    ]);
                }
                Constraint::Tangent { a, b, center, radius } => {
                    let (a, b, c) = (*a, *b, *center);
                    let (dx, dy) = (px(b) - px(a), py(b) - py(a));
                    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let cross = dx * (py(c) - py(a)) - dy * (px(c) - px(a));
                    let s = cross / len; // signed perpendicular distance centre↔line
                    let sg = sgn(s);
                    // Live radius: the solver variable when the circle exists,
                    // falling back to the constraint's baked value.
                    let rvar = layout.radius_var(c);
                    let rad = rvar.map_or(*radius, |v| x[v]);
                    // ∂|s|/∂q = sg · (∂cross/∂q − s·∂len/∂q) / len
                    let dcross = [
                        (2 * a, py(b) - py(c)),
                        (2 * a + 1, px(c) - px(b)),
                        (2 * b, py(c) - py(a)),
                        (2 * b + 1, px(a) - px(c)),
                        (2 * c, -dy),
                        (2 * c + 1, dx),
                    ];
                    let dlen = [-dx / len, -dy / len, dx / len, dy / len, 0.0, 0.0];
                    let mut grads: Vec<(usize, f64)> = dcross
                        .iter()
                        .zip(dlen.iter())
                        .map(|(&(col, dc), &dl)| (col, sg * (dc - s * dl) / len))
                        .collect();
                    if let Some(v) = rvar {
                        grads.push((v, -1.0));
                    }
                    put_row(&mut r, jac, &mut k, s.abs() - rad, &grads);
                }
                Constraint::Angle { a, b, c, d, value, .. } => {
                    // Signed angle from line (a→b) to line (c→d) should equal `value`.
                    let (a, b, c, d) = (*a, *b, *c, *d);
                    let (v1x, v1y) = (px(b) - px(a), py(b) - py(a));
                    let (v2x, v2y) = (px(d) - px(c), py(d) - py(c));
                    let cross = v1x * v2y - v1y * v2x;
                    let dot = v1x * v2x + v1y * v2y;
                    let diff = cross.atan2(dot) - *value;
                    let l1 = (v1x * v1x + v1y * v1y).max(1e-12);
                    let l2 = (v2x * v2x + v2y * v2y).max(1e-12);
                    // θ = θ(v2) − θ(v1); ∂θ/∂v1 = (v1y, −v1x)/|v1|², ∂θ/∂v2 = (−v2y, v2x)/|v2|².
                    put_row(&mut r, jac, &mut k, diff.sin().atan2(diff.cos()), &[
                        (2 * a, -v1y / l1),
                        (2 * a + 1, v1x / l1),
                        (2 * b, v1y / l1),
                        (2 * b + 1, -v1x / l1),
                        (2 * c, v2y / l2),
                        (2 * c + 1, -v2x / l2),
                        (2 * d, -v2y / l2),
                        (2 * d + 1, v2x / l2),
                    ]);
                }
                Constraint::PointLineDistance { p, a, b, value, .. } => {
                    // Perpendicular distance from point p to the line (a,b) → value.
                    let (p, a, b) = (*p, *a, *b);
                    let (dx, dy) = (px(b) - px(a), py(b) - py(a));
                    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let cross = dx * (py(p) - py(a)) - dy * (px(p) - px(a));
                    let s = cross / len;
                    let sg = sgn(s);
                    let dcross = [
                        (2 * a, py(b) - py(p)),
                        (2 * a + 1, px(p) - px(b)),
                        (2 * b, py(p) - py(a)),
                        (2 * b + 1, px(a) - px(p)),
                        (2 * p, -dy),
                        (2 * p + 1, dx),
                    ];
                    let dlen = [-dx / len, -dy / len, dx / len, dy / len, 0.0, 0.0];
                    let grads: Vec<(usize, f64)> = dcross
                        .iter()
                        .zip(dlen.iter())
                        .map(|(&(col, dc), &dl)| (col, sg * (dc - s * dl) / len))
                        .collect();
                    put_row(&mut r, jac, &mut k, s.abs() - *value, &grads);
                }
                Constraint::PointOnLine { p, a, b } => {
                    // Signed perpendicular distance from point p to the line (a,b) → zero.
                    let (p, a, b) = (*p, *a, *b);
                    let (dx, dy) = (px(b) - px(a), py(b) - py(a));
                    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let cross = dx * (py(p) - py(a)) - dy * (px(p) - px(a));
                    let s = cross / len;
                    let dcross = [
                        (2 * a, py(b) - py(p)),
                        (2 * a + 1, px(p) - px(b)),
                        (2 * b, py(p) - py(a)),
                        (2 * b + 1, px(a) - px(p)),
                        (2 * p, -dy),
                        (2 * p + 1, dx),
                    ];
                    let dlen = [-dx / len, -dy / len, dx / len, dy / len, 0.0, 0.0];
                    let grads: Vec<(usize, f64)> = dcross
                        .iter()
                        .zip(dlen.iter())
                        .map(|(&(col, dc), &dl)| (col, (dc - s * dl) / len))
                        .collect();
                    put_row(&mut r, jac, &mut k, s, &grads);
                }
                Constraint::PointOnArc { p, cx, cy, radius } => {
                    // Distance from p to the baked arc centre equals its radius.
                    let p = *p;
                    let (ddx, ddy) = (px(p) - *cx, py(p) - *cy);
                    let d = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                    put_row(&mut r, jac, &mut k, d - *radius, &[(2 * p, ddx / d), (2 * p + 1, ddy / d)]);
                }
                Constraint::PointOnCircle { p, center } => {
                    // Distance from p to the centre equals the circle's radius (a
                    // solver variable, so the point and the size can co-solve).
                    let (p, c) = (*p, *center);
                    let (dx, dy) = (px(p) - px(c), py(p) - py(c));
                    let d = (dx * dx + dy * dy).sqrt().max(1e-12);
                    match layout.radius_var(c) {
                        Some(rv) => put_row(&mut r, jac, &mut k, d - x[rv], &[
                            (2 * p, dx / d),
                            (2 * p + 1, dy / d),
                            (2 * c, -dx / d),
                            (2 * c + 1, -dy / d),
                            (rv, -1.0),
                        ]),
                        // Circle gone → no-op residual (keeps the row count stable).
                        None => put_row(&mut r, jac, &mut k, 0.0, &[]),
                    }
                }
                Constraint::Radius { center, value, .. } => {
                    // Driving radius dimension on the circle's radius variable.
                    if let Some(rv) = layout.radius_var(*center) {
                        put_row(&mut r, jac, &mut k, x[rv] - value.max(1e-4), &[(rv, 1.0)]);
                    }
                }
                // Enforced after the solve: EqualRadius keeps its "a drives b"
                // semantics; slot width isn't a solver variable.
                Constraint::EqualRadius { .. } | Constraint::SlotWidth { .. } => {}
            }
        }

        // Implicit arc consistency: both endpoints the same distance from the
        // centre (an arc's radius is |centre→a|; without this a solve could pull
        // the other endpoint off the arc).
        for e in &self.entities {
            if let SketchEntity::Arc { center, a, b, .. } = e {
                let (c, a, b) = (*center, *a, *b);
                let (dax, day) = (px(a) - px(c), py(a) - py(c));
                let (dbx, dby) = (px(b) - px(c), py(b) - py(c));
                let la = (dax * dax + day * day).sqrt().max(1e-12);
                let lb = (dbx * dbx + dby * dby).sqrt().max(1e-12);
                put_row(&mut r, jac, &mut k, la - lb, &[
                    (2 * a, dax / la),
                    (2 * a + 1, day / la),
                    (2 * b, -dbx / lb),
                    (2 * b + 1, -dby / lb),
                    (2 * c, -dax / la + dbx / lb),
                    (2 * c + 1, -day / la + dby / lb),
                ]);
            }
        }
        r
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
    fn tangent_resizes_the_circle_when_the_line_is_locked() {
        // The line can't move (fixed endpoints) and neither can the centre, so the
        // ONLY way to satisfy tangency is to grow the circle: r 2 → 3.
        let mut s = Sketch::default();
        let center = s.add_fixed_point(0.0, 0.0);
        s.add_circle(center, 2.0);
        let a = s.add_fixed_point(-5.0, 3.0);
        let b = s.add_fixed_point(5.0, 3.0);
        s.constraints.push(Constraint::Tangent { a, b, center, radius: 2.0 });
        s.solve();
        let r = s
            .entities
            .iter()
            .find_map(|e| match e {
                SketchEntity::Circle { radius, .. } => Some(*radius),
                _ => None,
            })
            .unwrap();
        assert!((r - 3.0).abs() < 1e-5, "circle should grow to touch the line, radius was {r}");
    }

    #[test]
    fn tangent_moves_the_line_when_it_can_and_keeps_the_radius() {
        // Same setup but the line is free: the cheap fix is moving the line, so the
        // radius must NOT change (stage 1 satisfies everything with points alone).
        let mut s = Sketch::default();
        let center = s.add_fixed_point(0.0, 0.0);
        s.add_circle(center, 2.0);
        let a = s.add_point(-5.0, 3.0);
        let b = s.add_point(5.0, 3.0);
        s.constraints.push(Constraint::Horizontal(a, b));
        s.constraints.push(Constraint::Tangent { a, b, center, radius: 2.0 });
        s.solve();
        let r = s
            .entities
            .iter()
            .find_map(|e| match e {
                SketchEntity::Circle { radius, .. } => Some(*radius),
                _ => None,
            })
            .unwrap();
        assert!((r - 2.0).abs() < 1e-6, "radius should stay 2, was {r}");
        let (dx, dy) = (s.points[b].x - s.points[a].x, s.points[b].y - s.points[a].y);
        let len = (dx * dx + dy * dy).sqrt();
        let cross = dx * (0.0 - s.points[a].y) - dy * (0.0 - s.points[a].x);
        assert!((cross.abs() / len - 2.0).abs() < 1e-5, "line should touch the circle");
    }

    #[test]
    fn radius_dimension_pulls_a_rim_point_along() {
        // A driving radius dim resizes the circle in-solve, and a point constrained
        // onto the rim rides out with it.
        let mut s = Sketch::default();
        let center = s.add_fixed_point(0.0, 0.0);
        s.add_circle(center, 2.0);
        let p = s.add_point(2.0, 0.0);
        s.constraints.push(Constraint::PointOnCircle { p, center });
        s.constraints.push(Constraint::Radius { center, value: 6.0, diameter: false });
        s.solve();
        let r = s
            .entities
            .iter()
            .find_map(|e| match e {
                SketchEntity::Circle { radius, .. } => Some(*radius),
                _ => None,
            })
            .unwrap();
        assert!((r - 6.0).abs() < 1e-6, "radius should be driven to 6, was {r}");
        let d = (s.points[p].x.powi(2) + s.points[p].y.powi(2)).sqrt();
        assert!((d - 6.0).abs() < 1e-4, "rim point should follow the radius out: {d}");
    }

    #[test]
    fn arc_endpoints_stay_equidistant_from_the_center() {
        // The implicit arc-consistency residual pulls a stray endpoint back onto
        // the arc's radius (|c→a| = |c→b|).
        let mut s = Sketch::default();
        let c = s.add_fixed_point(0.0, 0.0);
        let a = s.add_fixed_point(2.0, 0.0);
        let b = s.add_point(0.0, 3.0); // off the r=2 arc
        s.add_arc(c, a, b, true, false);
        s.solve();
        let lb = (s.points[b].x.powi(2) + s.points[b].y.powi(2)).sqrt();
        assert!((lb - 2.0).abs() < 1e-5, "arc endpoint should sit at r=2, was {lb}");
    }

    #[test]
    fn horizontal_axis_distance_solves_from_a_zero_gap() {
        // dx starts at exactly 0 — the |dx| kink — and must still open to 10.
        let mut s = Sketch::default();
        let a = s.add_fixed_point(0.0, 0.0);
        let b = s.add_point(0.0, 5.0);
        s.constraints.push(Constraint::Distance { a, b, value: 10.0, offset: 0.5, axis: DimAxis::Horizontal });
        s.solve();
        let dx = (s.points[a].x - s.points[b].x).abs();
        assert!((dx - 10.0).abs() < 1e-4, "horizontal extent was {dx}");
    }

    #[test]
    fn a_dimensioned_rectangle_without_an_anchor_keeps_rigid_body_freedom() {
        // H/V + two driving dims pin the shape, but nothing pins it to the plane:
        // 3 DOF remain (translate ×2, and the H/V pair leaves… no rotation here,
        // so exactly 2) — the point is it is NOT fully defined and all points stay free.
        let mut s = Sketch::default();
        let p0 = s.add_point(0.0, 0.0);
        let p1 = s.add_point(2.0, 0.0);
        let p2 = s.add_point(2.0, 1.0);
        let p3 = s.add_point(0.0, 1.0);
        s.constraints.push(Constraint::Horizontal(p0, p1));
        s.constraints.push(Constraint::Horizontal(p3, p2));
        s.constraints.push(Constraint::Vertical(p1, p2));
        s.constraints.push(Constraint::Vertical(p0, p3));
        s.constraints.push(Constraint::Distance { a: p0, b: p1, value: 2.0, offset: 0.5, axis: DimAxis::Aligned });
        s.constraints.push(Constraint::Distance { a: p1, b: p2, value: 1.0, offset: 0.5, axis: DimAxis::Aligned });
        s.solve();
        let rep = s.dof_report();
        assert_eq!(rep.dof, 2, "un-anchored rectangle should have exactly its translation DOF");
        assert!(!rep.over_defined);
        assert!(rep.free_points.iter().all(|&f| f), "every point can still translate");
    }

    #[test]
    fn anchoring_to_the_origin_fully_defines_the_rectangle() {
        // Same rectangle, but one corner coincident with the fixed origin anchor:
        // zero DOF, every point fully defined (black).
        let mut s = Sketch::default();
        let origin = s.ensure_origin();
        let p0 = s.add_point(0.1, 0.1);
        let p1 = s.add_point(2.0, 0.0);
        let p2 = s.add_point(2.0, 1.0);
        let p3 = s.add_point(0.0, 1.0);
        s.constraints.push(Constraint::Horizontal(p0, p1));
        s.constraints.push(Constraint::Horizontal(p3, p2));
        s.constraints.push(Constraint::Vertical(p1, p2));
        s.constraints.push(Constraint::Vertical(p0, p3));
        s.constraints.push(Constraint::Distance { a: p0, b: p1, value: 2.0, offset: 0.5, axis: DimAxis::Aligned });
        s.constraints.push(Constraint::Distance { a: p1, b: p2, value: 1.0, offset: 0.5, axis: DimAxis::Aligned });
        s.constraints.push(Constraint::Coincident(p0, origin));
        s.solve();
        let rep = s.dof_report();
        assert_eq!(rep.dof, 0, "anchored + dimensioned rectangle is fully defined");
        assert!(!rep.over_defined);
        assert!(rep.free_points.iter().all(|&f| !f), "no point should be free");
    }

    #[test]
    fn conflicting_dimensions_report_over_defined() {
        let mut s = Sketch::default();
        let a = s.add_point(0.0, 0.0);
        let b = s.add_point(2.0, 0.0);
        s.constraints.push(Constraint::Distance { a, b, value: 2.0, offset: 0.5, axis: DimAxis::Aligned });
        s.constraints.push(Constraint::Distance { a, b, value: 3.0, offset: 0.5, axis: DimAxis::Aligned });
        s.solve();
        let rep = s.dof_report();
        assert!(rep.over_defined, "two different lengths on one line must flag a conflict");
    }

    #[test]
    fn an_undimensioned_circle_radius_counts_as_freedom() {
        // Circle centred on the origin anchor: only the radius is free → 1 DOF.
        let mut s = Sketch::default();
        let origin = s.ensure_origin();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 2.0);
        s.constraints.push(Constraint::Coincident(c, origin));
        s.solve();
        let rep = s.dof_report();
        assert_eq!(rep.dof, 1, "free radius is the one remaining DOF");
        // Dimension the radius → fully defined.
        s.constraints.push(Constraint::Radius { center: c, value: 2.0, diameter: false });
        s.solve();
        assert_eq!(s.dof_report().dof, 0, "radius dim completes the definition");
    }

    #[test]
    fn ensure_origin_is_idempotent_and_detected() {
        let mut s = Sketch::default();
        assert!(!s.has_geometry());
        let o1 = s.ensure_origin();
        let o2 = s.ensure_origin();
        assert_eq!(o1, o2, "second call must find the first anchor");
        assert_eq!(s.origin_point(), Some(o1));
        assert!(!s.has_geometry(), "the anchor alone isn't user geometry");
        let p = s.add_point(1.0, 1.0);
        s.entities.push(SketchEntity::Point { at: p });
        assert!(s.has_geometry(), "a real point entity is geometry");
    }

    #[test]
    fn a_lone_circle_region_is_one_full_arc_span() {
        let mut s = Sketch::default();
        let c = s.add_point(1.0, 2.0);
        s.add_circle(c, 3.0);
        let regions = s.regions();
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.outer_arcs.len(), 1, "one span covering the whole rim");
        let span = r.outer_arcs[0];
        assert_eq!(span.count, r.outer.len(), "span covers every edge");
        assert!((span.center[0] - 1.0).abs() < 1e-9 && (span.center[1] - 2.0).abs() < 1e-9);
        assert!((span.radius - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_circular_hole_keeps_its_arc_span() {
        let mut s = Sketch::default();
        add_rect(&mut s, -10.0, -10.0, 10.0, 10.0);
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 2.0);
        let regions = s.regions();
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert!(r.outer_arcs.is_empty(), "rectangle outer is all lines");
        assert_eq!(r.holes.len(), 1);
        assert_eq!(r.hole_arcs.len(), 1);
        assert_eq!(r.hole_arcs[0].len(), 1, "hole is one full circle span");
        assert!((r.hole_arcs[0][0].radius - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_chord_cut_circle_yields_partial_arc_spans() {
        // A chord splits the disk: each region's boundary is one arc run + the chord.
        let mut s = Sketch::default();
        let c = s.add_point(0.0, 0.0);
        s.add_circle(c, 2.0);
        let a = s.add_point(-3.0, 0.5);
        let b = s.add_point(3.0, 0.5);
        s.add_line(a, b, false);
        let regions = s.regions();
        assert_eq!(regions.len(), 2);
        for r in &regions {
            assert_eq!(r.outer_arcs.len(), 1, "each piece has exactly one arc run");
            let span = r.outer_arcs[0];
            assert!(span.count < r.outer.len(), "chord edges must not be inside the span");
            assert!((span.radius - 2.0).abs() < 1e-9);
            // Every vertex the span covers (excluding its intersection endpoints)
            // sits exactly on the circle.
            let n = r.outer.len();
            for k in 1..span.count {
                let p = r.outer[(span.first_edge + k) % n];
                let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!((d - 2.0).abs() < 1e-9, "span vertex off the circle: {d}");
            }
        }
    }

    #[test]
    fn a_large_sketch_solves_accurately() {
        // A ladder of 40 dimensioned, constrained rectangles sharing corners —
        // ~320 variables. Checks the analytic-Jacobian solver converges on a
        // sketch this size (and prints how long it took under --nocapture).
        let mut s = Sketch::default();
        let mut prev_right = None::<(usize, usize)>; // (bottom, top) of the shared edge
        for i in 0..40 {
            let x0 = i as f64 * 2.1; // slightly off the 2.0 the dims will enforce
            let (p0, p3) = match prev_right {
                Some((b, t)) => (b, t),
                None => (s.add_point(x0, 0.0), s.add_point(x0, 1.0)),
            };
            let p1 = s.add_point(x0 + 2.1, 0.05);
            let p2 = s.add_point(x0 + 2.1, 1.05);
            s.add_line(p0, p1, false);
            s.add_line(p1, p2, false);
            s.add_line(p2, p3, false);
            if prev_right.is_none() {
                s.add_line(p3, p0, false);
            }
            s.constraints.push(Constraint::Horizontal(p0, p1));
            s.constraints.push(Constraint::Vertical(p1, p2));
            s.constraints.push(Constraint::Horizontal(p3, p2));
            s.constraints.push(Constraint::Distance { a: p0, b: p1, value: 2.0, offset: 0.5, axis: DimAxis::Aligned });
            s.constraints.push(Constraint::Distance { a: p1, b: p2, value: 1.0, offset: 0.5, axis: DimAxis::Aligned });
            prev_right = Some((p1, p2));
        }
        let t0 = std::time::Instant::now();
        s.solve();
        println!("large sketch ({} points, {} constraints) solved in {:?}", s.points.len(), s.constraints.len(), t0.elapsed());
        // Every bay must come out exactly 2 × 1.
        for i in 0..40 {
            let (p0, p1) = (2 * i, 2 * i + 2);
            let w = ((s.points[p1].x - s.points[p0].x).powi(2) + (s.points[p1].y - s.points[p0].y).powi(2)).sqrt();
            assert!((w - 2.0).abs() < 1e-5, "bay {i} width {w}");
        }
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
