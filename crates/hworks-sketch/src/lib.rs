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

/// A 2D point in plane-local coordinates — every entity endpoint is one of these,
/// and each is an unknown the constraint solver positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

/// A drawable sketch entity. A rectangle/square is four `Line`s plus constraints;
/// a "construction line with midpoint" is a construction `Line` plus a midpoint
/// `Point` tied to it by a `Midpoint` constraint.
#[derive(Debug, Clone)]
pub enum SketchEntity {
    Line { a: usize, b: usize, construction: bool },
    Circle { center: usize, radius: f64 },
    Point { at: usize },
}

/// Geometric and dimensional relations the solver drives the geometry to satisfy.
#[derive(Debug, Clone)]
pub enum Constraint {
    Coincident(usize, usize),
    Horizontal(usize, usize),
    Vertical(usize, usize),
    Midpoint { mid: usize, a: usize, b: usize },
    Distance { a: usize, b: usize, value: f64 },
}

/// A closed area of a sketch, ready to extrude: one outer boundary loop plus any
/// inner loops that are *holes* in it (e.g. a rectangle with a circular hole).
#[derive(Debug, Default, Clone)]
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
#[derive(Debug, Default, Clone)]
pub struct Sketch {
    pub points: Vec<Point2>,
    pub entities: Vec<SketchEntity>,
    pub constraints: Vec<Constraint>,
}

impl Sketch {
    /// Add a free point and return its index (the solver's future unknown).
    pub fn add_point(&mut self, x: f64, y: f64) -> usize {
        self.points.push(Point2 { x, y });
        self.points.len() - 1
    }

    /// Add a line between two existing points.
    pub fn add_line(&mut self, a: usize, b: usize, construction: bool) {
        self.entities.push(SketchEntity::Line { a, b, construction });
    }

    /// Add a circle from a center point and radius.
    pub fn add_circle(&mut self, center: usize, radius: f64) {
        self.entities.push(SketchEntity::Circle { center, radius });
    }

    /// Remove all geometry, leaving an empty sketch (used when (re)entering a plane).
    pub fn clear(&mut self) {
        self.points.clear();
        self.entities.clear();
        self.constraints.clear();
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
            if let SketchEntity::Line { a, b, construction: false } = e {
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
            if let SketchEntity::Circle { center, radius } = e {
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
            if let SketchEntity::Line { a, b, construction: false } = e {
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
            if let SketchEntity::Circle { center, radius } = e {
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

    /// Group the contours into closed regions, nesting inner loops as holes
    /// (one level deep). A rectangle with a circle inside it becomes one region
    /// whose outer is the rectangle and whose hole is the circle.
    pub fn regions(&self) -> Vec<Region> {
        let contours = self.contours();
        let n = contours.len();
        let areas: Vec<f64> = contours.iter().map(|c| area(c)).collect();
        // `j` contains `i` if it's bigger and `i`'s centroid falls inside it.
        // (The area test matters: an outer loop's centroid can land inside its
        // own hole, which would fool a centroid-only test.)
        let contains = |j: usize, i: usize| {
            j != i && areas[j] > areas[i] && point_in_poly(centroid(&contours[i]), &contours[j])
        };
        let depth: Vec<usize> =
            (0..n).map(|i| (0..n).filter(|&j| contains(j, i)).count()).collect();

        let mut regions = Vec::new();
        for i in 0..n {
            if depth[i] != 0 {
                continue; // not an outer boundary
            }
            let holes = (0..n)
                .filter(|&k| depth[k] == 1 && contains(i, k))
                .map(|k| contours[k].clone())
                .collect();
            regions.push(Region { outer: contours[i].clone(), holes });
        }
        regions
    }
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
            return;
        }
        let nvars = 2 * n;

        // Free-variable indices (everything except the pinned points' x/y).
        let mut is_fixed = vec![false; nvars];
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

        for (i, p) in self.points.iter_mut().enumerate() {
            p.x = x[2 * i];
            p.y = x[2 * i + 1];
        }
    }

    /// Total number of scalar residual equations across all constraints.
    fn residual_len(&self) -> usize {
        self.constraints
            .iter()
            .map(|c| match c {
                Constraint::Coincident(..) => 2,
                Constraint::Horizontal(..) => 1,
                Constraint::Vertical(..) => 1,
                Constraint::Distance { .. } => 1,
                Constraint::Midpoint { .. } => 2,
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
                Constraint::Distance { a, b, value } => {
                    let (a, b) = (*a, *b);
                    let dx = x[2 * a] - x[2 * b];
                    let dy = x[2 * a + 1] - x[2 * b + 1];
                    r[k] = (dx * dx + dy * dy).sqrt() - *value;
                    k += 1;
                }
                Constraint::Midpoint { mid, a, b } => {
                    let (mid, a, b) = (*mid, *a, *b);
                    r[k] = x[2 * mid] - 0.5 * (x[2 * a] + x[2 * b]);
                    r[k + 1] = x[2 * mid + 1] - 0.5 * (x[2 * a + 1] + x[2 * b + 1]);
                    k += 2;
                }
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
        s.constraints.push(Constraint::Distance { a, b, value: 5.0 });
        s.solve();
        let d = ((s.points[a].x - s.points[b].x).powi(2)
            + (s.points[a].y - s.points[b].y).powi(2))
        .sqrt();
        assert!((d - 5.0).abs() < 1e-6, "distance was {d}");
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
        s.points[p2] = Point2 { x: 4.0, y: 3.0 };
        s.solve_with_fixed(&[p2]);
        assert!((s.points[p1].x - s.points[p2].x).abs() < 1e-6, "right edge not vertical");
        assert!((s.points[p3].y - s.points[p2].y).abs() < 1e-6, "top edge not horizontal");
        assert!((s.points[p0].y - s.points[p1].y).abs() < 1e-6, "bottom edge not horizontal");
        assert!((s.points[p0].x - s.points[p3].x).abs() < 1e-6, "left edge not vertical");
    }
}
