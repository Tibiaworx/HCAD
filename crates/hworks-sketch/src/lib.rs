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
