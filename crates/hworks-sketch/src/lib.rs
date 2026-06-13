//! `hworks-sketch` — Layer 2: the 2D sketcher and constraint solver.
//!
//! A sketch is a 2D problem in a plane's local UV coordinates: a set of points
//! (the solver's unknowns), entities built on those points (lines, circles,
//! rectangles, construction geometry), and constraints/dimensions that the
//! solver satisfies all at once. See `DESIGN.md` §5.
//!
//! At milestone **M0** this is a stub describing the data model. Free drawing
//! arrives at **M1**, the Newton/least-squares solver at **M2**.

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
