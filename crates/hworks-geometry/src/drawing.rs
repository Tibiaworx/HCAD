//! Orthographic projection with hidden-line removal — the engine behind 2D drawing sheets.
//!
//! A drawing view is the part's feature edges flattened along a view direction, with each edge
//! split into the runs that are actually visible and the runs hidden behind material. Visible
//! runs draw solid, hidden runs draw dashed (the drafting convention).
//!
//! Visibility is decided in view space rather than by ray casting: every triangle is projected
//! to 2D once and bucketed into a uniform grid, then a sample point is hidden if any triangle
//! covers it at a shallower depth. That turns the per-sample cost from "test every triangle"
//! into "test the handful in this cell", which is what makes a real part project in
//! milliseconds instead of seconds.

use crate::TriMesh;

/// The frame a view is projected through: `f` is the direction the viewer looks (into the
/// sheet), `r` runs right across the sheet and `u` runs up it.
#[derive(Clone, Copy, Debug)]
pub struct ViewBasis {
    pub f: [f64; 3],
    pub r: [f64; 3],
    pub u: [f64; 3],
}

fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 1e-12 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0; 3] }
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl ViewBasis {
    /// Build a frame looking along `dir`, with `up_hint` orienting the sheet. If `up_hint` is
    /// parallel to `dir` (looking straight down, say) a fallback axis is used so the frame
    /// stays well-defined.
    pub fn looking_along(dir: [f64; 3], up_hint: [f64; 3]) -> ViewBasis {
        let f = norm(dir);
        let mut up = norm(up_hint);
        if dot(f, up).abs() > 0.999 {
            up = if f[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        }
        let r = norm(cross(f, up));
        let u = norm(cross(r, f));
        ViewBasis { f, r, u }
    }

    /// Project a world point to sheet coordinates plus depth. Smaller depth is nearer the
    /// viewer, so "hidden" means something sits at a smaller depth over the same spot.
    pub fn project(&self, p: [f64; 3]) -> ([f64; 2], f64) {
        ([dot(p, self.r), dot(p, self.u)], dot(p, self.f))
    }
}

/// One run of a projected edge, in sheet units (model millimetres, before view scaling).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjEdge {
    pub a: [f64; 2],
    pub b: [f64; 2],
    /// True when the run is behind material — drawn dashed.
    pub hidden: bool,
}

/// Triangles projected to 2D and bucketed for fast point queries.
struct DepthGrid {
    tris: Vec<([[f64; 2]; 3], [f64; 3])>, // 2D corners + per-corner depth
    cells: Vec<Vec<u32>>,
    lo: [f64; 2],
    inv: f64,
    nx: usize,
    ny: usize,
}

impl DepthGrid {
    fn build(mesh: &TriMesh, basis: &ViewBasis) -> DepthGrid {
        let mut tris = Vec::with_capacity(mesh.indices.len() / 3);
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for t in mesh.indices.chunks_exact(3) {
            let mut c2 = [[0.0f64; 2]; 3];
            let mut cd = [0.0f64; 3];
            for (k, &i) in t.iter().enumerate() {
                let p = mesh.positions[i as usize];
                let (xy, d) = basis.project([p[0] as f64, p[1] as f64, p[2] as f64]);
                c2[k] = xy;
                cd[k] = d;
                for a in 0..2 {
                    lo[a] = lo[a].min(xy[a]);
                    hi[a] = hi[a].max(xy[a]);
                }
            }
            tris.push((c2, cd));
        }
        if tris.is_empty() {
            return DepthGrid { tris, cells: Vec::new(), lo: [0.0; 2], inv: 1.0, nx: 0, ny: 0 };
        }
        // Aim for a few triangles per cell; clamp so a huge mesh can't allocate wildly.
        let span = ((hi[0] - lo[0]).max(hi[1] - lo[1])).max(1e-9);
        let target = (tris.len() as f64).sqrt().clamp(4.0, 256.0);
        let cell = span / target;
        let inv = 1.0 / cell.max(1e-9);
        let nx = (((hi[0] - lo[0]) * inv).ceil() as usize + 1).clamp(1, 512);
        let ny = (((hi[1] - lo[1]) * inv).ceil() as usize + 1).clamp(1, 512);
        let mut cells = vec![Vec::new(); nx * ny];
        for (ti, (c2, _)) in tris.iter().enumerate() {
            let (mut tlo, mut thi) = ([f64::MAX; 2], [f64::MIN; 2]);
            for q in c2 {
                for a in 0..2 {
                    tlo[a] = tlo[a].min(q[a]);
                    thi[a] = thi[a].max(q[a]);
                }
            }
            let x0 = (((tlo[0] - lo[0]) * inv) as usize).min(nx - 1);
            let x1 = (((thi[0] - lo[0]) * inv) as usize).min(nx - 1);
            let y0 = (((tlo[1] - lo[1]) * inv) as usize).min(ny - 1);
            let y1 = (((thi[1] - lo[1]) * inv) as usize).min(ny - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    cells[y * nx + x].push(ti as u32);
                }
            }
        }
        DepthGrid { tris, cells, lo, inv, nx, ny }
    }

    /// Is `p` (sheet coords, at `depth`) covered by material nearer the viewer?
    fn occluded(&self, p: [f64; 2], depth: f64, eps: f64) -> bool {
        if self.cells.is_empty() {
            return false;
        }
        let x = ((p[0] - self.lo[0]) * self.inv).floor();
        let y = ((p[1] - self.lo[1]) * self.inv).floor();
        if x < 0.0 || y < 0.0 {
            return false;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.nx || y >= self.ny {
            return false;
        }
        for &ti in &self.cells[y * self.nx + x] {
            let (c, d) = &self.tris[ti as usize];
            // Barycentric containment; also gives the interpolated depth for free.
            let v0 = [c[1][0] - c[0][0], c[1][1] - c[0][1]];
            let v1 = [c[2][0] - c[0][0], c[2][1] - c[0][1]];
            let v2 = [p[0] - c[0][0], p[1] - c[0][1]];
            let den = v0[0] * v1[1] - v1[0] * v0[1];
            if den.abs() < 1e-15 {
                continue; // edge-on sliver covers nothing
            }
            let a = (v2[0] * v1[1] - v1[0] * v2[1]) / den;
            let b = (v0[0] * v2[1] - v2[0] * v0[1]) / den;
            if a < 0.0 || b < 0.0 || a + b > 1.0 {
                continue;
            }
            let zd = d[0] + a * (d[1] - d[0]) + b * (d[2] - d[0]);
            if zd < depth - eps {
                return true;
            }
        }
        false
    }
}

/// Project `edges` through `basis`, splitting each into visible and hidden runs against
/// `mesh`. `samples` sets how finely an edge is tested (and so how precisely a run's endpoint
/// lands where it crosses a silhouette); 24 is a good default.
pub fn project_edges(mesh: &TriMesh, edges: &[[[f32; 3]; 2]], basis: &ViewBasis, samples: usize) -> Vec<ProjEdge> {
    if edges.is_empty() {
        return Vec::new();
    }
    let grid = DepthGrid::build(mesh, basis);
    // Depth tolerance scaled to the model, so an edge isn't occluded by its own faces.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a] as f64);
            hi[a] = hi[a].max(p[a] as f64);
        }
    }
    let diag = if mesh.positions.is_empty() {
        1.0
    } else {
        ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(1e-9)
    };
    let eps = diag * 1e-4;
    let n = samples.max(2);

    let mut out = Vec::new();
    for e in edges {
        let p0 = [e[0][0] as f64, e[0][1] as f64, e[0][2] as f64];
        let p1 = [e[1][0] as f64, e[1][1] as f64, e[1][2] as f64];
        let (a2, ad) = basis.project(p0);
        let (b2, bd) = basis.project(p1);
        if (a2[0] - b2[0]).abs() < 1e-12 && (a2[1] - b2[1]).abs() < 1e-12 {
            continue; // edge points straight at the viewer — no length on the sheet
        }
        // Classify each sample, then emit maximal same-state runs.
        let mut state: Option<bool> = None;
        let mut run_start = 0usize;
        let lerp2 = |t: f64| [a2[0] + (b2[0] - a2[0]) * t, a2[1] + (b2[1] - a2[1]) * t];
        let mut push = |from: usize, to: usize, hidden: bool, out: &mut Vec<ProjEdge>| {
            let t0 = from as f64 / (n - 1) as f64;
            let t1 = to as f64 / (n - 1) as f64;
            let (a, b) = (lerp2(t0), lerp2(t1));
            if (a[0] - b[0]).abs() > 1e-12 || (a[1] - b[1]).abs() > 1e-12 {
                out.push(ProjEdge { a, b, hidden });
            }
        };
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            // Sample midway between steps at the ends so an endpoint sitting exactly on a
            // silhouette doesn't decide the whole run.
            let tt = t.clamp(0.5 / (n - 1) as f64, 1.0 - 0.5 / (n - 1) as f64);
            let p = lerp2(tt);
            let d = ad + (bd - ad) * tt;
            let hidden = grid.occluded(p, d, eps);
            match state {
                None => {
                    state = Some(hidden);
                    run_start = 0;
                }
                Some(s) if s != hidden => {
                    push(run_start, i, s, &mut out);
                    state = Some(hidden);
                    run_start = i;
                }
                _ => {}
            }
        }
        if let Some(s) = state {
            push(run_start, n - 1, s, &mut out);
        }
    }
    out
}

/// Sheet-space bounds of a set of projected edges: `(min, max)`.
pub fn edges_bounds(edges: &[ProjEdge]) -> ([f64; 2], [f64; 2]) {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for e in edges {
        for q in [e.a, e.b] {
            for a in 0..2 {
                lo[a] = lo[a].min(q[a]);
                hi[a] = hi[a].max(q[a]);
            }
        }
    }
    if lo[0] > hi[0] {
        return ([0.0; 2], [0.0; 2]);
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{extrude_solid, mesh_tessellation, tessellate, PlaneBasis};

    fn xy_basis() -> PlaneBasis {
        PlaneBasis { origin: [0.0, 0.0, 0.0], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
    }

    /// A 10mm cube seen face-on: the four silhouette edges are visible, and the four edges of
    /// the far face are hidden behind them.
    #[test]
    fn cube_front_view_hides_the_back_face() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 10.0).expect("cube");
        let tess = tessellate(&solid, 0.05);
        let t = mesh_tessellation(tess.mesh.clone());

        // Viewer on +Z looking toward -Z (the app's "Front").
        let basis = ViewBasis::looking_along([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        let proj = project_edges(&t.mesh, &t.edges, &basis, 24);
        assert!(!proj.is_empty(), "nothing projected");

        let vis: Vec<_> = proj.iter().filter(|e| !e.hidden).collect();
        let hid: Vec<_> = proj.iter().filter(|e| e.hidden).collect();
        assert!(!vis.is_empty(), "no visible edges");
        assert!(!hid.is_empty(), "the far face should be hidden behind the near one");

        // The silhouette is a 10x10 square.
        let (lo, hi) = edges_bounds(&proj);
        assert!((hi[0] - lo[0] - 10.0).abs() < 0.1, "width {:.2}", hi[0] - lo[0]);
        assert!((hi[1] - lo[1] - 10.0).abs() < 0.1, "height {:.2}", hi[1] - lo[1]);

        // Total visible length is the 40mm outline (the near face's four edges); the far
        // face contributes the same length again, hidden.
        let len = |v: &[&ProjEdge]| v.iter().map(|e| ((e.b[0] - e.a[0]).powi(2) + (e.b[1] - e.a[1]).powi(2)).sqrt()).sum::<f64>();
        assert!((len(&vis) - 40.0).abs() < 4.0, "visible length {:.1}, wanted ~40", len(&vis));
        assert!(len(&hid) > 20.0, "hidden length {:.1} — the back face barely registered", len(&hid));
    }

    /// Looking down the length of a tall box, the far end is hidden — a sanity check that
    /// depth ordering (not just "some edges differ") drives the classification.
    #[test]
    fn depth_ordering_decides_visibility() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 40.0).expect("bar");
        let tess = tessellate(&solid, 0.05);
        let t = mesh_tessellation(tess.mesh.clone());

        let near = ViewBasis::looking_along([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        let far = ViewBasis::looking_along([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let a = project_edges(&t.mesh, &t.edges, &near, 24);
        let b = project_edges(&t.mesh, &t.edges, &far, 24);
        // Flipping the view swaps which end is hidden, so both directions must produce
        // hidden runs — and the same total edge length.
        assert!(a.iter().any(|e| e.hidden) && b.iter().any(|e| e.hidden));
        let tot = |v: &[ProjEdge]| v.iter().map(|e| ((e.b[0] - e.a[0]).powi(2) + (e.b[1] - e.a[1]).powi(2)).sqrt()).sum::<f64>();
        assert!((tot(&a) - tot(&b)).abs() < 1.0, "{:.1} vs {:.1}", tot(&a), tot(&b));
    }

    /// A side view of the same bar must be 10 wide and 40 tall — the projection has to honour
    /// the view direction, not just flatten Z.
    #[test]
    fn side_view_shows_the_long_dimension() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 40.0).expect("bar");
        let tess = tessellate(&solid, 0.05);
        let t = mesh_tessellation(tess.mesh.clone());

        // Looking along -X ("Right" view): the sheet shows Z across and Y up.
        let basis = ViewBasis::looking_along([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = project_edges(&t.mesh, &t.edges, &basis, 24);
        let (lo, hi) = edges_bounds(&proj);
        assert!((hi[0] - lo[0] - 40.0).abs() < 0.1, "width {:.2}, wanted the 40mm length", hi[0] - lo[0]);
        assert!((hi[1] - lo[1] - 10.0).abs() < 0.1, "height {:.2}", hi[1] - lo[1]);
    }

    /// An empty edge set must not panic or invent geometry.
    #[test]
    fn empty_input_is_safe() {
        let m = TriMesh::default();
        let basis = ViewBasis::looking_along([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);
        assert!(project_edges(&m, &[], &basis, 24).is_empty());
        assert_eq!(edges_bounds(&[]), ([0.0; 2], [0.0; 2]));
    }
}
