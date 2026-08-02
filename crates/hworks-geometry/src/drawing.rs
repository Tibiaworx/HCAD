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
            // Only FRONT-facing triangles can occlude.
            //
            // On a closed solid the front faces already cover everything the viewer can see,
            // while a back face sits behind its own surface — but in projection it lands on
            // the same spot and, on a curved patch seen at a grazing angle, can come out a
            // hair nearer than the edge lying between them. Including back faces therefore
            // hid huge numbers of perfectly visible edges: a filleted bar reported 280 of its
            // 303 runs hidden, and only 6 of 288 visible from the top.
            let p0 = mesh.positions[t[0] as usize];
            let p1 = mesh.positions[t[1] as usize];
            let p2 = mesh.positions[t[2] as usize];
            let e1 = [(p1[0] - p0[0]) as f64, (p1[1] - p0[1]) as f64, (p1[2] - p0[2]) as f64];
            let e2 = [(p2[0] - p0[0]) as f64, (p2[1] - p0[1]) as f64, (p2[2] - p0[2]) as f64];
            let fnrm = cross(e1, e2);
            // `basis.f` points away from the viewer, so a face pointing back at us has a
            // negative dot. Zero-area slivers are dropped either way.
            if dot(fnrm, basis.f) >= 0.0 {
                continue;
            }
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

/// One view as it should land on an exported sheet.
pub struct SheetItem<'a> {
    pub edges: &'a [ProjEdge],
    /// Sheet position of the view's centre, in millimetres from the bottom-left.
    pub center: [f64; 2],
    pub scale: f64,
    pub show_hidden: bool,
    /// Caption printed under the view, if any.
    pub label: Option<String>,
    /// Section hatching, in the same pre-scale view units as `edges`.
    pub hatch: &'a [([f64; 2], [f64; 2])],
}

/// Render a sheet to SVG, in real millimetres so it prints at true scale.
///
/// SVG's y axis runs down the page while sheet coordinates run up from the bottom-left, so
/// every point is flipped on the way out. Hidden lines use the drafting dash pattern.
pub fn to_svg(sheet_w: f64, sheet_h: f64, items: &[SheetItem], title: &[(String, String)]) -> String {
    to_svg_with_dims(sheet_w, sheet_h, items, &[], title)
}

/// As `to_svg`, plus dimensions already laid out in final sheet coordinates.
pub fn to_svg_with_dims(
    sheet_w: f64,
    sheet_h: f64,
    items: &[SheetItem],
    dims: &[SheetDim],
    title: &[(String, String)],
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{sheet_w}mm\" height=\"{sheet_h}mm\" viewBox=\"0 0 {sheet_w} {sheet_h}\">\n"
    ));
    out.push_str("<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
    // Sheet border, inset the conventional 10mm.
    out.push_str(&format!(
        "<rect x=\"10\" y=\"10\" width=\"{:.3}\" height=\"{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"/>\n",
        (sheet_w - 20.0).max(0.0),
        (sheet_h - 20.0).max(0.0)
    ));

    for it in items {
        let map = |q: [f64; 2]| {
            let x = it.center[0] + q[0] * it.scale;
            let y = it.center[1] + q[1] * it.scale;
            (x, sheet_h - y) // flip into SVG's y-down space
        };
        let mut solid = String::new();
        let mut hidden = String::new();
        for e in it.edges {
            if e.hidden && !it.show_hidden {
                continue;
            }
            let (x0, y0) = map(e.a);
            let (x1, y1) = map(e.b);
            let seg = format!("M{x0:.3},{y0:.3} L{x1:.3},{y1:.3} ");
            if e.hidden {
                hidden.push_str(&seg);
            } else {
                solid.push_str(&seg);
            }
        }
        if !solid.is_empty() {
            out.push_str(&format!("<path d=\"{solid}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.35\"/>\n"));
        }
        if !hidden.is_empty() {
            out.push_str(&format!(
                "<path d=\"{hidden}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.25\" stroke-dasharray=\"1.5,1\"/>\n"
            ));
        }
        // Section hatching: thinner than the outline, so the cut face reads as fill.
        if !it.hatch.is_empty() {
            let mut d = String::new();
            for (a, b) in it.hatch {
                let (x0, y0) = map(*a);
                let (x1, y1) = map(*b);
                d.push_str(&format!("M{x0:.3},{y0:.3} L{x1:.3},{y1:.3} "));
            }
            out.push_str(&format!("<path d=\"{d}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.18\"/>
"));
        }
        if let Some(lbl) = &it.label {
            // Caption under the view's own extents, so it never lands on the geometry.
            let (lo, hi) = edges_bounds(it.edges);
            let cx = it.center[0] + (lo[0] + hi[0]) * 0.5 * it.scale;
            let below = it.center[1] + lo[1] * it.scale - 5.0;
            out.push_str(&format!(
                "<text x=\"{cx:.3}\" y=\"{:.3}\" font-family=\"sans-serif\" font-size=\"4\" text-anchor=\"middle\" fill=\"black\">{}</text>\n",
                sheet_h - below,
                xml_escape(lbl)
            ));
        }
    }

    // Dimensions. Arrowheads are drawn as filled triangles at each end of the dimension
    // line, with witness lines running from the measured points back to it.
    for dim in dims {
        let fy = |y: f64| sheet_h - y;
        let (g, text) = match dim {
            SheetDim::Linear(g, t) => (g, t),
            SheetDim::Mark(m) => {
                for (a, b) in &m.cross {
                    out.push_str(&format!(
                        "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.25\"/>\n",
                        a[0], fy(a[1]), b[0], fy(b[1])
                    ));
                }
                for (a, b) in &m.arms {
                    out.push_str(&format!(
                        "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.2\" stroke-dasharray=\"3,1,0.6,1\"/>\n",
                        a[0], fy(a[1]), b[0], fy(b[1])
                    ));
                }
                continue;
            }
            SheetDim::Radial(r, t) => {
                // Leader, landing, and an arrow biting each rim.
                let d = [r.label[0] - r.centre[0], r.label[1] - r.centre[1]];
                let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
                let u = [d[0] / l, d[1] / l];
                let start = r.rim_far.unwrap_or(r.centre);
                out.push_str(&format!(
                    "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.25\"/>\n",
                    start[0], fy(start[1]), r.label[0], fy(r.label[1])
                ));
                out.push_str(&format!(
                    "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.25\"/>\n",
                    r.label[0], fy(r.label[1]), r.shoulder[0], fy(r.shoulder[1])
                ));
                let head = 1.6_f64;
                let mut heads = vec![(r.rim, [-u[0], -u[1]])];
                if let Some(f) = r.rim_far {
                    heads.push((f, u));
                }
                for (tip, into) in heads {
                    let (px, py) = (-into[1], into[0]);
                    let bx = tip[0] + into[0] * head;
                    let by = tip[1] + into[1] * head;
                    out.push_str(&format!(
                        "<polygon points=\"{:.3},{:.3} {:.3},{:.3} {:.3},{:.3}\" fill=\"black\"/>\n",
                        tip[0], fy(tip[1]),
                        bx + px * head * 0.32, fy(by + py * head * 0.32),
                        bx - px * head * 0.32, fy(by - py * head * 0.32)
                    ));
                }
                let tx = (r.label[0] + r.shoulder[0]) * 0.5;
                out.push_str(&format!(
                    "<text x=\"{:.3}\" y=\"{:.3}\" font-family=\"sans-serif\" font-size=\"3.2\" text-anchor=\"middle\" fill=\"black\">{}</text>\n",
                    tx, fy(r.label[1]) - 1.0, xml_escape(t)
                ));
                continue;
            }
        };
        let (dx, dy) = (g.p1[0] - g.p0[0], g.p1[1] - g.p0[1]);
        let l = (dx * dx + dy * dy).sqrt().max(1e-9);
        let (ux, uy) = (dx / l, dy / l);
        let (px, py) = (-uy, ux);
        out.push_str(&format!(
            "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.25\"/>\n",
            g.p0[0], fy(g.p0[1]), g.p1[0], fy(g.p1[1])
        ));
        // Witness lines, extended slightly past the dimension line as drafting expects.
        for (from, to) in [(g.a, g.p0), (g.b, g.p1)] {
            let ex = [to[0] + (to[0] - from[0]) * 0.0 + px * 1.2, to[1] + py * 1.2];
            out.push_str(&format!(
                "<path d=\"M{:.3},{:.3} L{:.3},{:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.18\"/>\n",
                from[0], fy(from[1]), ex[0], fy(ex[1])
            ));
        }
        let head = 1.6_f64;
        for (tip, s) in [(g.p0, 1.0), (g.p1, -1.0)] {
            let bx = tip[0] + ux * head * s;
            let by = tip[1] + uy * head * s;
            out.push_str(&format!(
                "<polygon points=\"{:.3},{:.3} {:.3},{:.3} {:.3},{:.3}\" fill=\"black\"/>\n",
                tip[0], fy(tip[1]),
                bx + px * head * 0.32, fy(by + py * head * 0.32),
                bx - px * head * 0.32, fy(by - py * head * 0.32)
            ));
        }
        out.push_str(&format!(
            "<text x=\"{:.3}\" y=\"{:.3}\" font-family=\"sans-serif\" font-size=\"3.2\" text-anchor=\"middle\" fill=\"black\">{}</text>\n",
            g.label[0], fy(g.label[1]) - 1.0, xml_escape(text)
        ));
    }

    // Title block, bottom-right inside the border.
    if !title.is_empty() {
        let rows = title.len() as f64;
        let (bw, rh) = (85.0f64, 7.0f64);
        let bh = rows * rh;
        let (bx, by) = (sheet_w - 10.0 - bw, sheet_h - 10.0 - bh);
        out.push_str(&format!(
            "<rect x=\"{bx:.3}\" y=\"{by:.3}\" width=\"{bw:.3}\" height=\"{bh:.3}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"/>\n"
        ));
        for (i, (k, v)) in title.iter().enumerate() {
            let y = by + (i as f64 + 0.72) * rh;
            if i > 0 {
                let ly = by + i as f64 * rh;
                out.push_str(&format!(
                    "<line x1=\"{bx:.3}\" y1=\"{ly:.3}\" x2=\"{:.3}\" y2=\"{ly:.3}\" stroke=\"black\" stroke-width=\"0.25\"/>\n",
                    bx + bw
                ));
            }
            out.push_str(&format!(
                "<text x=\"{:.3}\" y=\"{y:.3}\" font-family=\"sans-serif\" font-size=\"3\" fill=\"black\">{}</text>\n",
                bx + 2.0,
                xml_escape(k)
            ));
            out.push_str(&format!(
                "<text x=\"{:.3}\" y=\"{y:.3}\" font-family=\"sans-serif\" font-size=\"3.4\" fill=\"black\">{}</text>\n",
                bx + 26.0,
                xml_escape(v)
            ));
        }
    }
    out.push_str("</svg>\n");
    out
}

/// Escape the five XML metacharacters so a part name with an ampersand can't corrupt the file.
fn xml_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&apos;"),
            _ => o.push(c),
        }
    }
    o
}

// ---------------------------------------------------------------------------
// Dimension references: what a drawing dimension attaches to
// ---------------------------------------------------------------------------

/// The kind of geometry a dimension end is attached to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefKind {
    /// A corner — an endpoint shared by feature edges.
    Vertex,
    /// A straight edge, referenced at its midpoint and carrying its direction.
    Edge,
    /// A circular rim (hole or boss), carrying centre, axis and radius.
    Circle,
}

/// Something in a view a dimension can be attached to, with both where it APPEARS on the
/// sheet and what it IS in the model.
#[derive(Clone, Copy, Debug)]
pub struct SnapTarget {
    pub kind: RefKind,
    /// Position in sheet units for this view, before the view's scale is applied — this is
    /// what a click is matched against, because it is what the user sees.
    pub sheet: [f64; 2],
    /// The model-space anchor: the corner, the edge midpoint, or the circle centre.
    pub model: [f64; 3],
    /// Edge direction, or circle axis. Zero for a vertex.
    pub dir: [f64; 3],
    /// Circle radius; 0 otherwise.
    pub radius: f64,
    /// Whether the target sits on visible geometry (vs behind material).
    pub hidden: bool,
}

/// A stored, associative attachment point for a dimension.
///
/// It keeps a **geometry sample** rather than any index into the mesh: tessellation is rebuilt
/// from scratch on every regeneration, so an index means nothing across edits. The same
/// approach already carries assembly mates and region picks through rebuilds.
///
/// Resolution is nearest-match against the current geometry, scored on position, kind, and
/// (for circles) radius. That survives the edits dimensions actually need to survive — a
/// fillet changing, a hole moving, a neighbouring feature being added. It cannot follow an
/// edit that moves the geometry further than it is from its neighbours; `resolve` returns
/// `None` there so the dimension can be shown as dangling rather than silently measuring the
/// wrong thing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimRef {
    pub kind: RefKind,
    pub point: [f64; 3],
    pub dir: [f64; 3],
    pub radius: f64,
}

impl DimRef {
    pub fn from_target(t: &SnapTarget) -> DimRef {
        DimRef { kind: t.kind, point: t.model, dir: t.dir, radius: t.radius }
    }
}

/// Where a [`DimRef`] currently sits, after resolving against rebuilt geometry.
#[derive(Clone, Copy, Debug)]
pub struct ResolvedRef {
    pub point: [f64; 3],
    pub dir: [f64; 3],
    pub radius: f64,
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn len3(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Everything in this view a dimension could attach to: corners, straight-edge midpoints, and
/// circular rims.
///
/// `hidden` is carried through so a pick can prefer visible geometry, and the sheet position
/// is the projected one, so matching a click means comparing what the user actually sees.
pub fn snap_targets(mesh: &TriMesh, edges: &[[[f32; 3]; 2]], basis: &ViewBasis) -> Vec<SnapTarget> {
    let mut out: Vec<SnapTarget> = Vec::new();
    if edges.is_empty() {
        return out;
    }
    let grid = DepthGrid::build(mesh, basis);
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(q[a] as f64);
            hi[a] = hi[a].max(q[a] as f64);
        }
    }
    let diag = if mesh.positions.is_empty() { 1.0 } else { len3(sub3(hi, lo)).max(1e-9) };
    let eps = diag * 1e-4;
    let weld = diag * 1e-5;

    let mut push = |kind: RefKind, model: [f64; 3], dir: [f64; 3], radius: f64, out: &mut Vec<SnapTarget>| {
        let (sheet, depth) = basis.project(model);
        // Nudge toward the viewer so a target sitting exactly on the surface isn't judged
        // hidden by its own face. Smaller depth is NEARER, so this subtracts — adding pushed
        // every target behind its own surface and reported the whole model hidden.
        let hidden = grid.occluded(sheet, depth - eps * 2.0, eps);
        if out.iter().any(|t| t.kind == kind && len3(sub3(t.model, model)) < weld) {
            return;
        }
        out.push(SnapTarget { kind, sheet, model, dir, radius, hidden });
    };

    for e in edges {
        let a = [e[0][0] as f64, e[0][1] as f64, e[0][2] as f64];
        let b = [e[1][0] as f64, e[1][1] as f64, e[1][2] as f64];
        let d = sub3(b, a);
        let l = len3(d);
        if l < weld {
            continue;
        }
        push(RefKind::Vertex, a, [0.0; 3], 0.0, &mut out);
        push(RefKind::Vertex, b, [0.0; 3], 0.0, &mut out);
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
        push(RefKind::Edge, mid, [d[0] / l, d[1] / l, d[2] / l], 0.0, &mut out);
    }

    for c in circle_rims(edges, weld) {
        push(RefKind::Circle, c.centre, c.axis, c.radius, &mut out);
    }
    out
}

/// A circular rim recovered from a closed chain of feature edges.
struct Rim {
    centre: [f64; 3],
    axis: [f64; 3],
    radius: f64,
}

/// Find circular rims: closed chains of edges whose points are equidistant from their own
/// centroid and lie in one plane. That is what a hole or a boss end presents, and it is what
/// a diameter dimension will attach to.
fn circle_rims(edges: &[[[f32; 3]; 2]], weld: f64) -> Vec<Rim> {
    // Weld endpoints so a chain can be walked.
    let key = |q: [f64; 3]| {
        let s = if weld > 0.0 { weld } else { 1e-9 };
        ((q[0] / s).round() as i64, (q[1] / s).round() as i64, (q[2] / s).round() as i64)
    };
    let mut pts: Vec<[f64; 3]> = Vec::new();
    let mut ids: std::collections::HashMap<(i64, i64, i64), usize> = std::collections::HashMap::new();
    let mut adj: Vec<Vec<usize>> = Vec::new();
    let mut id_of = |q: [f64; 3], pts: &mut Vec<[f64; 3]>, adj: &mut Vec<Vec<usize>>, ids: &mut std::collections::HashMap<(i64, i64, i64), usize>| {
        *ids.entry(key(q)).or_insert_with(|| {
            pts.push(q);
            adj.push(Vec::new());
            pts.len() - 1
        })
    };
    for e in edges {
        let a = [e[0][0] as f64, e[0][1] as f64, e[0][2] as f64];
        let b = [e[1][0] as f64, e[1][1] as f64, e[1][2] as f64];
        let (ia, ib) = (id_of(a, &mut pts, &mut adj, &mut ids), id_of(b, &mut pts, &mut adj, &mut ids));
        if ia != ib {
            adj[ia].push(ib);
            adj[ib].push(ia);
        }
    }
    // Walk components made only of degree-2 vertices — a clean closed chain.
    let mut seen = vec![false; pts.len()];
    let mut out = Vec::new();
    for start in 0..pts.len() {
        if seen[start] || adj[start].len() != 2 {
            continue;
        }
        let mut chain = vec![start];
        seen[start] = true;
        let mut prev = start;
        let mut cur = adj[start][0];
        let mut closed = false;
        while adj[cur].len() == 2 && !seen[cur] {
            seen[cur] = true;
            chain.push(cur);
            let nxt = if adj[cur][0] == prev { adj[cur][1] } else { adj[cur][0] };
            prev = cur;
            cur = nxt;
            if cur == start {
                closed = true;
                break;
            }
            if chain.len() > 100_000 {
                break;
            }
        }
        // A circle needs enough segments to be a curve rather than a polygon corner.
        if !closed || chain.len() < 8 {
            continue;
        }
        let n = chain.len() as f64;
        let mut c = [0.0f64; 3];
        for &i in &chain {
            for a in 0..3 {
                c[a] += pts[i][a];
            }
        }
        for a in 0..3 {
            c[a] /= n;
        }
        // Equidistant from the centroid?
        let rs: Vec<f64> = chain.iter().map(|&i| len3(sub3(pts[i], c))).collect();
        let rmean = rs.iter().sum::<f64>() / n;
        if rmean <= weld * 10.0 || rs.iter().any(|r| (r - rmean).abs() > rmean * 0.02) {
            continue;
        }
        // Planar? Take the axis from two spokes and check every point lies in that plane.
        let a0 = sub3(pts[chain[0]], c);
        let a1 = sub3(pts[chain[chain.len() / 4]], c);
        let axis = norm(cross(a0, a1));
        if len3(axis) < 0.5 {
            continue;
        }
        if chain.iter().any(|&i| dot(sub3(pts[i], c), axis).abs() > rmean * 0.02) {
            continue;
        }
        out.push(Rim { centre: c, axis, radius: rmean });
    }
    out
}

/// Re-find what a stored reference points at, in freshly rebuilt geometry.
///
/// Scores candidates on distance first, then agreement of kind, direction and radius, and
/// requires the winner to be within `tol` — so an edit that moves geometry further than that
/// yields `None` (a dangling dimension) instead of silently latching onto a neighbour.
pub fn resolve_ref(targets: &[SnapTarget], r: &DimRef, tol: f64) -> Option<ResolvedRef> {
    let mut best: Option<(f64, &SnapTarget)> = None;
    for t in targets {
        if t.kind != r.kind {
            continue;
        }
        let d = len3(sub3(t.model, r.point));
        if d > tol {
            continue;
        }
        // Direction agreement (edges are undirected, so compare absolutely); radius match for
        // circles. Both only break ties — position leads.
        let mut score = d;
        if r.kind == RefKind::Edge {
            let align = dot(t.dir, r.dir).abs().clamp(0.0, 1.0);
            score += (1.0 - align) * tol;
        }
        if r.kind == RefKind::Circle {
            score += (t.radius - r.radius).abs();
            let align = dot(t.axis_or_dir(), r.dir).abs().clamp(0.0, 1.0);
            score += (1.0 - align) * tol * 0.5;
        }
        if best.is_none_or(|(bs, _)| score < bs) {
            best = Some((score, t));
        }
    }
    best.map(|(_, t)| ResolvedRef { point: t.model, dir: t.dir, radius: t.radius })
}

impl SnapTarget {
    fn axis_or_dir(&self) -> [f64; 3] {
        self.dir
    }
}

/// The target nearest a click, in sheet units. Visible geometry wins ties within `bias` so
/// tracing an outline doesn't keep catching hidden detail behind it.
pub fn pick_target<'a>(targets: &'a [SnapTarget], sheet: [f64; 2], tol: f64) -> Option<&'a SnapTarget> {
    let mut best: Option<(f64, &SnapTarget)> = None;
    for t in targets {
        let d = ((t.sheet[0] - sheet[0]).powi(2) + (t.sheet[1] - sheet[1]).powi(2)).sqrt();
        if d > tol {
            continue;
        }
        // Prefer, in order: circles (a hole centre is what you usually want), then corners,
        // then edges; and visible over hidden. Encoded as a small penalty so position still
        // dominates.
        let kind_pen = match t.kind {
            RefKind::Circle => 0.0,
            RefKind::Vertex => tol * 0.15,
            RefKind::Edge => tol * 0.35,
        };
        let score = d + kind_pen + if t.hidden { tol * 0.5 } else { 0.0 };
        if best.is_none_or(|(bs, _)| score < bs) {
            best = Some((score, t));
        }
    }
    best.map(|(_, t)| t)
}

/// How a linear dimension measures. Mirrors the document's `DimStyle`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DimStyle {
    Aligned,
    Horizontal,
    Vertical,
}

/// A laid-out dimension, ready to draw: the dimension line, the witness lines back to the
/// measured points, the text anchor, and the measured value.
#[derive(Clone, Copy, Debug)]
pub struct DimGeom {
    /// The measured points themselves.
    pub a: [f64; 2],
    pub b: [f64; 2],
    /// The dimension line, offset off the geometry.
    pub p0: [f64; 2],
    pub p1: [f64; 2],
    /// Where the text sits.
    pub label: [f64; 2],
    pub value: f64,
}

/// Lay out a linear dimension between two projected points.
///
/// The value is measured in SHEET space, which is what a drawing states: an orthographic view
/// shows the projected length, and a feature that is foreshortened in this view should read
/// as its foreshortened size, not its true 3D length.
pub fn dim_geometry(a: [f64; 2], b: [f64; 2], style: DimStyle, offset: f64, slide: f64) -> DimGeom {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (dir, value) = match style {
        DimStyle::Aligned => {
            let l = (dx * dx + dy * dy).sqrt();
            if l > 1e-9 { ([dx / l, dy / l], l) } else { ([1.0, 0.0], 0.0) }
        }
        DimStyle::Horizontal => ([1.0, 0.0], dx.abs()),
        DimStyle::Vertical => ([0.0, 1.0], dy.abs()),
    };
    let perp = [-dir[1], dir[0]];
    let off = [perp[0] * offset, perp[1] * offset];
    // For an axis-locked style the dimension line spans the projection of each point onto
    // that axis, so the witness lines run square off the geometry as drafting expects.
    let (e0, e1) = match style {
        DimStyle::Aligned => (a, b),
        DimStyle::Horizontal => {
            let y = if offset >= 0.0 { a[1].max(b[1]) } else { a[1].min(b[1]) };
            ([a[0], y], [b[0], y])
        }
        DimStyle::Vertical => {
            let x = if offset >= 0.0 { a[0].max(b[0]) } else { a[0].min(b[0]) };
            ([x, a[1]], [x, b[1]])
        }
    };
    let p0 = [e0[0] + off[0], e0[1] + off[1]];
    let p1 = [e1[0] + off[0], e1[1] + off[1]];
    let mid = [(p0[0] + p1[0]) * 0.5 + dir[0] * slide, (p0[1] + p1[1]) * 0.5 + dir[1] * slide];
    DimGeom { a, b, p0, p1, label: mid, value }
}

/// Format a measured length the way a drawing states it: trailing zeros trimmed, but never
/// bare-integer when the value isn't one.
pub fn format_dim(value: f64) -> String {
    let r = (value * 100.0).round() / 100.0;
    if (r - r.round()).abs() < 1e-9 {
        format!("{}", r.round() as i64)
    } else {
        let t = format!("{r:.2}");
        t.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// A laid-out radius or diameter dimension.
///
/// Drafting convention: a leader bites the rim with an arrow, runs out to the text, and the
/// text sits on a short horizontal landing. A diameter takes the leader clean across the
/// circle with an arrow at each rim.
#[derive(Clone, Copy, Debug)]
pub struct RadialGeom {
    pub centre: [f64; 2],
    /// Where the arrow meets the rim, on the side the label was dragged to.
    pub rim: [f64; 2],
    /// The opposite rim — only for a diameter, which is arrowed at both ends.
    pub rim_far: Option<[f64; 2]>,
    /// Where the text sits.
    pub label: [f64; 2],
    /// Far end of the horizontal landing under the text.
    pub shoulder: [f64; 2],
    /// The measured value: the radius, or the full diameter.
    pub value: f64,
}

/// Lay out a radial dimension. `angle` is the leader's direction from the centre and `dist`
/// how far out the text sits; together they let the label be dragged anywhere around the
/// circle. `dist` is clamped so the text can never sit inside the rim, where the leader would
/// have nothing to point at.
pub fn radial_dim_geometry(centre: [f64; 2], radius: f64, angle: f64, dist: f64, diameter: bool) -> RadialGeom {
    let d = [angle.cos(), angle.sin()];
    let out = dist.max(radius * 1.25 + 1.0);
    let rim = [centre[0] + d[0] * radius, centre[1] + d[1] * radius];
    let label = [centre[0] + d[0] * out, centre[1] + d[1] * out];
    // The landing runs away from the circle, so the text never overhangs back over it.
    let side = if d[0] >= 0.0 { 1.0 } else { -1.0 };
    let shoulder = [label[0] + side * (radius * 0.5).clamp(1.5, 6.0), label[1]];
    RadialGeom {
        centre,
        rim,
        rim_far: diameter.then(|| [centre[0] - d[0] * radius, centre[1] - d[1] * radius]),
        label,
        shoulder,
        value: if diameter { radius * 2.0 } else { radius },
    }
}

/// A laid-out centre mark: the little cross at a hole's centre, with centrelines running out
/// past the rim.
#[derive(Clone, Debug)]
pub struct CentreMarkGeom {
    pub centre: [f64; 2],
    /// The short solid cross at the centre.
    pub cross: Vec<([f64; 2], [f64; 2])>,
    /// The centrelines reaching out past the rim, drawn dash-dot.
    pub arms: Vec<([f64; 2], [f64; 2])>,
}

/// Lay out a centre mark, drafting-style: a small cross on the centre, then a break, then a
/// centreline out through the rim by `overshoot`. Arms run along the sheet axes, which is the
/// convention for a hole on an orthographic view.
pub fn centre_mark_geometry(centre: [f64; 2], radius: f64, overshoot: f64) -> CentreMarkGeom {
    let r = radius.max(1e-6);
    // The cross stays small and readable; the arms scale with the hole.
    let cross_arm = (r * 0.28).clamp(0.4, 3.0);
    let inner = r * 0.55;
    let outer = r + overshoot.max(r * 0.15).min(r * 3.0);
    let mut cross = Vec::with_capacity(2);
    let mut arms = Vec::with_capacity(4);
    for (dx, dy) in [(1.0f64, 0.0f64), (0.0, 1.0)] {
        cross.push((
            [centre[0] - dx * cross_arm, centre[1] - dy * cross_arm],
            [centre[0] + dx * cross_arm, centre[1] + dy * cross_arm],
        ));
    }
    for (dx, dy) in [(1.0f64, 0.0f64), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        arms.push((
            [centre[0] + dx * inner, centre[1] + dy * inner],
            [centre[0] + dx * outer, centre[1] + dy * outer],
        ));
    }
    CentreMarkGeom { centre, cross, arms }
}

/// A dimension ready for export, in final sheet coordinates.
pub enum SheetDim {
    Linear(DimGeom, String),
    Radial(RadialGeom, String),
    Mark(CentreMarkGeom),
}

// ---------------------------------------------------------------------------
// Section views
// ---------------------------------------------------------------------------

/// A part cut by a plane, ready to draw as a section view.
pub struct SectionCut {
    /// The half of the part BEHIND the plane — what a section view shows.
    pub mesh: TriMesh,
    /// 45-degree hatch lines across the exposed cut face, in model space.
    pub hatch: Vec<[[f32; 3]; 2]>,
}

/// Cut `mesh` with the plane through `point` with normal `n`, keeping the half the normal
/// points AWAY from, and hatch the exposed face.
///
/// The cut is a boolean against a half-space box, the same way the 3D section view does it —
/// a real solid, so the projector's hidden-line removal works on it unchanged and the cut
/// face is genuine geometry rather than an overlay.
pub fn section_cut(mesh: &TriMesh, point: [f64; 3], n: [f64; 3], hatch_spacing: f64) -> Option<SectionCut> {
    if mesh.indices.len() < 3 {
        return None;
    }
    let n = norm(n);
    if len3(n) < 0.5 {
        return None;
    }
    // A frame on the cutting plane.
    let seed = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
    let u = norm(cross(seed, n));
    let v = norm(cross(n, u));

    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &mesh.positions {
        for a in 0..3 {
            lo[a] = lo[a].min(q[a] as f64);
            hi[a] = hi[a].max(q[a] as f64);
        }
    }
    let l = len3(sub3(hi, lo)).max(1.0);

    // A box covering everything on the +n side of the plane, subtracted away.
    let basis = crate::PlaneBasis { origin: point, u, v, normal: n };
    let sq = vec![[-2.0 * l, -2.0 * l], [2.0 * l, -2.0 * l], [2.0 * l, 2.0 * l], [-2.0 * l, 2.0 * l]];
    let tool = crate::extrude_tool_mesh(&sq, &[], &basis, 0.0, 2.0 * l)?;
    let cut = crate::mesh_difference(mesh, &tool);
    let _ = crate::take_fallback_count(); // a section is a view, never a torn-surface warning
    if cut.indices.len() < 3 {
        return None;
    }

    // Hatch the exposed face: the triangles the boolean left lying ON the plane, facing the
    // side that was removed. A pre-existing face that merely touches the plane from behind
    // faces the other way and is skipped, so it isn't hatched as if it were cut.
    let eps = (l * 5e-4).max(1e-4);
    let d0 = dot(point, n);
    let sweep = norm([u[0] - v[0], u[1] - v[1], u[2] - v[2]]); // hatch lines run 45° to u
    let along = norm(cross(n, sweep));
    let step = if hatch_spacing > 1e-6 { hatch_spacing } else { (l / 45.0).max(1e-3) };
    let mut hatch: Vec<[[f32; 3]; 2]> = Vec::new();
    for t in cut.indices.chunks_exact(3) {
        let p: Vec<[f64; 3]> = t
            .iter()
            .map(|&i| {
                let q = cut.positions[i as usize];
                [q[0] as f64, q[1] as f64, q[2] as f64]
            })
            .collect();
        if p.iter().any(|q| (dot(*q, n) - d0).abs() > eps) {
            continue;
        }
        let gn = cross(sub3(p[1], p[0]), sub3(p[2], p[0]));
        if len3(gn) < 1e-12 || dot(norm(gn), n) < 0.5 {
            continue;
        }
        // Sweep hatch planes across the triangle and clip each to it.
        let w: Vec<f64> = p.iter().map(|q| dot(*q, sweep)).collect();
        let (wmin, wmax) = (w[0].min(w[1]).min(w[2]), w[0].max(w[1]).max(w[2]));
        let mut k = (wmin / step).ceil();
        while k * step <= wmax {
            let wk = k * step;
            k += 1.0;
            // Where the hatch plane crosses this triangle's edges.
            let mut xs: Vec<[f64; 3]> = Vec::new();
            for e in 0..3 {
                let (a, b) = (e, (e + 1) % 3);
                let (wa, wb) = (w[a], w[b]);
                if (wa - wk) * (wb - wk) > 0.0 || (wa - wb).abs() < 1e-12 {
                    continue;
                }
                let tt = (wk - wa) / (wb - wa);
                xs.push([
                    p[a][0] + (p[b][0] - p[a][0]) * tt,
                    p[a][1] + (p[b][1] - p[a][1]) * tt,
                    p[a][2] + (p[b][2] - p[a][2]) * tt,
                ]);
            }
            if xs.len() < 2 {
                continue;
            }
            // Two crossings bound the segment; sort along the line so it spans them.
            xs.sort_by(|x, y| dot(*x, along).total_cmp(&dot(*y, along)));
            let (a, b) = (xs[0], xs[xs.len() - 1]);
            if len3(sub3(b, a)) > 1e-6 {
                hatch.push([[a[0] as f32, a[1] as f32, a[2] as f32], [b[0] as f32, b[1] as f32, b[2] as f32]]);
            }
        }
    }
    Some(SectionCut { mesh: cut, hatch })
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

    #[test]
    fn svg_is_well_formed_and_scales() {
        let edges = [
            ProjEdge { a: [0.0, 0.0], b: [10.0, 0.0], hidden: false },
            ProjEdge { a: [0.0, 0.0], b: [0.0, 10.0], hidden: true },
        ];
        let items = [SheetItem { edges: &edges, center: [100.0, 100.0], scale: 2.0, show_hidden: true, label: Some("Front".into()), hatch: &[] }];
        let svg = to_svg(297.0, 210.0, &items, &[("PART".into(), "bracket & pin".into())]);
        assert!(svg.starts_with("<svg"), "no svg root");
        assert!(svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("<svg").count(), 1);
        // The visible edge runs 10mm at 2x from x=100, so it must end at x=120.
        assert!(svg.contains("M100.000,110.000 L120.000,110.000"), "solid edge misplaced:\n{svg}");
        // Hidden edges are dashed, and only drawn when asked for.
        assert!(svg.contains("stroke-dasharray"));
        let no_hidden = to_svg(297.0, 210.0, &[SheetItem { edges: &edges, center: [100.0, 100.0], scale: 2.0, show_hidden: false, label: None, hatch: &[] }], &[]);
        assert!(!no_hidden.contains("stroke-dasharray"), "hidden lines drawn when switched off");
        // A part name with an ampersand must not produce invalid XML.
        assert!(svg.contains("bracket &amp; pin"));
        assert!(!svg.contains("bracket & pin"));
    }

    /// Sheet coordinates run up from the bottom-left; SVG runs down. A view near the sheet
    /// bottom must land near the BOTTOM of the file, not the top.
    #[test]
    fn svg_flips_the_y_axis() {
        let e = [ProjEdge { a: [0.0, 0.0], b: [1.0, 0.0], hidden: false }];
        let low = to_svg(100.0, 100.0, &[SheetItem { edges: &e, center: [50.0, 10.0], scale: 1.0, show_hidden: false, label: None, hatch: &[] }], &[]);
        assert!(low.contains("M50.000,90.000"), "y not flipped:\n{low}");
    }

    /// A cylinder seen from the SIDE must draw its barrel outline. Its silhouette is not a
    /// sharp edge in the mesh — it depends on where you look from — so projecting only the
    /// feature edges leaves the cylinder invisible.
    #[test]
    fn a_cylinder_shows_its_silhouette_from_the_side() {
        // A 10mm-radius, 30mm-long cylinder along +Z.
        let mut sk = crate::TriMesh::default();
        let _ = &mut sk;
        let n = 64;
        let circle: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [10.0 * a.cos(), 10.0 * a.sin()]
            })
            .collect();
        let solid = extrude_solid(&circle, &[], &xy_basis(), 30.0).expect("cylinder");
        let tess = tessellate(&solid, 0.05);
        let t = mesh_tessellation(tess.mesh.clone());

        // Look along -X: the sheet should show the 30mm length across and 20mm diameter up.
        let basis = ViewBasis::looking_along([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = project_edges(&t.mesh, &t.edges, &basis, 24);
        assert!(!proj.is_empty(), "cylinder projected nothing at all");
        let (lo, hi) = edges_bounds(&proj);
        let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
        assert!((w - 30.0).abs() < 0.3, "width {w:.2}, wanted the 30mm length");
        assert!((h - 20.0).abs() < 0.3, "height {h:.2}, wanted the 20mm diameter — the barrel silhouette is missing");
    }

    /// The NEAR face must be the visible one. A cube is symmetric, so the earlier
    /// length-based tests pass whether or not the depth test is inverted — this one names
    /// which face won.
    #[test]
    fn the_near_face_is_the_visible_one() {
        let sq = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 10.0).expect("cube");
        let tess = tessellate(&solid, 0.05);
        let t = mesh_tessellation(tess.mesh.clone());

        // Viewer on +Z looking toward -Z, so the z=10 face is nearest.
        let basis = ViewBasis::looking_along([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]);

        // Classify each ORIGINAL edge by the z of its midpoint, then ask what the projection
        // said about it. Project one edge at a time so runs map back unambiguously.
        let (mut near_vis, mut near_hid, mut far_vis, mut far_hid) = (0, 0, 0, 0);
        for e in &t.edges {
            let zmid = (e[0][2] + e[1][2]) * 0.5;
            let runs = project_edges(&t.mesh, std::slice::from_ref(e), &basis, 24);
            let vis = runs.iter().any(|r| !r.hidden);
            if zmid > 9.9 {
                if vis { near_vis += 1 } else { near_hid += 1 }
            } else if zmid < 0.1 {
                if vis { far_vis += 1 } else { far_hid += 1 }
            }
        }
        eprintln!("near face: {near_vis} visible / {near_hid} hidden;  far face: {far_vis} visible / {far_hid} hidden");
        assert!(near_vis > 0 && near_hid == 0, "the NEAR face should be fully visible ({near_vis} vis / {near_hid} hidden)");
        assert!(far_hid > 0 && far_vis == 0, "the FAR face should be fully hidden ({far_vis} vis / {far_hid} hidden)");
    }

    /// Build a box `w` x `d` x `h` and return the pieces a view needs.
    fn box_geom(w: f64, d: f64, h: f64) -> (TriMesh, Vec<[[f32; 3]; 2]>) {
        let sq = vec![[0.0, 0.0], [w, 0.0], [w, d], [0.0, d]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), h).expect("box");
        let t = mesh_tessellation(tessellate(&solid, 0.05).mesh);
        (t.mesh, t.edges)
    }

    fn front() -> ViewBasis {
        ViewBasis::looking_along([0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
    }

    /// A box must offer its eight corners and twelve edge midpoints to attach to.
    #[test]
    fn snap_targets_cover_a_box() {
        let (mesh, edges) = box_geom(20.0, 15.0, 10.0);
        let ts = snap_targets(&mesh, &edges, &front());

        let corners: Vec<&SnapTarget> = ts.iter().filter(|t| t.kind == RefKind::Vertex).collect();
        assert_eq!(corners.len(), 8, "expected 8 corners, got {}", corners.len());
        let mids: Vec<&SnapTarget> = ts.iter().filter(|t| t.kind == RefKind::Edge).collect();
        assert_eq!(mids.len(), 12, "expected 12 edge midpoints, got {}", mids.len());

        // Every corner of the box is present.
        for c in [
            [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 15.0, 0.0], [0.0, 15.0, 0.0],
            [0.0, 0.0, 10.0], [20.0, 0.0, 10.0], [20.0, 15.0, 10.0], [0.0, 15.0, 10.0],
        ] {
            assert!(corners.iter().any(|t| len3(sub3(t.model, c)) < 1e-6), "missing corner {c:?}");
        }
        // Corners on the far face are behind the near one.
        assert!(corners.iter().any(|t| t.hidden), "no corner reported hidden");
        assert!(corners.iter().any(|t| !t.hidden), "no corner reported visible");
    }

    /// A hole's rim must be offered as a Circle, with its true centre and radius — that is
    /// what a diameter dimension will attach to.
    #[test]
    fn snap_targets_find_a_hole_rim() {
        let outer = vec![[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]];
        let n = 48;
        let hole: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [20.0 + 6.0 * a.cos(), 20.0 + 6.0 * a.sin()]
            })
            .collect();
        let solid = extrude_solid(&outer, &[hole], &xy_basis(), 8.0).expect("plate with hole");
        let t = mesh_tessellation(tessellate(&solid, 0.05).mesh);
        let ts = snap_targets(&t.mesh, &t.edges, &front());

        let circles: Vec<&SnapTarget> = ts.iter().filter(|c| c.kind == RefKind::Circle).collect();
        assert!(!circles.is_empty(), "the hole rim was not offered as a circle");
        let top = circles
            .iter()
            .find(|c| (c.model[2] - 8.0).abs() < 0.1)
            .expect("no rim on the top face");
        assert!((top.radius - 6.0).abs() < 0.05, "radius {:.3}, wanted 6", top.radius);
        assert!((top.model[0] - 20.0).abs() < 0.05 && (top.model[1] - 20.0).abs() < 0.05, "centre off: {:?}", top.model);
        // Its axis is the hole's, i.e. the extrude direction.
        assert!(dot(top.dir, [0.0, 0.0, 1.0]).abs() > 0.99, "axis not along Z: {:?}", top.dir);
    }

    /// THE associativity test: a reference taken on one build must resolve to the SAME
    /// feature after the part is rebuilt at a different size — following the geometry to its
    /// new position rather than staying at the old coordinates or latching onto a neighbour.
    #[test]
    fn a_reference_follows_the_geometry_through_an_edit() {
        let (m1, e1) = box_geom(20.0, 15.0, 10.0);
        let t1 = snap_targets(&m1, &e1, &front());

        // Attach to the top-front-right corner (20, 0, 10).
        let picked = t1
            .iter()
            .find(|t| t.kind == RefKind::Vertex && len3(sub3(t.model, [20.0, 0.0, 10.0])) < 1e-6)
            .expect("corner not offered");
        let r = DimRef::from_target(picked);

        // Rebuild 2mm taller and 2mm wider: that corner moves to (22, 0, 12).
        let (m2, e2) = box_geom(22.0, 15.0, 12.0);
        let t2 = snap_targets(&m2, &e2, &front());
        let got = resolve_ref(&t2, &r, 5.0).expect("reference went dangling on a small edit");
        assert!(
            len3(sub3(got.point, [22.0, 0.0, 12.0])) < 1e-6,
            "resolved to {:?}, wanted the moved corner (22, 0, 12)",
            got.point
        );
        // And emphatically NOT the stale coordinates.
        assert!(len3(sub3(got.point, r.point)) > 1.0, "resolved back to the old position");
    }

    /// An edit that moves geometry further than the tolerance must go dangling, not silently
    /// grab a neighbouring corner and measure something else.
    #[test]
    fn a_reference_goes_dangling_rather_than_grabbing_a_neighbour() {
        let (m1, e1) = box_geom(20.0, 15.0, 10.0);
        let t1 = snap_targets(&m1, &e1, &front());
        let picked = t1
            .iter()
            .find(|t| t.kind == RefKind::Vertex && len3(sub3(t.model, [20.0, 0.0, 10.0])) < 1e-6)
            .unwrap();
        let r = DimRef::from_target(picked);

        // Rebuild far larger, with a tight tolerance: nothing is close enough to be honest.
        let (m2, e2) = box_geom(60.0, 15.0, 40.0);
        let t2 = snap_targets(&m2, &e2, &front());
        assert!(resolve_ref(&t2, &r, 2.0).is_none(), "latched onto a neighbour instead of going dangling");

        // A kind mismatch never resolves either.
        let edge_ref = DimRef { kind: RefKind::Edge, ..r };
        let vertex_only: Vec<SnapTarget> = t2.iter().copied().filter(|t| t.kind == RefKind::Vertex).collect();
        assert!(resolve_ref(&vertex_only, &edge_ref, 100.0).is_none(), "an Edge ref matched a Vertex target");
    }

    /// A click on the sheet resolves to the target under it, preferring a hole centre over
    /// the edges around it, and visible geometry over hidden.
    #[test]
    fn picking_prefers_the_useful_target() {
        let (mesh, edges) = box_geom(20.0, 15.0, 10.0);
        let ts = snap_targets(&mesh, &edges, &front());

        // The front-top-right corner projects to sheet (20, 15) in this view.
        let (sheet, _) = front().project([20.0, 15.0, 10.0]);
        let hit = pick_target(&ts, sheet, 1.0).expect("nothing picked at a corner");
        assert_eq!(hit.kind, RefKind::Vertex, "expected the corner, got {:?}", hit.kind);
        assert!(!hit.hidden, "picked the hidden corner behind the visible one");

        // Far from anything, nothing is picked.
        assert!(pick_target(&ts, [500.0, 500.0], 1.0).is_none(), "picked something from far away");
    }

    /// Cutting a box in half must leave half the volume and hatch the exposed face.
    #[test]
    fn a_section_cut_halves_the_part_and_hatches_the_face() {
        let sq = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 10.0).expect("box");
        let mesh = tessellate(&solid, 0.05).mesh;
        let full = crate::signed_mesh_volume(&mesh).abs();
        assert!((full - 4000.0).abs() < 5.0, "box volume {full:.1}");

        // Cut at x = 10, keeping the -X half.
        let cut = section_cut(&mesh, [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0).expect("section");
        let half = crate::signed_mesh_volume(&cut.mesh).abs();
        assert!((half - 2000.0).abs() < 20.0, "kept {half:.1}, wanted half of {full:.1}");

        // The hatch must exist and lie ON the cut plane.
        assert!(!cut.hatch.is_empty(), "the cut face wasn't hatched");
        for h in &cut.hatch {
            for q in h {
                assert!((q[0] as f64 - 10.0).abs() < 0.05, "a hatch line left the cut plane: {q:?}");
            }
        }
        // ...and inside the face, not running off it.
        for h in &cut.hatch {
            for q in h {
                assert!(q[1] >= -0.05 && q[1] <= 20.05 && q[2] >= -0.05 && q[2] <= 10.05, "hatch outside the face: {q:?}");
            }
        }
    }

    /// Flipping the normal keeps the OTHER half — the two together make the whole part.
    #[test]
    fn flipping_a_section_keeps_the_other_half() {
        let sq = vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]];
        let solid = extrude_solid(&sq, &[], &xy_basis(), 10.0).expect("box");
        let mesh = tessellate(&solid, 0.05).mesh;

        let a = section_cut(&mesh, [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2.0).expect("a");
        let b = section_cut(&mesh, [10.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2.0).expect("b");
        let (va, vb) = (crate::signed_mesh_volume(&a.mesh).abs(), crate::signed_mesh_volume(&b.mesh).abs());
        assert!((va + vb - 4000.0).abs() < 40.0, "the halves ({va:.0} + {vb:.0}) don't make the whole");
        // Each half sits on its own side of the cut.
        let bbox = |m: &TriMesh| {
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for q in &m.positions {
                for k in 0..3 {
                    lo[k] = lo[k].min(q[k] as f64);
                    hi[k] = hi[k].max(q[k] as f64);
                }
            }
            (lo, hi)
        };
        let (lo_a, hi_a) = bbox(&a.mesh);
        let (lo_b, _) = bbox(&b.mesh);
        assert!(hi_a[0] < 10.05 && lo_a[0] < 1.0, "the kept half is on the wrong side: {lo_a:?}..{hi_a:?}");
        assert!(lo_b[0] > 9.95, "the flipped half is on the wrong side: {lo_b:?}");
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
