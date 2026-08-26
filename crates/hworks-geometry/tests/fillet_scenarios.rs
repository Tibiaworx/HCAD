//! Scenario tests for the fillet tool, run end to end the way regen runs it: mesh surgery
//! first, the CSG round as the fallback — because that pair is what a user's click reaches,
//! and a fillet that is only right in one engine is a fillet that is sometimes wrong.
//!
//! Every scenario asserts the same four things, because every reported fillet bug has been a
//! failure of one of them:
//!
//!  * the result is a 2-manifold solid;
//!  * it moves exactly the material a rolling ball moves (Pappus / analytic volume);
//!  * the fillet BAND is smooth: every band vertex lies on the ideal surface (cylinder,
//!    torus, sphere corner, or swept tube), and no crease inside the band is sharper than
//!    the band's own tessellation steps — a fold or a stripe is a 90-degree wall of them;
//!  * the edges work: the picked rim is present in the base body's selectable display edges,
//!    the fillet emits its seam lines (bevel_feature_edges), and the filleted body draws NO
//!    sharp display edge inside the band — a striped or creased band lights up with them.

use hworks_geometry::{
    bevel_feature_edges, bevel_mesh_selected, extrude_tool_mesh, is_manifold, mesh_tessellation,
    mesh_union, round_mesh, PlaneBasis, TriMesh,
};

// ---------------------------------------------------------------------------------- vector bits

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn xy() -> PlaneBasis {
    PlaneBasis { origin: [0.0; 3], u: [1.0, 0.0, 0.0], v: [0.0, 1.0, 0.0], normal: [0.0, 0.0, 1.0] }
}

fn vol(m: &TriMesh) -> f64 {
    let mut v = 0.0f64;
    for t in m.indices.chunks_exact(3) {
        let g = |i: u32| {
            let q = m.positions[i as usize];
            [q[0] as f64, q[1] as f64, q[2] as f64]
        };
        v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
    }
    v.abs()
}

/// The rolling ball's cross-section area and its centroid's distance in from the wall —
/// the two numbers every Pappus check is made of.
fn sliver_area(r: f64) -> f64 {
    (1.0 - std::f64::consts::PI / 4.0) * r * r
}
fn sliver_ubar(r: f64) -> f64 {
    r * (5.0 / 6.0 - std::f64::consts::PI / 4.0) / (1.0 - std::f64::consts::PI / 4.0)
}

fn point_seg_dist(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub(b, a);
    let l2 = dot(d, d);
    if l2 < 1e-18 {
        return len(sub(p, a));
    }
    let t = (dot(sub(p, a), d) / l2).clamp(0.0, 1.0);
    len(sub(p, [a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t]))
}

// ------------------------------------------------------------------------------- the app's path

/// Fillet the way regenerate_mesh does: the surgery engine, with the CSG round as fallback.
fn app_fillet(body: &TriMesh, r: f64, picked: &[Vec<[f64; 3]>]) -> (TriMesh, &'static str) {
    if let Some(m) = bevel_mesh_selected(body, r, 12, picked) {
        return (m, "surgery");
    }
    let m = round_mesh(body, r, picked).expect("both fillet engines declined the pick");
    (m, "csg")
}

// ------------------------------------------------------------------------------ quality helpers

/// Worst deviation of any vertex satisfying `pred` from the ideal surface (`dist` returns the
/// deviation at a point). Returns (worst, how many vertices were measured) — a scenario must
/// also check the count, because a predicate that matches nothing proves nothing.
fn band_worst(m: &TriMesh, pred: impl Fn([f64; 3]) -> bool, dist: impl Fn([f64; 3]) -> f64) -> (f64, usize) {
    let (mut worst, mut n) = (0.0f64, 0usize);
    for q in &m.positions {
        let p = [q[0] as f64, q[1] as f64, q[2] as f64];
        if pred(p) {
            worst = worst.max(dist(p));
            n += 1;
        }
    }
    (worst, n)
}

/// The sharpest crease (dihedral angle, degrees) between adjacent triangles that BOTH sit in
/// the band. Coincident duplicated sheets read as 180 and a fold as ~90+, so this is the test
/// that sees what volume cannot.
///
/// `min_feat` is the smallest triangle altitude that counts: a crease is only a crease if it
/// has visible extent. The CSG path leaves hairline steps where two tangent tools' different
/// tessellations meet (measured 0.007 on an r=1 box-rim corner, sphere against cylinder) —
/// walls the display tessellation does not even draw. A real stripe is an arc step tall,
/// ~13% of r, and a fold is full-size, so a floor of a couple percent of r hides neither.
fn max_band_crease(m: &TriMesh, pred: impl Fn([f64; 3]) -> bool, min_feat: f64) -> f64 {
    max_band_crease_at(m, pred, min_feat).0
}

/// As `max_band_crease`, but also says WHERE — the midpoint of the worst edge.
fn max_band_crease_at(m: &TriMesh, pred: impl Fn([f64; 3]) -> bool, min_feat: f64) -> (f64, [f64; 3]) {
    use std::collections::HashMap;
    let p = |i: u32| {
        let q = m.positions[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    let key = |q: [f64; 3]| ((q[0] * 1e4).round() as i64, (q[1] * 1e4).round() as i64, (q[2] * 1e4).round() as i64);
    let mut edge_owners: HashMap<((i64, i64, i64), (i64, i64, i64)), Vec<[f64; 3]>> = HashMap::new();
    for t in m.indices.chunks_exact(3) {
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        if !(pred(a) && pred(b) && pred(c)) {
            continue;
        }
        let n = cross(sub(b, a), sub(c, a));
        let nl = len(n);
        let longest = len(sub(b, a)).max(len(sub(c, b))).max(len(sub(a, c)));
        if longest < 1e-9 || nl / longest < min_feat.max(1e-3) {
            continue; // a needle, or a sub-visible hairline wall
        }
        let nrm = [n[0] / nl, n[1] / nl, n[2] / nl];
        for (u, v) in [(a, b), (b, c), (c, a)] {
            let (ku, kv) = (key(u), key(v));
            let e = if ku <= kv { (ku, kv) } else { (kv, ku) };
            edge_owners.entry(e).or_default().push(nrm);
        }
    }
    let (mut worst, mut at) = (0.0f64, [0.0f64; 3]);
    for (e, owners) in &edge_owners {
        for i in 0..owners.len() {
            for j in i + 1..owners.len() {
                let d = dot(owners[i], owners[j]).clamp(-1.0, 1.0);
                let ang = d.acos().to_degrees();
                if ang > worst {
                    worst = ang;
                    at = [
                        (e.0 .0 + e.1 .0) as f64 * 0.5e-4,
                        (e.0 .1 + e.1 .1) as f64 * 0.5e-4,
                        (e.0 .2 + e.1 .2) as f64 * 0.5e-4,
                    ];
                }
            }
        }
    }
    (worst, at)
}

/// The picked polylines must be clickable: every pick segment lies on a sharp display edge of
/// the base body's tessellation (that is what the app lets you select in the viewport).
fn assert_pick_selectable(base: &TriMesh, picked: &[Vec<[f64; 3]>], label: &str) {
    let t = mesh_tessellation(base.clone());
    for chain in picked {
        for w in chain.windows(2) {
            let mid = [(w[0][0] + w[1][0]) * 0.5, (w[0][1] + w[1][1]) * 0.5, (w[0][2] + w[1][2]) * 0.5];
            let hit = t.edges.iter().any(|e| {
                point_seg_dist(
                    mid,
                    [e[0][0] as f64, e[0][1] as f64, e[0][2] as f64],
                    [e[1][0] as f64, e[1][1] as f64, e[1][2] as f64],
                ) < 0.01
            });
            assert!(
                hit,
                "{label}: pick segment near ({:.2},{:.2},{:.2}) is not on any selectable display edge",
                mid[0], mid[1], mid[2]
            );
        }
    }
}

/// The fillet's seam lines must be emitted (they become the selectable edges after regen) and
/// every endpoint must lie on the expected tangent locus.
fn assert_seams(
    base: &TriMesh,
    r: f64,
    picked: &[Vec<[f64; 3]>],
    on_locus: impl Fn([f64; 3]) -> f64,
    tol: f64,
    label: &str,
) {
    let seams = bevel_feature_edges(base, r, picked);
    assert!(!seams.is_empty(), "{label}: the fillet emits no seam edges — nothing selectable after regen");
    let mut worst = 0.0f64;
    for e in &seams {
        for q in e {
            worst = worst.max(on_locus([q[0] as f64, q[1] as f64, q[2] as f64]));
        }
    }
    assert!(worst < tol, "{label}: a seam endpoint sits {worst:.4} off the tangent locus (tol {tol:.4})");
}

/// A smooth band draws no sharp display edge in its interior. This is the user-visible form of
/// the crease check: stripes on a fillet are precisely sharp edges the tessellation found there.
fn assert_no_sharp_edges_in(m: &TriMesh, pred: impl Fn([f64; 3]) -> bool, label: &str) {
    let t = mesh_tessellation(m.clone());
    let inside: Vec<_> = t
        .edges
        .iter()
        .filter(|e| e.iter().all(|q| pred([q[0] as f64, q[1] as f64, q[2] as f64])))
        .collect();
    assert!(
        inside.is_empty(),
        "{label}: {} sharp display edge(s) inside the fillet band, first near ({:.2},{:.2},{:.2}) — the band is creased or striped",
        inside.len(),
        inside[0][0][0],
        inside[0][0][1],
        inside[0][0][2]
    );
}

// ============================================================================== the scenarios

/// A straight convex edge on a box — the simplest fillet there is, and the baseline every
/// other scenario degrades from. The band is a quarter-cylinder; both ends stop square in the
/// side walls.
#[test]
fn straight_box_edge_fillets_smoothly() {
    let (lx, ly, h) = (20.0f64, 14.0, 6.0);
    let body = extrude_tool_mesh(&[[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]], &[], &xy(), 0.0, h).unwrap();
    let picked = vec![vec![[0.0, ly, h], [lx, ly, h]]];
    assert_pick_selectable(&body, &picked, "box edge");
    for &r in &[0.5f64, 1.5] {
        let (m, engine) = app_fillet(&body, r, &picked);
        assert!(is_manifold(&m), "box edge r={r} [{engine}]: not manifold");
        let removed = vol(&body) - vol(&m);
        let want = sliver_area(r) * lx;
        assert!(
            (removed - want).abs() < want * 0.02,
            "box edge r={r} [{engine}]: removed {removed:.4}, a fillet takes {want:.4}"
        );

        // The vertex measure includes the strip's end stations (a straight edge's band has
        // vertices ONLY there) — they sit in the side walls, still on the ideal cylinder. The
        // sharp-edge check below excludes them: the terminal arcs where the band meets those
        // walls ARE sharp, and rightly so.
        let margin = r * 0.05;
        let pred = |p: [f64; 3]| p[1] > ly - r + margin && p[2] > h - r + margin;
        let interior = |p: [f64; 3]| pred(p) && p[0] > 1e-3 && p[0] < lx - 1e-3;
        let axis = |p: [f64; 3]| (((p[1] - (ly - r)).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs();
        let (worst, n) = band_worst(&m, pred, axis);
        assert!(n > 8, "box edge r={r} [{engine}]: only {n} band vertices measured");
        assert!(worst < r * 0.02, "box edge r={r} [{engine}]: band strays {worst:.4} from the tangent cylinder");
        let crease = max_band_crease(
            &m,
            |p| p[1] > ly - r - 0.02 && p[2] > h - r - 0.02 && p[0] > r * 0.5 && p[0] < lx - r * 0.5,
            0.02 * r,
        );
        assert!(crease < 20.0, "box edge r={r} [{engine}]: {crease:.1} degree crease inside the band");
        assert_no_sharp_edges_in(&m, interior, &format!("box edge r={r} [{engine}]"));

        // Seams: the two tangent lines, full length.
        let seam_locus = |p: [f64; 3]| {
            let on_cap = ((p[1] - (ly - r)).abs()).max((p[2] - h).abs());
            let on_wall = ((p[1] - ly).abs()).max((p[2] - (h - r)).abs());
            on_cap.min(on_wall)
        };
        assert_seams(&body, r, &picked, seam_locus, 1e-3, &format!("box edge r={r}"));
    }
}

/// The whole top rim of a box: four straight fillets that must MEET — each corner blends into
/// a sphere patch. The ideal surface for band and corners together is one shape: distance r
/// from the top face's outline inset by r (a capsule offset of that rectangle).
///
/// This is the pick that exposed the corner-weld bug: the surgery used to blend two
/// perpendicular end rings into a mitre-average and drag every strip off the fillet surface
/// (removed 6.08 where a fillet takes 12.21). It declines real-corner welds now, so the CSG
/// round — straight runs plus a sphere octant per corner — does this shape.
#[test]
fn box_top_rim_corners_blend_smoothly() {
    let (lx, ly, h, r) = (16.0f64, 12.0, 6.0, 1.0);
    let body = extrude_tool_mesh(&[[0.0, 0.0], [lx, 0.0], [lx, ly], [0.0, ly]], &[], &xy(), 0.0, h).unwrap();
    let rim = vec![[0.0, 0.0, h], [lx, 0.0, h], [lx, ly, h], [0.0, ly, h], [0.0, 0.0, h]];
    let picked = vec![rim];
    assert_pick_selectable(&body, &picked, "box rim");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "box rim [{engine}]: not manifold");

    // Exact removal: the edge slivers over the shortened runs, plus a corner cube less the
    // sphere octant at each of the four corners.
    let removed = vol(&body) - vol(&m);
    let per = 2.0 * (lx - 2.0 * r) + 2.0 * (ly - 2.0 * r);
    let corner = r.powi(3) - std::f64::consts::PI * r.powi(3) / 6.0;
    let want = sliver_area(r) * per + 4.0 * corner;
    assert!(
        (removed - want).abs() < want * 0.03,
        "box rim [{engine}]: removed {removed:.4}, the exact answer is {want:.4}"
    );

    // Ideal surface: distance r from the inset rectangle's perimeter at z = h-r. That single
    // capsule formula covers the four cylinder bands AND the four sphere corners.
    let rect_perimeter = |x: f64, y: f64| -> f64 {
        let (cx, cy) = (lx * 0.5, ly * 0.5);
        let (hx, hy) = (lx * 0.5 - r, ly * 0.5 - r);
        let (qx, qy) = ((x - cx).abs() - hx, (y - cy).abs() - hy);
        let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
        let inside = qx.max(qy).min(0.0);
        (outside + inside).abs()
    };
    let ideal = |p: [f64; 3]| ((rect_perimeter(p[0], p[1]).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs();
    let margin = r * 0.05;
    let pred = |p: [f64; 3]| p[2] > h - r + margin && rect_perimeter(p[0], p[1]) > margin;
    let (worst, n) = band_worst(&m, pred, ideal);
    assert!(n > 40, "box rim [{engine}]: only {n} band vertices measured");
    assert!(worst < r * 0.04, "box rim [{engine}]: band strays {worst:.4} from the rolled rim (corners included)");
    // Interior only: the box's own vertical corner edges legitimately stay sharp right up to
    // z = h-r, where the corner spheres take over — the crease measure must not reach below
    // the band and read them. Tangency ACROSS the band's seams is covered by the sharp-display
    // check below (any dihedral past 35 degrees would be drawn as an edge there).
    let (crease, at) = max_band_crease_at(&m, |p| p[2] > h - r + 0.02, 0.02 * r);
    assert!(
        crease < 25.0,
        "box rim [{engine}]: {crease:.1} degree crease on the rolled rim at ({:.3},{:.3},{:.3})",
        at[0], at[1], at[2]
    );
    // Above the tangent height the whole top is smooth: band, corners and cap draw no sharp line.
    assert_no_sharp_edges_in(&m, |p| p[2] > h - r + margin, &format!("box rim [{engine}]"));
}

/// A cylinder's top rim: the round case. The band is a torus; the seams are two full circles.
#[test]
fn cylinder_rim_fillets_to_a_torus() {
    let n = 96usize;
    let (rad, h) = (5.0f64, 8.0);
    let circ: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            [rad * a.cos(), rad * a.sin()]
        })
        .collect();
    let body = extrude_tool_mesh(&circ, &[], &xy(), 0.0, h).unwrap();
    let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], h]).collect();
    rim.push(rim[0]);
    let picked = vec![rim];
    assert_pick_selectable(&body, &picked, "cylinder rim");
    let sagitta = rad * (1.0 - (std::f64::consts::PI / n as f64).cos());
    for &r in &[0.8f64, 2.0] {
        let (m, engine) = app_fillet(&body, r, &picked);
        assert!(is_manifold(&m), "cylinder rim r={r} [{engine}]: not manifold");
        let removed = vol(&body) - vol(&m);
        let want = 2.0 * std::f64::consts::PI * (rad - sliver_ubar(r)) * sliver_area(r);
        assert!(
            (removed - want).abs() < want * 0.02,
            "cylinder rim r={r} [{engine}]: removed {removed:.4}, Pappus says {want:.4}"
        );

        let margin = r * 0.05;
        let pred = |p: [f64; 3]| {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            p[2] > h - r + margin && d > rad - r + margin
        };
        let torus = |p: [f64; 3]| {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            (((d - (rad - r)).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs()
        };
        let (worst, cnt) = band_worst(&m, pred, torus);
        assert!(cnt > n, "cylinder rim r={r} [{engine}]: only {cnt} band vertices measured");
        let tol = (r * 0.02).max(3.0 * sagitta);
        assert!(
            worst < tol,
            "cylinder rim r={r} [{engine}]: band strays {worst:.4} from the ideal torus (tol {tol:.4})"
        );
        let crease = max_band_crease(
            &m,
            |p| {
                let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
                p[2] > h - r - 0.02 && d > rad - r - 0.02
            },
            0.02 * r,
        );
        assert!(crease < 20.0, "cylinder rim r={r} [{engine}]: {crease:.1} degree crease in the torus band");
        assert_no_sharp_edges_in(&m, pred, &format!("cylinder rim r={r} [{engine}]"));

        // Seams: the cap circle (rad-r at z=h) and the wall circle (rad at z=h-r).
        let seam_locus = |p: [f64; 3]| {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let cap = (d - (rad - r)).abs().max((p[2] - h).abs());
            let wall = (d - rad).abs().max((p[2] - (h - r)).abs());
            cap.min(wall)
        };
        assert_seams(&body, r, &picked, seam_locus, 3.0 * sagitta + 1e-3, &format!("cylinder rim r={r}"));
    }
}

/// A tube (pipe) end: the outer rim, the bore rim, and both at once. The bore band is a torus
/// too — its axis circle inside the bore's air at bore+r — and it is the case tubes kept
/// tripping: an inner rim is convex just like the outer one, but every sign in the tool flips.
#[test]
fn tube_end_rims_fillet_to_tori() {
    let n = 96usize;
    let (ro, ri, h, r) = (6.0f64, 3.5f64, 8.0f64, 0.8f64);
    let ring = |rad: f64| -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [rad * a.cos(), rad * a.sin()]
            })
            .collect()
    };
    let outer = ring(ro);
    let bore: Vec<[f64; 2]> = ring(ri).iter().rev().copied().collect();
    let body = extrude_tool_mesh(&outer, &[bore], &xy(), 0.0, h).unwrap();
    let rim_at = |rad: f64| -> Vec<[f64; 3]> {
        let mut c: Vec<[f64; 3]> = ring(rad).iter().map(|p| [p[0], p[1], h]).collect();
        c.push(c[0]);
        c
    };
    let sagitta = ro * (1.0 - (std::f64::consts::PI / n as f64).cos());
    let tol = (r * 0.02).max(3.0 * sagitta);
    let pi = std::f64::consts::PI;
    let outer_want = 2.0 * pi * (ro - sliver_ubar(r)) * sliver_area(r);
    let bore_want = 2.0 * pi * (ri + sliver_ubar(r)) * sliver_area(r);

    let outer_torus = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        (((d - (ro - r)).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs()
    };
    let bore_torus = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        (((d - (ri + r)).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs()
    };
    let margin = r * 0.05;
    let outer_pred = move |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        p[2] > h - r + margin && d > ro - r + margin
    };
    let bore_pred = move |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        p[2] > h - r + margin && d < ri + r - margin
    };

    // --- outer rim alone
    let picked = vec![rim_at(ro)];
    assert_pick_selectable(&body, &picked, "tube outer rim");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "tube outer rim [{engine}]: not manifold");
    let removed = vol(&body) - vol(&m);
    assert!(
        (removed - outer_want).abs() < outer_want * 0.02,
        "tube outer rim [{engine}]: removed {removed:.4}, Pappus says {outer_want:.4}"
    );
    let (worst, cnt) = band_worst(&m, outer_pred, outer_torus);
    assert!(cnt > n && worst < tol, "tube outer rim [{engine}]: band strays {worst:.4} over {cnt} verts (tol {tol:.4})");
    assert_no_sharp_edges_in(&m, outer_pred, &format!("tube outer rim [{engine}]"));

    // --- bore rim alone
    let picked = vec![rim_at(ri)];
    assert_pick_selectable(&body, &picked, "tube bore rim");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "tube bore rim [{engine}]: not manifold");
    let moved = vol(&body) - vol(&m);
    assert!(
        (moved - bore_want).abs() < bore_want * 0.02,
        "tube bore rim [{engine}]: removed {moved:.4}, Pappus says {bore_want:.4}"
    );
    let (worst, cnt) = band_worst(&m, bore_pred, bore_torus);
    assert!(cnt > n && worst < tol, "tube bore rim [{engine}]: band strays {worst:.4} over {cnt} verts (tol {tol:.4})");
    let crease = max_band_crease(
        &m,
        |p| {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            p[2] > h - r - 0.02 && d < ri + r + 0.02
        },
        0.02 * r,
    );
    assert!(crease < 20.0, "tube bore rim [{engine}]: {crease:.1} degree crease in the bore band");
    assert_no_sharp_edges_in(&m, bore_pred, &format!("tube bore rim [{engine}]"));

    // --- both rims in one pick
    let picked = vec![rim_at(ro), rim_at(ri)];
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "tube both rims [{engine}]: not manifold");
    let moved = vol(&body) - vol(&m);
    let want = outer_want + bore_want;
    assert!(
        (moved - want).abs() < want * 0.02,
        "tube both rims [{engine}]: removed {moved:.4}, the two tori take {want:.4}"
    );
    let (wo, co) = band_worst(&m, outer_pred, outer_torus);
    let (wi, ci) = band_worst(&m, bore_pred, bore_torus);
    assert!(co > n && wo < tol, "tube both rims [{engine}]: outer band strays {wo:.4}");
    assert!(ci > n && wi < tol, "tube both rims [{engine}]: bore band strays {wi:.4}");
    // The whole end face region above the tangent height is smooth now.
    assert_no_sharp_edges_in(&m, |p| p[2] > h - r + margin, &format!("tube both rims [{engine}]"));
}

/// A boss standing on a plate, filleted where it meets it: the CONCAVE round case. The fillet
/// adds a quarter-round fill whose band is a torus outside the boss wall.
#[test]
fn boss_base_concave_junction_fills_smoothly() {
    let n = 96usize;
    let (rad, plate_h, boss_top, r) = (5.0f64, 3.0, 9.0, 1.0);
    let plate =
        extrude_tool_mesh(&[[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]], &[], &xy(), 0.0, plate_h)
            .unwrap();
    let circ: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            [rad * a.cos(), rad * a.sin()]
        })
        .collect();
    let boss = extrude_tool_mesh(&circ, &[], &xy(), plate_h, boss_top - plate_h).unwrap();
    let body = mesh_union(&plate, &boss);
    let mut rim: Vec<[f64; 3]> = circ.iter().map(|p| [p[0], p[1], plate_h]).collect();
    rim.push(rim[0]);
    let picked = vec![rim];
    assert_pick_selectable(&body, &picked, "boss base");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "boss base [{engine}]: not manifold");
    let added = vol(&m) - vol(&body);
    let want = 2.0 * std::f64::consts::PI * (rad + sliver_ubar(r)) * sliver_area(r);
    assert!((added - want).abs() < want * 0.02, "boss base [{engine}]: added {added:+.4}, Pappus says {want:.4}");

    let sagitta = rad * (1.0 - (std::f64::consts::PI / n as f64).cos());
    let tol = (r * 0.02).max(3.0 * sagitta);
    let margin = r * 0.05;
    let pred = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        d > rad + margin && d < rad + r - margin && p[2] > plate_h + margin && p[2] < plate_h + r - margin
    };
    let torus = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        (((d - (rad + r)).powi(2) + (p[2] - (plate_h + r)).powi(2)).sqrt() - r).abs()
    };
    let (worst, cnt) = band_worst(&m, pred, torus);
    assert!(cnt > n / 2, "boss base [{engine}]: only {cnt} band vertices measured");
    assert!(worst < tol, "boss base [{engine}]: fill strays {worst:.4} from the concave torus (tol {tol:.4})");
    let crease = max_band_crease(
        &m,
        |p| {
            let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
            d > rad - 0.02 && d < rad + r + 0.02 && p[2] > plate_h - 0.02 && p[2] < plate_h + r + 0.02
        },
        0.02 * r,
    );
    assert!(crease < 20.0, "boss base [{engine}]: {crease:.1} degree crease in the fill band");
    assert_no_sharp_edges_in(&m, pred, &format!("boss base [{engine}]"));
}

/// Part of a rim — the pick that used to gouge. Through the APP's path this time, not
/// round_mesh directly: whichever engine takes it must produce the same smooth partial torus.
#[test]
fn partial_arc_pick_fillets_its_span_smoothly() {
    let n = 128usize;
    let (rad, h, r, k) = (8.0f64, 2.0, 0.5, 12usize);
    let circ: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            [rad * a.cos(), rad * a.sin()]
        })
        .collect();
    let body = extrude_tool_mesh(&circ, &[], &xy(), 0.0, h).unwrap();
    let pt = |i: usize| -> [f64; 3] {
        let a = std::f64::consts::TAU * (i % n) as f64 / n as f64;
        [rad * a.cos(), rad * a.sin(), h]
    };
    let chain: Vec<[f64; 3]> = (n / 4..=n / 4 + k).map(pt).collect();
    let picked = vec![chain.clone()];
    assert_pick_selectable(&body, &picked, "partial arc");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "partial arc [{engine}]: not manifold");
    let removed = vol(&body) - vol(&m);
    let arc: f64 = chain.windows(2).map(|w| len(sub(w[1], w[0]))).sum();
    let want = sliver_area(r) * arc;
    assert!(
        removed > want * 0.95 && removed < want * 1.15,
        "partial arc [{engine}]: removed {removed:.4}, the picked span's fillet takes {want:.4}"
    );

    // The band, inside the picked angular span with one facet of margin at each end.
    let (a0, a1) = (
        std::f64::consts::TAU * (n as f64 / 4.0 + 1.0) / n as f64,
        std::f64::consts::TAU * (n as f64 / 4.0 + k as f64 - 1.0) / n as f64,
    );
    let margin = r * 0.05;
    let pred = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let th = p[1].atan2(p[0]).rem_euclid(std::f64::consts::TAU);
        p[2] > h - r + margin && d > rad - r + margin && th > a0 && th < a1
    };
    let torus = |p: [f64; 3]| {
        let d = (p[0] * p[0] + p[1] * p[1]).sqrt();
        (((d - (rad - r)).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs()
    };
    let (worst, cnt) = band_worst(&m, pred, torus);
    let sagitta = rad * (1.0 - (std::f64::consts::PI / n as f64).cos());
    let tol = (r * 0.02).max(3.0 * sagitta);
    assert!(cnt > k, "partial arc [{engine}]: only {cnt} band vertices measured");
    assert!(worst < tol, "partial arc [{engine}]: band strays {worst:.4} from the partial torus (tol {tol:.4})");
    let crease = max_band_crease(&m, pred, 0.02 * r);
    assert!(crease < 20.0, "partial arc [{engine}]: {crease:.1} degree crease inside the picked span");
    assert_no_sharp_edges_in(&m, pred, &format!("partial arc [{engine}]"));
    // The seams for a partial pick must still be emitted for selection.
    let seams = bevel_feature_edges(&body, r, &picked);
    assert!(!seams.is_empty(), "partial arc: no selectable seam edges emitted");
}

/// A stadium (slot-shaped) boss top rim: straight runs blending into arcs with no corner — the
/// mixed curve that must sweep as ONE smooth tube. The ideal surface is distance r from the
/// rim's inset spine, one capsule formula for straights and arcs alike.
#[test]
fn stadium_slot_rim_fillets_as_one_tube() {
    let (half_l, a, h, r) = (4.0f64, 3.0f64, 5.0f64, 0.8f64);
    let n_arc = 48usize;
    let mut prof: Vec<[f64; 2]> = Vec::new();
    for i in 0..=n_arc {
        let t = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * i as f64 / n_arc as f64;
        prof.push([half_l + a * t.cos(), a * t.sin()]);
    }
    for i in 0..=n_arc {
        let t = std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * i as f64 / n_arc as f64;
        prof.push([-half_l + a * t.cos(), a * t.sin()]);
    }
    prof.dedup_by(|p, q| (p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9);
    if let (Some(first), Some(last)) = (prof.first().copied(), prof.last().copied()) {
        if (first[0] - last[0]).abs() < 1e-9 && (first[1] - last[1]).abs() < 1e-9 {
            prof.pop();
        }
    }
    let body = extrude_tool_mesh(&prof, &[], &xy(), 0.0, h).unwrap();
    let mut rim: Vec<[f64; 3]> = prof.iter().map(|p| [p[0], p[1], h]).collect();
    rim.push(rim[0]);
    let picked = vec![rim];
    assert_pick_selectable(&body, &picked, "stadium rim");
    let (m, engine) = app_fillet(&body, r, &picked);
    assert!(is_manifold(&m), "stadium rim [{engine}]: not manifold");

    // Pappus: the straights sweep in a line, the two half-circle arcs together in a circle.
    let removed = vol(&body) - vol(&m);
    let want = sliver_area(r) * (4.0 * half_l)
        + 2.0 * std::f64::consts::PI * (a - sliver_ubar(r)) * sliver_area(r);
    assert!(
        (removed - want).abs() < want * 0.02,
        "stadium rim [{engine}]: removed {removed:.4}, Pappus says {want:.4}"
    );

    // Distance in-plane to the stadium's spine segment, minus the inset — the capsule trick.
    let spine = |x: f64, y: f64| -> f64 {
        let dx = (x.abs() - half_l).max(0.0);
        ((dx * dx + y * y).sqrt() - (a - r)).abs()
    };
    let ideal = |p: [f64; 3]| ((spine(p[0], p[1]).powi(2) + (p[2] - (h - r)).powi(2)).sqrt() - r).abs();
    let margin = r * 0.05;
    let pred = |p: [f64; 3]| p[2] > h - r + margin && spine(p[0], p[1]) > margin;
    let sagitta = a * (1.0 - (std::f64::consts::PI / (2 * n_arc) as f64).cos());
    let tol = (r * 0.02).max(3.0 * sagitta);
    let (worst, cnt) = band_worst(&m, pred, ideal);
    assert!(cnt > n_arc, "stadium rim [{engine}]: only {cnt} band vertices measured");
    assert!(worst < tol, "stadium rim [{engine}]: band strays {worst:.4} from the swept tube (tol {tol:.4})");
    let crease = max_band_crease(&m, |p| p[2] > h - r - 0.02, 0.02 * r);
    assert!(crease < 20.0, "stadium rim [{engine}]: {crease:.1} degree crease along the tube");
    assert_no_sharp_edges_in(&m, pred, &format!("stadium rim [{engine}]"));
}

/// The area of material thinner than `max_thick` in the mesh, plus the area of exactly
/// coincident fold-back sheets (zero thickness — invisible to a ray with a self-hit guard).
/// This is what "a very thin sliver coming from the fillet" is made of.
fn thin_area(m: &TriMesh, max_thick: f64) -> f64 {
    let g = |i: u32| {
        let q = m.positions[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    let tris: Vec<[[f64; 3]; 3]> = m.indices.chunks_exact(3).map(|t| [g(t[0]), g(t[1]), g(t[2])]).collect();
    let geom = |t: &[[f64; 3]; 3]| {
        let n = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let nl = len(n);
        (nl, [n[0] / nl.max(1e-30), n[1] / nl.max(1e-30), n[2] / nl.max(1e-30)])
    };
    let mut area = 0.0f64;
    for (ti, t) in tris.iter().enumerate() {
        let (nl, nrm) = geom(t);
        let longest = len(sub(t[1], t[0])).max(len(sub(t[2], t[1]))).max(len(sub(t[0], t[2])));
        if nl < 1e-12 || longest < 1e-9 || nl / longest < 1e-3 {
            continue;
        }
        let cen = [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
            (t[0][2] + t[1][2] + t[2][2]) / 3.0,
        ];
        let dir = [-nrm[0], -nrm[1], -nrm[2]];
        let mut thin = false;
        for (ui, u) in tris.iter().enumerate() {
            if ui == ti {
                continue;
            }
            let (unl, un) = geom(u);
            if unl < 1e-12 {
                continue;
            }
            // A real-but-hair-thin wall: a ray along -normal meets an opposing face close by.
            // (Exactly coincident zero-thickness folds are invisible to this ray — its self-hit
            // guard rejects the opposite face at distance ~0 — so the caller pairs this with a
            // fold check via max_band_crease.)
            let e1 = sub(u[1], u[0]);
            let e2 = sub(u[2], u[0]);
            let pv = cross(dir, e2);
            let det = dot(e1, pv);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let tv = sub(cen, u[0]);
            let uu = dot(tv, pv) * inv;
            if !(-1e-9..=1.0 + 1e-9).contains(&uu) {
                continue;
            }
            let qv = cross(tv, e1);
            let vv = dot(dir, qv) * inv;
            if vv < -1e-9 || uu + vv > 1.0 + 1e-9 {
                continue;
            }
            let tt = dot(e2, qv) * inv;
            if tt > 1e-6 && tt < max_thick && dot(un, dir) > 0.7 {
                thin = true;
                break;
            }
        }
        if thin {
            area += nl * 0.5;
        }
    }
    area
}

/// sliver.hcad's disease, synthetically: a tube loses a quadrant to a cut; a CONCAVE fillet
/// fills the corner where the cut floor meets a radial wall, leaning its tangent wedge on
/// that wall; then a CONVEX fillet rounds the same wall's top rim — cutting away the wall
/// the wedge leans on. Three things went wrong here once: the fill's flush union left
/// 180-degree fins, the convex tool's flush difference left a 1.5e-4 film of wall over its
/// whole band, and the fill's wedge stood as a cantilevered blade — 7.97 of thin area in
/// all, "the thin walls on the fillets". The fix set: embedded fill flanks, the boolean
/// judge measuring film area across both signs of every nudge, and the blade-gated slab
/// that truncates the wedge at the band's foot.
#[test]
fn a_fillet_on_a_wall_a_fill_leans_on_leaves_no_blade() {
    let n = 96usize;
    let (ro, ri, h) = (6.4f64, 5.6f64, 7.0f64);
    let ring = |rad: f64| -> Vec<[f64; 2]> {
        (0..n)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / n as f64;
                [rad * a.cos(), rad * a.sin()]
            })
            .collect()
    };
    let bore: Vec<[f64; 2]> = ring(ri).iter().rev().copied().collect();
    let tube = extrude_tool_mesh(&ring(ro), &[bore], &xy(), 0.0, h).unwrap();
    // Cut the x>0, y<0 quadrant away from z=4 up.
    let tool = extrude_tool_mesh(&[[0.0, 0.0], [10.0, 0.0], [10.0, -10.0], [0.0, -10.0]], &[], &xy(), 4.0, 4.0).unwrap();
    let body = hworks_geometry::mesh_difference(&tube, &tool);
    let vol = |m: &TriMesh| {
        let mut v = 0.0f64;
        for t in m.indices.chunks_exact(3) {
            let g = |i: u32| {
                let q = m.positions[i as usize];
                [q[0] as f64, q[1] as f64, q[2] as f64]
            };
            v += dot(g(t[0]), cross(g(t[1]), g(t[2]))) / 6.0;
        }
        v.abs()
    };
    // Concave fillet on the floor edge under the y=0 radial wall.
    let (r_fill, r_rim) = (2.4f64, 2.0f64);
    let fill_edge = vec![vec![[ri, 0.0, 4.0], [ro, 0.0, 4.0]]];
    let filled = round_mesh(&body, r_fill, &fill_edge).expect("the concave fill applies");
    let added = vol(&filled) - vol(&body);
    let want_fill = (1.0 - std::f64::consts::PI / 4.0) * r_fill * r_fill * (ro - ri);
    assert!(
        (added - want_fill).abs() < want_fill * 0.1,
        "the fill added {added:.4}, a fillet adds {want_fill:.4}"
    );
    // Convex fillet on the same wall's top rim — the cut that used to expose the blade.
    let rim_edge = vec![vec![[ri, 0.0, h], [ro, 0.0, h]]];
    let (m, engine) = app_fillet(&filled, r_rim, &rim_edge);
    assert!(is_manifold(&m), "[{engine}] result not manifold");
    let removed = vol(&filled) - vol(&m);
    assert!(
        removed > 0.5 * (1.0 - std::f64::consts::PI / 4.0) * r_rim * r_rim * (ro - ri),
        "[{engine}] the rim fillet barely cut ({removed:.4})"
    );
    // The point: no blade, no film, no folds. Seam debris measures ~0.01; the blade alone
    // was 3, the flush films 5 more, and the doubled-membrane fins read as 180-degree folds.
    let thin = thin_area(&m, 0.05);
    assert!(
        thin < 0.1,
        "[{engine}] {thin:.4} of hair-thin material stands on the body — the blade or a film is back"
    );
    let crease = max_band_crease(&m, |_| true, 0.02);
    assert!(
        crease < 150.0,
        "[{engine}] a {crease:.0}-degree fold stands on the body — a zero-thickness membrane is back"
    );
}
