//! Edge rounding (fillet) for triangle meshes.
//!
//! Two strategies:
//!
//!  * **Boolean rolling-ball** ([`fillet_boolean`]) for *picked* edges between flat faces:
//!    a cylinder of radius `r` tangent to both faces replaces the sharp corner. We carve
//!    `(corner-triangle prism − cylinder)` out of the body with Manifold, so the cylinder
//!    arc becomes the new surface — an exact, crisp CAD-style fillet.
//!
//!  * **SDF blur** ([`round_mesh`] fallback) when no edges are given (a global round) or
//!    the boolean path can't handle an edge: sample the signed-distance field on a voxel
//!    grid, Gaussian-blur it by the radius (flat faces stay put, edges round to ≈ r), and
//!    re-extract the surface with surface nets.

use crate::TriMesh;
use std::collections::HashMap;

type V3 = [f64; 3];

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn norm(a: V3) -> V3 {
    let l = len(a);
    if l > 1e-12 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Squared distance from point `p` to triangle `abc`.
fn point_tri_dist2(p: V3, a: V3, b: V3, c: V3) -> f64 {
    // Ericson, "Real-Time Collision Detection" — closest point on triangle.
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
        let d = sub(p, q);
        return dot(d, d);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
        let d = sub(p, q);
        return dot(d, d);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = [b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2])];
        let d = sub(p, q);
        return dot(d, d);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ];
    let d = sub(p, q);
    dot(d, d)
}

/// Solid angle subtended by triangle `abc` at point `p` (Van Oosterom–Strackee). Summed
/// over a watertight mesh and divided by 4π this is the winding number — robust inside/
/// outside classification independent of per-triangle normals.
fn solid_angle(p: V3, a: V3, b: V3, c: V3) -> f64 {
    let va = sub(a, p);
    let vb = sub(b, p);
    let vc = sub(c, p);
    let (la, lb, lc) = (len(va), len(vb), len(vc));
    if la == 0.0 || lb == 0.0 || lc == 0.0 {
        return 0.0;
    }
    let num = dot(va, cross(vb, vc));
    let den = la * lb * lc + dot(va, vb) * lc + dot(vb, vc) * la + dot(vc, va) * lb;
    2.0 * num.atan2(den)
}

/// Squared distance from point `p` to a polyline segment `a`–`b`.
fn point_seg_dist2(p: V3, a: V3, b: V3) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 1e-12 { (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let q = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    let d = sub(p, q);
    dot(d, d)
}

/// Triangle face normal (assumes CCW outward winding).
fn tri_normal(a: V3, b: V3, c: V3) -> V3 {
    norm(cross(sub(b, a), sub(c, a)))
}

/// The two distinct adjacent face normals of the mesh edge `a`–`b` (the faces meeting at
/// that edge). `None` if the edge isn't shared by two clearly-different faces.
fn adjacent_normals(tris: &[[V3; 3]], a: V3, b: V3, tol: f64) -> Option<(V3, V3)> {
    let near = |p: V3, q: V3| sub(p, q).iter().fold(0.0_f64, |m, &c| m.max(c.abs())) < tol;
    let has = |t: &[V3; 3], p: V3| t.iter().any(|&v| near(v, p));
    let mut normals: Vec<V3> = Vec::new();
    for t in tris {
        if has(t, a) && has(t, b) {
            let n = tri_normal(t[0], t[1], t[2]);
            if normals.iter().all(|m| dot(*m, n) < 0.999) {
                normals.push(n);
            }
        }
    }
    if normals.len() >= 2 {
        // The two most divergent normals (a feature edge can touch >2 coplanar tris).
        let mut best = (0, 1, 2.0);
        for i in 0..normals.len() {
            for j in i + 1..normals.len() {
                let d = dot(normals[i], normals[j]);
                if d < best.2 {
                    best = (i, j, d);
                }
            }
        }
        Some((normals[best.0], normals[best.1]))
    } else {
        None
    }
}

/// Extrude a planar polygon `cross` (in order) along `axis` (unit) by `length`, starting
/// at the given `cross` positions — a closed prism mesh (two caps + side walls). The cross
/// is auto-wound CCW about `axis` so the prism is always outward-oriented (a solid, not an
/// inside-out shell), regardless of the caller's vertex order — otherwise a Manifold
/// difference would carve the *complement*.
fn extrude_prism(cross_in: &[V3], axis: V3, length: f64) -> TriMesh {
    // Newell normal of the cross polygon; flip the order if it faces against the axis.
    let mut newell = [0.0; 3];
    for i in 0..cross_in.len() {
        let p = cross_in[i];
        let q = cross_in[(i + 1) % cross_in.len()];
        newell[0] += (p[1] - q[1]) * (p[2] + q[2]);
        newell[1] += (p[2] - q[2]) * (p[0] + q[0]);
        newell[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    let cross: Vec<V3> = if dot(newell, axis) < 0.0 {
        cross_in.iter().rev().copied().collect()
    } else {
        cross_in.to_vec()
    };
    let n = cross.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * 2);
    let f = |p: V3| [p[0] as f32, p[1] as f32, p[2] as f32];
    for &p in &cross {
        positions.push(f(p));
    }
    for &p in &cross {
        positions.push(f(add(p, scale(axis, length))));
    }
    let mut indices: Vec<u32> = Vec::new();
    // Bottom cap (fan), reversed; top cap (fan).
    for i in 1..n - 1 {
        indices.extend_from_slice(&[0, (i + 1) as u32, i as u32]);
        indices.extend_from_slice(&[n as u32, (n + i) as u32, (n + i + 1) as u32]);
    }
    // Side walls.
    for i in 0..n {
        let j = (i + 1) % n;
        let (a0, b0, a1, b1) = (i as u32, j as u32, (n + i) as u32, (n + j) as u32);
        indices.extend_from_slice(&[a0, b0, b1, a0, b1, a1]);
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    fill_normals(&mut m);
    m
}

/// A closed cylinder of `radius` about `center` along `axis` (unit), `length` long, with
/// `seg` facets. `u`,`w` span the plane perpendicular to `axis`.
fn make_cylinder(center: V3, axis: V3, u: V3, w: V3, radius: f64, length: f64, seg: usize) -> TriMesh {
    let mut cross: Vec<V3> = Vec::with_capacity(seg);
    for k in 0..seg {
        let t = std::f64::consts::TAU * k as f64 / seg as f64;
        cross.push(add(center, add(scale(u, radius * t.cos()), scale(w, radius * t.sin()))));
    }
    extrude_prism(&cross, axis, length)
}

fn fill_normals(m: &mut TriMesh) {
    let mut normals = vec![[0f32; 3]; m.positions.len()];
    for t in m.indices.chunks_exact(3) {
        let p = |i: u32| m.positions[i as usize];
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let nrm = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
        for &vi in t {
            for k in 0..3 {
                normals[vi as usize][k] += nrm[k];
            }
        }
    }
    for nv in &mut normals {
        let l = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        if l > 1e-9 {
            nv[0] /= l;
            nv[1] /= l;
            nv[2] /= l;
        }
    }
    m.normals = normals;
}

/// Flip a mesh's triangle winding if its signed volume is negative, so the boolean tool is
/// a properly outward-oriented solid (a revolution's winding is otherwise hard to predict).
fn orient_outward(m: &mut TriMesh) {
    let mut vol = 0.0;
    for t in m.indices.chunks_exact(3) {
        let p = |i: u32| {
            let v = m.positions[i as usize];
            [v[0] as f64, v[1] as f64, v[2] as f64]
        };
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        vol += dot(a, cross(b, c));
    }
    if vol < 0.0 {
        for t in m.indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }
}

/// Revolve a closed 2D `profile` (points in `(s, z)` = radius-from-axis, distance-along-
/// axis) around the axis through `center` (direction `axis`, in-plane reference `radial0`),
/// placing one ring at each angle in `angles`. Passing the rim's own vertex angles makes
/// the tool's rings coincide with the body's facets, so the union/cut has no step or beat.
fn revolve(profile: &[(f64, f64)], center: V3, axis: V3, radial0: V3, angles: &[f64]) -> TriMesh {
    let tangent = cross(axis, radial0);
    let seg = angles.len();
    let np = profile.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(seg * np);
    for &th in angles {
        let rad = add(scale(radial0, th.cos()), scale(tangent, th.sin()));
        for &(s, z) in profile {
            let p = add(center, add(scale(rad, s), scale(axis, z)));
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    for k in 0..seg {
        let k1 = (k + 1) % seg;
        for i in 0..np {
            let i1 = (i + 1) % np;
            let (a, b, c, d) =
                ((k * np + i) as u32, (k1 * np + i) as u32, (k1 * np + i1) as u32, (k * np + i1) as u32);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// Fit a circle to a loop of points: `(centre, axis, radius)`. `None` if the points aren't
/// (near-)planar and equidistant from their centroid — i.e. not a circular edge.
fn fit_circle(pts: &[[f64; 3]]) -> Option<(V3, V3, f64)> {
    if pts.len() < 6 {
        return None;
    }
    let n = pts.len() as f64;
    let center = pts.iter().fold([0.0; 3], |a, p| add(a, *p));
    let center = scale(center, 1.0 / n);
    // Newell normal (best-fit plane).
    let mut nrm = [0.0; 3];
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        nrm[0] += (p[1] - q[1]) * (p[2] + q[2]);
        nrm[1] += (p[2] - q[2]) * (p[0] + q[0]);
        nrm[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    if len(nrm) < 1e-9 {
        return None;
    }
    let axis = norm(nrm);
    let radius = pts.iter().map(|p| len(sub(*p, center))).sum::<f64>() / n;
    if radius < 1e-6 {
        return None;
    }
    // Validate: planar (small axial deviation) and constant radius.
    for p in pts {
        let d = sub(*p, center);
        if (dot(d, axis)).abs() > radius * 0.05 {
            return None; // not planar enough
        }
        let r = len(sub(d, scale(axis, dot(d, axis))));
        if (r - radius).abs() > radius * 0.08 {
            return None; // not a constant radius
        }
    }
    Some((center, axis, radius))
}

/// Torus rolling-ball fillet on a circular edge (e.g. a cylinder's rim). Returns the
/// revolved tool plus whether it should be *unioned* (a concave corner — the fillet adds a
/// fill) or *subtracted* (a convex rim — it shaves the corner). `None` if the edge isn't a
/// circular flat-cap + round-wall corner, or the radius is too steep.
fn fillet_circular(tris: &[[V3; 3]], radius: f64, loop_pts: &[[f64; 3]], _tol: f64) -> Option<(TriMesh, bool)> {
    let (center, axis0, big_r) = fit_circle(loop_pts)?;
    if big_r <= radius * 1.05 {
        return None; // radius too big for this rim
    }
    // The cap normal is the loop's axis and the wall normal is radial — derive both (and
    // their orientation) from a winding-based corner probe, NOT from matching mesh vertices
    // (the committed mesh-kernel rebuild tessellates the rim differently than the preview,
    // so vertex matching is unreliable; winding is tessellation-independent).
    let radial0 = {
        let r = sub(loop_pts[0], center);
        norm(sub(r, scale(axis0, dot(r, axis0)))) // make it exactly ⟂ to the axis
    };
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    let rim0 = loop_pts[0];
    let e = radius * 0.4;
    // Probe the four (axis, radial) quadrants; `sc`/`sw` ∈ {−1,+1}.
    let signs = [-1.0_f64, 1.0];
    let mut q = [[false; 2]; 2];
    let mut solid = 0;
    for (i, &sc) in signs.iter().enumerate() {
        for (j, &sw) in signs.iter().enumerate() {
            let p = add(rim0, add(scale(axis0, sc * e), scale(radial0, sw * e)));
            q[i][j] = inside(p);
            if q[i][j] {
                solid += 1;
            }
        }
    }
    // Orient the axis so +axis is the cap's outward normal, and find which radial side the
    // wall material is on (`sign_w`: +1 boss, −1 hole).
    let (axis, sign_w, is_concave) = if solid == 1 {
        // Convex: the lone solid quadrant (sc0, sw0) holds the material; the cap/wall
        // outward normals point away from it.
        let (mut sc0, mut sw0) = (1.0, 1.0);
        for (i, &sc) in signs.iter().enumerate() {
            for (j, &sw) in signs.iter().enumerate() {
                if q[i][j] {
                    sc0 = sc;
                    sw0 = sw;
                }
            }
        }
        (scale(axis0, -sc0), -sw0, false)
    } else if solid == 3 {
        // Concave: the lone air quadrant (sc_a, sw_a) is the notch the fillet fills.
        let (mut sca, mut swa) = (1.0, 1.0);
        for (i, &sc) in signs.iter().enumerate() {
            for (j, &sw) in signs.iter().enumerate() {
                if !q[i][j] {
                    sca = sc;
                    swa = sw;
                }
            }
        }
        (scale(axis0, sca), swa, true)
    } else {
        return None; // ambiguous (e.g. a flat coplanar ring)
    };

    // Revolve at the angles of the *actual mesh's* rim vertices (the ring at radius `big_r`
    // in the cap plane) so the torus rings line up with the body's wall facets — no beat or
    // step. We read these from the live mesh, not the stored edge polyline, because the
    // committed mesh-kernel rebuild tessellates the rim differently than the preview did.
    let tangent = cross(axis, radial0);
    let mut ring: Vec<(f64, [i64; 3])> = Vec::new();
    let plane_tol = big_r * 0.03;
    for t in tris {
        for &v in t {
            let d = sub(v, center);
            let axial = dot(d, axis);
            let radial = len(sub(d, scale(axis, axial)));
            if axial.abs() < plane_tol && (radial - big_r).abs() < plane_tol {
                let ang = dot(d, tangent).atan2(dot(d, radial0));
                let key = [(v[0] * 1e3) as i64, (v[1] * 1e3) as i64, (v[2] * 1e3) as i64];
                ring.push((ang, key));
            }
        }
    }
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ring.dedup_by_key(|r| r.1);
    let angles: Vec<f64> = if ring.len() >= 8 {
        ring.iter().map(|r| r.0).collect()
    } else {
        // Couldn't read a clean rim from the mesh — fall back to the stored edge's angles.
        loop_pts
            .iter()
            .map(|p| {
                let d = sub(*p, center);
                dot(d, tangent).atan2(dot(d, radial0))
            })
            .collect()
    };
    let r = radius;
    let pad = r;
    const ARC: usize = 20;
    // Map a normal-aligned cross-section point (u = offset along the wall normal, v =
    // offset along the cap normal/axis) to the lathe's (s = radius, z = axial). `sign_w`
    // flips the radial direction for a hole so the same profiles serve boss and hole.
    let sz = |u: f64, v: f64| -> (f64, f64) { (big_r + sign_w * u, v) };

    if !is_concave {
        // Convex rim (cylinder top / hole lip): subtract the corner sliver. The tool's
        // outer edges run into open air, so the cut is transversal (no curved coincidence).
        // Material is the (−wall, −cap) corner; the rolling circle centre is (−r, −r).
        let mut profile: Vec<(f64, f64)> = vec![sz(-r, 0.0)]; // cap contact
        for k in 1..ARC {
            let a = std::f64::consts::FRAC_PI_2 * (1.0 - k as f64 / ARC as f64);
            profile.push(sz(-r + r * a.cos(), -r + r * a.sin()));
        }
        profile.push(sz(0.0, -r)); // wall contact
        profile.push(sz(pad, -r)); // out into air past the wall
        profile.push(sz(pad, pad)); // air
        profile.push(sz(-r, pad)); // back over the cap (air)
        let tool = revolve(&profile, center, axis, radial0, &angles);
        Some((tool, false))
    } else {
        // Concave rim (a boss base / counterbore floor): union a rounded fill. The notch is
        // the (+wall, +cap) quadrant; the rolling circle centre is (+r, +r). The tool's
        // non-arc edges run into existing material, so only the arc is a new surface.
        let mut profile: Vec<(f64, f64)> = vec![sz(0.0, r)]; // wall contact
        for k in 1..ARC {
            let a = std::f64::consts::PI + std::f64::consts::FRAC_PI_2 * (k as f64 / ARC as f64);
            profile.push(sz(r + r * a.cos(), r + r * a.sin()));
        }
        profile.push(sz(r, 0.0)); // cap contact
        profile.push(sz(r, -pad)); // down into material
        profile.push(sz(-pad, -pad)); // deep material
        profile.push(sz(-pad, r)); // up into the wall's material, back to wall contact
        let tool = revolve(&profile, center, axis, radial0, &angles);
        Some((tool, true))
    }
}

/// Boolean rolling-ball fillet on the picked `edges` (world-space polylines). For each
/// straight, convex segment between two flat faces it carves the corner with a tangent
/// cylinder. Returns the rounded mesh, or `None` if it couldn't round any segment (caller
/// falls back to the SDF round).
fn fillet_boolean(mesh: &TriMesh, radius: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    let tris: Vec<[V3; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let g = |i: u32| {
                let p = mesh.positions[i as usize];
                [p[0] as f64, p[1] as f64, p[2] as f64]
            };
            [g(t[0]), g(t[1]), g(t[2])]
        })
        .collect();
    if tris.is_empty() {
        return None;
    }
    // Match tolerance scaled to the model size.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for t in &tris {
        for v in t {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let diag = len(sub(hi, lo)).max(1.0);
    let tol = diag * 1e-4;
    // Clamp the radius to under half the smallest body dimension — a steeper fillet would
    // build a degenerate/self-overlapping tool that can abort the mesh kernel.
    let min_ext = (hi[0] - lo[0]).min(hi[1] - lo[1]).min(hi[2] - lo[2]).max(1e-6);
    let radius = radius.min(0.49 * min_ext);

    let mut body = mesh.clone();
    let mut any = false;
    // Convex straight-edge endpoints → the distinct face normals meeting there, so a corner
    // shared by ≥3 faces (a box vertex) can be blended with a sphere afterwards.
    let mut corners: HashMap<[i64; 3], (V3, Vec<V3>)> = HashMap::new();
    let vkey = |p: V3| [(p[0] * 1e3).round() as i64, (p[1] * 1e3).round() as i64, (p[2] * 1e3).round() as i64];
    for chain in edges {
        // A circular rim (e.g. a cylinder cap edge) is filleted *only* by the toroidal
        // cut — never by the straight per-segment path (which, run over a whole circle of
        // segments with a steep radius, builds dozens of huge overlapping tools and can
        // abort Manifold). If the radius is too steep for the torus, the edge is skipped.
        if fit_circle(chain).is_some() {
            if let Some((tool, is_union)) = fillet_circular(&tris, radius, chain, tol) {
                if tool.indices.len() >= 3 {
                    body = if is_union {
                        crate::mesh_union(&body, &tool)
                    } else {
                        crate::mesh_difference(&body, &tool)
                    };
                    any = true;
                }
            }
            continue;
        }
        for seg in chain.windows(2) {
            let (a, b) = (seg[0], seg[1]);
            let d = sub(b, a);
            let l = len(d);
            if l < tol {
                continue;
            }
            let axis = norm(d);
            let Some((n1, n2)) = adjacent_normals(&tris, a, b, tol.max(1e-3)) else {
                continue;
            };
            let c = dot(n1, n2);
            if (1.0 + c).abs() < 1e-3 {
                continue; // faces nearly opposite → no corner to round
            }
            // Convex vs concave: probe the four (n1, n2) quadrants at the edge midpoint and
            // count solid ones (1 = convex ridge, 3 = concave notch).
            let inside = |p: V3| {
                let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
                (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
            };
            let m = scale(add(a, b), 0.5);
            let e = radius * 0.4;
            let mut solid = 0;
            for s1 in [-1.0_f64, 1.0] {
                for s2 in [-1.0_f64, 1.0] {
                    if inside(add(m, add(scale(n1, s1 * e), scale(n2, s2 * e)))) {
                        solid += 1;
                    }
                }
            }
            let margin = radius * 1.5;
            let start_shift = scale(axis, -margin);
            let total_len = l + 2.0 * margin;
            let u = |c0: V3| norm(sub(add(c0, scale(n1, radius)), c0));

            if solid == 1 {
                // Convex ridge: subtract the corner sliver (corner-prism − tangent cylinder).
                let c0 = add(a, scale(add(n1, n2), -radius / (1.0 + c))); // centre, in material
                let t1 = add(c0, scale(n1, radius));
                let t2 = add(c0, scale(n2, radius));
                let cross = [add(a, start_shift), add(t1, start_shift), add(t2, start_shift)];
                let prism = extrude_prism(&cross, axis, total_len);
                let uu = u(c0);
                let w = norm(cross_perp(axis, uu));
                let cyl = make_cylinder(add(c0, start_shift), axis, uu, w, radius, total_len, 48);
                let tool = crate::mesh_difference(&prism, &cyl);
                if tool.indices.len() >= 3 {
                    body = crate::mesh_difference(&body, &tool);
                    any = true;
                }
                // Record this convex edge's faces at both endpoints, for corner blending.
                for &v in &[a, b] {
                    let entry = corners.entry(vkey(v)).or_insert_with(|| (v, Vec::new()));
                    for &nrm in &[n1, n2] {
                        if entry.1.iter().all(|m| dot(*m, nrm) < 0.99) {
                            entry.1.push(nrm);
                        }
                    }
                }
            } else if solid == 3 {
                // Concave notch (e.g. an inside L): union a quarter-round fill. The rolling
                // circle sits in the notch (+n1,+n2 side); the fill region runs into the
                // material so only the arc is exposed.
                let c0 = add(a, scale(add(n1, n2), radius / (1.0 + c))); // centre, in the notch
                let t1 = add(c0, scale(n1, -radius)); // contact on face 1
                let t2 = add(c0, scale(n2, -radius)); // contact on face 2
                let pad = radius;
                // Cross-section: arc t1→t2 (notch surface), then back through the material.
                let mut cross: Vec<V3> = vec![add(t1, start_shift)];
                const ARC: usize = 16;
                for k in 1..ARC {
                    let f = k as f64 / ARC as f64;
                    // Slerp-ish: interpolate the contact directions around the circle.
                    let dir = norm(add(scale(sub(t1, c0), 1.0 - f), scale(sub(t2, c0), f)));
                    cross.push(add(add(c0, scale(dir, radius)), start_shift));
                }
                cross.push(add(t2, start_shift));
                cross.push(add(add(t2, scale(n2, -pad)), start_shift)); // into material behind face 2
                cross.push(add(add(a, scale(add(n1, n2), -pad)), start_shift)); // deep material
                cross.push(add(add(t1, scale(n1, -pad)), start_shift)); // into material behind face 1
                let fill = extrude_prism(&cross, axis, total_len);
                if fill.indices.len() >= 3 {
                    body = crate::mesh_union(&body, &fill);
                    any = true;
                }
            }
        }
    }
    // Corner blends: where ≥3 distinct convex faces meet at a vertex (a box corner), round
    // it to a sphere octant so the edge fillets join smoothly instead of leaving a notch.
    for (_, (corner, normals)) in corners {
        if normals.len() >= 3 {
            if let Some(tool) = corner_sphere_tool(corner, normals[0], normals[1], normals[2], radius) {
                body = crate::mesh_difference(&body, &tool);
                any = true;
            }
        }
    }
    if any {
        Some(body)
    } else {
        None
    }
}

/// A unit vector perpendicular to `axis` within the plane spanned with `u` (axis ⟂ u
/// assumed); returns axis × u.
fn cross_perp(axis: V3, u: V3) -> V3 {
    cross(axis, u)
}

/// A UV sphere of `radius` centred at `c`.
fn make_sphere(c: V3, radius: f64, nlat: usize, nlong: usize) -> TriMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let f = |p: V3| [p[0] as f32, p[1] as f32, p[2] as f32];
    for i in 0..=nlat {
        let theta = std::f64::consts::PI * i as f64 / nlat as f64; // 0..π
        let (st, ct) = theta.sin_cos();
        for j in 0..nlong {
            let phi = std::f64::consts::TAU * j as f64 / nlong as f64;
            let (sp, cp) = phi.sin_cos();
            positions.push(f([c[0] + radius * st * cp, c[1] + radius * st * sp, c[2] + radius * ct]));
        }
    }
    let idx = |i: usize, j: usize| (i * nlong + (j % nlong)) as u32;
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..nlat {
        for j in 0..nlong {
            let (a, b, cc, d) = (idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j));
            indices.extend_from_slice(&[a, b, cc, a, cc, d]);
        }
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// An axis-aligned-in-frame box: centred near `c`, spanning `[lo, hi]` along each of the
/// orthonormal axes `e1`,`e2`,`e3`.
fn make_box(c: V3, e1: V3, e2: V3, e3: V3, lo: f64, hi: f64) -> TriMesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(8);
    for &z in &[lo, hi] {
        for &y in &[lo, hi] {
            for &x in &[lo, hi] {
                let p = add(c, add(scale(e1, x), add(scale(e2, y), scale(e3, z))));
                positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
    }
    // 12 triangles (corner indexing: bit0=x, bit1=y, bit2=z).
    let q = |a: u32, b: u32, c: u32, d: u32| [a, b, c, a, c, d];
    let mut indices: Vec<u32> = Vec::new();
    for f in [
        q(0, 1, 3, 2), // z=lo
        q(4, 6, 7, 5), // z=hi
        q(0, 4, 5, 1), // y=lo
        q(2, 3, 7, 6), // y=hi
        q(0, 2, 6, 4), // x=lo
        q(1, 5, 7, 3), // x=hi
    ] {
        indices.extend_from_slice(&f);
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// Spherical blend tool for a convex corner where three faces (outward normals `n1,n2,n3`)
/// meet at `corner`: carve `(corner-box − sphere)` so the corner rounds to a sphere octant
/// tangent to all three faces. Assumes near-orthogonal faces. `None` if degenerate.
fn corner_sphere_tool(corner: V3, n1: V3, n2: V3, n3: V3, r: f64) -> Option<TriMesh> {
    // Orthonormalise the frame (Gram–Schmidt); bail if the three faces are near-parallel.
    let e1 = norm(n1);
    let e2 = norm(sub(n2, scale(e1, dot(n2, e1))));
    if len(e2) < 0.3 {
        return None;
    }
    let e3 = norm(sub(sub(n3, scale(e1, dot(n3, e1))), scale(e2, dot(n3, e2))));
    if len(e3) < 0.3 {
        return None;
    }
    // Centre at distance r from each of the three faces, inside material.
    let center = add(corner, scale(add(add(e1, e2), e3), -r));
    let pad = r;
    // The box covers only the corner *octant* (from the sphere-centre planes, coord 0, out
    // past the vertex into air at r+pad). `(box − sphere)` is then just the pointy bit
    // outside the sphere — carving it rounds the corner without leaving a ball in a cavity.
    let bbox = make_box(center, e1, e2, e3, 0.0, r + pad);
    let sphere = make_sphere(center, r, 20, 28);
    let tool = crate::mesh_difference(&bbox, &sphere);
    if tool.indices.len() >= 3 {
        Some(tool)
    } else {
        None
    }
}

/// Round the edges of `mesh` to roughly `radius`. If `edges` is non-empty, only the parts
/// of the body within ≈`radius` of those edge polylines are rounded (a *selective* fillet
/// on the picked edges); otherwise every edge is rounded. `None` if the mesh is empty or
/// the radius is non-positive (caller keeps the original).
pub fn round_mesh(mesh: &TriMesh, radius: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    if radius <= 1e-6 || mesh.indices.len() < 3 {
        return None;
    }
    // Picked edges → exact boolean rolling-ball fillet (crisp). Fall back to the SDF round
    // only if it couldn't handle any picked edge.
    if !edges.is_empty() {
        if let Some(m) = fillet_boolean(mesh, radius, edges) {
            return Some(m);
        }
    }
    let tris: Vec<[V3; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let g = |i: u32| {
                let p = mesh.positions[i as usize];
                [p[0] as f64, p[1] as f64, p[2] as f64]
            };
            [g(t[0]), g(t[1]), g(t[2])]
        })
        .collect();
    if tris.is_empty() {
        return None;
    }

    // Bounding box, padded so the rounded surface (and blur kernel) fit inside the grid.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for t in &tris {
        for v in t {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let pad = radius * 2.0;
    for k in 0..3 {
        lo[k] -= pad;
        hi[k] += pad;
    }
    let ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let max_ext = ext[0].max(ext[1]).max(ext[2]);

    // Voxel size: small enough to resolve the radius (~4 voxels across it), but capped so
    // the grid never explodes past ~100 cells on its longest axis.
    const MAX_DIM: usize = 100;
    let h = (radius / 4.0).max(max_ext / MAX_DIM as f64);
    let dim = |e: f64| ((e / h).ceil() as usize + 1).clamp(2, MAX_DIM + 2);
    let (nx, ny, nz) = (dim(ext[0]), dim(ext[1]), dim(ext[2]));
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let pos = |i: usize, j: usize, k: usize| {
        [lo[0] + i as f64 * h, lo[1] + j as f64 * h, lo[2] + k as f64 * h]
    };

    // Signed distance field: unsigned nearest-triangle distance, signed by the winding
    // number (negative inside).
    let mut sdf = vec![0f32; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = pos(i, j, k);
                let mut d2 = f64::MAX;
                let mut wind = 0.0;
                for t in &tris {
                    d2 = d2.min(point_tri_dist2(p, t[0], t[1], t[2]));
                    wind += solid_angle(p, t[0], t[1], t[2]);
                }
                let inside = (wind / (4.0 * std::f64::consts::PI)).abs() > 0.5;
                let d = d2.sqrt() * if inside { -1.0 } else { 1.0 };
                sdf[idx(i, j, k)] = d as f32;
            }
        }
    }

    // Separable Gaussian blur of the SDF; sigma = radius (in voxels). This is what rounds
    // the edges. Truncate the kernel at 3σ.
    let sigma = (radius / h).max(0.6);
    let krad = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f32> = (-krad..=krad)
        .map(|x| (-(x * x) as f32 / (2.0 * (sigma * sigma) as f32)).exp())
        .collect();
    let ksum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|w| w / ksum).collect();
    let blur_axis = |src: &[f32], dst: &mut [f32], axis: usize| {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut acc = 0f32;
                    for (ki, w) in kernel.iter().enumerate() {
                        let off = ki as isize - krad;
                        let (mut ii, mut jj, mut kk) = (i as isize, j as isize, k as isize);
                        match axis {
                            0 => ii += off,
                            1 => jj += off,
                            _ => kk += off,
                        }
                        let ii = ii.clamp(0, nx as isize - 1) as usize;
                        let jj = jj.clamp(0, ny as isize - 1) as usize;
                        let kk = kk.clamp(0, nz as isize - 1) as usize;
                        acc += w * src[idx(ii, jj, kk)];
                    }
                    dst[idx(i, j, k)] = acc;
                }
            }
        }
    };
    // Keep the original (sharp) field so a selective fillet can leave unselected edges
    // untouched.
    let sharp = if edges.is_empty() { Vec::new() } else { sdf.clone() };
    let mut tmp = vec![0f32; nx * ny * nz];
    blur_axis(&sdf, &mut tmp, 0);
    blur_axis(&tmp, &mut sdf, 1);
    blur_axis(&sdf, &mut tmp, 2);
    let mut field = tmp;

    // Selective fillet: blend the rounded (blurred) field toward the original sharp field
    // away from the chosen edges, so only those edges round. The blend happens on flat
    // faces (where sharp ≈ blurred), so there's no seam. `w` = 1 on the edge, fading to 0
    // beyond ≈2·radius.
    if !edges.is_empty() {
        let segs: Vec<(V3, V3)> = edges
            .iter()
            .flat_map(|chain| chain.windows(2).map(|w| (w[0], w[1])))
            .collect();
        if !segs.is_empty() {
            let near = (radius * 1.3) as f32;
            let far = (radius * 2.3).max(radius * 1.3 + h) as f32;
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let p = pos(i, j, k);
                        let mut d2 = f64::MAX;
                        for &(a, b) in &segs {
                            d2 = d2.min(point_seg_dist2(p, a, b));
                        }
                        let d = d2.sqrt() as f32;
                        let w = (1.0 - ((d - near) / (far - near)).clamp(0.0, 1.0)).clamp(0.0, 1.0);
                        let n = idx(i, j, k);
                        field[n] = sharp[n] * (1.0 - w) + field[n] * w;
                    }
                }
            }
        }
    }

    // Naive surface nets: one vertex per surface-straddling cell, quads across grid edges
    // whose endpoints differ in sign.
    let corner = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ];
    // 12 cube edges as pairs of corner indices.
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    let cell_vert = |i: usize, j: usize, k: usize| (k * (ny - 1) + j) * (nx - 1) + i;
    let mut verts: Vec<[f32; 3]> = vec![[0.0; 3]; (nx - 1) * (ny - 1) * (nz - 1)];
    let mut has_vert = vec![false; (nx - 1) * (ny - 1) * (nz - 1)];

    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let mut s = [0f32; 8];
                for (c, &(dx, dy, dz)) in corner.iter().enumerate() {
                    s[c] = field[idx(i + dx, j + dy, k + dz)];
                }
                let mut sum = [0f64; 3];
                let mut cnt = 0;
                for &(a, b) in &edges {
                    let (sa, sb) = (s[a], s[b]);
                    if (sa <= 0.0) != (sb <= 0.0) {
                        let t = (sa / (sa - sb)) as f64;
                        let (cax, cay, caz) = corner[a];
                        let (cbx, cby, cbz) = corner[b];
                        let pa = pos(i + cax, j + cay, k + caz);
                        let pb = pos(i + cbx, j + cby, k + cbz);
                        for m in 0..3 {
                            sum[m] += pa[m] + t * (pb[m] - pa[m]);
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    let v = cell_vert(i, j, k);
                    verts[v] = [
                        (sum[0] / cnt as f64) as f32,
                        (sum[1] / cnt as f64) as f32,
                        (sum[2] / cnt as f64) as f32,
                    ];
                    has_vert[v] = true;
                }
            }
        }
    }

    // Quads: for each interior grid edge with a sign change, join the 4 cells around it.
    let mut indices: Vec<u32> = Vec::new();
    let mut quad = |a: usize, b: usize, c: usize, d: usize, flip: bool| {
        if has_vert[a] && has_vert[b] && has_vert[c] && has_vert[d] {
            let (a, b, c, d) = (a as u32, b as u32, c as u32, d as u32);
            if flip {
                indices.extend_from_slice(&[a, d, c, a, c, b]);
            } else {
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    };
    // X-edges
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 0..nx - 1 {
                let s0 = field[idx(i, j, k)];
                let s1 = field[idx(i + 1, j, k)];
                if (s0 <= 0.0) != (s1 <= 0.0) {
                    quad(
                        cell_vert(i, j - 1, k - 1),
                        cell_vert(i, j, k - 1),
                        cell_vert(i, j, k),
                        cell_vert(i, j - 1, k),
                        s0 <= 0.0,
                    );
                }
            }
        }
    }
    // Y-edges
    for k in 1..nz - 1 {
        for j in 0..ny - 1 {
            for i in 1..nx - 1 {
                let s0 = field[idx(i, j, k)];
                let s1 = field[idx(i, j + 1, k)];
                if (s0 <= 0.0) != (s1 <= 0.0) {
                    quad(
                        cell_vert(i - 1, j, k - 1),
                        cell_vert(i, j, k - 1),
                        cell_vert(i, j, k),
                        cell_vert(i - 1, j, k),
                        s0 > 0.0,
                    );
                }
            }
        }
    }
    // Z-edges
    for k in 0..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let s0 = field[idx(i, j, k)];
                let s1 = field[idx(i, j, k + 1)];
                if (s0 <= 0.0) != (s1 <= 0.0) {
                    quad(
                        cell_vert(i - 1, j - 1, k),
                        cell_vert(i, j - 1, k),
                        cell_vert(i, j, k),
                        cell_vert(i - 1, j, k),
                        s0 <= 0.0,
                    );
                }
            }
        }
    }

    if indices.is_empty() {
        return None;
    }
    // Compact to only referenced vertices.
    let mut remap = vec![u32::MAX; verts.len()];
    let mut positions: Vec<[f32; 3]> = Vec::new();
    for i in indices.iter_mut() {
        let old = *i as usize;
        if remap[old] == u32::MAX {
            remap[old] = positions.len() as u32;
            positions.push(verts[old]);
        }
        *i = remap[old];
    }
    // Per-vertex normals from face normals (smooth shading).
    let mut normals = vec![[0f32; 3]; positions.len()];
    for t in indices.chunks_exact(3) {
        let p = |i: u32| positions[i as usize];
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
        for &vi in t {
            for m in 0..3 {
                normals[vi as usize][m] += n[m];
            }
        }
    }
    for n in &mut normals {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 1e-9 {
            n[0] /= l;
            n[1] /= l;
            n[2] /= l;
        }
    }
    Some(TriMesh { positions, normals, indices })
}
