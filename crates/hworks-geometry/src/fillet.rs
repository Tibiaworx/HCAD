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
fn fillet_circular(tris: &[[V3; 3]], radius: f64, loop_pts: &[[f64; 3]], tol: f64) -> Option<(TriMesh, bool)> {
    let (center, mut axis, big_r) = fit_circle(loop_pts)?;
    if big_r <= radius * 1.05 {
        return None; // radius too big for this rim
    }
    // Adjacent faces from one rim segment: one ≈ axial (the cap), one ≈ radial (the wall).
    let (n1, n2) = adjacent_normals(tris, loop_pts[0], loop_pts[1], tol.max(1e-3))?;
    let (cap_n, wall_n) = if dot(n1, axis).abs() > dot(n2, axis).abs() { (n1, n2) } else { (n2, n1) };
    if dot(cap_n, axis).abs() < 0.9 || dot(wall_n, axis).abs() > 0.3 {
        return None; // not a flat-cap + round-wall corner
    }
    // Orient `axis` so +axis is the cap's outward normal; radial0 outward to the rim.
    if dot(cap_n, axis) < 0.0 {
        axis = scale(axis, -1.0);
    }
    let radial0 = norm(sub(loop_pts[0], center));
    let wall_n = if dot(wall_n, radial0) < 0.0 { scale(wall_n, -1.0) } else { wall_n };
    // Revolve at the rim vertices' own angles so the torus rings line up with the body's
    // facets (no beat / step at the union seam).
    let tangent = cross(axis, radial0);
    let angles: Vec<f64> = loop_pts
        .iter()
        .map(|p| {
            let d = sub(*p, center);
            dot(d, tangent).atan2(dot(d, radial0))
        })
        .collect();

    // Convex vs concave: probe the four (cap, wall) quadrants around the rim and count how
    // many are solid. A protruding (convex) rim has just one solid quadrant (inside both
    // faces); a notch (concave) has three (only outside-both is air).
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    let rim0 = loop_pts[0];
    let e = radius * 0.4;
    let mut solid = 0;
    for sc in [-1.0_f64, 1.0] {
        for sw in [-1.0_f64, 1.0] {
            let p = add(rim0, add(scale(axis, sc * e), scale(wall_n, sw * e)));
            if inside(p) {
                solid += 1;
            }
        }
    }
    let r = radius;
    let pad = r;
    const ARC: usize = 20;

    if solid == 1 {
        // Convex (e.g. a cylinder top rim): subtract the corner sliver. The tool's outer
        // edges run into open air so the cut is transversal — no curved coincidence.
        let cc = (big_r - r, -r);
        let mut profile: Vec<(f64, f64)> = vec![(big_r - r, 0.0)];
        for k in 1..ARC {
            let ang = std::f64::consts::FRAC_PI_2 * (1.0 - k as f64 / ARC as f64);
            profile.push((cc.0 + r * ang.cos(), cc.1 + r * ang.sin()));
        }
        profile.push((big_r, -r));
        profile.push((big_r + pad, -r));
        profile.push((big_r + pad, pad));
        profile.push((big_r - r, pad));
        let tool = revolve(&profile, center, axis, radial0, &angles);
        Some((tool, false))
    } else if solid == 3 {
        // Concave (e.g. a boss base): union a rounded fill. The tool's non-arc edges run
        // *into* existing material (overlap), so the only exposed new surface is the arc.
        let cc = (big_r + r, r); // rolling-circle centre, in the notch
        let mut profile: Vec<(f64, f64)> = vec![(big_r, r)]; // wall contact
        for k in 1..ARC {
            // Arc from wall contact (180°) to cap contact (270°), bulging to the corner.
            let ang = std::f64::consts::PI + std::f64::consts::FRAC_PI_2 * (k as f64 / ARC as f64);
            profile.push((cc.0 + r * ang.cos(), cc.1 + r * ang.sin()));
        }
        profile.push((big_r + r, 0.0)); // cap contact
        profile.push((big_r + r, -pad)); // down into material
        profile.push((big_r - pad, -pad)); // deep material
        profile.push((big_r - pad, r)); // up into the wall's material, back to wall contact
        let tool = revolve(&profile, center, axis, radial0, &angles);
        Some((tool, true))
    } else {
        None // ambiguous (tangent/coplanar) — leave it sharp
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
            // Cylinder centre offset from the edge: equidistant `radius` from both faces.
            let off = scale(add(n1, n2), -radius / (1.0 + c));
            // Convex test: the centre should sit inside the material. Outward face normals
            // mean "inside" is the −normal side; for a convex ridge `off` points inward.
            let inward = norm(scale(add(n1, n2), -1.0));
            if dot(off, inward) <= 0.0 {
                continue; // concave (or flat) — skip for now (would need a union)
            }
            // Cross-section at the `a` end: the sharp corner `a`, and the two tangent
            // contact points on each face. Extrude along the edge (with end margins).
            let c0 = add(a, off); // cylinder centre at the a-end
            let t1 = add(c0, scale(n1, radius));
            let t2 = add(c0, scale(n2, radius));
            let margin = radius.max(l * 0.0) * 1.5;
            let start_shift = scale(axis, -margin);
            let total_len = l + 2.0 * margin;
            // Corner prism (triangle a, t1, t2) extruded along the edge.
            let cross = [add(a, start_shift), add(t1, start_shift), add(t2, start_shift)];
            let prism = extrude_prism(&cross, axis, total_len);
            // Tangent cylinder, slightly oversized in length to fully span the prism.
            let u = norm(sub(t1, c0));
            let w = norm(cross_perp(axis, u));
            let cyl = make_cylinder(add(c0, start_shift), axis, u, w, radius, total_len, 48);
            // tool = corner sliver outside the cylinder; carve it out of the body.
            let tool = crate::mesh_difference(&prism, &cyl);
            if tool.indices.len() >= 3 {
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
