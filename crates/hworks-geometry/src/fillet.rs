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
/// that edge). Faces are found by *proximity to the segment* (a triangle with ≥2 vertices
/// lying on the line a–b within its span), so it's robust to the committed mesh
/// re-tessellating or subdividing the edge differently than the stored polyline. `None` if
/// the edge isn't shared by two clearly-different faces.
fn adjacent_normals(tris: &[[V3; 3]], a: V3, b: V3, tol: f64) -> Option<(V3, V3)> {
    let ab = sub(b, a);
    let span = len(ab);
    if span < 1e-9 {
        return None;
    }
    let dir = scale(ab, 1.0 / span);
    let on_seg = |p: V3| {
        let t = dot(sub(p, a), dir);
        if t < -tol || t > span + tol {
            return false;
        }
        let proj = add(a, scale(dir, t.clamp(0.0, span)));
        len(sub(p, proj)) < tol
    };
    let mut normals: Vec<V3> = Vec::new();
    for t in tris {
        if t.iter().filter(|&&v| on_seg(v)).count() >= 2 {
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
/// The solid swept by every body triangle lying in the plane (`n`, `off`), pushed `height` along
/// `n` — that face's own footprint, as a solid.
///
/// A concave fillet rests in the corner between two faces and cannot leave either of them, so
/// intersecting the fill with this trims BOTH ends at once, and in the two opposite directions
/// they need. Where the face's boundary curves away from the fill (a convex outer wall) the fill
/// is cut back off the tab it would otherwise leave standing proud. Where the boundary curves
/// TOWARD it (the r=5 bore) the fill — swept long on purpose — is cut to follow the bore round
/// instead of stopping in the straight chord that leaves a square step. Facet for facet, since the
/// footprint is the tessellated face itself.
///
/// Returns `None` if no such face is in the mesh, or its triangles don't close into a region.
fn face_footprint_prism(tris: &[[V3; 3]], n: V3, off: f64, height: f64, tol: f64) -> Option<TriMesh> {
    let face: Vec<[V3; 3]> = tris
        .iter()
        .filter(|t| {
            let nf = norm(cross_perp(sub(t[1], t[0]), sub(t[2], t[0])));
            dot(nf, n) > 0.999 && t.iter().all(|&v| (dot(n, v) - off).abs() < tol)
        })
        .copied()
        .collect();
    if face.is_empty() {
        return None;
    }
    // Sink the base slightly so the intersection is robust where the fill touches the face.
    let base = scale(n, -tol.max(1e-6) * 10.0);
    let top = scale(n, height);
    let mut b = MeshBuild::default();
    for t in &face {
        b.tri(add(t[0], top), add(t[1], top), add(t[2], top));
        b.tri(add(t[2], base), add(t[1], base), add(t[0], base)); // reversed: outward-facing
    }
    // Walls on the region's boundary — the edges used by exactly one triangle.
    let key = |p: V3| [(p[0] * 1e4).round() as i64, (p[1] * 1e4).round() as i64, (p[2] * 1e4).round() as i64];
    let mut count: HashMap<([i64; 3], [i64; 3]), (usize, V3, V3)> = HashMap::new();
    for t in &face {
        for &(p, q) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (kp, kq) = (key(p), key(q));
            let k = if kp <= kq { (kp, kq) } else { (kq, kp) };
            let e = count.entry(k).or_insert((0, p, q));
            e.0 += 1;
        }
    }
    for (_, (c, p, q)) in count {
        if c != 1 {
            continue; // interior edge
        }
        b.tri(add(p, base), add(q, base), add(q, top));
        b.tri(add(p, base), add(q, top), add(p, top));
    }
    Some(b.finish())
}

/// Minimal triangle-soup accumulator for the footprint prism.
#[derive(Default)]
struct MeshBuild {
    pos: Vec<[f32; 3]>,
}
impl MeshBuild {
    fn tri(&mut self, a: V3, b: V3, c: V3) {
        for p in [a, b, c] {
            self.pos.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    fn finish(self) -> TriMesh {
        let n = self.pos.len();
        let mut normals = Vec::with_capacity(n);
        for t in self.pos.chunks_exact(3) {
            let g = |p: [f32; 3]| [p[0] as f64, p[1] as f64, p[2] as f64];
            let nf = norm(cross_perp(sub(g(t[1]), g(t[0])), sub(g(t[2]), g(t[0]))));
            for _ in 0..3 {
                normals.push([nf[0] as f32, nf[1] as f32, nf[2] as f32]);
            }
        }
        TriMesh { positions: self.pos, normals, indices: (0..n as u32).collect() }
    }
}

/// A big box filling the material side of the plane through `p0` with outward normal `n`.
/// Intersecting a tool with it clips the tool back to that side.
fn halfspace_box(p0: V3, n: V3, size: f64) -> TriMesh {
    let seed = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let u = norm(sub(seed, scale(n, dot(seed, n))));
    let w = norm(cross_perp(n, u));
    let square: Vec<V3> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .map(|&(a, b)| add(p0, add(scale(u, a * size), scale(w, b * size))))
        .collect();
    extrude_prism(&square, scale(n, -1.0), size)
}

/// Trim a concave fillet's fill so it cannot add material outside the walls it runs out into.
///
/// The fill is a prism swept along the edge with FLAT ends, square to the edge. Where the edge
/// terminates on a wall that curves away — the circular junction where a raised feature meets the
/// part below it — the part's boundary bends inward while the prism's end does not, so the fill's
/// far corner is left standing outside the body: the tab you see proud of the wall. Measured on
/// the boss-sector junction before this trim: 134 vertices up to 0.397 outside an r=8 wall, each
/// one the edge's endpoint displaced by the fillet's own radius.
///
/// The right end condition is the fillet trimmed by the faces it runs into, and those faces are
/// already in the body. Take every triangle near the edge's ends that is not part of either face
/// being blended, and clip the fill back to the material side of any whose plane it crosses. On a
/// tessellated convex wall that reproduces the wall exactly, facet for facet; a plane the fill
/// never crosses costs nothing, so a fillet ending in open air is left alone.
fn trim_fill_to_walls(fill: &TriMesh, tris: &[[V3; 3]], ends: [V3; 2], n1: V3, n2: V3, radius: f64, tol: f64) -> TriMesh {
    let verts: Vec<V3> = fill
        .positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    if verts.is_empty() {
        return fill.clone();
    }
    // Only walls within reach of an end can clip anything: the fill never extends further than
    // its own radius from the edge.
    let reach = 2.5 * radius;
    let near_end = |p: V3| ends.iter().any(|&e| len(sub(p, e)) <= reach);
    let mut planes: Vec<(V3, f64)> = Vec::new(); // (outward normal, offset) deduped on a grid
    let mut seen: std::collections::HashSet<[i64; 4]> = std::collections::HashSet::new();
    for t in tris {
        if !t.iter().any(|&v| near_end(v)) {
            continue;
        }
        let nf = norm(cross_perp(sub(t[1], t[0]), sub(t[2], t[0])));
        if len(nf) < 0.5 {
            continue; // degenerate sliver
        }
        // The two faces being blended are where the fill is SUPPOSED to sit proud — clipping to
        // them would erase the fillet itself.
        if dot(nf, n1).abs() > 0.996 || dot(nf, n2).abs() > 0.996 {
            continue;
        }
        let off = dot(nf, t[0]);
        let key = [
            (nf[0] * 1e3).round() as i64,
            (nf[1] * 1e3).round() as i64,
            (nf[2] * 1e3).round() as i64,
            (off * 1e3).round() as i64,
        ];
        if !seen.insert(key) {
            continue;
        }
        // Only a plane the WHOLE body stays behind may be used as a clip. On a convex wall the
        // facet's plane supports the body and cutting the fill back to it is exactly the trim; on
        // a CONCAVE wall — the r=5 bore here — the tangent plane slices clean through the part, so
        // clipping to it would carve away the fillet along with the tab. (That is what happened
        // the first time: bore facets up to 60° away were trimming the fill down to a shaving,
        // leaving the run-out flush but the fillet itself all but gone.)
        let supports = tris
            .iter()
            .flatten()
            .all(|&v| dot(nf, v) - off <= tol.max(1e-6) * 10.0);
        if supports {
            planes.push((nf, off));
        }
    }
    // Box big enough to swallow the fill on the material side.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for v in &verts {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    let size = len(sub(hi, lo)).max(radius) * 4.0;
    let mut out = fill.clone();
    for (nf, off) in planes {
        let outside = verts.iter().map(|&v| dot(nf, v) - off).fold(f64::MIN, f64::max);
        if outside <= tol {
            continue; // the fill stays behind this wall already
        }
        let clipped = crate::mesh_intersection(&out, &halfspace_box(scale(nf, off), nf, size));
        if std::env::var("HCAD_TRIM_DEBUG").is_ok() {
            eprintln!(
                "  clip by n=({:.3},{:.3},{:.3}) off={off:.3}: fill was {outside:.3} past it; tris {} → {}",
                nf[0], nf[1], nf[2], out.indices.len() / 3, clipped.indices.len() / 3
            );
        }
        if clipped.indices.len() >= 3 {
            out = clipped;
        }
    }
    out
}

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
            // A small overrun past the edge ends so adjacent fillets meet at a corner,
            // but not so much that the cut's flat end shows on the neighbouring face.
            let margin = radius * 0.5;
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
                // material so only the arc is exposed. Unlike the convex subtract, the fill
                // must NOT overhang the edge ends (a union past the edge adds a stray tab),
                // so it spans exactly the edge length.
                let c0 = add(a, scale(add(n1, n2), radius / (1.0 + c))); // centre, in the notch
                let t1 = add(c0, scale(n1, -radius)); // contact on face 1
                let t2 = add(c0, scale(n2, -radius)); // contact on face 2
                // Minimal fill: the notch corner bounded by the two faces (a→t1, t2→a) and
                // the arc. Those side edges sit ON the existing faces (a coplanar union),
                // so the fill never reaches past a wall the way a "dig into material"
                // extension would (which pokes out a thin wall at a large radius).
                let mut cross: Vec<V3> = vec![a, t1];
                const ARC: usize = 16;
                for k in 1..ARC {
                    let f = k as f64 / ARC as f64;
                    let dir = norm(add(scale(sub(t1, c0), 1.0 - f), scale(sub(t2, c0), f)));
                    cross.push(add(c0, scale(dir, radius)));
                }
                cross.push(t2);
                // Swept LONG at both ends on purpose, then trimmed to the two faces it rests in.
                // The old fill spanned exactly the edge, on the reasoning that a union past the
                // end adds a stray tab — true, and it does, but stopping square to the edge is
                // just as wrong the other way: where the face's boundary curves toward the fill
                // (the bore) the band ends in a straight chord and leaves a step. Running long
                // and trimming to the footprint gets both ends from one rule.
                let over = scale(axis, -radius);
                let long: Vec<V3> = cross.iter().map(|&p| add(p, over)).collect();
                let fill = extrude_prism(&long, axis, l + 2.0 * radius);
                // Over EITHER face, not both. The two footprints are intersected with the fill as
                // a UNION because the fillet is entitled to run out over whichever face reaches
                // furthest: the ring top's boundary follows the bore round, while the sector's
                // radial face is planar and stops in a straight line at the bore's radius. Taking
                // them separately re-imposes that straight line and puts the step back — the
                // fillet may go where either face still supports it, and ends only where both
                // have run out.
                let dbg = std::env::var("HCAD_TRIM_DEBUG").is_ok();
                let mut allowed: Option<TriMesh> = None;
                for &(nf, on) in &[(n1, a), (n2, a)] {
                    if let Some(fp) = face_footprint_prism(&tris, nf, dot(nf, on), radius * 4.0, tol) {
                        allowed = Some(match allowed {
                            Some(prev) => crate::mesh_union(&prev, &fp),
                            None => fp,
                        });
                    }
                }
                let fill = match allowed {
                    Some(fp) => {
                        let cut = crate::mesh_intersection(&fill, &fp);
                        if dbg {
                            eprintln!("  footprint union: {} tris, fill {} → {}", fp.indices.len() / 3, fill.indices.len() / 3, cut.indices.len() / 3);
                        }
                        if cut.indices.len() >= 3 { cut } else { fill }
                    }
                    None => fill,
                };
                // Belt and braces for anything the footprints couldn't bound.
                let fill = trim_fill_to_walls(&fill, &tris, [a, b], n1, n2, radius, tol);
                if fill.indices.len() >= 3 {
                    body = crate::mesh_union(&body, &fill);
                    any = true;
                }
                // Record this concave edge's faces at both endpoints, for corner blending.
                for &v in &[a, b] {
                    let entry = corners.entry(vkey(v)).or_insert_with(|| (v, Vec::new()));
                    for &nrm in &[n1, n2] {
                        if entry.1.iter().all(|m| dot(*m, nrm) < 0.99) {
                            entry.1.push(nrm);
                        }
                    }
                }
            }
        }
    }
    // Corner blends: where three filleted faces meet at a vertex, round it to a sphere so
    // the edge fillets join smoothly. A convex corner (material in 1 of the 8 octants the
    // three faces carve) subtracts the sphere; a concave corner (a pocket's inside corner,
    // 7 octants solid) unions a rounded fill.
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    for (_, (corner, normals)) in corners {
        if normals.len() >= 3 {
            let (n1, n2, n3) = (normals[0], normals[1], normals[2]);
            let e = radius * 0.4;
            let mut solid = 0;
            for &s1 in &[-1.0_f64, 1.0] {
                for &s2 in &[-1.0_f64, 1.0] {
                    for &s3 in &[-1.0_f64, 1.0] {
                        let p = add(corner, add(scale(n1, s1 * e), add(scale(n2, s2 * e), scale(n3, s3 * e))));
                        if inside(p) {
                            solid += 1;
                        }
                    }
                }
            }
            // Only the clean corner topologies the sphere models correctly: 1 solid octant
            // (a protruding convex corner → subtract) or 7 (a recessed concave corner →
            // union). A 3-octant pocket-rim corner is *not* a sphere (it would scoop a
            // dimple), so it's left to the edge fillets meeting (a small mark, no scoop).
            let concave = match solid {
                1 => false,
                7 => true,
                _ => continue,
            };
            if let Some(tool) = corner_sphere_tool(corner, n1, n2, n3, radius, concave) {
                let cand = if concave { crate::mesh_union(&body, &tool) } else { crate::mesh_difference(&body, &tool) };
                let n0 = body.indices.len();
                if cand.indices.len() >= n0 / 2 && cand.indices.len() <= n0 * 4 {
                    body = cand;
                }
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

/// A box spanning `r1`/`r2`/`r3` (each `(lo, hi)`) along orthonormal axes `e1,e2,e3` from
/// `c`. Used as a clipping region.
fn make_box_ranges(c: V3, e1: V3, e2: V3, e3: V3, r1: (f64, f64), r2: (f64, f64), r3: (f64, f64)) -> TriMesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(8);
    for &z in &[r3.0, r3.1] {
        for &y in &[r2.0, r2.1] {
            for &x in &[r1.0, r1.1] {
                let p = add(c, add(scale(e1, x), add(scale(e2, y), scale(e3, z))));
                positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
    }
    let q = |a: u32, b: u32, cc: u32, d: u32| [a, b, cc, a, cc, d];
    let mut indices: Vec<u32> = Vec::new();
    for f in [q(0, 1, 3, 2), q(4, 6, 7, 5), q(0, 4, 5, 1), q(2, 3, 7, 6), q(0, 2, 6, 4), q(1, 5, 7, 3)] {
        indices.extend_from_slice(&f);
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// A clipping region that hugs the `edges` loop as a *tube*: the union of a box per rim
/// segment (each ±`2·radius` perpendicular and along the loop's normal). This keeps an SDF
/// round local to the edge curve itself, without filling its bounding box (which, for a big
/// pocket in a thin frame, would reach the part's outer edges). `None` if not planar.
fn edge_band_box(edges: &[Vec<[f64; 3]>], radius: f64) -> Option<TriMesh> {
    // Loop normal (Newell), for the perpendicular/axial box extents.
    let mut nrm = [0.0; 3];
    for chain in edges {
        for i in 0..chain.len() {
            let p = chain[i];
            let qn = chain[(i + 1) % chain.len()];
            nrm[0] += (p[1] - qn[1]) * (p[2] + qn[2]);
            nrm[1] += (p[2] - qn[2]) * (p[0] + qn[0]);
            nrm[2] += (p[0] - qn[0]) * (p[1] + qn[1]);
        }
    }
    if len(nrm) < 1e-9 {
        return None;
    }
    let axis = norm(nrm);
    let m = radius * 2.0;
    let mut band: Option<TriMesh> = None;
    for chain in edges {
        for i in 0..chain.len() {
            let a = chain[i];
            let b = chain[(i + 1) % chain.len()];
            let d = sub(b, a);
            let l = len(d);
            if l < 1e-6 {
                continue;
            }
            let e1 = scale(d, 1.0 / l);
            let e2 = norm(cross(e1, axis));
            let mid = scale(add(a, b), 0.5);
            let box_seg = make_box_ranges(mid, e1, e2, axis, (-l * 0.5 - m, l * 0.5 + m), (-m, m), (-m, m));
            band = Some(match band {
                Some(acc) => crate::mesh_union(&acc, &box_seg),
                None => box_seg,
            });
        }
    }
    band
}

/// Sweep a closed 2D `profile` (points in `(w, z)` = horizontal-into-material, along-axis)
/// along the closed loop `pts`, with the profile's `w` axis = the in-plane normal of the
/// loop (`cross(tangent, axis)·w_sign`) and `z` axis = `axis`. A tube whose cross-section
/// is constant but rotates to follow the rim — exact for straight runs and arcs alike.
fn sweep_profile(profile: &[(f64, f64)], pts: &[V3], axis: V3, w_sign: f64) -> TriMesh {
    let n = pts.len();
    let np = profile.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * np);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let next = pts[(i + 1) % n];
        let t = norm(sub(next, prev));
        let w = scale(norm(cross(t, axis)), w_sign);
        for &(wv, zv) in profile {
            let p = add(pts[i], add(scale(w, wv), scale(axis, zv)));
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..n {
        let i1 = (i + 1) % n;
        for j in 0..np {
            let j1 = (j + 1) % np;
            let (a, b, c, d) =
                ((i * np + j) as u32, (i1 * np + j) as u32, (i1 * np + j1) as u32, (i * np + j1) as u32);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// Exact swept rolling-ball fillet on a *planar convex* rim (a slot pocket, a rounded-
/// rectangle rim, …): sweep the corner-sliver cross-section along the rim and subtract it.
/// `None` if the rim isn't planar+convex (caller falls back to the SDF crop).
fn fillet_swept(mesh: &TriMesh, radius: f64, chain: &[[f64; 3]]) -> Option<(TriMesh, bool)> {
    if chain.len() < 6 {
        return None;
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
    let nn = chain.len();
    let center = scale(chain.iter().fold([0.0; 3], |a, p| add(a, *p)), 1.0 / nn as f64);
    let mut nrm = [0.0; 3];
    for i in 0..nn {
        let p = chain[i];
        let q = chain[(i + 1) % nn];
        nrm[0] += (p[1] - q[1]) * (p[2] + q[2]);
        nrm[1] += (p[2] - q[2]) * (p[0] + q[0]);
        nrm[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    if len(nrm) < 1e-9 {
        return None;
    }
    let axis0 = norm(nrm);
    // Planarity: every rim point near the plane.
    if chain.iter().any(|p| dot(sub(*p, center), axis0).abs() > radius * 0.5 + 1e-3) {
        return None;
    }
    // Smoothness: the sweep assumes a tangent-continuous rim (a slot, a rounded rect). A
    // sharp corner (a square pocket) makes the swept tool self-intersect, so bail and let
    // the SDF crop handle it.
    for i in 0..nn {
        let a = chain[(i + nn - 1) % nn];
        let b = chain[i];
        let c = chain[(i + 1) % nn];
        let (t_in, t_out) = (norm(sub(b, a)), norm(sub(c, b)));
        if len(sub(b, a)) > 1e-9 && len(sub(c, b)) > 1e-9 && dot(t_in, t_out) < 0.7 {
            return None; // turn > ~45°
        }
    }
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    let e = radius * 0.4;
    let p0 = chain[0];
    let t0 = norm(sub(chain[1], chain[nn - 1]));
    // Orient the axis to the cap's outward (air) side.
    let axis = if inside(add(p0, scale(axis0, e))) { scale(axis0, -1.0) } else { axis0 };
    // The in-plane normal pointing into material (the +w of the cross-section).
    let mut w0 = norm(cross(t0, axis));
    if !inside(add(p0, add(scale(w0, e), scale(axis, -e)))) {
        w0 = scale(w0, -1.0);
    }
    let w_sign = if dot(w0, cross(t0, axis)) > 0.0 { 1.0 } else { -1.0 };
    // Convex only (one solid quadrant); concave rims fall back.
    let mut solid = 0;
    for sc in [-1.0_f64, 1.0] {
        for sw in [-1.0_f64, 1.0] {
            if inside(add(p0, add(scale(axis, sc * e), scale(w0, sw * e)))) {
                solid += 1;
            }
        }
    }
    if solid != 1 {
        return None;
    }
    // Corner-sliver cross-section in (w, z): material at (+w, −z), rolling circle (r, −r).
    // Outer edges run into air (w<0 / z>0) so the cut is transversal.
    let r = radius;
    let pad = r;
    const ARC: usize = 14;
    let mut profile: Vec<(f64, f64)> = vec![(r, 0.0)]; // cap contact
    for k in 1..ARC {
        let a = std::f64::consts::FRAC_PI_2 * (1.0 + k as f64 / ARC as f64); // 90°→180°
        profile.push((r + r * a.cos(), -r + r * a.sin()));
    }
    profile.push((0.0, -r)); // wall contact
    profile.push((-pad, -r)); // into the pocket (air)
    profile.push((-pad, pad)); // air
    profile.push((r, pad)); // above the cap (air)
    let chain_v3: Vec<V3> = chain.to_vec();
    let tool = sweep_profile(&profile, &chain_v3, axis, w_sign);
    Some((tool, false))
}

/// Round a curved/mixed edge (a slot rim, a freeform loop) while keeping the rest of the
/// body sharp: SDF-round the whole body (which softens every edge), then splice back only
/// the change inside a box hugging the edge. Convex edges lose their shaved shell; concave
/// edges gain their fill — both clipped to the edge band.
fn local_round(body: &TriMesh, radius: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    let rounded = sdf_round(body, radius, &[])?;
    let band = edge_band_box(edges, radius)?;
    let removed = crate::mesh_intersection(&crate::mesh_difference(body, &rounded), &band);
    let mut out = crate::mesh_difference(body, &removed);
    let added = crate::mesh_intersection(&crate::mesh_difference(&rounded, body), &band);
    if added.indices.len() >= 3 {
        out = crate::mesh_union(&out, &added);
    }
    if out.indices.len() >= 3 {
        Some(out)
    } else {
        None
    }
}

/// Spherical blend tool for a convex corner where three faces (outward normals `n1,n2,n3`)
/// meet at `corner`: carve `(corner-box − sphere)` so the corner rounds to a sphere octant
/// tangent to all three faces. Assumes near-orthogonal faces. `None` if degenerate.
fn corner_sphere_tool(corner: V3, n1: V3, n2: V3, n3: V3, r: f64, concave: bool) -> Option<TriMesh> {
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
    let s = add(add(e1, e2), e3); // the diagonal toward the air octant (outward normals)
    let tool = if concave {
        // Concave corner (a pocket's inside corner): the sphere sits in the air octant
        // tangent to all three faces; the bit of the corner cube outside it is the fill —
        // union it. The cube spans only [0, r] from the corner so it doesn't add material
        // past the fillet into the open pocket.
        let center = add(corner, scale(s, r));
        let bbox = make_box(corner, e1, e2, e3, 0.0, r);
        crate::mesh_difference(&bbox, &make_sphere(center, r, 20, 28))
    } else {
        // Convex corner: the sphere sits in the material; carving the corner octant outside
        // it (out past the vertex into air) rounds the corner.
        let center = add(corner, scale(s, -r));
        let bbox = make_box(center, e1, e2, e3, 0.0, r + r);
        crate::mesh_difference(&bbox, &make_sphere(center, r, 20, 28))
    };
    if tool.indices.len() >= 3 {
        Some(tool)
    } else {
        None
    }
}

/// A picked edge is "straight" if all its points are collinear (so the tangent-cylinder
/// path applies). A single 2-point segment counts.
fn is_straight_edge(chain: &[[f64; 3]]) -> bool {
    if chain.len() < 2 {
        return false;
    }
    let (a, b) = (chain[0], chain[chain.len() - 1]);
    let span = len(sub(b, a));
    if span < 1e-9 {
        return false;
    }
    let dir = norm(sub(b, a));
    chain.iter().all(|p| {
        let d = sub(*p, a);
        len(sub(d, scale(dir, dot(d, dir)))) < span * 0.02 + 1e-4
    })
}

/// A loop is "piecewise straight" (a rectangle, a polygon) if every turn between segments
/// is either ~collinear or a sharp corner — i.e. there are no *gradual* arcs. Such a loop
/// is filleted as separate straight sides (crisp cylinders), not a swept/SDF curve.
fn is_piecewise_straight(chain: &[[f64; 3]]) -> bool {
    let n = chain.len();
    if n < 3 {
        return false;
    }
    // An arc is SUSTAINED turning in one direction, not one big turn.
    //
    // This used to judge each vertex alone: a turn between ~5.7° and ~45° meant "arc". But a
    // tessellated arc's per-facet turn shrinks as the tessellation gets finer — the junction rim
    // in roundfilleterror.hcad turns only 2.8° per facet — so the test fell through and a smooth
    // arc was declared piecewise straight. It was then split into a handful of long straight runs
    // and filleted with one crisp cylinder each: faceted where it should curve, and the
    // mismatched cylinders unioned into a non-manifold body.
    //
    // Accumulating same-sign turn is independent of how finely the arc is divided.
    let mut nrm = [0.0; 3];
    for i in 0..n {
        let p = chain[i];
        let q = chain[(i + 1) % n];
        nrm[0] += (p[1] - q[1]) * (p[2] + q[2]);
        nrm[1] += (p[2] - q[2]) * (p[0] + q[0]);
        nrm[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    let axis = if len(nrm) > 1e-9 { norm(nrm) } else { [0.0, 0.0, 1.0] };
    const ARC_TOTAL: f64 = 0.26; // ~15° of accumulated bend is a curve, not tessellation noise
    let (mut run, mut sign) = (0.0_f64, 0.0_f64);
    for i in 0..n {
        let a = chain[(i + n - 1) % n];
        let b = chain[i];
        let c = chain[(i + 1) % n];
        if len(sub(b, a)) < 1e-9 || len(sub(c, b)) < 1e-9 {
            continue;
        }
        let (t_in, t_out) = (norm(sub(b, a)), norm(sub(c, b)));
        let d = dot(t_in, t_out);
        if d < 0.7 {
            // A real corner: the bend so far belonged to the run that just ended.
            run = 0.0;
            sign = 0.0;
            continue;
        }
        let ang = d.clamp(-1.0, 1.0).acos();
        if ang < 1e-5 {
            continue; // dead straight, and its turn direction is meaningless
        }
        let s = if dot(cross(t_in, t_out), axis) < 0.0 { -1.0 } else { 1.0 };
        if s != sign {
            run = 0.0;
            sign = s;
        }
        run += ang;
        if run > ARC_TOTAL {
            return false; // sustained bend ⇒ an arc
        }
    }
    true
}

/// Split a piecewise-straight loop into its straight runs at the sharp corners.
fn split_straight_runs(chain: &[[f64; 3]]) -> Vec<Vec<[f64; 3]>> {
    let n = chain.len();
    let mut corners = Vec::new();
    for i in 0..n {
        let a = chain[(i + n - 1) % n];
        let b = chain[i];
        let c = chain[(i + 1) % n];
        if len(sub(b, a)) > 1e-9 && len(sub(c, b)) > 1e-9 && dot(norm(sub(b, a)), norm(sub(c, b))) < 0.7 {
            corners.push(i);
        }
    }
    if corners.len() < 2 {
        return vec![chain.to_vec()];
    }
    let mut runs = Vec::new();
    for k in 0..corners.len() {
        let (start, end) = (corners[k], corners[(k + 1) % corners.len()]);
        let mut run = Vec::new();
        let mut i = start;
        loop {
            run.push(chain[i]);
            if i == end {
                break;
            }
            i = (i + 1) % n;
        }
        runs.push(run);
    }
    runs
}

/// Round the picked `edges`: a circular rim → torus, a straight edge → tangent cylinder
/// (both exact booleans), and anything curved or mixed (a slot rim, a freeform loop) → the
/// SDF round masked to that edge — which follows the whole outline smoothly instead of
/// faceting it into starbursts. With no edges, rounds every edge (global SDF).
pub fn round_mesh(mesh: &TriMesh, radius: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    if radius <= 1e-6 || mesh.indices.len() < 3 {
        return None;
    }
    if edges.is_empty() {
        return sdf_round(mesh, radius, &[]);
    }
    let mut boolean_edges: Vec<Vec<[f64; 3]>> = Vec::new();
    let mut curved_edges: Vec<Vec<[f64; 3]>> = Vec::new();
    for chain in edges {
        if fit_circle(chain).is_some() || is_straight_edge(chain) {
            boolean_edges.push(chain.clone());
        } else if is_piecewise_straight(chain) {
            // A rectangle / polygon rim → fillet each straight side with a crisp cylinder.
            boolean_edges.extend(split_straight_runs(chain));
        } else {
            curved_edges.push(chain.clone());
        }
    }
    let mut body = mesh.clone();
    let mut any = false;
    if !boolean_edges.is_empty() {
        if let Some(m) = fillet_boolean(&body, radius, &boolean_edges) {
            body = m;
            any = true;
        }
    }
    for chain in &curved_edges {
        // Exact swept fillet for a planar convex rim (a slot); else crop the SDF round.
        if let Some((tool, is_union)) = fillet_swept(&body, radius, chain) {
            if tool.indices.len() >= 3 {
                body = if is_union { crate::mesh_union(&body, &tool) } else { crate::mesh_difference(&body, &tool) };
                any = true;
            }
        } else if let Some(m) = local_round(&body, radius, std::slice::from_ref(chain)) {
            body = m;
            any = true;
        }
    }
    if any {
        Some(body)
    } else {
        None
    }
}

/// Chamfer (flat bevel) on the picked `edges` by `dist`. Same edge classification and
/// boolean machinery as the fillet, but the cross-section is a straight cut instead of an
/// arc: a straight edge loses a triangular wedge; a circular rim loses a cone frustum.
/// Convex edges only (a chamfer adds no material). `None` if nothing was beveled.
pub fn chamfer_mesh(mesh: &TriMesh, dist: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    if dist <= 1e-6 || mesh.indices.len() < 3 || edges.is_empty() {
        return None;
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
    // Clamp to under half the smallest body dimension (same guard as the fillet).
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for t in &tris {
        for v in t {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let min_ext = (hi[0] - lo[0]).min(hi[1] - lo[1]).min(hi[2] - lo[2]).max(1e-6);
    let dist = dist.min(0.49 * min_ext);
    let diag = len(sub(hi, lo)).max(1.0);
    let tol = diag * 1e-4;

    // Build one wedge tool per straight *run* (using the run's endpoints — a run is
    // collinear, so a single wedge spans it) and one cone per circular rim. Crucially this
    // is *not* per tessellated point: a straight body edge is often split into many
    // collinear points, and a Manifold boolean per pair would re-mesh the whole body dozens
    // of times and freeze the app. One boolean per run keeps it snappy, like the fillet.
    let mut tools: Vec<TriMesh> = Vec::new();
    for chain in edges {
        if fit_circle(chain).is_some() {
            if let Some(tool) = chamfer_circular(&tris, dist, chain) {
                if tool.indices.len() >= 3 {
                    tools.push(tool);
                }
            }
            continue;
        }
        let runs: Vec<Vec<[f64; 3]>> = if is_straight_edge(chain) {
            vec![chain.clone()]
        } else if is_piecewise_straight(chain) {
            split_straight_runs(chain)
        } else {
            continue; // curved/slot chamfer not handled in v1
        };
        for run in &runs {
            if run.len() < 2 {
                continue;
            }
            let (a, b) = (run[0], run[run.len() - 1]);
            if let Some(tool) = chamfer_straight_seg(&tris, dist, a, b, tol) {
                if tool.indices.len() >= 3 {
                    tools.push(tool);
                }
            }
        }
    }
    if tools.is_empty() {
        return None;
    }
    let mut body = mesh.clone();
    for tool in &tools {
        body = crate::mesh_difference(&body, tool);
    }
    Some(body)
}

/// Wedge tool for a straight convex edge `a`–`b`: the triangle (edge, contact-on-face-1,
/// contact-on-face-2) extruded along the edge. Subtracting it leaves the flat bevel.
fn chamfer_straight_seg(tris: &[[V3; 3]], dist: f64, a: V3, b: V3, tol: f64) -> Option<TriMesh> {
    let d = sub(b, a);
    let l = len(d);
    if l < tol {
        return None;
    }
    let axis = norm(d);
    let (n1, n2) = adjacent_normals(tris, a, b, tol.max(1e-3))?;
    // Convex only: one solid quadrant around the edge midpoint.
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    let m = scale(add(a, b), 0.5);
    let e = dist * 0.4;
    let mut solid = 0;
    for s1 in [-1.0_f64, 1.0] {
        for s2 in [-1.0_f64, 1.0] {
            if inside(add(m, add(scale(n1, s1 * e), scale(n2, s2 * e)))) {
                solid += 1;
            }
        }
    }
    if solid != 1 {
        return None;
    }
    // In-plane directions along each face, away from the edge into the material (the
    // inward bisector projected onto each face). A nearly-flat edge (n1 ≈ −n2) makes the
    // bisector degenerate — bail rather than emit a NaN/blown-up prism that would make the
    // Manifold boolean choke.
    let bis = add(n1, n2);
    if len(bis) < 1e-3 {
        return None;
    }
    let inward = norm(scale(bis, -1.0));
    let f1 = norm(sub(inward, scale(n1, dot(inward, n1))));
    let f2 = norm(sub(inward, scale(n2, dot(inward, n2))));
    let p1 = add(a, scale(f1, dist));
    let p2 = add(a, scale(f2, dist));
    let margin = dist * 0.5;
    let shift = scale(axis, -margin);
    let cross = [add(a, shift), add(p1, shift), add(p2, shift)];
    // Reject any non-finite vertex — a degenerate cross-section can otherwise hang the CSG.
    if cross.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
        return None;
    }
    Some(extrude_prism(&cross, axis, l + 2.0 * margin))
}

/// Cone-frustum tool for a circular convex rim: revolve the straight chamfer cross-section
/// (cap contact → wall contact) around the axis.
fn chamfer_circular(tris: &[[V3; 3]], dist: f64, loop_pts: &[[f64; 3]]) -> Option<TriMesh> {
    let (center, axis0, big_r) = fit_circle(loop_pts)?;
    if big_r <= dist * 1.05 {
        return None;
    }
    let radial0 = {
        let r = sub(loop_pts[0], center);
        norm(sub(r, scale(axis0, dot(r, axis0))))
    };
    let inside = |p: V3| {
        let w: f64 = tris.iter().map(|t| solid_angle(p, t[0], t[1], t[2])).sum();
        (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
    };
    let e = dist * 0.4;
    let rim0 = loop_pts[0];
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
    if solid != 1 {
        return None; // convex rims only
    }
    let (mut sc0, mut sw0) = (1.0, 1.0);
    for (i, &sc) in signs.iter().enumerate() {
        for (j, &sw) in signs.iter().enumerate() {
            if q[i][j] {
                sc0 = sc;
                sw0 = sw;
            }
        }
    }
    let axis = scale(axis0, -sc0);
    let sign_w = -sw0;
    let tangent = cross(axis, radial0);
    // Align the revolve to the mesh rim vertices (no step).
    let mut ring: Vec<(f64, [i64; 3])> = Vec::new();
    let plane_tol = big_r * 0.03;
    for t in tris {
        for &v in t {
            let dd = sub(v, center);
            let axial = dot(dd, axis);
            let radial = len(sub(dd, scale(axis, axial)));
            if axial.abs() < plane_tol && (radial - big_r).abs() < plane_tol {
                ring.push((dot(dd, tangent).atan2(dot(dd, radial0)), [(v[0] * 1e3) as i64, (v[1] * 1e3) as i64, (v[2] * 1e3) as i64]));
            }
        }
    }
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ring.dedup_by_key(|r| r.1);
    let angles: Vec<f64> = if ring.len() >= 8 {
        ring.iter().map(|r| r.0).collect()
    } else {
        loop_pts.iter().map(|p| { let dd = sub(*p, center); dot(dd, tangent).atan2(dot(dd, radial0)) }).collect()
    };
    let pad = dist;
    let sz = |u: f64, v: f64| -> (f64, f64) { (big_r + sign_w * u, v) };
    // Straight bevel: cap contact (−d, 0) → wall contact (0, −d); outer edges into air.
    let profile = vec![sz(-dist, 0.0), sz(0.0, -dist), sz(pad, -dist), sz(pad, pad), sz(-dist, pad)];
    Some(revolve(&profile, center, axis, radial0, &angles))
}

/// SDF rolling-ball round: sample the signed-distance field on a voxel grid, blur it by the
/// radius, re-extract with surface nets. If `edges` is non-empty, the round is masked to
/// the body within ≈radius of those polylines; otherwise every edge rounds.
fn sdf_round(mesh: &TriMesh, radius: f64, edges: &[Vec<[f64; 3]>]) -> Option<TriMesh> {
    if radius <= 1e-6 || mesh.indices.len() < 3 {
        return None;
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
            // Keep the rounded band tight to the edge so the fillet doesn't reach into
            // nearby geometry: full effect to ≈1·r, fading out by ≈1.7·r.
            let near = radius as f32;
            let far = (radius * 1.7).max(radius + h) as f32;
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

// ---------------------------------------------------------------------------
// Threads — the "Hole Genie": tap a threaded hole, or add a thread to a hole/boss.
// A thread is a helical triangular ridge (a coil) swept around the axis and booleaned in:
// internal threads (tapped hole) union ridges into a clearance hole; external threads on a
// boss subtract a helical V-groove. Pitch sets the helix advance per turn; the 60° ISO V is
// approximated by a triangular tooth (pointed crest/root) — fine for a visual model.
// ---------------------------------------------------------------------------

/// A unit vector perpendicular to `a`.
fn any_perp(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    norm(cross(a, t))
}

/// A helical triangular coil around `axis` through `o`, advancing `pitch` per turn over
/// `length`. Each cross-section is a tooth: two outer vertices at `r_out` (± half-width along
/// the axis) and an apex at `r_in`. `rh` = right-handed. Used as a boolean tool.
fn thread_coil(o: V3, axis: V3, r_out: f64, r_in: f64, pitch: f64, length: f64, rh: bool) -> TriMesh {
    let a = norm(axis);
    let u = any_perp(a);
    let v = cross(a, u);
    let hw = pitch * 0.25; // tooth half-width along the axis
    let spt = 32usize; // angular steps per turn
    let turns = (length / pitch).max(0.0);
    let n = ((turns * spt as f64).ceil() as usize).max(2);
    let sign = if rh { 1.0 } else { -1.0 };
    // Start a hair below the face so the first tooth doesn't poke above it.
    let z0 = -hw;
    // Run-out: over the first/last half-turn fade the tooth apex back out to the wall
    // (`r_out`), so the coil's flat end caps collapse onto the wall — no tab at the top, no
    // stray geometry at the bottom of the bore.
    let ramp = (spt / 2).max(1);
    let cs = |k: usize| -> [V3; 3] {
        let theta = std::f64::consts::TAU * k as f64 / spt as f64;
        let z = z0 - pitch * theta / std::f64::consts::TAU; // spiral into the material (−axis)
        let rad = add(scale(u, (sign * theta).cos()), scale(v, (sign * theta).sin()));
        let base = add(o, scale(a, z));
        // Fade the tooth out over the run-out — but NEVER fully: at taper 0 the apex lands
        // exactly on the base circle, the cross-section goes collinear (zero area), and
        // Manifold's vertex merge folds those degenerate stations into broken topology —
        // rejecting the coil and kicking EVERY thread boolean to the lossy BSP fallback.
        // A 25% floor keeps every section a real triangle; the stub ends sit buried/blunt.
        let taper = (k as f64 / ramp as f64).min((n - k) as f64 / ramp as f64).clamp(0.25, 1.0);
        let r_apex = r_out + (r_in - r_out) * taper; // r_out at the ends, r_in in the middle
        let outer = |dz: f64| add(add(base, scale(a, dz)), scale(rad, r_out));
        [outer(-hw), outer(hw), add(base, scale(rad, r_apex))]
    };
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((n + 1) * 3);
    for k in 0..=n {
        for p in cs(k) {
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let vi = |k: usize, j: usize| (k * 3 + j) as u32;
    let mut indices: Vec<u32> = Vec::new();
    for k in 0..n {
        for e in 0..3 {
            let (j0, j1) = (e, (e + 1) % 3);
            let (a0, a1, b1, b0) = (vi(k, j0), vi(k, j1), vi(k + 1, j1), vi(k + 1, j0));
            indices.extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
        }
    }
    // End caps, wound OPPOSITE to the tube walls' traversal of the ring edges — a consistently
    // oriented closed mesh uses every directed edge exactly once. Both caps used to run the same
    // way as the walls (6 duplicated directed edges), which made Manifold reject the coil and
    // dropped EVERY thread boolean to the lossy BSP fallback (the "surface may be torn" banner).
    indices.extend_from_slice(&[vi(0, 2), vi(0, 1), vi(0, 0)]); // start cap
    indices.extend_from_slice(&[vi(n, 0), vi(n, 1), vi(n, 2)]); // end cap
    let mut m = TriMesh { positions, normals: Vec::new(), indices };
    orient_outward(&mut m);
    fill_normals(&mut m);
    m
}

/// Tap a threaded hole: drill a clearance hole of `major_d` from the face at `origin` (with
/// outward normal `axis`) down `depth`, then union the internal thread ridges (crests at the
/// minor diameter). `internal` true taps a hole; false threads an existing boss of `major_d`
/// (subtract a helical V-groove instead). Returns the modified body.
pub fn threaded_hole(
    body: &TriMesh,
    origin: V3,
    axis: V3,
    major_d: f64,
    pitch: f64,
    depth: f64,
    internal: bool,
    rh: bool,
) -> Option<TriMesh> {
    if major_d <= 1e-4 || pitch <= 1e-4 || depth <= 1e-4 || body.indices.len() < 3 {
        return None;
    }
    let a = norm(axis);
    let u = any_perp(a);
    let w = cross(a, u);
    let minor_d = (major_d - 1.0825 * pitch).max(major_d * 0.55);
    let hw = 0.25 * pitch;
    // Clamp the thread depth to the material actually behind the hole: cast a ray from the
    // face along −axis and find where it exits the solid (the back face). The thread can't be
    // built deeper than that, so an over-specified depth never spawns geometry past the body.
    let tris: Vec<[V3; 3]> = body
        .indices
        .chunks_exact(3)
        .map(|t| {
            let g = |i: u32| {
                let p = body.positions[i as usize];
                [p[0] as f64, p[1] as f64, p[2] as f64]
            };
            [g(t[0]), g(t[1]), g(t[2])]
        })
        .collect();
    let avail = axis_exit(&tris, origin, scale(a, -1.0)).unwrap_or(depth + 1.0);
    // THROUGH hole: the user asked for a depth at/past the far side. The bore must punch clean
    // out the back (clamping it left a paper-thin floor skin); the thread runs to just shy of
    // the exit so the last tooth doesn't poke out of the bottom face.
    let through = depth >= avail - 2.5 * hw;
    // The bottom tooth's outer base reaches ~hw below the last helix point (which itself sits
    // hw below the face), so leave ~2.5·hw of clearance to keep the whole tooth inside.
    let thread_depth = if through { (avail - 2.5 * hw).max(0.5) } else { depth.min((avail - 2.5 * hw).max(0.5)) };
    let mut out = body.clone();
    if internal {
        // Drill the clearance hole (major), starting just above the face going inward. A blind
        // hole runs a bit past the thread so the bore floor sits below the run-out; a through
        // hole overshoots the far side so no skin survives the subtraction.
        let bore_len = if through { avail + 1.0 } else { thread_depth + 0.5 * pitch + 0.4 };
        let start = add(origin, scale(a, 0.2));
        let hole = make_cylinder(start, scale(a, -1.0), u, w, major_d * 0.5, bore_len + 0.2, 48);
        out = crate::mesh_difference(&out, &hole);
        // Union the thread ridges (apex at minor, base buried a hair in the wall).
        let r_out = major_d * 0.5 + 0.1 * pitch;
        let coil = thread_coil(origin, a, r_out, minor_d * 0.5, pitch, thread_depth, rh);
        if coil.indices.len() >= 3 {
            out = crate::mesh_union(&out, &coil);
        }
    } else {
        // External: cut a helical V-groove into the boss (crest stays at major, root at minor).
        let r_out = major_d * 0.5 + 0.1 * pitch; // base outside the surface
        let groove = thread_coil(origin, a, r_out, minor_d * 0.5, pitch, thread_depth, rh);
        if groove.indices.len() >= 3 {
            out = crate::mesh_difference(&out, &groove);
        }
    }
    (out.indices.len() >= 3).then_some(out)
}

/// Distance from `o` along unit `dir` to where the ray exits the solid `tris` (the first
/// front/back surface it crosses after stepping just inside). `None` if it never exits.
fn axis_exit(tris: &[[V3; 3]], o: V3, dir: V3) -> Option<f64> {
    // Step WELL inside before looking for the exit: the placement point comes from a face
    // pick/snap and can sit a few thousandths OFF the surface — a 1e-3 step then "exited"
    // at the entry face itself, read the body as paper-thin, and clamped every thread to
    // the 0.5 minimum depth (the "hole only goes a small depth" bug). 0.05 clears any
    // realistic snap error while staying far below a tappable wall thickness.
    const EPS: f64 = 0.05;
    let oo = add(o, scale(dir, EPS));
    let mut best: Option<f64> = None;
    for t in tris {
        if let Some(td) = ray_tri(oo, dir, t[0], t[1], t[2]) {
            if td > 1e-3 {
                best = Some(best.map_or(td, |b: f64| b.min(td)));
            }
        }
    }
    best.map(|t| t + EPS)
}

/// Möller–Trumbore ray/triangle: distance `t > 0` along `dir` from `o` to triangle `abc`.
fn ray_tri(o: V3, dir: V3, a: V3, b: V3, c: V3) -> Option<f64> {
    let e1 = sub(b, a);
    let e2 = sub(c, a);
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let tv = sub(o, a);
    let uu = dot(tv, p) * inv;
    if !(-1e-7..=1.0 + 1e-7).contains(&uu) {
        return None;
    }
    let q = cross(tv, e1);
    let vv = dot(dir, q) * inv;
    if vv < -1e-7 || uu + vv > 1.0 + 1e-7 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t > 0.0).then_some(t)
}

#[cfg(test)]
mod coil_tests {
    use super::*;

    #[test]
    fn thread_coil_is_manifold() {
        let coil = thread_coil([0.0, 0.0, 20.0], [0.0, 0.0, 1.0], 2.58, 2.07, 0.8, 9.0, true);
        eprintln!("coil: {} verts {} tris", coil.positions.len(), coil.indices.len() / 3);
        let (v, t, boundary, nonman) = crate::mesh_bool::weld_edge_stats(&coil);
        eprintln!("welded: v={v} t={t} boundary={boundary} nonman={nonman}");
        // Orientation audit: in a consistently-wound closed mesh every DIRECTED edge appears
        // exactly once (its partner traverses the reverse direction).
        {
            use std::collections::HashMap;
            let mut dir_edges: HashMap<(u32, u32), u32> = HashMap::new();
            for tri in coil.indices.chunks(3) {
                for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    *dir_edges.entry((a, b)).or_default() += 1;
                }
            }
            let dup = dir_edges.values().filter(|&&c| c > 1).count();
            let zero_area = coil
                .indices
                .chunks(3)
                .filter(|t| {
                    let p = |i: u32| {
                        let q = coil.positions[t[i as usize % 3] as usize];
                        [q[0] as f64, q[1] as f64, q[2] as f64]
                    };
                    let (a, b, c) = (p(0), p(1), p(2));
                    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
                    (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() < 1e-10
                })
                .count();
            eprintln!("directed dup edges={dup} zero-area tris={zero_area}");
        }
        assert!(crate::mesh_bool::is_manifold(&coil), "thread coil mesh is not 2-manifold");
    }

    #[test]
    fn bore_cylinder_is_manifold() {
        let cyl = make_cylinder([0.0, 0.0, 20.2], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.5, 9.8, 48);
        assert!(crate::mesh_bool::is_manifold(&cyl), "bore cylinder mesh is not 2-manifold");
    }

    /// A curve must not read as "piecewise straight" just because it is finely tessellated.
    ///
    /// The test used to judge each vertex alone — a turn between ~5.7° and ~45° meant "arc" — so
    /// an arc's per-facet turn fell below the window as soon as the tessellation got good. The
    /// junction rim in roundfilleterror.hcad turns 2.8° per facet, was declared straight, and was
    /// then filleted as a handful of long cylinders: faceted where it should curve, and
    /// non-manifold where the mismatched cylinders met.
    #[test]
    fn a_finely_tessellated_arc_is_not_mistaken_for_a_straight_run() {
        let arc = |r: f64, n: usize, sweep: f64| -> Vec<[f64; 3]> {
            (0..n).map(|k| {
                let t = sweep * k as f64 / (n - 1) as f64;
                [r * t.cos(), r * t.sin(), 0.0]
            }).collect()
        };
        // A full circle, at tessellations from coarse to very fine. Every one is a curve.
        for n in [12usize, 24, 48, 96, 192] {
            let ring: Vec<[f64; 3]> = arc(10.0, n + 1, std::f64::consts::TAU)[..n].to_vec();
            assert!(!is_piecewise_straight(&ring), "a {n}-facet circle read as piecewise straight");
        }
        // A genuine polygon still does — that is what the classification is FOR. (Not an octagon:
        // its 45° turn sits exactly on the "is this a corner" threshold, so it read as a curve
        // before this change too. Left alone rather than quietly retuned here.)
        for sides in [3usize, 4, 6] {
            let poly: Vec<[f64; 3]> = (0..sides)
                .map(|k| {
                    let t = std::f64::consts::TAU * k as f64 / sides as f64;
                    [10.0 * t.cos(), 10.0 * t.sin(), 0.0]
                })
                .collect();
            assert!(is_piecewise_straight(&poly), "a {sides}-gon was not seen as piecewise straight");
        }
        // The case from the file: a closed rim of two arcs joined by two straight radial ends —
        // an annular sector. It contains curves, so it is not piecewise straight.
        let mut rim: Vec<[f64; 3]> = Vec::new();
        let sweep = 2.0;
        for p in arc(12.5, 60, sweep) {
            rim.push(p);
        }
        for p in arc(9.5, 60, sweep).into_iter().rev() {
            rim.push(p);
        }
        assert!(!is_piecewise_straight(&rim), "an annular-sector rim read as piecewise straight");
        // ...and a rectangle of the same point count still does, however finely its SIDES are
        // subdivided — subdivision alone must not change the answer either way.
        let mut rect: Vec<[f64; 3]> = Vec::new();
        for (a, b) in [([0.0, 0.0], [20.0, 0.0]), ([20.0, 0.0], [20.0, 8.0]), ([20.0, 8.0], [0.0, 8.0]), ([0.0, 8.0], [0.0, 0.0])] {
            for k in 0..30 {
                let t = k as f64 / 30.0;
                rect.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0]);
            }
        }
        assert!(is_piecewise_straight(&rect), "a finely subdivided rectangle stopped reading as straight");
    }
}
