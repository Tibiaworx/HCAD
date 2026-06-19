//! Kernel-agnostic edge rounding (fillet) for triangle meshes.
//!
//! truck has no fillet operator and Manifold exposes no offset, so we round edges with a
//! classic signed-distance-field technique that works on *any* watertight mesh:
//!
//!  1. Sample the body's signed distance field (SDF) on a voxel grid.
//!  2. Gaussian-blur the field by the fillet radius. Blurring leaves flat faces in place
//!     (their SDF is linear) but rounds every convex/concave edge to ≈ the blur radius —
//!     exactly a fillet/round.
//!  3. Re-extract the zero iso-surface with naive surface nets.
//!
//! It rounds *all* edges by one radius — a global fillet driven by a single "amount".

use crate::TriMesh;

type V3 = [f64; 3];

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

/// Round every edge of `mesh` to roughly `radius`. Returns `None` if the mesh is empty or
/// the radius is non-positive (caller keeps the original).
pub fn round_mesh(mesh: &TriMesh, radius: f64) -> Option<TriMesh> {
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
    let mut tmp = vec![0f32; nx * ny * nz];
    blur_axis(&sdf, &mut tmp, 0);
    blur_axis(&tmp, &mut sdf, 1);
    blur_axis(&sdf, &mut tmp, 2);
    let field = tmp;

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
