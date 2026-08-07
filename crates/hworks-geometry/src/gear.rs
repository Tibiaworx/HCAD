//! Involute spur-gear profiles — the geometry behind Gear Genie.
//!
//! A real gear tooth is an **involute of the base circle**: the curve traced by unwinding a
//! taut string off it. That shape is what makes gears run smoothly, because two involutes in
//! mesh keep a constant velocity ratio however far the teeth have rolled through contact. A
//! tooth drawn as an arc or a trapezoid looks similar and runs badly.
//!
//! Everything here works in the standard terms: **module** (mm of pitch diameter per tooth)
//! and **pressure angle** (20° almost always). Two gears mesh if their module and pressure
//! angle match — the tooth counts set the ratio and can be anything.

/// A spur gear's defining numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GearSpec {
    pub teeth: u32,
    /// Millimetres of pitch diameter per tooth. Pitch diameter = module × teeth.
    pub module: f64,
    /// Degrees; 20 is the modern standard, 14.5 shows up on older stock.
    pub pressure_angle: f64,
    /// Centre bore diameter. 0 leaves the gear solid.
    pub bore: f64,
    /// Trimmed off both flanks so a printed pair isn't a press fit, in mm of arc.
    pub backlash: f64,
}

impl Default for GearSpec {
    fn default() -> Self {
        GearSpec { teeth: 20, module: 2.0, pressure_angle: 20.0, bore: 6.0, backlash: 0.0 }
    }
}

impl GearSpec {
    /// The circle where two meshing gears roll without slipping. All the standard sizes hang
    /// off this one.
    pub fn pitch_radius(&self) -> f64 {
        self.module * self.teeth as f64 * 0.5
    }

    /// The circle the involute unwinds from.
    pub fn base_radius(&self) -> f64 {
        self.pitch_radius() * self.pressure_angle.to_radians().cos()
    }

    /// Outside (tip) radius: one module of addendum above the pitch circle.
    pub fn tip_radius(&self) -> f64 {
        self.pitch_radius() + self.module
    }

    /// Root radius: 1.25 modules of dedendum below it, the extra quarter giving tip clearance.
    pub fn root_radius(&self) -> f64 {
        (self.pitch_radius() - 1.25 * self.module).max(0.1)
    }

    /// Centre-to-centre spacing to mesh with another gear of the same module.
    pub fn centre_distance(&self, other: &GearSpec) -> f64 {
        self.pitch_radius() + other.pitch_radius()
    }

    /// Below this tooth count a standard 20° tooth is undercut at the root — the cutter digs
    /// into the flank, weakening the tooth. Printable, but worth flagging.
    pub fn undercut_limit(&self) -> u32 {
        let a = self.pressure_angle.to_radians();
        (2.0 / (a.sin() * a.sin())).ceil() as u32
    }

    pub fn is_undercut(&self) -> bool {
        self.teeth < self.undercut_limit()
    }
}

/// The involute function: how far round the base circle you have unwound to reach a point
/// where the pressure angle is `a`.
fn inv(a: f64) -> f64 {
    a.tan() - a
}

/// Half the tooth's angular thickness at radius `rho`.
///
/// Fixed at the pitch circle by the standard tooth thickness (half the circular pitch), and
/// shrinking outward exactly as the involute does — which is why a tooth is fat at the root
/// and pointed at the tip.
fn half_tooth_angle(spec: &GearSpec, rho: f64) -> f64 {
    let rb = spec.base_radius();
    let alpha = spec.pressure_angle.to_radians();
    // Half the tooth's share of the pitch circle.
    let psi_p = std::f64::consts::PI / (2.0 * spec.teeth as f64);
    // Below the base circle the involute doesn't exist; hold the base-circle value and run
    // radially down to the root.
    let rho = rho.max(rb * (1.0 + 1e-12));
    let a_rho = (rb / rho).clamp(-1.0, 1.0).acos();
    let backlash = if spec.backlash > 0.0 { spec.backlash / (2.0 * spec.pitch_radius()) } else { 0.0 };
    psi_p + inv(alpha) - inv(a_rho) - backlash
}

/// The closed outline of a spur gear, counter-clockwise, plus its bore as a hole loop.
///
/// `steps` is how finely each flank is sampled; 12 is plenty for printing.
pub fn gear_profile(spec: &GearSpec, steps: usize) -> Option<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    if spec.teeth < 3 || spec.module <= 0.0 || !(1.0..=45.0).contains(&spec.pressure_angle) {
        return None;
    }
    let n = spec.teeth as f64;
    let rb = spec.base_radius();
    let rf = spec.root_radius();
    let mut ra = spec.tip_radius();

    // A tooth whose flanks meet before the tip circle would come to a point and then cross
    // itself. Pull the tip in to just before that happens rather than emitting a bow tie.
    if half_tooth_angle(spec, ra) <= 1e-3 {
        let mut lo = spec.pitch_radius();
        let mut hi = ra;
        for _ in 0..60 {
            let mid = (lo + hi) * 0.5;
            if half_tooth_angle(spec, mid) > 1e-3 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        ra = lo;
    }
    if ra <= rf + 1e-9 {
        return None; // no tooth height at all
    }

    let steps = steps.max(4);
    let pitch = std::f64::consts::TAU / n;
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(spec.teeth as usize * (steps * 2 + 8));
    let at = |rho: f64, ang: f64| [rho * ang.cos(), rho * ang.sin()];

    // Radii sampled up one flank, stepped uniformly in the involute's own roll angle
    // (rho = rb·sqrt(1 + theta²)). That naturally clusters points low down, where the curve
    // turns hardest, without the arbitrary bunching a t² ramp in radius produces — that ramp
    // put its first two points 5 µm apart on a 44 mm gear, below the mesh kernel's weld
    // tolerance, so the welder collapsed them and left degenerate triangles that tore the
    // surface and wrecked every boolean downstream.
    let start = rf.max(rb);
    let roll = |rho: f64| ((rho / rb).powi(2) - 1.0).max(0.0).sqrt();
    let (th0, th1) = (roll(start), roll(ra));
    let flank: Vec<f64> = (0..=steps)
        .map(|i| {
            let th = th0 + (th1 - th0) * i as f64 / steps as f64;
            (rb * (1.0 + th * th).sqrt()).clamp(start, ra)
        })
        .collect();

    for i in 0..spec.teeth {
        let beta = pitch * i as f64;
        let h_root = half_tooth_angle(spec, rf.max(rb));

        // Root of the gap, then radially out to the base circle when the root sits below it.
        out.push(at(rf, beta - h_root));
        if rf < rb {
            out.push(at(rb, beta - h_root));
        }
        // Up the leading flank. `h` shrinks with radius, so the angle grows: counter-clockwise.
        for &rho in &flank {
            out.push(at(rho, beta - half_tooth_angle(spec, rho)));
        }
        // Across the tip.
        let h_tip = half_tooth_angle(spec, ra);
        let tip_steps = 3;
        for k in 1..tip_steps {
            let t = k as f64 / tip_steps as f64;
            out.push(at(ra, beta - h_tip + 2.0 * h_tip * t));
        }
        // Down the trailing flank, mirrored.
        for &rho in flank.iter().rev() {
            out.push(at(rho, beta + half_tooth_angle(spec, rho)));
        }
        if rf < rb {
            out.push(at(rb, beta + h_root));
        }
        out.push(at(rf, beta + h_root));
        // Round the root to the next tooth.
        let gap_start = beta + h_root;
        let gap_end = beta + pitch - h_root;
        let gap_steps = 3;
        for k in 1..gap_steps {
            let t = k as f64 / gap_steps as f64;
            out.push(at(rf, gap_start + (gap_end - gap_start) * t));
        }
    }

    // Guarantee a floor on segment length. Even with even sampling, a nearly-pointed tooth or
    // a root that all but touches the base circle can put two points a few microns apart, and
    // the mesh kernel welds anything closer than ~2e-4 of the body diagonal — collapsing the
    // segment into a degenerate triangle and tearing the solid. Dropping the point outright is
    // the honest fix: at this spacing it is invisible, and the outline stays closed and simple.
    let min_gap = ra * 4e-3;
    let far_enough = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]) >= min_gap;
    let mut thinned: Vec<[f64; 2]> = Vec::with_capacity(out.len());
    for p in out {
        match thinned.last() {
            Some(prev) if !far_enough(*prev, p) => {}
            _ => thinned.push(p),
        }
    }
    // The wrap-around segment gets the same treatment; dropping the last point can only bring
    // it closer to a neighbour it already cleared, so one pass is enough.
    while thinned.len() > 3 && !far_enough(thinned[thinned.len() - 1], thinned[0]) {
        thinned.pop();
    }
    if thinned.len() < 3 {
        return None;
    }
    let out = thinned;

    // The bore, wound clockwise so it reads as a hole.
    let mut holes = Vec::new();
    if spec.bore > 1e-6 {
        let r = spec.bore * 0.5;
        if r >= rf * 0.95 {
            return None; // the bore would eat the teeth
        }
        let m = 64;
        holes.push(
            (0..m)
                .map(|k| {
                    let a = -std::f64::consts::TAU * k as f64 / m as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect(),
        );
    }
    Some((out, holes))
}

/// GT2 belt geometry, 2 mm pitch — the profile on every 3D-printer belt drive.
///
/// A GT2 pulley is not a gear: it drives a toothed belt, so its "teeth" are round grooves cut
/// *into* the rim rather than involute teeth standing proud of it, and only the 2 mm pitch has
/// to match — tooth counts are free. These are the published Gates numbers for the 2M profile.
pub mod gt2 {
    /// Belt pitch: the distance between adjacent belt teeth.
    pub const PITCH: f64 = 2.0;
    /// How deep each groove is cut below the outside diameter.
    pub const TOOTH_DEPTH: f64 = 0.764;
    /// Radius of the arc that forms the groove.
    pub const GROOVE_RADIUS: f64 = 0.555;
    /// Pitch line differential: the belt's pitch line sits this far outside the pulley's own
    /// outside diameter, so the OD is turned down by it on each side. Get this wrong and a
    /// printed pulley runs at the wrong ratio and the belt rides badly.
    pub const PLD: f64 = 0.254;

    /// Where the belt's pitch line runs — the diameter that actually sets the drive ratio.
    pub fn pitch_radius(teeth: u32) -> f64 {
        teeth as f64 * PITCH / (2.0 * std::f64::consts::PI)
    }

    /// The pulley's own outside radius, PLD inside the pitch radius.
    pub fn outside_radius(teeth: u32) -> f64 {
        pitch_radius(teeth) - PLD
    }

    /// The radius at the bottom of a groove.
    pub fn root_radius(teeth: u32) -> f64 {
        outside_radius(teeth) - TOOTH_DEPTH
    }
}

/// The outline of a GT2 timing-belt pulley, counter-clockwise, plus its bore as a hole.
///
/// Each groove is the arc of a circle of radius `GROOVE_RADIUS` whose centre sits one groove
/// radius above the root, tangent-blended into the flat land that runs along the outside
/// diameter between grooves. `steps` samples each groove arc.
pub fn gt2_profile(teeth: u32, bore: f64, steps: usize) -> Option<(Vec<[f64; 2]>, Vec<Vec<[f64; 2]>>)> {
    if teeth < 6 {
        return None; // below this the grooves overlap and there is no land left
    }
    let ro = gt2::outside_radius(teeth);
    let rr = gt2::root_radius(teeth);
    let rg = gt2::GROOVE_RADIUS;
    // Centre of the groove arc, measured out from the axis.
    let d = rr + rg;
    // Half-angle (about the axis) subtended by where the groove arc meets the OD circle.
    let cos_phi = ((ro * ro + d * d - rg * rg) / (2.0 * ro * d)).clamp(-1.0, 1.0);
    let phi = cos_phi.acos();
    let pitch_ang = std::f64::consts::TAU / teeth as f64;
    if 2.0 * phi >= pitch_ang {
        return None; // grooves would run into each other
    }
    // Angle, seen from the arc centre, of the point where the arc meets the OD.
    let meet = (ro * phi.sin()).atan2(ro * cos_phi - d);
    // Sweep the long way round, through the deepest point, so the arc cuts inward.
    let sweep = std::f64::consts::TAU - 2.0 * meet;

    let steps = steps.max(6);
    let land_steps = 3;
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(teeth as usize * (steps + land_steps + 2));
    for i in 0..teeth {
        let beta = pitch_ang * i as f64;
        // The land, along the outside diameter, from this groove's edge to the next one's.
        let (a0, a1) = (beta + phi, beta + pitch_ang - phi);
        for k in 0..=land_steps {
            let a = a0 + (a1 - a0) * k as f64 / land_steps as f64;
            out.push([ro * a.cos(), ro * a.sin()]);
        }
        // The groove for the NEXT tooth, centred on beta + pitch_ang.
        let gc = beta + pitch_ang;
        let (cx, cy) = (d * gc.cos(), d * gc.sin());
        for k in 1..steps {
            let t = k as f64 / steps as f64;
            let a = gc - meet - sweep * t;
            out.push([cx + rg * a.cos(), cy + rg * a.sin()]);
        }
    }

    // Same weld-tolerance floor the involute profile enforces, and for the same reason: a
    // segment below it collapses in the mesh kernel and tears the solid.
    let min_gap = ro * 4e-3;
    let far = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]) >= min_gap;
    let mut thinned: Vec<[f64; 2]> = Vec::with_capacity(out.len());
    for q in out {
        match thinned.last() {
            Some(prev) if !far(*prev, q) => {}
            _ => thinned.push(q),
        }
    }
    while thinned.len() > 3 && !far(thinned[thinned.len() - 1], thinned[0]) {
        thinned.pop();
    }
    if thinned.len() < 3 {
        return None;
    }

    let mut holes = Vec::new();
    if bore > 1e-6 {
        let r = bore * 0.5;
        if r >= rr * 0.95 {
            return None; // the bore would eat the grooves
        }
        let m = 64;
        holes.push(
            (0..m)
                .map(|k| {
                    let a = -std::f64::consts::TAU * k as f64 / m as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect(),
        );
    }
    Some((thinned, holes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(p: &[[f64; 2]]) -> f64 {
        let mut a = 0.0;
        for i in 0..p.len() {
            let (u, v) = (p[i], p[(i + 1) % p.len()]);
            a += u[0] * v[1] - v[0] * u[1];
        }
        a * 0.5
    }

    /// The standard sizes must come out of the standard formulas — these are the numbers a
    /// machinist would check the gear against.
    #[test]
    fn gear_sizes_follow_the_standard_formulas() {
        let g = GearSpec { teeth: 20, module: 2.0, pressure_angle: 20.0, bore: 6.0, backlash: 0.0 };
        assert!((g.pitch_radius() - 20.0).abs() < 1e-9, "pitch dia should be module x teeth = 40");
        assert!((g.tip_radius() - 22.0).abs() < 1e-9, "tip = pitch + one module");
        assert!((g.root_radius() - 17.5).abs() < 1e-9, "root = pitch - 1.25 modules");
        assert!((g.base_radius() - 20.0 * 20.0_f64.to_radians().cos()).abs() < 1e-9);

        // Meshing: a 20T and a 40T of the same module sit 60mm apart.
        let big = GearSpec { teeth: 40, ..g };
        assert!((g.centre_distance(&big) - 60.0).abs() < 1e-9);

        // The classic 20° undercut limit is 18 teeth.
        assert_eq!(g.undercut_limit(), 18);
        assert!(!g.is_undercut());
        assert!(GearSpec { teeth: 12, ..g }.is_undercut());
    }

    /// The profile must be a sane closed CCW loop with the right tooth count, and its area
    /// must sit between the root and tip circles — a self-crossing or bow-tie profile fails
    /// this even when it looks plausible plotted.
    #[test]
    /// The bug behind geartest.hcad: the flank sampler put its first two points ~5 um apart on
    /// a 44 mm gear. The mesh kernel welds anything closer than ~2e-4 of the body diagonal, so
    /// those pairs collapsed into degenerate triangles, the prism came out non-manifold, and
    /// every boolean stacked on top fell back to the lossy BSP path. The profile must never
    /// emit a segment the downstream welder would swallow.
    #[test]
    fn no_segment_is_short_enough_for_the_mesh_welder_to_swallow() {
        // Across the range of gears the panel can produce, not just the default.
        for teeth in [8_u32, 12, 20, 37, 60, 120] {
            for module in [0.5_f64, 1.0, 2.0, 6.0] {
                for pa in [14.5_f64, 20.0, 25.0] {
                    let g = GearSpec { teeth, module, pressure_angle: pa, bore: module * 2.0, backlash: 0.0 };
                    let Some((outer, holes)) = gear_profile(&g, 24) else { continue };
                    // What the mesh kernel would weld: 2e-4 of the bounding diagonal.
                    let weld = 2.0 * g.tip_radius() * std::f64::consts::SQRT_2 * 2.0e-4;
                    for loop_ in std::iter::once(&outer).chain(holes.iter()) {
                        let n = loop_.len();
                        for k in 0..n {
                            let (a, b) = (loop_[k], loop_[(k + 1) % n]);
                            let d = (b[0] - a[0]).hypot(b[1] - a[1]);
                            assert!(
                                d > weld * 2.0,
                                "{teeth}T m{module} PA{pa}: segment {k} is {d:.3e} mm, at or under the {weld:.3e} mm weld tolerance"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Thinning must not quietly reshape the gear. The area has to stay where the standard
    /// numbers put it, and each tooth must still reach the tip circle exactly once.
    #[test]
    fn thinning_short_segments_leaves_the_gear_intact() {
        let g = GearSpec::default();
        let (outer, _) = gear_profile(&g, 24).expect("profile");
        let (rf, ra) = (g.root_radius(), g.tip_radius());
        let a = area(&outer);
        let (lo, hi) = (std::f64::consts::PI * rf * rf, std::f64::consts::PI * ra * ra);
        assert!(a > lo && a < hi, "area {a:.1} left the root..tip band {lo:.1}..{hi:.1}");
        let mut runs = 0;
        let mut inside = false;
        for p in outer.iter().chain(outer.first()) {
            let r = p[0].hypot(p[1]);
            let near_tip = r > ra - (ra - rf) * 0.02;
            if near_tip && !inside {
                runs += 1;
            }
            inside = near_tip;
        }
        assert_eq!(runs, g.teeth, "thinning lost or merged teeth: {runs} tip runs");
    }

    fn the_profile_is_a_clean_closed_loop() {
        let g = GearSpec::default();
        let (outer, holes) = gear_profile(&g, 12).expect("profile");
        assert!(outer.len() > 100, "too coarse: {} points", outer.len());

        // Counter-clockwise, as the extruder expects.
        let a = area(&outer);
        assert!(a > 0.0, "profile wound clockwise (area {a:.2})");

        // Between the root and tip circles by area.
        let (rf, ra) = (g.root_radius(), g.tip_radius());
        let lo = std::f64::consts::PI * rf * rf;
        let hi = std::f64::consts::PI * ra * ra;
        assert!(a > lo && a < hi, "area {a:.1} outside the root..tip band {lo:.1}..{hi:.1}");

        // Every point lies in that band too.
        for p in &outer {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(r >= rf - 1e-6 && r <= ra + 1e-6, "point at r={r:.3} outside {rf:.3}..{ra:.3}");
        }

        // Exactly `teeth` points reach the tip circle's neighbourhood, one run per tooth.
        let mut runs = 0;
        let mut inside = false;
        for p in outer.iter().chain(outer.first()) {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            let tip = r > ra - 1e-6;
            if tip && !inside {
                runs += 1;
            }
            inside = tip;
        }
        assert_eq!(runs, g.teeth as usize, "found {runs} tips, expected {} teeth", g.teeth);

        // The bore is a clockwise hole of the right size.
        assert_eq!(holes.len(), 1);
        assert!(area(&holes[0]) < 0.0, "the bore should wind clockwise");
        for p in &holes[0] {
            assert!(((p[0] * p[0] + p[1] * p[1]).sqrt() - 3.0).abs() < 1e-6, "bore radius wrong");
        }
    }

    /// The flanks must actually be involutes: the pressure angle measured off the geometry
    /// has to come back as the one asked for. This is what separates a real gear from a
    /// tooth-shaped decoration.
    #[test]
    fn the_flanks_are_true_involutes() {
        let g = GearSpec { teeth: 30, module: 1.5, pressure_angle: 20.0, bore: 0.0, backlash: 0.0 };
        let rb = g.base_radius();
        // Sample the flank angle at two radii and recover the base circle from them.
        for &rho in &[g.pitch_radius(), g.pitch_radius() * 1.03] {
            let h = half_tooth_angle(&g, rho);
            // Invert: h = psi_p + inv(alpha) - inv(a_rho)  =>  a_rho, and rb = rho cos(a_rho).
            let psi_p = std::f64::consts::PI / (2.0 * g.teeth as f64);
            let target = psi_p + inv(g.pressure_angle.to_radians()) - h;
            // Solve inv(a) = target for a.
            let (mut lo, mut hi) = (0.0_f64, 1.4_f64);
            for _ in 0..80 {
                let mid = (lo + hi) * 0.5;
                if inv(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let recovered = rho * lo.cos();
            assert!((recovered - rb).abs() < 1e-6, "flank at r={rho:.2} implies base {recovered:.4}, wanted {rb:.4}");
        }
        // And the tooth is thinner at the tip than the root — the involute's signature.
        assert!(half_tooth_angle(&g, g.tip_radius()) < half_tooth_angle(&g, g.root_radius().max(rb)));
    }

    /// A tooth count so low the flanks would meet before the tip must pull the tip in rather
    /// than emit a self-crossing bow tie.
    #[test]
    fn a_pointed_tooth_is_trimmed_not_crossed() {
        let g = GearSpec { teeth: 5, module: 3.0, pressure_angle: 20.0, bore: 0.0, backlash: 0.0 };
        let (outer, _) = gear_profile(&g, 10).expect("even a 5-tooth gear should build");
        assert!(area(&outer) > 0.0, "the trimmed profile wound the wrong way");
        // No point may exceed the nominal tip radius, and the loop stays simple: consecutive
        // points never jump more than a tooth's worth of arc.
        let ra = g.tip_radius();
        for p in &outer {
            assert!((p[0] * p[0] + p[1] * p[1]).sqrt() <= ra + 1e-6);
        }
        let max_step = std::f64::consts::TAU / g.teeth as f64 * ra * 1.5;
        for w in outer.windows(2) {
            let d = ((w[1][0] - w[0][0]).powi(2) + (w[1][1] - w[0][1]).powi(2)).sqrt();
            assert!(d < max_step, "a {d:.2}mm jump suggests the profile crossed itself");
        }
    }

    /// Backlash thins the teeth without moving the pitch circle, so a printed pair still
    /// meshes at the nominal centre distance.
    #[test]
    fn backlash_thins_the_teeth_only() {
        let tight = GearSpec { backlash: 0.0, ..GearSpec::default() };
        let loose = GearSpec { backlash: 0.2, ..tight };
        assert!((tight.pitch_radius() - loose.pitch_radius()).abs() < 1e-12, "backlash moved the pitch circle");
        let a1 = area(&gear_profile(&tight, 12).unwrap().0);
        let a2 = area(&gear_profile(&loose, 12).unwrap().0);
        assert!(a2 < a1, "backlash should remove material ({a2:.2} vs {a1:.2})");
        assert!(a2 > a1 * 0.9, "backlash removed far too much ({a2:.2} vs {a1:.2})");
    }

    /// Nonsense specs decline rather than produce junk.
    #[test]
    fn impossible_gears_are_refused() {
        let g = GearSpec::default();
        assert!(gear_profile(&GearSpec { teeth: 2, ..g }, 12).is_none(), "2 teeth isn't a gear");
        assert!(gear_profile(&GearSpec { module: 0.0, ..g }, 12).is_none());
        assert!(gear_profile(&GearSpec { pressure_angle: 0.0, ..g }, 12).is_none());
        // A bore that would swallow the teeth.
        assert!(gear_profile(&GearSpec { bore: 100.0, ..g }, 12).is_none());
    }
}
