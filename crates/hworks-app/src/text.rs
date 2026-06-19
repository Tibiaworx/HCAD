//! System-font discovery and glyph-outline extraction for the sketch Text tool.
//!
//! We avoid heavyweight font stacks (font-kit pulls in a C FreeType build): instead we
//! scan the platform font directory once, read each face's family/style from its `name`
//! and header tables with the pure-Rust `ttf-parser`, and extract glyph outlines on
//! demand. Outlines come back as closed polylines in *normalized EM space* — baseline at
//! y = 0, cap height ≈ 1 unit, x advancing right — so the sketch entity can scale/rotate
//! /warp them freely.

use std::path::PathBuf;
use std::sync::OnceLock;

/// One font face found on disk.
struct FaceEntry {
    family: String,
    path: PathBuf,
    index: u32,
    bold: bool,
    italic: bool,
}

static REGISTRY: OnceLock<Vec<FaceEntry>> = OnceLock::new();

/// Platform font directories to scan.
fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(windir) = std::env::var("WINDIR") {
        dirs.push(PathBuf::from(windir).join("Fonts"));
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        dirs.push(PathBuf::from(local).join("Microsoft").join("Windows").join("Fonts"));
    }
    dirs
}

fn registry() -> &'static Vec<FaceEntry> {
    REGISTRY.get_or_init(|| {
        let mut out = Vec::new();
        for dir in font_dirs() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for ent in rd.flatten() {
                let path = ent.path();
                let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                if !matches!(ext.as_deref(), Some("ttf" | "otf" | "ttc")) {
                    continue;
                }
                let Ok(data) = std::fs::read(&path) else { continue };
                let faces = ttf_parser::fonts_in_collection(&data).unwrap_or(1);
                for index in 0..faces {
                    let Ok(face) = ttf_parser::Face::parse(&data, index) else { continue };
                    let Some(family) = face_family(&face) else { continue };
                    out.push(FaceEntry {
                        family,
                        path: path.clone(),
                        index,
                        bold: face.is_bold(),
                        italic: face.is_italic() || face.is_oblique(),
                    });
                }
            }
        }
        out.sort_by(|a, b| a.family.to_lowercase().cmp(&b.family.to_lowercase()));
        out
    })
}

/// The English family name (prefer the typographic family, name id 16, then id 1).
fn face_family(face: &ttf_parser::Face) -> Option<String> {
    let mut id1 = None;
    for name in face.names() {
        // `to_string` returns Some only for encodings it can decode (Unicode / Mac Roman).
        let Some(s) = name.to_string() else { continue };
        match name.name_id {
            16 => return Some(s), // typographic family — best
            1 if id1.is_none() => id1 = Some(s),
            _ => {}
        }
    }
    id1
}

/// Distinct family names available, sorted, ready for a font picker.
pub fn families() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for e in registry() {
        if names.last().map(|n| n.eq_ignore_ascii_case(&e.family)) != Some(true)
            && !names.iter().any(|n| n.eq_ignore_ascii_case(&e.family))
        {
            names.push(e.family.clone());
        }
    }
    names
}

/// A sensible default family present on the system (Arial / Segoe UI / first available).
pub fn default_family() -> String {
    let fams = families();
    for want in ["Arial", "Segoe UI", "Calibri", "Tahoma", "Verdana"] {
        if let Some(f) = fams.iter().find(|f| f.eq_ignore_ascii_case(want)) {
            return f.clone();
        }
    }
    fams.into_iter().next().unwrap_or_else(|| "Arial".to_string())
}

/// One representative (regular) face per family with its file bytes + face index, for
/// registering live previews with egui. Files larger than `max_bytes` are skipped (big
/// CJK fonts would balloon memory and aren't needed to preview a Latin family name).
pub fn family_preview_data(max_bytes: u64) -> Vec<(String, Vec<u8>, u32)> {
    let mut out = Vec::new();
    for fam in families() {
        let Some(e) = best_face(&fam, false, false) else { continue };
        if std::fs::metadata(&e.path).map(|m| m.len()).unwrap_or(u64::MAX) > max_bytes {
            continue;
        }
        let Ok(bytes) = std::fs::read(&e.path) else { continue };
        // egui (epaint) parses every registered font eagerly with ab_glyph and *panics*
        // on any it can't read (bitmap fonts, some .ttc faces). ttf-parser is more
        // lenient, so gate registration on ab_glyph actually accepting the face.
        if ab_glyph::FontVec::try_from_vec_and_index(bytes.clone(), e.index).is_err() {
            continue;
        }
        out.push((fam, bytes, e.index));
    }
    out
}

/// Pick the face of `family` that best matches the requested bold/italic.
fn best_face(family: &str, bold: bool, italic: bool) -> Option<&'static FaceEntry> {
    let reg = registry();
    let mut best: Option<(&FaceEntry, i32)> = None;
    for e in reg {
        if !e.family.eq_ignore_ascii_case(family) {
            continue;
        }
        // Lower score = better match.
        let score = (e.bold != bold) as i32 + (e.italic != italic) as i32;
        if best.map_or(true, |(_, bs)| score < bs) {
            best = Some((e, score));
        }
    }
    best.map(|(e, _)| e)
}

/// Tessellate a quadratic Bézier (from `p0` via control `c` to `p1`) into points,
/// excluding `p0` (already emitted) and including `p1`.
fn quad(out: &mut Vec<[f64; 2]>, p0: [f64; 2], c: [f64; 2], p1: [f64; 2]) {
    const N: usize = 8;
    for i in 1..=N {
        let t = i as f64 / N as f64;
        let u = 1.0 - t;
        let x = u * u * p0[0] + 2.0 * u * t * c[0] + t * t * p1[0];
        let y = u * u * p0[1] + 2.0 * u * t * c[1] + t * t * p1[1];
        out.push([x, y]);
    }
}

/// Tessellate a cubic Bézier into points (excluding `p0`, including `p1`).
fn cubic(out: &mut Vec<[f64; 2]>, p0: [f64; 2], c1: [f64; 2], c2: [f64; 2], p1: [f64; 2]) {
    const N: usize = 12;
    for i in 1..=N {
        let t = i as f64 / N as f64;
        let u = 1.0 - t;
        let x = u * u * u * p0[0] + 3.0 * u * u * t * c1[0] + 3.0 * u * t * t * c2[0] + t * t * t * p1[0];
        let y = u * u * u * p0[1] + 3.0 * u * u * t * c1[1] + 3.0 * u * t * t * c2[1] + t * t * t * p1[1];
        out.push([x, y]);
    }
}

/// Collects one glyph's outline into closed contours, applying a uniform `scale`
/// (1/units_per_em), an italic `shear`, and a pen `offset` (all in normalized EM space).
struct Outliner {
    contours: Vec<Vec<[f64; 2]>>,
    cur: Vec<[f64; 2]>,
    last: [f64; 2],
    scale: f64,
    shear: f64,
    off_x: f64,
}

impl Outliner {
    fn pt(&self, x: f32, y: f32) -> [f64; 2] {
        let ny = y as f64 * self.scale;
        let nx = x as f64 * self.scale + self.shear * ny + self.off_x;
        [nx, ny]
    }
    fn flush(&mut self) {
        if self.cur.len() >= 3 {
            self.contours.push(std::mem::take(&mut self.cur));
        } else {
            self.cur.clear();
        }
    }
}

impl ttf_parser::OutlineBuilder for Outliner {
    fn move_to(&mut self, x: f32, y: f32) {
        self.flush();
        let p = self.pt(x, y);
        self.last = p;
        self.cur.push(p);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.last = p;
        self.cur.push(p);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let (p0, c, p1) = (self.last, self.pt(x1, y1), self.pt(x, y));
        quad(&mut self.cur, p0, c, p1);
        self.last = p1;
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let (p0, c1, c2, p1) = (self.last, self.pt(x1, y1), self.pt(x2, y2), self.pt(x, y));
        cubic(&mut self.cur, p0, c1, c2, p1);
        self.last = p1;
    }
    fn close(&mut self) {
        self.flush();
    }
}

/// Bake `text` into closed glyph-outline contours in normalized EM space using the best
/// matching face of `family`. `spacing` is extra inter-glyph advance in EM units (so 0.1
/// ≈ a tenth of the height). Returns an empty vec if the family/text yields no outlines.
pub fn glyph_contours(family: &str, bold: bool, italic: bool, text: &str, spacing: f64) -> Vec<Vec<[f64; 2]>> {
    let Some(entry) = best_face(family, bold, italic) else { return Vec::new() };
    let Ok(data) = std::fs::read(&entry.path) else { return Vec::new() };
    let Ok(face) = ttf_parser::Face::parse(&data, entry.index) else { return Vec::new() };
    let units = face.units_per_em() as f64;
    if units <= 0.0 {
        return Vec::new();
    }
    let scale = 1.0 / units;
    // Synthetic italic only when the chosen face isn't already italic.
    let shear = if italic && !entry.italic { 0.22 } else { 0.0 };

    let mut all: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut pen_x = 0.0_f64; // normalized
    for ch in text.chars() {
        if ch == '\n' {
            continue;
        }
        let gid = face.glyph_index(ch);
        if let Some(gid) = gid {
            let mut outliner = Outliner {
                contours: Vec::new(),
                cur: Vec::new(),
                last: [0.0, 0.0],
                scale,
                shear,
                off_x: pen_x,
            };
            face.outline_glyph(gid, &mut outliner);
            outliner.flush();
            all.append(&mut outliner.contours);
            let adv = face.glyph_hor_advance(gid).unwrap_or(0) as f64 * scale;
            pen_x += adv + spacing;
        } else {
            pen_x += 0.5 + spacing; // unknown glyph: advance a reasonable gap
        }
    }
    all
}
