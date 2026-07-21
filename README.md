<div align="center">

<img src="crates/hworks-app/assets/logo.png" width="140" alt="HCAD logo">

# HCAD

**A SolidWorks-style parametric CAD modeler, written in Rust.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-informational)
![Built with Rust](https://img.shields.io/badge/built%20with-Rust-orange)

</div>

---

HCAD is a desktop CAD application for **feature-based parametric solid modeling** — the
same workflow you'd recognize from SolidWorks, Fusion 360, or FreeCAD: sketch in 2D, turn
sketches into 3D solids, combine parts into assemblies, and edit any earlier step to have
the whole model rebuild itself.

## Download

Grab the latest build from the [**Releases**](../../releases) page:

| Platform | File | Notes |
|----------|------|-------|
| **Windows** | `hcad-vX.Y.Z-windows-x86_64.exe` | Installer — bundles the VC++ runtime, registers `.hcad`/`.hasm` file types, and creates shortcuts. |
| **Linux** | `hcad-vX.Y.Z-linux-x86_64.AppImage` | Self-contained; `chmod +x` and run. Needs a working **Vulkan** driver and **xdg-desktop-portal** (standard on modern desktops). x86_64, glibc 2.39+ (Ubuntu 24.04+ / Fedora 40+). |

No account, telemetry, or internet connection required — HCAD runs fully offline.

---

## Screenshots

| Sketching | Boss extrude |
|---|---|
| ![Sketching a profile on a face with relations and dimensions](docs/screenshots/sketch.jpg) | ![Boss-extrude PropertyManager with direction gizmo and live preview](docs/screenshots/boss-extrude.jpg) |
| *Sketch mode: a profile on a face with inference, geometric relations, and driving dimensions.* | *Boss extrude: PropertyManager, draggable direction gizmo, and live preview.* |

| Cut + feature tree | Solid body |
|---|---|
| ![A square cut through a cylinder with the parametric feature tree](docs/screenshots/cut.jpg) | ![Finished cylinder with feature tree and edge selection highlight](docs/screenshots/cylinder.jpg) |
| *Cut feature through a boss, with the editable feature tree and the selected feature's profile highlighted.* | *A finished solid — feature tree, view triad, and edge-selection highlight.* |

---

## Capabilities

### 2D Sketching
- Tools: line, circle, rectangle/square, arc, point, and **construction geometry**.
- **Geometric constraints:** coincident, horizontal, vertical, parallel, perpendicular,
  equal, midpoint, tangent, concentric — all solved together so the sketch stays
  parametric, not just drawn.
- **Smart dimensions:** type-as-you-draw live measurements, click-two-points dimensioning,
  double-click to edit, and edge-to-edge dimensions (drive a circle's size off an existing
  bore rim with no radius math).
- **Inference & snapping:** midpoint, centre, and quadrant snaps; hover a round edge to
  reveal and snap to its centre so new features stay concentric with existing bores.

### 3D Part Features
- **Extrude** (boss), **Cut**, **Revolve**, **Sweep / Sweep Cut** (profile along a path).
- **Fillet**, **Chamfer**, **Shell** (hollow to a wall thickness), **Draft/bevel**.
- **Loft** between profiles, **Thin feature**, **Mirror**, and **Text-as-feature**.
- **Patterns:** linear and circular (rows and bolt circles from one modelled feature).
- **Hole Genie:** tapped/threaded holes with standard pitch tables (metric coarse/fine,
  imperial UNC/UNF), depth quick-sets, and tap-drill hints.
- **Sketch on 3D faces:** select a planar face of a solid and start a new sketch on it.

### Parametric Feature Tree
- Every step lives in an **editable feature tree**. Roll back to any earlier feature,
  change a dimension or sketch, and the model **regenerates** downstream.
- Stable topological naming so references survive edits, 64-level undo/redo, and
  save/load of `.hcad` documents.

### Assemblies (`.hasm`)
- Insert saved parts as components with **move/rotate gizmos** and face snapping.
- **Mates:** coincident, distance, concentric, parallel — flat faces pull flush, round
  faces align bolt-into-hole. Mates re-solve on drag and survive part edits.
- **Edit parts in-context** while neighbours ghost around you; **interference check**
  (overlaps glow red with per-pair volumes), **exploded view**, and **BOM** export.

### Reverse Engineering / Scans
- Import an **STL** as an editable body or a reference scan; 3-point alignment, automatic
  mesh repair, and measure directly on scans.
- **Trace Section** fitted curves from a scan, **Solidify** a scan shell into a solid, and
  **fit primitives** (datum planes / axes) from a scan region.

### Reference Geometry & Views
- Default Front/Top/Right planes plus user offset, angled, free, and 3-point planes; axes.
- **Section view** with a draggable cutting-plane gizmo, **orthographic** camera by default
  (true CAD views) with a perspective toggle, and a wireframe overlay.

### Interface
- **CommandManager tabs** (SolidWorks-style), selectable **mouse schemes**
  (HCAD / Blender / SolidWorks), persisted settings, and `.hcad` / `.hasm` file associations.

---

## Architecture (short version)

HCAD is built in four decoupled layers. The full write-up — including the
feature-tree/regeneration model and the topological-naming strategy — is in
[`DESIGN.md`](DESIGN.md).

| Layer | Responsibility | Built on |
|-------|----------------|----------|
| Geometry kernel | B-rep solids, extrude (sweep), boolean cut, tessellation | [`truck`](https://github.com/ricosjp/truck) (exact B-rep, pure Rust) + [`Manifold`](https://github.com/elalish/manifold) (robust mesh booleans, C++) |
| Sketcher + solver | 2D entities, constraints, Newton/least-squares solve | our code (nalgebra) |
| Document / feature tree | parametric-history DAG, regeneration — **the source of truth** | our code |
| Viewport / UI | 3D rendering, camera, face/edge picking, gizmos, panels | [`Bevy`](https://bevyengine.org/) + `bevy_egui` |

The kernel, sketcher, and document crates are **Bevy-free and testable headless**; the
Bevy app is a thin presentation shell on top.

---

## Building from source

HCAD's geometry kernel uses [`Manifold`](https://github.com/elalish/manifold) (a C++
library) for robust mesh booleans, so building from source needs a **C++ toolchain and
CMake** in addition to Rust:

- **Rust** (stable; MSVC toolchain on Windows).
- **CMake** on `PATH` — e.g. `winget install Kitware.CMake`.
- **A C++ compiler** — on Windows, the MSVC "Desktop development with C++" workload.
  Manifold is compiled once and **statically linked** into the `hcad` binary.

> These are **build-time only**. The shipped binaries are self-contained — end users who
> run the installer or AppImage need none of this.

```sh
cargo run -p hworks-app          # run the hcad binary (debug)
cargo build --release            # optimized build → target/release/hcad
cargo test                       # headless tests (geometry, sketch, document)
```

### Flickering on a laptop with hybrid graphics

By default HCAD renders on the **discrete** GPU. On some hybrid laptops that causes
flicker (each frame is copied to the integrated GPU for display). Force the integrated
GPU, which is wired straight to the screen:

```sh
HCAD_GPU=integrated   # no cross-GPU flicker
HCAD_GPU=discrete     # default
```

---

## License

HCAD is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE)
file for full terms.

Copyright (C) 2026 Tibiaworx. You may use, modify, and distribute this software (including
commercially), but any distributed derivative work must also be released as open source
under the GPL.
