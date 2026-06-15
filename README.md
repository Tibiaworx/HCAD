# HCAD

**A SolidWorks-style parametric CAD modeler, written in Rust.**

HCAD is an in-development desktop CAD application for **feature-based parametric solid
modeling** — the same workflow you'd recognize from SolidWorks, Fusion 360, or FreeCAD:
sketch in 2D, turn sketches into 3D solids, and edit any earlier step to have the whole
model rebuild itself.

> ⚠️ **Status: early / pre-alpha.** This repository currently contains the architecture
> and design (see [`DESIGN.md`](DESIGN.md)). The application code is being built
> incrementally against the milestone roadmap below. There is **no runnable build yet.**

---

## What it does (the intended workflow)

1. **Reference planes.** Start with three datum planes (Front / Top / Right) and create
   your own offset or angled planes.
2. **2D sketching.** Open a plane and draw with basic tools — lines, **construction
   lines (with midpoints)**, circles, rectangles/squares, arcs, and points.
3. **Constraints & dimensions.** Apply geometric relations (coincident, horizontal,
   vertical, parallel, perpendicular, equal, midpoint, tangent, concentric) and driving
   dimensions. A constraint solver positions the geometry to satisfy all of them at once,
   so the sketch is *parametric*, not just drawn.
4. **Features (2D → 3D).** Turn closed sketch profiles into solids:
   - **Extrude** (boss / add material)
   - **Cut** (subtractive extrude / remove material)
   - *(planned: Revolve, Fillet, Chamfer, linear/circular Pattern)*
5. **Sketch on 3D faces.** Select a planar face of an existing solid and start a new
   sketch on it, then extrude or cut again.
6. **Editable history.** Every step lives in a **feature tree**. Roll back to any earlier
   feature, change a dimension or sketch, and the model **regenerates** downstream — this
   parametric history is the core of the application.

---

## Architecture (short version)

HCAD is built in four decoupled layers. The full write-up — including the
feature-tree/regeneration model and the topological-naming strategy — is in
[`DESIGN.md`](DESIGN.md).

| Layer | Responsibility | Built on |
|-------|----------------|----------|
| Geometry kernel | B-rep solids, extrude (sweep), boolean cut, tessellation | [`truck`](https://github.com/ricosjp/truck) (pure Rust), behind a swappable `GeometryKernel` trait |
| Sketcher + solver | 2D entities, constraints, Newton/least-squares solve | our code (nalgebra) |
| Document / feature tree | parametric-history DAG, regeneration — **the source of truth** | our code |
| Viewport / UI | 3D rendering, camera, face/edge picking, gizmos, panels | [`Bevy`](https://bevyengine.org/) + `bevy_egui` |

The kernel, sketcher, and document crates are **Bevy-free and testable headless**; the
Bevy app is a thin presentation shell on top.

### Planned workspace layout
```
hworks-geometry/   # kernel trait + truck impl, tessellation
hworks-sketch/     # sketch entities + constraint solver
hworks-document/   # feature tree, regeneration, save/load
hworks-app/        # Bevy viewport + egui UI
```

---

## Roadmap

Each milestone is independently runnable.

| Milestone | Deliverable |
|-----------|-------------|
| **M0** | Bevy window, orbit camera, render the three reference planes |
| **M1** | Pick a plane → flat (unconstrained) sketch mode |
| **M2** | Constraint solver v1, rectangle tool, construction lines, drag-to-resolve |
| **M3** | First solid — extrude a sketch profile |
| **M4** | Extrude-cut (boolean) + feature-tree panel |
| **M5** | Sketch on a solid face (stable topological IDs introduced) |
| **M6** | Edit an earlier feature → regenerate downstream; rollback bar |
| **M7** | Save / load documents |
| **M8+** | Revolve, Fillet, Chamfer, Pattern, more constraints, dimensions display |

---

## Building & running

```sh
cargo run -p hworks-app          # or: cargo build --release
```

Prebuilt Windows binaries are on the **Releases** page.

### Flickering on a laptop with hybrid graphics

Many laptops have two GPUs — an integrated one (Intel/AMD) wired to the display, and a
discrete one (NVIDIA/AMD) for performance. By default HCAD renders on the **discrete**
GPU, which is stable. On some hybrid laptops this causes flickering, because each frame
is copied from the discrete GPU to the integrated one for display.

The flicker-free fix is to render on the **integrated** GPU (it's connected straight to
the screen). Set the `HCAD_GPU` environment variable:

```sh
HCAD_GPU=integrated   # render on the integrated GPU (no cross-GPU flicker)
HCAD_GPU=discrete     # render on the discrete GPU (default)
```

> ⚠️ If the integrated GPU crashes with a *"device lost / suspended"* error, its driver
> is likely out of date — update the integrated GPU's driver (AMD Adrenalin / Intel
> Graphics) and try again. A CAD viewport runs fine on integrated graphics.

---

## License

TBD.
