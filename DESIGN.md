# Hworks — A Parametric CAD Application in Rust

> A SolidWorks-style parametric solid modeler. Reference planes, a constraint-based
> 2D sketcher, feature-based 3D modeling (extrude / cut), sketching on 3D faces, and a
> fully editable feature history. Built as an ongoing, incremental project.

---

## 1. Vision & Scope

Hworks is a **feature-based parametric CAD modeler**. The user:

1. Starts with three default reference **planes** (Front / Top / Right).
2. Opens a plane and draws a **2D sketch** — lines, construction lines (with
   midpoints), circles, rectangles — and applies **constraints** and **dimensions**.
3. Turns sketches into solids with **features**: extrude (boss/add) and
   cut (subtractive extrude).
4. Selects a **face of the resulting 3D solid** to start a new sketch on, and repeats.
5. Goes back to **any earlier feature**, edits it, and the whole model **regenerates**.

The defining property is **parametric history**: the model is not the final geometry,
it is the *recipe* that produces the geometry. Editing a step re-runs the recipe.

### In scope (eventually)
- Reference planes (default + user-created offset/angled planes).
- 2D sketcher: line, construction line, midpoint, circle, rectangle/square,
  arc, point. Trim/extend later.
- Geometric constraints: coincident, horizontal, vertical, parallel, perpendicular,
  equal, midpoint, tangent, concentric. Dimensional constraints: distance, radius,
  diameter, angle.
- Features: Extrude (boss), Extrude-Cut, plus later Revolve, Fillet, Chamfer, Pattern.
- Sketch on a planar face of an existing solid.
- Feature tree with rollback bar; edit-and-regenerate.
- Save / load a document.

### Out of scope (for now)
- Assemblies (multiple parts mated together).
- Drawings/drafting sheets (2D engineering drawings with dimensions/title blocks).
- Surfacing, sheet metal, simulation, CAM.
- STEP/IGES import-export (nice later; `truck` has some glTF/obj, OCCT would add STEP).

---

## 2. The Four Subsystems

A CAD app is four layers stacked. Keeping them decoupled is the single most important
architectural discipline in this project.

```
┌───────────────────────────────────────────────────────────┐
│  4. UI / Viewport          Bevy + bevy_egui                 │
│     3D scene, camera, picking, gizmos, feature-tree panel   │
├───────────────────────────────────────────────────────────┤
│  3. Document / Feature Tree   (our own code — the "brain")  │
│     parametric history DAG, regeneration, references        │
├───────────────────────────────────────────────────────────┤
│  2. Sketcher + Constraint Solver  (our own code)            │
│     2D entities, constraints, Newton/least-squares solve    │
├───────────────────────────────────────────────────────────┤
│  1. Geometry Kernel           truck                         │
│     B-rep solids, extrude (sweep), boolean cut, tessellate  │
└───────────────────────────────────────────────────────────┘
```

**Rule:** Layer 3 (the Document) is the source of truth. Bevy entities in Layer 4 are
*derived* render artifacts — never the authoritative model. We can throw away and
rebuild every Bevy mesh on regeneration; the Document persists.

---

## 3. Technology Stack

| Concern              | Choice                          | Why |
|----------------------|---------------------------------|-----|
| Geometry kernel      | **truck** 0.6.x (`truck-modeling`, `truck-shapeops`, `truck-meshalgo`) | Pure Rust → trivial Windows builds, fully debuggable; B-rep + sweep (extrude) + booleans (cut). |
| App / rendering      | **Bevy** 0.18.x                 | ECS engine: 3D renderer, scene hierarchy, built-in ray **picking** (click a face), **gizmos**, asset/mesh pipeline. |
| UI panels            | **bevy_egui**                   | Immediate-mode panels for the feature tree, dimension dialogs, toolbar. |
| Camera               | **bevy_panorbit_camera** (or hand-rolled orbit) | Orbit/pan/zoom around the part. |
| Math                 | **cgmath** (truck's native) + Bevy's **glam** | truck uses cgmath; convert at the boundary. Keep a `convert.rs`. |
| Constraint solver    | Hand-rolled (nalgebra/Newton) → optionally `slvs` (SolveSpace) later | Start small, swap up when needed. |
| Serialization        | **serde** + **ron** (human-readable) | Save/load the Document. |
| Error handling       | **thiserror** (libs) / **anyhow** (app) | |
| Logging              | **tracing** (Bevy uses it)      | |

> **Version pinning note:** Bevy and truck both evolve fast. At scaffold time we pin
> exact versions in the workspace and only bump deliberately. Bevy 0.18 folded picking
> and gizmos into core, which is why it's our floor.

### Kernel escape hatch
If `truck`'s booleans/fillets prove too fragile down the line, the kernel lives behind
**our own trait (`GeometryKernel`)** so we can swap in `opencascade-rs` (OCCT bindings)
without touching Layers 2–4. Designing this seam now is cheap; retrofitting it later is
not. See §7.

---

## 4. The Feature Tree — Parametric History (the core)

This is the heart of "go back and change geometry." It is **our** code, not a library.

### 4.1 Model
The document is an ordered list of **Features**, each a node in a dependency DAG:

```rust
struct Document {
    features: Vec<Feature>,   // timeline order
    rollback: usize,          // rollback-bar position; features[rollback..] are "rolled back"
    params: ParamTable,       // named dimensions/variables (enables a future equation editor)
}

enum FeatureKind {
    Plane(PlaneDef),          // datum plane (default, offset, or on-face)
    Sketch(Sketch),           // 2D sketch bound to a plane or face
    Extrude { sketch: FeatureId, distance: Param, direction: Dir },
    Cut     { sketch: FeatureId, distance: Param, direction: Dir },
    // later: Revolve, Fillet, Chamfer, Pattern, ...
}

struct Feature {
    id: FeatureId,            // stable, never reused
    kind: FeatureKind,
    inputs: Vec<Ref>,         // what this feature consumes (sketch ids, face refs, ...)
    cached: Option<KernelSolid>, // last computed B-rep result (dirty-tracked)
    dirty: bool,
}
```

### 4.2 Regeneration
When the user edits feature *k* (or moves the rollback bar):

1. Mark *k* and every downstream dependent **dirty**.
2. Topologically walk the DAG from the first dirty node.
3. For each dirty feature, call the kernel to recompute its `KernelSolid`, threading
   the previous feature's solid in as the "base" for Extrude/Cut.
4. Re-tessellate changed solids and rebuild their Bevy meshes.

Clean features reuse their cached solid → editing late features is cheap.

### 4.3 The Topological Naming Problem (READ THIS)
The deepest hazard in parametric CAD. When a sketch is created **on face #3** of a
solid and an *earlier* feature is edited, the kernel regenerates the solid and the face
that *was* "#3" may now be a different index — or may have split, merged, or vanished.
Naively storing "face index 3" gives the classic FreeCAD "broken model after edit" bug.

**Our mitigation strategy (incremental):**
- Phase 1: tolerate it. References are simple; editing upstream may break downstream and
  we tell the user. Acceptable while learning.
- Phase 2: assign **stable persistent IDs** to topological entities (faces/edges/verts)
  and a **matching heuristic** after regeneration — match new faces to old by geometry
  (plane normal + centroid + adjacency signature), carrying the stable ID forward.
- Phase 3: richer "selection by feature lineage" (which feature created this face) à la
  modern kernels.

We bake stable-ID slots into the topology wrapper from the start (cheap), and grow the
matching logic over time.

---

## 5. The Sketcher + Constraint Solver

### 5.1 Sketch model
A sketch is a 2D problem living in the plane's local UV coordinates:

```rust
struct Sketch {
    plane: Ref,                       // owning plane or face
    points: SlotMap<PointId, Point2>, // every entity endpoint is a solved variable
    entities: Vec<SketchEntity>,
    constraints: Vec<Constraint>,
}

enum SketchEntity {
    Line { a: PointId, b: PointId, construction: bool },
    Circle { center: PointId, radius: f64 },
    Arc { center: PointId, start: PointId, end: PointId },
    Point(PointId),
}
// A "square/rectangle" = 4 Line entities + perpendicular/equal/coincident constraints.
// "Construction line with midpoint" = a construction Line + an auto-created midpoint
//   Point with a Midpoint constraint, usable as a constraint target.
```

### 5.2 Constraint solving
All point coordinates are unknowns **x ∈ ℝⁿ**. Each constraint is a residual function
**fᵢ(x) = 0** (e.g. coincident → `pa - pb = 0`; horizontal → `ay - by = 0`; distance →
`‖pa-pb‖ - d = 0`; midpoint → `pm - (pa+pb)/2 = 0`). Solve the system with
**Newton / Gauss-Newton least squares** (nalgebra), seeded from current positions so the
solution stays near where the user drew. Under-constrained sketches have remaining DOF —
that's fine; dragging picks one solution. Over/conflicting constraints are detected and
flagged.

Start with ~6 constraint types and a dense solver (sketches are small, dozens of
points). Optimize / switch to `slvs` only if needed.

---

## 6. The Viewport & UI (Bevy)

- **Scene:** the part's tessellated faces as Bevy meshes; edges as line meshes; planes as
  semi-transparent quads with a normal gizmo.
- **Camera:** orbit/pan/zoom; standard view presets (front/top/iso).
- **Picking:** Bevy's built-in ray picking returns the entity + triangle; we map the
  hit back to a kernel face/edge via a `MeshId → (FeatureId, TopoId)` side table. This
  is how "click a face to sketch on it" and "select an edge to fillet" work.
- **Sketch mode:** when sketching, we lock the camera to look at the plane, project the
  cursor ray onto the plane → UV coords, and run the active draw tool. egui shows the
  sketch toolbar; live constraint inference (snap-to-endpoint, auto-horizontal) gives
  the SolidWorks feel.
- **Feature tree panel:** egui tree view of `Document.features` with the rollback bar;
  double-click to edit, right-click to suppress/delete, drag the bar to roll back.

---

## 7. Crate / Workspace Layout

A Cargo **workspace** keeps the layers physically separated and independently testable
(critical: the kernel and solver must be testable headless, with no Bevy/window).

```
hworks/
├─ Cargo.toml                  # [workspace]
├─ DESIGN.md
├─ crates/
│  ├─ hworks-geometry/         # Layer 1+: GeometryKernel trait + truck impl, tessellation
│  │     trait GeometryKernel { fn extrude(..); fn cut(..); fn tessellate(..); }
│  ├─ hworks-sketch/           # Layer 2: sketch entities + constraint solver (no kernel dep)
│  ├─ hworks-document/         # Layer 3: Feature tree, regeneration, serde save/load
│  │                           #          depends on geometry + sketch, NOT on bevy
│  └─ hworks-app/              # Layer 4: Bevy app, viewport, picking, egui panels
└─ assets/
```

The **headless core** (`geometry` + `sketch` + `document`) compiles and is unit-tested
without a GPU or window. The Bevy app is a thin presentation shell on top. This split is
what keeps a years-long project sane.

---

## 8. Roadmap (incremental milestones)

Each milestone is independently runnable and demoable.

- **M0 — Skeleton.** Workspace, Bevy window, orbit camera, render the three default
  reference planes. *You can fly around 3 planes.*
- **M1 — Pick a plane → flat sketch mode.** Click a plane, camera aligns, draw raw
  (unconstrained) lines and circles in UV; render them. *No solver yet.*
- **M2 — Constraint solver v1.** coincident, horizontal, vertical, distance, midpoint.
  Rectangle tool (built from constraints). Construction lines. Dragging re-solves.
- **M3 — First solid.** Extrude a closed sketch profile into a solid via truck;
  tessellate; render shaded with edges. *2D → 3D.*
- **M4 — Cut + feature tree.** Extrude-Cut (boolean). Feature-tree panel listing
  Plane/Sketch/Extrude/Cut. *History becomes visible.*
- **M5 — Sketch on a face.** Pick a planar face of the solid, start a sketch on it,
  extrude/cut again. Introduces stable topo-IDs (naming Phase 2 begins).
- **M6 — Edit & regenerate.** Double-click an earlier feature, change a dimension or
  sketch, regenerate downstream. Rollback bar. *The parametric payoff.*
- **M7 — Save / load.** serde+ron document persistence.
- **M8+ —** Revolve, Fillet, Chamfer, linear/circular Pattern, more constraints
  (tangent, equal, parallel), dimensions display, equation/variable table.

---

## 9. Key Risks & Decisions Log

| Risk / Decision | Stance |
|---|---|
| Topological naming | Anticipated in the topo wrapper from M5; solved incrementally. |
| truck boolean robustness | Kernel behind a trait (§7) so OCCT swap stays cheap. |
| cgmath (truck) vs glam (Bevy) | Single `convert.rs` boundary; never mix mid-module. |
| Bevy churn between versions | Pin versions; Layers 1–3 are Bevy-free so churn is contained to `hworks-app`. |
| Solver scope creep | Dense Newton first; only adopt `slvs` if profiling demands it. |

---

## 10. Open Questions (to revisit)
- Units & precision policy (mm internally? tolerance for coincidence?).
- Undo/redo: command pattern over the Document vs. snapshot diffing.
- How much of SolidWorks' *constraint inference UX* (auto-snapping) to chase early.
```
