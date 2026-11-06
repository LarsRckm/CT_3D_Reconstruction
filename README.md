# CT 3D Reconstruction

Build an inspectable **3D model of a battery cell from a folder of 2D CT slices**.

This is a purely geometric tool (no machine learning): it stacks the real cross‑section
images along the cell axis, places each slice at its true physical depth, and writes a
set of 3D outputs — including a **self‑contained interactive HTML viewer** you can open
with a double‑click.

## What it does

1. Load every image in a folder and sort the slices by depth (parsed from the file name, in mm).
2. Place each slice at its real depth and resample onto a regular voxel grid — this keeps the
   proportions correct even when the scan samples unevenly along the axis.
3. Derive physical voxel sizes: in‑plane from the cell format (18650 → 14.4 µm, 2170 → 16.4 µm,
   4680 → 35 µm), and the slice spacing from the measured depths.
4. Export the volume, a surface mesh, orthogonal previews, and an interactive viewer.

## Install

```bash
pip install -r requirements.txt
```

`tkinter` (used for the folder‑picker dialog) ships with Python. Three.js for the viewer is
bundled in `lib/` and embedded into the HTML, so the viewer works fully offline.

## Usage

Pick a folder via a dialog:

```bash
python build_3d_from_slices.py
```

Or pass the folder directly:

```bash
python build_3d_from_slices.py --input "path/to/slices" --output "path/to/out" --max-inplane 320
```

Useful options:

| Option | Meaning |
|---|---|
| `--input` / `--output` | slice folder / output folder (dialog if `--input` omitted) |
| `--max-inplane N` | load/resize slices to at most N px in‑plane |
| `--html-max-inplane N` / `--html-max-z N` | resolution of the volume embedded in the viewer (higher = sharper, larger file) |
| `--voxel-um` / `--z-um` | override the in‑plane voxel size / slice spacing (µm) |
| `--no-true-depth` | stack slices evenly instead of at their true depths |
| `--no-mesh` / `--no-html` | skip the STL mesh / the HTML viewer |

## Outputs

- **`view_3d.html`** — interactive viewer (volume rendering you cut into + synchronized 2D slice); opens offline by double‑click.
- `volume.tif` / `volume_geo.tif` — 3D stacks (Fiji/ImageJ, napari, 3D Slicer).
- `surface.stl` — surface mesh (Blender, 3D viewer).
- `preview_ortho.png`, `metadata.json` — previews and the parameters used.

## Notes

- File names are expected to end in the slice depth in mm (e.g. `-0.5.png`, `32.0.png`).
- The in‑plane voxel size is inferred from the cell format in the path (`18650` / `2170` / `4680`);
  use `--voxel-um` otherwise.
- Vertical detail is limited by how densely the scan was sampled along the axis.
