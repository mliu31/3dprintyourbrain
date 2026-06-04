# 3D Print Your Brain

Turn a 3D anatomical brain MRI in NIfTI format into files that are easier to inspect, validate, and 3D print.

The workflow uses HD-BET for skull stripping, then exports a brain mask, QA images, and an optional STL mesh generated from the extracted brain surface.

## Contents

- `extract_brain_from_nifti.py` - skull-strips a 3D NIfTI file and writes the extracted brain, mask, QA image, surface preview, and optional STL.
- `generate_brain_slices_png.py` - creates sagittal, coronal, and axial slice montages, with optional mask overlay.
- `generate_brain_spin_gif.py` - renders a z-axis spinning GIF from a binary STL.

## Requirements

- Python 3.10+
- A 3D anatomical NIfTI file (`.nii` or `.nii.gz`)
- Python packages:

```powershell
python -m pip install hd-bet nibabel numpy scipy scikit-image pillow
```

HD-BET may download model weights the first time it runs.

## Quick Start

Run the main workflow from the repo root:

```powershell
python .\extract_brain_from_nifti.py .\data\subject_T1w.nii.gz `
  -o .\outputs `
  --prefix subject `
  --stl-factor 2
```

Expected outputs:

- `outputs/subject_brain.nii.gz`
- `outputs/subject_brain_mask.nii.gz`
- `outputs/subject_brain_QA.png`
- `outputs/subject_brain_surface_preview.png`
- `outputs/subject_brain.stl`

Use `--stl-factor 1` for a smaller, faster mesh. Use `--stl-factor 2`, `3`, or `4` for denser STL output.

## Main Workflow Options

```powershell
python .\extract_brain_from_nifti.py .\data\subject_T1w.nii.gz `
  -o .\outputs `
  --prefix subject `
  --device cpu `
  --stl-factor 2
```

Useful options:

- `--device cpu|cuda|mps` selects the HD-BET device.
- `--tta` enables HD-BET test-time augmentation. This is slower, but can improve some masks.
- `--no-stl` skips STL generation.
- `--stl-factor` controls STL upsampling density.
- `--stl-smoothing` controls smoothing for `--stl-factor 1`.
- `--no-qa` skips the QA overlay PNG.
- `--no-preview` skips the surface preview PNG.

## Generate Slice Montages

Create raw MRI slices:

```powershell
python .\generate_brain_slices_png.py .\data\subject_T1w.nii.gz `
  -o .\outputs\subject_slices.png
```

Create slices with a brain mask overlay:

```powershell
python .\generate_brain_slices_png.py .\data\subject_T1w.nii.gz `
  --mask .\outputs\subject_brain_mask.nii.gz `
  -o .\outputs\subject_mask_overlay.png
```

Useful options:

- `--slices-per-axis` controls how many sagittal, coronal, and axial slices are shown.
- `--lower-pct` and `--upper-pct` control intensity scaling.

## Generate A Spin GIF

Render a GIF from an existing binary STL:

```powershell
python .\generate_brain_spin_gif.py .\outputs\subject_brain.stl `
  -o .\outputs\subject_brain_spin_z.gif
```

Useful options:

- `--frames` controls animation length.
- `--size` controls output image dimensions.
- `--max-faces` controls how many STL triangles are sampled for rendering.
- `--duration-ms` controls frame duration.
- `--splat-radius` controls the rendered point size.

## Notes

The STL is generated from the extracted brain mask. It is useful for visualization and 3D printing, but it is not a true cortical pial surface reconstruction. For finer sulci and gyri anatomy, use FreeSurfer `recon-all` or a similar cortical reconstruction pipeline.

Review the QA image and surface preview before printing. Skull-stripping quality depends on the input MRI and HD-BET output.
