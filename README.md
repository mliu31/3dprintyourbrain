# Brain Extraction From NIfTI

## Files

- `extract_brain_from_nifti.py` - runs HD-BET skull stripping, writes a brain NIfTI, mask, QA PNG, surface preview PNG, and optional STL.
- `generate_brain_slices_png.py` - creates a sagittal/coronal/axial slice montage, with optional mask overlay.
- `generate_brain_spin_gif.py` - renders a z-axis spinning GIF from a binary STL.

## Requirements

Install the Python packages once:

```powershell
python -m pip install hd-bet nibabel numpy scipy scikit-image pillow
```

HD-BET may download model weights the first time it runs.

## Recreate The Current Output

This command recreates the 4x high-resolution STL workflow used for the current output. It writes outputs to `C:\Users\mgnli\Downloads\brain_output`:

```powershell
python C:\Users\mgnli\Downloads\brain_extraction_workflow\extract_brain_from_nifti.py `
  "C:\Users\mgnli\Downloads\sub-sid001879_acq-MPRAGE_T1w.nii\sub-sid001879_acq-MPRAGE_T1w.nii" `
  -o "C:\Users\mgnli\Downloads\brain_output" `
  --prefix sub-sid001879 `
  --stl-factor 4
```

Expected outputs:

- `sub-sid001879_brain.nii.gz`
- `sub-sid001879_brain_mask.nii.gz`
- `sub-sid001879_brain.stl`
- `sub-sid001879_brain_QA.png`
- `sub-sid001879_brain_surface_preview.png`

## Generate Slices PNG

Raw slices only:

```powershell
python C:\Users\mgnli\Downloads\brain_extraction_workflow\generate_brain_slices_png.py `
  "C:\Users\mgnli\Downloads\sub-sid001879_acq-MPRAGE_T1w.nii\sub-sid001879_acq-MPRAGE_T1w.nii" `
  -o "C:\Users\mgnli\Downloads\sub-sid001879_raw_slices.png"
```

Mask overlay slices:

```powershell
python C:\Users\mgnli\Downloads\brain_extraction_workflow\generate_brain_slices_png.py `
  "C:\Users\mgnli\Downloads\sub-sid001879_acq-MPRAGE_T1w.nii\sub-sid001879_acq-MPRAGE_T1w.nii" `
  --mask "C:\Users\mgnli\Downloads\sub-sid001879_brain_only_mask.nii.gz" `
  -o "C:\Users\mgnli\Downloads\sub-sid001879_brain_only_QA.png"
```

## Generate Spin GIF

This renders a GIF from an existing STL. It does not change the STL.

```powershell
python C:\Users\mgnli\Downloads\brain_extraction_workflow\generate_brain_spin_gif.py `
  "C:\Users\mgnli\Downloads\sub-sid001879_brain_only.stl" `
  -o "C:\Users\mgnli\Downloads\sub-sid001879_brain_only_spin_z.gif"
```

Useful options:

- `--frames 48` controls animation length.
- `--size 640` controls image dimensions.
- `--max-faces 520000` controls how many STL triangles are sampled for rendering.

## Notes

HD-BET performs skull stripping. The STL is a surface generated from the extracted brain mask. For a true cortical pial surface with fine sulci/gyri anatomy, use FreeSurfer `recon-all` or a similar cortical reconstruction pipeline.

The current generated files in `C:\Users\mgnli\Downloads` are separate from this workflow folder:

- `sub-sid001879_brain_only.stl`
- `sub-sid001879_brain_only_mask.nii.gz`
- `sub-sid001879_brain_only_QA.png`
- `sub-sid001879_brain_only_surface_preview.png`
- `sub-sid001879_brain_only_spin_z.gif`
