# GOTCHA pipeline

- reference images have known poses (`project/empty_rec`) - for example from opendronemap/opensfm reconstruction
- queries are localized against the triangulated reference model
- optional GT camera centers can guide selection and fixed-center refinement (when you know the 3d position e.g., from CCTV or PTZ cams)

## Project layout

```text
project/
  images/          # reference images
  queries/         # query images (optionally grouped in subfolders)
  empty_rec/       # COLMAP model with known reference poses
  outputs/         # generated outputs
```

## Chronological workflow

### Step 1: Import from ODM/OpenSfM

Use the ODM/OpenSfM undistorted reference images as `project/images` (copy or symlink).

`prepare_empty_rec` uses fixed project-layout paths:
- `project/opensfm/undistorted/reconstruction.nvm`
- `project/images`

The `reconstruction.nvm` file and the images in `project/images` must correspond.

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_empty_rec --project /path/to/project
```

Output of this step:
- `project/empty_rec/cameras.txt`, `images.txt`, `points3D.txt`
- `project/empty_rec/cameras.bin`, `images.bin`, `points3D.bin`

Camera/georeference note:
- `prepare_empty_rec` writes one shared `SIMPLE_RADIAL` camera (single-camera assumption).
- It does not create georeferencing by itself; it preserves the OpenSfM/ODM reference frame.
- If your ODM reconstruction is georeferenced (depending on your ODM setup, e.g. GPS/RTK/calibration), `empty_rec` follows that frame.

### Step 2: Triangulate reference points

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference --project /path/to/project
```

Default behavior in this step:
- Imported reference poses are kept fixed (pose adjustment disabled by default).
- This default is intended to keep compatibility with reusing an existing ODM mesh reconstruction.

Optional:
- Enable pose adjustment/refinement with:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference \
  --project /path/to/project \
  --allow-pose-adjustment
```

- If you enable pose adjustment/refinement, re-run downstream dense/mesh reconstruction (MVS) with updated poses for consistency.

### Step 3: Localize query images

Base command:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries --project /path/to/project
```

Common options:
- Fallback exhaustive:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --fallback-exhaustive
```

- Single GT center value:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --gt-center 7.65 2.50 -0.95 \
  --use-fixed-center
```

- Per camera/PTZ workflow:
  run `localize_queries` again with a different `--gt-center` value for each known camera position.

- Manual query-reference pairs:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --manual-pairs /path/to/query-pairs.txt
```

## Outputs

`localize_queries` writes:
- `project/outputs/results.txt`
- `project/outputs/cameras_extra.txt`
- `project/outputs/images_extra.txt`
- `project/outputs/query_summary.json`

`query_summary.json` includes per query:
- status
- inliers
- center distance (if GT center is available)
- fallback usage
- reprojection stats for fixed-center refinement (if enabled)

## Assumptions

- Single camera setup for the full dataset.
- A valid OpenSfM/ODM NVM file exists at `project/opensfm/undistorted/reconstruction.nvm`.
- `project/images` contains the same images as the NVM with unique basenames.
- `points3D` in `empty_rec` is intentionally empty.
