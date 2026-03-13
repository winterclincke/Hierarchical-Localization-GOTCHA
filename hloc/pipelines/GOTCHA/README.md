# GOTCHA pipeline

- reference images have known poses (`project/empty_rec`) - for example from opendronemap/opensfm reconstruction
- queries are localized against the triangulated reference model
- optional GT camera centers can guide selection and fixed-center refinement (when you know the 3d position e.g., from CCTV or PTZ cams)

## Project layout

```text
project/
  opensfm/
    undistorted/
      reconstruction.nvm
  images/          # undistorted reference images
  queries/         # query images (optional subfolders)
  empty_rec/       # generated COLMAP known-pose model
  outputs/         # generated outputs
```

## Chronological workflow

### Step 1: Import from ODM/OpenSfM and build `empty_rec`

1. Copy/symlink ODM/OpenSfM undistorted images into `project/images`.
2. Place the matching NVM file at `project/opensfm/undistorted/reconstruction.nvm`.
3. Run:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_empty_rec --project /path/to/project
```

This creates:
- `project/empty_rec/cameras.txt`, `images.txt`, `points3D.txt`
- `project/empty_rec/cameras.bin`, `images.bin`, `points3D.bin`

Camera/georeference note:
- `prepare_empty_rec` uses one shared `SIMPLE_RADIAL` camera (single-camera assumption).
- Georeferencing is not created by this step; it follows the OpenSfM/ODM frame from `reconstruction.nvm`.

### Step 2: Triangulate reference points

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference --project /path/to/project
```

Default behavior:
- imported reference poses are kept fixed (`--allow-pose-adjustment` is off by default), so an existing ODM mesh can be reused.

Optional pose adjustment:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference \
  --project /path/to/project \
  --allow-pose-adjustment
```

If poses are adjusted, re-run downstream dense/mesh reconstruction (MVS) with the updated poses.

### Step 3: Localize query images

Base command:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries --project /path/to/project
```

Common options:

```bash
# enable exhaustive fallback when retrieval localization is weak/failing
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --fallback-exhaustive

# use one GT center for this run and apply fixed-center refinement
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --gt-center 7.65 2.50 -0.95 \
  --use-fixed-center

# override initial focal length for query cameras
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --initial-focal-length 2000
```

Per camera/PTZ workflow:
- run `localize_queries` again with a different single `--gt-center` value.

## Outputs

`localize_queries` writes:
- `project/outputs/results.txt` (HLOC-compatible pose output)
- `project/outputs/poses.json` (downstream-friendly per-query records)

Each `poses.json` entry contains:
- `query`, `status`, `pose_source`
- `qvec`, `tvec`
- `camera` (`model`, `width`, `height`, `f`, `cx`, `cy`, `k`)
- `inliers`, `center_distance_m`

Failed queries are still included with `status: "failed"` and null pose/camera fields.

## Assumptions

- Single-camera dataset.
- Valid OpenSfM/ODM NVM file at `project/opensfm/undistorted/reconstruction.nvm`.
- `project/images` corresponds to the NVM images (unique basenames).
- `empty_rec` intentionally has no 3D points.
