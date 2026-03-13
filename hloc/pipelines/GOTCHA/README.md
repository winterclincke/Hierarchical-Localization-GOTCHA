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

Behavior in this step:
- imported reference poses are fixed.
- intrinsics refinement is disabled.
- this keeps compatibility with reusing an existing ODM mesh reconstruction.

`--allow-pose-adjustment` is kept only for backward CLI compatibility and is intentionally disabled.
If you run external BA/refinement (for example PixSfM), regenerate downstream dense/MVS mesh so the updated poses stay aligned with the 3D digital twin.

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
```

Per camera/PTZ workflow:
- run `localize_queries` again with a different single `--gt-center` value.

Localization trials (experimental):
- This pipeline uses experimental repeated localization trials; they can improve robustness in some scenes but do not guarantee better accuracy.
- Stage 1 (bootstrap pass): one first pose estimation per query is used to bootstrap focal length (`bootstrap_focal_estimated`).
- Stage 2 (primary trials): only used when GT center is provided; remaining `LOCALIZATION_TRIALS - 1` runs are evaluated and the best pose is selected by smallest center distance.
- Stage 3 (fallback trials): only when `--fallback-exhaustive` is enabled and primary localization fails or is weak.
  Without GT center, fallback runs once.
  With GT center, fallback uses `FALLBACK_TRIALS`.
- To simplify/omit most trial behavior: run without `--fallback-exhaustive`, and set `LOCALIZATION_TRIALS=1` (optionally `FALLBACK_TRIALS=1`) in `hloc/pipelines/GOTCHA/pipeline.py`.

Future note:
- a pending COLMAP/pycolmap release adds camera-center prior support during absolute-pose refinement.
- after that release, this pipeline can replace the current custom fixed-center solver with the native solver path.
- if refined poses are adopted, regenerate dense/MVS mesh to keep alignment with the 3D digital twin.

## Outputs

`localize_queries` writes:
- `project/outputs/poses.json` (downstream-friendly per-query records)

Each `poses.json` entry contains:
- `query`, `status`, `pose_source`
- `qvec`, `tvec`
- `camera` (`model`, `width`, `height`, `f`, `cx`, `cy`, `k`)
- `inliers`, `center_distance_m`
- `bootstrap_success`, `bootstrap_focal_initial`, `bootstrap_focal_estimated`

Failed queries are still included with `status: "failed"` and null pose/camera fields.

## Assumptions

- Single-camera dataset.
- Valid OpenSfM/ODM NVM file at `project/opensfm/undistorted/reconstruction.nvm`.
- `project/images` corresponds to the NVM images (unique basenames).
- `empty_rec` intentionally has no 3D points.
