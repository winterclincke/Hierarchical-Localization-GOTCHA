# GOTCHA pipeline

This pipeline is an upstream-safe extension for fixed-layout projects where:
- reference images have known poses (`project/empty_rec`)
- queries are localized against the triangulated reference model
- optional GT camera centers can guide selection and fixed-center refinement

## Project layout

```text
project/
  images/          # reference images
  queries/         # query images (can include subfolders, e.g. cluster names)
  empty_rec/       # COLMAP text/binary model with known camera poses
  outputs/         # generated artifacts
```

## ODM quick start (3 commands)

Use the same undistorted images in step 1 that you will use later in `prepare_reference`.
In practice: place the ODM undistorted images in `project/images` (copy or symlink).

1. Convert ODM/OpenSfM NVM to `empty_rec`:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_empty_rec --project /path/to/project
```

2. Triangulate sparse reference points from known poses:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference --project /path/to/project
```

3. Localize query images:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries --project /path/to/project
```

## NVM + images relation

- The converter expects `reconstruction.nvm` and the undistorted reference images to match.
- Default NVM path: `project/opensfm/undistorted/reconstruction.nvm`.
- Default image path used for NVM matching: `project/images`.
- If your undistorted images are located elsewhere, use:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_empty_rec \
  --project /path/to/project \
  --nvm /path/to/opensfm/undistorted/reconstruction.nvm \
  --images-dir /path/to/project/images
```

After this, `prepare_reference` must still run on the images in `project/images`.

## Assumptions

- Single camera setup for the full dataset.
- ODM/OpenSfM provides `project/opensfm/undistorted/reconstruction.nvm`.
- `project/images` contains the same images as the NVM (unique basename per file).
- `points3D` in `empty_rec` is intentionally empty.

## Camera model and geo note

- `prepare_empty_rec` writes one shared camera as `SIMPLE_RADIAL` (single-camera assumption).
- This does not create georeferencing by itself; it preserves the OpenSfM/ODM reference frame.
- If your ODM reconstruction is georeferenced (depending on your ODM setup, e.g. GPS/RTK/calibration), `empty_rec` will follow that frame.

## Localization options

Standard retrieval-based localization:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries --project /path/to/project
```

With fallback exhaustive search for hard queries:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --fallback-exhaustive
```

With one shared GT center and fixed-center refinement:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --gt-center 7.65 2.50 -0.95 \
  --use-fixed-center
```

With per-image GT centers:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --gt-center-image-map /path/to/gt_centers_per_image.json \
  --use-fixed-center
```

With per-cluster GT centers:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --gt-center-cluster-map /path/to/gt_centers_per_cluster.json \
  --use-fixed-center
```

Manual query-reference pairs (one `query ref` pair per line):

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline localize_queries \
  --project /path/to/project \
  --manual-pairs /path/to/query-pairs.txt
```

## Outputs

`localize_queries` writes:
- `project/outputs/results.txt` (raw localized poses)
- `project/outputs/cameras_extra.txt`
- `project/outputs/images_extra.txt`
- `project/outputs/query_summary.json`

`query_summary.json` includes per query:
- status
- inliers
- center distance (if GT center available)
- fallback usage
- reprojection stats for fixed-center refinement (if enabled)

## GT center formats

Per-image map (`--gt-center-image-map`):

```json
{
  "default": [0.0, 0.0, 0.0],
  "queries/T2/StaticCam_123.png": [-1.46, 0.92, -0.86]
}
```

Per-cluster map (`--gt-center-cluster-map`):

```json
{
  "T1": [5.37, -2.30, -0.90],
  "T2": [-1.46, 0.92, -0.86]
}
```

Cluster name is derived from the first subfolder under `project/queries`.
Example: `project/queries/T2/StaticCam_123.png` belongs to cluster `T2`.
