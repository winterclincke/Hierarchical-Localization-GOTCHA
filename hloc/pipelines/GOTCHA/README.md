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

## 1) Prepare reference model

Build sparse points from known poses:

```bash
python3 -m hloc.pipelines.GOTCHA.pipeline prepare_reference --project /path/to/project
```

Main artifacts:
- `project/outputs/features.h5`
- `project/outputs/pairs-db-poses.txt`
- `project/outputs/matches.h5`
- `project/outputs/sfm_triangulated/`

## 2) Localize queries

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
