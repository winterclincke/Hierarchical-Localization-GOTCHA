import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as ScipyRotation

from ... import (
    extract_features,
    logger,
    match_features,
    pairs_from_exhaustive,
    pairs_from_poses,
    pairs_from_retrieval,
    triangulation,
)
from ...localize_sfm import QueryLocalizer, pose_from_cluster
from ...utils.parsers import parse_retrieval
from .fixed_center_solver import compute_center, refine_pose_fixed_center
from .opensfm_to_empty_rec import create_empty_rec_from_nvm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_OPENSFM_NVM = Path("opensfm/undistorted/reconstruction.nvm")
DEFAULT_REFERENCE_IMAGES = Path("images")
DEFAULT_QUERY_IMAGES = Path("queries")
DEFAULT_EMPTY_REC_DIR = Path("empty_rec")
DEFAULT_OUTPUT_DIR = Path("outputs")

LOCAL_FEATURE_CONF = "superpoint_max"
GLOBAL_FEATURE_CONF = "netvlad"
MATCHER_CONF = "superpoint+lightglue"
MAX_KEYPOINTS = 10000
NUM_POSE_PAIRS = 35
ROTATION_THRESHOLD = 120.0
NUM_MATCHED = 35
# Experimental heuristic: repeated trials can improve robustness, but gains are scene-dependent.
LOCALIZATION_TRIALS = 5
# Experimental heuristic for fallback exhaustive localization.
FALLBACK_TRIALS = 10
FALLBACK_MIN_INLIERS = 50

LOCALIZER_CONFIG = {
    "estimation": {
        "ransac": {
            "max_error": 12.0,
            "confidence": 0.99,
            "min_inlier_ratio": 0.3,
            "min_num_trials": 100,
            "max_num_trials": 20000,
        },
        "estimate_focal_length": False,
    },
    "refinement": {
        "max_num_iterations": 10000,
        "refine_focal_length": True,
        "refine_extra_params": False,
        "print_summary": False,
    },
}


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def to_relative_paths(paths: List[Path], project: Path) -> List[str]:
    return [p.relative_to(project).as_posix() for p in paths]


def get_num_inliers(ret: Optional[Dict[str, Any]]) -> int:
    if ret is None:
        return 0
    if "inlier_mask" in ret and ret["inlier_mask"] is not None:
        return int(np.sum(np.asarray(ret["inlier_mask"], dtype=bool)))
    if "num_inliers" in ret:
        return int(ret["num_inliers"])
    return 0


def get_pose_metrics(
    ret: Dict[str, Any], gt_center: Optional[np.ndarray]
) -> Dict[str, Any]:
    inliers = get_num_inliers(ret)
    center_distance = None
    if gt_center is not None:
        center = compute_center(ret["cam_from_world"])
        center_distance = float(np.linalg.norm(center - gt_center))
    return {"inliers": int(inliers), "center_distance_m": center_distance}


def get_selection_key(
    metrics: Dict[str, Any], gt_center: Optional[np.ndarray]
) -> Tuple[float, ...]:
    if gt_center is not None:
        return (float(metrics["center_distance_m"]), -int(metrics["inliers"]))
    return (-int(metrics["inliers"]),)


def extract_inlier_correspondences(
    model: pycolmap.Reconstruction,
    log: Dict[str, Any],
    inlier_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(log.get("keypoints_query", []), dtype=float)
    point3d_ids = np.asarray(log.get("points3D_ids", []), dtype=np.int64)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)

    n = min(len(keypoints), len(point3d_ids), len(inlier_mask))
    if n == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)

    keypoints = keypoints[:n]
    point3d_ids = point3d_ids[:n]
    inlier_mask = inlier_mask[:n]
    selected = inlier_mask & (point3d_ids != -1)
    if not np.any(selected):
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)

    points2d = []
    points3d = []
    for p2d, pid in zip(keypoints[selected], point3d_ids[selected]):
        pid_int = int(pid)
        if pid_int in model.points3D:
            points2d.append(p2d)
            points3d.append(model.points3D[pid_int].xyz)

    if not points2d:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(points2d, dtype=float), np.asarray(points3d, dtype=float)


def camera_to_simple_radial(camera: pycolmap.Camera, focal_override: Optional[float] = None) -> Dict[str, Any]:
    params = list(np.asarray(camera.params, dtype=float)) if hasattr(camera, "params") else []
    focal = float(
        focal_override
        if focal_override is not None
        else getattr(camera, "focal_length", params[0] if params else 1.0)
    )
    cx = float(
        getattr(
            camera,
            "principal_point_x",
            params[1] if len(params) > 1 else camera.width / 2.0,
        )
    )
    cy = float(
        getattr(
            camera,
            "principal_point_y",
            params[2] if len(params) > 2 else camera.height / 2.0,
        )
    )
    k = float(params[3]) if len(params) > 3 else 0.0
    return {
        "model": "SIMPLE_RADIAL",
        "width": int(camera.width),
        "height": int(camera.height),
        "f": focal,
        "cx": cx,
        "cy": cy,
        "k": k,
    }


def rigid3d_from_rt(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> pycolmap.Rigid3d:
    rotation_matrix = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    translation_vector = np.asarray(translation_vector, dtype=float).reshape(3)

    try:
        return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation_matrix), translation_vector)
    except Exception:
        pass

    quat_xyzw = ScipyRotation.from_matrix(rotation_matrix).as_quat()
    try:
        return pycolmap.Rigid3d(pycolmap.Rotation3d(quat_xyzw), translation_vector)
    except Exception:
        pass

    try:
        return pycolmap.Rigid3d(rotation_matrix, translation_vector)
    except Exception as exc:
        raise RuntimeError("Failed to convert fixed-center pose to pycolmap.Rigid3d.") from exc


def localize_with_trials(
    localizer: QueryLocalizer,
    query_name: str,
    camera: pycolmap.Camera,
    db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    trials: int,
    gt_center: Optional[np.ndarray],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    best_ret = None
    best_log = None
    best_key = None
    best_metrics = {"inliers": 0, "center_distance_m": None}

    for _ in range(trials):
        ret, log = pose_from_cluster(
            localizer, query_name, camera, db_ids, features_path, matches_path
        )
        if ret is None:
            continue

        metrics = get_pose_metrics(ret, gt_center)
        key = get_selection_key(metrics, gt_center)

        if best_key is None or key < best_key:
            best_key = key
            best_ret = ret
            best_log = log
            best_metrics = metrics

    return best_ret, best_log, best_metrics


def run_fallback_exhaustive(
    outputs_dir: Path,
    query_name: str,
    ref_list: List[str],
    matcher_conf: Dict[str, Any],
    localizer: QueryLocalizer,
    camera: pycolmap.Camera,
    all_ref_db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    trials: int,
    gt_center: Optional[np.ndarray],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    fallback_pairs_path = outputs_dir / "fallback_pairs" / f"{query_name.replace('/', '__')}.txt"
    fallback_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_from_exhaustive.main(
        output=fallback_pairs_path,
        image_list=[query_name],
        ref_list=ref_list,
    )
    match_features.main(
        matcher_conf,
        fallback_pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=False,
    )
    return localize_with_trials(
        localizer=localizer,
        query_name=query_name,
        camera=camera,
        db_ids=all_ref_db_ids,
        features_path=features_path,
        matches_path=matches_path,
        trials=trials,
        gt_center=gt_center,
    )


def run_prepare_empty_rec(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    nvm_path = project / DEFAULT_OPENSFM_NVM
    if not nvm_path.exists():
        raise FileNotFoundError(
            f"Missing NVM file: {nvm_path}. Expected at project/opensfm/undistorted/reconstruction.nvm"
        )
    create_empty_rec_from_nvm(
        project=project,
        images_dir=project / DEFAULT_REFERENCE_IMAGES,
        output_dir=project / DEFAULT_EMPTY_REC_DIR,
        nvm_path=nvm_path,
    )


def run_prepare_reference(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    images_dir = project / DEFAULT_REFERENCE_IMAGES
    empty_rec_dir = project / DEFAULT_EMPTY_REC_DIR
    outputs_dir = project / DEFAULT_OUTPUT_DIR

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing directory: {images_dir}")
    if not empty_rec_dir.exists():
        raise FileNotFoundError(f"Missing directory: {empty_rec_dir}")

    refs = list_images(images_dir)
    if not refs:
        raise ValueError(f"No reference images found in {images_dir}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    features_path = outputs_dir / "features.h5"
    matches_path = outputs_dir / "matches.h5"
    pairs_path = outputs_dir / "pairs-db-poses.txt"
    sfm_dir = outputs_dir / "sfm_triangulated"

    ref_list = to_relative_paths(refs, project)

    feature_conf = copy.deepcopy(extract_features.confs[LOCAL_FEATURE_CONF])
    if "model" in feature_conf and "max_keypoints" in feature_conf["model"]:
        feature_conf["model"]["max_keypoints"] = MAX_KEYPOINTS

    extract_features.main(
        feature_conf,
        project,
        image_list=ref_list,
        feature_path=features_path,
        overwrite=False,
    )

    pairs_from_poses.main(
        empty_rec_dir,
        pairs_path,
        num_matched=NUM_POSE_PAIRS,
        rotation_threshold=ROTATION_THRESHOLD,
    )

    matcher_conf = match_features.confs[MATCHER_CONF]
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=False,
    )

    if args.allow_pose_adjustment:
        raise RuntimeError(
            "--allow-pose-adjustment is intentionally disabled in prepare_reference. "
            "For pose refinement, use a dedicated BA/PixSfM workflow and regenerate dense/MVS mesh afterwards "
            "to keep alignment with the 3D digital twin."
        )

    triangulation_options = pycolmap.IncrementalPipelineOptions()
    triangulation_options.fix_existing_frames = True
    triangulation_options.mapper.fix_existing_frames = True
    triangulation_options.ba_refine_focal_length = False
    triangulation_options.ba_refine_extra_params = False
    # TODO: evaluate PixSfM triangulation/refiner integration for higher-accuracy refinement workflows.
    logger.info("Fixed-pose reference triangulation: frames and intrinsics are locked.")

    triangulation.main(
        sfm_dir=sfm_dir,
        reference_model=empty_rec_dir,
        image_dir=project,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        skip_geometric_verification=False,
        estimate_two_view_geometries=False,
        verbose=False,
        mapper_options=triangulation_options,
    )
    logger.info("Reference model written to %s", sfm_dir)


def run_localize_queries(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    images_dir = project / DEFAULT_REFERENCE_IMAGES
    queries_dir = project / DEFAULT_QUERY_IMAGES
    outputs_dir = project / DEFAULT_OUTPUT_DIR
    sfm_dir = outputs_dir / "sfm_triangulated"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing directory: {images_dir}")
    if not queries_dir.exists():
        raise FileNotFoundError(f"Missing directory: {queries_dir}")
    if not sfm_dir.exists():
        raise FileNotFoundError(f"Missing model: {sfm_dir}. Run prepare_reference first.")

    refs = list_images(images_dir)
    queries = list_images(queries_dir)
    if not refs:
        raise ValueError(f"No reference images found in {images_dir}")
    if not queries:
        raise ValueError(f"No query images found in {queries_dir}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    features_path = outputs_dir / "features.h5"
    matches_path = outputs_dir / "matches.h5"
    global_desc_path = outputs_dir / "global-features.h5"
    pairs_path = outputs_dir / "pairs-query-ref.txt"
    poses_json_path = outputs_dir / "poses.json"

    ref_list = to_relative_paths(refs, project)
    query_list = to_relative_paths(queries, project)

    feature_conf = copy.deepcopy(extract_features.confs[LOCAL_FEATURE_CONF])
    if "model" in feature_conf and "max_keypoints" in feature_conf["model"]:
        feature_conf["model"]["max_keypoints"] = MAX_KEYPOINTS
    extract_features.main(
        feature_conf,
        project,
        image_list=ref_list + query_list,
        feature_path=features_path,
        overwrite=False,
    )

    extract_features.main(
        extract_features.confs[GLOBAL_FEATURE_CONF],
        project,
        image_list=query_list + ref_list,
        feature_path=global_desc_path,
        overwrite=False,
    )

    pairs_from_retrieval.main(
        descriptors=global_desc_path,
        output=pairs_path,
        num_matched=NUM_MATCHED,
        query_list=query_list,
        db_list=ref_list,
    )
    retrieval = parse_retrieval(pairs_path)

    matcher_conf = match_features.confs[MATCHER_CONF]
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=False,
    )

    model = pycolmap.Reconstruction(sfm_dir)
    localizer = QueryLocalizer(model, LOCALIZER_CONFIG)

    name_to_id = {image.name: image_id for image_id, image in model.images.items()}
    all_ref_db_ids = [name_to_id[name] for name in ref_list if name in name_to_id]

    gt_center = np.array(args.gt_center, dtype=float) if args.gt_center else None

    localized_count = 0
    pose_records: List[Dict[str, Any]] = []

    for query_path in queries:
        query_name = query_path.relative_to(project).as_posix()

        camera = pycolmap.infer_camera_from_image(query_path)
        bootstrap_focal_initial = float(camera.focal_length)
        bootstrap_focal_estimated: Optional[float] = None
        bootstrap_success = False

        db_ids = [name_to_id[n] for n in retrieval.get(query_name, []) if n in name_to_id]
        ret = None
        log = None
        metrics = {"inliers": 0, "center_distance_m": None}
        pose_source = "primary"

        if db_ids:
            bootstrap_ret, bootstrap_log = pose_from_cluster(
                localizer, query_name, camera, db_ids, features_path, matches_path
            )
            if bootstrap_ret is not None:
                bootstrap_success = True
                bootstrap_metrics = get_pose_metrics(bootstrap_ret, gt_center)
                bootstrap_focal_estimated = float(
                    getattr(bootstrap_ret.get("camera", camera), "focal_length", camera.focal_length)
                )
                camera.focal_length = bootstrap_focal_estimated
                ret = bootstrap_ret
                log = bootstrap_log
                metrics = bootstrap_metrics

            # Only use repeated primary trials when a GT center is available.
            if gt_center is not None:
                remaining_ret, remaining_log, remaining_metrics = localize_with_trials(
                    localizer=localizer,
                    query_name=query_name,
                    camera=camera,
                    db_ids=db_ids,
                    features_path=features_path,
                    matches_path=matches_path,
                    trials=max(LOCALIZATION_TRIALS - 1, 0),
                    gt_center=gt_center,
                )
                # Keep the best-scoring pose over trials, not simply the first valid estimate.
                if remaining_ret is not None and (
                    ret is None
                    or get_selection_key(remaining_metrics, gt_center)
                    < get_selection_key(metrics, gt_center)
                ):
                    ret = remaining_ret
                    log = remaining_log
                    metrics = remaining_metrics

        fallback_used = False
        if args.fallback_exhaustive and (ret is None or metrics["inliers"] < FALLBACK_MIN_INLIERS):
            fallback_trials = FALLBACK_TRIALS if gt_center is not None else 1
            fallback_ret, fallback_log, fallback_metrics = run_fallback_exhaustive(
                outputs_dir=outputs_dir,
                query_name=query_name,
                ref_list=ref_list,
                matcher_conf=matcher_conf,
                localizer=localizer,
                camera=camera,
                all_ref_db_ids=all_ref_db_ids,
                features_path=features_path,
                matches_path=matches_path,
                trials=fallback_trials,
                gt_center=gt_center,
            )
            if fallback_ret is not None:
                fallback_used = True
                ret = fallback_ret
                log = fallback_log
                metrics = fallback_metrics
                pose_source = "fallback"

        if ret is None or log is None:
            pose_records.append(
                {
                    "query": query_name,
                    "status": "failed",
                    "pose_source": "none",
                    "qvec": None,
                    "tvec": None,
                    "camera": None,
                    "inliers": 0,
                    "center_distance_m": None,
                    "fallback_used": fallback_used,
                    "bootstrap_success": bootstrap_success,
                    "bootstrap_focal_initial": bootstrap_focal_initial,
                    "bootstrap_focal_estimated": bootstrap_focal_estimated,
                }
            )
            continue

        selected_pose = ret["cam_from_world"]
        selected_camera = ret.get("camera", camera)

        if args.use_fixed_center and gt_center is not None:
            inlier_mask = np.asarray(ret.get("inlier_mask", []), dtype=bool)
            points2d, points3d = extract_inlier_correspondences(model, log, inlier_mask)
            if len(points2d) >= 4:
                # TODO: replace this custom solver once pycolmap exposes center-prior absolute-pose refinement in a stable release.
                # feature is implemented but not yet exposed (will be available in pycolmap 3.14+): https://github.com/colmap/colmap/pull/4107
                fixed_ret = refine_pose_fixed_center(
                    points2d=points2d,
                    points3d=points3d,
                    camera=selected_camera,
                    fixed_center=gt_center,
                    initial_rotation=selected_pose.rotation.matrix(),
                    initial_translation=np.asarray(selected_pose.translation, dtype=float),
                    optimize_focal=True,
                )
                if fixed_ret["success"]:
                    selected_pose = rigid3d_from_rt(
                        fixed_ret["rotation_matrix"], fixed_ret["translation_vector"]
                    )
                    pose_source = "fixed"
                    selected_camera_dict = camera_to_simple_radial(
                        selected_camera, focal_override=float(fixed_ret["focal_length"])
                    )
                else:
                    selected_camera_dict = camera_to_simple_radial(selected_camera)
            else:
                selected_camera_dict = camera_to_simple_radial(selected_camera)
        else:
            selected_camera_dict = camera_to_simple_radial(selected_camera)

        localized_count += 1
        qxyzw = np.asarray(selected_pose.rotation.quat, dtype=float)
        qvec = [float(qxyzw[3]), float(qxyzw[0]), float(qxyzw[1]), float(qxyzw[2])]
        tvec = [float(v) for v in np.asarray(selected_pose.translation, dtype=float).reshape(3)]

        center_distance_m = None
        if gt_center is not None:
            center_distance_m = float(np.linalg.norm(compute_center(selected_pose) - gt_center))

        pose_records.append(
            {
                "query": query_name,
                "status": "localized",
                "pose_source": pose_source,
                "qvec": qvec,
                "tvec": tvec,
                "camera": selected_camera_dict,
                "inliers": int(metrics["inliers"]),
                "center_distance_m": center_distance_m,
                "fallback_used": fallback_used,
                "bootstrap_success": bootstrap_success,
                "bootstrap_focal_initial": bootstrap_focal_initial,
                "bootstrap_focal_estimated": bootstrap_focal_estimated,
            }
        )

    with open(poses_json_path, "w", encoding="utf-8") as fh:
        json.dump(pose_records, fh, indent=2)

    logger.info("Localized %d / %d queries.", localized_count, len(queries))
    logger.info("Saved poses JSON to %s", poses_json_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GOTCHA pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    empty_rec = subparsers.add_parser("prepare_empty_rec", help="Create empty_rec from OpenSfM/ODM NVM.")
    empty_rec.add_argument("--project", type=Path, required=True)
    empty_rec.set_defaults(func=run_prepare_empty_rec)

    prepare = subparsers.add_parser("prepare_reference", help="Triangulate reference points from empty_rec.")
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument(
        "--allow-pose-adjustment",
        action="store_true",
        help="Deprecated for this workflow: currently disabled and will raise an explicit error.",
    )
    prepare.set_defaults(func=run_prepare_reference)

    localize = subparsers.add_parser("localize_queries", help="Localize query images against sfm_triangulated.")
    localize.add_argument("--project", type=Path, required=True)
    localize.add_argument("--gt-center", nargs=3, type=float)
    localize.add_argument("--use-fixed-center", action="store_true")
    localize.add_argument("--fallback-exhaustive", action="store_true")
    localize.set_defaults(func=run_localize_queries)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
