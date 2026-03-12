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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def to_relative_paths(paths: List[Path], project: Path) -> List[str]:
    return [p.relative_to(project).as_posix() for p in paths]


def sanitize_filename(value: str) -> str:
    sanitized = value.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return sanitized.replace(":", "_")


def resolve_optional_path(path: Path, project: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = project / path
    return candidate if candidate.exists() else path


def parse_center_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if {"x", "y", "z"} <= set(value.keys()):
            return np.array([value["x"], value["y"], value["z"]], dtype=float)
        raise ValueError("Center dict must contain x, y, and z.")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return np.array(value, dtype=float)
    raise ValueError("Center value must be [x, y, z] or {x, y, z}.")


def load_center_map(path: Path) -> Dict[str, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Center map must be a JSON object: {path}")
    return {str(key): parse_center_value(value) for key, value in raw.items()}


def get_query_name(query_path: Path, project: Path) -> str:
    return query_path.relative_to(project).as_posix()


def get_query_cluster(query_path: Path, queries_dir: Path) -> Optional[str]:
    rel_query = query_path.relative_to(queries_dir)
    return rel_query.parts[0] if len(rel_query.parts) > 1 else None


def build_query_gt_centers(
    query_paths: List[Path],
    project: Path,
    queries_dir: Path,
    global_center: Optional[np.ndarray],
    per_image_map: Dict[str, np.ndarray],
    per_cluster_map: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centers: Dict[str, np.ndarray] = {}
    default_center = per_image_map.get("default", per_image_map.get("*"))
    if default_center is None:
        default_center = per_cluster_map.get("default", per_cluster_map.get("*"))

    for query_path in query_paths:
        query_name = get_query_name(query_path, project)
        if global_center is not None:
            centers[query_name] = global_center
            continue

        if query_name in per_image_map:
            centers[query_name] = per_image_map[query_name]
            continue

        cluster = get_query_cluster(query_path, queries_dir)
        if cluster is not None and cluster in per_cluster_map:
            centers[query_name] = per_cluster_map[cluster]
            continue

        if default_center is not None:
            centers[query_name] = default_center

    return centers


def get_num_inliers(ret: Optional[Dict[str, Any]]) -> int:
    if ret is None:
        return 0
    if "inlier_mask" in ret and ret["inlier_mask"] is not None:
        return int(np.sum(np.asarray(ret["inlier_mask"], dtype=bool)))
    if "num_inliers" in ret:
        return int(ret["num_inliers"])
    return 0


def get_simple_radial_params(
    camera: pycolmap.Camera, focal_override: Optional[float] = None
) -> Tuple[float, float, float, float]:
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
    return focal, cx, cy, k


def write_camera_line(
    handle, camera_id: int, width: int, height: int, focal: float, cx: float, cy: float, k: float
) -> None:
    handle.write(
        f"{camera_id} SIMPLE_RADIAL {width} {height} {focal} {cx} {cy} {k}\n"
    )


def write_image_line_from_pose(
    handle, image_id: int, pose: pycolmap.Rigid3d, camera_id: int, image_name: str
) -> None:
    quat_xyzw = np.asarray(pose.rotation.quat, dtype=float)
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    tx, ty, tz = np.asarray(pose.translation, dtype=float)
    handle.write(
        f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n"
    )


def write_image_line_from_rt(
    handle,
    image_id: int,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    camera_id: int,
    image_name: str,
) -> None:
    quat_xyzw = ScipyRotation.from_matrix(rotation_matrix).as_quat()
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    tx, ty, tz = np.asarray(translation_vector, dtype=float).reshape(3)
    handle.write(
        f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n"
    )


def write_results_file(poses: Dict[str, pycolmap.Rigid3d], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for query_name in sorted(poses.keys()):
            pose = poses[query_name]
            quat_xyzw = np.asarray(pose.rotation.quat, dtype=float)
            qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
            tx, ty, tz = np.asarray(pose.translation, dtype=float).reshape(3)
            handle.write(f"{query_name} {qw} {qx} {qy} {qz} {tx} {ty} {tz}\n")


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

    points2d_out = []
    points3d_out = []
    for p2d, pid in zip(keypoints[selected], point3d_ids[selected]):
        pid_int = int(pid)
        if pid_int in model.points3D:
            points2d_out.append(p2d)
            points3d_out.append(model.points3D[pid_int].xyz)

    if not points2d_out:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)
    return np.asarray(points2d_out, dtype=float), np.asarray(points3d_out, dtype=float)


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
    best_ret: Optional[Dict[str, Any]] = None
    best_log: Optional[Dict[str, Any]] = None
    best_key = None
    best_metrics: Dict[str, Any] = {"num_inliers": 0, "center_distance_m": None}

    for _ in range(trials):
        ret, log = pose_from_cluster(
            localizer, query_name, camera, db_ids, features_path, matches_path
        )
        if ret is None:
            continue

        num_inliers = get_num_inliers(ret)
        center = compute_center(ret["cam_from_world"])
        center_distance = (
            float(np.linalg.norm(center - gt_center)) if gt_center is not None else None
        )

        if gt_center is not None:
            current_key = (center_distance, -num_inliers)
        else:
            current_key = (-num_inliers,)

        if best_key is None or current_key < best_key:
            best_key = current_key
            best_ret = ret
            best_log = log
            best_metrics = {
                "num_inliers": int(num_inliers),
                "center_distance_m": center_distance,
                "center": center.tolist(),
            }

    return best_ret, best_log, best_metrics


def run_fallback_exhaustive(
    outputs_dir: Path,
    matcher_conf: Dict[str, Any],
    query_name: str,
    ref_list: List[str],
    localizer: QueryLocalizer,
    camera: pycolmap.Camera,
    all_ref_db_ids: List[int],
    features_path: Path,
    matches_path: Path,
    trials: int,
    gt_center: Optional[np.ndarray],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    fallback_pairs_dir = outputs_dir / "fallback_pairs"
    fallback_pairs_dir.mkdir(parents=True, exist_ok=True)
    fallback_pairs_path = fallback_pairs_dir / f"{sanitize_filename(query_name)}.txt"

    pairs_from_exhaustive.main(
        output=fallback_pairs_path, image_list=[query_name], ref_list=ref_list
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


def build_localizer_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "estimation": {
            "ransac": {
                "max_error": args.ransac_max_error,
                "confidence": args.ransac_confidence,
                "min_inlier_ratio": args.ransac_min_inlier_ratio,
                "min_num_trials": args.ransac_min_num_trials,
                "max_num_trials": args.ransac_max_num_trials,
            },
            "estimate_focal_length": False,
        },
        "refinement": {
            "max_num_iterations": args.refine_max_num_iterations,
            "refine_focal_length": True,
            "refine_extra_params": False,
            "print_summary": False,
        },
    }


def run_prepare_reference(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    images_dir = project / "images"
    empty_rec_dir = project / "empty_rec"
    outputs_dir = project / "outputs"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing reference images directory: {images_dir}")
    if not empty_rec_dir.exists():
        raise FileNotFoundError(f"Missing empty reconstruction directory: {empty_rec_dir}")

    refs = list_images(images_dir)
    if not refs:
        raise ValueError(f"No reference images found under {images_dir}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    features_path = outputs_dir / "features.h5"
    matches_path = outputs_dir / "matches.h5"
    pairs_path = outputs_dir / "pairs-db-poses.txt"
    sfm_dir = outputs_dir / "sfm_triangulated"

    ref_list = to_relative_paths(refs, project)
    logger.info("Preparing reference model with %d reference images.", len(ref_list))

    feature_conf = copy.deepcopy(extract_features.confs[args.local_feature_conf])
    if "model" in feature_conf and "max_keypoints" in feature_conf["model"]:
        feature_conf["model"]["max_keypoints"] = args.max_keypoints
    extract_features.main(
        feature_conf,
        project,
        image_list=ref_list,
        feature_path=features_path,
        overwrite=args.overwrite_features,
    )

    pairs_from_poses.main(
        empty_rec_dir,
        pairs_path,
        num_matched=args.num_pose_pairs,
        rotation_threshold=args.rotation_threshold,
    )

    matcher_conf = match_features.confs[args.matcher_conf]
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=args.overwrite_matches,
    )

    triangulation.main(
        sfm_dir=sfm_dir,
        reference_model=empty_rec_dir,
        image_dir=project,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        skip_geometric_verification=args.skip_geometric_verification,
        estimate_two_view_geometries=args.estimate_two_view_geometries,
        verbose=args.verbose,
    )
    logger.info("Reference triangulation complete: %s", sfm_dir)


def run_localize_queries(args: argparse.Namespace) -> None:
    project = args.project.resolve()
    images_dir = project / "images"
    queries_dir = project / "queries"
    outputs_dir = project / "outputs"
    sfm_dir = outputs_dir / "sfm_triangulated"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing reference images directory: {images_dir}")
    if not queries_dir.exists():
        raise FileNotFoundError(f"Missing query images directory: {queries_dir}")
    if not sfm_dir.exists():
        raise FileNotFoundError(
            f"Missing triangulated reference model: {sfm_dir}. "
            "Run prepare_reference first."
        )

    refs = list_images(images_dir)
    queries = list_images(queries_dir)
    if not refs:
        raise ValueError(f"No reference images found under {images_dir}")
    if not queries:
        raise ValueError(f"No query images found under {queries_dir}")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    features_path = outputs_dir / "features.h5"
    matches_path = outputs_dir / "matches.h5"
    global_desc_path = outputs_dir / "global-features.h5"
    pairs_path = outputs_dir / "pairs-query-ref.txt"
    results_path = outputs_dir / "results.txt"
    extra_cams_path = outputs_dir / "cameras_extra.txt"
    extra_images_path = outputs_dir / "images_extra.txt"
    summary_path = outputs_dir / "query_summary.json"

    ref_list = to_relative_paths(refs, project)
    query_list = to_relative_paths(queries, project)

    logger.info("Found %d references and %d queries.", len(ref_list), len(query_list))

    feature_conf = copy.deepcopy(extract_features.confs[args.local_feature_conf])
    if "model" in feature_conf and "max_keypoints" in feature_conf["model"]:
        feature_conf["model"]["max_keypoints"] = args.max_keypoints

    local_feature_images = query_list
    if args.ensure_ref_features or not features_path.exists():
        local_feature_images = ref_list + query_list

    extract_features.main(
        feature_conf,
        project,
        image_list=local_feature_images,
        feature_path=features_path,
        overwrite=args.overwrite_features,
    )

    if args.manual_pairs is not None:
        pairs_path = resolve_optional_path(args.manual_pairs, project)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Manual pairs file not found: {pairs_path}")
        retrieval = parse_retrieval(pairs_path)
        logger.info("Using manual query-reference pairs from %s", pairs_path)
    else:
        global_conf = copy.deepcopy(extract_features.confs[args.global_feature_conf])
        extract_features.main(
            global_conf,
            project,
            image_list=query_list + ref_list,
            feature_path=global_desc_path,
            overwrite=args.overwrite_global_descriptors,
        )
        pairs_from_retrieval.main(
            descriptors=global_desc_path,
            output=pairs_path,
            num_matched=args.num_matched,
            query_list=query_list,
            db_list=ref_list,
        )
        retrieval = parse_retrieval(pairs_path)
        logger.info("Generated retrieval pairs in %s", pairs_path)

    matcher_conf = match_features.confs[args.matcher_conf]
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=args.overwrite_matches,
    )

    model = pycolmap.Reconstruction(sfm_dir)
    localizer = QueryLocalizer(model, build_localizer_config(args))

    name_to_id = {image.name: image_id for image_id, image in model.images.items()}
    all_ref_db_ids = [name_to_id[name] for name in ref_list if name in name_to_id]
    if not all_ref_db_ids:
        raise ValueError("No reference image names from project/images were found in sfm model.")

    gt_center_global = np.array(args.gt_center, dtype=float) if args.gt_center else None
    gt_center_image_map: Dict[str, np.ndarray] = {}
    gt_center_cluster_map: Dict[str, np.ndarray] = {}
    if args.gt_center_image_map is not None:
        gt_image_path = resolve_optional_path(args.gt_center_image_map, project)
        if not gt_image_path.exists():
            raise FileNotFoundError(f"GT per-image center map not found: {gt_image_path}")
        gt_center_image_map = load_center_map(gt_image_path)
    if args.gt_center_cluster_map is not None:
        gt_cluster_path = resolve_optional_path(args.gt_center_cluster_map, project)
        if not gt_cluster_path.exists():
            raise FileNotFoundError(
                f"GT per-cluster center map not found: {gt_cluster_path}"
            )
        gt_center_cluster_map = load_center_map(gt_cluster_path)
    if args.gt_center_map is not None:
        legacy_path = resolve_optional_path(args.gt_center_map, project)
        if not legacy_path.exists():
            raise FileNotFoundError(f"GT center map not found: {legacy_path}")
        legacy_map = load_center_map(legacy_path)
        for key, center in legacy_map.items():
            if key.startswith("queries/") or "/" in key:
                gt_center_image_map[key] = center
            elif key in {"default", "*"}:
                gt_center_image_map[key] = center
            else:
                gt_center_cluster_map[key] = center

    query_gt_centers = build_query_gt_centers(
        query_paths=queries,
        project=project,
        queries_dir=queries_dir,
        global_center=gt_center_global,
        per_image_map=gt_center_image_map,
        per_cluster_map=gt_center_cluster_map,
    )

    localized_raw_poses: Dict[str, pycolmap.Rigid3d] = {}
    summary: List[Dict[str, Any]] = []

    cam_id = 1
    image_id = 1
    with open(extra_cams_path, "w", encoding="utf-8") as cam_fh, open(
        extra_images_path, "w", encoding="utf-8"
    ) as img_fh:
        for query_path in queries:
            query_name = get_query_name(query_path, project)
            query_gt_center = query_gt_centers.get(query_name)

            camera = pycolmap.infer_camera_from_image(query_path)
            if args.initial_focal_length is not None:
                camera.focal_length = float(args.initial_focal_length)

            db_names = retrieval.get(query_name, [])
            db_ids = [name_to_id[n] for n in db_names if n in name_to_id]

            primary_ret = None
            primary_log = None
            primary_metrics = {"num_inliers": 0, "center_distance_m": None}
            if db_ids:
                primary_ret, primary_log, primary_metrics = localize_with_trials(
                    localizer=localizer,
                    query_name=query_name,
                    camera=camera,
                    db_ids=db_ids,
                    features_path=features_path,
                    matches_path=matches_path,
                    trials=args.trials,
                    gt_center=query_gt_center,
                )

            selected_ret = primary_ret
            selected_log = primary_log
            selected_metrics = primary_metrics
            selected_source = "primary"

            fallback_attempted = False
            fallback_used = False
            fallback_success = False
            fallback_reason = None

            needs_fallback = (
                args.fallback_exhaustive
                and (selected_ret is None or get_num_inliers(selected_ret) < args.fallback_min_inliers)
            )
            if needs_fallback:
                fallback_attempted = True
                if not all_ref_db_ids:
                    fallback_reason = "no_reference_db_ids"
                else:
                    fallback_ret, fallback_log, fallback_metrics = run_fallback_exhaustive(
                        outputs_dir=outputs_dir,
                        matcher_conf=matcher_conf,
                        query_name=query_name,
                        ref_list=ref_list,
                        localizer=localizer,
                        camera=camera,
                        all_ref_db_ids=all_ref_db_ids,
                        features_path=features_path,
                        matches_path=matches_path,
                        trials=args.fallback_trials,
                        gt_center=query_gt_center,
                    )
                    if fallback_ret is not None:
                        selected_ret = fallback_ret
                        selected_log = fallback_log
                        selected_metrics = fallback_metrics
                        selected_source = "fallback"
                        fallback_used = True
                        fallback_success = True
                    else:
                        fallback_reason = "no_fallback_solution"

            if selected_ret is None or selected_log is None:
                logger.warning("Failed to localize %s", query_name)
                summary.append(
                    {
                        "query": query_name,
                        "status": "failed",
                        "source": "none",
                        "inliers": 0,
                        "center_distance_m": None,
                        "fallback_attempted": fallback_attempted,
                        "fallback_used": fallback_used,
                        "fallback_success": fallback_success,
                        "fallback_reason": fallback_reason,
                        "reprojection_stats": None,
                    }
                )
                continue

            pose = selected_ret["cam_from_world"]
            localized_raw_poses[query_name] = pose
            inliers = get_num_inliers(selected_ret)
            center = compute_center(pose)
            center_distance_m = (
                float(np.linalg.norm(center - query_gt_center))
                if query_gt_center is not None
                else None
            )

            raw_camera = selected_ret.get("camera", camera)
            raw_f, raw_cx, raw_cy, raw_k = get_simple_radial_params(raw_camera)
            raw_width, raw_height = int(raw_camera.width), int(raw_camera.height)
            raw_cam_id = cam_id
            raw_img_id = image_id
            cam_id += 1
            image_id += 1
            write_camera_line(
                cam_fh, raw_cam_id, raw_width, raw_height, raw_f, raw_cx, raw_cy, raw_k
            )
            write_image_line_from_pose(
                img_fh, raw_img_id, pose, raw_cam_id, f"{query_name}_raw"
            )

            fixed_result = None
            if args.use_fixed_center and query_gt_center is not None:
                inlier_mask = np.asarray(selected_ret.get("inlier_mask", []), dtype=bool)
                points2d_inliers, points3d_inliers = extract_inlier_correspondences(
                    model=model, log=selected_log, inlier_mask=inlier_mask
                )
                if len(points2d_inliers) >= 4:
                    fixed_result = refine_pose_fixed_center(
                        points2d=points2d_inliers,
                        points3d=points3d_inliers,
                        camera=raw_camera,
                        fixed_center=query_gt_center,
                        initial_rotation=pose.rotation.matrix(),
                        initial_translation=np.asarray(pose.translation, dtype=float),
                        optimize_focal=not args.keep_focal_fixed,
                        max_nfev=args.fixed_center_max_nfev,
                    )

                    if fixed_result["success"]:
                        fixed_f = (
                            float(raw_f)
                            if args.keep_focal_fixed
                            else float(fixed_result["focal_length"])
                        )
                        fixed_cam_id = cam_id
                        fixed_img_id = image_id
                        cam_id += 1
                        image_id += 1
                        write_camera_line(
                            cam_fh,
                            fixed_cam_id,
                            raw_width,
                            raw_height,
                            fixed_f,
                            raw_cx,
                            raw_cy,
                            raw_k,
                        )
                        write_image_line_from_rt(
                            img_fh,
                            fixed_img_id,
                            fixed_result["rotation_matrix"],
                            fixed_result["translation_vector"],
                            fixed_cam_id,
                            f"{query_name}_fixed",
                        )

            reprojection_stats = None
            if fixed_result is not None:
                initial = float(fixed_result["reproj_error_initial"])
                optimized = float(fixed_result["mean_reprojection_error"])
                change_pct = (
                    100.0 * (optimized - initial) / initial
                    if np.isfinite(initial) and initial > 0.0
                    else None
                )
                reprojection_stats = {
                    "fixed_success": bool(fixed_result["success"]),
                    "initial": initial,
                    "optimized": optimized,
                    "change_pct": change_pct,
                }

            summary.append(
                {
                    "query": query_name,
                    "status": "localized",
                    "source": selected_source,
                    "inliers": int(inliers),
                    "center": center.tolist(),
                    "center_distance_m": center_distance_m,
                    "gt_center": query_gt_center.tolist()
                    if query_gt_center is not None
                    else None,
                    "fallback_attempted": fallback_attempted,
                    "fallback_used": fallback_used,
                    "fallback_success": fallback_success,
                    "fallback_reason": fallback_reason,
                    "reprojection_stats": reprojection_stats,
                }
            )

    write_results_file(localized_raw_poses, results_path)
    with open(summary_path, "w", encoding="utf-8") as summary_fh:
        json.dump(summary, summary_fh, indent=2)

    logger.info("Localized %d / %d queries.", len(localized_raw_poses), len(queries))
    logger.info("Saved poses: %s", results_path)
    logger.info("Saved extra COLMAP exports: %s and %s", extra_cams_path, extra_images_path)
    logger.info("Saved summary JSON: %s", summary_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GOTCHA pipeline for known-pose SfM and query localization.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare_reference",
        help="Triangulate a sparse reference model from known poses in project/empty_rec.",
    )
    prepare.add_argument("--project", type=Path, required=True)
    prepare.add_argument("--local-feature-conf", default="superpoint_max")
    prepare.add_argument("--matcher-conf", default="superpoint+lightglue")
    prepare.add_argument("--max-keypoints", type=int, default=10000)
    prepare.add_argument("--num-pose-pairs", type=int, default=35)
    prepare.add_argument("--rotation-threshold", type=float, default=120.0)
    prepare.add_argument("--overwrite-features", action="store_true")
    prepare.add_argument("--overwrite-matches", action="store_true")
    prepare.add_argument("--skip-geometric-verification", action="store_true")
    prepare.add_argument("--estimate-two-view-geometries", action="store_true")
    prepare.add_argument("--verbose", action="store_true")
    prepare.set_defaults(func=run_prepare_reference)

    localize = subparsers.add_parser(
        "localize_queries",
        help="Localize query images with optional fixed-center refinement.",
    )
    localize.add_argument("--project", type=Path, required=True)
    localize.add_argument("--local-feature-conf", default="superpoint_max")
    localize.add_argument("--global-feature-conf", default="netvlad")
    localize.add_argument("--matcher-conf", default="superpoint+lightglue")
    localize.add_argument("--max-keypoints", type=int, default=10000)
    localize.add_argument("--num-matched", type=int, default=35)
    localize.add_argument("--manual-pairs", type=Path)
    localize.add_argument("--trials", type=int, default=5)
    localize.add_argument("--ransac-max-error", type=float, default=12.0)
    localize.add_argument("--ransac-confidence", type=float, default=0.99)
    localize.add_argument("--ransac-min-inlier-ratio", type=float, default=0.3)
    localize.add_argument("--ransac-min-num-trials", type=int, default=100)
    localize.add_argument("--ransac-max-num-trials", type=int, default=20000)
    localize.add_argument("--refine-max-num-iterations", type=int, default=10000)
    localize.add_argument("--fallback-exhaustive", action="store_true")
    localize.add_argument("--fallback-trials", type=int, default=10)
    localize.add_argument("--fallback-min-inliers", type=int, default=15)
    localize.add_argument("--gt-center", nargs=3, type=float)
    localize.add_argument("--gt-center-image-map", type=Path)
    localize.add_argument("--gt-center-cluster-map", type=Path)
    localize.add_argument("--gt-center-map", type=Path)
    localize.add_argument("--use-fixed-center", action="store_true")
    localize.add_argument("--keep-focal-fixed", action="store_true")
    localize.add_argument("--fixed-center-max-nfev", type=int, default=200)
    localize.add_argument("--initial-focal-length", type=float)
    localize.add_argument("--ensure-ref-features", action="store_true")
    localize.add_argument("--overwrite-features", action="store_true")
    localize.add_argument("--overwrite-matches", action="store_true")
    localize.add_argument("--overwrite-global-descriptors", action="store_true")
    localize.set_defaults(func=run_localize_queries)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
