from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# TODO: Deprecate this custom solver once pycolmap exposes camera-center-prior
# absolute-pose refinement in a released COLMAP/pycolmap version.


def compute_center(cam_from_world: Any) -> np.ndarray:
    """Compute camera center in world coordinates from a pycolmap pose."""
    return -cam_from_world.rotation.inverse().matrix() @ np.asarray(
        cam_from_world.translation, dtype=float
    )


def _project_points(
    points3d: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    points_cam = (rotation_matrix @ points3d.T).T + translation_vector.reshape(1, 3)
    depth = points_cam[:, 2]
    valid = depth > 1e-9

    proj = np.full((len(points3d), 2), np.nan, dtype=float)
    proj[valid, 0] = focal_length * (points_cam[valid, 0] / depth[valid]) + cx
    proj[valid, 1] = focal_length * (points_cam[valid, 1] / depth[valid]) + cy
    return proj, valid


def _mean_reprojection_error(
    points2d: np.ndarray,
    points3d: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vector: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
) -> float:
    proj, valid = _project_points(
        points3d, rotation_matrix, translation_vector, focal_length, cx, cy
    )
    if not np.any(valid):
        return float("inf")
    errors = np.linalg.norm(proj[valid] - points2d[valid], axis=1)
    return float(np.mean(errors))


def refine_pose_fixed_center(
    points2d: np.ndarray,
    points3d: np.ndarray,
    camera: Any,
    fixed_center: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: Optional[np.ndarray] = None,
    optimize_focal: bool = True,
    max_nfev: int = 200,
) -> Dict[str, Any]:
    """Refine orientation (and optionally focal length) with fixed camera center."""
    del initial_translation

    points2d = np.asarray(points2d, dtype=float)
    points3d = np.asarray(points3d, dtype=float)
    fixed_center = np.asarray(fixed_center, dtype=float).reshape(3)
    initial_rotation = np.asarray(initial_rotation, dtype=float).reshape(3, 3)

    if len(points2d) < 4 or len(points3d) < 4:
        t0 = -initial_rotation @ fixed_center
        return {
            "success": False,
            "message": "Need at least 4 inlier correspondences.",
            "rotation_matrix": initial_rotation,
            "translation_vector": t0,
            "focal_length": float(getattr(camera, "focal_length", camera.params[0])),
            "reproj_error_initial": float("inf"),
            "mean_reprojection_error": float("inf"),
            "nfev": 0,
        }

    cx = float(getattr(camera, "principal_point_x", camera.width / 2.0))
    cy = float(getattr(camera, "principal_point_y", camera.height / 2.0))
    focal0 = float(getattr(camera, "focal_length", camera.params[0]))

    rot0 = Rotation.from_matrix(initial_rotation).as_rotvec()
    t0 = -initial_rotation @ fixed_center
    reproj_init = _mean_reprojection_error(
        points2d, points3d, initial_rotation, t0, focal0, cx, cy
    )

    def residual_fn(params: np.ndarray) -> np.ndarray:
        rotvec = params[:3]
        focal = params[3] if optimize_focal else focal0
        rotation = Rotation.from_rotvec(rotvec).as_matrix()
        translation = -rotation @ fixed_center

        proj, valid = _project_points(points3d, rotation, translation, focal, cx, cy)
        residuals = proj - points2d
        residuals[~valid, :] = 1e3
        return residuals.reshape(-1)

    x0 = np.concatenate([rot0, [focal0]]) if optimize_focal else rot0

    if optimize_focal:
        image_diag = float(np.hypot(camera.width, camera.height))
        lower = max(0.1 * image_diag, 1e-3)
        upper = max(20.0 * image_diag, lower * 2.0)
        x0[3] = float(np.clip(x0[3], lower, upper))
        result = least_squares(
            residual_fn,
            x0,
            bounds=(
                np.array([-np.inf, -np.inf, -np.inf, lower], dtype=float),
                np.array([np.inf, np.inf, np.inf, upper], dtype=float),
            ),
            loss="soft_l1",
            max_nfev=max_nfev,
        )
    else:
        result = least_squares(
            residual_fn,
            x0,
            loss="soft_l1",
            max_nfev=max_nfev,
        )

    rotvec_opt = result.x[:3]
    focal_opt = float(result.x[3] if optimize_focal else focal0)
    rotation_opt = Rotation.from_rotvec(rotvec_opt).as_matrix()
    translation_opt = -rotation_opt @ fixed_center
    reproj_opt = _mean_reprojection_error(
        points2d, points3d, rotation_opt, translation_opt, focal_opt, cx, cy
    )

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "rotation_matrix": rotation_opt,
        "translation_vector": translation_opt,
        "focal_length": focal_opt,
        "reproj_error_initial": reproj_init,
        "mean_reprojection_error": reproj_opt,
        "nfev": int(result.nfev),
    }
