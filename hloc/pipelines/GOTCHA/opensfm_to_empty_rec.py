from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ... import logger
from ...utils.read_write_model import Camera, Image, qvec2rotmat, write_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_unique_basename_map(project: Path, images_dir: Path) -> Dict[str, str]:
    basename_map: Dict[str, str] = {}
    for image_path in list_images(images_dir):
        rel = image_path.relative_to(project).as_posix()
        basename = image_path.name
        if basename in basename_map:
            raise ValueError(
                f"Duplicate basename '{basename}' in images dir. "
                f"Found at both '{basename_map[basename]}' and '{rel}'."
            )
        basename_map[basename] = rel
    if not basename_map:
        raise ValueError(f"No images found in: {images_dir}")
    return basename_map


def get_image_size(image_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


def parse_nvm_entries(nvm_path: Path) -> List[Dict[str, Any]]:
    lines = nvm_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines):
        token = lines[idx].strip()
        if token and token.isdigit():
            break
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"Could not parse NVM image count: {nvm_path}")

    num_images = int(lines[idx].strip())
    idx += 1

    entries = []
    read_images = 0
    while idx < len(lines) and read_images < num_images:
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Invalid NVM camera line: {line}")

        qvec = np.array([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
        qvec = qvec / np.linalg.norm(qvec)
        center = np.array([float(parts[6]), float(parts[7]), float(parts[8])], dtype=float)
        tvec = -qvec2rotmat(qvec) @ center

        entries.append(
            {
                "source_name": parts[0],
                "focal": float(parts[1]),
                "k1": float(parts[9]),
                "qvec": qvec,
                "tvec": tvec,
            }
        )
        read_images += 1

    if read_images != num_images:
        raise ValueError(f"NVM declared {num_images} images but parsed {read_images}.")
    if not entries:
        raise ValueError(f"No image entries found in NVM: {nvm_path}")
    return entries


def resolve_image_name(source_name: str, basename_map: Dict[str, str]) -> str:
    basename = Path(source_name.replace("\\", "/")).name
    if basename not in basename_map:
        raise FileNotFoundError(f"Image '{basename}' from NVM not found in project/images.")
    return basename_map[basename]


def create_empty_rec_from_nvm(
    project: Path, images_dir: Path, output_dir: Path, nvm_path: Path
) -> None:
    if not nvm_path.exists():
        raise FileNotFoundError(f"NVM file not found: {nvm_path}")

    entries = parse_nvm_entries(nvm_path)
    basename_map = build_unique_basename_map(project, images_dir)

    resolved_entries: List[Tuple[Dict[str, Any], str]] = []
    for entry in entries:
        image_name = resolve_image_name(entry["source_name"], basename_map)
        resolved_entries.append((entry, image_name))

    first_image_path = project / resolved_entries[0][1]
    width, height = get_image_size(first_image_path)
    focal = float(np.median([entry["focal"] for entry, _ in resolved_entries]))
    k1 = float(np.median([entry["k1"] for entry, _ in resolved_entries]))

    cameras = {
        1: Camera(
            id=1,
            model="SIMPLE_RADIAL",
            width=width,
            height=height,
            params=np.array([focal, width / 2.0, height / 2.0, k1], dtype=float),
        )
    }

    images = {}
    for image_id, (entry, image_name) in enumerate(resolved_entries, start=1):
        images[image_id] = Image(
            id=image_id,
            qvec=np.asarray(entry["qvec"], dtype=float),
            tvec=np.asarray(entry["tvec"], dtype=float),
            camera_id=1,
            name=image_name,
            xys=np.zeros((0, 2), dtype=float),
            point3D_ids=np.full(0, -1, dtype=int),
        )

    points3d: Dict[int, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3d, str(output_dir), ext=".txt")
    write_model(cameras, images, points3d, str(output_dir), ext=".bin")
    logger.info("Wrote empty_rec to %s", output_dir)

