"""COLMAP sparse reconstruction I/O 라이브러리 (최적화 없음 / pass-through).

COLMAP 의 `sparse/0/{cameras,images,points3D}.bin` 은 이미 COLMAP 내부 BA 를 거친
결과이므로, 기본 동작은 "읽어서 그대로 내보내기" (pass-through) 이다.

추가 최적화를 하려면 `methods/{namespace}.py` 에 다음 시그니처의 함수를 정의한다:

    def {namespace}(recon: Reconstruction, **kwargs) -> Reconstruction
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from .bundle_adjustment_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
    intrinsics_to_K,
    qvec_to_rotmat,
    rotmat_to_rvec,
    reprojection_error,
)


PathLike = Union[str, Path]


@dataclass
class Reconstruction:
    """COLMAP sparse reconstruction 데이터 컨테이너.

    Attributes:
        cameras : {camera_id: {id, model, width, height, params}}
        images  : {image_id:  {id, qvec, tvec, camera_id, name, points2D, point3D_ids}}
        points3D: {point3D_id:{id, xyz, rgb, error, track}}
    """
    cameras: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    images: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    points3D: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # -------------------------------------------------- I/O
    @classmethod
    def load(cls, sparse_dir: PathLike) -> "Reconstruction":
        """COLMAP sparse/0/ 디렉토리에서 로드."""
        p = Path(sparse_dir)
        return cls(
            cameras=read_cameras_binary(str(p / "cameras.bin")),
            images=read_images_binary(str(p / "images.bin")),
            points3D=read_points3D_binary(str(p / "points3D.bin")),
        )

    def save(self, sparse_dir: PathLike) -> None:
        """COLMAP sparse/0/ 디렉토리에 저장."""
        p = Path(sparse_dir)
        p.mkdir(parents=True, exist_ok=True)
        write_cameras_binary(self.cameras, str(p / "cameras.bin"))
        write_images_binary(self.images, str(p / "images.bin"))
        write_points3D_binary(self.points3D, str(p / "points3D.bin"))

    # -------------------------------------------------- 진단
    def reprojection_errors(self) -> np.ndarray:
        """관측 per reprojection error (px) 배열."""
        cam = next(iter(self.cameras.values()))
        K = intrinsics_to_K(cam["params"])
        errs = []
        for pt in self.points3D.values():
            xyz = pt["xyz"].reshape(1, 3)
            for img_id, kp_idx in pt["track"]:
                if img_id not in self.images:
                    continue
                img = self.images[img_id]
                rvec = rotmat_to_rvec(qvec_to_rotmat(img["qvec"]))
                x, y, _ = img["points2D"][kp_idx]
                errs.append(
                    reprojection_error(xyz, np.array([[x, y]]), rvec, img["tvec"], K)[0]
                )
        return np.array(errs)

    def summary(self, label: str = "") -> None:
        e = self.reprojection_errors()
        prefix = f"[{label}] " if label else ""
        print(
            f"{prefix}cameras={len(self.cameras)} "
            f"images={len(self.images)} "
            f"points={len(self.points3D):,} "
            f"obs={len(e):,} "
            f"reproj mean={e.mean():.4f}px median={np.median(e):.4f}px"
        )


def passthrough(input_path: PathLike, output_path: PathLike) -> None:
    """최적화 없이 sparse/0/ 를 읽어 그대로 내보낸다.

    method 이름이 지정되지 않은 경우 run_sba.py 가 호출하는 기본 경로.
    """
    recon = Reconstruction.load(input_path)
    recon.save(output_path)
