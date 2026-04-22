"""scipy.optimize.least_squares (LM) + soft-L1 robust loss 기반 BA.

첫 카메라를 고정하여 gauge ambiguity 를 제거하고, 나머지 카메라 (rvec+tvec, 6D)
와 모든 3D 포인트 (xyz, 3D) 를 동시에 최적화한다.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..bundle_adjustment import Reconstruction
from ..bundle_adjustment_utils import (
    intrinsics_to_K,
    qvec_to_rotmat,
    rotmat_to_qvec,
    rotmat_to_rvec,
    rvec_to_rotmat,
)


def scipy_lm(recon: Reconstruction, max_iterations: int = 50, **_) -> Reconstruction:
    """Levenberg-Marquardt + soft-L1 로 BA 재실행.

    Args:
        recon          : 입력 reconstruction (COLMAP BA 결과)
        max_iterations : LM 최대 iteration (scipy max_nfev)

    Returns:
        in-place 수정된 recon (카메라 포즈 + 3D 포인트 좌표 업데이트)
    """
    recon.summary("initial")

    cam = next(iter(recon.cameras.values()))
    fx, fy, cx, cy = cam["params"][:4]
    _ = intrinsics_to_K(cam["params"])  # validate shape

    img_ids = sorted(recon.images.keys())
    n_cameras = len(img_ids)
    img_id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

    pt_ids = sorted(recon.points3D.keys())
    n_points = len(pt_ids)
    pt_id_to_idx = {pid: i for i, pid in enumerate(pt_ids)}

    # --- 관측 모으기 ---
    cam_indices, pt_indices, observed_2d = [], [], []
    for pid in pt_ids:
        pt = recon.points3D[pid]
        pi = pt_id_to_idx[pid]
        for img_id, kp_idx in pt["track"]:
            if img_id not in img_id_to_idx:
                continue
            ci = img_id_to_idx[img_id]
            x, y, _unused = recon.images[img_id]["points2D"][kp_idx]
            cam_indices.append(ci)
            pt_indices.append(pi)
            observed_2d.append([x, y])
    cam_indices = np.array(cam_indices, dtype=int)
    pt_indices = np.array(pt_indices, dtype=int)
    observed_2d = np.array(observed_2d)
    n_obs = len(cam_indices)
    print(f"  cameras={n_cameras} points={n_points:,} obs={n_obs:,}")

    # --- 초기 파라미터 ---
    camera_params = np.zeros((n_cameras, 6))
    for i, img_id in enumerate(img_ids):
        img = recon.images[img_id]
        camera_params[i, :3] = rotmat_to_rvec(qvec_to_rotmat(img["qvec"]))
        camera_params[i, 3:6] = img["tvec"]

    point_params = np.zeros((n_points, 3))
    for i, pid in enumerate(pt_ids):
        point_params[i] = recon.points3D[pid]["xyz"]

    # --- residuals ---
    def residuals(cam_flat, pt_flat):
        cams = cam_flat.reshape(n_cameras, 6)
        pts = pt_flat.reshape(n_points, 3)
        R_all = np.zeros((n_cameras, 3, 3))
        for ci in range(n_cameras):
            R_all[ci] = rvec_to_rotmat(cams[ci, :3])
        R_obs = R_all[cam_indices]
        t_obs = cams[cam_indices, 3:6]
        pts3d = pts[pt_indices]
        pts_cam = np.einsum("nij,nj->ni", R_obs, pts3d) + t_obs
        z = pts_cam[:, 2]
        z_safe = np.where(z > 0, z, 1e-6)
        px = fx * pts_cam[:, 0] / z_safe + cx
        py = fy * pts_cam[:, 1] / z_safe + cy
        res = np.zeros(n_obs * 2)
        res[0::2] = px - observed_2d[:, 0]
        res[1::2] = py - observed_2d[:, 1]
        behind = z <= 0
        res[0::2][behind] = 1000.0
        res[1::2][behind] = 1000.0
        return res

    # 첫 카메라 고정
    fixed_cam = camera_params[0].copy()
    opt_cams = camera_params[1:].copy()
    x0 = np.concatenate([opt_cams.ravel(), point_params.ravel()])
    n_opt_cams = n_cameras - 1

    def residuals_opt(x):
        cam_flat = np.concatenate([fixed_cam, x[: n_opt_cams * 6]])
        pt_flat = x[n_opt_cams * 6:]
        return residuals(cam_flat, pt_flat)

    # Jacobian sparsity
    n_x = len(x0)
    A = lil_matrix((n_obs * 2, n_x), dtype=int)
    for obs_i in range(n_obs):
        ci = cam_indices[obs_i]
        pi = pt_indices[obs_i]
        if ci > 0:
            opt_ci = ci - 1
            for j in range(6):
                A[obs_i * 2, opt_ci * 6 + j] = 1
                A[obs_i * 2 + 1, opt_ci * 6 + j] = 1
        pt_offset = n_opt_cams * 6 + pi * 3
        for j in range(3):
            A[obs_i * 2, pt_offset + j] = 1
            A[obs_i * 2 + 1, pt_offset + j] = 1

    rmse_before = np.sqrt(np.mean(residuals_opt(x0) ** 2))
    print(f"  RMSE before : {rmse_before:.4f} px")

    t0 = time.time()
    result = least_squares(
        residuals_opt, x0,
        jac_sparsity=A,
        verbose=2,
        x_scale="jac",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_iterations,
    )
    elapsed = time.time() - t0
    rmse_after = np.sqrt(np.mean(result.fun ** 2))
    print(f"  RMSE after  : {rmse_after:.4f} px")
    print(f"  time={elapsed:.1f}s nfev={result.nfev}")

    # --- recon 업데이트 ---
    opt_cams_r = result.x[: n_opt_cams * 6].reshape(n_opt_cams, 6)
    opt_pts_r = result.x[n_opt_cams * 6:].reshape(n_points, 3)

    for i, img_id in enumerate(img_ids):
        if i == 0:
            continue
        c = opt_cams_r[i - 1]
        R = rvec_to_rotmat(c[:3])
        recon.images[img_id]["qvec"] = rotmat_to_qvec(R)
        recon.images[img_id]["tvec"] = c[3:6]

    for i, pid in enumerate(pt_ids):
        recon.points3D[pid]["xyz"] = opt_pts_r[i]

    recon.summary("final")
    return recon
