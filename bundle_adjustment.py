"""
Custom Bundle Adjustment
COLMAP mapper 결과를 읽어서 BA만 재실행합니다.

흐름:
    COLMAP (Step 1-2-3) → cameras.bin, images.bin, points3D.bin
    → 이 스크립트로 BA 재실행 → 결과 덮어쓰기 (또는 별도 폴더)

Usage:
    python bundle_adjustment.py \
        --input_path colmap_ws_custom/bicycle/sparse/0 \
        --output_path colmap_ws_custom/bicycle/sparse_ba/0
"""

import os
import argparse
import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from bundle_adjustment_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
    intrinsics_to_K,
    qvec_to_rotmat,
    rotmat_to_qvec,
    rotmat_to_rvec,
    rvec_to_rotmat,
    reprojection_error,
)


class BundleAdjustment:
    """COLMAP 결과를 읽어서 BA를 재실행하는 클래스."""

    def __init__(self, input_path):
        """COLMAP sparse reconstruction을 로드합니다.

        Args:
            input_path: sparse/0/ 경로 (cameras.bin, images.bin, points3D.bin)
        """
        print(f"Loading COLMAP reconstruction: {input_path}")
        self.cameras = read_cameras_binary(os.path.join(input_path, "cameras.bin"))
        self.images = read_images_binary(os.path.join(input_path, "images.bin"))
        self.points3D = read_points3D_binary(os.path.join(input_path, "points3D.bin"))

        print(f"  Cameras: {len(self.cameras)}")
        print(f"  Images:  {len(self.images)}")
        print(f"  Points:  {len(self.points3D):,}")

        # Camera intrinsics (assume single camera)
        cam = list(self.cameras.values())[0]
        self.cam_params = cam["params"]
        self.K = intrinsics_to_K(self.cam_params)
        fx, fy, cx, cy = self.cam_params[:4]
        print(f"  Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        # Compute initial reprojection error
        self._print_reproj_stats("Initial")

    def _print_reproj_stats(self, label):
        """Reprojection error 통계 출력."""
        errors = []
        for pid, pt in self.points3D.items():
            xyz = pt["xyz"].reshape(1, 3)
            for img_id, kp_idx in pt["track"]:
                if img_id not in self.images:
                    continue
                img = self.images[img_id]
                R = qvec_to_rotmat(img["qvec"])
                rvec = rotmat_to_rvec(R)
                t = img["tvec"]
                x, y, _ = img["points2D"][kp_idx]
                obs = np.array([[x, y]])
                err = reprojection_error(xyz, obs, rvec, t, self.K)
                errors.append(err[0])

        errors = np.array(errors)
        print(f"  {label} reprojection error:")
        print(f"    Mean:   {errors.mean():.4f} px")
        print(f"    Median: {np.median(errors):.4f} px")
        print(f"    Observations: {len(errors):,}")

    def run(self, max_iterations=50):
        """Bundle Adjustment 실행.

        카메라 포즈(R, t)와 3D 포인트(xyz)를 동시 최적화.
        Reprojection error를 최소화.

        Args:
            max_iterations: 최대 LM iteration 수
        """
        print("\nRunning Bundle Adjustment...")

        # --- 파라미터 구성 ---
        img_ids = sorted(self.images.keys())
        n_cameras = len(img_ids)
        img_id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

        pt_ids = sorted(self.points3D.keys())
        n_points = len(pt_ids)
        pt_id_to_idx = {pid: i for i, pid in enumerate(pt_ids)}

        # Observations 수집
        cam_indices = []
        pt_indices = []
        observed_2d = []

        for pid in pt_ids:
            pt = self.points3D[pid]
            pi = pt_id_to_idx[pid]
            for img_id, kp_idx in pt["track"]:
                if img_id not in img_id_to_idx:
                    continue
                ci = img_id_to_idx[img_id]
                x, y, _ = self.images[img_id]["points2D"][kp_idx]
                cam_indices.append(ci)
                pt_indices.append(pi)
                observed_2d.append([x, y])

        cam_indices = np.array(cam_indices, dtype=int)
        pt_indices = np.array(pt_indices, dtype=int)
        observed_2d = np.array(observed_2d)
        n_obs = len(cam_indices)

        print(f"  Cameras: {n_cameras}, Points: {n_points:,}, Observations: {n_obs:,}")

        # Camera params: rvec(3) + tvec(3) per camera
        camera_params = np.zeros((n_cameras, 6))
        for i, img_id in enumerate(img_ids):
            img = self.images[img_id]
            R = qvec_to_rotmat(img["qvec"])
            camera_params[i, :3] = rotmat_to_rvec(R)
            camera_params[i, 3:6] = img["tvec"]

        # Point params: xyz per point
        point_params = np.zeros((n_points, 3))
        for i, pid in enumerate(pt_ids):
            point_params[i] = self.points3D[pid]["xyz"]

        fx, fy, cx, cy = self.cam_params[:4]

        # --- Vectorized residuals ---
        def residuals(cam_flat, pt_flat):
            cams = cam_flat.reshape(n_cameras, 6)
            pts = pt_flat.reshape(n_points, 3)

            # Build rotation matrices
            R_all = np.zeros((n_cameras, 3, 3))
            for ci in range(n_cameras):
                R_all[ci] = rvec_to_rotmat(cams[ci, :3])

            # Gather per observation
            R_obs = R_all[cam_indices]
            t_obs = cams[cam_indices, 3:6]
            pts3d = pts[pt_indices]

            # Transform: pts_cam = R @ pt + t
            pts_cam = np.einsum("nij,nj->ni", R_obs, pts3d) + t_obs

            # Project
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

        # --- Fix first camera, optimize the rest ---
        fixed_cam = camera_params[0].copy()
        opt_cams = camera_params[1:].copy()
        x0 = np.concatenate([opt_cams.ravel(), point_params.ravel()])
        n_opt_cams = n_cameras - 1

        def residuals_opt(x):
            cam_flat = np.concatenate([fixed_cam, x[:n_opt_cams * 6]])
            pt_flat = x[n_opt_cams * 6:]
            return residuals(cam_flat, pt_flat)

        # Sparse Jacobian structure
        n_x = len(x0)
        A = lil_matrix((n_obs * 2, n_x), dtype=int)
        for obs_i in range(n_obs):
            ci = cam_indices[obs_i]
            pi = pt_indices[obs_i]
            # Camera params (skip fixed camera 0)
            if ci > 0:
                opt_ci = ci - 1
                for j in range(6):
                    A[obs_i * 2, opt_ci * 6 + j] = 1
                    A[obs_i * 2 + 1, opt_ci * 6 + j] = 1
            # Point params
            pt_offset = n_opt_cams * 6 + pi * 3
            for j in range(3):
                A[obs_i * 2, pt_offset + j] = 1
                A[obs_i * 2 + 1, pt_offset + j] = 1

        # Initial error
        res_before = residuals_opt(x0)
        rmse_before = np.sqrt(np.mean(res_before ** 2))
        print(f"  RMSE before: {rmse_before:.4f} px")

        # --- Solve ---
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

        res_after = residuals_opt(result.x)
        rmse_after = np.sqrt(np.mean(res_after ** 2))
        print(f"  RMSE after:  {rmse_after:.4f} px")
        print(f"  Time: {elapsed:.1f}s, Iterations: {result.nfev}")

        # --- Update state ---
        opt_cams_result = result.x[:n_opt_cams * 6].reshape(n_opt_cams, 6)
        opt_pts_result = result.x[n_opt_cams * 6:].reshape(n_points, 3)

        # Update images
        for i, img_id in enumerate(img_ids):
            if i == 0:
                continue  # fixed camera
            cam = opt_cams_result[i - 1]
            R = rvec_to_rotmat(cam[:3])
            self.images[img_id]["qvec"] = rotmat_to_qvec(R)
            self.images[img_id]["tvec"] = cam[3:6]

        # Update points3D
        for i, pid in enumerate(pt_ids):
            self.points3D[pid]["xyz"] = opt_pts_result[i]

        # Update reprojection errors
        for pid in pt_ids:
            pt = self.points3D[pid]
            xyz = pt["xyz"].reshape(1, 3)
            errs = []
            for img_id, kp_idx in pt["track"]:
                if img_id not in self.images:
                    continue
                img = self.images[img_id]
                R = qvec_to_rotmat(img["qvec"])
                rvec = rotmat_to_rvec(R)
                x, y, _ = img["points2D"][kp_idx]
                obs = np.array([[x, y]])
                err = reprojection_error(xyz, obs, rvec, img["tvec"], self.K)
                errs.append(err[0])
            self.points3D[pid]["error"] = float(np.mean(errs)) if errs else 0.0

        self._print_reproj_stats("Final")

    def export(self, output_path):
        """결과를 COLMAP binary format으로 저장."""
        os.makedirs(output_path, exist_ok=True)

        write_cameras_binary(self.cameras, os.path.join(output_path, "cameras.bin"))
        write_images_binary(self.images, os.path.join(output_path, "images.bin"))
        write_points3D_binary(self.points3D, os.path.join(output_path, "points3D.bin"))

        print(f"\nExported to {output_path}/")
        print(f"  cameras.bin  ({len(self.cameras)} cameras)")
        print(f"  images.bin   ({len(self.images)} images)")
        print(f"  points3D.bin ({len(self.points3D):,} points)")


def main():
    parser = argparse.ArgumentParser(description="Custom BA on COLMAP reconstruction")
    parser.add_argument("--input_path", required=True,
                        help="Path to COLMAP sparse/0/ (cameras.bin, images.bin, points3D.bin)")
    parser.add_argument("--output_path", required=True,
                        help="Output path for BA results")
    parser.add_argument("--max_iterations", type=int, default=50,
                        help="Max LM iterations")
    args = parser.parse_args()

    ba = BundleAdjustment(args.input_path)
    ba.run(max_iterations=args.max_iterations)
    ba.export(args.output_path)


if __name__ == "__main__":
    main()
