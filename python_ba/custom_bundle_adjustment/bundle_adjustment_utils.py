"""
Bundle Adjustment Utilities
- COLMAP database reader (features, matches)
- COLMAP binary file writer (cameras.bin, images.bin, points3D.bin)
- Geometric utilities (rotation, projection, triangulation)
"""

import sqlite3
import struct
import numpy as np
import cv2
import os


# ============================================================
# 1. COLMAP Database Reader
# ============================================================

MAX_IMAGE_ID = 2147483647  # 2^31 - 1


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return image_id1, image_id2


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


class COLMAPDatabase:
    """COLMAP database.db reader"""

    def __init__(self, database_path):
        self.conn = sqlite3.connect(database_path)

    def close(self):
        self.conn.close()

    def read_cameras(self):
        """Returns dict: camera_id -> {model_id, width, height, params}"""
        cameras = {}
        cursor = self.conn.execute(
            "SELECT camera_id, model, width, height, params FROM cameras"
        )
        for cam_id, model, width, height, params_blob in cursor:
            num_params = len(params_blob) // 8
            params = np.array(struct.unpack(f"<{num_params}d", params_blob))
            cameras[cam_id] = {
                "model_id": model,
                "width": width,
                "height": height,
                "params": params,
            }
        return cameras

    def read_images(self):
        """Returns dict: image_id -> {name, camera_id}"""
        images = {}
        cursor = self.conn.execute("SELECT image_id, name, camera_id FROM images")
        for image_id, name, camera_id in cursor:
            images[image_id] = {"name": name, "camera_id": camera_id}
        return images

    def read_keypoints(self, image_id):
        """Returns (N, 2) array of keypoint (x, y) coordinates"""
        cursor = self.conn.execute(
            "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
            (image_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return np.zeros((0, 2))
        rows, cols, data = row
        kps = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
        return kps[:, :2].copy()

    def read_all_keypoints(self):
        """Returns dict: image_id -> (N, 2) keypoints"""
        keypoints = {}
        cursor = self.conn.execute("SELECT image_id, rows, cols, data FROM keypoints")
        for image_id, rows, cols, data in cursor:
            kps = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
            keypoints[image_id] = kps[:, :2].copy()
        return keypoints

    def read_two_view_geometries(self):
        """Returns dict: (id1, id2) -> inlier_matches (N, 2) uint32

        two_view_geometries contains geometrically verified matches.
        """
        matches = {}
        cursor = self.conn.execute(
            "SELECT pair_id, rows, cols, data, config FROM two_view_geometries "
            "WHERE rows > 0"
        )
        for pair_id, rows, cols, data, config in cursor:
            if rows == 0 or len(data) == 0:
                continue
            id1, id2 = pair_id_to_image_ids(pair_id)
            inlier_matches = np.frombuffer(data, dtype=np.uint32).reshape((rows, cols))
            matches[(id1, id2)] = inlier_matches
        return matches


# ============================================================
# 2. Rotation Utilities
# ============================================================

def rotmat_to_qvec(R):
    """Rotation matrix (3x3) -> quaternion (w, x, y, z)"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def qvec_to_rotmat(qvec):
    """Quaternion (w, x, y, z) -> rotation matrix (3x3)"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def rvec_to_rotmat(rvec):
    """Rodrigues vector (3,) -> rotation matrix (3x3)"""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def rotmat_to_rvec(R):
    """Rotation matrix (3x3) -> Rodrigues vector (3,)"""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


# ============================================================
# 3. Projection
# ============================================================

def project(points_3d, rvec, tvec, K):
    """Project 3D points to 2D.

    Args:
        points_3d: (N, 3) world points
        rvec: (3,) Rodrigues rotation
        tvec: (3,) translation
        K: (3, 3) intrinsic matrix

    Returns:
        (N, 2) projected 2D points
    """
    R = rvec_to_rotmat(rvec)
    pts_cam = (R @ points_3d.T).T + tvec
    pts_proj = (K @ pts_cam.T).T
    return pts_proj[:, :2] / pts_proj[:, 2:3]


def reprojection_error(points_3d, points_2d, rvec, tvec, K):
    """Per-point reprojection error in pixels."""
    projected = project(points_3d, rvec, tvec, K)
    return np.linalg.norm(projected - points_2d, axis=1)


def intrinsics_to_K(params):
    """PINHOLE params [fx, fy, cx, cy] -> 3x3 matrix"""
    fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# ============================================================
# 4. Triangulation
# ============================================================

def triangulate_points(K1, R1, t1, K2, R2, t2, pts1, pts2):
    """Triangulate 3D points from two views.

    Returns:
        (N, 3) 3D points in world frame
    """
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64),
                                  pts2.T.astype(np.float64))
    return (pts4d[:3] / pts4d[3:4]).T


def filter_triangulated(points_3d, R1, t1, R2, t2,
                         K1=None, K2=None, pts1=None, pts2=None,
                         max_reproj=4.0, min_angle_deg=1.0):
    """Filter by cheirality, angle, and reprojection error.

    Returns:
        (N,) boolean mask
    """
    N = len(points_3d)
    mask = np.ones(N, dtype=bool)

    # Cheirality: z > 0 in both cameras
    pts_cam1 = (R1 @ points_3d.T).T + t1
    pts_cam2 = (R2 @ points_3d.T).T + t2
    mask &= pts_cam1[:, 2] > 0
    mask &= pts_cam2[:, 2] > 0

    # Triangulation angle
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    ray1 = points_3d - C1
    ray2 = points_3d - C2
    ray1 /= np.linalg.norm(ray1, axis=1, keepdims=True) + 1e-10
    ray2 /= np.linalg.norm(ray2, axis=1, keepdims=True) + 1e-10
    cos_a = np.clip(np.sum(ray1 * ray2, axis=1), -1, 1)
    mask &= np.degrees(np.arccos(cos_a)) > min_angle_deg

    # Reprojection error
    if K1 is not None and pts1 is not None:
        rvec1 = rotmat_to_rvec(R1)
        rvec2 = rotmat_to_rvec(R2)
        err1 = reprojection_error(points_3d, pts1, rvec1, t1, K1)
        err2 = reprojection_error(points_3d, pts2, rvec2, t2, K2)
        mask &= err1 < max_reproj
        mask &= err2 < max_reproj

    return mask


# ============================================================
# 5. COLMAP Binary Readers
# ============================================================

CAMERA_MODEL_NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}


def read_cameras_binary(path):
    """Read cameras.bin -> dict of cam_id -> {model_id, width, height, params}"""
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id = struct.unpack("<ii", f.read(8))
            width, height = struct.unpack("<QQ", f.read(16))
            n_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[cam_id] = {"model_id": model_id, "width": width, "height": height, "params": params}
    return cameras


def read_images_binary(path):
    """Read images.bin -> dict of img_id -> {qvec, tvec, camera_id, name, points2D}
    points2D: list of (x, y, point3D_id)
    """
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            props = struct.unpack("<idddddddi", f.read(64))
            img_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            num_pts = struct.unpack("<Q", f.read(8))[0]
            points2D = []
            for _ in range(num_pts):
                x, y, p3d_id = struct.unpack("<ddq", f.read(24))
                points2D.append((x, y, p3d_id))
            images[img_id] = {
                "qvec": qvec, "tvec": tvec, "camera_id": camera_id,
                "name": name, "points2D": points2D,
            }
    return images


def read_points3D_binary(path):
    """Read points3D.bin -> dict of pid -> {xyz, rgb, error, track}
    track: list of (image_id, point2D_idx)
    """
    points3D = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            props = struct.unpack("<QdddBBBd", f.read(43))
            pid = props[0]
            xyz = np.array(props[1:4])
            rgb = list(props[4:7])
            error = props[7]
            track_len = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_len):
                img_id, pt2d_idx = struct.unpack("<ii", f.read(8))
                track.append((img_id, pt2d_idx))
            points3D[pid] = {"xyz": xyz, "rgb": rgb, "error": error, "track": track}
    return points3D


# ============================================================
# 6. COLMAP Binary Writers
# ============================================================

def write_cameras_binary(cameras, path):
    """Write cameras.bin.

    cameras: dict of cam_id -> {model_id, width, height, params}
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            f.write(struct.pack("<ii", cam_id, cam["model_id"]))
            f.write(struct.pack("<QQ", cam["width"], cam["height"]))
            for p in cam["params"]:
                f.write(struct.pack("<d", p))


def write_images_binary(images, path):
    """Write images.bin.

    images: dict of img_id -> {qvec, tvec, camera_id, name, points2D}
        points2D: list of (x, y, point3D_id)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img_id, img in images.items():
            f.write(struct.pack("<i", img_id))
            for q in img["qvec"]:
                f.write(struct.pack("<d", q))
            for t in img["tvec"]:
                f.write(struct.pack("<d", t))
            f.write(struct.pack("<i", img["camera_id"]))
            f.write(img["name"].encode("utf-8"))
            f.write(b"\x00")
            pts2d = img.get("points2D", [])
            f.write(struct.pack("<Q", len(pts2d)))
            for x, y, p3d_id in pts2d:
                f.write(struct.pack("<ddq", x, y, p3d_id))


def write_points3D_binary(points3D, path):
    """Write points3D.bin.

    points3D: dict of pid -> {xyz, rgb, error, track}
        track: list of (image_id, point2D_idx)
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points3D)))
        for pid, pt in points3D.items():
            f.write(struct.pack("<Q", pid))
            for v in pt["xyz"]:
                f.write(struct.pack("<d", v))
            rgb = pt.get("rgb", [128, 128, 128])
            for c in rgb[:3]:
                f.write(struct.pack("<B", int(c)))
            f.write(struct.pack("<d", pt.get("error", 0.0)))
            track = pt.get("track", [])
            f.write(struct.pack("<Q", len(track)))
            for img_id, pt2d_idx in track:
                f.write(struct.pack("<ii", img_id, pt2d_idx))
