import sys
import os
import csv
import time
import cv2
import numpy as np
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGridLayout, QGroupBox,
    QSizePolicy, QMessageBox, QProgressBar
)

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

FEATURE_NAMES = [
    "roi_forehead_score",
    "roi_left_cheek_score",
    "roi_right_cheek_score",
    "roi_nose_score",
    "global_skin_score",
    "texture_score",
    "global_smoothness",
    "global_detail_loss",
    "global_uniformity",
    "global_edge_suspicion",
    "eye_ratio",
    "eye_height_ratio",
    "jaw_cheek_ratio",
    "face_aspect_ratio",
    "frontality",
    "big_eye_score",
    "slim_face_score",
    "geo_score",
    "temporal_flicker",
    "temporal_warp",
    "temporal_jump",
    "temporal_score",
    "heuristic_score",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32)
]

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cv_to_pixmap(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

def blend_mask(img, mask, color_bgr, alpha=0.30):
    idx = mask > 0
    if not np.any(idx):
        return
    color = np.array(color_bgr, dtype=np.float32)
    img[idx] = (img[idx].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)

class LogisticBeautyModel:
    def __init__(self):
        self.feature_names = FEATURE_NAMES[:]
        self.mean = None
        self.std = None
        self.w = None
        self.b = 0.0

    def fit(self, X, y, epochs=2200, lr=0.05, l2=1e-3):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.mean) / self.std

        self.w = np.zeros((X.shape[1], 1), dtype=np.float32)
        self.b = 0.0

        for _ in range(epochs):
            z = Xn @ self.w + self.b
            z = np.clip(z, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (Xn.T @ (p - y)) / len(X) + l2 * self.w
            grad_b = float(np.mean(p - y))
            self.w -= lr * grad_w
            self.b -= lr * grad_b

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn = (X - self.mean) / self.std
        z = Xn @ self.w + self.b
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        return p.reshape(-1)

    def predict_proba_from_dict(self, feat_dict):
        vec = np.array([feat_dict.get(k, 0.0) for k in self.feature_names], dtype=np.float32)
        return float(self.predict_proba(vec)[0])

    def predict_label_from_dict(self, feat_dict):
        p = self.predict_proba_from_dict(feat_dict)
        return ("beautified" if p >= 0.5 else "natural"), p

    def save(self, path):
        np.savez(
            path,
            feature_names=np.array(self.feature_names, dtype=object),
            mean=self.mean,
            std=self.std,
            w=self.w,
            b=np.array([self.b], dtype=np.float32)
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        obj = cls()
        obj.feature_names = list(data["feature_names"])
        obj.mean = data["mean"]
        obj.std = data["std"]
        obj.w = data["w"]
        obj.b = float(data["b"][0])
        return obj

class MultiFaceBodyAnalyzer:
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    LEFT_BROW = [70, 63, 105, 66, 107]
    RIGHT_BROW = [336, 296, 334, 293, 300]
    NOSE_OUTLINE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327]

    def __init__(self, face_model_path=None, pose_model_path=None, num_faces=5, num_poses=4):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.script_dir, "models")
        self.num_faces = num_faces
        self.num_poses = num_poses

        self.face_model_path = face_model_path or self._find_default_face_model()
        self.pose_model_path = pose_model_path or self._find_default_pose_model()

        self.face_landmarker = None
        self.pose_landmarker = None
        self.model = None

        self.reset_temporal()

        if self.face_model_path and os.path.isfile(self.face_model_path):
            self._create_face_landmarker()

        if self.pose_model_path and os.path.isfile(self.pose_model_path):
            self._create_pose_landmarker()

    def _find_default_face_model(self):
        candidate = os.path.join(self.models_dir, "face_landmarker.task")
        return candidate if os.path.isfile(candidate) else None

    def _find_default_pose_model(self):
        candidates = [
            "pose_landmarker_full.task",
            "pose_landmarker_lite.task",
            "pose_landmarker_heavy.task",
            "pose_landmarker.task"
        ]
        for name in candidates:
            p = os.path.join(self.models_dir, name)
            if os.path.isfile(p):
                return p
        return None

    def _create_face_landmarker(self):
        if not self.face_model_path or not os.path.isfile(self.face_model_path):
            raise FileNotFoundError(f"Face model not found:\n{self.face_model_path}")

        base_options = BaseOptions(model_asset_path=self.face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

    def _create_pose_landmarker(self):
        if not self.pose_model_path or not os.path.isfile(self.pose_model_path):
            raise FileNotFoundError(f"Pose model not found:\n{self.pose_model_path}")

        base_options = BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    def set_face_model_path(self, new_model_path):
        self.face_model_path = new_model_path
        self.reset_temporal()
        self._create_face_landmarker()

    def set_pose_model_path(self, new_model_path):
        self.pose_model_path = new_model_path
        self._create_pose_landmarker()

    def set_num_faces(self, num_faces):
        self.num_faces = max(1, int(num_faces))
        if self.face_model_path and os.path.isfile(self.face_model_path):
            self._create_face_landmarker()

    def reset_temporal(self):
        self.prev_gray = None
        self.prev_ring_edges = None
        self.prev_center = None
        self.prev_heuristic = None

    def ready_face(self):
        return self.face_landmarker is not None

    def ready_pose(self):
        return self.pose_landmarker is not None

    def _tasks_landmarks_to_pixels(self, face_landmarks, width, height):
        pts = []
        for lm in face_landmarks:
            x = int(clamp(lm.x, 0.0, 1.0) * (width - 1))
            y = int(clamp(lm.y, 0.0, 1.0) * (height - 1))
            pts.append((x, y))
        return pts

    def _pose_landmarks_to_pixels(self, pose_landmarks, width, height):
        pts = []
        for lm in pose_landmarks:
            x = int(clamp(lm.x, 0.0, 1.0) * (width - 1))
            y = int(clamp(lm.y, 0.0, 1.0) * (height - 1))
            vis = float(getattr(lm, "visibility", 1.0))
            pres = float(getattr(lm, "presence", 1.0))
            pts.append((x, y, vis, pres))
        return pts

    def _dist(self, a, b):
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def _mean_pt(self, pts):
        arr = np.array(pts, dtype=np.float32)
        return tuple(np.mean(arr, axis=0))

    def _poly_mask(self, shape_hw, lm_pixels, indices):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array([lm_pixels[i] for i in indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask

    def _rect_mask(self, shape_hw, x1, y1, x2, y2):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = int(clamp(x1, 0, w - 1))
        x2 = int(clamp(x2, 0, w - 1))
        y1 = int(clamp(y1, 0, h - 1))
        y2 = int(clamp(y2, 0, h - 1))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def _build_masks(self, frame_shape, lm):
        h, w = frame_shape[:2]
        face_mask = self._poly_mask((h, w), lm, self.FACE_OVAL)
        left_eye_mask = self._poly_mask((h, w), lm, self.LEFT_EYE)
        right_eye_mask = self._poly_mask((h, w), lm, self.RIGHT_EYE)
        lips_mask = self._poly_mask((h, w), lm, self.OUTER_LIPS)
        nose_mask = self._poly_mask((h, w), lm, self.NOSE_OUTLINE)

        eye_mask = cv2.bitwise_or(left_eye_mask, right_eye_mask)
        eye_mask = cv2.dilate(eye_mask, np.ones((11, 11), np.uint8), iterations=1)
        lips_mask = cv2.dilate(lips_mask, np.ones((11, 11), np.uint8), iterations=1)
        nose_core = cv2.dilate(nose_mask, np.ones((7, 7), np.uint8), iterations=1)

        skin_mask = face_mask.copy()
        skin_mask[eye_mask > 0] = 0
        skin_mask[lips_mask > 0] = 0
        skin_mask = cv2.erode(skin_mask, np.ones((5, 5), np.uint8), iterations=1)

        face_pts = np.array([lm[i] for i in self.FACE_OVAL], dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(face_pts)

        brow_y = int(min(
            np.mean([lm[i][1] for i in self.LEFT_BROW]),
            np.mean([lm[i][1] for i in self.RIGHT_BROW])
        ))
        brow_y = max(brow_y, y + int(0.18 * bh))

        forehead_rect = self._rect_mask((h, w), x + 0.22 * bw, y + 0.06 * bh, x + 0.78 * bw, brow_y - 0.03 * bh)
        left_cheek_rect = self._rect_mask((h, w), x + 0.08 * bw, y + 0.35 * bh, x + 0.42 * bw, y + 0.72 * bh)
        right_cheek_rect = self._rect_mask((h, w), x + 0.58 * bw, y + 0.35 * bh, x + 0.92 * bw, y + 0.72 * bh)
        nose_rect = self._rect_mask((h, w), x + 0.42 * bw, y + 0.24 * bh, x + 0.58 * bw, y + 0.62 * bh)

        forehead_mask = cv2.bitwise_and(forehead_rect, face_mask)
        forehead_mask[eye_mask > 0] = 0

        left_cheek_mask = cv2.bitwise_and(left_cheek_rect, skin_mask)
        left_cheek_mask[nose_core > 0] = 0

        right_cheek_mask = cv2.bitwise_and(right_cheek_rect, skin_mask)
        right_cheek_mask[nose_core > 0] = 0

        nose_roi_mask = cv2.bitwise_and(nose_rect, face_mask)
        nose_roi_mask[eye_mask > 0] = 0
        nose_roi_mask[lips_mask > 0] = 0

        ring = cv2.dilate(face_mask, np.ones((9, 9), np.uint8), iterations=1) - \
               cv2.erode(face_mask, np.ones((9, 9), np.uint8), iterations=1)

        return {
            "face": face_mask,
            "skin": skin_mask,
            "ring": ring,
            "forehead": forehead_mask,
            "left_cheek": left_cheek_mask,
            "right_cheek": right_cheek_mask,
            "nose": nose_roi_mask,
            "bbox": (x, y, bw, bh)
        }

    def _region_metrics(self, lap, residual, lab_a, lab_b, edges, ring_mask, mask):
        if cv2.countNonZero(mask) < 250:
            return {
                "lap_var": 0.0,
                "residual_mean": 0.0,
                "chroma_std": 0.0,
                "edge_density": 0.0,
                "texture_ratio": 0.0,
                "smoothness": 0.0,
                "detail_loss": 0.0,
                "uniformity": 0.0,
                "edge_suspicion": 0.0,
                "score": 0.0
            }

        idx = mask > 0
        lap_var = float(np.var(lap[idx]))
        residual_mean = float(np.mean(residual[idx]))
        chroma_std = float((np.std(lab_a[idx]) + np.std(lab_b[idx])) / 2.0)
        edge_density = float(np.mean(edges[idx] > 0) * 100.0)

        ring_idx = ring_mask > 0
        ring_edge_density = float(np.mean(edges[ring_idx] > 0) * 100.0) if np.any(ring_idx) else 1.0
        texture_ratio = edge_density / max(ring_edge_density, 1e-6)

        smoothness = clamp(1.0 - lap_var / 180.0, 0.0, 1.0)
        detail_loss = clamp(1.0 - residual_mean / 12.0, 0.0, 1.0)
        uniformity = clamp(1.0 - chroma_std / 14.0, 0.0, 1.0)
        edge_suspicion = clamp(1.0 - texture_ratio / 1.15, 0.0, 1.0)

        score = (
            0.38 * smoothness +
            0.27 * detail_loss +
            0.20 * uniformity +
            0.15 * edge_suspicion
        )

        return {
            "lap_var": lap_var,
            "residual_mean": residual_mean,
            "chroma_std": chroma_std,
            "edge_density": edge_density,
            "texture_ratio": texture_ratio,
            "smoothness": smoothness,
            "detail_loss": detail_loss,
            "uniformity": uniformity,
            "edge_suspicion": edge_suspicion,
            "score": float(score)
        }

    def _compute_geometry(self, lm):
        face_width = max(self._dist(lm[234], lm[454]), 1e-6)
        face_height = max(self._dist(lm[10], lm[152]), 1e-6)
        jaw_width = self._dist(lm[172], lm[397])

        left_eye_w = self._dist(lm[33], lm[133])
        right_eye_w = self._dist(lm[362], lm[263])
        left_eye_h = 0.5 * (self._dist(lm[159], lm[145]) + self._dist(lm[160], lm[144]))
        right_eye_h = 0.5 * (self._dist(lm[386], lm[374]) + self._dist(lm[385], lm[380]))

        eye_ratio = ((left_eye_w + right_eye_w) / 2.0) / face_width
        eye_height_ratio = ((left_eye_h + right_eye_h) / 2.0) / face_height
        jaw_cheek_ratio = jaw_width / face_width
        face_aspect_ratio = face_height / face_width

        left_eye_center = self._mean_pt([lm[i] for i in self.LEFT_EYE])
        right_eye_center = self._mean_pt([lm[i] for i in self.RIGHT_EYE])
        face_center_x = (lm[234][0] + lm[454][0]) / 2.0
        nose_x = lm[1][0]

        eye_tilt = abs(left_eye_center[1] - right_eye_center[1]) / face_height
        nose_shift = abs(nose_x - face_center_x) / face_width
        frontality = clamp(1.0 - (3.2 * eye_tilt + 2.0 * nose_shift), 0.0, 1.0)

        big_eye_score = clamp((eye_ratio - 0.175) / 0.05, 0.0, 1.0) * frontality
        slim_face_score = (
            clamp((0.80 - jaw_cheek_ratio) / 0.20, 0.0, 1.0) *
            clamp((face_aspect_ratio - 1.18) / 0.35, 0.0, 1.0) *
            frontality
        )
        geo_score = 0.58 * big_eye_score + 0.42 * slim_face_score

        return {
            "eye_ratio": float(eye_ratio),
            "eye_height_ratio": float(eye_height_ratio),
            "jaw_cheek_ratio": float(jaw_cheek_ratio),
            "face_aspect_ratio": float(face_aspect_ratio),
            "frontality": float(frontality),
            "big_eye_score": float(big_eye_score),
            "slim_face_score": float(slim_face_score),
            "geo_score": float(geo_score),
            "face_center": (
                int((lm[234][0] + lm[454][0]) / 2),
                int((lm[10][1] + lm[152][1]) / 2)
            )
        }

    def _compute_temporal(self, gray, face_mask, face_center, current_heuristic):
        ring = cv2.dilate(face_mask, np.ones((9, 9), np.uint8), iterations=1) - \
               cv2.erode(face_mask, np.ones((9, 9), np.uint8), iterations=1)
        edges = cv2.Canny(gray, 70, 140)
        ring_edges = np.zeros_like(edges)
        ring_edges[ring > 0] = edges[ring > 0]

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_ring_edges = ring_edges.copy()
            self.prev_center = face_center
            self.prev_heuristic = current_heuristic
            return {"flicker": 0.0, "warp": 0.0, "jump": 0.0, "score": 0.0}

        dx = int(face_center[0] - self.prev_center[0])
        dy = int(face_center[1] - self.prev_center[1])

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        prev_shifted = cv2.warpAffine(
            self.prev_ring_edges, M, (gray.shape[1], gray.shape[0]),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        union = ((ring > 0) | (prev_shifted > 0))
        if np.any(union):
            flicker_raw = float(np.mean(np.logical_xor(ring_edges[union] > 0, prev_shifted[union] > 0)))
        else:
            flicker_raw = 0.0
        flicker_score = clamp((flicker_raw - 0.08) / 0.25, 0.0, 1.0)

        face_idx = face_mask > 0
        ring_idx = ring > 0
        if np.sum(face_idx) > 400:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.1, 0
            )
            medx = float(np.median(flow[:, :, 0][face_idx]))
            medy = float(np.median(flow[:, :, 1][face_idx]))
            residual_flow = np.sqrt((flow[:, :, 0] - medx) ** 2 + (flow[:, :, 1] - medy) ** 2)

            interior_mean = float(np.mean(residual_flow[face_idx]))
            ring_mean = float(np.mean(residual_flow[ring_idx])) if np.any(ring_idx) else interior_mean
            warp_ratio = ring_mean / max(interior_mean, 1e-6)
        else:
            warp_ratio = 1.0

        warp_score = clamp((warp_ratio - 1.15) / 0.90, 0.0, 1.0)
        jump_score = clamp(abs(current_heuristic - self.prev_heuristic) / 18.0, 0.0, 1.0)
        temporal_score = 0.42 * flicker_score + 0.33 * warp_score + 0.25 * jump_score

        self.prev_gray = gray.copy()
        self.prev_ring_edges = ring_edges.copy()
        self.prev_center = face_center
        self.prev_heuristic = current_heuristic

        return {
            "flicker": float(flicker_score),
            "warp": float(warp_score),
            "jump": float(jump_score),
            "score": float(temporal_score)
        }

    def _analyze_single_face(self, frame_bgr, lm, gray, lab, static_image=False):
        masks = self._build_masks(frame_bgr.shape, lm)
        face_mask = masks["face"]
        skin_mask = masks["skin"]
        ring_mask = masks["ring"]

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        bilateral = cv2.bilateralFilter(gray, 9, 35, 35)
        residual = np.abs(gray.astype(np.float32) - bilateral.astype(np.float32))
        edges = cv2.Canny(gray, 60, 120)
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]

        global_metrics = self._region_metrics(lap, residual, lab_a, lab_b, edges, ring_mask, skin_mask)
        forehead_metrics = self._region_metrics(lap, residual, lab_a, lab_b, edges, ring_mask, masks["forehead"])
        left_cheek_metrics = self._region_metrics(lap, residual, lab_a, lab_b, edges, ring_mask, masks["left_cheek"])
        right_cheek_metrics = self._region_metrics(lap, residual, lab_a, lab_b, edges, ring_mask, masks["right_cheek"])
        nose_metrics = self._region_metrics(lap, residual, lab_a, lab_b, edges, ring_mask, masks["nose"])

        roi_score = (
            0.23 * forehead_metrics["score"] +
            0.29 * left_cheek_metrics["score"] +
            0.29 * right_cheek_metrics["score"] +
            0.19 * nose_metrics["score"]
        )
        texture_score = 0.45 * global_metrics["score"] + 0.55 * roi_score
        geo = self._compute_geometry(lm)

        base_heuristic = 100.0 * (0.72 * texture_score + 0.28 * geo["geo_score"])
        temporal = self._compute_temporal(gray, face_mask, geo["face_center"], base_heuristic) if not static_image else {
            "flicker": 0.0, "warp": 0.0, "jump": 0.0, "score": 0.0
        }

        heuristic_score = base_heuristic if static_image else 100.0 * (
            0.50 * texture_score +
            0.22 * geo["geo_score"] +
            0.28 * temporal["score"]
        )
        heuristic_score = float(clamp(heuristic_score, 0.0, 100.0))
        heuristic_label = "beautified" if heuristic_score >= 50.0 else "natural"

        feature_dict = {
            "roi_forehead_score": float(forehead_metrics["score"]),
            "roi_left_cheek_score": float(left_cheek_metrics["score"]),
            "roi_right_cheek_score": float(right_cheek_metrics["score"]),
            "roi_nose_score": float(nose_metrics["score"]),
            "global_skin_score": float(global_metrics["score"]),
            "texture_score": float(texture_score),
            "global_smoothness": float(global_metrics["smoothness"]),
            "global_detail_loss": float(global_metrics["detail_loss"]),
            "global_uniformity": float(global_metrics["uniformity"]),
            "global_edge_suspicion": float(global_metrics["edge_suspicion"]),
            "eye_ratio": float(geo["eye_ratio"]),
            "eye_height_ratio": float(geo["eye_height_ratio"]),
            "jaw_cheek_ratio": float(geo["jaw_cheek_ratio"]),
            "face_aspect_ratio": float(geo["face_aspect_ratio"]),
            "frontality": float(geo["frontality"]),
            "big_eye_score": float(geo["big_eye_score"]),
            "slim_face_score": float(geo["slim_face_score"]),
            "geo_score": float(geo["geo_score"]),
            "temporal_flicker": float(temporal["flicker"]),
            "temporal_warp": float(temporal["warp"]),
            "temporal_jump": float(temporal["jump"]),
            "temporal_score": float(temporal["score"]),
            "heuristic_score": float(heuristic_score),
        }

        feature_vector = np.array([feature_dict[k] for k in FEATURE_NAMES], dtype=np.float32)

        if self.model is not None:
            model_label, model_prob = self.model.predict_label_from_dict(feature_dict)
        else:
            model_prob = heuristic_score / 100.0
            model_label = heuristic_label

        x, y, bw, bh = masks["bbox"]
        return {
            "lm": lm,
            "bbox": (x, y, bw, bh),
            "bbox_area": float(bw * bh),
            "masks": masks,
            "texture_score": float(texture_score),
            "geo": geo,
            "temporal": temporal,
            "heuristic_score": heuristic_score,
            "heuristic_label": heuristic_label,
            "model_label": model_label,
            "model_prob": float(model_prob),
            "feature_dict": feature_dict,
            "feature_vector": feature_vector,
            "global_metrics": global_metrics,
            "roi_scores": {
                "forehead": float(forehead_metrics["score"]),
                "left_cheek": float(left_cheek_metrics["score"]),
                "right_cheek": float(right_cheek_metrics["score"]),
                "nose": float(nose_metrics["score"]),
            }
        }

    def _analyze_poses(self, pose_result, frame_shape):
        poses = []
        h, w = frame_shape[:2]
        if not pose_result or not pose_result.pose_landmarks:
            return poses

        for pose_landmarks in pose_result.pose_landmarks:
            pts = self._pose_landmarks_to_pixels(pose_landmarks, w, h)
            vis_vals = [p[2] for p in pts]
            avg_vis = float(np.mean(vis_vals)) if vis_vals else 0.0

            xs = [p[0] for p in pts if p[2] > 0.3]
            ys = [p[1] for p in pts if p[2] > 0.3]

            if xs and ys:
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                x1 = y1 = x2 = y2 = 0

            poses.append({
                "pts": pts,
                "avg_vis": avg_vis,
                "bbox": (x1, y1, x2, y2),
                "bbox_area": float(max(0, x2 - x1) * max(0, y2 - y1))
            })

        return poses

    def analyze(self, frame_bgr, static_image=False):
        if static_image:
            self.reset_temporal()

        out = frame_bgr.copy()

        if not self.ready_face():
            cv2.putText(out, "Load face model first (.task)", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            return {
                "ok": False,
                "frame": out,
                "summary": "Face model not loaded.",
                "faces_count": 0,
                "poses_count": 0,
                "heuristic_score": 0.0,
                "heuristic_label": "n/a",
                "model_label": "n/a",
                "model_prob": 0.0,
                "feature_dict": {},
                "feature_vector": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            }

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self.face_landmarker.detect(mp_image)
        pose_result = self.pose_landmarker.detect(mp_image) if self.ready_pose() else None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        faces = []
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                lm = self._tasks_landmarks_to_pixels(face_landmarks, frame_bgr.shape[1], frame_bgr.shape[0])
                face_info = self._analyze_single_face(frame_bgr, lm, gray, lab, static_image=static_image)
                faces.append(face_info)

        poses = self._analyze_poses(pose_result, frame_bgr.shape)

        for i, pose in enumerate(poses, 1):
            pts = pose["pts"]
            color = (255, 255, 0)

            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    x1, y1, v1, _ = pts[a]
                    x2, y2, v2, _ = pts[b]
                    if v1 > 0.35 and v2 > 0.35:
                        cv2.line(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            for x, y, vis, _ in pts:
                if vis > 0.35:
                    cv2.circle(out, (x, y), 3, color, -1)

            x1, y1, x2, y2 = pose["bbox"]
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, f"P{i} vis={pose['avg_vis']:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        if not faces:
            cv2.putText(out, "No face detected", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2, cv2.LINE_AA)

            summary = f"No face detected.\nWhole body count: {len(poses)}"
            if not self.ready_pose():
                summary += "\nPose model not loaded."

            return {
                "ok": False,
                "frame": out,
                "summary": summary,
                "faces_count": 0,
                "poses_count": len(poses),
                "heuristic_score": 0.0,
                "heuristic_label": "no_face",
                "model_label": "n/a",
                "model_prob": 0.0,
                "feature_dict": {},
                "feature_vector": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
                "faces": [],
                "poses": poses,
            }

        faces.sort(key=lambda d: d["bbox_area"], reverse=True)
        primary = faces[0]

        for i, face in enumerate(faces, 1):
            lm = face["lm"]
            masks = face["masks"]
            score = face["heuristic_score"]
            color = (0, 200, 0) if score < 35 else ((0, 185, 255) if score < 60 else (0, 0, 255))

            blend_mask(out, masks["forehead"], (255, 128, 0), 0.18)
            blend_mask(out, masks["left_cheek"], (255, 0, 180), 0.18)
            blend_mask(out, masks["right_cheek"], (180, 0, 255), 0.18)
            blend_mask(out, masks["nose"], (0, 180, 255), 0.18)

            face_poly = np.array([lm[j] for j in self.FACE_OVAL], dtype=np.int32)
            cv2.polylines(out, [face_poly], True, color, 2, cv2.LINE_AA)

            x, y, bw, bh = face["bbox"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(out, f"F{i} {score:.1f} {face['heuristic_label']}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

        p = primary
        line1 = f"Faces={len(faces)}  Poses={len(poses)}"
        line2 = f"Primary Face: {p['heuristic_score']:.1f} [{p['heuristic_label']}]  model={p['model_label']} ({p['model_prob']:.2f})"
        line3 = f"Geo big-eye/slim={p['geo']['big_eye_score']:.2f}/{p['geo']['slim_face_score']:.2f}  frontal={p['geo']['frontality']:.2f}"
        line4 = f"Temp flicker/warp/jump={p['temporal']['flicker']:.2f}/{p['temporal']['warp']:.2f}/{p['temporal']['jump']:.2f}"

        cv2.putText(out, line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line2, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line3, (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(out, line4, (16, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (220, 220, 220), 2, cv2.LINE_AA)

        summary_lines = [
            f"Faces detected: {len(faces)}",
            f"Whole bodies detected: {len(poses)}",
            f"Primary face heuristic={p['heuristic_score']:.2f} ({p['heuristic_label']})",
            f"Primary face model={p['model_label']}  prob={p['model_prob']:.4f}",
            f"Primary face texture={p['texture_score']:.4f}",
            f"Primary face ROI forehead={p['roi_scores']['forehead']:.4f}  left cheek={p['roi_scores']['left_cheek']:.4f}  right cheek={p['roi_scores']['right_cheek']:.4f}  nose={p['roi_scores']['nose']:.4f}",
            f"Primary face geo eye_ratio={p['geo']['eye_ratio']:.4f}  eye_height_ratio={p['geo']['eye_height_ratio']:.4f}  jaw_cheek_ratio={p['geo']['jaw_cheek_ratio']:.4f}  face_ar={p['geo']['face_aspect_ratio']:.4f}",
            f"Primary face big-eye={p['geo']['big_eye_score']:.4f}  slim-face={p['geo']['slim_face_score']:.4f}  frontality={p['geo']['frontality']:.4f}",
            f"Primary face temporal flicker={p['temporal']['flicker']:.4f}  warp={p['temporal']['warp']:.4f}  jump={p['temporal']['jump']:.4f}  total={p['temporal']['score']:.4f}",
        ]
        if not self.ready_pose():
            summary_lines.append("Pose model not loaded.")
        elif poses:
            for idx, pose in enumerate(poses[:5], 1):
                summary_lines.append(f"Pose {idx}: avg visibility={pose['avg_vis']:.4f}  bbox={pose['bbox']}")

        return {
            "ok": True,
            "frame": out,
            "summary": "\n".join(summary_lines),
            "faces_count": len(faces),
            "poses_count": len(poses),
            "faces": faces,
            "poses": poses,
            "heuristic_score": p["heuristic_score"],
            "heuristic_label": p["heuristic_label"],
            "model_label": p["model_label"],
            "model_prob": p["model_prob"],
            "texture_score": p["texture_score"],
            "roi_scores": p["roi_scores"],
            "geo": p["geo"],
            "temporal": p["temporal"],
            "feature_dict": p["feature_dict"],
            "feature_vector": p["feature_vector"],
        }

class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    status_ready = pyqtSignal(str)

    def __init__(self, source, analyzer, is_camera=False):
        super().__init__()
        self.source = source
        self.analyzer = analyzer
        self.is_camera = is_camera
        self._running = True
        self.scores = deque(maxlen=60)

    def stop(self):
        self._running = False

    def run(self):
        self.analyzer.reset_temporal()
        cap = cv2.VideoCapture(0 if self.is_camera else self.source)
        if not cap.isOpened():
            self.status_ready.emit("Failed to open video source.")
            return

        self.status_ready.emit("Video processing started.")
        last_t = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] > 960:
                scale = 960.0 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            result = self.analyzer.analyze(frame, static_image=False)
            self.scores.append(result.get("heuristic_score", 0.0))
            result["rolling_avg"] = float(np.mean(self.scores)) if self.scores else 0.0

            now = time.time()
            result["fps"] = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            self.frame_ready.emit(result["frame"], result)
            if not self.is_camera:
                time.sleep(0.01)

        cap.release()
        self.status_ready.emit("Video processing stopped.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = MultiFaceBodyAnalyzer(num_faces=5, num_poses=4)
        self.worker = None
        self.last_frame = None
        self.last_result = None

        self.setWindowTitle(self._title_text())
        self.resize(1600, 1000)

        self._build_ui()
        self._apply_style()

        if not self.analyzer.ready_face():
            self.log_msg("Face model not loaded. Please click 'Choose Face Model'.")
        if not self.analyzer.ready_pose():
            self.log_msg("Pose model not loaded. Please click 'Choose Pose Model' for whole body detect.")

    def _title_text(self):
        face_name = os.path.basename(self.analyzer.face_model_path) if self.analyzer.face_model_path else "no_face_model"
        pose_name = os.path.basename(self.analyzer.pose_model_path) if self.analyzer.pose_model_path else "no_pose_model"
        return f"Multi Face + Whole Body Detector - face:{face_name} | pose:{pose_name}"

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self.preview = QLabel("Open image / video / webcam")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(940, 740)
        self.preview.setStyleSheet("background:#111;border:1px solid #444;color:#ddd;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.preview)

        btn_row1 = QHBoxLayout()
        self.btn_image = QPushButton("Open Image")
        self.btn_video = QPushButton("Open Video")
        self.btn_cam = QPushButton("Open Webcam")
        self.btn_stop = QPushButton("Stop")
        self.btn_save = QPushButton("Save Frame")

        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_save.clicked.connect(self.save_frame)

        for b in [self.btn_image, self.btn_video, self.btn_cam, self.btn_stop, self.btn_save]:
            btn_row1.addWidget(b)
        left.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_choose_face_model = QPushButton("Choose Face Model")
        self.btn_choose_pose_model = QPushButton("Choose Pose Model")
        self.btn_train = QPushButton("Train Beauty Model")
        self.btn_load_model = QPushButton("Load Beauty Model")
        self.btn_save_model = QPushButton("Save Beauty Model")
        self.btn_reset_temporal = QPushButton("Reset Temporal")

        self.btn_choose_face_model.clicked.connect(self.choose_face_model)
        self.btn_choose_pose_model.clicked.connect(self.choose_pose_model)
        self.btn_train.clicked.connect(self.train_model_from_folder)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_reset_temporal.clicked.connect(self.reset_temporal)

        for b in [
            self.btn_choose_face_model, self.btn_choose_pose_model,
            self.btn_train, self.btn_load_model, self.btn_save_model,
            self.btn_reset_temporal
        ]:
            btn_row2.addWidget(b)
        left.addLayout(btn_row2)

        right = QVBoxLayout()

        g1 = QGroupBox("Result")
        g1l = QGridLayout(g1)
        self.lbl_faces = QLabel("-")
        self.lbl_poses = QLabel("-")
        self.lbl_heuristic = QLabel("-")
        self.lbl_model = QLabel("-")
        self.lbl_texture = QLabel("-")
        self.lbl_geo = QLabel("-")
        self.lbl_temporal = QLabel("-")
        self.lbl_roi = QLabel("-")
        self.lbl_roll = QLabel("-")
        self.lbl_fps = QLabel("-")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFormat("%p")

        g1l.addWidget(QLabel("Faces:"), 0, 0)
        g1l.addWidget(self.lbl_faces, 0, 1)
        g1l.addWidget(QLabel("Bodies:"), 1, 0)
        g1l.addWidget(self.lbl_poses, 1, 1)
        g1l.addWidget(QLabel("Primary Heuristic:"), 2, 0)
        g1l.addWidget(self.lbl_heuristic, 2, 1)
        g1l.addWidget(QLabel("Beauty Model:"), 3, 0)
        g1l.addWidget(self.lbl_model, 3, 1)
        g1l.addWidget(QLabel("Texture:"), 4, 0)
        g1l.addWidget(self.lbl_texture, 4, 1)
        g1l.addWidget(QLabel("Geometry:"), 5, 0)
        g1l.addWidget(self.lbl_geo, 5, 1)
        g1l.addWidget(QLabel("Temporal:"), 6, 0)
        g1l.addWidget(self.lbl_temporal, 6, 1)
        g1l.addWidget(QLabel("ROI:"), 7, 0)
        g1l.addWidget(self.lbl_roi, 7, 1)
        g1l.addWidget(QLabel("Rolling Avg:"), 8, 0)
        g1l.addWidget(self.lbl_roll, 8, 1)
        g1l.addWidget(QLabel("FPS:"), 9, 0)
        g1l.addWidget(self.lbl_fps, 9, 1)
        g1l.addWidget(self.bar, 10, 0, 1, 2)

        g2 = QGroupBox("Details")
        g2l = QVBoxLayout(g2)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        g2l.addWidget(self.details)

        g3 = QGroupBox("Log")
        g3l = QVBoxLayout(g3)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        g3l.addWidget(self.log)

        right.addWidget(g1)
        right.addWidget(g2, 1)
        right.addWidget(g3, 1)

        main_layout.addLayout(left, 3)
        main_layout.addLayout(right, 2)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background:#20242b; color:#e8e8e8; }
            QWidget { color:#e8e8e8; font-size:15px; }
            QGroupBox {
                border:1px solid #4a5568;
                margin-top:10px;
                padding-top:10px;
                font-weight:bold;
                background:#262b33;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left:10px;
                padding:0 3px;
            }
            QPushButton {
                background:#3b82f6;
                border:none;
                padding:8px 12px;
                border-radius:6px;
                color:white;
                font-weight:600;
            }
            QPushButton:hover { background:#2563eb; }
            QPushButton:pressed { background:#1d4ed8; }
            QTextEdit {
                background:#111827;
                border:1px solid #374151;
                color:#e5e7eb;
            }
            QProgressBar {
                border:1px solid #555;
                border-radius:4px;
                text-align:center;
                background:#111827;
            }
        """)

    def log_msg(self, msg):
        self.log.append(msg)

    def closeEvent(self, event):
        self.stop_worker()
        event.accept()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1200)
            self.worker = None
            self.log_msg("Worker stopped.")

    def reset_temporal(self):
        self.analyzer.reset_temporal()
        self.log_msg("Temporal state reset.")

    def choose_face_model(self):
        self.stop_worker()
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Face Landmarker Model", "", "Task models (*.task)"
        )
        if not path:
            return
        try:
            self.analyzer.set_face_model_path(path)
            self.setWindowTitle(self._title_text())
            self.log_msg(f"Face model switched to: {path}")
            QMessageBox.information(self, "Face Model Loaded", f"Using face model:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Face Model Load Error", f"Failed to load face model:\n{e}")

    def choose_pose_model(self):
        self.stop_worker()
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Pose Landmarker Model", "", "Task models (*.task)"
        )
        if not path:
            return
        try:
            self.analyzer.set_pose_model_path(path)
            self.setWindowTitle(self._title_text())
            self.log_msg(f"Pose model switched to: {path}")
            QMessageBox.information(self, "Pose Model Loaded", f"Using pose model:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Pose Model Load Error", f"Failed to load pose model:\n{e}")

    def _ensure_face_model(self):
        if self.analyzer.ready_face():
            return True
        QMessageBox.warning(self, "Face Model Required", "Please load a face model first.")
        return False

    def open_image(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", "Failed to read image.")
            return

        if frame.shape[1] > 1400:
            scale = 1400.0 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        result = self.analyzer.analyze(frame, static_image=True)
        self.last_frame = result["frame"]
        self.last_result = result

        self.preview.setPixmap(
            cv_to_pixmap(result["frame"]).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.update_result_ui(result)
        self.log_msg(f"Loaded image: {path}")

    def open_video(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if not path:
            return
        self.start_worker(path, is_camera=False)

    def open_camera(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return
        self.start_worker(0, is_camera=True)

    def start_worker(self, source, is_camera):
        self.worker = VideoWorker(source, self.analyzer, is_camera=is_camera)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_ready.connect(self.log_msg)
        self.worker.start()

    def on_frame_ready(self, frame, result):
        self.last_frame = frame
        self.last_result = result
        self.preview.setPixmap(
            cv_to_pixmap(frame).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.update_result_ui(result)

    def update_result_ui(self, result):
        hs = result.get("heuristic_score", 0.0)
        hl = result.get("heuristic_label", "-")
        ml = result.get("model_label", "-")
        model_prob = result.get("model_prob", 0.0)
        ts = result.get("texture_score", 0.0)
        geo = result.get("geo", {})
        temporal = result.get("temporal", {})
        roi = result.get("roi_scores", {})
        rolling = result.get("rolling_avg", None)
        fps = result.get("fps", None)

        self.lbl_faces.setText(str(result.get("faces_count", 0)))
        self.lbl_poses.setText(str(result.get("poses_count", 0)))
        self.lbl_heuristic.setText(f"{hs:.2f} [{hl}]")
        self.lbl_model.setText(f"{ml} ({model_prob:.3f})")
        self.lbl_texture.setText(f"{ts:.3f}")
        self.lbl_geo.setText(
            f"geo={geo.get('geo_score', 0.0):.3f}  big-eye={geo.get('big_eye_score', 0.0):.3f}  "
            f"slim-face={geo.get('slim_face_score', 0.0):.3f}  frontal={geo.get('frontality', 0.0):.3f}"
        )
        self.lbl_temporal.setText(
            f"total={temporal.get('score', 0.0):.3f}  flicker={temporal.get('flicker', 0.0):.3f}  "
            f"warp={temporal.get('warp', 0.0):.3f}  jump={temporal.get('jump', 0.0):.3f}"
        )
        self.lbl_roi.setText(
            f"forehead={roi.get('forehead', 0.0):.3f}  lcheek={roi.get('left_cheek', 0.0):.3f}  "
            f"rcheek={roi.get('right_cheek', 0.0):.3f}  nose={roi.get('nose', 0.0):.3f}"
        )
        self.lbl_roll.setText(f"{rolling:.2f}" if rolling is not None else "-")
        self.lbl_fps.setText(f"{fps:.1f}" if fps is not None else "-")
        self.details.setPlainText(result.get("summary", ""))

        self.bar.setValue(int(clamp(hs, 0, 100)))
        if hs < 35:
            color = "#22c55e"
        elif hs < 60:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border:1px solid #555;
                border-radius:4px;
                text-align:center;
                background:#111827;
            }}
            QProgressBar::chunk {{
                background-color:{color};
            }}
        """)

    def save_frame(self):
        if self.last_frame is None:
            QMessageBox.information(self, "Info", "No annotated frame available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Frame", "multi_face_body_result.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return

        if cv2.imwrite(path, self.last_frame):
            self.log_msg(f"Saved frame: {path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save frame.")

    def train_model_from_folder(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        root = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not root:
            return

        natural_dir = os.path.join(root, "natural")
        beautified_dir = os.path.join(root, "beautified")

        if not os.path.isdir(natural_dir) or not os.path.isdir(beautified_dir):
            QMessageBox.warning(
                self,
                "Dataset Error",
                "Folder must contain:\n\nnatural/\nbeautified/"
            )
            return

        files = []
        for label, d in [(0, natural_dir), (1, beautified_dir)]:
            for fn in os.listdir(d):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    files.append((os.path.join(d, fn), label))

        if len(files) < 10:
            QMessageBox.warning(self, "Dataset Error", "Need at least 10 images.")
            return

        self.log_msg(f"Training dataset scan started. Total files: {len(files)}")
        X, y, rows = [], [], []
        self.analyzer.reset_temporal()

        for idx, (path, label) in enumerate(files, 1):
            img = cv2.imread(path)
            if img is None:
                self.log_msg(f"Skip unreadable: {path}")
                continue

            if img.shape[1] > 1400:
                scale = 1400.0 / img.shape[1]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            res = self.analyzer.analyze(img, static_image=True)
            if not res["ok"]:
                self.log_msg(f"Skip no-face: {path}")
                continue

            X.append(res["feature_vector"])
            y.append(label)
            rows.append([path, label] + [res["feature_dict"][k] for k in FEATURE_NAMES])

            if idx % 10 == 0:
                self.log_msg(f"Processed {idx}/{len(files)}")

        if len(X) < 8:
            QMessageBox.warning(self, "Training Error", "Too few valid face samples.")
            return

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(X))
        split = max(1, int(0.8 * len(X)))
        tr = perm[:split]
        te = perm[split:] if len(X) - split > 0 else perm[:split]

        model = LogisticBeautyModel()
        model.fit(X[tr], y[tr])

        tr_pred = (model.predict_proba(X[tr]) >= 0.5).astype(np.float32)
        te_pred = (model.predict_proba(X[te]) >= 0.5).astype(np.float32)
        tr_acc = float(np.mean(tr_pred == y[tr]))
        te_acc = float(np.mean(te_pred == y[te]))

        self.analyzer.model = model

        csv_path = os.path.join(root, "beauty_dataset_features.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "label"] + FEATURE_NAMES)
            writer.writerows(rows)

        self.log_msg(f"Training finished. Valid samples: {len(X)}")
        self.log_msg(f"Train accuracy: {tr_acc:.4f}")
        self.log_msg(f"Test accuracy: {te_acc:.4f}")
        self.log_msg(f"Feature CSV exported: {csv_path}")

        QMessageBox.information(
            self,
            "Training Done",
            f"Model trained.\n\nValid samples: {len(X)}\nTrain acc: {tr_acc:.4f}\nTest acc: {te_acc:.4f}\n\nCSV: {csv_path}"
        )

    def save_model(self):
        if self.analyzer.model is None:
            QMessageBox.information(self, "Info", "No trained/loaded beauty model available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Beauty Model", "beauty_logreg_model.npz", "NPZ (*.npz)"
        )
        if not path:
            return
        self.analyzer.model.save(path)
        self.log_msg(f"Beauty model saved: {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Beauty Model", "", "NPZ (*.npz)"
        )
        if not path:
            return

        try:
            self.analyzer.model = LogisticBeautyModel.load(path)
            self.log_msg(f"Beauty model loaded: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load beauty model:\n{e}")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()