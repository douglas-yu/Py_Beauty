import sys
import os
import csv
import time
import math
import cv2
import numpy as np
from collections import deque
from datetime import datetime

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

ALL_CLASS_NAMES = ["natural", "skin-smoothing", "geometry-warp", "mixed"]

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

def blend_mask(img, mask, color_bgr, alpha=0.25):
    idx = mask > 0
    if not np.any(idx):
        return
    color = np.array(color_bgr, dtype=np.float32)
    img[idx] = (img[idx].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)

def list_images(folder):
    if not os.path.isdir(folder):
        return []
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    return [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(exts)
    ]

def find_split_root(dataset_root, split_name):
    p = os.path.join(dataset_root, split_name)
    return p if os.path.isdir(p) else dataset_root

def discover_classes(dataset_root):
    """
    4类优先；如果 mixed 不存在或为空，则退化为 3 类。
    """
    candidate_root = find_split_root(dataset_root, "train")
    present = []
    for c in ALL_CLASS_NAMES:
        if len(list_images(os.path.join(candidate_root, c))) > 0:
            present.append(c)

    if "natural" not in present:
        return []

    base3 = ["natural", "skin-smoothing", "geometry-warp"]
    if all(c in present for c in base3):
        if "mixed" in present:
            return ALL_CLASS_NAMES[:]
        return base3[:]

    # 最低要求：至少 2 类
    if len(present) >= 2:
        return present
    return []

def augment_image(img, rng):
    """
    轻量通用增强：亮度/对比度、轻微模糊、轻微压缩、轻微缩放裁剪
    不引入标签语义变化，适合当前特征提取方案
    """
    out = img.copy()

    # brightness / contrast
    alpha = rng.uniform(0.9, 1.12)
    beta = rng.uniform(-12, 12)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # occasional blur
    if rng.rand() < 0.35:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # jpeg compression simulation
    if rng.rand() < 0.45:
        q = int(rng.uniform(50, 92))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # small random crop + resize back
    if rng.rand() < 0.4:
        h, w = out.shape[:2]
        crop_ratio = rng.uniform(0.92, 0.98)
        nw, nh = int(w * crop_ratio), int(h * crop_ratio)
        if nw > 50 and nh > 50:
            x = rng.randint(0, max(1, w - nw + 1))
            y = rng.randint(0, max(1, h - nh + 1))
            crop = out[y:y+nh, x:x+nw]
            out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    return out

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def classification_report_from_cm(cm, class_names):
    num_classes = len(class_names)
    rows = []

    total = int(cm.sum())
    acc = float(np.trace(cm) / max(total, 1))

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0

    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        rows.append({
            "class": class_names[i],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })

    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": rows
    }

def render_confusion_matrix_image(cm, class_names, out_path):
    cell = 120
    margin_left = 180
    margin_top = 120
    n = len(class_names)
    width = margin_left + n * cell + 40
    height = margin_top + n * cell + 60

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.putText(canvas, "Confusion Matrix", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)

    maxv = max(int(cm.max()), 1)

    for i, name in enumerate(class_names):
        cv2.putText(canvas, name, (margin_left + i * cell + 10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, name, (10, margin_top + i * cell + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Predicted", (margin_left + 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "True", (55, margin_top - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    for r in range(n):
        for c in range(n):
            v = int(cm[r, c])
            intensity = 255 - int(215 * (v / maxv))
            color = (255, intensity, intensity)
            x1 = margin_left + c * cell
            y1 = margin_top + r * cell
            x2 = x1 + cell
            y2 = y1 + cell

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), 1)
            cv2.putText(canvas, str(v), (x1 + 38, y1 + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)

class MultiClassBeautyModel:
    """
    轻量 softmax 多分类器
    """
    def __init__(self):
        self.feature_names = FEATURE_NAMES[:]
        self.class_names = []
        self.mean = None
        self.std = None
        self.W = None
        self.b = None

    def fit(self, X, y, class_names, epochs=2600, lr=0.05, l2=1e-3):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        self.class_names = list(class_names)

        n, d = X.shape
        k = len(self.class_names)

        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.mean) / self.std

        self.W = np.zeros((d, k), dtype=np.float32)
        self.b = np.zeros((1, k), dtype=np.float32)

        Y = np.zeros((n, k), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        for _ in range(epochs):
            logits = Xn @ self.W + self.b
            logits -= logits.max(axis=1, keepdims=True)
            expv = np.exp(logits)
            P = expv / np.clip(expv.sum(axis=1, keepdims=True), 1e-12, None)

            grad_W = (Xn.T @ (P - Y)) / n + l2 * self.W
            grad_b = np.mean(P - Y, axis=0, keepdims=True)

            self.W -= lr * grad_W
            self.b -= lr * grad_b

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn = (X - self.mean) / self.std
        logits = Xn @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        expv = np.exp(logits)
        return expv / np.clip(expv.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)

    def predict_from_dict(self, feat_dict):
        vec = np.array([feat_dict.get(k, 0.0) for k in self.feature_names], dtype=np.float32)
        proba = self.predict_proba(vec)[0]
        idx = int(np.argmax(proba))
        return self.class_names[idx], float(proba[idx]), proba

    def save(self, path):
        np.savez(
            path,
            feature_names=np.array(self.feature_names, dtype=object),
            class_names=np.array(self.class_names, dtype=object),
            mean=self.mean,
            std=self.std,
            W=self.W,
            b=self.b
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        obj = cls()
        obj.feature_names = list(data["feature_names"])
        obj.class_names = list(data["class_names"])
        obj.mean = data["mean"]
        obj.std = data["std"]
        obj.W = data["W"]
        obj.b = data["b"]
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
        self.classifier = None

        self.reset_temporal()

        if self.face_model_path and os.path.isfile(self.face_model_path):
            self._create_face_landmarker()
        if self.pose_model_path and os.path.isfile(self.pose_model_path):
            self._create_pose_landmarker()

    def _find_default_face_model(self):
        p = os.path.join(self.models_dir, "face_landmarker.task")
        return p if os.path.isfile(p) else None

    def _find_default_pose_model(self):
        for name in [
            "pose_landmarker_full.task",
            "pose_landmarker_lite.task",
            "pose_landmarker_heavy.task",
            "pose_landmarker.task"
        ]:
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

    def set_face_model_path(self, path):
        self.face_model_path = path
        self.reset_temporal()
        self._create_face_landmarker()

    def set_pose_model_path(self, path):
        self.pose_model_path = path
        self._create_pose_landmarker()

    def ready_face(self):
        return self.face_landmarker is not None

    def ready_pose(self):
        return self.pose_landmarker is not None

    def reset_temporal(self):
        self.prev_gray = None
        self.prev_ring_edges = None
        self.prev_center = None
        self.prev_heuristic = None

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

    def _build_feature_dict(self, global_metrics, roi_scores, geo, temporal, heuristic_score, texture_score):
        return {
            "roi_forehead_score": float(roi_scores["forehead"]),
            "roi_left_cheek_score": float(roi_scores["left_cheek"]),
            "roi_right_cheek_score": float(roi_scores["right_cheek"]),
            "roi_nose_score": float(roi_scores["nose"]),
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

    def _rule_based_subclass(self, feature_dict):
        """
        无训练模型时的 4类规则退路
        """
        tex = float(feature_dict["texture_score"])
        geo = float(feature_dict["geo_score"])
        temporal = float(feature_dict["temporal_score"])
        smooth = float(feature_dict["global_smoothness"])
        detail_loss = float(feature_dict["global_detail_loss"])

        skin_evidence = 0.65 * tex + 0.20 * smooth + 0.15 * detail_loss
        geom_evidence = 0.75 * geo + 0.25 * temporal

        if skin_evidence < 0.42 and geom_evidence < 0.32:
            label = "natural"
            conf = 1.0 - max(skin_evidence, geom_evidence)
        elif skin_evidence >= 0.42 and geom_evidence < 0.38:
            label = "skin-smoothing"
            conf = max(0.55, min(0.98, skin_evidence))
        elif skin_evidence < 0.42 and geom_evidence >= 0.38:
            label = "geometry-warp"
            conf = max(0.55, min(0.98, geom_evidence))
        else:
            label = "mixed"
            conf = max(0.55, min(0.99, 0.5 * (skin_evidence + geom_evidence)))

        return label, float(conf)

    def _make_conclusion_summary(self, pred_label, pred_prob, feature_dict, faces_count, poses_count):
        tex = feature_dict["texture_score"]
        geo = feature_dict["geo_score"]
        smooth = feature_dict["global_smoothness"]
        detail = feature_dict["global_detail_loss"]
        temporal = feature_dict["temporal_score"]

        reasons = []
        if tex >= 0.45:
            reasons.append("皮肤纹理/细节损失明显")
        if smooth >= 0.45:
            reasons.append("皮肤区域平滑度偏高")
        if geo >= 0.40:
            reasons.append("五官或脸型几何异常偏高")
        if temporal >= 0.35:
            reasons.append("视频时序形变/边缘波动偏高")

        if not reasons:
            reasons.append("未发现明显磨皮或几何变形迹象")

        mapping = {
            "natural": "更接近自然人脸表现",
            "skin-smoothing": "更像磨皮/美白类美颜",
            "geometry-warp": "更像瘦脸/大眼类几何美颜",
            "mixed": "同时存在磨皮与几何美颜迹象"
        }

        text = [
            f"结论：{mapping.get(pred_label, pred_label)}",
            f"分类标签：{pred_label}  |  置信度：{pred_prob:.3f}",
            f"检测对象：人脸 {faces_count} 个，人体 {poses_count} 个",
            f"主要依据：{'; '.join(reasons)}",
            f"特征摘要：texture={tex:.3f}, geo={geo:.3f}, smooth={smooth:.3f}, detail_loss={detail:.3f}, temporal={temporal:.3f}"
        ]
        return "\n".join(text)

    def _analyze_single_face_basic(self, frame_bgr, lm, gray, lab):
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

        roi_scores = {
            "forehead": float(forehead_metrics["score"]),
            "left_cheek": float(left_cheek_metrics["score"]),
            "right_cheek": float(right_cheek_metrics["score"]),
            "nose": float(nose_metrics["score"]),
        }

        roi_score = (
            0.23 * roi_scores["forehead"] +
            0.29 * roi_scores["left_cheek"] +
            0.29 * roi_scores["right_cheek"] +
            0.19 * roi_scores["nose"]
        )
        texture_score = 0.45 * global_metrics["score"] + 0.55 * roi_score
        geo = self._compute_geometry(lm)

        x, y, bw, bh = masks["bbox"]

        return {
            "lm": lm,
            "bbox": (x, y, bw, bh),
            "bbox_area": float(bw * bh),
            "masks": masks,
            "texture_score": float(texture_score),
            "geo": geo,
            "global_metrics": global_metrics,
            "roi_scores": roi_scores
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
                "conclusion_summary": "未加载 face model。",
                "faces_count": 0,
                "poses_count": 0,
                "heuristic_score": 0.0,
                "pred_label": "n/a",
                "pred_prob": 0.0,
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
                faces.append(self._analyze_single_face_basic(frame_bgr, lm, gray, lab))

        poses = self._analyze_poses(pose_result, frame_bgr.shape)

        # draw pose
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
                cv2.putText(out, f"P{i} vis={pose['avg_vis']:.2f}", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        if not faces:
            cv2.putText(out, "No face detected", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2, cv2.LINE_AA)

            return {
                "ok": False,
                "frame": out,
                "summary": f"No face detected.\nWhole body count: {len(poses)}",
                "conclusion_summary": f"未检测到人脸。人体数：{len(poses)}",
                "faces_count": 0,
                "poses_count": len(poses),
                "heuristic_score": 0.0,
                "pred_label": "no_face",
                "pred_prob": 0.0,
                "feature_dict": {},
                "feature_vector": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
                "faces": [],
                "poses": poses,
            }

        faces.sort(key=lambda d: d["bbox_area"], reverse=True)
        primary = faces[0]

        base_heuristic = 100.0 * (0.72 * primary["texture_score"] + 0.28 * primary["geo"]["geo_score"])
        temporal = self._compute_temporal(
            gray,
            primary["masks"]["face"],
            primary["geo"]["face_center"],
            base_heuristic
        ) if not static_image else {"flicker": 0.0, "warp": 0.0, "jump": 0.0, "score": 0.0}

        heuristic_score = base_heuristic if static_image else 100.0 * (
            0.50 * primary["texture_score"] +
            0.22 * primary["geo"]["geo_score"] +
            0.28 * temporal["score"]
        )
        heuristic_score = float(clamp(heuristic_score, 0.0, 100.0))

        feature_dict = self._build_feature_dict(
            primary["global_metrics"],
            primary["roi_scores"],
            primary["geo"],
            temporal,
            heuristic_score,
            primary["texture_score"]
        )
        feature_vector = np.array([feature_dict[k] for k in FEATURE_NAMES], dtype=np.float32)

        if self.classifier is not None:
            pred_label, pred_prob, _ = self.classifier.predict_from_dict(feature_dict)
        else:
            pred_label, pred_prob = self._rule_based_subclass(feature_dict)

        # draw faces
        for i, face in enumerate(faces, 1):
            score = 100.0 * (0.72 * face["texture_score"] + 0.28 * face["geo"]["geo_score"])
            color = (0, 200, 0) if score < 35 else ((0, 185, 255) if score < 60 else (0, 0, 255))
            masks = face["masks"]
            lm = face["lm"]

            blend_mask(out, masks["forehead"], (255, 128, 0), 0.16)
            blend_mask(out, masks["left_cheek"], (255, 0, 180), 0.16)
            blend_mask(out, masks["right_cheek"], (180, 0, 255), 0.16)
            blend_mask(out, masks["nose"], (0, 180, 255), 0.16)

            face_poly = np.array([lm[j] for j in self.FACE_OVAL], dtype=np.int32)
            cv2.polylines(out, [face_poly], True, color, 2, cv2.LINE_AA)

            x, y, bw, bh = face["bbox"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            if i == 1:
                label_text = f"F{i} {pred_label} {pred_prob:.2f}"
            else:
                label_text = f"F{i}"
            cv2.putText(out, label_text, (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        line1 = f"Faces={len(faces)}  Bodies={len(poses)}"
        line2 = f"Primary: {pred_label} ({pred_prob:.2f})  heuristic={heuristic_score:.1f}"
        line3 = f"Texture={primary['texture_score']:.2f}  Geo={primary['geo']['geo_score']:.2f}  Temporal={temporal['score']:.2f}"

        cv2.putText(out, line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line2, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line3, (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)

        conclusion_summary = self._make_conclusion_summary(
            pred_label, pred_prob, feature_dict, len(faces), len(poses)
        )

        summary_lines = [
            f"Faces detected: {len(faces)}",
            f"Whole bodies detected: {len(poses)}",
            f"Primary label: {pred_label}",
            f"Primary confidence: {pred_prob:.4f}",
            f"Heuristic score: {heuristic_score:.4f}",
            f"Texture score: {primary['texture_score']:.4f}",
            f"ROI forehead={primary['roi_scores']['forehead']:.4f}  left cheek={primary['roi_scores']['left_cheek']:.4f}  right cheek={primary['roi_scores']['right_cheek']:.4f}  nose={primary['roi_scores']['nose']:.4f}",
            f"Geo score={primary['geo']['geo_score']:.4f}  big-eye={primary['geo']['big_eye_score']:.4f}  slim-face={primary['geo']['slim_face_score']:.4f}",
            f"Temporal total={temporal['score']:.4f}  flicker={temporal['flicker']:.4f}  warp={temporal['warp']:.4f}  jump={temporal['jump']:.4f}",
        ]

        return {
            "ok": True,
            "frame": out,
            "summary": "\n".join(summary_lines),
            "conclusion_summary": conclusion_summary,
            "faces_count": len(faces),
            "poses_count": len(poses),
            "faces": faces,
            "poses": poses,
            "heuristic_score": heuristic_score,
            "pred_label": pred_label,
            "pred_prob": float(pred_prob),
            "texture_score": float(primary["texture_score"]),
            "roi_scores": primary["roi_scores"],
            "geo": primary["geo"],
            "temporal": temporal,
            "feature_dict": feature_dict,
            "feature_vector": feature_vector,
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
        self.last_validation_summary = ""

        self.setWindowTitle(self._title_text())
        self.resize(1550, 980)

        self._build_ui()
        self._apply_style()

        if not self.analyzer.ready_face():
            self.log_msg("Face model not loaded. Please click 'Choose Face Model'.")
        if not self.analyzer.ready_pose():
            self.log_msg("Pose model not loaded. Whole body detect will be unavailable until you load one.")

    def _title_text(self):
        face_name = os.path.basename(self.analyzer.face_model_path) if self.analyzer.face_model_path else "no_face_model"
        pose_name = os.path.basename(self.analyzer.pose_model_path) if self.analyzer.pose_model_path else "no_pose_model"
        return f"Beauty Detector Pro - face:{face_name} | pose:{pose_name}"

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = QVBoxLayout()
        self.preview = QLabel("Open image / video / webcam")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(940, 760)
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
        self.btn_train = QPushButton("Train Multi-Class Model")
        self.btn_validate = QPushButton("Validate Dataset")
        self.btn_load_model = QPushButton("Load Classifier")
        self.btn_save_model = QPushButton("Save Classifier")
        self.btn_reset_temporal = QPushButton("Reset Temporal")

        self.btn_choose_face_model.clicked.connect(self.choose_face_model)
        self.btn_choose_pose_model.clicked.connect(self.choose_pose_model)
        self.btn_train.clicked.connect(self.train_multiclass_model)
        self.btn_validate.clicked.connect(self.validate_dataset)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_reset_temporal.clicked.connect(self.reset_temporal)

        for b in [
            self.btn_choose_face_model, self.btn_choose_pose_model,
            self.btn_train, self.btn_validate,
            self.btn_load_model, self.btn_save_model,
            self.btn_reset_temporal
        ]:
            btn_row2.addWidget(b)
        left.addLayout(btn_row2)

        right = QVBoxLayout()

        g1 = QGroupBox("Result")
        g1l = QGridLayout(g1)
        self.lbl_faces = QLabel("-")
        self.lbl_poses = QLabel("-")
        self.lbl_pred = QLabel("-")
        self.lbl_prob = QLabel("-")
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
        g1l.addWidget(QLabel("Pred Label:"), 2, 0)
        g1l.addWidget(self.lbl_pred, 2, 1)
        g1l.addWidget(QLabel("Probability:"), 3, 0)
        g1l.addWidget(self.lbl_prob, 3, 1)
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

        g2 = QGroupBox("Detection Conclusion")
        g2l = QVBoxLayout(g2)
        self.conclusion = QTextEdit()
        self.conclusion.setReadOnly(True)
        g2l.addWidget(self.conclusion)

        g3 = QGroupBox("Details")
        g3l = QVBoxLayout(g3)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        g3l.addWidget(self.details)

        g4 = QGroupBox("Log / Validation Summary")
        g4l = QVBoxLayout(g4)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        g4l.addWidget(self.log)

        right.addWidget(g1)
        right.addWidget(g2, 1)
        right.addWidget(g3, 1)
        right.addWidget(g4, 1)

        main_layout.addLayout(left, 3)
        main_layout.addLayout(right, 2)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background:#20242b; color:#e8e8e8; }
            QWidget { color:#e8e8e8; font-size:13px; }
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

    def _scan_dataset(self, root_dir, class_names):
        dataset = []
        for cls in class_names:
            folder = os.path.join(root_dir, cls)
            for path in list_images(folder):
                dataset.append((path, cls))
        return dataset

    def _extract_feature_from_image(self, img):
        if img is None:
            return None
        if img.shape[1] > 1400:
            scale = 1400.0 / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        res = self.analyzer.analyze(img, static_image=True)
        if not res["ok"]:
            return None
        return res

    def train_multiclass_model(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        dataset_root = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if not dataset_root:
            return

        class_names = discover_classes(dataset_root)
        if not class_names:
            QMessageBox.warning(
                self, "Dataset Error",
                "Cannot discover valid classes.\nNeed at least:\n"
                "natural + skin-smoothing + geometry-warp\n"
                "mixed is optional."
            )
            return

        train_root = find_split_root(dataset_root, "train")
        val_root = find_split_root(dataset_root, "val")

        train_items = self._scan_dataset(train_root, class_names)
        if len(train_items) < 12:
            QMessageBox.warning(self, "Dataset Error", "Too few training images.")
            return

        self.log_msg(f"Training classes: {class_names}")
        if "mixed" not in class_names:
            self.log_msg("Mixed class not found -> auto fallback to 3 classes.")

        X, y = [], []
        train_feature_csv_rows = []
        rng = np.random.RandomState(42)

        class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.log_msg(f"Training samples scan started. Raw images: {len(train_items)}")

        for idx, (path, cls) in enumerate(train_items, 1):
            img = cv2.imread(path)
            if img is None:
                self.log_msg(f"Skip unreadable: {path}")
                continue

            # original
            res = self._extract_feature_from_image(img)
            if res is not None:
                X.append(res["feature_vector"])
                y.append(class_to_idx[cls])
                train_feature_csv_rows.append([path, cls, "original"] + [res["feature_dict"][k] for k in FEATURE_NAMES])

            # automatic augmentation
            for aug_i in range(2):
                aug = augment_image(img, rng)
                res_aug = self._extract_feature_from_image(aug)
                if res_aug is not None:
                    X.append(res_aug["feature_vector"])
                    y.append(class_to_idx[cls])
                    train_feature_csv_rows.append([path, cls, f"aug_{aug_i+1}"] + [res_aug["feature_dict"][k] for k in FEATURE_NAMES])

            if idx % 10 == 0:
                self.log_msg(f"Processed {idx}/{len(train_items)} training images")

        if len(X) < 20:
            QMessageBox.warning(self, "Training Error", "Too few valid face samples after filtering.")
            return

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)

        model = MultiClassBeautyModel()
        model.fit(X, y, class_names=class_names)
        self.analyzer.classifier = model

        report_root = os.path.join(dataset_root, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_root, exist_ok=True)

        feature_csv = os.path.join(report_root, "train_features.csv")
        with open(feature_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["source_file", "label", "variant"] + FEATURE_NAMES)
            writer.writerows(train_feature_csv_rows)

        self.log_msg(f"Training finished. Feature CSV: {feature_csv}")
        self.log_msg(f"Classifier classes: {self.analyzer.classifier.class_names}")

        # optional immediate val
        if os.path.isdir(os.path.join(dataset_root, "val")):
            self._run_validation(val_root, report_root, class_names, title_prefix="VAL")

        QMessageBox.information(
            self, "Training Done",
            f"Training complete.\n\nClasses: {class_names}\nReport folder:\n{report_root}"
        )

    def _predict_feature_dict(self, feature_dict):
        if self.analyzer.classifier is not None:
            label, prob, proba = self.analyzer.classifier.predict_from_dict(feature_dict)
            return label, prob, proba
        label, prob = self.analyzer._rule_based_subclass(feature_dict)
        class_names = ALL_CLASS_NAMES[:]  # for fallback display only
        proba = np.zeros(len(class_names), dtype=np.float32)
        if label in class_names:
            proba[class_names.index(label)] = prob
        return label, prob, proba

    def _run_validation(self, val_root, report_root, class_names, title_prefix="VAL"):
        items = self._scan_dataset(val_root, class_names)
        if len(items) == 0:
            self.log_msg(f"{title_prefix}: no validation images found.")
            return None

        y_true, y_pred = [], []
        predictions_rows = []
        class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.log_msg(f"{title_prefix}: validating {len(items)} images...")

        for idx, (path, cls) in enumerate(items, 1):
            img = cv2.imread(path)
            if img is None:
                continue
            res = self._extract_feature_from_image(img)
            if res is None:
                continue

            pred_label, pred_prob, proba = self._predict_feature_dict(res["feature_dict"])

            # 若当前训练模型是3类，但规则可能给mixed，则映射回 geometry-warp 或 skin-smoothing
            if pred_label not in class_names:
                if pred_label == "mixed":
                    if res["feature_dict"]["geo_score"] >= res["feature_dict"]["texture_score"]:
                        pred_label = "geometry-warp" if "geometry-warp" in class_names else class_names[-1]
                    else:
                        pred_label = "skin-smoothing" if "skin-smoothing" in class_names else class_names[-1]
                else:
                    pred_label = class_names[0]

            y_true.append(class_to_idx[cls])
            y_pred.append(class_to_idx[pred_label])

            row = [path, cls, pred_label, pred_prob]
            row.extend([res["feature_dict"][k] for k in FEATURE_NAMES])
            predictions_rows.append(row)

            if idx % 20 == 0:
                self.log_msg(f"{title_prefix}: processed {idx}/{len(items)}")

        if len(y_true) == 0:
            self.log_msg(f"{title_prefix}: no valid samples with detected faces.")
            return None

        cm = confusion_matrix_np(y_true, y_pred, len(class_names))
        report = classification_report_from_cm(cm, class_names)

        cm_png = os.path.join(report_root, f"{title_prefix.lower()}_confusion_matrix.png")
        render_confusion_matrix_image(cm, class_names, cm_png)

        pred_csv = os.path.join(report_root, f"{title_prefix.lower()}_predictions.csv")
        with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "true_label", "pred_label", "pred_prob"] + FEATURE_NAMES)
            writer.writerows(predictions_rows)

        report_csv = os.path.join(report_root, f"{title_prefix.lower()}_metrics.csv")
        with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["accuracy", report["accuracy"]])
            writer.writerow(["macro_precision", report["macro_precision"]])
            writer.writerow(["macro_recall", report["macro_recall"]])
            writer.writerow(["macro_f1", report["macro_f1"]])
            writer.writerow([])
            writer.writerow(["class", "precision", "recall", "f1", "support"])
            for row in report["per_class"]:
                writer.writerow([row["class"], row["precision"], row["recall"], row["f1"], row["support"]])

        summary_lines = [
            f"{title_prefix} Accuracy: {report['accuracy']:.4f}",
            f"{title_prefix} Macro Precision: {report['macro_precision']:.4f}",
            f"{title_prefix} Macro Recall: {report['macro_recall']:.4f}",
            f"{title_prefix} Macro F1: {report['macro_f1']:.4f}",
            "Per-class:"
        ]
        for row in report["per_class"]:
            summary_lines.append(
                f"  {row['class']}: P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, support={row['support']}"
            )

        summary_lines.append(f"Confusion matrix image: {cm_png}")
        summary_lines.append(f"Predictions CSV: {pred_csv}")
        summary_lines.append(f"Metrics CSV: {report_csv}")

        summary = "\n".join(summary_lines)
        self.log_msg(summary)
        self.last_validation_summary = summary
        self.conclusion.setPlainText(summary)
        return summary

    def validate_dataset(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        dataset_root = QFileDialog.getExistingDirectory(self, "Select Validation Dataset Root")
        if not dataset_root:
            return

        if self.analyzer.classifier is not None:
            class_names = self.analyzer.classifier.class_names
        else:
            class_names = discover_classes(dataset_root)

        if not class_names:
            QMessageBox.warning(self, "Validation Error", "Cannot discover valid classes.")
            return

        val_root = find_split_root(dataset_root, "val")
        report_root = os.path.join(dataset_root, f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(report_root, exist_ok=True)

        summary = self._run_validation(val_root, report_root, class_names, title_prefix="VAL")
        if summary is None:
            QMessageBox.warning(self, "Validation Error", "No valid validation samples found.")
            return

        QMessageBox.information(self, "Validation Done", f"Validation finished.\n\nReport folder:\n{report_root}")

    def save_model(self):
        if self.analyzer.classifier is None:
            QMessageBox.information(self, "Info", "No trained classifier available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Multi-Class Classifier", "beauty_multiclass_model.npz", "NPZ (*.npz)"
        )
        if not path:
            return
        self.analyzer.classifier.save(path)
        self.log_msg(f"Classifier saved: {path}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Multi-Class Classifier", "", "NPZ (*.npz)"
        )
        if not path:
            return
        try:
            self.analyzer.classifier = MultiClassBeautyModel.load(path)
            self.log_msg(f"Classifier loaded: {path}")
            self.log_msg(f"Classes: {self.analyzer.classifier.class_names}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load classifier:\n{e}")

    def open_image(self):
        self.stop_worker()
        if not self._ensure_face_model():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
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
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
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
        pred_label = result.get("pred_label", "-")
        pred_prob = result.get("pred_prob", 0.0)
        ts = result.get("texture_score", 0.0)
        geo = result.get("geo", {})
        temporal = result.get("temporal", {})
        roi = result.get("roi_scores", {})
        rolling = result.get("rolling_avg", None)
        fps = result.get("fps", None)

        self.lbl_faces.setText(str(result.get("faces_count", 0)))
        self.lbl_poses.setText(str(result.get("poses_count", 0)))
        self.lbl_pred.setText(pred_label)
        self.lbl_prob.setText(f"{pred_prob:.3f}")
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
        self.conclusion.setPlainText(result.get("conclusion_summary", ""))

        self.bar.setValue(int(clamp(hs, 0, 100)))
        if pred_label == "natural":
            color = "#22c55e"
        elif pred_label == "skin-smoothing":
            color = "#f59e0b"
        elif pred_label == "geometry-warp":
            color = "#ef4444"
        else:
            color = "#a855f7"

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
            self, "Save Frame", "beauty_result.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return

        if cv2.imwrite(path, self.last_frame):
            self.log_msg(f"Saved frame: {path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save frame.")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()