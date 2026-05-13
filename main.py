import sys
import os
import time
import cv2
import numpy as np
from collections import deque

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QProgressBar,
    QGroupBox, QSizePolicy, QMessageBox
)

import mediapipe as mp

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cv_to_pixmap(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

class BeautyAnalyzer:
    """
    Heuristic beauty-filter detector:
    - Face landmarks
    - Face skin mask
    - Texture / high-frequency loss
    - Bilateral residual
    - Skin chroma uniformity
    - Internal-vs-contour edge ratio
    """

    # Ordered face oval indices for polygon mask
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
        361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
        176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
        162, 21, 54, 103, 67, 109
    ]

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.static_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.video_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _landmarks_to_pixels(self, landmarks, width, height):
        pts = []
        for lm in landmarks.landmark:
            x = int(clamp(lm.x, 0.0, 1.0) * (width - 1))
            y = int(clamp(lm.y, 0.0, 1.0) * (height - 1))
            pts.append((x, y))
        return pts

    def _poly_mask(self, shape_hw, lm_pixels, indices):
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array([lm_pixels[i] for i in indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
        return mask

    def _build_skin_mask(self, frame_shape, lm_pixels):
        h, w = frame_shape[:2]

        face_mask = self._poly_mask((h, w), lm_pixels, self.FACE_OVAL)
        left_eye_mask = self._poly_mask((h, w), lm_pixels, self.LEFT_EYE)
        right_eye_mask = self._poly_mask((h, w), lm_pixels, self.RIGHT_EYE)
        lips_mask = self._poly_mask((h, w), lm_pixels, self.OUTER_LIPS)
        nose_mask = self._poly_mask((h, w), lm_pixels, self.NOSE)

        skin_mask = face_mask.copy()
        skin_mask[left_eye_mask > 0] = 0
        skin_mask[right_eye_mask > 0] = 0
        skin_mask[lips_mask > 0] = 0
        skin_mask[nose_mask > 0] = 0

        kernel = np.ones((7, 7), np.uint8)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
        _, skin_mask = cv2.threshold(skin_mask, 20, 255, cv2.THRESH_BINARY)

        return face_mask, skin_mask

    def _compute_metrics(self, frame_bgr, face_mask, skin_mask):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

        if cv2.countNonZero(skin_mask) < 1000:
            return None

        mask_idx = skin_mask > 0

        # 1) high-frequency texture
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(np.var(lap[mask_idx]))

        # 2) bilateral residual (already-smoothed skin -> lower residual)
        bilateral = cv2.bilateralFilter(gray, 9, 35, 35)
        residual = np.abs(gray.astype(np.float32) - bilateral.astype(np.float32))
        residual_mean = float(np.mean(residual[mask_idx]))

        # 3) chroma uniformity
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        chroma_std = float((np.std(a[mask_idx]) + np.std(b[mask_idx])) / 2.0)

        # 4) internal edge density vs face contour edge density
        edges = cv2.Canny(gray, 60, 120)
        internal_edge_density = float(np.mean(edges[mask_idx] > 0) * 100.0)

        ring = cv2.dilate(face_mask, np.ones((11, 11), np.uint8), 1) - \
               cv2.erode(face_mask, np.ones((11, 11), np.uint8), 1)
        ring_idx = ring > 0
        contour_edge_density = float(np.mean(edges[ring_idx] > 0) * 100.0) if np.any(ring_idx) else 1.0
        texture_ratio = internal_edge_density / max(contour_edge_density, 1e-6)

        # 5) brightness hint (mild whitening clue only)
        lch = lab[:, :, 0]
        lightness_mean = float(np.mean(lch[mask_idx]))

        # Normalize to suspicion-oriented scores
        smoothness_score = clamp(1.0 - lap_var / 180.0, 0.0, 1.0)
        detail_loss_score = clamp(1.0 - residual_mean / 12.0, 0.0, 1.0)
        uniformity_score = clamp(1.0 - chroma_std / 14.0, 0.0, 1.0)
        edge_ratio_score = clamp(1.0 - texture_ratio / 1.2, 0.0, 1.0)
        whitening_hint_score = clamp((lightness_mean - 150.0) / 55.0, 0.0, 1.0)

        beauty_score = (
            0.34 * smoothness_score +
            0.26 * detail_loss_score +
            0.20 * uniformity_score +
            0.12 * edge_ratio_score +
            0.08 * whitening_hint_score
        ) * 100.0

        beauty_score = float(clamp(beauty_score, 0.0, 100.0))

        if beauty_score < 35:
            level = "Low suspicion"
        elif beauty_score < 60:
            level = "Moderate suspicion"
        else:
            level = "High suspicion"

        return {
            "lap_var": lap_var,
            "residual_mean": residual_mean,
            "chroma_std": chroma_std,
            "internal_edge_density": internal_edge_density,
            "contour_edge_density": contour_edge_density,
            "texture_ratio": texture_ratio,
            "lightness_mean": lightness_mean,
            "beauty_score": beauty_score,
            "level": level
        }

    def analyze(self, frame_bgr, static_image=False):
        out = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mesh = self.static_mesh if static_image else self.video_mesh
        result = mesh.process(rgb)

        response = {
            "ok": False,
            "frame": out,
            "beauty_score": 0.0,
            "level": "No face",
            "metrics": None
        }

        if not result.multi_face_landmarks:
            cv2.putText(out, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            response["frame"] = out
            return response

        h, w = frame_bgr.shape[:2]
        lm_pixels = self._landmarks_to_pixels(result.multi_face_landmarks[0], w, h)
        face_mask, skin_mask = self._build_skin_mask(frame_bgr.shape, lm_pixels)
        metrics = self._compute_metrics(frame_bgr, face_mask, skin_mask)

        if metrics is None:
            cv2.putText(out, "Face detected, skin ROI too small", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
            response["frame"] = out
            return response

        # overlay
        overlay = out.copy()
        score = metrics["beauty_score"]
        if score < 35:
            color = (0, 200, 0)
        elif score < 60:
            color = (0, 180, 255)
        else:
            color = (0, 0, 255)

        overlay[skin_mask > 0] = (
            int(0.35 * color[0] + 0.65 * overlay[skin_mask > 0][:, 0]),
            int(0.35 * color[1] + 0.65 * overlay[skin_mask > 0][:, 1]),
            int(0.35 * color[2] + 0.65 * overlay[skin_mask > 0][:, 2]),
        )
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

        face_poly = np.array([lm_pixels[i] for i in self.FACE_OVAL], dtype=np.int32)
        cv2.polylines(out, [face_poly], True, color, 2, cv2.LINE_AA)

        cv2.putText(out, f"Beauty suspicion: {score:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
        cv2.putText(out, metrics["level"], (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        response.update({
            "ok": True,
            "frame": out,
            "beauty_score": score,
            "level": metrics["level"],
            "metrics": metrics
        })
        return response

class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, object)
    status_ready = pyqtSignal(str)

    def __init__(self, source, analyzer, is_camera=False):
        super().__init__()
        self.source = source
        self.analyzer = analyzer
        self.is_camera = is_camera
        self._running = True
        self.score_history = deque(maxlen=60)

    def stop(self):
        self._running = False

    def run(self):
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

            result = self.analyzer.analyze(frame, static_image=False)
            self.score_history.append(result["beauty_score"])
            rolling_avg = float(np.mean(self.score_history)) if self.score_history else 0.0
            result["rolling_avg"] = rolling_avg

            now = time.time()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            result["fps"] = fps

            self.frame_ready.emit(result["frame"], result)

            if not self.is_camera:
                time.sleep(0.01)

        cap.release()
        self.status_ready.emit("Video processing stopped.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beauty Filter Suspicion Detector - PyQt5/OpenCV/Face Landmarks")
        self.resize(1350, 860)

        self.analyzer = BeautyAnalyzer()
        self.worker = None
        self.last_frame = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)

        # Left: preview
        left_box = QVBoxLayout()
        self.preview = QLabel("Load an image / video or open webcam")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(900, 700)
        self.preview.setStyleSheet("background:#111;color:#ddd;border:1px solid #444;")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_box.addWidget(self.preview)

        btn_row = QHBoxLayout()
        self.btn_image = QPushButton("Open Image")
        self.btn_video = QPushButton("Open Video")
        self.btn_camera = QPushButton("Open Webcam")
        self.btn_stop = QPushButton("Stop")
        self.btn_save = QPushButton("Save Annotated Frame")

        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_worker)
        self.btn_save.clicked.connect(self.save_frame)

        for b in [self.btn_image, self.btn_video, self.btn_camera, self.btn_stop, self.btn_save]:
            btn_row.addWidget(b)

        left_box.addLayout(btn_row)

        # Right: metrics
        right_box = QVBoxLayout()

        score_group = QGroupBox("Detection Result")
        sg = QGridLayout(score_group)

        self.lbl_level = QLabel("No result")
        self.lbl_score = QLabel("0.0")
        self.lbl_avg = QLabel("-")
        self.lbl_fps = QLabel("-")

        self.bar_score = QProgressBar()
        self.bar_score.setRange(0, 100)
        self.bar_score.setValue(0)
        self.bar_score.setTextVisible(True)
        self.bar_score.setFormat("%p")

        sg.addWidget(QLabel("Suspicion Level:"), 0, 0)
        sg.addWidget(self.lbl_level, 0, 1)
        sg.addWidget(QLabel("Current Score:"), 1, 0)
        sg.addWidget(self.lbl_score, 1, 1)
        sg.addWidget(QLabel("Rolling Avg (video):"), 2, 0)
        sg.addWidget(self.lbl_avg, 2, 1)
        sg.addWidget(QLabel("FPS:"), 3, 0)
        sg.addWidget(self.lbl_fps, 3, 1)
        sg.addWidget(self.bar_score, 4, 0, 1, 2)

        metrics_group = QGroupBox("Raw Metrics")
        mg = QGridLayout(metrics_group)

        self.lbl_lap = QLabel("-")
        self.lbl_residual = QLabel("-")
        self.lbl_chroma = QLabel("-")
        self.lbl_inner_edge = QLabel("-")
        self.lbl_contour_edge = QLabel("-")
        self.lbl_ratio = QLabel("-")
        self.lbl_lightness = QLabel("-")

        rows = [
            ("Texture (Laplacian var)", self.lbl_lap),
            ("Residual after bilateral", self.lbl_residual),
            ("Skin chroma std", self.lbl_chroma),
            ("Internal edge density", self.lbl_inner_edge),
            ("Contour edge density", self.lbl_contour_edge),
            ("Texture/Contour ratio", self.lbl_ratio),
            ("Skin lightness mean", self.lbl_lightness),
        ]
        for i, (name, label) in enumerate(rows):
            mg.addWidget(QLabel(name + ":"), i, 0)
            mg.addWidget(label, i, 1)

        log_group = QGroupBox("Log")
        lg = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lg.addWidget(self.log)

        right_box.addWidget(score_group)
        right_box.addWidget(metrics_group)
        right_box.addWidget(log_group, 1)

        main_layout.addLayout(left_box, 3)
        main_layout.addLayout(right_box, 1)

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #20242b; color: #e8e8e8; }
            QWidget { color: #e8e8e8; font-size: 13px; }
            QGroupBox {
                border: 1px solid #4a5568;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                background: #262b33;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background: #3b82f6;
                border: none;
                padding: 8px 12px;
                border-radius: 6px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:pressed { background: #1d4ed8; }
            QTextEdit, QLabel {
                background: transparent;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background: #111827;
            }
            QProgressBar::chunk {
                background-color: #ef4444;
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

    def open_image(self):
        self.stop_worker()
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

        result = self.analyzer.analyze(frame, static_image=True)
        self.last_frame = result["frame"]
        self.preview.setPixmap(
            cv_to_pixmap(result["frame"]).scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.update_metrics(result)
        self.log_msg(f"Loaded image: {path}")

    def open_video(self):
        self.stop_worker()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if not path:
            return
        self.start_worker(path, is_camera=False)

    def open_camera(self):
        self.stop_worker()
        self.start_worker(0, is_camera=True)

    def start_worker(self, source, is_camera):
        self.worker = VideoWorker(source, self.analyzer, is_camera=is_camera)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_ready.connect(self.log_msg)
        self.worker.start()

    def on_frame_ready(self, frame, result):
        self.last_frame = frame
        pix = cv_to_pixmap(frame).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pix)
        self.update_metrics(result)

    def update_metrics(self, result):
        score = result.get("beauty_score", 0.0)
        level = result.get("level", "-")
        rolling_avg = result.get("rolling_avg", None)
        fps = result.get("fps", None)
        metrics = result.get("metrics", None)

        self.lbl_score.setText(f"{score:.2f}")
        self.lbl_level.setText(level)
        self.bar_score.setValue(int(score))

        if score < 35:
            color = "#22c55e"
        elif score < 60:
            color = "#f59e0b"
        else:
            color = "#ef4444"
        self.bar_score.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                background: #111827;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

        self.lbl_avg.setText(f"{rolling_avg:.2f}" if rolling_avg is not None else "-")
        self.lbl_fps.setText(f"{fps:.1f}" if fps is not None else "-")

        if metrics:
            self.lbl_lap.setText(f"{metrics['lap_var']:.3f}")
            self.lbl_residual.setText(f"{metrics['residual_mean']:.3f}")
            self.lbl_chroma.setText(f"{metrics['chroma_std']:.3f}")
            self.lbl_inner_edge.setText(f"{metrics['internal_edge_density']:.3f}")
            self.lbl_contour_edge.setText(f"{metrics['contour_edge_density']:.3f}")
            self.lbl_ratio.setText(f"{metrics['texture_ratio']:.3f}")
            self.lbl_lightness.setText(f"{metrics['lightness_mean']:.3f}")
        else:
            self.lbl_lap.setText("-")
            self.lbl_residual.setText("-")
            self.lbl_chroma.setText("-")
            self.lbl_inner_edge.setText("-")
            self.lbl_contour_edge.setText("-")
            self.lbl_ratio.setText("-")
            self.lbl_lightness.setText("-")

    def save_frame(self):
        if self.last_frame is None:
            QMessageBox.information(self, "Info", "No annotated frame available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Frame", "beauty_detection_result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.last_frame)
        if ok:
            self.log_msg(f"Saved annotated frame: {path}")
        else:
            QMessageBox.warning(self, "Error", "Failed to save frame.")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()