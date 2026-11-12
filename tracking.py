"""동영상에서 사람을 추적하고 Qt 화면에 프레임별 결과를 전달합니다."""
from collections import deque
from math import atan2, degrees, hypot, isfinite
from threading import Condition, Event
from time import perf_counter

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from models_runtime import MODEL_DIR, mosaic_faces, person_class, person_weights, track_people


class TrackingThread(QThread):
    metadata = Signal(int)
    frame_ready = Signal(QImage, list, int, float)
    failed = Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source
        self.control = Condition()
        self.paused = False
        self.cancelled = False
        self.displayed = Event()

    def set_paused(self, paused):
        with self.control:
            self.paused = paused
            self.control.notify_all()

    def cancel(self):
        with self.control:
            self.cancelled = True
            self.control.notify_all()
        self.displayed.set()

    def run(self):
        try:
            self.track_video()
        except Exception as error:
            self.failed.emit(str(error))

    def track_video(self):
        import cv2
        from ultralytics import YOLO

        capture = cv2.VideoCapture(self.source)
        try:
            ok, frame = capture.read()
            if not ok:
                raise ValueError("영상 프레임을 읽을 수 없습니다.")
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if isfinite(fps) and fps > 0 else 30.0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = round(frame_count * 1000 / fps) if isfinite(frame_count) else 0
            self.metadata.emit(max(0, duration))

            model = YOLO(person_weights())
            class_id = person_class(model)
            face_weights = MODEL_DIR / "face.pt"
            face_model = YOLO(str(face_weights)) if face_weights.is_file() else None
            if face_model is not None and set(face_model.names.values()) != {"face"}:
                raise ValueError("models/face.pt는 face 단일 클래스 가중치여야 합니다.")
            tracks = {}
            index = 0
            while ok:
                with self.control:
                    self.control.wait_for(lambda: not self.paused or self.cancelled)
                    if self.cancelled:
                        return

                started = perf_counter()
                boxes = track_people(model, frame, class_id)
                if face_model is not None:
                    mosaic_faces(frame, face_model)
                elapsed = perf_counter() - started
                rows = update_tracks(boxes, tracks)
                height, width, _ = frame.shape
                image = QImage(
                    frame.data, width, height, frame.strides[0], QImage.Format_BGR888
                ).copy()
                # 화면이 소비할 때까지 기다려 프레임이 신호 큐에 누적되는 것을 막습니다.
                self.displayed.clear()
                with self.control:
                    if self.cancelled:
                        return
                self.frame_ready.emit(image, rows, round(index * 1000 / fps), elapsed)
                self.displayed.wait()
                with self.control:
                    if self.cancelled:
                        return
                index += 1
                ok, frame = capture.read()
        finally:
            capture.release()


def update_tracks(boxes, tracks):
    """추적기가 부여한 ID를 기준으로 좌표와 누적 화면 이동 거리를 기록합니다."""
    if boxes is None or boxes.id is None:
        return []
    rows = []
    for track_id, coordinates in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist()):
        x1, y1, x2, y2 = coordinates
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        trail, distance = tracks.get(track_id, (deque(maxlen=45), 0.0))
        direction = "—"
        if trail:
            last_x, last_y = trail[-1]
            dx, dy = x - last_x, y - last_y
            step = hypot(dx, dy)
            distance += step
            if step >= 1:
                octant = round(degrees(atan2(dy, dx)) / 45) % 8
                direction = ("→", "↘", "↓", "↙", "←", "↖", "↑", "↗")[octant]
        trail.append((x, y))
        tracks[track_id] = trail, distance
        rows.append((track_id, coordinates, tuple(trail), direction, distance))
    return sorted(rows, key=lambda row: row[0])
