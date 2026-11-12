"""영상 재생 및 ByteTrack 사람 추적 GUI."""
import sys
import json
from pathlib import Path
from PySide6.QtCore import Qt, QUrl, QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QTabWidget, QTableWidget, QHeaderView,
    QCheckBox, QSlider, QFileDialog, QMessageBox, QGroupBox, QStackedWidget,
    QTableWidgetItem
)

from tracking import TrackingThread

STYLE = """
QWidget { background:#101923; color:#edf4fa; font-family:'Malgun Gothic'; font-size:14px; }
QMainWindow { background:#101923; }
QLabel#title { font-size:29px; font-weight:bold; }
QPushButton,QComboBox { background:#223140; border:1px solid #496073; border-radius:7px; padding:10px 18px; }
QPushButton:hover { border-color:#00d4e9; background:#293f50; }
QPushButton:disabled { color:#687786; }
QPushButton#primary { background:#00cddd; color:#081922; font-weight:bold; }
QTabWidget::pane { border:1px solid #344a5c; }
QTabBar::tab { background:#223140; padding:13px 28px; }
QTabBar::tab:selected { background:#00cddd; color:#081922; font-weight:bold; }
QGroupBox { border:1px solid #344a5c; border-radius:8px; margin-top:14px; padding:14px; }
QGroupBox::title { subcontrol-origin:margin; left:14px; }
QTableWidget { background:#16222e; gridline-color:#344a5c; border:0; }
QHeaderView::section { background:#223140; padding:8px; border:0; }
QSlider::groove:horizontal { height:6px; background:#435466; border-radius:3px; }
QSlider::handle:horizontal { background:#00d4e9; width:16px; margin:-5px 0; border-radius:8px; }
QCheckBox { padding:7px; }
"""

class TrackingView(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.rows = []
        self.display = {}

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0a1118"))
        if self.image is None:
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        painter.translate((self.width() - width) / 2, (self.height() - height) / 2)
        painter.scale(scale, scale)
        rect = QRectF(0, 0, self.image.width(), self.image.height())
        painter.setClipRect(rect)
        painter.drawImage(rect, self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        font = painter.font()
        font.setPixelSize(max(13, round(14 / scale)))
        painter.setFont(font)
        for track_id, box, trail, _, _ in self.rows:
            x1, y1, x2, y2 = box
            color = QColor.fromHsv((track_id * 137) % 360, 175, 255)
            pen = QPen(color)
            pen.setWidthF(2 / scale)
            painter.setPen(pen)
            if self.display["이동 궤적"] and len(trail) > 1:
                painter.drawPolyline(QPolygonF([QPointF(x, y) for x, y in trail]))
            if self.display["박스"]:
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            if self.display["ID"]:
                label = f"ID {track_id}"
                metrics = painter.fontMetrics()
                padding = 4 / scale
                label_height = metrics.height() + 2 * padding
                label_rect = QRectF(x1, max(0, y1 - label_height),
                                    metrics.horizontalAdvance(label) + 2 * padding, label_height)
                label_rect.moveLeft(min(x1, max(0, rect.width() - label_rect.width())))
                painter.fillRect(label_rect, color)
                painter.setPen(QColor("#101923"))
                painter.drawText(label_rect, Qt.AlignCenter, label)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("지능형 객체 추적 시스템")
        self.resize(1320, 820)
        self.source = ""
        self.worker = None
        self.stopping = False
        self.closing = False
        self.tracking_failed = False
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("지능형 객체 추적 시스템")
        title.setObjectName("title")
        layout.addWidget(title)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        page = QWidget()
        main = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.open_button = self.button(toolbar, "영상 열기", self.open_video)
        toolbar.addWidget(QLabel("추적기"))
        self.tracker = QComboBox()
        self.tracker.addItems(["ByteTrack", "원본 재생"])
        toolbar.addWidget(self.tracker)
        self.load_button = self.button(toolbar, "설정 불러오기", self.load_settings)
        self.button(toolbar, "설정 저장", self.save_settings)
        toolbar.addStretch()
        main.addLayout(toolbar)
        notice = QLabel("YOLO11n · ByteTrack · 사람 추적")
        notice.setStyleSheet("color:#00d4e9; padding:6px")
        main.addWidget(notice)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.video = QVideoWidget()
        self.preview = TrackingView()
        self.video_stack = QStackedWidget()
        self.video_stack.setMinimumSize(500, 300)
        self.video_stack.addWidget(self.video)
        self.video_stack.addWidget(self.preview)
        left.addWidget(self.video_stack, 1)
        controls = QHBoxLayout()
        self.play_button = self.button(controls, "▶ 시작", self.play)
        self.play_button.setObjectName("primary")
        self.pause_button = self.button(controls, "Ⅱ 일시정지", self.pause)
        self.stop_button = self.button(controls, "■ 중지", self.stop)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        controls.addWidget(self.slider, 1)
        self.clock = QLabel("00:00 / 00:00")
        controls.addWidget(self.clock)
        left.addLayout(controls)
        body.addLayout(left, 3)
        panel = QVBoxLayout()
        stats = QGroupBox("추적 상태")
        stats_layout = QHBoxLayout(stats)
        self.stats = [QLabel() for _ in range(3)]
        for stat in self.stats:
            stats_layout.addWidget(stat)
        self.set_stats()
        panel.addWidget(stats)
        objects = QGroupBox("객체 목록")
        objects_layout = QVBoxLayout(objects)
        self.objects = self.table(["ID", "상태", "방향", "거리(px)"])
        objects_layout.addWidget(self.objects)
        panel.addWidget(objects, 1)
        self.options = {}
        options = QHBoxLayout()
        for text in ["박스", "ID", "이동 궤적"]:
            check = QCheckBox(text)
            check.setChecked(True)
            check.toggled.connect(self.update_display)
            options.addWidget(check)
            self.options[text] = check
        panel.addLayout(options)
        self.update_display()
        body.addLayout(panel, 1)
        main.addLayout(body, 1)
        tabs.addTab(page, "실시간 추적")
        comparison = QWidget()
        compare_layout = QVBoxLayout(comparison)
        compare_layout.addWidget(QLabel("성능 비교 · 평가 엔진 연결 대기"))
        compare_layout.addWidget(self.table(
            ["알고리즘", "버전", "HOTA", "IDF1", "MOTA", "ID 변경", "FPS", "평균 ms"]))
        compare_layout.addWidget(QLabel("정답 데이터와 평가 엔진 연결 후 결과를 표시합니다."))
        tabs.addTab(comparison, "성능 비교")
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setMuted(True)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.media_error)
        self.tracker.currentTextChanged.connect(self.mode_changed)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.statusBar().showMessage("영상 파일을 열어 주세요. 소리는 기본 음소거입니다.")
        self.enable_controls(False)

    @staticmethod
    def button(layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def table(headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().hide()
        return table

    def enable_controls(self, enabled):
        for widget in (self.play_button, self.pause_button, self.stop_button, self.slider):
            widget.setEnabled(enabled)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 열기", "", "영상 (*.mp4 *.avi *.mkv *.mov *.webm);;모든 파일 (*)")
        if path:
            self.set_video(path)

    def set_video(self, path):
        """선택한 영상을 새 세션의 입력으로 지정합니다."""
        self.source = str(Path(path).resolve())
        self.mode_changed()
        self.statusBar().showMessage(Path(path).name + " · 시작을 눌러 주세요.")

    def mode_changed(self):
        self.player.stop()
        self.clear_tracking()
        self.enable_controls(bool(self.source))
        original = self.tracker.currentText() == "원본 재생"
        self.video_stack.setCurrentWidget(self.video if original else self.preview)
        self.slider.setEnabled(original and bool(self.source))
        self.player.setSource(QUrl.fromLocalFile(self.source) if original and self.source else QUrl())

    def play(self):
        if not self.source or self.stopping:
            return
        if self.tracker.currentText() == "원본 재생":
            if self.player.duration() and self.player.position() >= self.player.duration():
                self.player.setPosition(0)
            self.player.play()
            self.statusBar().showMessage("원본 영상 재생 중")
            return
        if self.worker:
            self.worker.set_paused(False)
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.statusBar().showMessage("사람 추적 중")
            return

        self.clear_tracking()
        self.tracking_failed = False
        self.worker = TrackingThread(self.source, self)
        self.worker.metadata.connect(self.tracking_duration)
        self.worker.frame_ready.connect(self.show_tracks)
        self.worker.failed.connect(self.tracking_error)
        self.worker.finished.connect(self.tracking_finished)
        for widget in (self.open_button, self.tracker, self.load_button, self.play_button):
            widget.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.statusBar().showMessage("YOLO11n 모델 준비 중 · 첫 실행 시 가중치를 다운로드할 수 있습니다.")
        self.worker.start()

    def pause(self):
        if self.worker and not self.stopping:
            self.worker.set_paused(True)
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
        else:
            self.player.pause()
        self.statusBar().showMessage("일시정지")

    def stop(self):
        if self.worker:
            self.stopping = True
            self.enable_controls(False)
            self.worker.cancel()
            self.statusBar().showMessage("중지 중 · 현재 프레임 처리가 끝날 때까지 기다려 주세요.")
        elif self.tracker.currentText() == "원본 재생":
            self.player.stop()
            self.statusBar().showMessage("중지")
        else:
            self.clear_tracking()
            self.statusBar().showMessage("중지 · 시작하면 처음부터 추적합니다.")

    def clear_tracking(self):
        self.preview.image = None
        self.preview.rows = []
        self.preview.update()
        self.objects.setRowCount(0)
        self.slider.setRange(0, 0)
        self.clock.setText("00:00 / 00:00")
        self.set_stats()

    def set_stats(self, count=None, elapsed=None):
        values = ("—", "—", "—") if elapsed is None else (
            str(count), f"{1 / elapsed:.1f}", f"{elapsed * 1000:.1f} ms"
        )
        for widget, label, value in zip(
            self.stats, ("추적 중", "처리 FPS", "프레임당 시간"), values
        ):
            widget.setText(f"{label}\n\n{value}")

    def update_display(self):
        self.preview.display = {key: check.isChecked() for key, check in self.options.items()}
        self.preview.update()

    def tracking_duration(self, duration):
        if not self.stopping:
            self.slider.setRange(0, duration)
            self.clock.setText(f"00:00 / {self.time_text(duration)}")

    def show_tracks(self, image, rows, position, elapsed):
        worker = self.worker
        if worker is None:
            return
        try:
            if self.stopping:
                return
            self.preview.image = image
            self.preview.rows = rows
            self.preview.update()
            self.slider.setValue(position)
            self.clock.setText(
                f"{self.time_text(position)} / {self.time_text(self.slider.maximum())}"
            )
            self.set_stats(len(rows), elapsed)
            self.objects.setRowCount(len(rows))
            for index, (track_id, _, _, direction, distance) in enumerate(rows):
                for column, value in enumerate((str(track_id), "추적 중", direction, f"{distance:.1f}")):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.objects.setItem(index, column, item)
            if not self.pause_button.isEnabled():
                return
            self.statusBar().showMessage("사람 추적 중")
        finally:
            worker.displayed.set()

    def tracking_error(self, message):
        self.tracking_failed = True
        if not self.stopping:
            self.statusBar().showMessage("추적 실패")
            # 비모달 대화상자로 완료 신호가 처리되는 동안 UI를 막지 않습니다.
            warning = QMessageBox(QMessageBox.Warning, "추적 실패", message, parent=self)
            warning.setAttribute(Qt.WA_DeleteOnClose)
            warning.show()

    def tracking_finished(self):
        self.worker.deleteLater()
        self.worker = None
        if self.closing:
            self.close()
            return
        for widget in (self.open_button, self.tracker, self.load_button, self.play_button):
            widget.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if self.stopping:
            self.clear_tracking()
            self.statusBar().showMessage("중지 · 시작하면 처음부터 추적합니다.")
        elif not self.tracking_failed:
            duration = self.slider.maximum()
            if duration:
                self.slider.setValue(duration)
                self.clock.setText(f"{self.time_text(duration)} / {self.time_text(duration)}")
            self.statusBar().showMessage("추적 완료 · 시작하면 처음부터 추적합니다.")
        self.stopping = False

    @staticmethod
    def time_text(milliseconds):
        seconds = milliseconds // 1000
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def position_changed(self, position):
        if self.tracker.currentText() != "원본 재생":
            return
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        self.clock.setText(
            f"{self.time_text(position)} / {self.time_text(self.player.duration())}")

    def duration_changed(self, duration):
        if self.tracker.currentText() != "원본 재생":
            return
        self.slider.setRange(0, duration)
        self.position_changed(self.player.position())

    def media_error(self, error, message):
        if self.tracker.currentText() != "원본 재생":
            return
        self.enable_controls(False)
        self.statusBar().showMessage("영상 오류 · 다른 파일을 열어 주세요.")
        QMessageBox.warning(self, "영상 재생 오류", message or "지원하지 않는 영상입니다.")

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "설정 저장", "settings.json", "JSON (*.json)")
        if path:
            data = {"version": 1, "tracker": self.tracker.currentText(),
                    "display": {key: value.isChecked() for key, value in self.options.items()}}
            try:
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as error:
                QMessageBox.warning(self, "저장 실패", str(error))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "설정 불러오기", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("지원하지 않는 설정 형식입니다.")
            tracker = data.get("tracker")
            if tracker in ("BoT-SORT", "DeepSORT"):
                tracker = "ByteTrack"
            display = data.get("display")
            if self.tracker.findText(tracker or "") < 0 or not isinstance(display, dict):
                raise ValueError("추적기 또는 표시 설정이 올바르지 않습니다.")
            if any(type(display.get(key)) is not bool for key in self.options):
                raise ValueError("표시 옵션은 true 또는 false여야 합니다.")
            self.tracker.setCurrentText(tracker)
            for key, check in self.options.items():
                check.setChecked(display[key])
        except (OSError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "불러오기 실패", str(error))

    def closeEvent(self, event):
        if self.worker:
            self.closing = True
            self.stop()
            event.ignore()
            return
        self.player.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())
