# hand_launcher.py
import sys, os, cv2
import mediapipe as mp
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor

# new_tracking_start.py 안의 MainWindow 불러오기
from new_tracking_start import MainWindow as AppWindow


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("실행기")

        logo_path = r"C:\Users\yhk49\OneDrive\바탕 화면\인텔 Edge AI\logo.png"
        self.logoLabel = QLabel(self)

        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logoLabel.setPixmap(scaled_pixmap)
            self.logoLabel.setScaledContents(True)

            w, h = scaled_pixmap.width(), scaled_pixmap.height()
            self.resize(w, h)
            self.logoLabel.resize(w, h)

        # 손가락 좌표 저장용
        self.finger_x, self.finger_y = None, None

        # Mediapipe 손 추적 초기화
        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 타이머로 좌표 갱신
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_hand)
        self.timer.start(50)  # 20fps 정도

    def detect_hand(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                ix = int(lm.landmark[8].x * self.logoLabel.width())   # 로고 QLabel 크기에 맞춰 좌표 변환
                iy = int(lm.landmark[8].y * self.logoLabel.height())
                self.finger_x, self.finger_y = ix, iy

                # 실행 버튼 영역 체크 (임의 영역, 필요시 조정)
                if 110 <= ix <= 210 and 290 <= iy <= 340:
                    print("손가락이 실행하기 버튼 위에 있음 → 실행")
                    self.start_app()
        else:
            self.finger_x, self.finger_y = None, None

        # 좌표 바뀌면 다시 그리기
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.finger_x is not None and self.finger_y is not None:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 12)  # 빨간 점
            painter.setPen(pen)
            painter.drawPoint(self.finger_x, self.finger_y)

    def start_app(self):
        """기존 Viewer 대신 new_tracking_start의 MainWindow 실행"""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.close()

        # 메인 윈도우 띄우기
        self.app_win = AppWindow()
        self.app_win.show()

    def closeEvent(self, e):
        if self.cap:
            self.cap.release()
        self.hands.close()
        super().closeEvent(e)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec())
