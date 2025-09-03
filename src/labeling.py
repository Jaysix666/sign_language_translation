# project.py
# 손 검출/랜드마크/좌우 구분/FPS/인덱스 토글 뷰어 + 자/모음 학습용 이미지 저장
# ▶ 저장 이미지는 항상 “오버레이 없는 원본(raw)에서 손 ROI만” 깔끔하게 저장
# ▶ 한글 텍스트는 Pillow로 렌더링(??? 방지)
# ▶ s: 현재 라벨로 ROI 10장 저장 | l: 라벨 변경 | i: 인덱스 on/off | 1/2: 손 개수 | q/ESC: 종료

import cv2
import time
import numpy as np
import mediapipe as mp
import os

# ========================= 사용자 설정 =========================
BASE_PATH = r"C:\Users\yhk49\OneDrive\바탕 화면\인텔 Edge AI\data"
LABELS = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ',
          'ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ']
# =============================================================

# 폴더 자동 생성
os.makedirs(BASE_PATH, exist_ok=True)
for label in LABELS:
    os.makedirs(os.path.join(BASE_PATH, label), exist_ok=True)

# -------- Pillow(한글 텍스트) --------
try:
    from PIL import ImageFont, ImageDraw, Image
    HAS_PIL = True
    FONT_CANDIDATES = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
    ]
    def _get_font(size=28):
        for p in FONT_CANDIDATES:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except:
                    pass
        return None
    def draw_text_kor(bgr, text, org, size=28, color=(255,255,0), stroke=2, stroke_color=(0,0,0)):
        font = _get_font(size)
        if font is None:
            cv2.putText(bgr, text.encode('ascii','ignore').decode('ascii'), org,
                        cv2.FONT_HERSHEY_SIMPLEX, size/40.0, color, 2, cv2.LINE_AA)
            return bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        draw.text(org, text, font=font, fill=tuple(color[::-1]),
                  stroke_width=stroke, stroke_fill=tuple(stroke_color[::-1]))
        bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return bgr
except Exception:
    HAS_PIL = False
    def draw_text_kor(bgr, text, org, size=28, color=(255,255,0), stroke=2, stroke_color=(0,0,0)):
        cv2.putText(bgr, text.encode('ascii','ignore').decode('ascii'), org,
                    cv2.FONT_HERSHEY_SIMPLEX, size/40.0, color, 2, cv2.LINE_AA)
        return bgr
# -----------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def next_index_in_dir(dir_path: str, label: str, ext: str = ".jpg") -> int:
    try:
        idx = 0
        for name in os.listdir(dir_path):
            if not name.lower().endswith(ext): continue
            if not name.startswith(label + "_"): continue
            try:
                num_part = name[len(label) + 1 : -len(ext)]
                idx = max(idx, int(num_part))
            except:
                pass
        return idx + 1
    except FileNotFoundError:
        return 1

def imwrite_unicode(filepath: str, image, params=None) -> bool:
    try:
        ext = os.path.splitext(filepath)[1]
        if not ext:
            ext = ".jpg"
            filepath = filepath + ext
        ok, buf = cv2.imencode(ext, image, params if params is not None else [])
        if not ok:
            return False
        buf.tofile(filepath)  # 유니코드 경로 안전
        return True
    except Exception as e:
        print(f"[에러] 이미지 저장 중 예외: {e}")
        return False

def bbox_from_pts(pts_xy: np.ndarray, w: int, h: int, margin: float = 0.20):
    if pts_xy is None or len(pts_xy) == 0:
        return None
    x = pts_xy[:, 0] * w
    y = pts_xy[:, 1] * h
    x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(x2-x1, y2-y1) * (1.0 + margin*2.0)
    x1 = int(max(0, cx - side/2)); y1 = int(max(0, cy - side/2))
    x2 = int(min(w, cx + side/2)); y2 = int(min(h, cy + side/2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

class HandCalibViewer:
    def __init__(self,
                 max_hands=2,
                 det_conf=0.5,
                 trk_conf=0.5,
                 show_indices=False,
                 cam_index=0):
        self.max_hands = max_hands
        self.det_conf = det_conf
        self.trk_conf = trk_conf
        self.show_indices = show_indices
        self.cam_index = cam_index

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.cap = None
        self.prev_t = time.time()
        self.fps_avg = 0.0
        self.fps_alpha = 0.9
        self.current_label = "ㄱ"
        self.last_bbox = None  # 최근 프레임의 손 ROI(저장용)

    @staticmethod
    def _put_fps(frame, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 240, 80), 2, cv2.LINE_AA)

    def _draw_landmark_indices(self, frame, pts_px):
        for i, (xx, yy) in enumerate(pts_px[:, :2].astype(int)):
            cv2.putText(frame, str(i), (xx, yy),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def _update_fps(self):
        now = time.time()
        dt = now - self.prev_t
        self.prev_t = now
        inst = 1.0 / dt if dt > 0 else 0.0
        self.fps_avg = self.fps_alpha * self.fps_avg + (1 - self.fps_alpha) * inst
        return self.fps_avg

    def _process_key(self, key):
        if key == 27 or key == ord('q'):
            return False
        if key == ord('i'):
            self.show_indices = not self.show_indices
        elif key == ord('1'):
            self.max_hands = 1
        elif key == ord('2'):
            self.max_hands = 2
        elif key == ord('l'):
            print("라벨 입력 모드: 새로운 라벨을 입력하세요. (엔터로 확정)")
            try:
                new_label = input("라벨 입력: ").strip()
            except Exception as e:
                print(f"[에러] 라벨 입력 실패: {e}")
                new_label = ""
            if new_label:
                self.current_label = new_label
                ensure_dir(os.path.join(BASE_PATH, self.current_label))
                print(f"[라벨 변경] 현재 라벨 = {self.current_label}")
        return True

    def _save_current_frame(self, raw_frame):
        label_dir = os.path.join(BASE_PATH, self.current_label)
        ensure_dir(label_dir)
        start_idx = next_index_in_dir(label_dir, self.current_label, ".jpg")

        if self.last_bbox is None:
            file_path = os.path.join(label_dir, f"{self.current_label}_{start_idx}.jpg")
            ok = imwrite_unicode(file_path, raw_frame)
            print(f"[{'저장됨' if ok else '실패'}] {file_path} (ROI 없음, 전체 저장)")
            return

        x1, y1, x2, y2 = self.last_bbox
        roi = raw_frame[y1:y2, x1:x2].copy()
        for i in range(10):
            idx = start_idx + i
            file_path = os.path.join(label_dir, f"{self.current_label}_{idx}.jpg")
            ok = imwrite_unicode(file_path, roi)
            print(f"[{'저장됨' if ok else '실패'}] {file_path}")

    def run(self):
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("카메라를 열 수 없습니다.")
                return

        cv2.namedWindow("Hand Calib Viewer", cv2.WINDOW_NORMAL)
        print("esc/q: 종료 | i: 인덱스 표시 on/off | 1/2: 손 개수 변경 | s: 저장(ROI 10장) | l: 라벨 입력")

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.trk_conf
        ) as hands:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                raw = frame.copy()
                fps = self._update_fps()
                h, w, _ = frame.shape

                # 손 검출
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                self.last_bbox = None

                # HUD
                hud = frame.copy()
                cv2.rectangle(hud, (0, 0), (w, 70), (0, 0, 0), -1)
                frame = cv2.addWeighted(hud, 0.35, frame, 0.65, 0)
                self._put_fps(frame, fps)
                draw_text_kor(frame, f"Current Label: {self.current_label} | s: 저장 | l: 라벨 변경",
                              (10, 46), size=28, color=(255,255,0), stroke=2)

                # 랜드마크/박스
                if res.multi_hand_landmarks:
                    for lm_list in res.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, lm_list,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style()
                        )
                        pts_norm = np.array([[lm.x, lm.y] for lm in lm_list.landmark], dtype=np.float32)
                        bbox = bbox_from_pts(pts_norm, w, h, margin=0.20)
                        if bbox is not None:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            self.last_bbox = bbox
                        if self.show_indices:
                            pts_px = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                            self._draw_landmark_indices(frame, pts_px)

                cv2.imshow("Hand Calib Viewer", frame)
                key = cv2.waitKey(1) & 0xFF

                if key != 255:
                    if key == ord('s'):
                        self._save_current_frame(raw)
                    else:
                        if not self._process_key(key):
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return

def main():
    viewer = HandCalibViewer(
        max_hands=1,
        det_conf=0.5,
        trk_conf=0.5,
        show_indices=False,
        cam_index=0
    )
    viewer.run()

if __name__ == "__main__":
    main()
