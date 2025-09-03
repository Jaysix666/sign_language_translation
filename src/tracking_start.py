# hand_tracking_plus.py
import os, time
import numpy as np
import cv2
import mediapipe as mp

try:
    from PIL import ImageFont, ImageDraw, Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False
    print("[경고] Pillow 미설치 → 한글이 ??? 로 보일 수 있습니다. pip install pillow")

# -------------------- PySide6 UI --------------------
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QMessageBox
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap

# -------------------- 번역용 --------------------
from googletrans import Translator

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
]

BASE_PATH  = r"C:\Users\yhk49\OneDrive\바탕 화면\인텔 Edge AI\data"
CACHE_PATH = os.path.join(BASE_PATH, "_features_cache.npz")

CHO  = ["ㄱ","ㄲ","ㄴ","ㄷ","ㄸ","ㄹ","ㅁ","ㅂ","ㅃ","ㅅ","ㅆ","ㅇ","ㅈ","ㅉ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]
JUNG = ["ㅏ","ㅐ","ㅑ","ㅒ","ㅓ","ㅔ","ㅕ","ㅖ","ㅗ","ㅘ","ㅙ","ㅚ","ㅛ","ㅜ","ㅝ","ㅞ","ㅟ","ㅠ","ㅡ","ㅢ","ㅣ"]
JONG = ["","ㄱ","ㄲ","ㄳ","ㄴ","ㄵ","ㄶ","ㄷ","ㄹ","ㄺ","ㄻ","ㄼ","ㄽ","ㄾ","ㄿ","ㅀ","ㅁ","ㅂ","ㅄ","ㅅ","ㅆ","ㅇ","ㅈ","ㅊ","ㅋ","ㅌ","ㅍ","ㅎ"]

COMBO_JUNG = {
    ("ㅗ","ㅏ"):"ㅘ",
    ("ㅗ","ㅐ"):"ㅙ",
    ("ㅗ","ㅣ"):"ㅚ",
    ("ㅜ","ㅓ"):"ㅝ",
    ("ㅜ","ㅔ"):"ㅞ",
    ("ㅜ","ㅣ"):"ㅟ",
    ("ㅡ","ㅣ"):"ㅢ",
    ("ㅏ","ㅣ"):"ㅐ",
    ("ㅑ","ㅣ"):"ㅒ",
    ("ㅓ","ㅣ"):"ㅔ",
    ("ㅕ","ㅣ"):"ㅖ",
}

COMBO_CHO = {
    ("ㄱ","ㄱ"):"ㄲ",
    ("ㄷ","ㄷ"):"ㄸ",
    ("ㅂ","ㅂ"):"ㅃ",
    ("ㅅ","ㅅ"):"ㅆ",
    ("ㅈ","ㅈ"):"ㅉ",
}

COMBO_JONG = {
    ("ㄱ","ㄱ"):"ㄲ",
    ("ㄱ","ㅅ"):"ㄳ",
    ("ㄴ","ㅈ"):"ㄵ",
    ("ㄴ","ㅎ"):"ㄶ",
    ("ㄹ","ㄱ"):"ㄺ",
    ("ㄹ","ㅁ"):"ㄻ",
    ("ㄹ","ㅂ"):"ㄼ",
    ("ㄹ","ㅅ"):"ㄽ",
    ("ㄹ","ㅌ"):"ㄾ",
    ("ㄹ","ㅍ"):"ㄿ",
    ("ㄹ","ㅎ"):"ㅀ",
    ("ㅂ","ㅅ"):"ㅄ",
}

def combine_jamos(cho, jung, jong=""):
    if isinstance(jung, (list,tuple)) and len(jung)==2:
        jung1,jung2 = jung
        jung = COMBO_JUNG.get((jung1,jung2), COMBO_JUNG.get((jung2,jung1), jung1+jung2))
    try:
        cho_idx = CHO.index(cho)
        jung_idx = JUNG.index(jung)
        jong_idx = JONG.index(jong) if jong else 0
        return chr(0xAC00 + cho_idx*21*28 + jung_idx*28 + jong_idx)
    except:
        return cho + (jung or '') + (jong or '')

def imwrite_unicode(filepath: str, image, params=None) -> bool:
    try:
        ext = os.path.splitext(filepath)[1] or ".jpg"
        if not filepath.lower().endswith(ext):
            filepath += ext
        ok, buf = cv2.imencode(ext, image, params or [])
        if not ok: return False
        buf.tofile(filepath)
        return True
    except Exception:
        return False

def get_korean_font(size=44):
    if not HAS_PIL: return None
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return None

def draw_text_kor(img_bgr, text, org, font_size=44, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    if not HAS_PIL:
        cv2.putText(img_bgr, text.encode('ascii','ignore').decode('ascii'), org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size/40.0, color, 2, cv2.LINE_AA)
        return img_bgr
    font = get_korean_font(font_size)
    if font is None:
        cv2.putText(img_bgr, text.encode('ascii','ignore').decode('ascii'), org,
                    cv2.FONT_HERSHEY_SIMPLEX, font_size/40.0, color, 2, cv2.LINE_AA)
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    x,y = org
    draw.text((x,y), text, font=font, fill=tuple(color[::-1]),
              stroke_width=stroke, stroke_fill=tuple(stroke_color[::-1]))
    img_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr

def landmarks_to_feature(pts_xy: np.ndarray) -> np.ndarray:
    pts = pts_xy.astype(np.float32).copy()
    pts -= pts[0]
    rng = np.max(np.abs(pts), axis=0)
    scale = max(rng[0], rng[1], 1e-6)
    pts /= scale
    base_feats = pts.flatten()
    fingertip_ids = [4,8,12,16,20]
    tips = pts[fingertip_ids]
    dists = [np.linalg.norm(tips[i]-tips[0]) for i in range(1,len(tips))]
    angles = []
    for i in range(1,len(tips)-1):
        v1 = tips[i]-tips[i-1]
        v2 = tips[i+1]-tips[i]
        dot = np.dot(v1,v2)
        norm = np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6
        ang = np.arccos(np.clip(dot/norm,-1.0,1.0))/np.pi
        angles.append(ang)
    extra_feats = np.array(dists+angles,dtype=np.float32)
    return np.concatenate([base_feats,extra_feats])

def extract_feat_from_result(lm):
    pts = np.array([[p.x,p.y] for p in lm.landmark],dtype=np.float32)
    return landmarks_to_feature(pts)

class SimpleKNN:
    def __init__(self,k=7):
        self.k = k
        self.X = None; self.y = None
    def load_from_cache(self,cache_path):
        npz = np.load(cache_path,allow_pickle=True)
        self.X = np.asarray(npz["X"],dtype=np.float32)
        self.y = npz["y"]
        labels = npz["labels"].tolist() if "labels" in npz.files else sorted(list(set(self.y)))
        return labels,len(self.X)
    def predict_one(self,x):
        if self.X is None or len(self.X)==0: return None,0.0
        d = np.linalg.norm(self.X-x,axis=1)
        idx = np.argpartition(d, kth=min(self.k,len(self.X)-1)-1)[:self.k]
        votes, counts = np.unique(self.y[idx],return_counts=True)
        best = votes[np.argmax(counts)]
        conf = counts.max()/self.k
        return best,float(conf)
    def ready(self): return self.X is not None and len(self.X)>0

class Viewer:
    def __init__(self,cam_index=0,max_hands=1,det_conf=0.3,trk_conf=0.3):
        self.cam_index = cam_index
        self.max_hands = max_hands
        self.det_conf  = det_conf
        self.trk_conf  = trk_conf

        self.knn = SimpleKNN(k=7)
        self.prev_t = time.time(); self.fps=0.0; self.alpha=0.9

        self.current_syllable = ""
        self.output_text = ""
        self.cho = None; self.jung = None; self.jong = None

        self.stable_char = None
        self.stable_start_time = None
        self.hold_time = 1.0

        self.last_space_time = 0
        self.last_del_time = 0
        self.SPACE_HOLD = 1.0
        self.DEL_HOLD   = 1.0

    def _update_fps(self):
        now = time.time()
        dt = now - self.prev_t
        self.prev_t = now
        self.fps = self.alpha*self.fps + (1-self.alpha)*(1.0/dt if dt>0 else 0.0)

    def load_cache(self):
        if not os.path.exists(CACHE_PATH):
            print(f"[오류] 캐시가 없습니다: {CACHE_PATH}")
            return False
        self.knn.load_from_cache(CACHE_PATH)
        print("[로드] 캐시 로드 완료")
        return True

    def update_syllable(self, char):
        if char in CHO:
            if self.cho and not self.jung:
                combo = COMBO_CHO.get((self.cho, char), COMBO_CHO.get((char, self.cho), None))
                if combo:
                    self.cho = combo
                else:
                    if self.current_syllable:
                        self.output_text += self.current_syllable
                    self.reset_syllable()
                    self.cho = char
            elif self.cho and self.jung:
                if self.jong:
                    combo = COMBO_JONG.get((self.jong,char), COMBO_JONG.get((char,self.jong), None))
                    if combo: self.jong = combo
                    else:
                        if self.current_syllable:
                            self.output_text += self.current_syllable
                        self.reset_syllable()
                        self.cho = char
                else:
                    self.jong = char
            else:
                self.cho = char
        elif char in JUNG:
            if self.jung:
                combo = COMBO_JUNG.get((self.jung,char), COMBO_JUNG.get((char,self.jung), None))
                if combo:
                    self.jung = combo
                else:
                    if self.cho and self.jung:
                        self.current_syllable = combine_jamos(self.cho, self.jung, self.jong or '')
                        self.output_text += self.current_syllable
                    self.jung = char
                    self.cho = None
                    self.jong = None
            else:
                self.jung = char
        elif char in JONG:
            if self.cho and self.jung:
                if self.jong:
                    combo = COMBO_JONG.get((self.jong,char), COMBO_JONG.get((char,self.jong), None))
                    if combo: self.jong = combo
                    else: self.jong = char
                else:
                    self.jong = char
        if self.cho and self.jung:
            self.current_syllable = combine_jamos(self.cho, self.jung, self.jong or '')
        elif self.cho:
            self.current_syllable = self.cho
        elif self.jung:
            self.current_syllable = self.jung
        else:
            self.current_syllable = ''

    def reset_syllable(self):
        self.current_syllable = ""
        self.cho = self.jung = self.jong = None

    def process_frame(self, frame, hands):
        self._update_fps()
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        shown = "(예측 없음)"
        if res.multi_hand_landmarks and self.knn.ready():
            for lm in res.multi_hand_landmarks:
                feat = extract_feat_from_result(lm)
                pred, conf = self.knn.predict_one(feat)
                if pred is not None:
                    now = time.time()
                    if pred == self.stable_char:
                        if self.stable_start_time and (now-self.stable_start_time >= self.hold_time):
                            if pred == "space":
                                if now-self.last_space_time >= self.SPACE_HOLD:
                                    self.last_space_time = now
                                    if self.current_syllable:
                                        self.output_text += self.current_syllable
                                    self.output_text += " "
                                    self.reset_syllable()
                            elif pred == "del":
                                if now-self.last_del_time >= self.DEL_HOLD:
                                    self.last_del_time = now
                                    if self.current_syllable:
                                        self.reset_syllable()
                                    else:
                                        words = self.output_text.rstrip().split(' ')
                                        if words:
                                            words = words[:-1]
                                        self.output_text = ' '.join(words) + (' ' if words else '')
                            else:
                                self.update_syllable(pred)
                            shown = pred
                            self.stable_char, self.stable_start_time = None,None
                    else:
                        self.stable_char = pred
                        self.stable_start_time = now
                        shown = pred

        return frame

    def run(self):
        if not self.load_cache(): return
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다."); return

        cv2.namedWindow("KSL Inference", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("KSL Inference", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("q/ESC 종료 | r 캐시 리로드 | SPACE/del 기능 적용")
        mp_hands_mod = mp.solutions.hands
        with mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=self.det_conf,
            min_tracking_confidence=self.trk_conf
        ) as hands:
            while True:
                ok,frame = cap.read()
                if not ok: break
                frame = self.process_frame(frame, hands)
                cv2.imshow("KSL Inference",frame)
                key = cv2.waitKey(1)&0xFF
                if key in (27, ord('q')): break
                elif key==ord('r'): self.load_cache()

        cap.release()
        cv2.destroyAllWindows()

# -------------------- PySide6 메인 윈도우 --------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KSL Inference - PySide6")
        self.setMinimumSize(QSize(960, 640))

        self.viewer = Viewer(cam_index=0, max_hands=1, det_conf=0.3, trk_conf=0.3)
        if not self.viewer.load_cache():
            QMessageBox.warning(self, "오류", f"캐시가 없습니다:\n{CACHE_PATH}")
        self.cap = None
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.viewer.max_hands,
            model_complexity=1,
            min_detection_confidence=self.viewer.det_conf,
            min_tracking_confidence=self.viewer.trk_conf
        )

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border-radius:16px;")

        self.pred_label = QLabel("예측: -")
        self.word_label = QLabel("현재 단어: -")
        self.sent_edit = QLineEdit()
        self.sent_edit.setReadOnly(True)
        self.sent_edit.setPlaceholderText("여기에 인식된 문장이 표시됩니다")

        self.pred_label.setStyleSheet(
            "font-family:'Segoe UI','Malgun Gothic','Pretendard',sans-serif;"
            "font-size:18px; font-weight:700; color:#000000;"
        )
        self.word_label.setStyleSheet(
            "font-family:'Segoe UI','Malgun Gothic','Pretendard',sans-serif;"
            "font-size:18px; font-weight:600; color:#000000;"
        )
        self.sent_edit.setStyleSheet(
            "font-family:'Segoe UI','Malgun Gothic','Pretendard',sans-serif;"
            "font-size:16px; color:#CFEFFF; background:#0F1720;"
            "border:1px solid #263344; border-radius:8px; padding:8px;"
        )

        self.btn_start = QPushButton("시작")
        self.btn_stop  = QPushButton("중지")
        self.btn_clear = QPushButton("문장 지우기")
        self.btn_translate = QPushButton("번역하기")
        self.btn_full  = QPushButton("최대화/원복")

        for b in (self.btn_start,self.btn_stop,self.btn_clear,self.btn_translate,self.btn_full):
            b.setMinimumHeight(36)
            b.setStyleSheet("font-size:15px; border-radius:10px;")

        top_info = QHBoxLayout()
        top_info.addWidget(self.pred_label)
        top_info.addSpacing(100)
        top_info.addWidget(self.word_label)
        top_info.addStretch(1)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_translate)
        btns.addWidget(self.btn_full)

        right = QVBoxLayout()
        right.addLayout(top_info)
        right.addWidget(self.sent_edit)
        right.addWidget(self.video_label, stretch=1)
        right.addLayout(btns)

        container = QWidget()
        container.setLayout(right)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_clear.clicked.connect(self.clear_sentence)
        self.btn_translate.clicked.connect(self.translate_sentence)
        self.btn_full.clicked.connect(self.toggle_fullscreen)

        self.translator = Translator()

        self.showFullScreen()

    # -------------------- ESC 키 종료 --------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)
    # -------------------- ESC 키 종료 끝 --------------------

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.viewer.cam_index, cv2.CAP_ANY)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
                return
        self.timer.start(15)

    def stop_camera(self):
        self.timer.stop()

    def clear_sentence(self):
        self.viewer.reset_syllable()
        self.viewer.output_text = ""
        self.sent_edit.setText("")
        self.pred_label.setText("예측: -")
        self.word_label.setText("현재 단어: -")

    def translate_sentence(self):
        text = self.viewer.output_text.strip()
        if not text:
            return
        try:
            result = self.translator.translate(text, src='ko', dest='en')
            self.viewer.output_text = result.text
            self.sent_edit.setText(result.text)
        except Exception as e:
            QMessageBox.warning(self, "번역 오류", f"번역 실패:\n{str(e)}")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def closeEvent(self, e):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        cv2.destroyAllWindows()
        super().closeEvent(e)

    def update_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        frame = self.viewer.process_frame(frame, self.hands)

        self.pred_label.setText(f"예측: {self.viewer.stable_char or '-'}")
        self.word_label.setText(f"현재 단어: {self.viewer.current_syllable or '-'}")
        self.sent_edit.setText(self.viewer.output_text)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        show_pix = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(show_pix)

    def resizeEvent(self, e):
        if self.video_label.pixmap():
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        super().resizeEvent(e)

if __name__=="__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
