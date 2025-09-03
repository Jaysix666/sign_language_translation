# BASE_PATH/<라벨>/*.jpg|png → Mediapipe 손 랜드마크(확장 특징 포함) 추출 → _features_cache.npz 저장
# - 한글 경로 안전 (imdecode/fromfile)
# - 진행률/속도 출력, 빠른 모드(--fast) 지원
# - 성공/실패 이미지 경로 로그 저장 (_success.txt, _fail.txt)
#
# 실행 예:
#   python build_features_cache.py --fast --print-every 50 --min-det-conf 0.6
#
# 개선 사항:
#   - min_det_conf 기본값 0.6 (더 높은 신뢰도)
#   - fast 모드 사용 시 정확도↓ → 필요 시 옵션 빼고 실행
#   - 특징 벡터에 좌표(42차원) + 거리/각도 특징 추가

import os, sys, time, argparse, math
import numpy as np
import cv2
import mediapipe as mp
from collections import defaultdict

# ===== 경로 설정 =====
BASE_PATH = r"C:\Users\yhk49\OneDrive\바탕 화면\인텔 Edge AI\data"
CACHE_PATH = os.path.join(BASE_PATH, "_features_cache.npz")
# ====================

os.makedirs(BASE_PATH, exist_ok=True)
mp_hands = mp.solutions.hands

def imread_unicode(path: str):
    """한글/특수문자 경로에서도 안전하게 이미지 로드"""
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def maybe_resize(img, max_side: int | None):
    """긴 변 기준 리사이즈"""
    if img is None or not max_side:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def landmarks_to_feature(pts_xy: np.ndarray) -> np.ndarray:
    """
    (21,2) → 기준점(0) 평행이동 + 크기 정규화 → (추가 특징 포함 벡터)
    - 기본 좌표 42차원
    - 손가락 끝 거리 + 각도 특징 추가
    """
    pts = pts_xy.astype(np.float32).copy()
    pts -= pts[0]  # 기준점 이동
    rng = np.max(np.abs(pts), axis=0)
    scale = max(rng[0], rng[1], 1e-6)
    pts /= scale

    base_feats = pts.flatten()  # (42,)

    # === 추가 특징 ===
    # 손가락 끝 포인트 인덱스: 엄지(4), 검지(8), 중지(12), 약지(16), 새끼(20)
    fingertip_ids = [4, 8, 12, 16, 20]
    tips = pts[fingertip_ids]

    # 1) 손가락 끝 거리들 (엄지-다른 손가락)
    dists = []
    for i in range(1, len(tips)):
        d = np.linalg.norm(tips[i] - tips[0])
        dists.append(d)

    # 2) 손가락 끝 각도들 (검지~중지, 중지~약지, 약지~새끼)
    angles = []
    for i in range(1, len(tips)-1):
        v1 = tips[i] - tips[i-1]
        v2 = tips[i+1] - tips[i]
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        ang = math.acos(np.clip(dot / norm, -1.0, 1.0))  # 라디안
        angles.append(ang / math.pi)  # 0~1 정규화

    extra_feats = np.array(dists + angles, dtype=np.float32)

    return np.concatenate([base_feats, extra_feats])  # (42 + 추가 특징)

def extract_batch(images_bgr, paths, min_det_conf=0.6, print_every=50):
    """
    Hands()를 1회만 생성해 배치 처리. 진행률/속도 출력 + 성공/실패 리스트 기록
    """
    feats = []
    total = len(images_bgr)
    ok_cnt = 0
    start = time.time()
    success_list, fail_list = [], []

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                        min_detection_confidence=min_det_conf) as hands:
        for i, (img, p) in enumerate(zip(images_bgr, paths), 1):
            if img is None:
                feats.append(None)
                fail_list.append(p)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]  # 첫 번째 손만 사용
                    pts = np.array([[pt.x, pt.y] for pt in lm.landmark], dtype=np.float32)
                    feats.append(landmarks_to_feature(pts))
                    ok_cnt += 1
                    success_list.append(p)
                else:
                    feats.append(None)
                    fail_list.append(p)

            if print_every > 0 and (i % print_every == 0 or i == total):
                elapsed = time.time() - start
                speed = i / elapsed if elapsed > 0 else 0.0
                print(f"[진행] {i}/{total}  성공:{ok_cnt}  실패:{i-ok_cnt}  "
                      f"경과:{elapsed:5.1f}s  속도:{speed:4.1f} imgs/s", flush=True)

    return feats, success_list, fail_list

def scan_dataset(base):
    """라벨별 이미지 경로 수집"""
    labels = sorted([d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)) and not d.startswith("_")])
    paths, y = [], []
    for lab in labels:
        ldir = os.path.join(base, lab)
        for fn in os.listdir(ldir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(ldir, fn))
                y.append(lab)
    return labels, paths, y

def main():
    parser = argparse.ArgumentParser(description="Build hand-landmark feature cache (improved)")
    parser.add_argument("--fast", action="store_true",
                        help="긴 변을 256px로 리사이즈해 빠르게 처리 (정확도↓ 가능성 있음)")
    parser.add_argument("--max-side", type=int, default=256,
                        help="--fast일 때 사용할 최대 변 길이 (기본 256)")
    parser.add_argument("--print-every", type=int, default=50,
                        help="N장마다 진행상황 출력 (0이면 비활성)")
    parser.add_argument("--min-det-conf", type=float, default=0.6,
                        help="손 검출 최소 신뢰도 (0~1, 높을수록 안정적)")
    args = parser.parse_args()

    labels, paths, y = scan_dataset(BASE_PATH)
    if len(paths) == 0:
        print("[오류] 이미지가 없습니다. BASE_PATH 아래 라벨 폴더에 이미지를 넣어주세요.")
        sys.exit(1)

    print(f"[스캔] 라벨 {len(labels)}개, 이미지 {len(paths)}장")
    print(f"[옵션] fast={args.fast}  max_side={args.max_side if args.fast else '원본'}  "
          f"print_every={args.print_every}  min_det_conf={args.min_det_conf}")

    # 이미지 로드(+옵션 리사이즈)
    imgs = []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = imread_unicode(p)
        img = maybe_resize(img, args.max_side if args.fast else None)
        imgs.append(img)
        if args.print_every > 0 and (i % args.print_every == 0 or i == len(paths)):
            el = time.time() - t0
            sp = i / el if el > 0 else 0.0
            print(f"[로드] {i}/{len(paths)}  경과:{el:5.1f}s  속도:{sp:4.1f} imgs/s", flush=True)

    # 특징 추출
    feats, success_list, fail_list = extract_batch(imgs, paths,
                                                   min_det_conf=args.min_det_conf,
                                                   print_every=args.print_every)

    # 통계 및 저장
    total_per = defaultdict(int); used_per = defaultdict(int)
    for lab in y:
        total_per[lab] += 1
    X, Y = [], []
    for lab, f in zip(y, feats):
        if f is not None:
            X.append(f); Y.append(lab); used_per[lab] += 1
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=object)

    np.savez_compressed(CACHE_PATH, X=X, y=Y, labels=np.array(labels, dtype=object))
    print(f"[저장] 캐시 → {CACHE_PATH}")

    # 성공/실패 로그 저장
    with open(os.path.join(BASE_PATH, "_success.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(success_list))
    with open(os.path.join(BASE_PATH, "_fail.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(fail_list))

    print("=== 요약 ===")
    for lab in labels:
        print(f"{lab}: 총 {total_per[lab]}장, 사용 {used_per[lab]}장")
    print(f"[로그] 성공 {len(success_list)}장 (_success.txt), 실패 {len(fail_list)}장 (_fail.txt)")
    print(f"[완료] 사용 가능한 샘플: {len(X)}")

if __name__ == "__main__":
    main()