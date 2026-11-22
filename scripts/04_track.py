"""
학습된 모델로 동영상에서 객체를 감지 + 추적(ByteTrack)하고,
객체 ID별 이동 궤적(trajectory)을 그려서 동영상과 CSV로 저장한다.

시각화:
- 박스/궤적은 지수이동평균(EMA)으로 떨림을 줄여서 표시 (CSV에는 원본 좌표 저장)
- 궤적은 최근일수록 굵고 진하게, 오래될수록 가늘고 흐리게
- 좌상단 정보 패널: 모델/추적기, 부류별 색상 범례와 현재 프레임의 객체 수
- 장면 전환(클립 경계)을 감지하면 추적기와 궤적을 초기화

사용법:
    python scripts/04_track.py --model models/sky3_yolov8n_best.pt --source videos/input_5s.mp4 --out results/tracking_result.mp4
"""
import argparse
import csv
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

# BGR 색상: airplane=하늘색, eagle=주황, crow=보라
COLORS = {0: (235, 180, 40), 1: (40, 150, 255), 2: (200, 70, 190)}
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA


def ema(prev, cur, a):
    return cur if prev is None else tuple(a * c + (1 - a) * p for p, c in zip(prev, cur))


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    """반투명 사각형"""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)


def draw_trail(img, pts, color):
    """오래된 점일수록 가늘고 흐리게"""
    n = len(pts)
    if n < 2:
        return
    overlay = img.copy()
    for i in range(1, n):
        t = i / (n - 1)  # 0(오래됨) → 1(최근)
        thick = max(1, int(round(1 + 5 * t)))
        p0 = tuple(int(v) for v in pts[i - 1])
        p1 = tuple(int(v) for v in pts[i])
        cv2.line(overlay, p0, p1, color, thick, AA)
    # 전체를 한 번에 섞되, 최근 구간은 한 번 더 진하게
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, dst=img)
    recent = [tuple(int(v) for v in p) for p in list(pts)[-max(2, n // 3):]]
    for i in range(1, len(recent)):
        cv2.line(img, recent[i - 1], recent[i], color, 5, AA)
    cv2.circle(img, tuple(int(v) for v in pts[-1]), 7, (255, 255, 255), -1, AA)
    cv2.circle(img, tuple(int(v) for v in pts[-1]), 5, color, -1, AA)


PANEL = (24, 24, 464, 214)  # 정보 패널 영역 (x1, y1, x2, y2) — draw_panel과 일치


def overlaps(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def draw_box(img, box, color, label):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, AA)
    # 모서리 강조
    L = max(12, min(x2 - x1, y2 - y1) // 6)
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, 7, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, 7, AA)
    scale, th = 0.9, 2
    (tw, tht), base = cv2.getTextSize(label, FONT, scale, th)
    lx = x1
    ly = y1 - 10 if y1 - tht - 18 > 0 else y1 + tht + 16  # 박스 위, 공간이 없으면 박스 안쪽 위
    if overlaps((lx, ly - tht - 10, lx + tw + 16, ly + base + 2), PANEL):
        lx = PANEL[2] + 8  # 정보 패널과 겹치면 패널 오른쪽으로 옮긴다
    blend_rect(img, lx, ly - tht - 10, lx + tw + 16, ly + base + 2, color, 0.85)
    cv2.putText(img, label, (lx + 8, ly - 2), FONT, scale, (255, 255, 255), th, AA)


def draw_panel(img, names, counts, frame_idx, fps):
    x, y = PANEL[0], PANEL[1]
    w, h = PANEL[2] - PANEL[0], PANEL[3] - PANEL[1]
    blend_rect(img, x, y, x + w, y + h, (20, 20, 20), 0.6)
    cv2.putText(img, "YOLOv8n + ByteTrack", (x + 16, y + 36), FONT, 0.85, (255, 255, 255), 2, AA)
    cv2.putText(img, f"{frame_idx / fps:4.1f}s", (x + w - 84, y + 36), FONT, 0.7, (200, 200, 200), 1, AA)
    for i, (cid, name) in enumerate(names.items()):
        yy = y + 60 + 42 * i
        cv2.rectangle(img, (x + 18, yy), (x + 42, yy + 24), COLORS.get(cid, (0, 255, 0)), -1, AA)
        cv2.putText(img, name, (x + 56, yy + 20), FONT, 0.75, (255, 255, 255), 1, AA)
        cv2.putText(img, f"{counts.get(cid, 0)}", (x + w - 50, yy + 20), FONT, 0.75, (255, 255, 255), 2, AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="results/tracking_result.mp4")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=960, help="추론 해상도 (멀리 있는 작은 비행기 검출을 위해 학습 크기보다 크게)")
    ap.add_argument("--cut-thresh", type=float, default=40.0, help="장면 전환 판정 임계값 (프레임 간 평균 밝기 차)")
    ap.add_argument("--trail", type=int, default=45, help="궤적 길이 (프레임)")
    ap.add_argument("--smooth", type=float, default=0.3, help="EMA 계수 (작을수록 부드러움)")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = defaultdict(lambda: deque(maxlen=args.trail))
    sm_center, sm_box, last_seen, cls_of = {}, {}, {}, {}
    rows = []
    frame_idx = 0
    prev_small = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # 장면 전환 감지: 이전 프레임과 크게 다르면 추적기 상태와 궤적을 초기화
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY).astype("float32")
        if prev_small is not None and abs(small - prev_small).mean() > args.cut_thresh:
            print(f"[scene cut] frame {frame_idx} → tracker reset")
            if getattr(model, "predictor", None) is not None and getattr(model.predictor, "trackers", None):
                for t in model.predictor.trackers:
                    t.reset()
            trails.clear(); sm_center.clear(); sm_box.clear(); last_seen.clear(); cls_of.clear()
        prev_small = small

        res = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz, tracker=args.tracker, verbose=False)[0]
        counts = defaultdict(int)
        dets = []
        if res.boxes.id is not None:
            for box, tid, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.id.int().tolist(),
                                           res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                rows.append([frame_idx, round(frame_idx / fps, 3), tid, model.names[cls], round(conf, 3),
                             round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
                sm_center[tid] = ema(sm_center.get(tid), (cx, cy), args.smooth)
                sm_box[tid] = ema(sm_box.get(tid), tuple(box), 0.75)
                trails[tid].append(sm_center[tid])
                last_seen[tid] = frame_idx
                counts[cls] += 1
                dets.append((tid, cls, conf))

        # 오래 안 보인 객체의 궤적은 제거
        for tid in [t for t, f in last_seen.items() if frame_idx - f > 15]:
            trails.pop(tid, None); sm_center.pop(tid, None); sm_box.pop(tid, None); last_seen.pop(tid)

        for tid, cls, _ in dets:
            cls_of[tid] = cls  # 일시적으로 놓친 프레임에도 궤적 색을 유지
        for tid, pts in trails.items():
            draw_trail(frame, pts, COLORS.get(cls_of.get(tid, -1), (0, 255, 255)))
        for tid, cls, conf in dets:
            draw_box(frame, sm_box[tid], COLORS.get(cls, (0, 255, 0)), f"#{tid} {model.names[cls]} {conf:.2f}")
        draw_panel(frame, model.names, counts, frame_idx, fps)

        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()

    # GitHub/브라우저에서 재생되도록 H.264로 재인코딩
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()

    csv_path = out.with_name(out.stem + "_trajectory.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "track_id", "class", "conf", "cx", "cy", "x1", "y1", "x2", "y2"])
        wr.writerows(rows)
    print(f"[done] {out} ({frame_idx} frames, {frame_idx / fps:.1f}s), 궤적 CSV: {csv_path}")
    print(f"추적된 객체: {sorted({(r[2], r[3]) for r in rows})}")


if __name__ == "__main__":
    main()
