"""
Open Images V7에서 3개 클래스(비행기, 독수리, 까마귀) 이미지를 수집하고
YOLO 형식 데이터셋으로 변환하는 스크립트.

사용법:
    python scripts/01_build_dataset.py --meta <라벨 CSV 폴더> --out datasets/sky3 --per-class 250
"""
import argparse
import random
import concurrent.futures as cf
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# YOLO 클래스 번호 순서 = 이 리스트 순서
CLASSES = [
    ("airplane", "/m/0cmf2"),  # Airplane
    ("eagle", "/m/09csl"),     # Eagle
    ("crow", "/m/06j2d"),      # Raven (까마귀류)
]
LABEL2IDX = {mid: i for i, (_, mid) in enumerate(CLASSES)}
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf"]


def load_boxes(meta: Path) -> pd.DataFrame:
    files = {
        "validation": "validation-annotations-bbox.csv",
        "test": "test-annotations-bbox.csv",
        "train": "oidv6-train-annotations-bbox.csv",
    }
    frames = []
    for split, name in files.items():
        path = meta / name
        if not path.exists():
            print(f"[skip] {path} 없음")
            continue
        for chunk in pd.read_csv(path, usecols=COLS, chunksize=2_000_000):
            chunk = chunk[chunk.LabelName.isin(LABEL2IDX)]
            if len(chunk):
                chunk = chunk.assign(split=split)
                frames.append(chunk)
        print(f"[load] {split} 완료")
    return pd.concat(frames, ignore_index=True)


def pick_images(df: pd.DataFrame, per_class: int, seed: int) -> dict:
    """클래스별로 per_class장씩 고른다. 그룹 박스(IsGroupOf=1)가 있는 이미지는 제외."""
    rng = random.Random(seed)
    bad = set(df.loc[df.IsGroupOf == 1, "ImageID"])
    chosen = {}  # image_id -> (split, main_class)
    # 이미지 수가 적은 클래스(까마귀)부터 골라 중복을 피한다
    counts = df.groupby("LabelName").ImageID.nunique().sort_values()
    split_of = df.drop_duplicates("ImageID").set_index("ImageID").split
    for mid in counts.index:
        ids = sorted(set(df.loc[df.LabelName == mid, "ImageID"]) - bad - set(chosen))
        rng.shuffle(ids)
        for iid in ids[:per_class]:
            chosen[iid] = (split_of[iid], LABEL2IDX[mid])
        print(f"[pick] {CLASSES[LABEL2IDX[mid]][0]}: {min(per_class, len(ids))} / 후보 {len(ids)}")
    return chosen


def download(iid, split, dst: Path, max_side=800):
    if dst.exists():
        return True
    try:
        r = requests.get(IMG_URL.format(split=split, image_id=iid), timeout=30)
        r.raise_for_status()
        im = Image.open(BytesIO(r.content)).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst, quality=92)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[fail] {iid}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/sky3"))
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_boxes(args.meta)
    chosen = pick_images(df, args.per_class, args.seed)

    # 클래스별로 train/val 분할 (층화)
    rng = random.Random(args.seed)
    assign = {}
    for c in range(len(CLASSES)):
        ids = sorted(i for i, (_, mc) in chosen.items() if mc == c)
        rng.shuffle(ids)
        n_val = int(len(ids) * args.val_ratio)
        for k, iid in enumerate(ids):
            assign[iid] = "val" if k < n_val else "train"

    for sub in ("train", "val"):
        (args.out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / sub).mkdir(parents=True, exist_ok=True)

    with cf.ThreadPoolExecutor(16) as ex:
        futs = {
            ex.submit(download, iid, split, args.out / "images" / assign[iid] / f"{iid}.jpg"): iid
            for iid, (split, _) in chosen.items()
        }
        ok = {futs[f] for f in cf.as_completed(futs) if f.result()}
    print(f"[download] {len(ok)} / {len(chosen)} 성공")

    # YOLO 라벨: class x_center y_center width height (0~1 정규화)
    boxes = df[df.ImageID.isin(ok)]
    for iid, g in boxes.groupby("ImageID"):
        lines = []
        for r in g.itertuples():
            xc, yc = (r.XMin + r.XMax) / 2, (r.YMin + r.YMax) / 2
            w, h = r.XMax - r.XMin, r.YMax - r.YMin
            lines.append(f"{LABEL2IDX[r.LabelName]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        (args.out / "labels" / assign[iid] / f"{iid}.txt").write_text("\n".join(lines) + "\n")

    # 통계
    stats = boxes.assign(sub=boxes.ImageID.map(assign), cls=boxes.LabelName.map(lambda m: CLASSES[LABEL2IDX[m]][0]))
    print("\n이미지 수 (클래스가 하나라도 등장한 이미지):")
    print(stats.drop_duplicates(["ImageID", "cls"]).groupby(["sub", "cls"]).size().unstack(fill_value=0))
    print("\n박스 수:")
    print(stats.groupby(["sub", "cls"]).size().unstack(fill_value=0))

    yaml = (
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, (n, _) in enumerate(CLASSES))
    )
    (args.out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"\n[done] {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
