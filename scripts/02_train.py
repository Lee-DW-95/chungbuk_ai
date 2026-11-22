"""
YOLOv8n 사전학습 모델을 3개 클래스(비행기, 독수리, 까마귀)에 맞춰 전이학습한다.
COCO로 학습된 백본/넥의 가중치는 그대로 가져오고, 헤드는 3개 클래스에 맞게 새로 만들어진다.

사용법:
    python scripts/02_train.py --data datasets/sky3/data.yaml --epochs 50
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/sky3/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default="sky3_yolov8n")
    # fine-tuning(이미 학습된 모델에서 이어서 학습)용: 학습률을 낮추고 warmup을 끈다
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=0.01, help="optimizer=auto이면 무시되고 자동 결정")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        optimizer=args.optimizer,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        patience=15,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        seed=0,
        plots=True,
    )
    # 최종(best) 모델로 검증 → mAP 출력
    save_dir = model.trainer.save_dir
    print(f"학습 결과 폴더: {save_dir}")
    best = YOLO(f"{save_dir}/weights/best.pt")
    m = best.val(data=args.data, imgsz=args.imgsz, device=args.device, project=str(Path("runs").resolve()), name=f"{args.name}_val", exist_ok=True)
    print("\n=== 검증 결과 ===")
    print(f"mAP50     : {m.box.map50:.4f}")
    print(f"mAP50-95  : {m.box.map:.4f}")
    print(f"Precision : {m.box.mp:.4f}")
    print(f"Recall    : {m.box.mr:.4f}")
    for i, name in best.names.items():
        print(f"  {name:9s} AP50={m.box.ap50[i]:.4f}  AP50-95={m.box.ap[i]:.4f}")


if __name__ == "__main__":
    main()
