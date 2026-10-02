# -*- coding: utf-8 -*-
"""
问题2：目标检测模型训练（YOLOv8）
- 训练 YOLOv8n（主）与 YOLOv8s（尺度消融）
- 输出训练过程图、指标，best.pt 到 results/
用法：
  python src/train_detector.py --scale n --epochs 150 --batch 16

已知问题：close_mosaic=15 会在 epoch 135 触发 dataloader 重建，Windows 下 workers=4
可能死锁（GPU 0% 但进程存活）。本次 yolov8s 即因此提前停止于 epoch 135，best.pt 已保存。
"""
import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "work", "data.yaml")
OUT = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="n", choices=["n", "s"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"yolov8{args.scale}_detect"
    model = YOLO(f"yolov8{args.scale}.pt")  # 预训练权重
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        project=OUT,
        name=name,
        patience=30,
        workers=4,
        pretrained=True,
        # 针对复杂背景/小目标的增强
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
        close_mosaic=15,  # 最后 15 轮关闭 mosaic，稳定收敛
        plots=True,
        verbose=True,
    )
    print("训练完成，best.pt 位于 results/yolov8{args.scale}_detect/weights/best.pt")


if __name__ == "__main__":
    main()
