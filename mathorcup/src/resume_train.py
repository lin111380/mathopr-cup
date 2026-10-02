# -*- coding: utf-8 -*-
"""
从 last.pt 续训 YOLOv8（承接 train_detector.py 未跑完的 150 epochs）
用法：
  python src/resume_train.py --weights results/yolov8n_detect/weights/last.pt
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="results/yolov8n_detect/weights/last.pt")
    args = ap.parse_args()

    model = YOLO(args.weights)  # 加载 checkpoint（内含全部训练参数与优化器状态）
    model.train(resume=True)


if __name__ == "__main__":
    main()
