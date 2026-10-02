# -*- coding: utf-8 -*-
"""
用训练好的 YOLOv8 检测器对测试集推理，生成提交文件 test_result.csv
- 每张图按置信度降序，最多取 4 个框（赛题提示2）
- 输出列：image_id,class_id,x_center,y_center,width,height（YOLO 归一化）
用法：
  python src/predict.py --weights results/yolov8n_detect/weights/best.pt --conf 0.25
"""
import argparse
import glob
import os

import pandas as pd
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "数据集3713", "images", "test")
OUT_CSV = os.path.join(ROOT, "test_result.csv")
MAX_BOXES = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    model = YOLO(args.weights)
    imgs = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
    print(f"test images = {len(imgs)}")

    rows = []
    for img in imgs:
        name = os.path.basename(img)
        res = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        # 收集 (confidence, class_id, xc, yc, w, h)
        dets = []
        for box in res.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            xc = (xyxy[0] + xyxy[2]) / 2 / res.orig_shape[1]
            yc = (xyxy[1] + xyxy[3]) / 2 / res.orig_shape[0]
            w = (xyxy[2] - xyxy[0]) / res.orig_shape[1]
            h = (xyxy[3] - xyxy[1]) / res.orig_shape[0]
            dets.append((conf, cls, xc, yc, w, h))
        dets.sort(key=lambda d: -d[0])
        for conf, cls, xc, yc, w, h in dets[:MAX_BOXES]:
            rows.append([name, cls, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)])

    df = pd.DataFrame(rows, columns=["image_id", "class_id", "x_center", "y_center", "width", "height"])
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} with {len(df)} rows from {len(imgs)} images")


if __name__ == "__main__":
    main()
