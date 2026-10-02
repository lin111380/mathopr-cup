# -*- coding: utf-8 -*-
"""
问题3：模型评估（多维度）
- 检测：mAP@0.5 / mAP@0.5:0.95 / 各类 AP / P / R / 混淆矩阵
- 规模与速度：参数量、模型大小、单图推理时间
- 输出评估指标表与数据（供绘图）
用法：
  python src/evaluate.py --weights results/yolov8n_detect/weights/best.pt --scale n
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "work", "data.yaml")
OUT = os.path.join(ROOT, "results", "eval")
NAMES = {0: "dent", 1: "hole", 2: "rusty"}


def count_params(model):
    p = sum(x.numel() for x in model.model.parameters())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scale", default="n")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    model = YOLO(args.weights)

    # 检测指标：在测试集（有标注）上验证
    metrics = model.val(data=DATA_YAML, split="test", plots=True, verbose=False)
    mAP50 = metrics.box.map50
    mAP = metrics.box.map
    per_cls = metrics.box.ap50  # 每类 AP50 (列表，按类别顺序)
    mp = metrics.box.mp
    mr = metrics.box.mr
    print(f"mAP@0.5={mAP50:.4f}  mAP@0.5:0.95={mAP:.4f}  P={mp:.4f}  R={mr:.4f}")

    # 规模
    n_params = count_params(model)
    size_mb = os.path.getsize(args.weights) / 1e6

    # 速度：测试集前 20 张平均推理时间
    from PIL import Image
    import glob
    imgs = sorted(glob.glob(os.path.join(ROOT, "数据集3713", "images", "test", "*.jpg")))[:20]
    model.predict(imgs[0], verbose=False)  # warmup
    t0 = time.time()
    for img in imgs:
        model.predict(img, verbose=False)
    dt = (time.time() - t0) / len(imgs)

    result = {
        "scale": args.scale,
        "mAP50": float(mAP50),
        "mAP50_95": float(mAP),
        "precision": float(mp),
        "recall": float(mr),
        "per_class_AP50": {NAMES[i]: float(per_cls[i]) for i in range(len(NAMES))},
        "params": int(n_params),
        "model_size_MB": round(size_mb, 2),
        "infer_time_ms": round(dt * 1000, 2),
    }
    with open(os.path.join(OUT, f"eval_{args.scale}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
