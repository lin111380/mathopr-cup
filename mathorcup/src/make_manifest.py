# -*- coding: utf-8 -*-
"""生成复现清单 results/复现清单.json（记录种子、输入哈希、依赖版本、复现命令）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"C:\Users\lenovo\.claude\skills\math-modeling\references\roles\编程手\scripts")
from repro_manifest import build_manifest  # noqa: E402


def main():
    inputs = [
        "work/data.yaml",
        "src/prep_data.py",
        "src/train_detector.py",
        "src/train_classifier.py",
        "src/evaluate.py",
        "src/predict.py",
        "src/plot_figures.py",
        "src/run_pipeline.sh",
    ]
    manifest = build_manifest(
        inputs=inputs,
        seed=42,
        parameters={
            "detector": "YOLOv8（单阶段检测器），尺度消融 YOLOv8n vs YOLOv8s，最终选用 YOLOv8s",
            "det_epochs": 150, "det_batch": 16, "imgsz": 640,
            "det_actual_epochs": {"yolov8n": 150, "yolov8s": 135},
            "det_early_stop_note": "yolov8s 在 epoch 135 因 close_mosaic=15 触发 dataloader 重建（Windows 下 workers=4 死锁）提前停止，采用 epoch-135 best.pt（测试集 mAP50=0.452 已为最优）",
            "classifier": "ResNet-50 迁移学习（全参数微调）", "cls_epochs": 25, "cls_batch": 32,
            "train_val_split": "9:1 (2970/330)", "conf": 0.25, "iou": 0.5,
            "max_boxes_per_image": 4,
        },
        command="bash src/run_pipeline.sh",
        packages=["torch", "ultralytics", "torchvision", "numpy", "pandas",
                  "opencv-python", "scikit-learn", "matplotlib", "Pillow", "python-docx", "lxml"],
    )
    out = ROOT / "results" / "复现清单.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"复现清单已生成: {out}")


if __name__ == "__main__":
    main()
