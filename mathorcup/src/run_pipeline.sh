#!/usr/bin/env bash
# 赛后管道：检测基线评估 → yolov8s 训练/评估 → 最终模型推理 → 分类器训练 → 出图 → 复现清单 → 论文
# 用法：bash src/run_pipeline.sh
set -e
cd "$(dirname "$0")/.."
PY="C:/Users/lenovo/AppData/Local/Programs/Python/Python312/python"

echo "===== [1/8] yolov8n 评估（尺度消融基线） ====="
"$PY" src/evaluate.py --weights results/yolov8n_detect/weights/best.pt --scale n

echo "===== [2/8] yolov8s 训练（尺度消融） ====="
"$PY" src/train_detector.py --scale s --epochs 150 --batch 16 --name yolov8s_detect

echo "===== [3/8] yolov8s 评估 ====="
"$PY" src/evaluate.py --weights results/yolov8s_detect/weights/best.pt --scale s

echo "===== [4/8] 最终模型（yolov8s）推理 → test_result.csv ====="
"$PY" src/predict.py --weights results/yolov8s_detect/weights/best.pt --conf 0.25

echo "===== [5/8] ResNet-50 分类器训练 ====="
"$PY" src/train_classifier.py --epochs 25 --batch 32 --seed 42

echo "===== [6/8] 论文图表生成 ====="
"$PY" src/plot_figures.py

echo "===== [7/8] 复现清单 ====="
"$PY" src/make_manifest.py

echo "===== [8/8] 构建论文 ====="
"$PY" src/build_paper.py

echo "===== 管道完成 ====="
