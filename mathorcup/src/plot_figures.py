# -*- coding: utf-8 -*-
"""
统一出图脚本：生成三类图（raw / process / result），覆盖问题1/2/3。
输出 SVG + 300 DPI PNG 到 figures/。
依赖训练产物：results/yolov8{n,s}_detect/results.csv、results/classifier/*、results/eval/*.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
from plot_style import apply_publication_style, export_figure, PALETTE  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
apply_publication_style(language="zh", width="report")

NAMES = {0: "dent 凹陷", 1: "hole 破洞", 2: "rusty 锈蚀"}


def save(fig, stem):
    try:
        export_figure(fig, os.path.join(FIG, stem), dpi=300, strict_layout=False, strict_design=False)
    except Exception as e:  # 兼容旧版导出
        fig.savefig(os.path.join(FIG, stem + ".svg"))
        fig.savefig(os.path.join(FIG, stem + ".png"), dpi=300)
    plt.close(fig)


def imread_u(path):
    """读取图像（兼容 Windows 非 ASCII 路径，cv2.imread 对中文路径会失败）。"""
    import cv2
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------- raw ----------------
def raw_q1_class_distribution():
    counts = [3934, 946, 3158]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar([NAMES[i] for i in range(3)], counts, color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["contrast"]])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("训练集标注框数量")
    ax.set_title("问题1/2 原始数据：三类破损样本数量分布")
    save(fig, "raw_q1_class_distribution")


def raw_q2_gt_examples():
    import cv2
    imgs = sorted(glob.glob(os.path.join(ROOT, "数据集3713", "images", "train", "*.jpg")))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
    colors = [(0, 114, 178), (230, 159, 0), (213, 94, 0)]
    for ax, img in zip(axes.ravel(), imgs):
        im = cv2.cvtColor(imread_u(img), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        lbl = img.replace("images", "labels").replace(".jpg", ".txt")
        with open(lbl) as fh:
            for line in fh:
                c, xc, yc, w, h = [float(v) for v in line.split()]
                x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
                x2, y2 = (xc + w / 2) * W, (yc + h / 2) * H
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           edgecolor=np.array(colors[int(c)]) / 255, linewidth=1.2))
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("问题2 原始数据：训练样本与真值框示例", fontsize=9)
    save(fig, "raw_q2_gt_examples")


def raw_q3_boxes_per_image():
    lbls = glob.glob(os.path.join(ROOT, "数据集3713", "labels", "train", "*.txt"))
    counts = []
    for l in lbls:
        n = sum(1 for line in open(l) if line.strip())
        counts.append(n)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.hist(counts, bins=25, color=PALETTE["primary"], edgecolor="white")
    ax.set_xlabel("单张图像破损框数量")
    ax.set_ylabel("图像数量")
    ax.set_title("问题3 原始数据：单图破损框数量分布")
    save(fig, "raw_q3_boxes_per_image")


# ---------------- process ----------------
def _read_yolo_results(name):
    p = os.path.join(ROOT, "results", name, "results.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def process_q1_train_curve():
    # 分类器训练曲线：从日志重放不可得，改为从 metrics.json 生成占位说明（如有训练日志则覆盖）
    p = os.path.join(ROOT, "results", "classifier", "metrics.json")
    if not os.path.exists(p):
        return
    m = json.load(open(p, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    names = ["accuracy", "precision", "recall", "f1", "auc"]
    vals = [m[k] for k in names]
    ax.bar(names, vals, color=PALETTE["primary"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("取值")
    ax.set_title("问题1 过程：分类器验证指标")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
    save(fig, "process_q1_classifier_metrics")


def process_q2_train_loss(name="yolov8s_detect", tag="q2"):
    df = _read_yolo_results(name)
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].plot(df["epoch"], df["train/box_loss"], label="box", color=PALETTE["primary"])
    axes[0].plot(df["epoch"], df["train/cls_loss"], label="cls", color=PALETTE["secondary"])
    axes[0].plot(df["epoch"], df["train/dfl_loss"], label="dfl", color=PALETTE["contrast"])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("训练损失"); axes[0].legend()
    axes[0].set_title("训练损失曲线")
    m50 = "metrics/mAP50(B)" if "metrics/mAP50(B)" in df.columns else "metrics/mAP50"
    m5095 = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else "metrics/mAP50-95"
    axes[1].plot(df["epoch"], df[m50], color=PALETTE["positive"])
    axes[1].plot(df["epoch"], df[m5095], color=PALETTE["sky"])
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("mAP"); axes[1].legend(["mAP@0.5", "mAP@0.5:0.95"])
    axes[1].set_title("验证 mAP 曲线")
    fig.suptitle("问题2 过程：YOLOv8 训练收敛曲线", fontsize=9)
    save(fig, f"process_{tag}_train_loss")


def process_q3_threshold_scan():
    p = os.path.join(ROOT, "results", "classifier", "roc_data.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    y, pscore = d["y"], d["p"]
    ths = np.linspace(0.05, 0.95, 19)
    from sklearn.metrics import precision_score, recall_score, f1_score
    ps, rs, fs = [], [], []
    for t in ths:
        pred = (pscore >= t).astype(int)
        ps.append(precision_score(y, pred, zero_division=0))
        rs.append(recall_score(y, pred))
        fs.append(f1_score(y, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(ths, ps, label="Precision", color=PALETTE["primary"])
    ax.plot(ths, rs, label="Recall", color=PALETTE["secondary"])
    ax.plot(ths, fs, label="F1", color=PALETTE["positive"])
    ax.set_xlabel("判定阈值 τ"); ax.set_ylabel("指标")
    ax.legend(); ax.set_title("问题3 过程：分类阈值扫描")
    save(fig, "process_q3_threshold_scan")


# ---------------- result ----------------
def result_q1_roc():
    p = os.path.join(ROOT, "results", "classifier", "roc_data.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    y, pscore = d["y"], d["p"]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, pscore)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(fpr, tpr, color=PALETTE["primary"], label=f"AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    ax.set_xlabel("假阳性率 FPR"); ax.set_ylabel("真阳性率 TPR")
    ax.legend(); ax.set_title("问题1 结果：二分类 ROC 曲线")
    save(fig, "result_q1_roc")


def result_q2_detection(weights="results/yolov8s_detect/weights/best.pt"):
    import cv2
    from ultralytics import YOLO
    if not os.path.exists(os.path.join(ROOT, weights)):
        return
    model = YOLO(os.path.join(ROOT, weights))
    imgs = sorted(glob.glob(os.path.join(ROOT, "数据集3713", "images", "test", "*.jpg")))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.6))
    for ax, img in zip(axes.ravel(), imgs):
        im = cv2.cvtColor(imread_u(img), cv2.COLOR_BGR2RGB)
        res = model.predict(img, conf=0.25, verbose=False)[0]
        for box in res.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            ax.add_patch(plt.Rectangle((xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                                       fill=False, edgecolor=PALETTE["positive"], linewidth=1.4))
            ax.text(xyxy[0], xyxy[1] - 2, NAMES[cls], color="white", fontsize=6,
                    bbox=dict(facecolor=PALETTE["positive"], alpha=0.8, pad=0.5))
        ax.imshow(im); ax.axis("off")
    fig.suptitle("问题2 结果：测试集检测结果可视化", fontsize=9)
    save(fig, "result_q2_detection")


def result_q3_eval_compare():
    evals = glob.glob(os.path.join(ROOT, "results", "eval", "eval_*.json"))
    if not evals:
        return
    rows = []
    for e in evals:
        d = json.load(open(e, encoding="utf-8"))
        rows.append(d)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    x = df["scale"]
    axes[0].bar(x, df["mAP50"], color=PALETTE["primary"], label="mAP@0.5")
    axes[0].bar(x, df["mAP50_95"], color=PALETTE["sky"], label="mAP@0.5:0.95", bottom=0)
    axes[0].set_ylabel("mAP"); axes[0].legend(); axes[0].set_title("检测精度对比")
    axes[1].bar(x, df["infer_time_ms"], color=PALETTE["secondary"])
    axes[1].set_ylabel("单图推理时间 (ms)"); axes[1].set_title("推理速度对比")
    fig.suptitle("问题3 结果：不同尺度模型对比", fontsize=9)
    save(fig, "result_q3_eval_compare")


def process_all_roadmap():
    """技术路线图：数据→三问题→输出 的流程框图。"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc=PALETTE["primary"], tc="white", fs=8):
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.15", fc=fc, ec="none", lw=0)
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fs, weight="bold")

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                            color=PALETTE["neutral"], lw=1.1)
        ax.add_patch(a)

    # 顶部：数据
    box(5, 6.5, 8.6, 0.9, "数据勘察与预处理：类别分布统计、9:1 划分、\n背景裁剪合成负样本、数据增强",
        fc=PALETTE["dark"])
    # 三个子问题
    box(1.7, 4.6, 2.9, 1.3, "问题一\nResNet-50 迁移学习\n图像级二分类", fc=PALETTE["primary"])
    box(5, 4.6, 2.9, 1.3, "问题二\nYOLOv8 目标检测\n数据增强 + 多尺度融合", fc=PALETTE["secondary"])
    box(8.3, 4.6, 2.9, 1.3, "问题三\n五维评估 + 尺度消融\n精度/速度/规模对比", fc=PALETTE["positive"])
    # 各问题输出
    box(1.7, 2.9, 2.9, 0.8, "有/无残损判定", fc=PALETTE["sky"], tc="black")
    box(5, 2.9, 2.9, 0.8, "破损定位 + 类别", fc=PALETTE["sky"], tc="black")
    box(8.3, 2.9, 2.9, 0.8, "指标对比与选型结论", fc=PALETTE["sky"], tc="black")
    # 底部输出
    box(5, 1.2, 8.6, 1.0, "测试集推理（每图置信度降序至多 4 框）\n→ 写入 test_result.csv", fc=PALETTE["contrast"])

    arrow(5, 6.02, 1.7, 5.28)
    arrow(5, 6.02, 5, 5.28)
    arrow(5, 6.02, 8.3, 5.28)
    arrow(1.7, 3.92, 1.7, 3.32)
    arrow(5, 3.92, 5, 3.32)
    arrow(8.3, 3.92, 8.3, 3.32)
    arrow(1.7, 2.48, 4.4, 1.72)
    arrow(5, 2.48, 5, 1.72)
    arrow(8.3, 2.48, 5.6, 1.72)
    fig.suptitle("技术路线图", fontsize=9)
    save(fig, "process_all_roadmap")


def process_q1_train_curve():
    """分类器训练曲线（逐轮 loss / acc / auc）。"""
    p = os.path.join(ROOT, "results", "classifier", "history.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].plot(d["epoch"], d["loss"], color=PALETTE["primary"], label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("损失"); axes[0].legend()
    axes[0].set_title("训练损失曲线")
    axes[1].plot(d["epoch"], d["acc"], color=PALETTE["secondary"], label="accuracy")
    axes[1].plot(d["epoch"], d["auc"], color=PALETTE["positive"], label="AUC")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("指标"); axes[1].legend()
    axes[1].set_title("验证集准确率与 AUC")
    fig.suptitle("问题1 过程：ResNet-50 分类器训练曲线", fontsize=9)
    save(fig, "process_q1_train_curve")


def result_q2_perclass_ap():
    """问题2 各类别 AP 对比柱状图。"""
    s_eval = os.path.join(ROOT, "results", "eval", "eval_s.json")
    if not os.path.exists(s_eval):
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    d = json.load(open(s_eval, encoding="utf-8"))
    per = d.get("per_class_AP50", {})
    names = list(per.keys())
    vals = [per[k] for k in names]
    bars = ax.bar(names, vals, color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["contrast"]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("AP@0.5")
    ax.set_title("问题2 结果：各类别 AP@0.5")
    save(fig, "result_q2_perclass_ap")


def result_q3_confusion(weights="results/yolov8s_detect/weights/best.pt"):
    """问题3 混淆矩阵（基于验证集）。"""
    import cv2
    from ultralytics import YOLO
    wp = os.path.join(ROOT, weights)
    if not os.path.exists(wp):
        return
    model = YOLO(wp)
    # 在验证集上跑 val，ultralytics 会生成 confusion_matrix.png
    model.val(data=os.path.join(ROOT, "work", "data.yaml"), split="val", plots=True, verbose=False,
              project=os.path.join(ROOT, "results", "conf"), name="cm", exist_ok=True)


if __name__ == "__main__":
    process_all_roadmap()
    raw_q1_class_distribution()
    raw_q2_gt_examples()
    raw_q3_boxes_per_image()
    process_q1_train_curve()
    process_q2_train_loss()
    process_q3_threshold_scan()
    result_q1_roc()
    result_q2_detection()
    result_q2_perclass_ap()
    result_q3_eval_compare()
    result_q3_confusion()
    print("figures generated in", FIG)
