# -*- coding: utf-8 -*-
"""
数据准备：训练/验证划分 + 数据统计 + 生成 data.yaml
- 输入只读，只复制 val 子集到 work/，不修改原始 数据集3713
- 固定随机种子，保证可复现
"""
import os
import glob
import random
import shutil
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PROJECT_ROOT
DATA = os.path.join(ROOT, "数据集3713")
WORK = os.path.join(ROOT, "work")
VAL_FRAC = 0.10
SEED = 42
CLASS_NAMES = {0: "dent", 1: "hole", 2: "rusty"}


def list_images(split):
    return sorted(glob.glob(os.path.join(DATA, "images", split, "*.jpg")))


def label_path(img_path):
    p = os.path.splitext(img_path)[0]
    p = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return p + ".txt"


def count_boxes(lbl):
    n = 0
    with open(lbl) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def main():
    random.seed(SEED)
    train_imgs = list_images("train")
    test_imgs = list_images("test")
    print(f"train images = {len(train_imgs)}, test images = {len(test_imgs)}")

    # 划分：train -> train/val（9:1）
    random.shuffle(train_imgs)
    n_val = int(len(train_imgs) * VAL_FRAC)
    val_imgs = train_imgs[:n_val]
    train_keep = train_imgs[n_val:]
    print(f"after split: train={len(train_keep)}, val={len(val_imgs)}")

    # 复制 train/val 子集到 work/（不修改原始数据集，避免 train/val 数据泄漏）
    os.makedirs(os.path.join(WORK, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "labels", "val"), exist_ok=True)
    for img in train_keep:
        name = os.path.basename(img)
        lbl_name = os.path.splitext(name)[0] + ".txt"
        shutil.copy2(img, os.path.join(WORK, "images", "train", name))
        shutil.copy2(label_path(img), os.path.join(WORK, "labels", "train", lbl_name))
    for img in val_imgs:
        name = os.path.basename(img)
        lbl_name = os.path.splitext(name)[0] + ".txt"
        shutil.copy2(img, os.path.join(WORK, "images", "val", name))
        shutil.copy2(label_path(img), os.path.join(WORK, "labels", "val", lbl_name))

    # 统计（原始 train 全量）
    cls_counter = Counter()
    per_img = Counter()
    for img in train_imgs:
        lbl = label_path(img)
        cs = Counter()
        with open(lbl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                c = int(float(parts[0]))
                cs[c] += 1
                cls_counter[c] += 1
        per_img[sum(cs.values())] += 1
    print("train class dist (0=dent,1=hole,2=rusty):", dict(sorted(cls_counter.items())))
    print("train boxes per image dist:", dict(sorted(per_img.items())))

    # 写 data.yaml（绝对路径；train/val 指向 work/ 内的无重叠划分）
    yaml = f"""# YOLOv8 数据集配置（由 prep_data.py 生成, seed={SEED}）
path: {DATA}
train: {os.path.join(WORK, 'images', 'train')}
val: {os.path.join(WORK, 'images', 'val')}
test: {os.path.join(DATA, 'images', 'test')}
names:
  0: dent
  1: hole
  2: rusty
"""
    with open(os.path.join(WORK, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(yaml)
    print("wrote", os.path.join(WORK, "data.yaml"))


if __name__ == "__main__":
    main()
