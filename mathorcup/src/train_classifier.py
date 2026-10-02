# -*- coding: utf-8 -*-
"""
问题1：图像级二分类模型训练（ResNet-50 迁移学习）
- 正样本：含破损的整幅训练图
- 负样本：从训练图不含任何破损框的区域裁剪的"背景"块（合成无残损样本）
- 输出分类指标（Accuracy/Precision/Recall/F1/AUC）与 ROC 曲线数据
用法：
  python src/train_classifier.py --epochs 25 --batch 32 --seed 42
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "数据集3713")
OUT = os.path.join(ROOT, "results")


def read_boxes_yolo(lbl_path):
    boxes = []
    with open(lbl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c, xc, yc, w, h = [float(v) for v in line.split()]
            boxes.append((c, xc, yc, w, h))
    return boxes


def crop_background(img_path, lbl_path, size=320, max_tries=50, seed=None):
    """在图像中随机采样一块不含任何破损框的区域，作为负样本。"""
    rng = random.Random(seed)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    boxes = read_boxes_yolo(lbl_path)
    for _ in range(max_tries):
        x = rng.uniform(0, 1 - size / W)
        y = rng.uniform(0, 1 - size / H)
        # 归一化候选区域
        x1, y1 = x / W, y / H
        x2, y2 = (x + size) / W, (y + size) / H
        overlap = False
        for (_, bx, by, bw, bh) in boxes:
            bx1, by1 = bx - bw / 2, by - bh / 2
            bx2, by2 = bx + bw / 2, by + bh / 2
            if x2 > bx1 and bx2 > x1 and y2 > by1 and by2 > y1:
                overlap = True
                break
        if not overlap:
            crop = img.crop((int(x), int(y), int(x + size), int(y + size)))
            return crop
    return None  # 找不到干净区域（框太密）则放弃


class DefectDataset(Dataset):
    def __init__(self, items, transform, pos_label=1):
        self.items = items  # list of (path_or_pil, label)
        self.transform = transform
        self.pos_label = pos_label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src, label = self.items[i]
        if isinstance(src, str):
            img = Image.open(src).convert("RGB")
        else:
            img = src
        return self.transform(img), float(label)


def build_splits(seed=42):
    random.seed(seed)
    train_imgs = sorted(glob.glob(os.path.join(DATA, "images", "train", "*.jpg")))
    random.shuffle(train_imgs)
    n_val = int(len(train_imgs) * 0.1)
    val_pos = train_imgs[:n_val]
    train_pos = train_imgs[n_val:]

    def make_negs(img_list, prefix):
        negs = []
        for i, img in enumerate(img_list):
            lbl = img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).replace(".jpg", ".txt")
            crop = crop_background(img, lbl, seed=seed + i)
            if crop is not None:
                negs.append((crop, 0))
        return negs

    train_neg = make_negs(train_pos, "train")
    val_neg = make_negs(val_pos, "val")
    train_items = [(p, 1) for p in train_pos] + train_neg
    val_items = [(p, 1) for p in val_pos] + val_neg
    return train_items, val_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_items, val_items = build_splits(args.seed)
    print(f"train items = {len(train_items)}, val items = {len(val_items)}")
    train_ds = DefectDataset(train_items, transform)
    val_ds = DefectDataset(val_items, transform)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * x.size(0)
        tl /= len(train_ds)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                logits = model(x)
                p = torch.sigmoid(logits).cpu().numpy().ravel()
                ps.extend(p)
                ys.extend(y.numpy().ravel())
        ys = np.array(ys)
        ps = np.array(ps)
        pred = (ps >= 0.5).astype(int)
        acc = accuracy_score(ys, pred)
        auc = roc_auc_score(ys, ps)
        history.append([float(epoch), float(tl), float(acc), float(auc)])
        print(f"epoch {epoch:02d} loss={tl:.4f} acc={acc:.4f} auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            os.makedirs(os.path.join(OUT, "classifier"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUT, "classifier", "resnet50_best.pt"))

    # 保存逐轮训练历史（供绘制训练曲线）
    os.makedirs(os.path.join(OUT, "classifier"), exist_ok=True)
    np.savez(os.path.join(OUT, "classifier", "history.npz"),
             epoch=np.array([h[0] for h in history]),
             loss=np.array([h[1] for h in history]),
             acc=np.array([h[2] for h in history]),
             auc=np.array([h[3] for h in history]))

    # 最终指标
    model.load_state_dict(torch.load(os.path.join(OUT, "classifier", "resnet50_best.pt"), map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy().ravel()
            ps.extend(p)
            ys.extend(y.numpy().ravel())
    ys = np.array(ys)
    ps = np.array(ps)
    pred = (ps >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(ys, pred),
        "precision": precision_score(ys, pred, zero_division=0),
        "recall": recall_score(ys, pred, zero_division=0),
        "f1": f1_score(ys, pred, zero_division=0),
        "auc": roc_auc_score(ys, ps),
    }
    print("final metrics:", metrics)
    import json
    with open(os.path.join(OUT, "classifier", "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    np.savez(os.path.join(OUT, "classifier", "roc_data.npz"), y=ys, p=ps)
    print("best_auc =", best_auc)


if __name__ == "__main__":
    main()
