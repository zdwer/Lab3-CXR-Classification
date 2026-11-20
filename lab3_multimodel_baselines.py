
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

CLASSES = ["normal", "bacteria", "virus", "COVID-19"]
DEFAULT_MODELS = [
    "resnet18", "resnet50", "densenet121", "efficientnet_b0",
    "efficientnet_b3", "mobilenet_v3_large", "convnext_tiny", "swin_t",
]
ALL_MODELS = DEFAULT_MODELS + ["vit_b_16"]  # Optional large transformer.
FILENAME_COLS = ("new_filename", "filename", "image_filename", "image_name",
                 "img_name", "image", "file_name", "file", "path", "image_path")
LABEL_COLS = ("label", "class", "target", "category", "diagnosis", "class_name")
LABEL_LOOKUP = {
    "normal": 0, "healthy": 0, "bacteria": 1, "bacterial": 1,
    "bacterialpneumonia": 1, "virus": 2, "viral": 2,
    "viralpneumonia": 2, "covid19": 3, "covid": 3,
    "0": 0, "1": 1, "2": 2, "3": 3,
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def get_device(name: str):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check torch.cuda.is_available() or use --device cpu.")
    return torch.device(name)


def make_transform(size: int, is_train: bool):
    ops = [transforms.Resize((size, size))]
    if is_train:

        ops += [transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08)]
    ops += [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


class LabeledCSVImages(Dataset):
    """Read Kaggle flat image directories with one-hot or string-label CSV."""

    def __init__(self, root: str, csv_path: str, image_size: int, train: bool):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image directory missing: {self.root}")
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Label CSV missing: {self.csv_path}")
        self.records: list[tuple[Path, int, str]] = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            fn_col = next((col for col in FILENAME_COLS if col in fields), None)
            is_onehot = all(c in fields for c in CLASSES)
            label_col = next((col for col in LABEL_COLS if col in fields), None)
            if fn_col is None or not (is_onehot or label_col):
                raise ValueError(f"Unrecognized CSV columns: {fields} in {self.csv_path}")
            names = set()
            for line, row in enumerate(reader, 2):
                fn = (row.get(fn_col) or "").strip()
                if not fn:
                    raise ValueError(f"Empty filename: {self.csv_path}:{line}")
                if fn in names:
                    raise ValueError(f"Duplicate filename in CSV: {fn}")
                names.add(fn)
                p = self.root / fn
                if not p.is_file():
                    p = self.root / Path(fn).name
                if not p.is_file():
                    raise FileNotFoundError(f"Missing image {fn} at {self.root}")
                if is_onehot:
                    flags = [float(row[c]) for c in CLASSES]
                    if any(v not in (0.0, 1.0) for v in flags) or sum(flags) != 1.0:
                        raise ValueError(f"Invalid one-hot labels: {self.csv_path}:{line}, {flags}")
                    target = flags.index(1.0)
                else:
                    raw = (row.get(label_col) or "").strip()
                    label = raw.casefold().replace(" ", "").replace("_", "").replace("-", "")
                    if label not in LABEL_LOOKUP:
                        raise ValueError(f"Unknown class label: {self.csv_path}:{line}, {raw}")
                    target = LABEL_LOOKUP[label]
                self.records.append((p, target, fn))
        if not self.records:
            raise ValueError(f"No samples found: {self.csv_path}")
        self.transform = make_transform(image_size, train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, target, _ = self.records[i]
        with Image.open(path) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor, target


class KaggleTestImages(Dataset):
    def __init__(self, img_dir: str, sample_csv: str, image_size: int):
        self.root = Path(img_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        with Path(sample_csv).open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if (reader.fieldnames or []) != ["new_filename"] + CLASSES:
                raise ValueError(f"Unexpected sample columns: {reader.fieldnames}")
            self.names = [(r.get("new_filename") or "").strip() for r in reader]
        if not self.names or len(self.names) != len(set(self.names)):
            raise ValueError("Empty/duplicate image names in sample CSV")
        for name in self.names:
            if not (self.root / name).is_file():
                raise FileNotFoundError(f"Missing Kaggle test image: {self.root / name}")
        self.transform = make_transform(image_size, False)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(self.root / self.names[i]) as im:
            return self.transform(im.convert("RGB"))


def make_model(name: str, pretrained: bool = True):
    """Use torchvision ImageNet weights for all model families, same 4-class head."""
    if name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    entries = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT),
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
    }
    factory, weight = entries[name]
    net = factory(weights=weight if pretrained else None)
    if name.startswith("resnet"):
        net.fc = nn.Linear(net.fc.in_features, 4)
    elif name.startswith("densenet"):
        net.classifier = nn.Linear(net.classifier.in_features, 4)
    elif name.startswith("efficientnet") or name.startswith("mobilenet") or name == "convnext_tiny":
        last = net.classifier[-1]
        net.classifier[-1] = nn.Linear(last.in_features, 4)
    elif name == "swin_t":
        net.head = nn.Linear(net.head.in_features, 4)
    elif name == "vit_b_16":
        net.heads.head = nn.Linear(net.heads.head.in_features, 4)
    return net


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        loss = (1 - torch.exp(-ce)).pow(self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def calc_metrics(y_true, y_pred):
    labels = list(range(4))
    p, r, f, n = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(n[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels_order": CLASSES,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    total, n = 0.0, 0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)  # FP32, no autocast. Avoid prior Exp5 NaN test issue.
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite validation logits; no checkpoint saved")
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities.extend(probs.tolist())
        y_true.extend(y.cpu().tolist())
        y_pred.extend(probs.argmax(axis=1).tolist())
        total += float(loss.item()) * len(y)
        n += len(y)
    result = calc_metrics(y_true, y_pred)
    result["val_loss"] = total / max(1, n)
    return result, np.asarray(probabilities, dtype=np.float64), y_true, y_pred


def save_best_predictions(path, ds, y_true, pred, probs):
    rows = []
    for i, (_, y, filename) in enumerate(ds.records):
        if y != y_true[i]:
            raise AssertionError("Validation ordering mismatch")
        row = {"filename": filename, "true_class": CLASSES[y],
               "pred_class": CLASSES[pred[i]], "correct": int(y == pred[i])}
        for j, name in enumerate(CLASSES):
            row[f"prob_{name}"] = round(float(probs[i, j]), 9)
        rows.append(row)
    fields = ["filename", "true_class", "pred_class", "correct"] + [f"prob_{c}" for c in CLASSES]
    write_csv(path, rows, fields)


def make_summary(root: Path, verbose: bool = True):
    rows = []
    for name in ALL_MODELS:
        p = root / name / "best_val_metrics.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        row = {"model": name, "best_epoch": m["epoch"], "val_macro_f1": round(m["macro_f1"], 6),
               "status": "complete" if (root / name / "DONE.json").exists() else "in_progress"}
        for cl in CLASSES:
            row[f"f1_{cl}"] = round(m["per_class"][cl]["f1"], 6)
        rows.append(row)
    rows.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    keys = ["rank", "model", "status", "best_epoch", "val_macro_f1"] + [f"f1_{c}" for c in CLASSES]
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "model_leaderboard.csv", rows, keys)
    if verbose:
        print("\n=== MULTI-MODEL BASELINE LEADERBOARD (validation only) ===")
        print(f"{'RANK':<5} {'MODEL':<23} {'MACRO-F1':<10} {'VIRUS':<9} {'COVID-19':<9} {'STATUS'}")
        for row in rows:
            print(f"{row['rank']:<5} {row['model']:<23} {row['val_macro_f1']:<10.6f} "
                  f"{row['f1_virus']:<9.6f} {row['f1_COVID-19']:<9.6f} {row['status']}")
        print(f"CSV: {root / 'model_leaderboard.csv'}")
    return rows


def validate_input(args):

    train = LabeledCSVImages(args.train_dir, args.train_csv, args.image_size, train=True)
    val = LabeledCSVImages(args.val_dir, args.val_csv, args.image_size, train=False)
    a = {p.name for p, _, _ in train.records}
    b = {p.name for p, _, _ in val.records}
    overlap = a.intersection(b)
    if overlap:
        raise ValueError(f"Train/val have overlapping filenames: {sorted(overlap)[:5]}")
    tc, vc = Counter(y for _, y, _ in train.records), Counter(y for _, y, _ in val.records)
    if any(tc[i] == 0 or vc[i] == 0 for i in range(4)):
        raise ValueError("A class has no samples in train or val")
    print("TRAIN:", len(train), {CLASSES[i]: tc[i] for i in range(4)})
    print("VAL:  ", len(val), {CLASSES[i]: vc[i] for i in range(4)})
    print("Note: filename-overlap check is NOT a patient-level leakage check.")
    return train, val, tc


def train_one(name, args, ds_train, ds_val, counts, device):
    folder = Path(args.output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / "DONE.json"
    if done.is_file():
        print(f"SKIP {name}: DONE.json exists (completed). Use new --output-dir for new experiment.")
        return
    config = {
        "model": name, "size": args.image_size, "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "lr": args.lr, "weight_decay": args.weight_decay, "seed": args.seed,
        "pretrained": not args.no_pretrained, "loss": "focal", "gamma": 2.0,
        "precision": "fp32", "augmentation": "original_exp3", "classes": CLASSES,
        "train_dir": str(Path(args.train_dir).resolve()),
        "train_csv": str(Path(args.train_csv).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()),
    }
    cfg_path = folder / "config.json"
    if cfg_path.is_file():
        old_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old_cfg != config:
            raise ValueError(f"Config changed for {name}. Use a new output directory.")
        if not args.resume:
            raise FileExistsError(f"{folder} already started. Add --resume to continue.")
    else:
        if (folder / "last_checkpoint.pt").exists():
            raise FileExistsError(f"{folder} has a checkpoint but no config. Inspect manually.")
        write_json(cfg_path, config)
    seed_all(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              generator=g)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    inv_sqrt = torch.tensor([1.0 / math.sqrt(counts[i]) for i in range(4)],
                            device=device, dtype=torch.float32)
    alpha = inv_sqrt / inv_sqrt.mean()
    criterion = FocalLoss(gamma=2.0, alpha=alpha)
    has_resume_ckpt = args.resume and (folder / "last_checkpoint.pt").is_file()
    model = make_model(name, pretrained=not args.no_pretrained and not has_resume_ckpt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, start, best, patience = [], 1, -1.0, 0
    if args.resume:
        ckpt = folder / "last_checkpoint.pt"
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            history = state["history"]
            start = int(state["epoch"]) + 1
            best = float(state["best_macro_f1"])
            patience = int(state["stale"])
            if "torch_rng" in state:
                torch.set_rng_state(state["torch_rng"].cpu())
            if device.type == "cuda" and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if "numpy_rng" in state:
                np.random.set_state(state["numpy_rng"])
            if "python_rng" in state:
                random.setstate(state["python_rng"])
            if "loader_rng" in state:
                g.set_state(state["loader_rng"].cpu())
            print(f"Resume {name}: epoch {start}, best={best:.6f}")
        elif (folder / "best_model.pt").exists():
            raise FileNotFoundError(f"{name}: missing last_checkpoint.pt for resume")
    if start > args.epochs or patience >= args.patience:
        write_json(done, {"model": name, "best_val_macro_f1": best, "status": "completed_previously"})
        return
    print(f"\n--- TRAIN {name} | {args.image_size}x{args.image_size}, "
          f"FP32, batch {args.batch_size}x{args.accum_steps}, focal, {device} ---", flush=True)
    for epoch in range(start, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, n = 0.0, 0
        batches = len(train_loader)
        for j, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"{name} epoch {epoch}: non-finite train logits")
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} epoch {epoch}: non-finite train loss")
            total += float(loss.item()) * len(y)
            n += len(y)

            group_start = (j // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, batches - group_start)
            (loss / group_size).backward()
            if (j + 1) % args.accum_steps == 0 or j + 1 == batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        report, probs, y_true, y_pred = validate(model, val_loader, criterion, device)
        scheduler.step()
        score = report["macro_f1"]
        row = {"epoch": epoch, "train_loss": round(total/max(n, 1), 7),
               "val_loss": round(report["val_loss"], 7),
               "val_macro_f1": round(score, 7)}
        history.append(row)
        write_csv(folder / "history.csv", history, list(row.keys()))
        if score > best + 1e-12:
            best, patience = score, 0
            report["epoch"] = epoch
            report["model"] = name
            write_json(folder / "best_val_metrics.json", report)
            save_best_predictions(folder / "best_val_predictions.csv", ds_val, y_true, y_pred, probs)
            torch.save({"model_name": name, "image_size": args.image_size,
                        "model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_macro_f1": score, "classes": CLASSES}, folder / "best_model.pt")
        else:
            patience += 1

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "stale": patience, "best_macro_f1": best,
                    "history": history, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "loader_rng": g.get_state()}, folder / "last_checkpoint.pt")
        print(f"{name} EPOCH {epoch:03d} | train={row['train_loss']:.5f} "
              f"val={row['val_loss']:.5f} macro_f1={score:.6f} best={best:.6f}", flush=True)
        make_summary(Path(args.output_dir), verbose=False)
        if patience >= args.patience:
            print(f"Early stopping for {name}: {args.patience} non-improving epochs")
            break
    write_json(done, {"model": name, "best_val_macro_f1": best,
                      "last_epoch": history[-1]["epoch"], "status": "completed"})
    print(f"COMPLETE {name} | best Val Macro-F1={best:.6f}", flush=True)
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def training_main(args):
    if args.image_size <= 0 or args.batch_size <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("Image size, batch-size, accum-steps, epochs must be positive")
    selected = DEFAULT_MODELS if args.models == ["default"] else args.models
    if len(set(selected)) != len(selected) or any(m not in ALL_MODELS for m in selected):
        raise ValueError(f"Invalid or repeated model names. Available: {ALL_MODELS}")
    ds_train, ds_val, counts = validate_input(args)
    if args.check_data_only:
        print("DATA CHECK PASSED. No training, no checkpoint files changed.")
        return
    device = get_device(args.device)
    print(f"Models selected: {selected}; device={device}; loss=Focal (same as Exp3), FP32")
    print("All 8 baseline models use common resolution 224 by default. This is NOT Exp7A 380px.")
    for k, name in enumerate(selected, 1):
        print(f"\n[{k}/{len(selected)}] {name}", flush=True)
        train_one(name, args, ds_train, ds_val, counts, device)
        make_summary(Path(args.output_dir))
    print("\nALL REQUESTED MODELS FINISHED.")
    make_summary(Path(args.output_dir))


@torch.no_grad()
def predict_main(args):
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = ckpt["model_name"]
    size = int(ckpt["image_size"])
    if ckpt.get("classes") != CLASSES:
        raise ValueError("Checkpoint class order mismatch")
    print(f"Model={name}, image_size={size}, device={device}")
    model = make_model(name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = KaggleTestImages(args.test_dir, args.sample_csv, size)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    all_probs = []
    for batch in loader:
        logits = model(batch.to(device, non_blocking=True))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite test logits: stop, do not submit CSV")
        p = torch.softmax(logits, dim=1).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("Non-finite test probabilities: stop")
        all_probs.extend(p.tolist())
    probs = np.asarray(all_probs, dtype=np.float64)
    if probs.shape != (len(dataset), 4):
        raise AssertionError("Wrong number of test predictions")
    if not np.allclose(probs.sum(1), 1, atol=1e-4):
        raise AssertionError("Test probabilities do not sum to one")
    preds = probs.argmax(axis=1)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Never alter sample order or add index columns.
    rows = []
    for i, filename in enumerate(dataset.names):
        row = {"new_filename": filename}
        for j, cls in enumerate(CLASSES):
            row[cls] = int(preds[i] == j)
        rows.append(row)
    write_csv(out, rows, ["new_filename"] + CLASSES)
    write_csv(out.with_name(out.stem + "_probabilities.csv"),
              [{"new_filename": dataset.names[i],
                **{f"prob_{cls}": round(float(probs[i, j]), 9) for j, cls in enumerate(CLASSES)}}
               for i in range(len(dataset))],
              ["new_filename"] + [f"prob_{x}" for x in CLASSES])
    print(f"Kaggle CSV created: {out} ({len(dataset)} rows, original order, one-hot)")
    print("Do NOT upload the _probabilities.csv file.")


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Lab3 8-model baseline training and CSV prediction")
    sub = ap.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train multiple models sequentially")
    train.add_argument("--train-dir", required=True)
    train.add_argument("--train-csv", required=True)
    train.add_argument("--val-dir", required=True)
    train.add_argument("--val-csv", required=True)
    train.add_argument("--output-dir", default="runs/multimodel_baselines")
    train.add_argument("--models", nargs="+", default=["default"], help="default = 8 models; or model names")
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accum-steps", type=int, default=4)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--lr", type=float, default=0.0001)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--check-data-only", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-pretrained", action="store_true", help="TESTING ONLY: no ImageNet weights")
    rank = sub.add_parser("rank", help="Rebuild ranking from saved model metrics")
    rank.add_argument("--output-dir", default="runs/multimodel_baselines")
    pred = sub.add_parser("predict", help="Predict test images using one completed model")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--test-dir", required=True)
    pred.add_argument("--sample-csv", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--batch-size", type=int, default=2)
    pred.add_argument("--num-workers", type=int, default=0)
    pred.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "train":
        training_main(args)
    elif args.command == "rank":
        make_summary(Path(args.output_dir))
    elif args.command == "predict":
        predict_main(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
