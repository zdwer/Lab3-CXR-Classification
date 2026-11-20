

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lab3_efficientnet_b3_csv_v2 import CLASSES, CSVDataset, FocalLoss, build_model


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(image_size, train, experiment):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        if experiment == "A":
            ops += [transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
                    transforms.ColorJitter(brightness=0.08, contrast=0.08)]
        elif experiment == "B":
            ops += [transforms.RandomAffine(degrees=7, translate=(0.04, 0.04),
                                            scale=(0.95, 1.05)),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12)]
        else:
            raise ValueError(f"Unknown experiment: {experiment}")
    ops += [transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def metrics(y_true, y_pred):
    ids = list(range(4))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids,
                                   average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
        "labels_order": CLASSES,
    }


def validate(model, loader, device, loss_fn):
    model.eval()
    actual, predicted = [], []
    loss_sum, n = 0., 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)  
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite validation logits")
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            actual.extend(y.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    result = metrics(actual, predicted)
    result["val_loss"] = loss_sum / max(1, n)
    return result


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def save_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset_from_csv(root, csvfile, image_size, train, experiment):
    ds = CSVDataset(root, csvfile, train=False)
    ds.transform = make_transform(image_size, train=train, experiment=experiment)
    return ds


def check_split(train_ds, val_ds):
    tc = Counter(int(label) for _, label in train_ds.records)
    vc = Counter(int(label) for _, label in val_ds.records)
    print("Train:", len(train_ds), {CLASSES[i]: tc[i] for i in range(4)})
    print("Val:  ", len(val_ds), {CLASSES[i]: vc[i] for i in range(4)})
    tnames = {p.name for p, _ in train_ds.records}
    vnames = {p.name for p, _ in val_ds.records}
    overlap = tnames & vnames
    if overlap:
        raise ValueError(f"Train/val filename overlap: {list(sorted(overlap))[:8]}")
    if not tc or not vc or min(tc.values()) < 1 or min(vc.values()) < 1:
        raise ValueError("One or more classes missing from a split")


def train(args):
    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = dataset_from_csv(args.train_dir, args.train_csv, args.image_size,
                                train=True, experiment=args.experiment)
    val_ds = dataset_from_csv(args.val_dir, args.val_csv, args.image_size,
                              train=False, experiment=args.experiment)
    check_split(train_ds, val_ds)
    if args.check_data_only:
        print(f"DATA CHECK PASSED | Exp7{args.experiment} | image={args.image_size}x{args.image_size}")
        print("No model initialized or training performed.")
        return

    if args.batch_size <= 0 or args.accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch-size, accum-steps, and epochs must be positive")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": args.experiment, "image_size": args.image_size,
        "batch_size": args.batch_size, "accum_steps": args.accum_steps,
        "epochs": args.epochs, "patience": args.patience,
        "lr": args.lr, "weight_decay": 1e-4, "seed": args.seed,
        "loss": "focal", "gamma": 2.0, "precision": "fp32",
        "train_dir": str(Path(args.train_dir).resolve()),
        "train_csv": str(Path(args.train_csv).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()),
    }
    config_path = out / "config.json"
    if args.resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config for resume: {config_path}")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise ValueError("Resume configuration differs from existing run. Use a new output-dir.")
    else:
        if (out / "last_checkpoint.pt").exists() or (out / "best_model.pt").exists():
            raise FileExistsError("Output folder contains a trained model. Use --resume or a new output-dir.")
        save_json(config_path, config)

    counts = Counter(int(y) for _, y in train_ds.records)
    inv_sqrt = torch.tensor([1 / math.sqrt(counts[i]) for i in range(4)],
                            dtype=torch.float32, device=device)
    weights = inv_sqrt / inv_sqrt.mean()
    loss_fn = FocalLoss(gamma=2.0, alpha=weights)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(pretrained=(not args.no_pretrained and not args.resume)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, stale, start_epoch, history = -1., 0, 1, []
    if args.resume:
        path = out / "last_checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing resume checkpoint: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best = float(ckpt["best_macro_f1"])
        stale = int(ckpt["stale"])
        history = ckpt["history"]
        start_epoch = int(ckpt["epoch"]) + 1
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"].cpu())
        if device.type == "cuda" and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        if "rng_numpy" in ckpt:
            np.random.set_state(ckpt["rng_numpy"])
        if "rng_python" in ckpt:
            random.setstate(ckpt["rng_python"])
        print(f"Resuming at epoch {start_epoch}, best={best:.6f}")
        if stale >= args.patience or start_epoch > args.epochs:
            print("Previous run already stopped or reached max epochs; nothing to resume.")
            return

    print(f"Device={device} | Exp7{args.experiment} | image={args.image_size} | "
          f"batch={args.batch_size}x{args.accum_steps} | FP32")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_n = 0., 0
        batches = len(train_loader)
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite training logits at epoch {epoch} batch {step+1}")
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch} batch {step+1}")
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)
            # Tail group can be smaller than accum_steps.
            group_start = (step // args.accum_steps) * args.accum_steps
            group_size = min(args.accum_steps, batches - group_start)
            (loss / group_size).backward()
            if (step + 1) % args.accum_steps == 0 or step + 1 == batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        report = validate(model, val_loader, device, loss_fn)
        scheduler.step()
        score = report["macro_f1"]
        record = {"epoch": epoch, "train_loss": round(total_loss/max(1,total_n), 6),
                  "val_loss": round(report["val_loss"], 6),
                  "val_macro_f1": round(score, 6)}
        history.append(record)
        save_history(out / "history.csv", history)

        if score > best + 1e-6:
            best, stale = score, 0
            model_info = {"model_state_dict": model.state_dict(),
                          "classes": CLASSES, "epoch": epoch,
                          "val_macro_f1": score, "loss": "focal",
                          "image_size": args.image_size,
                          "experiment": args.experiment}
            torch.save(model_info, out / "best_model.pt")
            save_json(out / "best_val_metrics.json", {"epoch": epoch, **report})
        else:
            stale += 1

        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch, "best_macro_f1": best,
                    "stale": stale, "history": history,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "rng_numpy": np.random.get_state(),
                    "rng_python": random.getstate()},
                   out / "last_checkpoint.pt")
        print(f"EPOCH {epoch:03d} | train_loss={record['train_loss']:.6f} "
              f"val_loss={record['val_loss']:.6f} val_macro_f1={score:.6f} "
              f"best={best:.6f}", flush=True)
        if stale >= args.patience:
            print(f"EARLY STOP: no improvement for {stale} epochs")
            break
    print(f"TRAINING COMPLETE | Exp7{args.experiment} | best val Macro-F1={best:.6f} "
          f"| best checkpoint={out/'best_model.pt'}")


class TestImages(Dataset):
    def __init__(self, image_root, filenames, image_size):
        self.root = Path(image_root)
        self.names = filenames
        self.transform = make_transform(image_size, False, "A")
        for name in filenames:
            if not (self.root / name).is_file():
                raise FileNotFoundError(f"Missing test image: {self.root / name}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / name) as im:
            tensor = self.transform(im.convert("RGB"))
        return tensor


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("classes") != CLASSES:
        raise ValueError("Checkpoint class order differs from Kaggle's four-class order")
    image_size = int(ckpt["image_size"])
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with Path(args.sample_csv).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["new_filename"] + CLASSES:
            raise ValueError(f"Unexpected Kaggle CSV header: {reader.fieldnames}")
        filenames = [row["new_filename"].strip() for row in reader]
    if not filenames or len(set(filenames)) != len(filenames) or any(not x for x in filenames):
        raise ValueError("Invalid or duplicate test image filenames")
    ds = TestImages(args.test_dir, filenames, image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    predictions, probabilities = [], []
    with torch.no_grad():
        for x in loader:
            p = torch.softmax(model(x.to(device)).float(), dim=1).cpu().numpy()
            if not np.isfinite(p).all():
                raise FloatingPointError("Non-finite test probabilities: submission was not written")
            predictions.extend(np.argmax(p, axis=1).tolist())
            probabilities.extend(p.tolist())
    if len(predictions) != len(filenames):
        raise AssertionError("Test prediction count mismatch")

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_filename"] + CLASSES)
        for name, pred in zip(filenames, predictions):
            w.writerow([name] + [int(i == pred) for i in range(4)])
    probs_target = target.with_name(target.stem + "_probabilities.csv")
    with probs_target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_filename"] + ["prob_" + x for x in CLASSES])
        for name, p in zip(filenames, probabilities):
            w.writerow([name] + [f"{v:.9g}" for v in p])
    print(f"Wrote {target} ({len(filenames)} predictions, ordered as sample CSV)")
    print(f"Wrote probabilities (analysis only, not for Kaggle): {probs_target}")
    print("No Kaggle submission performed.")


def main():
    parser = argparse.ArgumentParser(description="Lab3 Exp7 high-resolution EfficientNet-B3")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train Exp7A or Exp7B on original train/val split")
    t.add_argument("--train-dir", required=True)
    t.add_argument("--train-csv", required=True)
    t.add_argument("--val-dir", required=True)
    t.add_argument("--val-csv", required=True)
    t.add_argument("--experiment", choices=["A", "B"], required=True)
    t.add_argument("--output-dir", required=True)
    t.add_argument("--image-size", type=int, default=380)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--patience", type=int, default=7)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--accum-steps", type=int, default=2)
    t.add_argument("--lr", type=float, default=0.0001)
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-pretrained", action="store_true")
    t.add_argument("--check-data-only", action="store_true")
    t.add_argument("--resume", action="store_true")
    p = sub.add_parser("predict", help="Predict Kaggle test set using Exp7 checkpoint")
    p.add_argument("--test-dir", required=True)
    p.add_argument("--sample-csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
