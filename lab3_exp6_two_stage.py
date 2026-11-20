
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
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lab3_efficientnet_b3_csv_v2 import (
    CLASSES, CSVDataset, FocalLoss, build_model, create_transform
)

CLASSES_2 = ['bacteria', 'virus']
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIELDS = ['new_filename'] + CLASSES


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path, fields, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def macro_f1(truth, pred, num_classes=4):
    return float(f1_score(truth, pred, average='macro',
                          labels=list(range(num_classes)), zero_division=0))


def metrics_4(truth, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=[0, 1, 2, 3], zero_division=0
    )
    return {
        'macro_f1': macro_f1(truth, pred),
        'per_class': {c: {'precision': float(precision[i]),
                          'recall': float(recall[i]),
                          'f1': float(f1[i]),
                          'support': int(support[i])}
                      for i, c in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(truth, pred, labels=[0, 1, 2, 3]).tolist(),
    }


def ensure_probs(p, nclasses, context):
    if p.ndim != 2 or p.shape[1] != nclasses:
        raise ValueError(f'{context}: invalid shape: {p.shape}')
    bad = np.where(~np.isfinite(p).all(axis=1))[0]
    if len(bad):
        raise ValueError(f'{context}: NaN/inf probability rows: {bad[:20].tolist()}')
    if np.any(p < -1e-6) or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError(f'{context}: invalid probability range or sums')
    return p


def fuse(p4, p2, alpha):
    mass = p4[:, 1] + p4[:, 2]
    merged = p4.copy()
    merged[:, 1] = (1 - alpha)*p4[:, 1] + alpha*mass*p2[:, 0]
    merged[:, 2] = (1 - alpha)*p4[:, 2] + alpha*mass*p2[:, 1]
    return ensure_probs(merged, 4, f'fusion alpha={alpha}')


def alpha_score(p4, p2, truth, alpha, subset=None):
    if subset is None:
        subset = np.arange(len(truth))
    pred = fuse(p4[subset], p2[subset], alpha).argmax(axis=1)
    return macro_f1(truth[subset], pred)


def choose_alpha(p4, p2, truth, indices):
    # When ties occur, prefer no modification of the existing Stage 1 model.
    scores = [(alpha_score(p4, p2, truth, a, indices), a) for a in ALPHAS]
    return sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]


def crossfit_fusion(p4, p2, truth, seed):

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(truth), dtype=np.int64)
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(truth)), truth), start=1):
        alpha = choose_alpha(p4, p2, truth, tr)
        fused = fuse(p4[va], p2[va], alpha)
        pred[va] = fused.argmax(1)
        baseline = macro_f1(truth[va], p4[va].argmax(1))
        tuned = macro_f1(truth[va], pred[va])
        rows.append({'fold': fold, 'selected_alpha': alpha,
                     'baseline_macro_f1': baseline,
                     'tuned_macro_f1': tuned,
                     'delta': tuned - baseline,
                     'val_count': len(va)})
    base = macro_f1(truth, p4.argmax(1))
    result = macro_f1(truth, pred)
    return {
        'crossfit_macro_f1': result,
        'baseline_macro_f1': base,
        'gain': result-base,
        'improved_folds': sum(r['delta'] > 1e-9 for r in rows),
        'folds': rows,
        'note': 'Crossfit checks alpha stability only. Both neural models selected on the same validation set.'
    }


class BinaryDataset(Dataset):
    def __init__(self, base):
        self.base = base
        self.indices = [idx for idx, (_, y) in enumerate(base.records) if y in (1, 2)]
        if not self.indices:
            raise ValueError('No Bacteria or Virus samples found in CSV')
        self.labels = [base.records[idx][1]-1 for idx in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, y = self.base[self.indices[index]]
        return image, int(y)-1


class TestImages(Dataset):
    def __init__(self, folder, filenames):
        self.paths = [Path(folder)/s for s in filenames]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing test images: {missing[:5]}')
        self.transform = create_transform(False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert('RGB'))


def sample_names(csv_path):
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'Wrong sample CSV fields: {reader.fieldnames}; expected {FIELDS}')
        names = [(r['new_filename'] or '').strip() for r in reader]
    if len(names) != 1182 or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Sample must have exactly 1182 unique, nonempty filenames')
    return names


def load_stage1(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if list(checkpoint.get('classes', [])) != list(CLASSES):
        raise ValueError(f'Stage1 checkpoint class order mismatched: {checkpoint.get("classes")}')
    model = build_model(False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    print(f'Stage1 loaded: {path}; epoch={checkpoint.get("epoch")}; '
          f'val_macro_f1={checkpoint.get("val_macro_f1")}')
    return model


def build_stage2(stage1_path=None):
    model = build_model(False)
    if stage1_path:
        checkpoint = torch.load(stage1_path, map_location='cpu', weights_only=False)
        if list(checkpoint.get('classes', [])) != list(CLASSES):
            raise ValueError('Stage1 checkpoint has unexpected classes')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print('Stage2 initialized from Exp3 Stage1 backbone')
    else:
        # ImageNet initialization is supported, but may require a network download.
        model = build_model(True)
        print('Stage2 initialized from ImageNet weights')
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def load_stage2(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('classes') != CLASSES_2:
        raise ValueError('Stage2 checkpoint must have classes [bacteria, virus]')
    model = build_model(False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    print(f'Stage2 loaded: {path}; epoch={ckpt.get("epoch")}; '
          f'binary_val_macro_f1={ckpt.get("val_macro_f1")}')
    return model


@torch.no_grad()
def predict_probs(model, loader, device, nclasses, name):
    model.eval()
    chunks = []
    for batch in loader:
        # Train/validation datasets also return integer labels.
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        logits = model(x).float()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(f'{name}: non-finite logits. Inference always uses FP32.')
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not chunks:
        raise ValueError(f'{name}: empty loader')
    return ensure_probs(np.concatenate(chunks), nclasses, name)


def train_stage2(args, train_base, val_base, device, out):
    train_ds = BinaryDataset(train_base)
    val_ds = BinaryDataset(val_base)
    counts = Counter(train_ds.labels)
    print('Stage2 train:', len(train_ds), 'counts:', dict(counts),
          'Stage2 val:', len(val_ds))
    weights = torch.tensor([1/math.sqrt(counts[i]) for i in range(2)],
                           dtype=torch.float32, device=device)
    weights = weights / weights.mean()
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=2.0, alpha=weights)
    elif args.loss == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    model = build_stage2(None if args.from_imagenet else args.stage1_ckpt).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type=='cuda')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    amp = args.amp and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history = []
    best_f1, stale = -1.0, 0
    ckpt_path = out/'best_binary_model.pt'

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f'Nonfinite training loss at epoch {epoch}; try FP32 (omit --amp)')
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
            loss_sum += float(loss.item())*len(y)
            seen += len(y)

        model.eval()
        labels, predictions = [], []
        val_loss_sum, val_n = 0.0, 0
        # Run validation in FP32 even when training with AMP.
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError('Nonfinite Stage2 validation logits')
                loss = criterion(logits, y)
                val_loss_sum += float(loss.item())*len(y)
                val_n += len(y)
                labels.extend(y.cpu().tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
        sched.step()
        score = macro_f1(labels, predictions, num_classes=2)
        row = {'epoch': epoch, 'train_loss': loss_sum/seen,
               'val_loss': val_loss_sum/val_n, 'binary_val_macro_f1': score}
        history.append(row)
        write_csv(out/'history_binary.csv', list(row), history)
        print(f'EPOCH {epoch:03d} | train_loss={row["train_loss"]:.5f} '
              f'val_loss={row["val_loss"]:.5f} '
              f'binary_val_macro_f1={score:.5f} | best={max(best_f1,score):.5f}')
        if score > best_f1 + 1e-6:
            best_f1, stale = score, 0
            torch.save({'model_state_dict': model.state_dict(),
                        'classes': CLASSES_2, 'epoch': epoch,
                        'val_macro_f1': score,
                        'loss': args.loss,
                        'init_stage1_checkpoint': None if args.from_imagenet else str(args.stage1_ckpt)},
                       ckpt_path)
        else:
            stale += 1
        if stale >= args.patience:
            print(f'Early stopping: no improvement for {args.patience} epochs')
            break
    print(f'Stage2 training complete: best binary val F1={best_f1:.6f}; {ckpt_path}')
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return ckpt_path


def write_submission(path, names, labels):
    labels = np.asarray(labels)
    if len(names) != 1182 or labels.shape != (1182,):
        raise ValueError('Prediction count does not match sample size 1182')
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Output has invalid class labels')
    rows = [{'new_filename': name,
             **{c: int(i == int(label)) for i, c in enumerate(CLASSES)}}
            for name, label in zip(names, labels)]
    if [r['new_filename'] for r in rows] != names:
        raise RuntimeError('Sample order changed unexpectedly')
    write_csv(path, FIELDS, rows)
    return rows


def run(args):
    seed_everything(args.seed)
    data = Path(args.data_root)
    out = Path(args.output_dir)
    train_csv, val_csv = data/'train_data.csv', data/'val_data.csv'
    train_base = CSVDataset(data/'train_images', train_csv, train=True)
    val_base = CSVDataset(data/'val_images', val_csv, train=False)
    y_train = np.asarray([label for _, label in train_base.records])
    y_val = np.asarray([label for _, label in val_base.records])
    if not set(y_train).issubset(set(range(4))) or not set(y_val).issubset(set(range(4))):
        raise ValueError('Unexpected CSV label encoding')
    if set(y_train) != set(range(4)) or set(y_val) != set(range(4)):
        raise ValueError('All four classes are required in both train and validation')
    if len(train_base) != 4017 or len(val_base) != 709:
        print(f'WARNING: expected 4017/709 but got {len(train_base)}/{len(val_base)}')
    print('Train class counts:', {CLASSES[i]: int((y_train == i).sum()) for i in range(4)})
    print('Validation class counts:', {CLASSES[i]: int((y_val == i).sum()) for i in range(4)})
    if args.check_data_only:
        print('DATA CHECK PASSED. No training or inference performed.')
        return
    if not args.stage1_ckpt.is_file():
        raise FileNotFoundError(f'Stage1 checkpoint missing: {args.stage1_ckpt}')
    names = sample_names(args.sample_csv)
    test_ds = TestImages(data/'test_images', names)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, '| Stage2 training FP32:', not args.amp)
    ckpt_path = out/'best_binary_model.pt'
    if not args.eval_only:
        ckpt_path = train_stage2(args, train_base, val_base, device, out)
    elif not ckpt_path.is_file():
        raise FileNotFoundError(f'--eval-only requires existing {ckpt_path}')

    val_loader = DataLoader(val_base, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type=='cuda')
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=device.type=='cuda')
    stage1 = load_stage1(args.stage1_ckpt, device)
    p4_val = predict_probs(stage1, val_loader, device, 4, 'Stage1 validation')
    p4_test = predict_probs(stage1, test_loader, device, 4, 'Stage1 test')
    del stage1
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    stage2 = load_stage2(ckpt_path, device)
    p2_val = predict_probs(stage2, val_loader, device, 2, 'Stage2 validation')
    p2_test = predict_probs(stage2, test_loader, device, 2, 'Stage2 test')
    del stage2
    print('Completed FP32 inference (no TTA) on val/test for both models.')

    baseline = metrics_4(y_val, p4_val.argmax(1))
    candidates = []
    for alpha in ALPHAS:
        pred = fuse(p4_val, p2_val, alpha).argmax(1)
        candidates.append({'alpha': alpha, 'macro_f1': macro_f1(y_val, pred),
                           'virus_f1': metrics_4(y_val, pred)['per_class']['virus']['f1'],
                           'bacteria_f1': metrics_4(y_val, pred)['per_class']['bacteria']['f1']})
    write_csv(out/'fusion_candidates.csv', list(candidates[0]), candidates)
    best_alpha = choose_alpha(p4_val, p2_val, y_val, np.arange(len(y_val)))
    best_score = alpha_score(p4_val, p2_val, y_val, best_alpha)
    fixed_alpha = 0.5
    fixed_score = alpha_score(p4_val, p2_val, y_val, fixed_alpha)
    crossfit = crossfit_fusion(p4_val, p2_val, y_val, args.seed)
    write_csv(out/'crossfit_folds.csv', list(crossfit['folds'][0]), crossfit['folds'])

    # Guard against overstating tiny same-validation tuning gains.
    robust = bool(crossfit['gain'] >= 0.005 and crossfit['improved_folds'] >= 3
                  and best_score > baseline['macro_f1'] + 0.005)
    chosen = float(best_alpha if robust else fixed_alpha)
    if args.alpha is not None:
        chosen = float(args.alpha)
        selection = 'user_fixed_alpha'
    else:
        selection = 'crossfit_supported_val_best' if robust else 'diagnostic_fixed_0.5'
    chosen_pred = fuse(p4_val, p2_val, chosen).argmax(1)
    chosen_metrics = metrics_4(y_val, chosen_pred)
    improvement = chosen_metrics['macro_f1']-baseline['macro_f1']
    results = {
        'baseline_stage1': baseline,
        'alpha_0_5_val_macro_f1': fixed_score,
        'best_full_val_alpha': best_alpha,
        'best_full_val_macro_f1_optimistic': best_score,
        'crossfit_alpha': crossfit,
        'selected_alpha': chosen,
        'selected_alpha_reason': selection,
        'selected_alpha_val_metrics': chosen_metrics,
        'selected_minus_baseline_val_macro_f1': improvement,
        'robust_alpha_gain_flag': robust,
        'cautions': [
            'Validation was already used for Stage1 and Stage2 checkpoint selection.',
            'Alpha crossfit is not an external unbiased test of model performance.',
            'Normal and COVID-19 probability masses remain unchanged under fusion.',
            'No Kaggle submission was performed.'
        ]
    }
    write_json(out/'exp6_validation_metrics.json', results)
    write_csv(out/'val_predictions_exp6.csv',
              ['filename','true_class','stage1_pred','fused_pred','prob_stage1_bacteria',
               'prob_stage1_virus','prob_binary_bacteria','prob_binary_virus'],
              [{'filename': Path(path).name, 'true_class': CLASSES[int(y)],
                'stage1_pred': CLASSES[int(p4_val[i].argmax())],
                'fused_pred': CLASSES[int(chosen_pred[i])],
                'prob_stage1_bacteria': float(p4_val[i,1]),
                'prob_stage1_virus': float(p4_val[i,2]),
                'prob_binary_bacteria': float(p2_val[i,0]),
                'prob_binary_virus': float(p2_val[i,1])}
               for i, ((path, _), y) in enumerate(zip(val_base.records, y_val))])

    p4_fused_test = fuse(p4_test, p2_test, chosen)
    test_pred = p4_fused_test.argmax(1)
    output_csv = out/'submission_exp6.csv'
    write_submission(output_csv, names, test_pred)
    write_csv(out/'test_probabilities_exp6.csv',
              ['new_filename']+['prob_'+c for c in CLASSES],
              [{'new_filename': n, **{'prob_'+c: float(p4_fused_test[i,j])
                                     for j,c in enumerate(CLASSES)}}
               for i,n in enumerate(names)])
    print('\n=== EXP 6 SUMMARY ===')
    print(f'Stage1 baseline validation Macro-F1:  {baseline["macro_f1"]:.6f}')
    print(f'Fixed alpha=0.50 validation Macro-F1: {fixed_score:.6f}')
    print(f'Best tuned val alpha: {best_alpha:.2f}, score={best_score:.6f} (optimistic)')
    print(f'Crossfit alpha Macro-F1: {crossfit["crossfit_macro_f1"]:.6f}, '
          f'gain={crossfit["gain"]:+.6f}, improved folds={crossfit["improved_folds"]}/5')
    print(f'Selected alpha={chosen:.2f} ({selection}); val Macro-F1={chosen_metrics["macro_f1"]:.6f}')
    print('Per-class F1:', {k: round(v['f1'], 4) for k,v in chosen_metrics['per_class'].items()})
    print(f'Output candidate (NOT uploaded): {output_csv} (1182 rows in sample order)')
    if not robust:
        print('ASSESSMENT: No strong support for tuning gain. Treat submission as a diagnostic candidate.')
    else:
        print('ASSESSMENT: Validation-only improvement; Kaggle gain is not guaranteed.')


def main():
    p = argparse.ArgumentParser(description='Lab3 Exp6 two-stage Bacteria/Virus specialist (Macro-F1)')
    p.add_argument('--data-root', type=Path, required=True,
                   help='Folder containing train_images, train_data.csv, val_images, val_data.csv, test_images')
    p.add_argument('--stage1-ckpt', type=Path, required=True,
                   help='Exp3 four-class checkpoint, e.g. runs/exp3_focal/best_model.pt')
    p.add_argument('--sample-csv', type=Path, default=Path('test_data_sample.csv'))
    p.add_argument('--output-dir', default='runs/exp6_two_stage')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--loss', choices=['ce','weighted_ce','focal'], default='focal')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--amp', action='store_true', help='Optional AMP TRAINING only; FP32 inference always')
    p.add_argument('--from-imagenet', action='store_true',
                   help='Use ImageNet backbone instead of warm-starting Stage2 from Stage1')
    p.add_argument('--check-data-only', action='store_true', help='Check train/val CSV and image paths only')
    p.add_argument('--eval-only', action='store_true', help='Skip training and use existing best_binary_model.pt')
    p.add_argument('--alpha', type=float, default=None,
                   help='Fix fusion alpha in [0,1]; default chooses via validation checks')
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.num_workers < 0:
        p.error('epochs, patience and batch-size must be positive; num-workers >= 0')
    if args.alpha is not None and not (0 <= args.alpha <= 1):
        p.error('--alpha must lie between 0 and 1')
    run(args)


if __name__ == '__main__':
    main()
