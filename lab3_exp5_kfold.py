
import argparse
import csv
import hashlib
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
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from lab3_efficientnet_b3_csv_v2 import (
    CLASSES, CSVDataset, FocalLoss, build_model, create_transform
)

FILENAME_CANDIDATES = CSVDataset.FILENAME_COLUMNS
GROUP_CANDIDATES = (
    'patient_id', 'patientid', 'patient', 'subject_id', 'subjectid', 'subject',
    'person_id', 'case_id', 'caseid', 'participant_id', 'participantid',
    'study_patient_id', 'patient_no', 'patient_number'
)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def save_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_group_column(csv_paths, requested):
    """Only infer patient grouping when an explicit ID column is present in BOTH CSVs."""
    header_maps = []
    for p in csv_paths:
        with Path(p).open(newline='', encoding='utf-8-sig') as f:
            fields = csv.DictReader(f).fieldnames or []
        header_maps.append({c.casefold().strip(): c for c in fields})
    if requested.lower() == 'none':
        return None, [None, None]
    if requested != 'auto':
        key = requested.casefold().strip()
        if not all(key in h for h in header_maps):
            raise ValueError(f'--group-column {requested!r} must exist in BOTH CSV files; '
                             f'headers={header_maps}')
        return requested, [h[key] for h in header_maps]
    for candidate in GROUP_CANDIDATES:
        if all(candidate in h for h in header_maps):
            return candidate, [h[candidate] for h in header_maps]
    return None, [None, None]


def read_groups(csv_path, group_col):
    if not group_col:
        return None
    values = []
    with Path(csv_path).open(newline='', encoding='utf-8-sig') as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            value = (row.get(group_col) or '').strip()
            if not value:
                raise ValueError(f'Empty patient ID in {csv_path}:{line}. '
                                 'Specify --group-column none only if you accept image-level splits.')
            values.append(value)
    return values


class PooledDataset(Dataset):
    """Paths/labels from train+val, with train or evaluation transform."""
    def __init__(self, records, train=False):
        self.records = records
        self.transform = create_transform(train)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        with Image.open(path) as im:
            x = self.transform(im.convert('RGB'))
        return x, int(label)


class TestDataset(Dataset):
    def __init__(self, test_dir, names):
        self.paths = [Path(test_dir) / name for name in names]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'Missing {len(missing)} test images: {missing[:5]}')
        self.transform = create_transform(False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert('RGB'))


def sample_filenames(sample_csv):
    with Path(sample_csv).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        expected = ['new_filename'] + CLASSES
        if reader.fieldnames != expected:
            raise ValueError(f'Sample header {reader.fieldnames}; expected {expected}')
        names = [(r['new_filename'] or '').strip() for r in reader]
    if len(names) != 1182 or any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Expected 1182 nonempty, unique sample filenames in sample CSV')
    return names


def full_metrics(y_true, y_pred):
    macro = float(f1_score(y_true, y_pred, average='macro', labels=list(range(4)), zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )
    return {
        'macro_f1': macro,
        'per_class': {
            c: {'precision': float(precision[i]), 'recall': float(recall[i]),
                'f1': float(f1[i]), 'support': int(support[i])}
            for i, c in enumerate(CLASSES)
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()
    }


def fingerprint(records, groups, splits, args):
    """Protect against silently using checkpoints from a different fold assignment."""
    obj = {
        'records': [(str(path.resolve()), int(y), str(groups[i]) if groups is not None else '')
                    for i, (path, y) in enumerate(records)],
        'splits': [[tr.tolist(), va.tolist()] for tr, va in splits],
        'classes': CLASSES,
        'folds': args.folds,
        'seed': args.seed,
        'loss': args.loss,
        'epochs': args.epochs,
        'patience': args.patience,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'accum_steps': args.accum_steps,
        'weight_decay': args.weight_decay,
        'pretrained': not args.no_pretrained,
    }
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode('utf-8')).hexdigest()


def prepare_data(args):
    root = Path(args.data_root)
    train_csv = root / 'train_data.csv'
    val_csv = root / 'val_data.csv'
    ds_train = CSVDataset(root / 'train_images', train_csv, train=False)
    ds_val = CSVDataset(root / 'val_images', val_csv, train=False)
    records = ds_train.records + ds_val.records
    labels = np.array([int(label) for _, label in records], dtype=np.int64)
    group_name, cols = parse_group_column([train_csv, val_csv], args.group_column)
    groups = None
    if group_name:
        groups = np.asarray(read_groups(train_csv, cols[0]) + read_groups(val_csv, cols[1]))
        if len(groups) != len(records):
            raise ValueError('Patient ID and image row counts mismatch')
        print(f'Patient grouping enabled using column {group_name!r}; '
              f'{len(set(groups))} distinct groups')
    else:
        print('WARNING: No patient-ID group column found. Using IMAGE-level StratifiedKFold.')
        print('WARNING: If multiple images belong to one patient, CV can be overly optimistic.')
    if args.folds < 2:
        raise ValueError('--folds must be >= 2')
    counts = Counter(labels.tolist())
    print('Combined data:', len(records), 'images; class counts =',
          {CLASSES[i]: counts[i] for i in range(4)})
    if min(counts[i] for i in range(4)) < args.folds:
        raise ValueError('Not enough examples of each class for the selected number of folds')
    # Duplicate paths should not appear twice in the combined training pool.
    resolved = [str(p.resolve()).casefold() for p, _ in records]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Duplicate image paths between train/val CSVs; fix before K-fold')
    if groups is None:
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(labels)), labels))
    else:
        if StratifiedGroupKFold is None:
            raise RuntimeError('Group folds require scikit-learn>=1.1')
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(splitter.split(np.zeros(len(labels)), labels, groups=groups))
    for k, (train_idx, val_idx) in enumerate(splits, start=1):
        counts_fold = np.bincount(labels[val_idx], minlength=4)
        if any(c == 0 for c in counts_fold):
            raise RuntimeError(f'Fold {k} has a missing class: {counts_fold.tolist()}. '
                               'Try fewer folds / inspect patient grouping.')
        if groups is not None and set(groups[train_idx]).intersection(groups[val_idx]):
            raise AssertionError('Patient leakage detected in fold split')
        print(f'Fold {k}: train={len(train_idx)}, val={len(val_idx)}, '
              f'val counts={dict(zip(CLASSES, map(int, counts_fold)))}')
    split_id = fingerprint(records, groups, splits, args)
    return records, labels, groups, splits, split_id, group_name


def make_criterion(labels, train_idx, args, device):
    counts = np.bincount(labels[train_idx], minlength=4)
    inv_sqrt = torch.tensor(1.0 / np.sqrt(counts), dtype=torch.float32, device=device)
    alpha = inv_sqrt / inv_sqrt.mean()
    if args.loss == 'focal':
        return FocalLoss(gamma=args.gamma, alpha=alpha)
    if args.loss == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=alpha)
    return nn.CrossEntropyLoss()


def get_amp(device):
    enabled = device.type == 'cuda'
    # PyTorch 2.4 (the user's previous environment) supports these APIs.
    scaler = torch.amp.GradScaler('cuda', enabled=enabled)
    return enabled, scaler


@torch.inference_mode()
def infer_probs(model, loader, device):
    model.eval()
    chunks = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            logits = model(x)
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if not chunks:
        raise ValueError('Empty prediction loader')
    return np.concatenate(chunks, axis=0)


def best_model_from_file(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if list(ckpt.get('classes', [])) != CLASSES:
        raise RuntimeError(f'Classes mismatch in checkpoint: {path}')
    model = build_model(False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device)
    return model, ckpt


def run_fold(args, fold, indices, records, labels, split_id, device):
    train_idx, val_idx = indices
    root = Path(args.output_dir)
    fold_dir = root / f'fold_{fold}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    last_path = fold_dir / 'last_checkpoint.pt'
    best_path = fold_dir / 'best_model.pt'
    complete_path = fold_dir / 'fold_metrics.json'
    prediction_path = fold_dir / 'val_predictions.npz'

    if complete_path.exists() and best_path.exists() and prediction_path.exists():
        saved = json.loads(complete_path.read_text(encoding='utf-8'))
        if saved.get('split_fingerprint') != split_id:
            raise RuntimeError(f'Fold {fold} split/config changed. Use a different --output-dir.')
        print(f'[Fold {fold}] completed; reusing checkpoints and validation probabilities')
        return saved
    if last_path.exists() and not args.resume:
        raise RuntimeError(f'Fold {fold} has incomplete run in {fold_dir}. '
                           'Rerun with --resume to continue, or choose a fresh output folder.')
    if best_path.exists() and not last_path.exists() and not complete_path.exists():
        raise RuntimeError(f'Fold {fold} has orphan best_model.pt. Do not overwrite; inspect folder.')

    train_data = PooledDataset(records, train=True)
    val_data = PooledDataset(records, train=False)
    train_loader = DataLoader(Subset(train_data, train_idx.tolist()),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(Subset(val_data, val_idx.tolist()),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    counts = np.bincount(labels[train_idx], minlength=4)
    criterion = make_criterion(labels, train_idx, args, device)
    model = build_model(not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled, scaler = get_amp(device)
    start_epoch, history, best_score, best_epoch, stale = 1, [], -1.0, 0, 0

    if last_path.exists() and args.resume:
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state.get('split_fingerprint') != split_id:
            raise RuntimeError('Stored fold split/config is different: cannot resume safely')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        scaler.load_state_dict(state['scaler_state_dict'])
        start_epoch = state['epoch'] + 1
        history = state['history']
        best_score, best_epoch, stale = state['best_score'], state['best_epoch'], state['stale']
        print(f'[Fold {fold}] resuming from epoch {start_epoch}; previous best={best_score:.5f}')
    else:
        print(f'[Fold {fold}] starting training (class counts: {counts.tolist()})')

    for epoch in range(start_epoch, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        n_train = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # Normalize by the actual number of micro-batches in this update,
            # including the final partial accumulation window.
            window_start = ((step - 1) // args.accum_steps) * args.accum_steps
            window_size = min(args.accum_steps, len(train_loader) - window_start)
            with torch.autocast(device_type='cuda', enabled=amp_enabled):
                logits = model(images)
                raw_loss = criterion(logits, targets)
                loss = raw_loss / window_size
            scaler.scale(loss).backward()
            if step % args.accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += float(raw_loss.detach().item()) * len(targets)
            n_train += len(targets)

        probs = infer_probs(model, val_loader, device)
        val_preds = probs.argmax(axis=1)
        val_macro_f1 = float(f1_score(labels[val_idx], val_preds,
                                      labels=list(range(4)), average='macro', zero_division=0))
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': round(train_loss_sum / max(1, n_train), 6),
               'val_macro_f1': round(val_macro_f1, 8)}
        history.append(row)
        save_csv(fold_dir / 'history.csv', list(row), history)
        print(f'Fold {fold}/{args.folds} | epoch {epoch:03d}/{args.epochs} '
              f'| loss={row["train_loss"]:.5f} | val Macro-F1={val_macro_f1:.5f} '
              f'| best={max(best_score, val_macro_f1):.5f}', flush=True)
        if val_macro_f1 > best_score + 1e-8:
            best_score, best_epoch, stale = val_macro_f1, epoch, 0
            torch.save({
                'model_state_dict': model.state_dict(), 'classes': CLASSES,
                'epoch': epoch, 'val_macro_f1': best_score, 'fold': fold,
                'split_fingerprint': split_id, 'loss': args.loss,
            }, best_path)
        else:
            stale += 1
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch, 'history': history, 'best_score': best_score,
            'best_epoch': best_epoch, 'stale': stale,
            'split_fingerprint': split_id,
        }, last_path)
        if stale >= args.patience:
            print(f'[Fold {fold}] early stop; best epoch={best_epoch}, '
                  f'best Macro-F1={best_score:.5f}')
            break
    if not best_path.exists():
        raise RuntimeError(f'Fold {fold} has no best checkpoint; check training log')

    del model, optimizer, scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    best_model, _ = best_model_from_file(best_path, device)
    probs = infer_probs(best_model, val_loader, device)
    predictions = probs.argmax(axis=1)
    metrics = full_metrics(labels[val_idx], predictions)
    if abs(metrics['macro_f1'] - best_score) > 1e-4:
        print(f'WARNING fold {fold}: best macro-F1 during training {best_score:.5f} '
              f'vs checkpoint repeat {metrics["macro_f1"]:.5f}')
    np.savez_compressed(prediction_path, indices=val_idx, labels=labels[val_idx], probs=probs)
    info = {
        'fold': fold, 'best_epoch': best_epoch, 'best_val_macro_f1': metrics['macro_f1'],
        'validation_size': len(val_idx), 'split_fingerprint': split_id,
        'details': metrics,
    }
    save_json(complete_path, info)
    del best_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return info


def produce_results(args, records, labels, groups, splits, split_id, device):
    out = Path(args.output_dir)
    n = len(records)
    oof_probs = np.zeros((n, len(CLASSES)), dtype=np.float32)
    seen = np.zeros(n, dtype=np.int64)
    summaries = []
    for fold, (_, val_idx) in enumerate(splits, start=1):
        folder = out / f'fold_{fold}'
        info_path = folder / 'fold_metrics.json'
        probs_path = folder / 'val_predictions.npz'
        ckpt_path = folder / 'best_model.pt'
        if not (info_path.is_file() and probs_path.is_file() and ckpt_path.is_file()):
            print(f'Fold {fold} is not complete; no Kaggle submission produced.')
            return False
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if info.get('split_fingerprint') != split_id:
            raise RuntimeError('Output folder contains mismatched fold config')
        with np.load(probs_path) as npz:
            idx = npz['indices'].astype(np.int64)
            probs = npz['probs'].astype(np.float32)
            truth = npz['labels'].astype(np.int64)
        if not np.array_equal(idx, val_idx) or not np.array_equal(truth, labels[val_idx]):
            raise RuntimeError(f'Fold {fold} OOF indices/labels mismatch')
        oof_probs[idx] = probs
        seen[idx] += 1
        summaries.append({'fold': fold, 'best_epoch': info['best_epoch'],
                          'val_count': info['validation_size'],
                          'val_macro_f1': round(info['best_val_macro_f1'], 8)})
    if np.any(seen != 1):
        raise AssertionError('OOF must predict each training image exactly once')
    oof_preds = oof_probs.argmax(axis=1)
    oof_stats = full_metrics(labels, oof_preds)
    scores = [r['val_macro_f1'] for r in summaries]
    oof_stats.update({
        'note': 'OOF selected by best epoch within each fold. Prior Exp2/3 validation tuning may influence this estimate.',
        'fold_macro_f1_mean': float(np.mean(scores)),
        'fold_macro_f1_std': float(np.std(scores)),
        'folds': summaries,
        'split_fingerprint': split_id,
        'group_aware': groups is not None,
    })
    save_csv(out / 'fold_summary.csv', ['fold', 'best_epoch', 'val_count', 'val_macro_f1'], summaries)
    save_json(out / 'oof_metrics.json', oof_stats)
    oof_rows = []
    fold_ids = np.zeros(n, dtype=np.int64)
    for j, (_, val_idx) in enumerate(splits, start=1):
        fold_ids[val_idx] = j
    for i, ((path, _), yi, pi) in enumerate(zip(records, labels, oof_probs)):
        fold = int(fold_ids[i])
        oof_rows.append({
            'filename': path.name, 'true_class': CLASSES[int(yi)],
            'pred_class': CLASSES[int(np.argmax(pi))], 'fold': fold,
            'correct': int(int(yi) == int(np.argmax(pi))),
            **{f'prob_{c}': round(float(pi[k]), 7) for k, c in enumerate(CLASSES)}
        })
    fields = ['filename', 'true_class', 'pred_class', 'fold', 'correct'] + [f'prob_{c}' for c in CLASSES]
    save_csv(out / 'oof_predictions.csv', fields, oof_rows)
    print(f'OOF Macro-F1: {oof_stats["macro_f1"]:.5f} | '
          f'fold mean: {oof_stats["fold_macro_f1_mean"]:.5f}')
    print('OOF per-class F1:', {k: round(v['f1'], 4)
                                for k, v in oof_stats['per_class'].items()})

    names = sample_filenames(args.sample_csv)
    test_ds = TestDataset(Path(args.data_root) / 'test_images', names)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    test_sum = np.zeros((len(names), len(CLASSES)), dtype=np.float64)
    for fold in range(1, args.folds + 1):
        path = out / f'fold_{fold}' / 'best_model.pt'
        model, ckpt = best_model_from_file(path, device)
        if ckpt.get('split_fingerprint') != split_id:
            raise RuntimeError(f'Fold {fold} test inference checkpoint mismatch')
        probs = infer_probs(model, test_loader, device)
        if probs.shape != test_sum.shape:
            raise RuntimeError('Unexpected test output shape')
        test_sum += probs
        print(f'Test inference: fold {fold}/{args.folds} done', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    avg = test_sum / args.folds
    pred = avg.argmax(axis=1)
    rows = []
    prob_rows = []
    for i, name in enumerate(names):
        rows.append({'new_filename': name, **{c: int(j == pred[i]) for j, c in enumerate(CLASSES)}})
        prob_rows.append({'new_filename': name,
                          **{f'prob_{c}': round(float(avg[i, j]), 8) for j, c in enumerate(CLASSES)}})
    if len(rows) != 1182 or [r['new_filename'] for r in rows] != names:
        raise AssertionError('Submission file row order incorrect')
    if any(sum(row[c] for c in CLASSES) != 1 for row in rows):
        raise AssertionError('Submission requires one-hot predictions')
    save_csv(out / 'submission_exp5.csv', ['new_filename'] + CLASSES, rows)
    save_csv(out / 'test_probabilities_exp5.csv', ['new_filename'] + [f'prob_{c}' for c in CLASSES], prob_rows)
    print(f'Wrote {out / "submission_exp5.csv"} (1182 rows, correct order).')
    print('Do not upload test_probabilities_exp5.csv. No Kaggle submission was performed.')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', required=True, help='Kaggle extracted directory with train_images, val_images and CSVs')
    parser.add_argument('--sample-csv', default='test_data_sample.csv', help='Original sample CSV for test order')
    parser.add_argument('--output-dir', default='runs/exp5_kfold')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--accum-steps', type=int, default=2, help='Gradient accumulation; effective batch ~ batch-size*accum-steps')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--loss', choices=['focal', 'weighted_ce', 'ce'], default='focal')
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--group-column', default='auto', help='auto, none, or exact patient-ID CSV header')
    parser.add_argument('--only-fold', type=int, default=0, help='Train just one fold; no Kaggle CSV until all 5 exist')
    parser.add_argument('--resume', action='store_true', help='Resume unfinished fold checkpoint safely')
    parser.add_argument('--predict-only', action='store_true', help='Re-create OOF + submission from completed folds')
    parser.add_argument('--check-data-only', action='store_true', help='Validate CSVs and fold distributions; do not train')
    parser.add_argument('--no-pretrained', action='store_true', help='For offline smoke testing only; performance usually worse')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.patience < 1:
        parser.error('epochs, batch-size, accum-steps and patience must all be >= 1')
    if args.only_fold and not (1 <= args.only_fold <= args.folds):
        parser.error('--only-fold must be between 1 and --folds')
    if args.check_data_only and args.predict_only:
        parser.error('Cannot specify both --check-data-only and --predict-only')
    set_seed(args.seed)
    records, labels, groups, splits, split_id, group_name = prepare_data(args)
    if args.check_data_only:
        print('DATA CHECK PASSED. No weights downloaded, no model training, no CSV submission.')
        return
    if not Path(args.sample_csv).is_file():
        raise FileNotFoundError(f'Sample CSV not found: {args.sample_csv}')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / 'run_config.json'
    config = vars(args).copy()
    config.update({'split_fingerprint': split_id, 'n_pool': len(records),
                   'class_counts': dict(Counter(CLASSES[int(y)] for y in labels)),
                   'group_column_used': group_name, 'torch': torch.__version__})
    if config_path.is_file():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old.get('split_fingerprint') != split_id:
            raise RuntimeError('The output folder already contains a different fold/data/config split. '
                               'Use a new --output-dir instead.')
    else:
        save_json(config_path, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; class order: {CLASSES}; split fingerprint: {split_id[:12]}')
    if not args.predict_only:
        for fold, split in enumerate(splits, start=1):
            if args.only_fold and args.only_fold != fold:
                continue
            set_seed(args.seed + fold)
            run_fold(args, fold, split, records, labels, split_id, device)
    produce_results(args, records, labels, groups, splits, split_id, device)


if __name__ == '__main__':
    main()
