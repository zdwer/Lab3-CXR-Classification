
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab3_exp5_kfold import CLASSES, TestDataset, best_model_from_file, sample_filenames

P_FIELDS = [f'prob_{c}' for c in CLASSES]
FIELDS = ['new_filename'] + P_FIELDS


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'Not found: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames != FIELDS:
            raise ValueError(f'Unexpected headers for {path}: {rdr.fieldnames}; expected {FIELDS}')
        return list(rdr)


def audit_existing(rows, filenames):
    if len(rows) != len(filenames):
        raise ValueError(f'Test probability rows={len(rows)}, sample rows={len(filenames)}; expected equal')
    if [r['new_filename'].strip() for r in rows] != filenames:
        raise ValueError('Test probability filenames/order differ from sample CSV; stopping.')
    invalid = []
    prob_array = np.empty((len(rows), len(CLASSES)), dtype=np.float64)
    for i, row in enumerate(rows):
        bad_fields = []
        for k, field in enumerate(P_FIELDS):
            raw = row.get(field)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = np.nan
            prob_array[i, k] = val
            if not np.isfinite(val) or not (0.0 <= val <= 1.00001):
                bad_fields.append(f'{field}={raw!r}')
        if not bad_fields and abs(float(np.sum(prob_array[i])) - 1.0) > 1e-3:
            bad_fields.append(f'row_sum={np.sum(prob_array[i]):.7g}')
        if bad_fields:
            invalid.append({'csv_line': i + 2, 'filename': filenames[i], 'issues': bad_fields})
    return invalid, prob_array


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def infer_fold_fp32(model, loader, device, names, fold):
    model.eval()
    all_chunks = []
    offset = 0
    for images in loader:
        if isinstance(images, (tuple, list)):
            images = images[0]
        # No CUDA autocast: use full float32 to avoid possible FP16 overflow.
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        if not torch.isfinite(images).all():
            bad = torch.where(~torch.isfinite(images).flatten(1).all(dim=1))[0].tolist()
            raise RuntimeError(f'Fold {fold}: input images contain NaN/inf, filenames={[names[offset+i] for i in bad]}')
        with torch.autocast(device_type='cuda', enabled=False):
            logits = model(images)
        ok = torch.isfinite(logits).all(dim=1)
        if not bool(ok.all()):
            bad = torch.where(~ok)[0].tolist()
            raise RuntimeError(f'Fold {fold}: FP32 logits remain invalid; check images/model for {[names[offset+i] for i in bad]}')
        probs = torch.softmax(logits.float(), dim=1)
        if not bool(torch.isfinite(probs).all()):
            raise RuntimeError(f'Fold {fold}: invalid softmax probabilities near row {offset+2}')
        all_chunks.append(probs.cpu().numpy().astype(np.float64))
        offset += images.shape[0]
    return np.concatenate(all_chunks, axis=0)


def main():
    ap = argparse.ArgumentParser(description='Diagnose and repair Exp5 test probabilities via five-fold FP32 inference; no training')
    ap.add_argument('--data-root', required=True, help='Extracted Kaggle dataset root')
    ap.add_argument('--sample-csv', default='test_data_sample.csv')
    ap.add_argument('--exp5-dir', default='runs/exp5_kfold')
    ap.add_argument('--output-dir', default='runs/exp5_kfold/fp32_recheck')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--audit-only', action='store_true', help='Show which CSV rows have invalid probabilities; do not run inference')
    args = ap.parse_args()
    if args.batch_size < 1 or args.folds < 1:
        ap.error('--batch-size and --folds must be >= 1')

    names = sample_filenames(args.sample_csv)
    exp5 = Path(args.exp5_dir)
    source = exp5 / 'test_probabilities_exp5.csv'
    rows = read_csv(source)
    invalid, original_probs = audit_existing(rows, names)
    print(f'Existing test probability CSV: {len(rows)} rows; invalid rows={len(invalid)}')
    for entry in invalid[:20]:
        print(f"  line={entry['csv_line']}, filename={entry['filename']}, issues={', '.join(entry['issues'])}")
    if len(invalid) > 20:
        print(f'  ... {len(invalid)-20} additional invalid rows')
    if args.audit_only:
        print('AUDIT ONLY: no model loaded, no files modified.')
        return

    test_dir = Path(args.data_root) / 'test_images'
    ds = TestDataset(test_dir, names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    print(f'Recomputing all {len(ds)} test images in full FP32 on {device}; no training.')

    total = np.zeros((len(names), len(CLASSES)), dtype=np.float64)
    for fold in range(1, args.folds + 1):
        checkpoint = exp5 / f'fold_{fold}' / 'best_model.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing fold checkpoint: {checkpoint}')
        model, ckpt = best_model_from_file(checkpoint, device)
        if int(ckpt.get('fold', -1)) != fold:
            raise ValueError(f'Fold metadata mismatch in {checkpoint}')
        fold_probs = infer_fold_fp32(model, loader, device, names, fold)
        if fold_probs.shape != total.shape:
            raise ValueError(f'Fold {fold} returned shape {fold_probs.shape}, expected {total.shape}')
        total += fold_probs
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f'FP32 inference completed: fold {fold}/{args.folds}', flush=True)

    avg = total / args.folds
    if not np.isfinite(avg).all() or np.any(avg < 0.0) or np.any(avg > 1.00001):
        raise RuntimeError('Recomputed probabilities are invalid; no CSV output written.')
    if not np.allclose(avg.sum(axis=1), 1.0, atol=1e-4, rtol=0):
        raise RuntimeError('Recomputed probabilities do not sum to 1; no CSV output written.')

    preds = np.argmax(avg, axis=1)
    fp32_prob_rows = []
    submission_rows = []
    for i, name in enumerate(names):
        fp32_prob_rows.append({'new_filename': name,
            **{f'prob_{c}': f'{avg[i,j]:.10f}' for j, c in enumerate(CLASSES)}})
        submission_rows.append({'new_filename': name,
            **{c: int(preds[i] == j) for j, c in enumerate(CLASSES)}})

    valid_mask = np.ones(len(names), dtype=bool)
    valid_mask[[item['csv_line']-2 for item in invalid]] = False
    valid_comparison = valid_mask & np.isfinite(original_probs).all(axis=1)
    old_preds = np.argmax(original_probs[valid_comparison], axis=1)
    new_preds = preds[valid_comparison]
    changed_valid = int(np.sum(old_preds != new_preds))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path = out_dir / 'test_probabilities_exp5_fp32.csv'
    sub_path = out_dir / 'submission_exp5_fp32.csv'
    write_csv(probs_path, FIELDS, fp32_prob_rows)
    write_csv(sub_path, ['new_filename'] + CLASSES, submission_rows)
    report = {
        'original_csv': str(source),
        'original_invalid_row_count': len(invalid),
        'original_invalid_rows': invalid,
        'inference_precision': 'FP32, autocast disabled',
        'folds_used': args.folds,
        'rows_with_changed_class_among_old_valid_rows': changed_valid,
        'new_probabilities': str(probs_path),
        'new_submission': str(sub_path),
        'kaggle_submission_performed': False,
        'note': 'The original test probabilities and submission were not overwritten.'
    }
    (out_dir / 'fp32_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'New valid test probabilities: {probs_path}')
    print(f'New one-hot Kaggle CSV (NOT submitted): {sub_path}')
    print(f'Changed predicted classes among original valid rows: {changed_valid}')
    print('Original Exp5 files and trained checkpoints are unchanged.')


if __name__ == '__main__':
    main()
