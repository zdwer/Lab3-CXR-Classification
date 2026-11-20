
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']
P_FIELDS = [f'prob_{c}' for c in CLASSES]
LABELS = list(range(len(CLASSES)))
B = CLASSES.index('bacteria')
V = CLASSES.index('virus')


def read_table(path: Path, required):
    if not path.is_file():
        raise FileNotFoundError(f'Missing file: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        heads = rd.fieldnames or []
        absent = [s for s in required if s not in heads]
        if absent:
            raise ValueError(f'{path} is missing fields: {absent}; present: {heads}')
        data = list(rd)
    if not data:
        raise ValueError(f'No rows: {path}')
    return data


def write_csv(path: Path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def get_probs(rows, source):
    try:
        arr = np.asarray([[float(r[c]) for c in P_FIELDS] for r in rows], dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f'Missing/non-numeric probability in {source}: {e}') from e
    if not np.isfinite(arr).all():
        raise ValueError(f'NaN/inf probabilities in {source}')
    if (arr < 0).any() or (arr > 1.00001).any():
        raise ValueError(f'Probability outside [0, 1] in {source}')
    total = arr.sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-3, rtol=0):
        k = int(np.argmax(np.abs(total-1)))
        raise ValueError(f'Class probabilities must sum to approximately 1 in {source}; row {k+2} sum={total[k]:.6g}')
    return arr


def score(y_true, pred):
    return float(f1_score(y_true, pred, labels=LABELS, average='macro', zero_division=0))


def metrics(y_true, y_pred):
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        'macro_f1': score(y_true, y_pred),
        'per_class': {c: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(sup[i])} for i, c in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def adjusted_predict(probs, threshold):
    pred = probs.argmax(axis=1).astype(int)
    bv_mask = (pred == B) | (pred == V)
    pred[bv_mask] = np.where(probs[bv_mask, V] > (threshold * probs[bv_mask, B]), V, B)
    return pred


def choose_threshold(y_true, probs, candidates):
    scores = [(float(t), score(y_true, adjusted_predict(probs, t))) for t in candidates]
    best = max(sc for _, sc in scores)
    near = [t for t, sc in scores if best - sc < 1e-10]
    selected = min(near, key=lambda t: (abs(math.log(t)), t))
    return selected, best, scores


def main():
    ap = argparse.ArgumentParser(description='Exp5: Bacteria vs Virus threshold tuning with fold-held-out evaluation; no retraining')
    ap.add_argument('--oof-csv', default='runs/exp5_kfold/oof_predictions.csv')
    ap.add_argument('--test-csv', default='runs/exp5_kfold/test_probabilities_exp5.csv')
    ap.add_argument('--sample-csv', default='test_data_sample.csv')
    ap.add_argument('--exp5-metrics', default='runs/exp5_kfold/oof_metrics.json')
    ap.add_argument('--original-submission', default='runs/exp5_kfold/submission_exp5.csv')
    ap.add_argument('--output-dir', default='runs/exp5_kfold/threshold_review')
    ap.add_argument('--min-gain', type=float, default=0.003, help='Minimum held-out Macro-F1 gain to regard result as promising, not a guarantee')
    args = ap.parse_args()

    oof_path, test_path, sample_path = Path(args.oof_csv), Path(args.test_csv), Path(args.sample_csv)
    oof = read_table(oof_path, ['filename', 'true_class', 'fold'] + P_FIELDS)
    test = read_table(test_path, ['new_filename'] + P_FIELDS)
    sample = read_table(sample_path, ['new_filename'] + CLASSES)
    if (csv.field_size_limit() < 100):
        raise RuntimeError('Unexpected small CSV parser field size')
    if len(sample) != 1182 or len(test) != 1182:
        raise ValueError(f'Kaggle expects 1,182 test rows; sample={len(sample)}, probabilities={len(test)}')
    names = [r['new_filename'].strip() for r in sample]
    test_names = [r['new_filename'].strip() for r in test]
    if not all(names) or len(set(names)) != len(names) or names != test_names:
        raise ValueError('Sample/test filename mismatch, duplicate or incorrect row order; stop to avoid invalid Kaggle CSV')
    if len(oof) != 4726:
        print(f'WARNING: expected 4726 OOF rows based on original task, found {len(oof)}. Continuing only if each fold is valid.')

    y_raw = [r['true_class'].strip() for r in oof]
    if any(c not in CLASSES for c in y_raw):
        bad = sorted(set(y_raw) - set(CLASSES))
        raise ValueError(f'Unknown OOF labels: {bad}')
    y = np.asarray([CLASSES.index(c) for c in y_raw], dtype=np.int64)
    try:
        folds = np.asarray([int(r['fold']) for r in oof], dtype=np.int64)
    except ValueError as e:
        raise ValueError('OOF fold field must be numeric') from e
    fold_ids = sorted(set(folds.tolist()))
    if len(fold_ids) < 3:
        raise ValueError('Need >=3 folds to cross-fit a post-processing threshold')
    if any(not np.any(folds != k) for k in fold_ids):
        raise ValueError('Not enough training folds')

    oof_p = get_probs(oof, oof_path)
    test_p = get_probs(test, test_path)
    baseline = adjusted_predict(oof_p, 1.0)
    base_score = score(y, baseline)
    if Path(args.exp5_metrics).is_file():
        previous = json.loads(Path(args.exp5_metrics).read_text(encoding='utf-8'))
        previous_score = float(previous['macro_f1'])
        if abs(base_score - previous_score) > 0.0001:
            raise ValueError(f'OOF baseline {base_score:.7f} differs from Exp5 recorded {previous_score:.7f}; possible data mismatch')
    if Path(args.original_submission).is_file():
        orig = read_table(Path(args.original_submission), ['new_filename'] + CLASSES)
        if [r['new_filename'].strip() for r in orig] != names:
            raise ValueError('Original submission filename order mismatch')
        base_test = test_p.argmax(axis=1)
        for k, row in enumerate(orig):
            expected = [int(j == base_test[k]) for j in LABELS]
            try:
                actual = [int(row[c]) for c in CLASSES]
            except ValueError as e:
                raise ValueError(f'Original submission has invalid values in row {k+2}') from e
            if actual != expected:
                raise ValueError(f'Original submission differs from test probability argmax at row {k+2}; stop and investigate')

    candidates = np.round(np.arange(0.75, 1.25001, 0.025), 6)
    tuned_full_t, fitted_full_f1, curve = choose_threshold(y, oof_p, candidates)

    crossfit_pred = baseline.copy()
    crossfit_rows = []
    for k in fold_ids:
        train_mask = folds != k
        hold_mask = folds == k
        threshold, train_score, _ = choose_threshold(y[train_mask], oof_p[train_mask], candidates)
        fold_pred = adjusted_predict(oof_p[hold_mask], threshold)
        crossfit_pred[hold_mask] = fold_pred
        crossfit_rows.append({
            'fold': int(k), 'validation_rows': int(hold_mask.sum()),
            'trained_threshold_excluding_fold': threshold,
            'other_four_fold_tuning_macro_f1': round(train_score, 7),
            'heldout_baseline_macro_f1': round(score(y[hold_mask], baseline[hold_mask]), 7),
            'heldout_tuned_macro_f1': round(score(y[hold_mask], fold_pred), 7),
            'heldout_improvement': round(score(y[hold_mask], fold_pred) - score(y[hold_mask], baseline[hold_mask]), 7),
        })
    crossfit_score = score(y, crossfit_pred)
    delta = crossfit_score - base_score
    fold_wins = sum(r['heldout_improvement'] > 0 for r in crossfit_rows)

    promising = (delta >= args.min_gain and fold_wins >= 3 and fitted_full_f1 >= base_score)
    
    test_pred = adjusted_predict(test_p, tuned_full_t)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kaggle_rows = [
        {'new_filename': names[i], **{c: int(test_pred[i] == j) for j, c in enumerate(CLASSES)}}
        for i in range(len(names))
    ]
    if any(sum(row[c] for c in CLASSES) != 1 for row in kaggle_rows):
        raise AssertionError('One-hot output validation failed')
    submit_path = out / 'submission_exp5_threshold.csv'
    write_csv(submit_path, ['new_filename'] + CLASSES, kaggle_rows)
    write_csv(out / 'threshold_candidates.csv', ['virus_vs_bacteria_threshold', 'full_oof_macro_f1'],
              [{'virus_vs_bacteria_threshold': t, 'full_oof_macro_f1': round(s, 7)} for t, s in curve])
    write_csv(out / 'crossfit_fold_results.csv', list(crossfit_rows[0]), crossfit_rows)

    changes = int(np.sum(test_pred != test_p.argmax(axis=1)))
    result = {
        'baseline_oof': metrics(y, baseline),
        'crossfit_oof': metrics(y, crossfit_pred),
        'full_oof_tuned_apparent_score': fitted_full_f1,
        'full_oof_selected_threshold': tuned_full_t,
        'crossfit_vs_baseline_delta': delta,
        'folds_improved': fold_wins,
        'folds_total': len(fold_ids),
        'minimum_gain_required': args.min_gain,
        'candidate_appears_promising': promising,
        'test_predictions_changed_vs_exp5': changes,
        'fold_holdout_results': crossfit_rows,
        'caveats': [
            'Full OOF tuned Macro-F1 is optimistic because the same OOF labels chose the threshold.',
            'Cross-fitted post-processing helps but does not undo checkpoint selection and other model-development bias.',
            'The test file uses a five-model ensemble; the fold-specific OOF probability distribution may differ.',
            'Exp5 used image-level StratifiedKFold (group_aware=false). If images share patients, evaluate patient leakage risk.',
            'Do not claim the Kaggle score has increased without actual leaderboard evaluation.'
        ],
    }
    (out / 'threshold_tuning_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('=== EXP 5 BACTERIA vs VIRUS POST-PROCESSING (NO TRAINING) ===')
    print(f'OOF rows: {len(oof)}; test rows: {len(test)}; folds: {fold_ids}')
    print(f'Baseline OOF Macro-F1:              {base_score:.6f}')
    print(f'Best full-OOF threshold:             {tuned_full_t:.3f}')
    print(f'Full OOF tuned Macro-F1 (optimistic):{fitted_full_f1:.6f}')
    print(f'Fold-held-out Macro-F1:             {crossfit_score:.6f}')
    print(f'Held-out gain over baseline:        {delta:+.6f}')
    print(f'Fold improvements:                  {fold_wins}/{len(fold_ids)}')
    for r in crossfit_rows:
        print(f"  Fold {r['fold']}: t={r['trained_threshold_excluding_fold']:.3f}, "
              f"baseline={r['heldout_baseline_macro_f1']:.5f}, "
              f"tuned={r['heldout_tuned_macro_f1']:.5f}, "
              f"delta={r['heldout_improvement']:+.5f}")
    print(f'Test predictions changed:           {changes}/{len(test)}')
    print('ASSESSMENT: ' + ('POTENTIALLY USEFUL; check validation details before any Kaggle submission'
                             if promising else 'NO ROBUST GAIN; do not upload this candidate yet'))
    print(f'Output (diagnostic candidate only): {submit_path}')
    print(f'Analysis: {out / "threshold_tuning_metrics.json"}')
    print('Keep the original submission_exp5.csv unchanged.')


if __name__ == '__main__':
    main()
