
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

from lab3_efficientnet_b3_csv_v2 import (CLASSES, CSVDataset, build_model, create_transform)
from lab3_exp7_highres import make_transform


class ImageList(Dataset):

    def __init__(self, root, filenames):
        self.root = Path(root)
        self.filenames = list(filenames)
        self.transform_300 = create_transform(False)
        self.transform_380 = make_transform(380, train=False, experiment='A')
        missing = [s for s in self.filenames if not (self.root / s).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing images under {self.root}; e.g. {missing[:4]}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with Image.open(self.root / self.filenames[index]) as im:
            rgb = im.convert('RGB')
            return self.transform_300(rgb), self.transform_380(rgb)


def load_model(checkpoint_path, expected_size, device):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}')
    data = torch.load(path, map_location='cpu', weights_only=False)
    if list(data.get('classes', [])) != CLASSES:
        raise ValueError(f'Class order mismatch: {path}, got {data.get("classes")}')
    size = int(data.get('image_size', 300))
    if size != expected_size:
        raise ValueError(f'Expected {expected_size}px checkpoint, but got {size}px: {path}')
    net = build_model(pretrained=False)
    net.load_state_dict(data['model_state_dict'], strict=True)
    net.to(device).eval()
    return net, data


@torch.inference_mode()
def inference(checkpoint_path, size, ds, device, batch_size, num_workers):
    model, data = load_model(checkpoint_path, size, device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    res = []
    for batch_no, (x_300, x_380) in enumerate(loader, start=1):
        x = x_300 if size == 300 else x_380
        x = x.to(device, non_blocking=True)
        # FP32 only. Avoid prior mixed-precision NaN issues in test predictions.
        logits = model(x)
        if not torch.isfinite(logits).all():
            start = (batch_no - 1) * batch_size
            raise FloatingPointError(f'Non-finite logits in {checkpoint_path}; '
                                     f'batch {batch_no}; filenames {ds.filenames[start:start+batch_size]}')
        p = torch.softmax(logits, dim=1).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError(f'Non-finite probabilities in {checkpoint_path}; batch {batch_no}')
        res.append(p)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    probs = np.concatenate(res, axis=0)
    if probs.shape != (len(ds), 4):
        raise AssertionError(f'Bad probability shape: {probs.shape}')
    if np.any(probs < -1e-6) or not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
        raise AssertionError('Invalid model probabilities')
    return probs, {'epoch': data.get('epoch'), 'val_macro_f1_from_checkpoint': data.get('val_macro_f1'),
                   'image_size': size}


def metric_report(y_true, probs):
    y_pred = probs.argmax(axis=1).astype(int)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred,
                                                       labels=[0, 1, 2, 3], zero_division=0)
    return {
        'macro_f1': float(f1_score(y_true, y_pred, labels=[0, 1, 2, 3],
                                   average='macro', zero_division=0)),
        'per_class': {
            label: {'precision': float(p[i]), 'recall': float(r[i]),
                    'f1': float(f[i]), 'support': int(support[i])}
            for i, label in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_filenames(sample_csv):
    with Path(sample_csv).open('r', newline='', encoding='utf-8-sig') as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames != ['new_filename'] + CLASSES:
            raise ValueError(f'Incorrect sample submission columns: {rdr.fieldnames}')
        names = [r['new_filename'].strip() for r in rdr]
    if len(names) != 1182 or len(set(names)) != len(names) or not all(names):
        raise ValueError('Expected exactly 1182 nonempty, unique filenames in sample order')
    return names


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_ds = CSVDataset(root / 'val_images', root / 'val_data.csv', train=False)
    names_val = [p.name for p, _ in val_ds.records]
    labels_val = np.array([int(y) for _, y in val_ds.records], dtype=int)
    names_test = test_filenames(args.sample_csv)
    val_images = ImageList(root / 'val_images', names_val)
    test_images = ImageList(root / 'test_images', names_test)

    print(f'Device={device}, precision=FP32, validation={len(names_val)}, test={len(names_test)}', flush=True)
    val3, info3 = inference(args.exp3_ckpt, 300, val_images, device,
                            args.batch_size, args.num_workers)
    print('Exp3 validation inference completed.', flush=True)
    val7, info7 = inference(args.exp7a_ckpt, 380, val_images, device,
                            args.batch_size, args.num_workers)
    print('Exp7A validation inference completed.', flush=True)

    reports = {
        'exp3_300': metric_report(labels_val, val3),
        'exp7a_380': metric_report(labels_val, val7),
        'fixed_50_50': metric_report(labels_val, (val3 + val7) / 2.0),
    }
    expected = [info3.get('val_macro_f1_from_checkpoint'), info7.get('val_macro_f1_from_checkpoint')]
    for key, checkpoint_score in zip(['exp3_300', 'exp7a_380'], expected):
        actual = reports[key]['macro_f1']
        if checkpoint_score is not None and abs(actual - float(checkpoint_score)) > 0.02:
            raise RuntimeError(f'{key} validation score {actual:.6f} differs significantly from checkpoint '
                               f'{float(checkpoint_score):.6f}. Please confirm data, preprocessing and split.')

    diagnostics = []
    for weight7 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pred = ((1 - weight7) * val3 + weight7 * val7).argmax(axis=1)
        diagnostics.append({'exp7a_weight': weight7, 'exp3_weight': 1 - weight7,
                            'val_macro_f1': float(f1_score(labels_val, pred, labels=[0, 1, 2, 3],
                                                           average='macro', zero_division=0))})
    write_csv(out / 'val_weights_diagnostic_only.csv',
              ['exp7a_weight', 'exp3_weight', 'val_macro_f1'], diagnostics)

    p3, p7 = val3.argmax(axis=1), val7.argmax(axis=1)
    p50 = ((val3 + val7) / 2).argmax(axis=1)
    disagreements = []
    for i, name in enumerate(names_val):
        if p3[i] != p7[i]:
            disagreements.append({
                'filename': name, 'true_class': CLASSES[labels_val[i]],
                'exp3_pred': CLASSES[p3[i]], 'exp7a_pred': CLASSES[p7[i]],
                'fusion_pred': CLASSES[p50[i]],
                'exp3_correct': int(p3[i] == labels_val[i]),
                'exp7a_correct': int(p7[i] == labels_val[i]),
                'fusion_correct': int(p50[i] == labels_val[i]),
            })
    write_csv(out / 'validation_disagreements.csv',
              ['filename', 'true_class', 'exp3_pred', 'exp7a_pred', 'fusion_pred',
               'exp3_correct', 'exp7a_correct', 'fusion_correct'], disagreements)

    summary = {
        'model_3': info3, 'model_7a': info7,
        'reports': reports, 'validation_model_disagreements': len(disagreements),
        'fixed_fusion_delta_over_exp3': reports['fixed_50_50']['macro_f1'] - reports['exp3_300']['macro_f1'],
        'fixed_fusion_delta_over_exp7a': reports['fixed_50_50']['macro_f1'] - reports['exp7a_380']['macro_f1'],
        'method_note': 'The 50/50 weight was fixed before validation. Same validation data used to select '
                       'individual models, so results are development estimates, not independent test scores.',
        'patient_group_note': 'No patient-group split can be verified from these files alone.',
    }
    (out / 'exp8_validation_metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                       encoding='utf-8')
    print('\n=== EXP 8 VALIDATION COMPARISON ===')
    for key, report in reports.items():
        print(f'{key}: Macro-F1={report["macro_f1"]:.6f}, '
              f'Virus F1={report["per_class"]["virus"]["f1"]:.6f}, '
              f'Bacteria F1={report["per_class"]["bacteria"]["f1"]:.6f}')
    print(f'Exp3 vs Exp7A disagreements: {len(disagreements)}/{len(val_ds)}')

    if args.val_only:
        print('Validation-only mode: No Kaggle test predictions generated.')
        return

    test3, _ = inference(args.exp3_ckpt, 300, test_images, device,
                         args.batch_size, args.num_workers)
    print('Exp3 test inference completed.', flush=True)
    test7, _ = inference(args.exp7a_ckpt, 380, test_images, device,
                         args.batch_size, args.num_workers)
    print('Exp7A test inference completed.', flush=True)
    test50 = (test3 + test7) / 2.0
    if not np.isfinite(test50).all() or not np.allclose(test50.sum(axis=1), 1, atol=1e-4):
        raise AssertionError('Invalid ensemble test probabilities: no Kaggle CSV written')
    preds = test50.argmax(axis=1)
    rows = [{'new_filename': name, **{label: int(int(idx) == cls)
                                      for cls, label in enumerate(CLASSES)}}
            for name, idx in zip(names_test, preds)]
    prob_rows = [{'new_filename': name, **{f'prob_{label}': f'{prob:.9g}'
                                                for label, prob in zip(CLASSES, test50[i])}}
                 for i, name in enumerate(names_test)]
    if [r['new_filename'] for r in rows] != names_test or any(sum(r[c] for c in CLASSES) != 1 for r in rows):
        raise AssertionError('Invalid Kaggle row order or one-hot values')
    write_csv(out / 'submission_exp8_fixed50.csv', ['new_filename'] + CLASSES, rows)
    write_csv(out / 'test_probabilities_exp8_fixed50.csv', ['new_filename'] + [f'prob_{c}' for c in CLASSES],
              prob_rows)
    print(f'Created candidate (NOT submitted): {out / "submission_exp8_fixed50.csv"} '
          f'({len(rows)} rows, original filename order).')
    print('Do not upload probability CSV to Kaggle.')
    if reports['fixed_50_50']['macro_f1'] <= max(reports['exp3_300']['macro_f1'],
                                                 reports['exp7a_380']['macro_f1']):
        print('ASSESSMENT: 50/50 fusion did not beat both single models on validation.')
    else:
        print('ASSESSMENT: 50/50 fusion improved validation Macro-F1; independent test gain not guaranteed.')


def main():
    parser = argparse.ArgumentParser(description='Lab3 Exp8 FP32 50/50 ensemble (no training, no upload)')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--sample-csv', default='test_data_sample.csv')
    parser.add_argument('--exp3-ckpt', default='runs/exp3_focal/best_model.pt')
    parser.add_argument('--exp7a-ckpt', default='runs/exp7A_380_original_aug/best_model.pt')
    parser.add_argument('--output-dir', default='runs/exp8_ensemble')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--val-only', action='store_true',
                        help='Only run 709 validation images; do not predict 1182 test images')
    a = parser.parse_args()
    if a.batch_size < 1 or a.num_workers < 0:
        parser.error('Invalid batch-size/num-workers')
    run(a)


if __name__ == '__main__':
    main()
