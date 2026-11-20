
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

from lab3_efficientnet_b3_csv_v2 import CLASSES, CSVDataset, build_model, create_transform


class PairedViewDataset(Dataset):
    def __init__(self, directory, filenames, labels=None):
        self.directory = Path(directory)
        self.filenames = list(filenames)
        self.labels = list(labels) if labels is not None else None
        self.transform = create_transform(False)
        missing = [name for name in self.filenames if not (self.directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} missing images; examples: {missing[:5]}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        with Image.open(self.directory / name) as im:
            rgb = im.convert('RGB')
            base = self.transform(rgb)
            # A deterministic, mild intensity-only test-time view.
            # Changes neither geometry nor left/right orientation.
            contrast = ImageEnhance.Contrast(rgb).enhance(1.05)
            second = ImageEnhance.Brightness(contrast).enhance(1.02)
            changed = self.transform(second)
        return base, changed


def filenames_from_sample(path):
    with Path(path).open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['new_filename'] + CLASSES:
            raise ValueError(f'Sample columns mismatch: {reader.fieldnames}')
        names = [r['new_filename'].strip() for r in reader]
    if len(names) != 1182 or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError('Expected 1182 unique, nonempty test filenames')
    return names


def checkpoint_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if list(ckpt.get('classes', [])) != CLASSES:
        raise ValueError(f'Checkpoint classes mismatch: {path}')
    model = build_model(False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(device).eval()
    print(f'Loaded checkpoint {path}: epoch={ckpt.get("epoch")}, val_macro_f1={ckpt.get("val_macro_f1")}')
    return model


@torch.inference_mode()
def predict_probabilities(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device.type == 'cuda'))
    base_probs, tta_probs = [], []
    for first, second in loader:
        first = first.to(device, non_blocking=True)
        second = second.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            logits_0 = model(first)
            logits_1 = model(second)
        p0 = torch.softmax(logits_0.float(), dim=1)
        p1 = torch.softmax(logits_1.float(), dim=1)
        base_probs.append(p0.cpu().numpy())
        tta_probs.append((0.5 * (p0 + p1)).cpu().numpy())
    return np.concatenate(base_probs), np.concatenate(tta_probs)


def candidate_probs(probs, focal_weight, use_tta):
    key = 'tta' if use_tta else 'base'
    return (1.0 - focal_weight) * probs['weighted'][key] + focal_weight * probs['focal'][key]


def save_csv(path, fieldnames, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run(args):
    data_root = Path(args.data_root)
    val_ds = CSVDataset(data_root / 'val_images', data_root / 'val_data.csv', train=False)
    val_names = [p.name for p, _ in val_ds.records]
    val_labels = np.asarray([int(label) for _, label in val_ds.records], dtype=int)
    test_names = filenames_from_sample(args.sample_csv)
    val_images = PairedViewDataset(data_root / 'val_images', val_names)
    test_images = PairedViewDataset(data_root / 'test_images', test_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; validation: {len(val_names)}; test: {len(test_names)}')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    probs_val = {}
    probs_test = {}
    for key, ckpt_path in [('weighted', args.weighted_ckpt), ('focal', args.focal_ckpt)]:
        net = checkpoint_model(ckpt_path, device)
        v0, vt = predict_probabilities(net, val_images, device, args.batch_size)
        t0, tt = predict_probabilities(net, test_images, device, args.batch_size)
        probs_val[key] = {'base': v0, 'tta': vt}
        probs_test[key] = {'base': t0, 'tta': tt}
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    results = []
    for tta in [False, True]:
        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val_prob = candidate_probs(probs_val, weight, tta)
            pred = val_prob.argmax(axis=1)
            score = float(f1_score(val_labels, pred, labels=[0, 1, 2, 3],
                                   average='macro', zero_division=0))
            results.append({'focal_weight':weight, 'weighted_weight':round(1-weight, 2),
                            'tta':int(tta), 'val_macro_f1':round(score, 8)})

    results.sort(key=lambda r: (-r['val_macro_f1'], r['tta'], abs(r['focal_weight'] - 0.5)))
    save_csv(output / 'val_candidates.csv',
             ['focal_weight', 'weighted_weight', 'tta', 'val_macro_f1'], results)
    best = results[0]
    selected_val = candidate_probs(probs_val, best['focal_weight'], bool(best['tta']))
    selected_test = candidate_probs(probs_test, best['focal_weight'], bool(best['tta']))
    val_pred = selected_val.argmax(axis=1)
    test_pred = selected_test.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        val_labels, val_pred, labels=[0, 1, 2, 3], zero_division=0)
    metrics = {
        'selected': best,
        'validation_count': len(val_labels),
        'per_class': {name: {'precision':float(precision[i]),'recall':float(recall[i]),
                             'f1':float(f1[i]),'support':int(support[i])}
                      for i, name in enumerate(CLASSES)},
        'confusion_matrix': confusion_matrix(val_labels, val_pred,
                                             labels=[0,1,2,3]).tolist(),
        'note': 'Model combination chosen ONLY using validation. Public Kaggle score not used for tuning.'
    }
    (output / 'best_val_metrics_exp4.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

    errors = []
    for i, (true, pred) in enumerate(zip(val_labels, val_pred)):
        if true != pred:
            errors.append({'filename':val_names[i], 'true_class':CLASSES[true],
                           'predicted_class':CLASSES[pred],
                           'predicted_confidence':round(float(selected_val[i,pred]), 5)})
    errors.sort(key=lambda r: -r['predicted_confidence'])
    save_csv(output / 'validation_errors.csv',
             ['filename','true_class','predicted_class','predicted_confidence'], errors)

    submission_rows = []
    prob_rows = []
    for i, name in enumerate(test_names):
        pred = int(test_pred[i])
        submission_rows.append({'new_filename': name,
                                **{cls:int(j == pred) for j,cls in enumerate(CLASSES)}})
        prob_rows.append({'new_filename': name,
                          **{f'prob_{cls}':round(float(selected_test[i,j]), 7)
                             for j,cls in enumerate(CLASSES)}})
    save_csv(output / 'submission_exp4.csv', ['new_filename'] + CLASSES, submission_rows)
    save_csv(output / 'test_probabilities_exp4.csv',
             ['new_filename'] + [f'prob_{cls}' for cls in CLASSES], prob_rows)

    assert [row['new_filename'] for row in submission_rows] == test_names
    assert all(sum(row[c] for c in CLASSES)==1 for row in submission_rows)
    print('\nTop candidate configurations (validation only):')
    for r in results[:5]:
        print(f'  focal={r["focal_weight"]:.2f}, weighted={r["weighted_weight"]:.2f}, '
              f'TTA={bool(r["tta"])}, Macro-F1={r["val_macro_f1"]:.5f}')
    print(f'BEST: focal={best["focal_weight"]:.2f}, weighted={best["weighted_weight"]:.2f}, '
          f'TTA={bool(best["tta"])}, val Macro-F1={best["val_macro_f1"]:.5f}')
    print(f'Validation errors: {len(errors)}')
    print(f'Kaggle file: {output / "submission_exp4.csv"}')
    print('Upload ONLY submission_exp4.csv, not test_probabilities_exp4.csv')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--sample-csv', default='test_data_sample.csv')
    p.add_argument('--weighted-ckpt', default='runs/exp2_weighted_ce/best_model.pt')
    p.add_argument('--focal-ckpt', default='runs/exp3_focal/best_model.pt')
    p.add_argument('--output-dir', default='runs/exp4_ensemble')
    p.add_argument('--batch-size', type=int, default=4)
    run(p.parse_args())


if __name__ == '__main__':
    main()
