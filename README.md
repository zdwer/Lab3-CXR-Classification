# Lab 3 - Multi-class CXR Classification


- `lab3_efficientnet_b3_csv_v2.py`: dataset loader, preprocessing, focal/weighted losses, Exp 3 model training.
- `lab3_exp7_highres.py`: Exp 7A / Exp 7B high-resolution training, FP32 validation, and checkpoint save.
- `lab3_exp8_ensemble.py`: fixed 50/50 fusion of Exp 3 and Exp 7A softmax probabilities. This is the best Kaggle submission method.
- `lab3_multimodel_baselines.py`: eight ImageNet-pretrained backbone benchmarks.
- Other `lab3_exp*.py`: previous experiments (two-stage, K-fold, threshold tests, etc.).
- `compare_exp7_metrics.py`: experiment comparison.
- `verify_kaggle_csv.py`: strict CSV format/order/one-hot checks.

**The Kaggle data, trained checkpoint files, and actual Kaggle submission CSV are not included in the GitHub source package.** Keep checkpoints in `runs/` locally, and obtain the dataset from the official competition.

## Setup

Tested locally by the student with Python 3.10 and NVIDIA RTX 2000 Ada. Create a Python environment with matching CUDA-enabled PyTorch and install `requirements.txt`. Requirements are version ranges, not an exact lockfile; record the actual environment versions for strict reproducibility.

Dataset directory:
```
nycu-aimi-2026-lab3-cxr/
  train_images/
  val_images/
  test_images/
  train_data.csv
  val_data.csv
```

## Reproduce best single models (PowerShell)

Run from the directory containing these Python files. Set `$data` to your local dataset root, e.g. `F:\...\nycu-aimi-2026-lab3-cxr`.

Exp 3 (300px):
```powershell
python .\lab3_efficientnet_b3_csv_v2.py train `
  --train-dir "$data\train_images" --train-csv "$data\train_data.csv" `
  --val-dir "$data\val_images" --val-csv "$data\val_data.csv" `
  --output-dir ".\runs\exp3_focal" --epochs 25 --patience 7 `
  --batch-size 8 --num-workers 0 --lr 0.0001 --loss focal
```

Exp 7A (380px):
```powershell
python .\lab3_exp7_highres.py train `
  --train-dir "$data\train_images" --train-csv "$data\train_data.csv" `
  --val-dir "$data\val_images" --val-csv "$data\val_data.csv" `
  --experiment A --output-dir ".\runs\exp7A_380_original_aug" `
  --image-size 380 --epochs 25 --patience 7 --batch-size 4 `
  --accum-steps 2 --num-workers 0 --lr 0.0001
```

Exp 8 (inference, two checkpoints required):
```powershell
python .\lab3_exp8_ensemble.py `
  --data-root "$data" --sample-csv ".\test_data_sample.csv" `
  --exp3-ckpt ".\runs\exp3_focal\best_model.pt" `
  --exp7a-ckpt ".\runs\exp7A_380_original_aug\best_model.pt" `
  --output-dir ".\runs\exp8_ensemble" --batch-size 4
```

The submission output is `runs/exp8_ensemble/submission_exp8_fixed50.csv`; upload **that one-hot CSV**, not `test_probabilities_exp8_fixed50.csv`. The sample CSV should be copied from the competition into the working directory.

## Metric and limitations

Kaggle uses **four-class macro-F1**. We use the same validation metric to select best checkpoints. Class order: Normal, Bacteria, Virus, COVID-19. The original class balance is highly skewed (39/4017 train COVID-19; 7/709 validation COVID-19), so class-specific F1 matters. All repeated experiments used the same development validation set, risking optimistic model selection. Patient-ID-level grouping cannot be confirmed from the available labels.

## Submission requirements

- Kaggle competition: submit one-hot CSV (1,182 rows) before Nov 23, 2026 at 12:30.
- E3: upload a PDF report named `LAB3_415551015_<Chinese name>.pdf` by Nov 27, 2026 at 12:30.
- GitHub: publish Python code and this README; put the actual repository URL in the report's Github Link section.

The Public Leaderboard score above was provided by the student; this repository does not query Kaggle or include private leaderboard labels.
