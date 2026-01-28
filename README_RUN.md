HiCervix Prototype — Quick Run Guide

Purpose
- Train and evaluate the HieRA hierarchical classifier (L1/L2/L3) on the HiCervix dataset.

Repository layout (important files)
- dataset/: image folders and CSV splits (`train.csv`, `val.csv`, `test.csv`, `train_prototype.csv`)
- HierSwin/: model implementation and utilities (hierarchy pickles under `HierSwin/data/`)
- train.py: training entrypoint
- inference.py: prediction script for single images, directories or CSV batches

Prerequisites
- Python 3.8+ (use a venv)
- Install requirements:

```powershell
pip install -r requirements.txt
```

Dataset expectations
- Images stored under `dataset/train/<CLASS>/...`
- CSV files with header `image_name,class_id,class_name` and `image_name` entries relative to `dataset/`, e.g. `CLASS_A/img1.jpg`
- If you have the 40k-image dataset: copy class subfolders into `dataset/train/` and generate `train.csv`, `val.csv`, `test.csv` (70/15/15 split). Validate that all `image_name` files exist.

Quick CSV generation snippet (runs in repo root; adjusts to Windows paths):

```powershell
python - <<'PY'
import os, csv, random
root = r'dataset\train'
images = []
for cls in sorted(os.listdir(root)):
    clsdir = os.path.join(root, cls)
    if os.path.isdir(clsdir):
        for f in os.listdir(clsdir):
            if f.lower().endswith(('.jpg','.png','.jpeg','.bmp','.tiff')):
                images.append((f"{cls}/{f}", cls))
random.shuffle(images)
n = len(images)
t1 = int(n*0.7); t2 = int(n*0.85)
splits = {'dataset\\train.csv': images[:t1], 'dataset\\val.csv': images[t1:t2], 'dataset\\test.csv': images[t2:]}
classes = sorted({c for _,c in images})
class_to_id = {c:i for i,c in enumerate(classes)}
for path, items in splits.items():
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['image_name','class_id','class_name'])
        for img, cls in items:
            w.writerow([img, class_to_id[cls], cls])
print('Wrote CSVs, totals:', {p:len(v) for p,v in splits.items()})
PY
```

Training (dry-run)
- Run a short dry-run to verify everything works:

```powershell
python train.py --epochs 1 --batch-size 4 --workers 2 --output output_test_small
```

Training (full)
- For the full dataset, tune:
  - `--batch-size` (reduce if GPU memory is low)
  - `--workers` (data loader workers)
  - `--epochs` and `--lr`
- Default model uses a Swin-Large backbone which needs substantial GPU memory. If you lack memory, use a smaller backbone (edit `HierSwin/better_mistakes/model/hiera.py`) or reduce batch size.

Inference
- Single image:

```powershell
python inference.py --image path\to\img.jpg --checkpoint output\checkpoint.pth.tar
```

- Directory or CSV batch:

```powershell
python inference.py --input_dir path\to\images --checkpoint output\checkpoint.pth.tar --output results.csv
python inference.py --csv dataset\test.csv --root_dir dataset --checkpoint output\checkpoint.pth.tar
```

Checkpoints & outputs
- `--output` folder contains `checkpoint.pth.tar`, `json/` summaries and `model_snapshots/`.

Validation steps I can run for you
- Verify that every `image_name` in the CSVs exists under `dataset/`.
- Produce `class_counts.csv` listing per-class counts.
- Run a single-batch dry-run locally (CPU) to confirm data loading and model init.

If you want one of the validation steps executed now, tell me which and I'll run it.
