# HiCervix 

This folder contains a standalone prototype for training the HieRA model on the HiCervix dataset (sampled version).

## Directory Structure

- `dataset/`: Contains the sampled images (50 per class) and `train_prototype.csv`.
- `HierSwin/`: Contains the model code and data files (pickle files).
- `train.py`: The script to run the training.

## Dataset Splits

The dataset has been split into:
- `dataset/train.csv`: Training set (70%)
- `dataset/val.csv`: Validation set (15%)
- `dataset/test.csv`: Test set (15%)

## How to Run

### Training

To start training with the default settings (using `train.csv` and `val.csv`):

```bash
python train.py
```

You can customize parameters:

```bash
python train.py --epochs 50 --batch-size 8 --output output_experiment_1
```

### Testing

To evaluate a trained model on the test set (`dataset/test.csv`):

```bash
python HierSwin/scripts/start_testing_hiera.py --checkpoint output/checkpoint.pth.tar --test-csv dataset/test.csv --data_dir dataset
```

*Note: Replace `output/checkpoint.pth.tar` with the actual path to your trained model checkpoint.*

