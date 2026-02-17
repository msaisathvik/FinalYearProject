#This is a test
import argparse
import os
import sys
# Add HierSwin to path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'HierSwin'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'HierSwin', 'scripts'))

import json
import shutil
import numpy as np
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import albumentations as albu
import pickle
import pandas as pd
import os.path as osp
import cv2
import time

from better_mistakes.util.rand import make_deterministic
from better_mistakes.util.folders import get_expm_folder
from better_mistakes.util.config import load_config
from better_mistakes.model.init import init_model_on_gpu
from better_mistakes.model.run_hiera import run_hiera
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss, SupervisedContrastiveHierarchicalLoss
from better_mistakes.trees import load_hierarchy, get_weighting, get_classes, DistanceDict

from dataset import InputDataset


MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.append('swinT')
MODEL_NAMES.append('hiera')
LOSS_NAMES = ["hierarchical-cross-entropy", "supervised-contrastive"]
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD"]
DATASET_NAMES = ["hicervix"]


print('==> Preparing data..')
def load_data(train_csv, val_csv, data_root, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=10, contrast=10, saturation=20, hue=0.1),
        transforms.ToTensor(),
        normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])

    train_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        #albu.CenterCrop(224, 224, always_apply=True),
        albu.RandomCrop(700, 700, always_apply=True),
        albu.RandomBrightnessContrast(),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    print("Loading training data")
    st = time.time()
    # Point to the train folder
    train_root = os.path.join(data_root, 'train')
    dataset = InputDataset(train_csv, train_root, True, train_transform,
            albu_transform=train_albu_transform)
    print("Took", time.time() - st)

    print("Loading validation data")
    # Point to the val folder (using train for prototype val as well)
    val_root = os.path.join(data_root, 'train')
    dataset_test = InputDataset(val_csv, val_root, False, test_transform,
            albu_transform=test_albu_transform)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler



class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.is_best = False

    def __call__(self, val_loss):
        self.is_best = False
        if self.best_loss is None:
            self.best_loss = val_loss
            self.is_best = True
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.is_best = True

def main_worker(gpus_per_node, opts):
    # Worker setup
    if opts.gpu is not None:
        print("Use GPU: {} for training".format(opts.gpu))

    # Enables the cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True

    # pretty printer for cmd line options
    pp = PrettyPrinter(indent=4)

    # Setup data loaders --------------------------------------------------------------------------------------------------------------------------------------
    batch_size = opts.batch_size
    # Pass the dataset root directory
    dataset, dataset_test, train_sampler, test_sampler = load_data(opts.train_csv, opts.val_csv, opts.data_dir, False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=opts.workers, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size,
        sampler=test_sampler, num_workers=opts.workers, pin_memory=True,drop_last=True)
    
    # Adjust the number of epochs to the size of the dataset
    num_batches = len(train_loader)
    print('Number of epoches: {}'.format(opts.epochs))

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    print(os.getcwd())
    # Ensure these files exist or update paths if needed
    with open('HierSwin/data/tct_distances.pkl', "rb") as f:
        distances = DistanceDict(pickle.load(f))
    with open('HierSwin/data/tct_tree.pkl', "rb") as f:
        hierarchy = pickle.load(f)

    classes =  ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US',
                    'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS',
                    'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']

    opts.num_classes = len(classes)

    # shuffle hierarchy nodes
    if opts.shuffle_classes:
        np.random.shuffle(classes)

    # Model, loss, optimizer ----------------------------------------------------------------------------------------------------------------------------------

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts)

    # setup optimizer
    optimizer = _select_optimizer(model, opts)

    # load from checkpoint if existing
    steps = _load_checkpoint(opts, model, optimizer)

    # setup loss
    if opts.loss == "hierarchical-cross-entropy":
        print("Using SupervisedContrastiveHierarchicalLoss (mapped from hierarchical-cross-entropy)...")
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        # Using SupervisedContrastiveHierarchicalLoss as it handles tuple output from HieRA
        loss_function = SupervisedContrastiveHierarchicalLoss(hierarchy, classes, weights, alpha=opts.alpha)
        if torch.cuda.is_available():
            loss_function = loss_function.cuda(opts.gpu)
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    # Training/evaluation -------------------------------------------------------------------------------------------------------------------------------------

    early_stopping = EarlyStopping(patience=opts.patience, min_delta=opts.min_delta)

    for epoch in range(opts.start_epoch, opts.epochs):

        # do we validate at this epoch?
        do_validate = epoch % opts.val_freq == 0

        # name for the json file s
        json_name = "epoch.%04d.json" % epoch

        if opts.arch == "hiera":
            summary_train, steps = run_hiera(
                train_loader, model, loss_function, opts, epoch, steps, optimizer, is_inference=False
            )
        else:
             raise RuntimeError("Only hiera architecture is supported in this script.")

        # dump results
        with open(os.path.join(opts.out_folder, "json/train", json_name), "w") as fp:
            json.dump(summary_train, fp)

        # print summary of the epoch and save checkpoint
        state = {"epoch": epoch + 1, "steps": steps, "arch": opts.arch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "is_best": is_best}
        _save_checkpoint(state, do_validate, epoch, opts.out_folder)

        # validation
        if do_validate:
            if opts.arch == "hiera":
                summary_val, steps = run_hiera(
                    val_loader, model, loss_function, opts, epoch, steps, optimizer=None, is_inference=True
                )
            else:
                 raise RuntimeError("Only hiera architecture is supported in this script.")

            print("\nSummary for epoch %04d (for val set):" % epoch)
            pp.pprint(summary_val)
            print("\n\n")
            with open(os.path.join(opts.out_folder, "json/val", json_name), "w") as fp:
                json.dump(summary_val, fp)

            # Early Stopping check
            early_stopping(summary_val['loss'])
            
            if early_stopping.is_best:
                print(f"Validation loss improved to {early_stopping.best_loss:.4f}. Saving best model...")
                best_filename = os.path.join(opts.out_folder, "model_best.pth.tar")
                torch.save(state, best_filename)
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break


def _load_checkpoint(opts, model, optimizer):
    if os.path.isfile(os.path.join(opts.out_folder, "checkpoint.pth.tar")):
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "checkpoint.pth.tar"))
        opts.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps = checkpoint["steps"]
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))
    elif opts.pretrained_folder is not None:
        if os.path.exists(opts.pretrained_folder):
            print("=> loading pretrained checkpoint '{}'".format(opts.pretrained_folder))
            if os.path.isdir(opts.pretrained_folder):
                checkpoint = torch.load(os.path.join(opts.pretrained_folder, "checkpoint.pth.tar"))
            else:
                checkpoint = torch.load(opts.pretrained_folder)
            
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            steps = 0
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_folder, checkpoint["epoch"]))
        else:
            raise FileNotFoundError("Can not find {}".format(opts.pretrained_folder))
    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps


def _save_checkpoint(state, do_validate, epoch, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    # ✅ Latest checkpoint (always overwritten)
    latest_path = os.path.join(out_folder, "latest_model.pth.tar")
    torch.save(state, latest_path)

    # ✅ Best checkpoint (saved only if best)
    if state.get("is_best", False):
        best_path = os.path.join(out_folder, "best_model.pth.tar")
        torch.save(state, best_path)

    print(f"Checkpoint saved → Epoch {epoch}")


def _select_optimizer(model, opts):
    if opts.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == "adam_amsgrad":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False, )
    else:
        raise ValueError("Unknown optimizer", opts.loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="hiera", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="hierarchical-cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None, help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.5, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--num_training_steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--epochs", default=5, type=int, help="training epochs")

    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=8, type=int, help="total batch size")
    parser.add_argument("--shuffle_classes", default=False, type=boolean, help="Shuffle classes in the hierarchy")
    parser.add_argument("--beta", default=0, type=float, help="Softness parameter: the higher, the closer to one-hot encoding")
    parser.add_argument("--alpha", type=float, default=0.4, help="Decay parameter for hierarchical cross entropy.")
    
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="hicervix", help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=384, type=int, help="Size of image input to the network (target resize after data augmentation)")
    # Update default paths to point to the correct location (relative to hicervix_prototype root)
    parser.add_argument("--train-csv", default='dataset/train.csv', help="train csv of HiCervix")
    parser.add_argument("--val-csv", default='dataset/val.csv', help="val csv of HiCervix")

    parser.add_argument("--data_dir", default="dataset", help="Folder containing the dataset (train/val/test folders)")
    parser.add_argument("--output", default='output', help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str, help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=10, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=1, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--patience", default=10, type=int, help="Early stopping patience")
    parser.add_argument("--min-delta", default=0.0, type=float, help="Early stopping min delta")

    opts = parser.parse_args()

    # Add missing attributes required by init_model_on_gpu
    opts.devise = False
    opts.barzdenzler = False

    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder, "json/train"))
        os.makedirs(os.path.join(opts.out_folder, "json/val"))
        os.makedirs(os.path.join(opts.out_folder, "model_snapshots"))

    # set if we want to output soft labels or one hot
    opts.soft_labels = opts.beta != 0

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(opts))
    with open(os.path.join(opts.out_folder, "opts.json"), "w") as fp:
        json.dump(vars(opts), fp)

    # setup random number generation
    if opts.seed is not None:
        make_deterministic(opts.seed)

    gpus_per_node = torch.cuda.device_count()
    main_worker(gpus_per_node, opts)
