import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
from pprint import PrettyPrinter

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import albumentations as albu
import pickle
import pandas as pd
import cv2
import time

from better_mistakes.util.rand import make_deterministic
from better_mistakes.model.init import init_model_on_gpu
from better_mistakes.model.run_hiera import run_hiera
from better_mistakes.model.losses import HierarchicalCrossEntropyLoss
from better_mistakes.trees import load_hierarchy, get_weighting, get_classes, DistanceDict

from dataset import InputDataset

MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.append('swinT')
MODEL_NAMES.append('hiera')
DATASET_NAMES = ["hicervix"]

def load_data(test_csv, data_root, distributed):
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])

    test_albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
        ])

    print("Loading test data")
    # Point to the test folder (assuming nested like train/val)
    test_root = os.path.join(data_root, 'test', 'test')
    # If nested structure is not consistent, we might need to check.
    # But assuming consistency with train/val.
    if not os.path.exists(test_root):
         # Fallback to non-nested if not found
         test_root = os.path.join(data_root, 'test')
         
    dataset_test = InputDataset(test_csv, test_root, False, test_transform,
            albu_transform=test_albu_transform)

    print("Creating data loaders")
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler

def main(opts):
    if opts.gpu is not None:
        print("Use GPU: {} for testing".format(opts.gpu))

    cudnn.benchmark = True
    pp = PrettyPrinter(indent=4)

    # Load hierarchy and classes
    with open('data/tct_distances.pkl', "rb") as f:
        distances = DistanceDict(pickle.load(f))
    with open('data/tct_tree.pkl', "rb") as f:
        hierarchy = pickle.load(f)

    classes =  ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US',
                    'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS',
                    'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']
    opts.num_classes = len(classes)

    # Setup model
    model = init_model_on_gpu(torch.cuda.device_count(), opts)
    
    # Load checkpoint
    if os.path.isfile(opts.checkpoint):
        print("=> loading checkpoint '{}'".format(opts.checkpoint))
        checkpoint = torch.load(opts.checkpoint, map_location='cpu') # Load to CPU first
        model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.checkpoint, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(opts.checkpoint))
        return

    # Setup loss
    # Using SupervisedContrastiveHierarchicalLoss to match training
    from better_mistakes.model.losses import SupervisedContrastiveHierarchicalLoss
    weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
    loss_function = SupervisedContrastiveHierarchicalLoss(hierarchy, classes, weights, alpha=opts.alpha)
    
    if torch.cuda.is_available():
        loss_function = loss_function.cuda(opts.gpu)

    # Load data
    dataset_test, test_sampler = load_data(opts.test_csv, opts.data_dir, False)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=opts.batch_size,
        sampler=test_sampler, num_workers=opts.workers, pin_memory=True)

    # Run testing
    print("Starting testing...")
    # Ensure opts has missing attributes for init_model_on_gpu if not added by parser
    if not hasattr(opts, 'devise'): opts.devise = False
    if not hasattr(opts, 'barzdenzler'): opts.barzdenzler = False
    
    summary_val, steps = run_hiera(
        test_loader, model, loss_function, opts, 0, 0, optimizer=None, is_inference=True
    )

    print("\nTest Summary:")
    pp.pprint(summary_val)
    
    # Save results
    if not os.path.exists(opts.output):
        os.makedirs(opts.output)
    
    with open(os.path.join(opts.output, "test_summary.json"), "w") as fp:
        json.dump(summary_val, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="hiera", choices=MODEL_NAMES, help="model architecture")
    parser.add_argument("--test-csv", default='../dataset/test.csv', help="test csv of HiCervix")
    parser.add_argument("--data_dir", default="../dataset", help="Folder containing the dataset")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default='output_test_results', help="path to save results")
    parser.add_argument("--batch-size", default=8, type=int, help="batch size")
    parser.add_argument("--workers", default=2, type=int, help="number of workers")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id")
    parser.add_argument("--alpha", type=float, default=0.4, help="Decay parameter for hierarchical cross entropy.")
    parser.add_argument("--loss", default="hierarchical-cross-entropy", help="loss type")
    parser.add_argument("--out_folder", default=None, help="Output folder for tensorboard (optional)")
    parser.add_argument("--log_freq", default=10, type=int, help="Log frequency")
    parser.add_argument("--epochs", default=1, type=int, help="Dummy epoch for logging")
    
    opts = parser.parse_args()
    main(opts)
