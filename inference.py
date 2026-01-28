import argparse
import os
import sys
import torch
import cv2
import numpy as np
import albumentations as albu
from torchvision import transforms
from PIL import Image
import json

# Add HierSwin to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'HierSwin'))

from better_mistakes.model.hiera import HieRA
from better_mistakes.trees import load_hierarchy, get_weighting, get_classes

def get_args():
    parser = argparse.ArgumentParser(description='HiCervix Inference Script')
    parser.add_argument('--image', type=str, help='Path to a single image file')
    parser.add_argument('--input_dir', type=str, help='Path to a directory of images')
    parser.add_argument('--csv', type=str, help='Path to a CSV file containing image paths')
    parser.add_argument('--root_dir', type=str, help='Root directory for images in CSV (optional)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (checkpoint.pth.tar)')
    parser.add_argument('--output', type=str, default='inference_results.csv', help='Path to save the results CSV')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (if available)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    # Initialize model
    # HiCervix counts: L1: 4, L2: 21, L3: 23
    model = HieRA(num_classes_l1=4, num_classes_l2=21, num_classes_l3=23, pretrained=False)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict keys if they start with 'module.' (DataParallel)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        # Filter out 'head' keys which are artifacts from init_model_on_gpu and unused in HieRA
        if name in ['head.weight', 'head.bias']:
            continue
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_transforms():
    # Match validation transforms from train.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Albumentations for resizing/cropping
    albu_transform = albu.Compose([
        albu.PadIfNeeded(min_height=1000, min_width=1000,
            border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), always_apply=True),
        albu.CenterCrop(700, 700, always_apply=True),
        albu.Resize(384, 384, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ])

    # Torchvision for tensor conversion and normalization
    torch_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    return albu_transform, torch_transform

def preprocess_image(image_path, albu_transform, torch_transform):
    # Read with OpenCV (BGR)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Albumentations
    augmented = albu_transform(image=img)
    img = augmented['image']
    
    # Convert to PIL for torchvision transforms (or just use ToTensor directly on numpy)
    # train.py does: Image.fromarray(img) -> transform(img)
    img = Image.fromarray(img)
    
    # Apply Torchvision transforms
    img_tensor = torch_transform(img)
    
    return img_tensor

def run_inference(model, img_tensor, device):
    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension
    
    with torch.no_grad():
        logits_l1, logits_l2, logits_l3, _ = model(img_tensor)
        
        # Get predictions
        pred_l1 = torch.argmax(logits_l1, dim=1).item()
        pred_l2 = torch.argmax(logits_l2, dim=1).item()
        pred_l3 = torch.argmax(logits_l3, dim=1).item()
        
        # Get probabilities (softmax)
        prob_l1 = torch.softmax(logits_l1, dim=1).cpu().numpy()[0]
        prob_l2 = torch.softmax(logits_l2, dim=1).cpu().numpy()[0]
        prob_l3 = torch.softmax(logits_l3, dim=1).cpu().numpy()[0]
        
    return {
        'l1': {'pred': pred_l1, 'prob': prob_l1},
        'l2': {'pred': pred_l2, 'prob': prob_l2},
        'l3': {'pred': pred_l3, 'prob': prob_l3}
    }

def main():
    args = get_args()
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    try:
        model = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Transforms
    albu_transform, torch_transform = get_transforms()
    
    # Class names (hardcoded for now based on train.py/HierSwin)
    classes =  ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US',
                    'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS',
                    'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']
    
    # Process images
    images_to_process = [] # List of dicts: {'path': str, 'gt_l3': int/None, 'gt_name': str/None}

    if args.image:
        images_to_process.append({'path': args.image, 'gt_l3': None, 'gt_name': None})
    elif args.input_dir:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    images_to_process.append({'path': os.path.join(root, file), 'gt_l3': None, 'gt_name': None})
    elif args.csv:
        import pandas as pd
        if not os.path.exists(args.csv):
            print(f"CSV file not found: {args.csv}")
            return
        
        df = pd.read_csv(args.csv)
        root_dir = args.root_dir if args.root_dir else ''
        
        for _, row in df.iterrows():
            img_path = os.path.join(root_dir, row['image_name'])
            gt_l3 = row['class_id'] if 'class_id' in row else None
            gt_name = row['class_name'] if 'class_name' in row else None
            images_to_process.append({'path': img_path, 'gt_l3': gt_l3, 'gt_name': gt_name})

    if not images_to_process:
        print("No images found to process.")
        return
        
    print(f"Processing {len(images_to_process)} images...")
    
    results = []
    correct_l3 = 0
    total_with_gt = 0
    
    for item in images_to_process:
        img_path = item['path']
        gt_l3 = item['gt_l3']
        gt_name = item['gt_name']
        
        try:
            img_tensor = preprocess_image(img_path, albu_transform, torch_transform)
            if img_tensor is None:
                continue
                
            out = run_inference(model, img_tensor, device)
            
            # Map L3 index to name
            l3_name = classes[out['l3']['pred']] if out['l3']['pred'] < len(classes) else "Unknown"
            
            result_row = {
                'image': os.path.basename(img_path),
                'l1_pred': out['l1']['pred'],
                'l2_pred': out['l2']['pred'],
                'l3_pred': out['l3']['pred'],
                'l3_label': l3_name,
                'l1_conf': float(out['l1']['prob'][out['l1']['pred']]),
                'l3_conf': float(out['l3']['prob'][out['l3']['pred']])
            }
            
            if gt_l3 is not None:
                result_row['gt_l3'] = gt_l3
                result_row['gt_name'] = gt_name
                result_row['correct'] = (out['l3']['pred'] == gt_l3)
                
                if result_row['correct']:
                    correct_l3 += 1
                total_with_gt += 1
                
                # print(f"Image: {os.path.basename(img_path)} -> Pred: {l3_name} (GT: {gt_name}) [{'Correct' if result_row['correct'] else 'Wrong'}]")
            else:
                pass
                # print(f"Image: {os.path.basename(img_path)} -> L3: {l3_name} (Conf: {result_row['l3_conf']:.4f})")
            
            # Progress indicator
            if (len(results)) % 10 == 0:
                print(f"Processed {len(results)}/{len(images_to_process)} images...", end='\r')
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # Save results
    if results:
        import pandas as pd
        from sklearn.metrics import accuracy_score, classification_report
        
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        
        if total_with_gt > 0:
            # Extract predictions and ground truths
            # Note: We need to ensure we have ground truth for all levels if we want to report them.
            # The CSV currently only provides 'class_id' which is L3.
            # We can infer L1/L2 from L3 if we have the hierarchy, but for now let's focus on L3 metrics 
            # as that's what the user provided.
            
            y_true_l3 = [r['gt_l3'] for r in results if 'gt_l3' in r]
            y_pred_l3 = [r['l3_pred'] for r in results if 'gt_l3' in r]
            
            # Filter out any None values just in case
            valid_indices = [i for i, x in enumerate(y_true_l3) if x is not None]
            y_true_l3 = [y_true_l3[i] for i in valid_indices]
            y_pred_l3 = [y_pred_l3[i] for i in valid_indices]

            print("\n" + "="*50)
            print("EVALUATION METRICS")
            print("="*50)
            
            # L3 Accuracy
            acc_l3 = accuracy_score(y_true_l3, y_pred_l3)
            print(f"Level 3 Accuracy: {acc_l3:.4f} ({len(y_true_l3)} samples)")
            
            # L3 Classification Report
            # Get unique classes present in the data to avoid errors if some classes are missing
            unique_labels = sorted(list(set(y_true_l3) | set(y_pred_l3)))
            target_names = [classes[i] for i in unique_labels if i < len(classes)]
            
            print("\nLevel 3 Classification Report:")
            print(classification_report(y_true_l3, y_pred_l3, labels=unique_labels, target_names=target_names, zero_division=0))
            print("="*50)

if __name__ == '__main__':
    main()
