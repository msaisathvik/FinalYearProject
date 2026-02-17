import time
import numpy as np
import os.path
import torch
from contextlib import nullcontext

# Optional dependency: tensorboardX (fallback to torch.utils.tensorboard if available)
try:
    import tensorboardX  # type: ignore
except Exception:  # pragma: no cover
    tensorboardX = None

# The original code depended on the external `conditional` package.
# To make training runnable in Kaggle (and other minimal envs), we provide a local equivalent.
def conditional(condition, context_manager):
    """
    Return `context_manager` if condition is True, else a no-op context.
    """
    return context_manager if condition else nullcontext()

from better_mistakes.model.performance import accuracy
import torch.nn.functional as F
from .tct_get_tree_target import get_order_family_target
from sklearn.metrics import precision_score, recall_score, f1_score

def run_hiera(loader, model, loss_function, opts, epoch, prev_steps, optimizer=None, is_inference=True):
    """
    Runs training or inference routine for HieRA
    """
    
    # Logging setup
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    
    with_tb = (opts.out_folder is not None) and (tensorboardX is not None)
    if with_tb:
        tb_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_folder, "tb", descriptor.lower()))

    loss_accum = 0.0
    acc_l1_accum = 0.0
    acc_l2_accum = 0.0
    acc_l3_accum = 0.0
    num_logged = 0
    time_accum = 0.0

    # Metrics accumulators
    all_preds_l1, all_targets_l1 = [], []
    all_preds_l2, all_targets_l2 = [], []
    all_preds_l3, all_targets_l3 = [], []

    # In case the loader is empty, keep a defined steps value.
    tot_steps = prev_steps
    
    if is_inference:
        model.eval()
    else:
        model.train()

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (images, target) in enumerate(loader):
            this_load_time = time.time() - time_load0
            this_rest0 = time.time()
            
            # Move to the same device as the model (robust for CPU/GPU and avoids device mismatch)
            device = next(model.parameters()).device
            images = images.to(device, non_blocking=torch.cuda.is_available())
            
            # Get targets for all levels
            target_l1, target_l2, target_l3 = get_order_family_target(target)
            target_l1 = target_l1.to(device)
            target_l2 = target_l2.to(device)
            target_l3 = target_l3.to(device)
            
            # Forward pass
            output = model(images) # (logits_l1, logits_l2, logits_l3, features)
            logits_l1, logits_l2, logits_l3, _ = output
            
            # Compute Loss
            loss = loss_function(output, (target_l1, target_l2, target_l3))
            
            if not is_inference:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Measure Accuracy
            # L1
            acc_l1, _ = accuracy(logits_l1, target_l1, ks=(1,))
            acc_l1_accum += acc_l1[0].item()
            
            # L2 (Handle -1)
            mask_l2 = target_l2 != -1
            if mask_l2.sum() > 0:
                acc_l2, _ = accuracy(logits_l2[mask_l2], target_l2[mask_l2], ks=(1,))
                acc_l2_accum += acc_l2[0].item()
            
            # L3
            mask_l3 = target_l3 != -1
            if mask_l3.sum() > 0:
                acc_l3, _ = accuracy(logits_l3[mask_l3], target_l3[mask_l3], ks=(1,))
                acc_l3_accum += acc_l3[0].item()
            
            loss_accum += loss.item()
            num_logged += 1

            # Collect predictions and targets for metrics
            all_preds_l1.append(logits_l1.argmax(dim=1).detach().cpu().numpy())
            all_targets_l1.append(target_l1.detach().cpu().numpy())
            
            all_preds_l2.append(logits_l2.argmax(dim=1).detach().cpu().numpy())
            all_targets_l2.append(target_l2.detach().cpu().numpy())
            
            all_preds_l3.append(logits_l3.argmax(dim=1).detach().cpu().numpy())
            all_targets_l3.append(target_l3.detach().cpu().numpy())
            
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()
            
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            if batch_idx % opts.log_freq == 0:
                print(
                    "**%8s [Epoch %03d/%03d, Batch %05d/%05d]\t"
                    "Time: %2.1f ms | \t"
                    "Loss: %2.3f\t"
                    "Acc L1: %2.2f%%\t"
                    "Acc L2: %2.2f%%\t"
                    "Acc L3: %2.2f%%"
                    % (descriptor, epoch, opts.epochs, batch_idx, len(loader), 
                       time_accum / (batch_idx + 1) * 1000, 
                       loss.item(),
                       acc_l1[0].item() * 100,
                       acc_l2[0].item() * 100 if mask_l2.sum() > 0 else 0,
                       acc_l3[0].item() * 100 if mask_l3.sum() > 0 else 0)
                )
                
                if with_tb:
                    tb_writer.add_scalar(loss_id, loss.item(), tot_steps)
                    tb_writer.add_scalar("accuracy/L1", acc_l1[0].item(), tot_steps)
                    tb_writer.add_scalar("accuracy/L3", acc_l3[0].item(), tot_steps)

    if num_logged == 0:
        raise RuntimeError(
            "No batches were processed. Check that your CSV paths resolve to existing images, "
            "and that dataset size is >= 1 (and not dropped by `drop_last`)."
        )

    # Calculate Epoch Metrics
    # L1
    y_true_l1 = np.concatenate(all_targets_l1)
    y_pred_l1 = np.concatenate(all_preds_l1)
    p_l1 = precision_score(y_true_l1, y_pred_l1, average='macro', zero_division=0)
    r_l1 = recall_score(y_true_l1, y_pred_l1, average='macro', zero_division=0)
    f1_l1 = f1_score(y_true_l1, y_pred_l1, average='macro', zero_division=0)

    # L2
    y_true_l2 = np.concatenate(all_targets_l2)
    y_pred_l2 = np.concatenate(all_preds_l2)
    mask_l2 = y_true_l2 != -1
    if mask_l2.sum() > 0:
        p_l2 = precision_score(y_true_l2[mask_l2], y_pred_l2[mask_l2], average='macro', zero_division=0)
        r_l2 = recall_score(y_true_l2[mask_l2], y_pred_l2[mask_l2], average='macro', zero_division=0)
        f1_l2 = f1_score(y_true_l2[mask_l2], y_pred_l2[mask_l2], average='macro', zero_division=0)
    else:
        p_l2, r_l2, f1_l2 = 0.0, 0.0, 0.0

    # L3
    y_true_l3 = np.concatenate(all_targets_l3)
    y_pred_l3 = np.concatenate(all_preds_l3)
    mask_l3 = y_true_l3 != -1
    if mask_l3.sum() > 0:
        p_l3 = precision_score(y_true_l3[mask_l3], y_pred_l3[mask_l3], average='macro', zero_division=0)
        r_l3 = recall_score(y_true_l3[mask_l3], y_pred_l3[mask_l3], average='macro', zero_division=0)
        f1_l3 = f1_score(y_true_l3[mask_l3], y_pred_l3[mask_l3], average='macro', zero_division=0)
    else:
        p_l3, r_l3, f1_l3 = 0.0, 0.0, 0.0

    print(f"\n{descriptor} Epoch Metrics:")
    print(f"L1 - P: {p_l1:.4f}, R: {r_l1:.4f}, F1: {f1_l1:.4f}")
    print(f"L2 - P: {p_l2:.4f}, R: {r_l2:.4f}, F1: {f1_l2:.4f}")
    print(f"L3 - P: {p_l3:.4f}, R: {r_l3:.4f}, F1: {f1_l3:.4f}")

    summary = {
        "loss": loss_accum / num_logged,
        "acc_l1": acc_l1_accum / num_logged,
        "acc_l2": acc_l2_accum / num_logged,
        "acc_l3": acc_l3_accum / num_logged,
        "precision_l1": p_l1, "recall_l1": r_l1, "f1_l1": f1_l1,
        "precision_l2": p_l2, "recall_l2": r_l2, "f1_l2": f1_l2,
        "precision_l3": p_l3, "recall_l3": r_l3, "f1_l3": f1_l3,
    }
    
    if with_tb:
        tb_writer.close()
        
    return summary, tot_steps
