#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find optimal classification threshold by plotting ROC curve and AUC,
and showing MCC vs threshold.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from CnnLstmModel import CnnLstmModel

# -------------------------------------------------------------------------
# Configuration (MUST MATCH fine_tune_dsa.py for consistent test split)
# -------------------------------------------------------------------------
DATA_ROOT_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/Channel0-DataTypeUnsignedShort-Values0to4000"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "fine_tuned_models", "china_data_unfrozen_cnn")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2  # Midpoint between the two label encodings

TEST_BATCH_SIZE = 1
NUM_WORKERS = 4
FRONTAL_CKPT = os.path.join(OUTPUT_PATH, "frontal_fine_tuned_best_mcc.pt")
LATERAL_CKPT = os.path.join(OUTPUT_PATH, "lateral_fine_tuned_best_mcc.pt")


def load_model(device, checkpoint_path):
    """Load model weights from checkpoint onto the given device."""
    model = CnnLstmModel(512, 3, 1, True, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def get_all_predictions(model_frontal, model_lateral, dataloader, device1, device2):
    """Get all predictions and labels from the test set."""
    all_probs_combined = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions", unit="batch"):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)

            # Combine outputs: average the probabilities
            prob_frontal = torch.sigmoid(output_frontal).item()
            prob_lateral = torch.sigmoid(output_lateral).item()
            prob_combined = (prob_frontal + prob_lateral) / 2.0
            
            all_probs_combined.append(prob_combined)
            all_labels.append(labels_frontal.item())
    
    return np.array(all_probs_combined), np.array(all_labels)


def calculate_roc_curve(probabilities, labels):
    """Calculate ROC curve (TPR vs FPR) and AUC."""
    # Convert labels to binary (0 or 1)
    binary_labels = (labels > LABEL_THRESHOLD).astype(int)
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_labels = binary_labels[sorted_indices]
    
    # Count positives and negatives
    n_positive = np.sum(binary_labels == 1)
    n_negative = np.sum(binary_labels == 0)
    
    if n_positive == 0 or n_negative == 0:
        print("Warning: Only one class present in labels. Cannot calculate ROC curve.")
        return None, None, None
    
    # Initialize arrays
    tpr = [0.0]  # True Positive Rate (Sensitivity)
    fpr = [0.0]  # False Positive Rate (1 - Specificity)
    
    TP = 0
    FP = 0
    
    # Calculate TPR and FPR at each threshold
    for i in range(len(sorted_probs)):
        if sorted_labels[i] == 1:
            TP += 1
        else:
            FP += 1
        
        tpr.append(TP / n_positive)
        fpr.append(FP / n_negative)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return np.array(fpr), np.array(tpr), auc


def calculate_mcc_at_threshold(probabilities, labels, threshold):
    """Calculate MCC at a given threshold."""
    # Convert probabilities to binary predictions
    predictions = (probabilities > threshold).astype(int)
    
    # Convert labels to binary (0 or 1)
    binary_labels = (labels > LABEL_THRESHOLD).astype(int)
    
    # Calculate confusion matrix
    TP = np.sum((predictions == 1) & (binary_labels == 1))
    TN = np.sum((predictions == 0) & (binary_labels == 0))
    FP = np.sum((predictions == 1) & (binary_labels == 0))
    FN = np.sum((predictions == 0) & (binary_labels == 1))
    
    # MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0.0:
        mcc = 0.0
    else:
        mcc = (TP * TN - FP * FN) / denominator
    
    return mcc


def find_best_mcc_threshold(probabilities, labels, thresholds):
    """Find threshold with best MCC."""
    mcc_values = []
    for threshold in thresholds:
        mcc = calculate_mcc_at_threshold(probabilities, labels, threshold)
        mcc_values.append(mcc)
    
    best_idx = np.argmax(mcc_values)
    best_threshold = thresholds[best_idx]
    best_mcc = mcc_values[best_idx]
    
    return thresholds, mcc_values, best_threshold, best_mcc


def plot_roc_curve_and_mcc(fpr, tpr, auc, thresholds, mcc_values, best_mcc_thresh, best_mcc, output_dir):
    """Plot ROC curve and MCC vs threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: ROC Curve
    ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier (AUC = 0.5)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: MCC vs Threshold
    ax2.plot(thresholds, mcc_values, 'g-', linewidth=2, label='MCC')
    ax2.plot(best_mcc_thresh, best_mcc, 'go', markersize=10, 
             label=f'Best MCC: {best_mcc:.4f} at threshold {best_mcc_thresh:.3f}')
    ax2.axvline(best_mcc_thresh, color='r', linestyle='--', alpha=0.5, label=f'Optimal Threshold: {best_mcc_thresh:.3f}')
    ax2.set_xlabel('Classification Threshold', fontsize=12)
    ax2.set_ylabel('Matthews Correlation Coefficient (MCC)', fontsize=12)
    ax2.set_title('MCC vs Classification Threshold', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(thresholds), max(thresholds)])
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'roc_curve_and_mcc.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    return output_file


def print_results(auc, best_mcc_thresh, best_mcc, probabilities, labels):
    """Print summary of results."""
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nAUC (Area Under ROC Curve): {auc:.4f}")
    if auc >= 0.9:
        interpretation = "Excellent"
    elif auc >= 0.8:
        interpretation = "Good"
    elif auc >= 0.7:
        interpretation = "Fair"
    else:
        interpretation = "Poor"
    print(f"Interpretation: {interpretation}")
    
    print(f"\nBest MCC Threshold: {best_mcc_thresh:.4f}")
    print(f"Best MCC Value: {best_mcc:.4f}")
    
    # Calculate metrics at best threshold
    binary_labels = (labels > LABEL_THRESHOLD).astype(int)
    predictions = (probabilities > best_mcc_thresh).astype(int)
    
    TP = np.sum((predictions == 1) & (binary_labels == 1))
    TN = np.sum((predictions == 0) & (binary_labels == 0))
    FP = np.sum((predictions == 1) & (binary_labels == 0))
    FN = np.sum((predictions == 0) & (binary_labels == 1))
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    print(f"\nMetrics at Best MCC Threshold ({best_mcc_thresh:.4f}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")


def main():
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device1)

    print("Preparing test split...")
    print(f"Using same split parameters as training: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}, seed={RANDOM_SEED}")
    train_ids, val_ids, test_ids = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print(f"Test set contains {len(test_ids)} sequences (not used during training)")
    data_set_test = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=test_ids, training=False)
    data_loader_test = DataLoader(
        dataset=data_set_test,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    if not os.path.isfile(FRONTAL_CKPT):
        raise FileNotFoundError(f"Frontal checkpoint not found: {FRONTAL_CKPT}")
    if not os.path.isfile(LATERAL_CKPT):
        raise FileNotFoundError(f"Lateral checkpoint not found: {LATERAL_CKPT}")

    print(f"Loading checkpoints:\n  Frontal: {FRONTAL_CKPT}\n  Lateral: {LATERAL_CKPT}")
    model_frontal = load_model(device1, FRONTAL_CKPT)
    model_lateral = load_model(device2, LATERAL_CKPT)

    # Get all predictions
    print("\nCollecting predictions from test set...")
    probabilities, labels = get_all_predictions(model_frontal, model_lateral, data_loader_test, device1, device2)
    
    print(f"\nCollected {len(probabilities)} predictions")
    print(f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    print(f"Label distribution: {np.sum(labels <= LABEL_THRESHOLD)} negative, {np.sum(labels > LABEL_THRESHOLD)} positive")
    
    # Calculate ROC curve and AUC
    print("\nCalculating ROC curve and AUC...")
    fpr, tpr, auc = calculate_roc_curve(probabilities, labels)
    
    if fpr is None:
        print("Error: Could not calculate ROC curve.")
        return
    
    print(f"AUC: {auc:.4f}")
    
    # Find best MCC threshold
    print("\nFinding best MCC threshold...")
    thresholds = np.arange(0.1, 0.95, 0.01)
    threshold_array, mcc_values, best_mcc_thresh, best_mcc = find_best_mcc_threshold(
        probabilities, labels, thresholds
    )
    
    print(f"Best MCC: {best_mcc:.4f} at threshold: {best_mcc_thresh:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_roc_curve_and_mcc(fpr, tpr, auc, threshold_array, mcc_values, 
                           best_mcc_thresh, best_mcc, SCRIPT_DIR)
    
    # Print summary
    print_results(auc, best_mcc_thresh, best_mcc, probabilities, labels)
    
    print("\n" + "="*70)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
