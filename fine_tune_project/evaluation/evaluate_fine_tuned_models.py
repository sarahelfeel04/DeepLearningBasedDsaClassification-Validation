#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the fine-tuned frontal and lateral models on the held-out test split.
Uses the same split logic as fine_tune_dsa.py to ensure consistency.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fine_tune_project.dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from fine_tune_project.utils.CnnLstmModel import CnnLstmModel
from fine_tune_project.evaluation.ModelEvaluation import ModelEvaluation

# -------------------------------------------------------------------------
# Configuration (MUST MATCH fine_tune_dsa.py for consistent test split)
# -------------------------------------------------------------------------
DATA_ROOT_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/Channel0-DataTypeUnsignedShort-Values0to4000"
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join("/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/DeepLearningBasedDsaClassification-Validation/fine_tune_project/fine_tuned_models", "china_data_unfrozen_cnn_20_epochs")
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


def save_results_to_csv(csv_results, output_path):
    """Save evaluation results to CSV file."""
    if not csv_results:
        print("Warning: No results to save to CSV.")
        return
    
    csv_file = os.path.join(output_path, "evaluation_results.csv")
    fieldnames = [
        'filename_frontal', 'filename_lateral', 'ground_truth', 'ground_truth_value',
        'prob_frontal', 'prob_lateral', 'prob_combined',
        'prediction_frontal', 'prediction_lateral', 'prediction_combined',
        'correct_combined'
    ]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_results)
    
    print(f"\nResults saved to: {csv_file}")
    return csv_file


def calculate_auc(probabilities, labels):
    """Calculate AUC (Area Under ROC Curve) using trapezoidal rule."""
    # Convert labels to binary (0 or 1)
    binary_labels = (labels > LABEL_THRESHOLD).astype(int)
    
    # Count positives and negatives
    n_positive = np.sum(binary_labels == 1)
    n_negative = np.sum(binary_labels == 0)
    
    if n_positive == 0 or n_negative == 0:
        return None  # Cannot calculate AUC with only one class
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_labels = binary_labels[sorted_indices]
    
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
    
    return auc


def evaluate(model_frontal, model_lateral, dataloader, device1, device2):
    loss_fn = nn.BCEWithLogitsLoss()
    eval_metrics = ModelEvaluation()
    combined_metrics = ModelEvaluation()  # For combined frontal+lateral predictions
    
    running_loss_frontal = 0.0
    running_loss_lateral = 0.0
    running_loss_combined = 0.0
    
    # Store all probabilities and labels for AUC calculation
    all_probs_combined = []
    all_labels = []
    
    # Store results for CSV export
    csv_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", unit="batch"):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)
            
            # Get file paths from batch (handle both list and single value cases)
            filename_frontal = batch["filename"][0] if isinstance(batch["filename"], (list, tuple)) else batch["filename"]
            filename_lateral = batch["filenameOtherView"][0] if isinstance(batch["filenameOtherView"], (list, tuple)) else batch["filenameOtherView"]

            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)

            running_loss_frontal += loss_fn(output_frontal, labels_frontal).item()
            running_loss_lateral += loss_fn(output_lateral, labels_lateral).item()

            # Combine outputs: average the probabilities
            prob_frontal = torch.sigmoid(output_frontal).item()
            prob_lateral = torch.sigmoid(output_lateral).item()
            prob_combined = (prob_frontal + prob_lateral) / 2.0
            
            # Store for AUC calculation
            all_probs_combined.append(prob_combined)
            all_labels.append(labels_frontal.item())
            
            # Convert combined probability back to logits for loss calculation
            # prob = sigmoid(logit) => logit = log(prob / (1 - prob))
            logit_combined = np.log(prob_combined / (1 - prob_combined + 1e-8))
            # Match the shape of labels_frontal which is [1, 1]
            logit_combined_tensor = torch.tensor([[logit_combined]], dtype=torch.float32).to(device1)
            running_loss_combined += loss_fn(logit_combined_tensor, labels_frontal).item()

            estimate_frontal = THROMBUS_NO if prob_frontal <= 0.55 else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if prob_lateral <= 0.55 else THROMBUS_YES
            estimate_combined = THROMBUS_NO if prob_combined <= 0.55 else THROMBUS_YES

            label_value = labels_frontal.item()
            is_thrombus_free = label_value <= LABEL_THRESHOLD
            
            # Store results for CSV
            csv_results.append({
                'filename_frontal': filename_frontal,
                'filename_lateral': filename_lateral,
                'ground_truth': 'Thrombus' if not is_thrombus_free else 'No Thrombus',
                'ground_truth_value': label_value,
                'prob_frontal': prob_frontal,
                'prob_lateral': prob_lateral,
                'prob_combined': prob_combined,
                'prediction_frontal': 'Thrombus' if estimate_frontal == THROMBUS_YES else 'No Thrombus',
                'prediction_lateral': 'Thrombus' if estimate_lateral == THROMBUS_YES else 'No Thrombus',
                'prediction_combined': 'Thrombus' if estimate_combined == THROMBUS_YES else 'No Thrombus',
                'correct_combined': 'Yes' if ((is_thrombus_free and estimate_combined == THROMBUS_NO) or 
                                             (not is_thrombus_free and estimate_combined == THROMBUS_YES)) else 'No'
            })

            # Individual model metrics
            if is_thrombus_free:
                eval_metrics.increaseTNfrontal() if estimate_frontal == THROMBUS_NO else eval_metrics.increaseFPfrontal()
                eval_metrics.increaseTNlateral() if estimate_lateral == THROMBUS_NO else eval_metrics.increaseFPlateral()
            else:
                eval_metrics.increaseTPfrontal() if estimate_frontal == THROMBUS_YES else eval_metrics.increaseFNfrontal()
                eval_metrics.increaseTPlateral() if estimate_lateral == THROMBUS_YES else eval_metrics.increaseFNlateral()
            
            # Combined model metrics (using frontal metrics structure for combined)
            if is_thrombus_free:
                combined_metrics.increaseTNfrontal() if estimate_combined == THROMBUS_NO else combined_metrics.increaseFPfrontal()
            else:
                combined_metrics.increaseTPfrontal() if estimate_combined == THROMBUS_YES else combined_metrics.increaseFNfrontal()

    n_batches = len(dataloader)
    avg_loss_frontal = running_loss_frontal / n_batches
    avg_loss_lateral = running_loss_lateral / n_batches
    avg_loss_combined = running_loss_combined / n_batches
    
    # Calculate AUC
    all_probs_combined = np.array(all_probs_combined)
    all_labels = np.array(all_labels)
    auc = calculate_auc(all_probs_combined, all_labels)
    
    return avg_loss_frontal, avg_loss_lateral, avg_loss_combined, eval_metrics, combined_metrics, auc, csv_results


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

    print("Evaluating on the test set...")
    test_loss_frontal, test_loss_lateral, test_loss_combined, metrics, combined_metrics, auc, csv_results = evaluate(
        model_frontal, model_lateral, data_loader_test, device1, device2
    )

    print("\n" + "="*60)
    print("INDIVIDUAL MODEL RESULTS:")
    print("="*60)
    print(f"Test Loss -> Frontal: {test_loss_frontal:.4f}, Lateral: {test_loss_lateral:.4f}")
    metrics.printAllStats()
    
    print("\n" + "="*60)
    print("COMBINED MODEL RESULTS (Frontal + Lateral averaged):")
    print("="*60)
    print(f"Test Loss (Combined): {test_loss_combined:.4f}")
    print(f"Accuracy: {combined_metrics.getAccuracyFrontal():.4f}")
    print(f"Precision: {combined_metrics.getPrecisionFrontal():.4f}")
    print(f"Recall: {combined_metrics.getRecallFrontal():.4f}")
    print(f"MCC: {combined_metrics.getMccFrontal():.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: Cannot calculate (only one class present in test set)")
    print(f"\nConfusion Matrix (Combined):")
    print(f"  TP: {combined_metrics.TP_frontal}, FP: {combined_metrics.FP_frontal}")
    print(f"  TN: {combined_metrics.TN_frontal}, FN: {combined_metrics.FN_frontal}")
    
    # Save results to CSV
    save_results_to_csv(csv_results, OUTPUT_PATH)


if __name__ == "__main__":
    main()

