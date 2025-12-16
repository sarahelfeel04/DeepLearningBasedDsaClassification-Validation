#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate K-fold mixed (China + German) fine-tuned models on the held-out test set.

Two approaches:
1) Per-fold evaluation: load each fold's best-MCC frontal/lateral checkpoints,
   evaluate them independently on the SAME mixed test set, and summarize metrics.
2) Ensemble evaluation: for each test sample, average probabilities across all folds
   (ensemble of fold models) and evaluate once.
"""

import os
import csv
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from fine_tune_project.dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from fine_tune_project.utils.CnnLstmModel import CnnLstmModel
from fine_tune_project.evaluation.ModelEvaluation import ModelEvaluation
from fine_tune_project.GermanDataUtils import (
    GermanFineTuneDataset,
    load_german_sequences,
    split_german_data_by_patient,
)

# -------------------------------------------------------------------------
# Configuration (MUST MATCH fine_tune_mix_kfold.py for consistent splits)
# -------------------------------------------------------------------------

# Chinese data
DATA_ROOT_PATH_CHINA = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/Channel0-DataTypeUnsignedShort-Values0to4000"

# German data
DATA_ROOT_PATH_GERMAN = "/media/nami/Volume/ThromboMap/dataClinic2024"
GERMAN_ANNOTATIONS_CSV = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/final_annotations_deutsch_2024.csv"
)

# K-fold models location (must match OUTPUT_PATH in fine_tune_mix_kfold.py)
K_FOLD_MODELS_PATH = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/fine_tuned_models/china_german_mixed_kfold"
)

# Split ratios and seed (must match training)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2

TEST_BATCH_SIZE = 1
NUM_WORKERS = 4
N_FOLDS = 5


def load_model(device, checkpoint_path: str) -> CnnLstmModel:
    """Load model weights from checkpoint onto the given device."""
    model = CnnLstmModel(512, 3, 1, True, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def calculate_auc(probabilities, labels):
    """Calculate AUC (Area Under ROC Curve) using sklearn's roc_auc_score."""
    binary_labels = (labels > LABEL_THRESHOLD).astype(int)
    # Need both classes present
    if np.unique(binary_labels).size < 2:
        return None
    try:
        return float(roc_auc_score(binary_labels, probabilities))
    except ValueError:
        return None


def compute_confusion_metrics(tp: int, tn: int, fp: int, fn: int):
    """Return (accuracy, sensitivity, specificity, mcc) from confusion counts."""
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if tp + fp > 0 and tp + fn > 0 and tn + fp > 0 and tn + fn > 0 else 0.0
    if denom == 0.0:
        mcc = 0.0
    else:
        mcc = ((tp * tn) - (fp * fn)) / denom
    return acc, sens, spec, mcc


def evaluate_single_pair(
    model_frontal,
    model_lateral,
    dataloader,
    device1,
    device2,
    threshold: float = 0.50,
    n_ch_samples: int | None = None,
    fold: int | None = None,
    output_dir: str | None = None,
):
    """
    Evaluate one frontal/lateral pair on the mixed test set.

    Records:
      - aggregate metrics (losses, ACC/PREC/REC/MCC, AUC)
      - per-sample predictions and inference time (ms) in a CSV:
        kfold_predictions_fold_{fold}.csv (if fold and output_dir are provided)
    """
    loss_fn = nn.BCEWithLogitsLoss()
    eval_metrics = ModelEvaluation()
    combined_metrics = ModelEvaluation()

    running_loss_frontal = 0.0
    running_loss_lateral = 0.0
    running_loss_combined = 0.0

    all_probs_combined = []
    all_labels = []

    # For timing and per-sample CSV
    per_sample_rows = []
    total_time = 0.0
    n_samples = 0

    # Domain-split confusion for combined classifier
    tp_ch = tn_ch = fp_ch = fn_ch = 0
    tp_de = tn_de = fp_de = fn_de = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing", unit="batch")):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            # Measure per-sample inference time (synchronize for accurate GPU timing)
            torch.cuda.synchronize(device1) if device1.type == "cuda" else None
            torch.cuda.synchronize(device2) if device2.type == "cuda" else None
            t0 = time.perf_counter()

            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)

            torch.cuda.synchronize(device1) if device1.type == "cuda" else None
            torch.cuda.synchronize(device2) if device2.type == "cuda" else None
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            running_loss_frontal += loss_fn(output_frontal, labels_frontal).item()
            running_loss_lateral += loss_fn(output_lateral, labels_lateral).item()

            prob_frontal = torch.sigmoid(output_frontal).item()
            prob_lateral = torch.sigmoid(output_lateral).item()
            prob_combined = (prob_frontal + prob_lateral) / 2.0

            all_probs_combined.append(prob_combined)
            all_labels.append(labels_frontal.item())

            # For combined loss, convert prob back to logit
            logit_combined = np.log(prob_combined / (1 - prob_combined + 1e-8))
            logit_combined_tensor = torch.tensor([[logit_combined]], dtype=torch.float32).to(device1)
            running_loss_combined += loss_fn(logit_combined_tensor, labels_frontal).item()

            estimate_frontal = THROMBUS_NO if prob_frontal <= threshold else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if prob_lateral <= threshold else THROMBUS_YES
            estimate_combined = THROMBUS_NO if prob_combined <= threshold else THROMBUS_YES

            label_value = labels_frontal.item()
            is_thrombus_free = label_value <= LABEL_THRESHOLD

            if is_thrombus_free:
                eval_metrics.increaseTNfrontal() if estimate_frontal == THROMBUS_NO else eval_metrics.increaseFPfrontal()
                eval_metrics.increaseTNlateral() if estimate_lateral == THROMBUS_NO else eval_metrics.increaseFPlateral()
            else:
                eval_metrics.increaseTPfrontal() if estimate_frontal == THROMBUS_YES else eval_metrics.increaseFNfrontal()
                eval_metrics.increaseTPlateral() if estimate_lateral == THROMBUS_YES else eval_metrics.increaseFNlateral()

            if is_thrombus_free:
                combined_metrics.increaseTNfrontal() if estimate_combined == THROMBUS_NO else combined_metrics.increaseFPfrontal()
            else:
                combined_metrics.increaseTPfrontal() if estimate_combined == THROMBUS_YES else combined_metrics.increaseFNfrontal()

            # Domain-specific confusion (combined)
            is_ch = (n_ch_samples is not None) and (batch_idx < n_ch_samples)
            if is_thrombus_free:
                if estimate_combined == THROMBUS_NO:
                    if is_ch:
                        tn_ch += 1
                    else:
                        tn_de += 1
                else:
                    if is_ch:
                        fp_ch += 1
                    else:
                        fp_de += 1
            else:
                if estimate_combined == THROMBUS_YES:
                    if is_ch:
                        tp_ch += 1
                    else:
                        tp_de += 1
                else:
                    if is_ch:
                        fn_ch += 1
                    else:
                        fn_de += 1

            # Track timing and basic prediction info per sample
            # Determine domain string for CSV
            domain_str = "china" if is_ch else "german"

            # Extract filenames if present in batch
            fname_f = ""
            fname_l = ""
            if "filename" in batch:
                v = batch["filename"]
                if isinstance(v, (list, tuple)):
                    fname_f = v[0]
                else:
                    fname_f = v
            if "filenameOtherView" in batch:
                v2 = batch["filenameOtherView"]
                if isinstance(v2, (list, tuple)):
                    fname_l = v2[0]
                else:
                    fname_l = v2

            total_time += elapsed_ms
            n_samples += 1
            per_sample_rows.append(
                {
                    "index": batch_idx,
                    "label_value": label_value,
                    "label_binary": 0 if is_thrombus_free else 1,
                    "prob_frontal": prob_frontal,
                    "prob_lateral": prob_lateral,
                    "prob_combined": prob_combined,
                    "pred_frontal": 1 if estimate_frontal == THROMBUS_YES else 0,
                    "pred_lateral": 1 if estimate_lateral == THROMBUS_YES else 0,
                    "pred_combined": 1 if estimate_combined == THROMBUS_YES else 0,
                    "correct_combined": int(
                        (is_thrombus_free and estimate_combined == THROMBUS_NO)
                        or (not is_thrombus_free and estimate_combined == THROMBUS_YES)
                    ),
                    "domain": domain_str,
                    "filename_frontal": fname_f,
                    "filename_lateral": fname_l,
                    "time_ms": elapsed_ms,
                }
            )

    n_batches = len(dataloader)
    avg_loss_frontal = running_loss_frontal / n_batches
    avg_loss_lateral = running_loss_lateral / n_batches
    avg_loss_combined = running_loss_combined / n_batches

    all_probs_combined = np.array(all_probs_combined)
    all_labels = np.array(all_labels)
    auc = calculate_auc(all_probs_combined, all_labels)

    # Per-domain AUC using split by index
    auc_ch = auc_de = None
    if n_ch_samples is not None:
        probs_ch = all_probs_combined[:n_ch_samples]
        labels_ch = all_labels[:n_ch_samples]
        probs_de = all_probs_combined[n_ch_samples:]
        labels_de = all_labels[n_ch_samples:]
        if probs_ch.size > 0:
            auc_ch = calculate_auc(probs_ch, labels_ch)
        if probs_de.size > 0:
            auc_de = calculate_auc(probs_de, labels_de)

    avg_time_ms = total_time / max(n_samples, 1)

    # Save per-sample predictions & timing to CSV if requested
    if fold is not None and output_dir is not None:
        pred_file = os.path.join(output_dir, f"kfold_predictions_fold_{fold}.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "label_value",
                    "label_binary",
                    "prob_frontal",
                    "prob_lateral",
                    "prob_combined",
                    "pred_frontal",
                    "pred_lateral",
                    "pred_combined",
                    "correct_combined",
                    "domain",
                    "filename_frontal",
                    "filename_lateral",
                    "time_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"Per-sample predictions for fold {fold} saved to: {pred_file}")

    # Extract metrics
    acc_front = eval_metrics.getAccuracyFrontal()
    prec_front = eval_metrics.getPrecisionFrontal()
    rec_front = eval_metrics.getRecallFrontal()
    mcc_front = eval_metrics.getMccFrontal()

    acc_lat = eval_metrics.getAccuracyLateral()
    prec_lat = eval_metrics.getPrecisionLateral()
    rec_lat = eval_metrics.getRecallLateral()
    mcc_lat = eval_metrics.getMccLateral()

    # Global combined confusion from ModelEvaluation
    tp_g = combined_metrics.TP_frontal
    tn_g = combined_metrics.TN_frontal
    fp_g = combined_metrics.FP_frontal
    fn_g = combined_metrics.FN_frontal
    acc_comb, sens_comb, spec_comb, mcc_comb = compute_confusion_metrics(tp_g, tn_g, fp_g, fn_g)

    # Domain-split metrics for combined predictions
    acc_ch, sens_ch, spec_ch, mcc_ch = compute_confusion_metrics(tp_ch, tn_ch, fp_ch, fn_ch)
    acc_de, sens_de, spec_de, mcc_de = compute_confusion_metrics(tp_de, tn_de, fp_de, fn_de)

    return {
        "avg_loss_frontal": avg_loss_frontal,
        "avg_loss_lateral": avg_loss_lateral,
        "avg_loss_combined": avg_loss_combined,
        "acc_front": acc_front,
        "prec_front": prec_front,
        "rec_front": rec_front,
        "mcc_front": mcc_front,
        "acc_lat": acc_lat,
        "prec_lat": prec_lat,
        "rec_lat": rec_lat,
        "mcc_lat": mcc_lat,
        "auc": auc,
        "avg_time_ms": avg_time_ms,
        # Combined (all test samples)
        "acc_combined": acc_comb,
        "sens_combined": sens_comb,
        "spec_combined": spec_comb,
        "mcc_combined": mcc_comb,
        "tp_combined": tp_g,
        "tn_combined": tn_g,
        "fp_combined": fp_g,
        "fn_combined": fn_g,
        # Chinese domain (combined)
        "acc_ch": acc_ch,
        "sens_ch": sens_ch,
        "spec_ch": spec_ch,
        "mcc_ch": mcc_ch,
        "auc_ch": auc_ch,
        "tp_ch": tp_ch,
        "tn_ch": tn_ch,
        "fp_ch": fp_ch,
        "fn_ch": fn_ch,
        # German domain (combined)
        "acc_de": acc_de,
        "sens_de": sens_de,
        "spec_de": spec_de,
        "mcc_de": mcc_de,
        "auc_de": auc_de,
        "tp_de": tp_de,
        "tn_de": tn_de,
        "fp_de": fp_de,
        "fn_de": fn_de,
    }


def evaluate_ensemble(
    fold_models,
    dataloader,
    device1,
    device2,
    threshold: float = 0.50,
    output_dir: str | None = None,
    n_ch_samples: int | None = None,
):
    """
    Evaluate an ensemble of fold models by averaging probabilities across folds.

    Also logs per-sample predictions and timing to kfold_ensemble_predictions.csv
    if output_dir is provided.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    eval_metrics = ModelEvaluation()
    combined_metrics = ModelEvaluation()

    running_loss_combined = 0.0
    all_probs_combined = []
    all_labels = []

    per_sample_rows = []
    total_time = 0.0
    n_samples = 0

    tp_ch = tn_ch = fp_ch = fn_ch = 0
    tp_de = tn_de = fp_de = fn_de = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing (Ensemble)", unit="batch")):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            torch.cuda.synchronize(device1) if device1.type == "cuda" else None
            torch.cuda.synchronize(device2) if device2.type == "cuda" else None
            t0 = time.perf_counter()

            probs_frontal = []
            probs_lateral = []

            for model_frontal, model_lateral in fold_models:
                output_frontal = model_frontal(images_frontal)
                output_lateral = model_lateral(images_lateral)
                probs_frontal.append(torch.sigmoid(output_frontal).item())
                probs_lateral.append(torch.sigmoid(output_lateral).item())

            torch.cuda.synchronize(device1) if device1.type == "cuda" else None
            torch.cuda.synchronize(device2) if device2.type == "cuda" else None
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            prob_frontal = float(np.mean(probs_frontal))
            prob_lateral = float(np.mean(probs_lateral))
            prob_combined = (prob_frontal + prob_lateral) / 2.0

            all_probs_combined.append(prob_combined)
            all_labels.append(labels_frontal.item())

            logit_combined = np.log(prob_combined / (1 - prob_combined + 1e-8))
            logit_combined_tensor = torch.tensor([[logit_combined]], dtype=torch.float32).to(device1)
            running_loss_combined += loss_fn(logit_combined_tensor, labels_frontal).item()

            estimate_combined = THROMBUS_NO if prob_combined <= threshold else THROMBUS_YES

            label_value = labels_frontal.item()
            is_thrombus_free = label_value <= LABEL_THRESHOLD

            if is_thrombus_free:
                combined_metrics.increaseTNfrontal() if estimate_combined == THROMBUS_NO else combined_metrics.increaseFPfrontal()
            else:
                combined_metrics.increaseTPfrontal() if estimate_combined == THROMBUS_YES else combined_metrics.increaseFNfrontal()

            # Domain-split confusion for ensemble (combined only)
            is_ch = (n_ch_samples is not None) and (batch_idx < n_ch_samples)
            if is_thrombus_free:
                if estimate_combined == THROMBUS_NO:
                    if is_ch:
                        tn_ch += 1
                    else:
                        tn_de += 1
                else:
                    if is_ch:
                        fp_ch += 1
                    else:
                        fp_de += 1
            else:
                if estimate_combined == THROMBUS_YES:
                    if is_ch:
                        tp_ch += 1
                    else:
                        tp_de += 1
                else:
                    if is_ch:
                        fn_ch += 1
                    else:
                        fn_de += 1

            # For CSV: domain + filenames
            domain_str = "china" if is_ch else "german"
            fname_f = ""
            fname_l = ""
            if "filename" in batch:
                v = batch["filename"]
                if isinstance(v, (list, tuple)):
                    fname_f = v[0]
                else:
                    fname_f = v
            if "filenameOtherView" in batch:
                v2 = batch["filenameOtherView"]
                if isinstance(v2, (list, tuple)):
                    fname_l = v2[0]
                else:
                    fname_l = v2

            total_time += elapsed_ms
            n_samples += 1
            per_sample_rows.append(
                {
                    "index": batch_idx,
                    "label_value": label_value,
                    "label_binary": 0 if is_thrombus_free else 1,
                    "prob_frontal": prob_frontal,
                    "prob_lateral": prob_lateral,
                    "prob_combined": prob_combined,
                    "pred_combined": 1 if estimate_combined == THROMBUS_YES else 0,
                    "correct_combined": int(
                        (is_thrombus_free and estimate_combined == THROMBUS_NO)
                        or (not is_thrombus_free and estimate_combined == THROMBUS_YES)
                    ),
                    "domain": domain_str,
                    "filename_frontal": fname_f,
                    "filename_lateral": fname_l,
                    "time_ms": elapsed_ms,
                }
            )

    n_batches = len(dataloader)
    avg_loss_combined = running_loss_combined / n_batches

    all_probs_combined = np.array(all_probs_combined)
    all_labels = np.array(all_labels)
    auc = calculate_auc(all_probs_combined, all_labels)

    acc = combined_metrics.getAccuracyFrontal()
    prec = combined_metrics.getPrecisionFrontal()
    rec = combined_metrics.getRecallFrontal()
    mcc = combined_metrics.getMccFrontal()

    # Combined global confusion
    tp_g = combined_metrics.TP_frontal
    tn_g = combined_metrics.TN_frontal
    fp_g = combined_metrics.FP_frontal
    fn_g = combined_metrics.FN_frontal
    _, sens_comb, spec_comb, _ = compute_confusion_metrics(tp_g, tn_g, fp_g, fn_g)

    # Domain-split metrics for ensemble combined
    acc_ch, sens_ch, spec_ch, mcc_ch = compute_confusion_metrics(tp_ch, tn_ch, fp_ch, fn_ch)
    acc_de, sens_de, spec_de, mcc_de = compute_confusion_metrics(tp_de, tn_de, fp_de, fn_de)

    # Per-domain AUC using split by index
    auc_ch = auc_de = None
    if n_ch_samples is not None:
        probs_ch = all_probs_combined[:n_ch_samples]
        labels_ch = all_labels[:n_ch_samples]
        probs_de = all_probs_combined[n_ch_samples:]
        labels_de = all_labels[n_ch_samples:]
        if probs_ch.size > 0:
            auc_ch = calculate_auc(probs_ch, labels_ch)
        if probs_de.size > 0:
            auc_de = calculate_auc(probs_de, labels_de)

    avg_time_ms = total_time / max(n_samples, 1)

    if output_dir is not None:
        pred_file = os.path.join(output_dir, "kfold_ensemble_predictions.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "label_value",
                    "label_binary",
                    "prob_frontal",
                    "prob_lateral",
                    "prob_combined",
                    "pred_combined",
                    "correct_combined",
                    "domain",
                    "filename_frontal",
                    "filename_lateral",
                    "time_ms",
                ],
            )
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"Per-sample predictions for ensemble saved to: {pred_file}")

    return {
        "avg_loss_combined": avg_loss_combined,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "mcc": mcc,
        "auc": auc,
        "avg_time_ms": avg_time_ms,
        # combined global
        "sens_combined": sens_comb,
        "spec_combined": spec_comb,
        "tp_combined": tp_g,
        "tn_combined": tn_g,
        "fp_combined": fp_g,
        "fn_combined": fn_g,
        # Chinese domain
        "acc_ch": acc_ch,
        "sens_ch": sens_ch,
        "spec_ch": spec_ch,
        "mcc_ch": mcc_ch,
        "auc_ch": auc_ch,
        "tp_ch": tp_ch,
        "tn_ch": tn_ch,
        "fp_ch": fp_ch,
        "fn_ch": fn_ch,
        # German domain
        "acc_de": acc_de,
        "sens_de": sens_de,
        "spec_de": spec_de,
        "mcc_de": mcc_de,
        "auc_de": auc_de,
        "tp_de": tp_de,
        "tn_de": tn_de,
        "fp_de": fp_de,
        "fn_de": fn_de,
    }


def main():
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device1)

    # 1. Build consistent Chinese test split
    print("Preparing CHINESE test split...")
    train_ids_ch, val_ids_ch, test_ids_ch = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH_CHINA, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 2. Build consistent German test split
    print("Loading GERMAN sequences and preparing test split...")
    german_sequences = load_german_sequences(DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV)
    train_ids_de, val_ids_de, test_ids_de = split_german_data_by_patient(
        german_sequences, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 3. Mixed test dataset (China + German)
    data_set_test_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=test_ids_ch, training=False)
    data_set_test_de = GermanFineTuneDataset(german_sequences, test_ids_de, DATA_ROOT_PATH_GERMAN, training=False)
    data_set_test = ConcatDataset([data_set_test_ch, data_set_test_de])

    data_loader_test = DataLoader(
        dataset=data_set_test,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ------------------------------------------------------------------
    # Approach 1: Per-fold evaluation
    # ------------------------------------------------------------------
    per_fold_results = []
    fold_models_for_ensemble = []

    for fold in range(1, N_FOLDS + 1):
        frontal_ckpt = os.path.join(
            K_FOLD_MODELS_PATH, f"frontal_fine_tuned_best_mcc_fold_{fold}.pt"
        )
        lateral_ckpt = os.path.join(
            K_FOLD_MODELS_PATH, f"lateral_fine_tuned_best_mcc_fold_{fold}.pt"
        )

        if not (os.path.isfile(frontal_ckpt) and os.path.isfile(lateral_ckpt)):
            print(f"[Fold {fold}] Checkpoints not found, skipping.")
            continue

        print(f"\n=== Evaluating Fold {fold} ===")
        print(f"Frontal checkpoint: {frontal_ckpt}")
        print(f"Lateral checkpoint: {lateral_ckpt}")

        model_frontal = load_model(device1, frontal_ckpt)
        model_lateral = load_model(device2, lateral_ckpt)

        fold_result = evaluate_single_pair(
            model_frontal,
            model_lateral,
            data_loader_test,
            device1,
            device2,
            threshold=0.50,
            n_ch_samples=len(test_ids_ch),
            fold=fold,
            output_dir=K_FOLD_MODELS_PATH,
        )
        fold_result["fold"] = fold
        per_fold_results.append(fold_result)
        fold_models_for_ensemble.append((model_frontal, model_lateral))

        # Safely format optional AUCs
        auc_global_str = f"{fold_result['auc']:.4f}" if fold_result["auc"] is not None else "N/A"
        auc_ch_str = f"{fold_result['auc_ch']:.4f}" if fold_result["auc_ch"] is not None else "N/A"
        auc_de_str = f"{fold_result['auc_de']:.4f}" if fold_result["auc_de"] is not None else "N/A"

        print(
            f"Fold {fold} -> MCC_front={fold_result['mcc_front']:.4f}, "
            f"MCC_lat={fold_result['mcc_lat']:.4f}, "
            f"AUC_global={auc_global_str}, "
            f"AUC_CN={auc_ch_str}, "
            f"AUC_DE={auc_de_str}, "
            f"ACC_CN={fold_result['acc_ch']:.4f}, "
            f"SEN_CN={fold_result['sens_ch']:.4f}, "
            f"SPEC_CN={fold_result['spec_ch']:.4f}, "
            f"ACC_DE={fold_result['acc_de']:.4f}, "
            f"SEN_DE={fold_result['sens_de']:.4f}, "
            f"SPEC_DE={fold_result['spec_de']:.4f}, "
            f"avg_time_per_sample_ms={fold_result['avg_time_ms']:.2f}"
        )

    # Save per-fold summary
    if per_fold_results:
        summary_file = os.path.join(K_FOLD_MODELS_PATH, "kfold_test_summary.csv")
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "fold",
                    "avg_loss_frontal",
                    "avg_loss_lateral",
                    "avg_loss_combined",
                    "acc_front",
                    "prec_front",
                    "rec_front",
                    "mcc_front",
                    "acc_lat",
                    "prec_lat",
                    "rec_lat",
                    "mcc_lat",
                    "auc",
                    "avg_time_ms",
                    # combined all
                    "acc_combined",
                    "sens_combined",
                    "spec_combined",
                    "mcc_combined",
                    "tp_combined",
                    "tn_combined",
                    "fp_combined",
                    "fn_combined",
                    # Chinese domain (combined)
                    "acc_ch",
                    "sens_ch",
                    "spec_ch",
                    "mcc_ch",
                    "auc_ch",
                    "tp_ch",
                    "tn_ch",
                    "fp_ch",
                    "fn_ch",
                    # German domain (combined)
                    "acc_de",
                    "sens_de",
                    "spec_de",
                    "mcc_de",
                    "auc_de",
                    "tp_de",
                    "tn_de",
                    "fp_de",
                    "fn_de",
                ]
            )
            for r in per_fold_results:
                writer.writerow(
                    [
                        r["fold"],
                        r["avg_loss_frontal"],
                        r["avg_loss_lateral"],
                        r["avg_loss_combined"],
                        r["acc_front"],
                        r["prec_front"],
                        r["rec_front"],
                        r["mcc_front"],
                        r["acc_lat"],
                        r["prec_lat"],
                        r["rec_lat"],
                        r["mcc_lat"],
                        r["auc"] if r["auc"] is not None else "",
                        r["avg_time_ms"],
                        r["acc_combined"],
                        r["sens_combined"],
                        r["spec_combined"],
                        r["mcc_combined"],
                        r["tp_combined"],
                        r["tn_combined"],
                        r["fp_combined"],
                        r["fn_combined"],
                        r["acc_ch"],
                        r["sens_ch"],
                        r["spec_ch"],
                        r["mcc_ch"],
                        r["auc_ch"] if r["auc_ch"] is not None else "",
                        r["tp_ch"],
                        r["tn_ch"],
                        r["fp_ch"],
                        r["fn_ch"],
                        r["acc_de"],
                        r["sens_de"],
                        r["spec_de"],
                        r["mcc_de"],
                        r["auc_de"] if r["auc_de"] is not None else "",
                        r["tp_de"],
                        r["tn_de"],
                        r["fp_de"],
                        r["fn_de"],
                    ]
                )
        print(f"\nPer-fold test summary saved to: {summary_file}")

        # Compute simple averages of metrics across folds
        mcc_fronts = [r["mcc_front"] for r in per_fold_results]
        mcc_lats = [r["mcc_lat"] for r in per_fold_results]
        aucs = [r["auc"] for r in per_fold_results if r["auc"] is not None]
        times = [r["avg_time_ms"] for r in per_fold_results]

        print("\n=== Per-fold MCC, AUC and domain metrics (Test set) ===")
        for r in per_fold_results:
            auc_global_str = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
            auc_ch_str = f"{r['auc_ch']:.4f}" if r["auc_ch"] is not None else "N/A"
            auc_de_str = f"{r['auc_de']:.4f}" if r["auc_de"] is not None else "N/A"
            print(
                f"Fold {r['fold']}: MCC_front={r['mcc_front']:.4f}, "
                f"MCC_lat={r['mcc_lat']:.4f}, "
                f"AUC_global={auc_global_str}, "
                f"AUC_CN={auc_ch_str}, "
                f"AUC_DE={auc_de_str}, "
                f"ACC_CN={r['acc_ch']:.4f}, SEN_CN={r['sens_ch']:.4f}, SPEC_CN={r['spec_ch']:.4f}, "
                f"ACC_DE={r['acc_de']:.4f}, SEN_DE={r['sens_de']:.4f}, SPEC_DE={r['spec_de']:.4f}, "
                f"avg_time_ms={r['avg_time_ms']:.2f}"
            )

        print(
            f"\nAverage MCC_front={np.mean(mcc_fronts):.4f} ± {np.std(mcc_fronts):.4f}, "
            f"Average MCC_lat={np.mean(mcc_lats):.4f} ± {np.std(mcc_lats):.4f}"
        )
        if aucs:
            print(f"Average AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

        print(
            f"Average time per sample={np.mean(times):.2f} ms ± {np.std(times):.2f} ms"
        )

        # Choose "best" fold (e.g., by average of frontal & lateral MCC)
        best_fold = max(
            per_fold_results,
            key=lambda r: 0.5 * (r["mcc_front"] + r["mcc_lat"]),
        )
        print(
            f"\nBest fold by mean MCC(front,lat): Fold {best_fold['fold']} "
            f"(MCC_front={best_fold['mcc_front']:.4f}, MCC_lat={best_fold['mcc_lat']:.4f})"
        )

    # ------------------------------------------------------------------
    # Approach 2: Ensemble evaluation (average probabilities across folds)
    # ------------------------------------------------------------------
    if fold_models_for_ensemble:
        print("\n=== Evaluating Ensemble of folds (averaged probabilities) ===")
        ensemble_result = evaluate_ensemble(
            fold_models_for_ensemble,
            data_loader_test,
            device1,
            device2,
            threshold=0.50,
            output_dir=K_FOLD_MODELS_PATH,
            n_ch_samples=len(test_ids_ch),
        )
        ens_auc_str = f"{ensemble_result['auc']:.4f}" if ensemble_result["auc"] is not None else "N/A"
        ens_auc_ch_str = f"{ensemble_result['auc_ch']:.4f}" if ensemble_result["auc_ch"] is not None else "N/A"
        ens_auc_de_str = f"{ensemble_result['auc_de']:.4f}" if ensemble_result["auc_de"] is not None else "N/A"
        print(
            f"Ensemble -> Loss_combined={ensemble_result['avg_loss_combined']:.4f}, "
            f"Accuracy={ensemble_result['acc']:.4f}, "
            f"Precision={ensemble_result['prec']:.4f}, "
            f"Recall={ensemble_result['rec']:.4f}, "
            f"MCC={ensemble_result['mcc']:.4f}, "
            f"AUC={ens_auc_str}, "
            f"AUC_CN={ens_auc_ch_str}, "
            f"AUC_DE={ens_auc_de_str}, "
            f"ACC_CN={ensemble_result['acc_ch']:.4f}, SEN_CN={ensemble_result['sens_ch']:.4f}, SPEC_CN={ensemble_result['spec_ch']:.4f}, "
            f"ACC_DE={ensemble_result['acc_de']:.4f}, SEN_DE={ensemble_result['sens_de']:.4f}, SPEC_DE={ensemble_result['spec_de']:.4f}, "
            f"avg_time_per_sample_ms={ensemble_result['avg_time_ms']:.2f}"
        )

        # Save ensemble summary as CSV alongside kfold_test_summary
        ensemble_summary_file = os.path.join(K_FOLD_MODELS_PATH, "kfold_ensemble_summary.csv")
        with open(ensemble_summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "loss_combined",
                    "acc_global",
                    "prec_global",
                    "rec_global",
                    "mcc_global",
                    "auc_global",
                    "avg_time_ms",
                    "sens_combined",
                    "spec_combined",
                    "tp_combined",
                    "tn_combined",
                    "fp_combined",
                    "fn_combined",
                    # Chinese domain
                    "acc_ch",
                    "sens_ch",
                    "spec_ch",
                    "mcc_ch",
                    "auc_ch",
                    "tp_ch",
                    "tn_ch",
                    "fp_ch",
                    "fn_ch",
                    # German domain
                    "acc_de",
                    "sens_de",
                    "spec_de",
                    "mcc_de",
                    "auc_de",
                    "tp_de",
                    "tn_de",
                    "fp_de",
                    "fn_de",
                ]
            )
            writer.writerow(
                [
                    ensemble_result["avg_loss_combined"],
                    ensemble_result["acc"],
                    ensemble_result["prec"],
                    ensemble_result["rec"],
                    ensemble_result["mcc"],
                    ensemble_result["auc"] if ensemble_result["auc"] is not None else "",
                    ensemble_result["avg_time_ms"],
                    ensemble_result["sens_combined"],
                    ensemble_result["spec_combined"],
                    ensemble_result["tp_combined"],
                    ensemble_result["tn_combined"],
                    ensemble_result["fp_combined"],
                    ensemble_result["fn_combined"],
                    ensemble_result["acc_ch"],
                    ensemble_result["sens_ch"],
                    ensemble_result["spec_ch"],
                    ensemble_result["mcc_ch"],
                    ensemble_result["auc_ch"] if ensemble_result["auc_ch"] is not None else "",
                    ensemble_result["tp_ch"],
                    ensemble_result["tn_ch"],
                    ensemble_result["fp_ch"],
                    ensemble_result["fn_ch"],
                    ensemble_result["acc_de"],
                    ensemble_result["sens_de"],
                    ensemble_result["spec_de"],
                    ensemble_result["mcc_de"],
                    ensemble_result["auc_de"] if ensemble_result["auc_de"] is not None else "",
                    ensemble_result["tp_de"],
                    ensemble_result["tn_de"],
                    ensemble_result["fp_de"],
                    ensemble_result["fn_de"],
                ]
            )
        print(f"Ensemble summary saved to: {ensemble_summary_file}")


if __name__ == "__main__":
    main()


