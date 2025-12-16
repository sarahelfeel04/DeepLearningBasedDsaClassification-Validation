#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate K-fold fine-tuned CHINESE models on the held-out test set.

Logic is based on the original evaluate_fine_tuned_models.py:
- Uses same split (FineTuneDsaDataset.split_data with same ratios/seed).
- Loads frontal and lateral checkpoints, averages probabilities for a
  combined prediction, and computes metrics + AUC.

This script:
1) Evaluates each fold's best-MCC frontal/lateral checkpoints independently
   on the same Chinese test set, saving per-fold metrics and per-sample
   predictions.
2) Evaluates an ensemble across folds by averaging probabilities from all
   fold models, and saves ensemble metrics and per-sample predictions.
"""

import os
import csv
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from fine_tune_project.dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from fine_tune_project.utils.CnnLstmModel import CnnLstmModel
from fine_tune_project.evaluation.ModelEvaluation import ModelEvaluation

# -------------------------------------------------------------------------
# Configuration (aligns with fine_tune_dsa_kfold)
# -------------------------------------------------------------------------
DATA_ROOT_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/FirstChannel-CorrectRange-uint16-reannotated"

K_FOLD_MODELS_PATH = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/fine_tuned_models/china_kfold"
)

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
    threshold: float = 0.55,
    fold: int | None = None,
    output_dir: str | None = None,
):
    """
    Evaluate one frontal/lateral pair on the Chinese test set.

    Records:
      - aggregate metrics (losses, ACC/PREC/REC/MCC for frontal/lateral/combined, AUC)
      - per-sample predictions and inference time (ms) in a CSV:
        china_kfold_predictions_fold_{fold}.csv (if fold and output_dir are provided)
    """
    loss_fn = nn.BCEWithLogitsLoss()
    eval_metrics = ModelEvaluation()
    combined_metrics = ModelEvaluation()  # For combined frontal+lateral predictions

    running_loss_frontal = 0.0
    running_loss_lateral = 0.0
    running_loss_combined = 0.0

    # Store all probabilities and labels for AUC calculation
    all_probs_combined = []
    all_labels = []

    # Per-sample CSV rows
    per_sample_rows = []
    total_time = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", unit="batch"):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            # Filenames for bookkeeping
            filename_frontal = batch["filename"][0] if isinstance(batch["filename"], (list, tuple)) else batch["filename"]
            filename_lateral = batch["filenameOtherView"][0] if isinstance(batch["filenameOtherView"], (list, tuple)) else batch["filenameOtherView"]

            # Time forward passes
            if device1.type == "cuda":
                torch.cuda.synchronize(device1)
            if device2.type == "cuda":
                torch.cuda.synchronize(device2)
            t0 = time.perf_counter()

            output_frontal = model_frontal(images_frontal)
            output_lateral = model_lateral(images_lateral)

            if device1.type == "cuda":
                torch.cuda.synchronize(device1)
            if device2.type == "cuda":
                torch.cuda.synchronize(device2)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0

            running_loss_frontal += loss_fn(output_frontal, labels_frontal).item()
            running_loss_lateral += loss_fn(output_lateral, labels_lateral).item()

            # Combine outputs: average the probabilities
            prob_frontal = torch.sigmoid(output_frontal).item()
            prob_lateral = torch.sigmoid(output_lateral).item()
            prob_combined = (prob_frontal + prob_lateral) / 2.0

            all_probs_combined.append(prob_combined)
            all_labels.append(labels_frontal.item())

            # Convert combined probability back to logits for loss calculation
            logit_combined = np.log(prob_combined / (1 - prob_combined + 1e-8))
            logit_combined_tensor = torch.tensor([[logit_combined]], dtype=torch.float32).to(device1)
            running_loss_combined += loss_fn(logit_combined_tensor, labels_frontal).item()

            estimate_frontal = THROMBUS_NO if prob_frontal <= threshold else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if prob_lateral <= threshold else THROMBUS_YES
            estimate_combined = THROMBUS_NO if prob_combined <= threshold else THROMBUS_YES

            label_value = labels_frontal.item()
            is_thrombus_free = label_value <= LABEL_THRESHOLD

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

            total_time += elapsed_ms
            n_samples += 1

            per_sample_rows.append(
                {
                    "filename_frontal": filename_frontal,
                    "filename_lateral": filename_lateral,
                    "label_value": label_value,
                    "prob_frontal": prob_frontal,
                    "prob_lateral": prob_lateral,
                    "prob_combined": prob_combined,
                    "prediction_frontal": 1 if estimate_frontal == THROMBUS_YES else 0,
                    "prediction_lateral": 1 if estimate_lateral == THROMBUS_YES else 0,
                    "prediction_combined": 1 if estimate_combined == THROMBUS_YES else 0,
                }
            )

    n_batches = len(dataloader)
    avg_loss_frontal = running_loss_frontal / n_batches
    avg_loss_lateral = running_loss_lateral / n_batches
    avg_loss_combined = running_loss_combined / n_batches

    all_probs_combined = np.array(all_probs_combined)
    all_labels = np.array(all_labels)
    auc = calculate_auc(all_probs_combined, all_labels)

    avg_time_ms = total_time / max(n_samples, 1)

    # Save per-sample CSV if requested
    if fold is not None and output_dir is not None:
        pred_file = os.path.join(output_dir, f"china_kfold_predictions_fold_{fold}.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename_frontal",
                    "filename_lateral",
                    "label_value",
                    "prob_frontal",
                    "prob_lateral",
                    "prob_combined",
                    "prediction_frontal",
                    "prediction_lateral",
                    "prediction_combined",
                ],
            )
            writer.writeheader()
            writer.writerows(per_sample_rows)
        print(f"Per-sample predictions for fold {fold} saved to: {pred_file}")

    # Individual metrics
    acc_front = eval_metrics.getAccuracyFrontal()
    prec_front = eval_metrics.getPrecisionFrontal()
    rec_front = eval_metrics.getRecallFrontal()
    mcc_front = eval_metrics.getMccFrontal()

    acc_lat = eval_metrics.getAccuracyLateral()
    prec_lat = eval_metrics.getPrecisionLateral()
    rec_lat = eval_metrics.getRecallLateral()
    mcc_lat = eval_metrics.getMccLateral()

    # Combined confusion for summary
    tp_c = combined_metrics.TP_frontal
    tn_c = combined_metrics.TN_frontal
    fp_c = combined_metrics.FP_frontal
    fn_c = combined_metrics.FN_frontal
    acc_comb, sens_comb, spec_comb, mcc_comb = compute_confusion_metrics(tp_c, tn_c, fp_c, fn_c)

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
        "acc_combined": acc_comb,
        "sens_combined": sens_comb,
        "spec_combined": spec_comb,
        "mcc_combined": mcc_comb,
        "tp_combined": tp_c,
        "tn_combined": tn_c,
        "fp_combined": fp_c,
        "fn_combined": fn_c,
    }


def evaluate_ensemble(
    fold_models,
    dataloader,
    device1,
    device2,
    threshold: float = 0.55,
    output_dir: str | None = None,
):
    """
    Evaluate an ensemble of fold models by averaging probabilities across folds.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    combined_metrics = ModelEvaluation()

    running_loss_combined = 0.0
    all_probs_combined = []
    all_labels = []

    per_sample_rows = []
    total_time = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing (Ensemble)", unit="batch"):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)
            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            filename_frontal = batch["filename"][0] if isinstance(batch["filename"], (list, tuple)) else batch["filename"]
            filename_lateral = batch["filenameOtherView"][0] if isinstance(batch["filenameOtherView"], (list, tuple)) else batch["filenameOtherView"]

            if device1.type == "cuda":
                torch.cuda.synchronize(device1)
            if device2.type == "cuda":
                torch.cuda.synchronize(device2)
            t0 = time.perf_counter()

            probs_frontal = []
            probs_lateral = []

            for model_frontal, model_lateral in fold_models:
                output_frontal = model_frontal(images_frontal)
                output_lateral = model_lateral(images_lateral)
                probs_frontal.append(torch.sigmoid(output_frontal).item())
                probs_lateral.append(torch.sigmoid(output_lateral).item())

            if device1.type == "cuda":
                torch.cuda.synchronize(device1)
            if device2.type == "cuda":
                torch.cuda.synchronize(device2)
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

            total_time += elapsed_ms
            n_samples += 1
            per_sample_rows.append(
                {
                    "filename_frontal": filename_frontal,
                    "filename_lateral": filename_lateral,
                    "label_value": label_value,
                    "prob_frontal": prob_frontal,
                    "prob_lateral": prob_lateral,
                    "prob_combined": prob_combined,
                    "prediction_combined": 1 if estimate_combined == THROMBUS_YES else 0,
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

    tp_c = combined_metrics.TP_frontal
    tn_c = combined_metrics.TN_frontal
    fp_c = combined_metrics.FP_frontal
    fn_c = combined_metrics.FN_frontal
    _, sens_comb, spec_comb, _ = compute_confusion_metrics(tp_c, tn_c, fp_c, fn_c)

    avg_time_ms = total_time / max(n_samples, 1)

    if output_dir is not None:
        pred_file = os.path.join(output_dir, "china_kfold_ensemble_predictions.csv")
        with open(pred_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename_frontal",
                    "filename_lateral",
                    "label_value",
                    "prob_frontal",
                    "prob_lateral",
                    "prob_combined",
                    "prediction_combined",
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
        "sens_combined": sens_comb,
        "spec_combined": spec_comb,
        "tp_combined": tp_c,
        "tn_combined": tn_c,
        "fp_combined": fp_c,
        "fn_combined": fn_c,
    }


def main():
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else device1)

    # 1. Build test split (same logic as training)
    print("Preparing CHINESE test split...")
    train_ids, val_ids, test_ids = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    print(f"Test set contains {len(test_ids)} sequences (not used during K-fold training)")

    data_set_test = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=test_ids, training=False)
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
            threshold=0.55,
            fold=fold,
            output_dir=K_FOLD_MODELS_PATH,
        )
        fold_result["fold"] = fold
        per_fold_results.append(fold_result)
        fold_models_for_ensemble.append((model_frontal, model_lateral))

        auc_str = f"{fold_result['auc']:.4f}" if fold_result["auc"] is not None else "N/A"
        print(
            f"Fold {fold} -> "
            f"LossF={fold_result['avg_loss_frontal']:.4f}, LossL={fold_result['avg_loss_lateral']:.4f}, "
            f"LossComb={fold_result['avg_loss_combined']:.4f}, "
            f"ACC_F={fold_result['acc_front']:.4f}, PREC_F={fold_result['prec_front']:.4f}, "
            f"REC_F={fold_result['rec_front']:.4f}, MCC_F={fold_result['mcc_front']:.4f}, "
            f"ACC_L={fold_result['acc_lat']:.4f}, PREC_L={fold_result['prec_lat']:.4f}, "
            f"REC_L={fold_result['rec_lat']:.4f}, MCC_L={fold_result['mcc_lat']:.4f}, "
            f"ACC_Comb={fold_result['acc_combined']:.4f}, SEN_Comb={fold_result['sens_combined']:.4f}, "
            f"SPEC_Comb={fold_result['spec_combined']:.4f}, MCC_Comb={fold_result['mcc_combined']:.4f}, "
            f"AUC={auc_str}, avg_time_ms={fold_result['avg_time_ms']:.2f}"
        )

    # Save per-fold summary CSV
    if per_fold_results:
        summary_file = os.path.join(K_FOLD_MODELS_PATH, "china_kfold_test_summary.csv")
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
                    "acc_combined",
                    "sens_combined",
                    "spec_combined",
                    "mcc_combined",
                    "tp_combined",
                    "tn_combined",
                    "fp_combined",
                    "fn_combined",
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
                    ]
                )
        print(f"\nPer-fold Chinese test summary saved to: {summary_file}")

    # ------------------------------------------------------------------
    # Approach 2: Ensemble evaluation (average probabilities across folds)
    # ------------------------------------------------------------------
    if fold_models_for_ensemble:
        print("\n=== Evaluating Chinese Ensemble (averaged probabilities across folds) ===")
        ensemble_result = evaluate_ensemble(
            fold_models_for_ensemble,
            data_loader_test,
            device1,
            device2,
            threshold=0.55,
            output_dir=K_FOLD_MODELS_PATH,
        )
        ens_auc_str = f"{ensemble_result['auc']:.4f}" if ensemble_result["auc"] is not None else "N/A"
        print(
            f"Ensemble -> LossComb={ensemble_result['avg_loss_combined']:.4f}, "
            f"ACC={ensemble_result['acc']:.4f}, PREC={ensemble_result['prec']:.4f}, "
            f"REC={ensemble_result['rec']:.4f}, MCC={ensemble_result['mcc']:.4f}, "
            f"SEN={ensemble_result['sens_combined']:.4f}, SPEC={ensemble_result['spec_combined']:.4f}, "
            f"AUC={ens_auc_str}, avg_time_ms={ensemble_result['avg_time_ms']:.2f}"
        )

        # Save ensemble summary
        ensemble_summary_file = os.path.join(K_FOLD_MODELS_PATH, "china_kfold_ensemble_summary.csv")
        with open(ensemble_summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "loss_combined",
                    "acc",
                    "prec",
                    "rec",
                    "mcc",
                    "auc",
                    "avg_time_ms",
                    "sens_combined",
                    "spec_combined",
                    "tp_combined",
                    "tn_combined",
                    "fp_combined",
                    "fn_combined",
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
                ]
            )
        print(f"Ensemble summary saved to: {ensemble_summary_file}")


if __name__ == "__main__":
    main()


