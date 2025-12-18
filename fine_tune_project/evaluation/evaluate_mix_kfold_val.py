#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run validation for mixed K-fold fine-tuning (China + German) using the
saved best-MCC checkpoints per fold from fine_tune_mix_kfold.py.

For each fold this script:
  - Reconstructs the same validation split (China + German) used during training
  - Loads the saved frontal and lateral checkpoints for that fold
  - Runs inference on the validation set on a SINGLE device (GPU if available)
  - Computes:
        * AUC for frontal, lateral, and combined (mean of probabilities)
        * Accuracy, Precision, Recall (Sensitivity), Specificity, MCC
        * Confusion matrix (TP, TN, FP, FN) for each view and combined
        * Average classification time per sample per view
  - Saves per-sample validation predictions (including combined) to CSV per fold
  - Saves a per-fold summary CSV with all metrics and timings
"""

import os
import csv
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from tqdm import tqdm

from ..dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from ..utils.CnnLstmModel import CnnLstmModel
from ..GermanDataUtils import (
    GermanFineTuneDataset,
    load_german_sequences,
    split_german_data_by_patient,
)
from ..fine_tune_mix_kfold import (
    DATA_ROOT_PATH_CHINA,
    DATA_ROOT_PATH_GERMAN,
    GERMAN_ANNOTATIONS_CSV,
    MODEL_BASE_PATH,
    CHECKPOINT_FRONTAL_NAME,
    CHECKPOINT_LATERAL_NAME,
    OUTPUT_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    LABEL_THRESHOLD,
    N_FOLDS,
)


def get_device() -> torch.device:
    """
    Always use a SINGLE device (GPU if available, otherwise CPU) for both views.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    return torch.device("cpu")


def load_model(device: torch.device, checkpoint_rel_path: str) -> CnnLstmModel:
    """
    Load a CnnLstmModel and its weights from a checkpoint path relative to MODEL_BASE_PATH.
    This mirrors the logic in fine_tune_mix_kfold.load_and_configure_model, but for inference
    we only need to load the weights and move to the given device.
    """
    model = CnnLstmModel(512, 3, 1, True, device)
    checkpoint_path = os.path.join(MODEL_BASE_PATH, checkpoint_rel_path)
    print(f"Loading initial weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_fine_tuned_model_for_fold(
    device: torch.device,
    base_checkpoint_rel_path: str,
    fold_idx: int,
    view: str,
) -> CnnLstmModel:
    """
    Load the fine-tuned model for a given fold and view (frontal/lateral).

    We start from the architecture defined in CnnLstmModel, then load the
    fine-tuned checkpoint saved by fine_tune_mix_kfold.py.
    """
    # Instantiate model (same architecture)
    model = CnnLstmModel(512, 3, 1, True, device)

    # Determine path to fine-tuned checkpoint
    if view == "frontal":
        ckpt_name = f"frontal_fine_tuned_best_mcc_fold_{fold_idx + 1}.pt"
    else:
        ckpt_name = f"lateral_fine_tuned_best_mcc_fold_{fold_idx + 1}.pt"

    ckpt_path = os.path.join(OUTPUT_PATH, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")

    print(f"[Fold {fold_idx + 1}] Loading fine-tuned {view} model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_confusion_and_metrics(
    y_true: List[int],
    y_prob: List[float],
    threshold: float = 0.5,
) -> Tuple[float, float, float, float, float, int, int, int, int, float]:
    """
    Compute standard classification metrics given true labels and predicted probabilities.

    Returns:
        acc, prec, rec, sens, spec, tp, tn, fp, fn, mcc
    """
    if not y_true:
        return (0.0,) * 10

    # Predictions at threshold
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall == sensitivity
    sens = rec
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return acc, prec, rec, sens, spec, tp, tn, fp, fn, mcc


def eval_fold(
    fold_idx: int,
    val_ids_ch_fold,
    val_ids_de_fold,
    german_sequences,
    device: torch.device,
):
    """
    Evaluate a single fold's validation set using the saved best-MCC checkpoints.
    """
    print(f"\n========== Re-validating Fold {fold_idx + 1}/{N_FOLDS} ==========")

    # Build datasets exactly as in fine_tune_mix_kfold.run_fold (validation only)
    data_set_val_ch = FineTuneDsaDataset(
        DATA_ROOT_PATH_CHINA, data_subset=val_ids_ch_fold, training=False
    )
    data_set_val_de = GermanFineTuneDataset(
        german_sequences, val_ids_de_fold, DATA_ROOT_PATH_GERMAN, training=False
    )

    data_set_val = ConcatDataset([data_set_val_ch, data_set_val_de])
    n_val_ch = len(data_set_val_ch)
    n_val_total = len(data_set_val)

    data_loader_val = DataLoader(
        dataset=data_set_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Load fine-tuned models for this fold
    model_frontal = load_fine_tuned_model_for_fold(
        device, CHECKPOINT_FRONTAL_NAME, fold_idx, view="frontal"
    )
    model_lateral = load_fine_tuned_model_for_fold(
        device, CHECKPOINT_LATERAL_NAME, fold_idx, view="lateral"
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # Storage for metrics
    y_true: List[int] = []
    probs_f: List[float] = []
    probs_l: List[float] = []
    probs_comb: List[float] = []

    # For per-sample CSV
    rows: List[list] = []

    # Timing
    total_time_f = 0.0
    total_time_l = 0.0

    # Loss (optional)
    total_loss_f = 0.0
    total_loss_l = 0.0

    val_bar = tqdm(data_loader_val, desc=f"Fold {fold_idx + 1} Re-Validation", unit="sample")
    with torch.no_grad():
        for idx, batch in enumerate(val_bar):
            labels = batch["target_label"].to(device=device, dtype=torch.float)
            images_f = batch["image"].to(device=device, dtype=torch.float)
            images_l = batch["imageOtherView"].to(device=device, dtype=torch.float)

            # Optional file info for CSV
            filename_f = batch.get("filename", [""])[0]
            filename_l = batch.get("filenameOtherView", [""])[0]

            # Domain: based on index in ConcatDataset (first n_val_ch are China)
            domain = "china" if idx < n_val_ch else "german"

            # Frontal timing and inference
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out_f = model_frontal(images_f)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            total_time_f += (t1 - t0)

            # Lateral timing and inference
            t2 = time.perf_counter()
            out_l = model_lateral(images_l)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t3 = time.perf_counter()
            total_time_l += (t3 - t2)

            # Compute losses (optional)
            loss_f = loss_fn(out_f, labels)
            loss_l = loss_fn(out_l, labels)
            total_loss_f += loss_f.item()
            total_loss_l += loss_l.item()

            # Probabilities
            prob_f = torch.sigmoid(out_f).item()
            prob_l = torch.sigmoid(out_l).item()
            prob_c = 0.5 * (prob_f + prob_l)

            # Ground-truth binary label using same convention as other scripts
            label_scalar = labels.item()
            y_bin = 1 if label_scalar > LABEL_THRESHOLD else 0

            y_true.append(y_bin)
            probs_f.append(prob_f)
            probs_l.append(prob_l)
            probs_comb.append(prob_c)

            # Thresholded predictions
            pred_f = 1 if prob_f >= 0.5 else 0
            pred_l = 1 if prob_l >= 0.5 else 0
            pred_c = 1 if prob_c >= 0.5 else 0

            # Per-sample CSV row
            rows.append(
                [
                    fold_idx + 1,
                    idx,
                    domain,
                    label_scalar,
                    y_bin,
                    prob_f,
                    prob_l,
                    prob_c,
                    pred_f,
                    pred_l,
                    pred_c,
                    filename_f,
                    filename_l,
                ]
            )

    # AUCs (handle potential edge cases where only one class present)
    def safe_auc(y, p):
        try:
            if len(set(y)) < 2:
                return None
            return roc_auc_score(y, p)
        except Exception:
            return None

    auc_f = safe_auc(y_true, probs_f)
    auc_l = safe_auc(y_true, probs_l)
    auc_c = safe_auc(y_true, probs_comb)

    # Metrics per view and combined
    acc_f, prec_f, rec_f, sens_f, spec_f, tp_f, tn_f, fp_f, fn_f, mcc_f = compute_confusion_and_metrics(
        y_true, probs_f, threshold=0.5
    )
    acc_l, prec_l, rec_l, sens_l, spec_l, tp_l, tn_l, fp_l, fn_l, mcc_l = compute_confusion_and_metrics(
        y_true, probs_l, threshold=0.5
    )
    acc_c, prec_c, rec_c, sens_c, spec_c, tp_c, tn_c, fp_c, fn_c, mcc_c = compute_confusion_and_metrics(
        y_true, probs_comb, threshold=0.5
    )

    n_samples = len(y_true)
    avg_time_f = total_time_f / n_samples if n_samples > 0 else 0.0
    avg_time_l = total_time_l / n_samples if n_samples > 0 else 0.0

    avg_loss_f = total_loss_f / n_samples if n_samples > 0 else 0.0
    avg_loss_l = total_loss_l / n_samples if n_samples > 0 else 0.0

    print(f"[Fold {fold_idx + 1}] Val samples: {n_samples}")
    print(
        f"[Fold {fold_idx + 1}] Frontal: "
        f"AUC={auc_f if auc_f is not None else 'N/A'}, "
        f"ACC={acc_f:.4f}, MCC={mcc_f:.4f}, "
        f"Sens={sens_f:.4f}, Spec={spec_f:.4f}"
    )
    print(
        f"[Fold {fold_idx + 1}] Lateral: "
        f"AUC={auc_l if auc_l is not None else 'N/A'}, "
        f"ACC={acc_l:.4f}, MCC={mcc_l:.4f}, "
        f"Sens={sens_l:.4f}, Spec={spec_l:.4f}"
    )
    print(
        f"[Fold {fold_idx + 1}] Combined: "
        f"AUC={auc_c if auc_c is not None else 'N/A'}, "
        f"ACC={acc_c:.4f}, MCC={mcc_c:.4f}, "
        f"Sens={sens_c:.4f}, Spec={spec_c:.4f}"
    )
    print(
        f"[Fold {fold_idx + 1}] Avg classification time per sample: "
        f"Frontal={avg_time_f*1000:.3f} ms, Lateral={avg_time_l*1000:.3f} ms"
    )

    # Save per-sample predictions CSV
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    preds_path = os.path.join(OUTPUT_PATH, f"mixed_kfold_val_predictions_fold_{fold_idx + 1}.csv")
    with open(preds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
                "index_in_fold",
                "domain",
                "label_continuous",
                "label_binary",
                "prob_frontal",
                "prob_lateral",
                "prob_combined",
                "pred_frontal",
                "pred_lateral",
                "pred_combined",
                "filename_frontal",
                "filename_lateral",
            ]
        )
        writer.writerows(rows)

    # Return summary dict for aggregation
    return {
        "fold": fold_idx + 1,
        "n_samples": n_samples,
        "val_loss_frontal": avg_loss_f,
        "val_loss_lateral": avg_loss_l,
        "auc_frontal": auc_f,
        "auc_lateral": auc_l,
        "auc_combined": auc_c,
        "acc_frontal": acc_f,
        "prec_frontal": prec_f,
        "recall_frontal": rec_f,
        "sens_frontal": sens_f,
        "spec_frontal": spec_f,
        "mcc_frontal": mcc_f,
        "tp_frontal": tp_f,
        "tn_frontal": tn_f,
        "fp_frontal": fp_f,
        "fn_frontal": fn_f,
        "acc_lateral": acc_l,
        "prec_lateral": prec_l,
        "recall_lateral": rec_l,
        "sens_lateral": sens_l,
        "spec_lateral": spec_l,
        "mcc_lateral": mcc_l,
        "tp_lateral": tp_l,
        "tn_lateral": tn_l,
        "fp_lateral": fp_l,
        "fn_lateral": fn_l,
        "acc_combined": acc_c,
        "prec_combined": prec_c,
        "recall_combined": rec_c,
        "sens_combined": sens_c,
        "spec_combined": spec_c,
        "mcc_combined": mcc_c,
        "tp_combined": tp_c,
        "tn_combined": tn_c,
        "fp_combined": fp_c,
        "fn_combined": fn_c,
        "avg_time_frontal": avg_time_f,
        "avg_time_lateral": avg_time_l,
    }


def main():
    device = get_device()
    print(f"Using device for evaluation: {device}")

    # 1. Recreate initial CHINESE split
    print("Re-splitting CHINESE data sequences (70/15/15) with stratification...")
    train_ids_ch, val_ids_ch, test_ids_ch = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH_CHINA, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 2. Recreate GERMAN sequences + split
    print("Re-loading GERMAN annotations and rebuilding sequences...")
    german_sequences = load_german_sequences(DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV)
    print("Re-splitting GERMAN data sequences (70/15/15) by patient...")
    train_ids_de, val_ids_de, test_ids_de = split_german_data_by_patient(
        german_sequences, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 3. Combine train+val for K-fold (test sets remain fixed and unused here)
    trainval_ch = train_ids_ch + val_ids_ch
    trainval_de = train_ids_de + val_ids_de

    from sklearn.model_selection import KFold

    kf_ch = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    kf_de = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Convert to lists for indexing
    trainval_ch = list(trainval_ch)
    trainval_de = list(trainval_de)

    summaries = []

    for fold_idx, ((train_idx_ch, val_idx_ch), (train_idx_de, val_idx_de)) in enumerate(
        zip(kf_ch.split(trainval_ch), kf_de.split(trainval_de))
    ):
        # Build fold-specific val ID lists (same as training script)
        val_ids_ch_fold = [trainval_ch[i] for i in val_idx_ch]
        val_ids_de_fold = [trainval_de[i] for i in val_idx_de]

        summary = eval_fold(
            fold_idx=fold_idx,
            val_ids_ch_fold=val_ids_ch_fold,
            val_ids_de_fold=val_ids_de_fold,
            german_sequences=german_sequences,
            device=device,
        )
        summaries.append(summary)

    # Save per-fold summary CSV
    summary_path = os.path.join(OUTPUT_PATH, "mixed_kfold_val_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "fold",
            "n_samples",
            "val_loss_frontal",
            "val_loss_lateral",
            "auc_frontal",
            "auc_lateral",
            "auc_combined",
            "acc_frontal",
            "prec_frontal",
            "recall_frontal",
            "sens_frontal",
            "spec_frontal",
            "mcc_frontal",
            "tp_frontal",
            "tn_frontal",
            "fp_frontal",
            "fn_frontal",
            "acc_lateral",
            "prec_lateral",
            "recall_lateral",
            "sens_lateral",
            "spec_lateral",
            "mcc_lateral",
            "tp_lateral",
            "tn_lateral",
            "fp_lateral",
            "fn_lateral",
            "acc_combined",
            "prec_combined",
            "recall_combined",
            "sens_combined",
            "spec_combined",
            "mcc_combined",
            "tp_combined",
            "tn_combined",
            "fp_combined",
            "fn_combined",
            "avg_time_frontal",
            "avg_time_lateral",
        ]
        writer.writerow(header)
        for s in summaries:
            writer.writerow(
                [
                    s["fold"],
                    s["n_samples"],
                    s["val_loss_frontal"],
                    s["val_loss_lateral"],
                    s["auc_frontal"] if s["auc_frontal"] is not None else "N/A",
                    s["auc_lateral"] if s["auc_lateral"] is not None else "N/A",
                    s["auc_combined"] if s["auc_combined"] is not None else "N/A",
                    s["acc_frontal"],
                    s["prec_frontal"],
                    s["recall_frontal"],
                    s["sens_frontal"],
                    s["spec_frontal"],
                    s["mcc_frontal"],
                    s["tp_frontal"],
                    s["tn_frontal"],
                    s["fp_frontal"],
                    s["fn_frontal"],
                    s["acc_lateral"],
                    s["prec_lateral"],
                    s["recall_lateral"],
                    s["sens_lateral"],
                    s["spec_lateral"],
                    s["mcc_lateral"],
                    s["tp_lateral"],
                    s["tn_lateral"],
                    s["fp_lateral"],
                    s["fn_lateral"],
                    s["acc_combined"],
                    s["prec_combined"],
                    s["recall_combined"],
                    s["sens_combined"],
                    s["spec_combined"],
                    s["mcc_combined"],
                    s["tp_combined"],
                    s["tn_combined"],
                    s["fp_combined"],
                    s["fn_combined"],
                    s["avg_time_frontal"],
                    s["avg_time_lateral"],
                ]
            )

    print(f"\nPer-fold validation summary written to: {summary_path}")


if __name__ == "__main__":
    main()


