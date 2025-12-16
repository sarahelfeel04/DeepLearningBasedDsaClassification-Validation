#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed fine-tuning script (China + German) with K-fold cross-validation
on the combined train+val set. The held-out test set (from the initial
70/15/15 split) is left untouched for final evaluation.

This is based on fine_tune_mix.py, but wraps the training loop in a
K-fold loop over the train+val IDs.
"""

import os
import csv
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from sklearn.model_selection import KFold

from .dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from .utils.CnnLstmModel import CnnLstmModel
from .evaluation.ModelEvaluation import ModelEvaluation
from .GermanDataUtils import (
    GermanFineTuneDataset,
    load_german_sequences,
    split_german_data_by_patient,
)

# -------------------------------------------------------------------------
# Configuration (MUST MATCH other fine-tuning scripts where relevant)
# -------------------------------------------------------------------------

# 1. Chinese Data Root Path
DATA_ROOT_PATH_CHINA = "/media/nami/Volume/ThromboMap/datasets/FirstChannel-CorrectRange-uint16-reannotated"

# 2. German Data Root Path
DATA_ROOT_PATH_GERMAN = "/media/nami/FastDataSpace/ThromboMap-Validation/dataClinic2024"
GERMAN_ANNOTATIONS_CSV = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/final_annotations_deutsch_2024.csv"
)

# 3. Model Checkpoint Paths (initial pre-trained weights)
MODEL_BASE_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/Classificator/Models"
CHECKPOINT_FRONTAL_NAME = "frontal/final_model.pt"
CHECKPOINT_LATERAL_NAME = "lateral/final_model.pt"

# 4. Output Path for mixed fine-tuning with K-fold
OUTPUT_PATH = "./fine_tuned_models/china_german_mixed_kfold/"

# Hyperparameters
FINE_TUNE_LR = 5e-6     # Low Learning Rate for Fine-Tuning (recommended when unfreezing CNN)
EPOCHS = 20
BATCH_SIZE = 1
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2  # Midpoint between the two label encodings

# K-fold settings
N_FOLDS = 5
# If you already completed some folds, you can skip them by setting this.
# 0-based index: e.g. set to 1 to start from Fold 2/5.
FOLD_START = 2

# --- Device Setup ---
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
if device2.type == "cpu":
    print("Warning: Only one GPU or none detected. Both models will use device:", device1)
    device2 = device1

# -------------------------------------------------------------------------
# Warning filtering / deprecation clean-up
# -------------------------------------------------------------------------

# Suppress Albumentations API warnings about ShiftScaleRotate/Downscale
warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform.*",
)
warnings.filterwarnings(
    "ignore",
    message="Argument\\(s\\) 'value' are not valid for transform ShiftScaleRotate",
)
warnings.filterwarnings(
    "ignore",
    message="Argument\\(s\\) 'scale_min, scale_max' are not valid for transform Downscale",
)


def load_and_configure_model(device, checkpoint_name: str) -> CnnLstmModel:
    """Loads pre-trained weights and ensures all layers, including the CNN, are trainable."""
    model = CnnLstmModel(512, 3, 1, True, device)

    checkpoint_path = os.path.join(MODEL_BASE_PATH, checkpoint_name)
    print(f"Attempting to load checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if checkpoint has a nested 'model_state_dict' key (as seen in original trainCnnLstm.py)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    # Unfreeze ALL layers for full fine-tuning
    print(f"Unfreezing ALL layers for full fine-tuning on device {device}...")
    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    return model


def run_fold(
    fold_idx: int,
    train_ids_ch_fold,
    val_ids_ch_fold,
    train_ids_de_fold,
    val_ids_de_fold,
    test_ids_ch,
    test_ids_de,
    german_sequences,
):
    """Run one K-fold training/validation cycle."""
    print(f"\n========== Fold {fold_idx + 1}/{N_FOLDS} ==========")

    # Datasets for this fold
    data_set_train_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=train_ids_ch_fold, training=True)
    data_set_val_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=val_ids_ch_fold, training=False)

    data_set_train_de = GermanFineTuneDataset(german_sequences, train_ids_de_fold, DATA_ROOT_PATH_GERMAN, training=True)
    data_set_val_de = GermanFineTuneDataset(german_sequences, val_ids_de_fold, DATA_ROOT_PATH_GERMAN, training=False)

    data_set_train = ConcatDataset([data_set_train_ch, data_set_train_de])
    data_set_val = ConcatDataset([data_set_val_ch, data_set_val_de])

    dataLoaderTrain = DataLoader(
        dataset=data_set_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    dataLoaderVal = DataLoader(
        dataset=data_set_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    # Fresh models per fold
    model_frontal = load_and_configure_model(device1, CHECKPOINT_FRONTAL_NAME)
    model_lateral = load_and_configure_model(device2, CHECKPOINT_LATERAL_NAME)

    optimizer_frontal = optim.AdamW(model_frontal.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
    scheduler_frontal = optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, "min", factor=0.1, patience=5)
    loss_function_frontal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

    optimizer_lateral = optim.AdamW(model_lateral.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
    scheduler_lateral = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lateral, "min", factor=0.1, patience=5)
    loss_function_lateral = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

    loss_function_validation = nn.BCEWithLogitsLoss()

    # Use new torch.amp.GradScaler API to avoid deprecation warnings
    scaler_frontal = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else torch.amp.GradScaler()
    scaler_lateral = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else torch.amp.GradScaler()

    modelEvaluationVal = ModelEvaluation()
    best_mcc_frontal = -1.0
    best_mcc_lateral = -1.0

    # Training loop (same as fine_tune_mix, but per fold)
    for epoch in range(EPOCHS):
        model_frontal.train()
        model_lateral.train()
        running_loss_frontal = 0.0
        running_loss_lateral = 0.0
        epoch_running_loss_frontal = 0.0
        epoch_running_loss_lateral = 0.0

        train_bar = tqdm(dataLoaderTrain, desc=f"Fold {fold_idx + 1} Epoch {epoch+1}/{EPOCHS} Training", unit="batch")

        for step, batch in enumerate(train_bar):
            labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
            images_frontal = batch["image"].to(device=device1, dtype=torch.float)

            labels_lateral = batch["target_label"].to(device=device2, dtype=torch.float)
            images_lateral = batch["imageOtherView"].to(device=device2, dtype=torch.float)

            optimizer_frontal.zero_grad()
            optimizer_lateral.zero_grad()

            with torch.autocast(device_type=device1.type, dtype=torch.float16):
                output_frontal = model_frontal(images_frontal)

            with torch.autocast(device_type=device2.type, dtype=torch.float16):
                output_lateral = model_lateral(images_lateral)

            loss_frontal = loss_function_frontal(output_frontal, labels_frontal)
            loss_lateral = loss_function_lateral(output_lateral, labels_lateral)

            scaler_frontal.scale(loss_frontal).backward()
            scaler_lateral.scale(loss_lateral).backward()

            scaler_frontal.step(optimizer_frontal)
            scaler_lateral.step(optimizer_lateral)

            scaler_frontal.update()
            scaler_lateral.update()

            running_loss_frontal += loss_frontal.detach().item()
            running_loss_lateral += loss_lateral.detach().item()

            step_loss_f = loss_frontal.detach().item()
            step_loss_l = loss_lateral.detach().item()

            epoch_running_loss_frontal += step_loss_f
            epoch_running_loss_lateral += step_loss_l

            train_bar.set_postfix(
                Loss_F=f"{epoch_running_loss_frontal / (step + 1):.4f}",
                Loss_L=f"{epoch_running_loss_lateral / (step + 1):.4f}",
                LR=optimizer_frontal.param_groups[0]["lr"],
            )
            train_bar.refresh()

        avg_train_loss_frontal = running_loss_frontal / len(dataLoaderTrain)
        avg_train_loss_lateral = running_loss_lateral / len(dataLoaderTrain)
        scheduler_frontal.step(avg_train_loss_frontal)
        scheduler_lateral.step(avg_train_loss_lateral)

        print(
            f"\n[Fold {fold_idx + 1}] --- Epoch {epoch+1}/{EPOCHS} ---\n"
            f"Train Loss (Mixed): Frontal={avg_train_loss_frontal:.4f}, Lateral={avg_train_loss_lateral:.4f}"
        )

        # Validation
        model_frontal.eval()
        model_lateral.eval()
        modelEvaluationVal.reset()
        validation_loss_frontal = 0.0
        validation_loss_lateral = 0.0

        val_bar = tqdm(dataLoaderVal, desc=f"Fold {fold_idx + 1} Validation", unit="batch")

        with torch.no_grad():
            for batch in val_bar:
                labels_frontal = batch["target_label"].to(device=device1, dtype=torch.float)
                images_frontal_val = batch["image"].to(device=device1, dtype=torch.float)
                labels_lateral_device = batch["target_label"].to(device=device2, dtype=torch.float)
                images_lateral_val = batch["imageOtherView"].to(device=device2, dtype=torch.float)

                output_frontal = model_frontal(images_frontal_val)
                output_lateral = model_lateral(images_lateral_val)

                current_val_loss_f = loss_function_validation(output_frontal, labels_frontal).item()
                current_val_loss_l = loss_function_validation(output_lateral, labels_lateral_device).item()

                validation_loss_frontal += current_val_loss_f
                validation_loss_lateral += current_val_loss_l

                estimate_frontal = (
                    THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
                )
                estimate_lateral = (
                    THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES
                )

                label_value = labels_frontal.item()
                is_thrombus_free = label_value <= LABEL_THRESHOLD

                if is_thrombus_free:
                    modelEvaluationVal.increaseTNfrontal() if estimate_frontal == THROMBUS_NO else modelEvaluationVal.increaseFPfrontal()
                    modelEvaluationVal.increaseTNlateral() if estimate_lateral == THROMBUS_NO else modelEvaluationVal.increaseFPlateral()
                else:
                    modelEvaluationVal.increaseTPfrontal() if estimate_frontal == THROMBUS_YES else modelEvaluationVal.increaseFNfrontal()
                    modelEvaluationVal.increaseTPlateral() if estimate_lateral == THROMBUS_YES else modelEvaluationVal.increaseFNlateral()

                val_bar.set_postfix(
                    Loss_F=f"{current_val_loss_f:.4f}",
                    Loss_L=f"{current_val_loss_l:.4f}",
                )
                val_bar.refresh()

        avg_val_loss_frontal = validation_loss_frontal / len(dataLoaderVal)
        avg_val_loss_lateral = validation_loss_lateral / len(dataLoaderVal)
        print(
            f"[Fold {fold_idx + 1}] Validation Loss (Mixed): "
            f"Frontal={avg_val_loss_frontal:.4f}, Lateral={avg_val_loss_lateral:.4f}"
        )

        # Compute metrics for logging
        acc_front = modelEvaluationVal.getAccuracyFrontal()
        prec_front = modelEvaluationVal.getPrecisionFrontal()
        rec_front = modelEvaluationVal.getRecallFrontal()
        mcc_front = modelEvaluationVal.getMccFrontal()

        acc_lat = modelEvaluationVal.getAccuracyLateral()
        prec_lat = modelEvaluationVal.getPrecisionLateral()
        rec_lat = modelEvaluationVal.getRecallLateral()
        mcc_lat = modelEvaluationVal.getMccLateral()

        modelEvaluationVal.printAllStats()

        # Save best models per fold based on MCC
        current_mcc_frontal = mcc_front
        if current_mcc_frontal > best_mcc_frontal:
            best_mcc_frontal = current_mcc_frontal
            print(f"[Fold {fold_idx + 1}] New best frontal MCC: {best_mcc_frontal:.4f}. Saving model.")
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": model_frontal.state_dict(),
                    "mcc": best_mcc_frontal,
                },
                os.path.join(OUTPUT_PATH, f"frontal_fine_tuned_best_mcc_fold_{fold_idx + 1}.pt"),
            )

        current_mcc_lateral = mcc_lat
        if current_mcc_lateral > best_mcc_lateral:
            best_mcc_lateral = current_mcc_lateral
            print(f"[Fold {fold_idx + 1}] New best lateral MCC: {best_mcc_lateral:.4f}. Saving model.")
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": model_lateral.state_dict(),
                    "mcc": best_mcc_lateral,
                },
                os.path.join(OUTPUT_PATH, f"lateral_fine_tuned_best_mcc_fold_{fold_idx + 1}.pt"),
            )

        # Log metrics to CSV for this fold/epoch
        metrics_file = os.path.join(OUTPUT_PATH, f"metrics_fold_{fold_idx + 1}.csv")
        file_exists = os.path.isfile(metrics_file)
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "fold",
                        "epoch",
                        "train_loss_frontal",
                        "train_loss_lateral",
                        "val_loss_frontal",
                        "val_loss_lateral",
                        "acc_front",
                        "prec_front",
                        "recall_front",
                        "mcc_front",
                        "acc_lat",
                        "prec_lat",
                        "recall_lat",
                        "mcc_lat",
                    ]
                )
            writer.writerow(
                [
                    fold_idx + 1,
                    epoch + 1,
                    avg_train_loss_frontal,
                    avg_train_loss_lateral,
                    avg_val_loss_frontal,
                    avg_val_loss_lateral,
                    acc_front,
                    prec_front,
                    rec_front,
                    mcc_front,
                    acc_lat,
                    prec_lat,
                    rec_lat,
                    mcc_lat,
                ]
            )

    print(
        f"\n[Fold {fold_idx + 1}] Finished. "
        f"Best MCC: Frontal={best_mcc_frontal:.4f}, Lateral={best_mcc_lateral:.4f}"
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. Initial CHINESE split (train/val/test)
    print("1. Splitting CHINESE data sequences (70/15/15) with stratification...")
    train_ids_ch, val_ids_ch, test_ids_ch = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH_CHINA, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 2. Initial GERMAN split (train/val/test) by patient
    print("2. Loading GERMAN annotations and building sequences...")
    german_sequences = load_german_sequences(DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV)
    print("3. Splitting GERMAN data sequences (70/15/15) by patient...")
    train_ids_de, val_ids_de, test_ids_de = split_german_data_by_patient(
        german_sequences, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 3. Combine train+val for K-fold (test sets remain fixed and unused here)
    trainval_ch = train_ids_ch + val_ids_ch
    trainval_de = train_ids_de + val_ids_de

    print(
        "4. K-fold setup (Mixed China + German):\n"
        f"   China train+val sequences: {len(trainval_ch)}, test: {len(test_ids_ch)}\n"
        f"   German train+val sequences: {len(trainval_de)}, test: {len(test_ids_de)}\n"
        f"   K folds: {N_FOLDS}"
    )

    # K-fold split indices separately for China and German
    kf_ch = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    kf_de = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Convert IDs to lists for indexing
    trainval_ch = list(trainval_ch)
    trainval_de = list(trainval_de)

    # zip over folds for China and German (same number of folds)
    for fold_idx, ((train_idx_ch, val_idx_ch), (train_idx_de, val_idx_de)) in enumerate(
        zip(kf_ch.split(trainval_ch), kf_de.split(trainval_de))
    ):
        # Optionally skip some folds if they were completed in a previous run
        if fold_idx < FOLD_START:
            continue
        train_ids_ch_fold = [trainval_ch[i] for i in train_idx_ch]
        val_ids_ch_fold = [trainval_ch[i] for i in val_idx_ch]

        train_ids_de_fold = [trainval_de[i] for i in train_idx_de]
        val_ids_de_fold = [trainval_de[i] for i in val_idx_de]

        run_fold(
            fold_idx,
            train_ids_ch_fold,
            val_ids_ch_fold,
            train_ids_de_fold,
            val_ids_de_fold,
            test_ids_ch,
            test_ids_de,
            german_sequences,
        )

    print(
        "\nK-fold mixed fine-tuning complete. "
        "Per-fold best models are saved in the output directory. "
        "Use your evaluation script to assess them on the fixed test sets."
    )


