#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-fold variant of fine_tune_dsa.py.
Logic, hyperparameters, and metrics are identical to fine_tune_dsa; the only
change is that train+val are split into K folds (test split kept separate)
and we train/evaluate on each fold, saving per-epoch metrics and best-MCC
checkpoints per fold.
"""

import os
import csv
from contextlib import nullcontext

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from .dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from .utils.CnnLstmModel import CnnLstmModel
from .evaluation.ModelEvaluation import ModelEvaluation

# --- Configuration (copied from fine_tune_dsa.py) ---
DATA_ROOT_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/FirstChannel-CorrectRange-uint16-reannotated"
MODEL_BASE_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/Classificator/Models"
CHECKPOINT_FRONTAL_NAME = "frontal/final_model.pt"
CHECKPOINT_LATERAL_NAME = "lateral/final_model.pt"
OUTPUT_PATH = "./fine_tuned_models/china_kfold/"

FINE_TUNE_LR = 5e-6
EPOCHS = 20
BATCH_SIZE = 1
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2

# K-fold setting
N_FOLDS = 5

# --- Device Setup (same as fine_tune_dsa) ---
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
if device2.type == 'cpu':
    print("Warning: Only one GPU or none detected. Both models will use device:", device1)
    device2 = device1

# --- Model Loading and Configuration (UNFREEZE CNN) ---
def load_and_configure_model(device, checkpoint_name):
    model = CnnLstmModel(512, 3, 1, True, device)
    checkpoint_path = os.path.join(MODEL_BASE_PATH, checkpoint_name)
    print(f"Attempting to load checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = True
    model.to(device)
    return model

def autocast_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()

def train_one_epoch(model_frontal, model_lateral, dataLoaderTrain,
                    optimizer_frontal, optimizer_lateral,
                    loss_function_frontal, loss_function_lateral,
                    scaler_frontal, scaler_lateral):
    model_frontal.train()
    model_lateral.train()
    running_loss_frontal = 0.0
    running_loss_lateral = 0.0
    epoch_running_loss_frontal = 0.0
    epoch_running_loss_lateral = 0.0

    train_bar = tqdm(dataLoaderTrain, desc="Training", unit="batch")
    for step, batch in enumerate(train_bar):
        labels_frontal = batch['target_label'].to(device=device1, dtype=torch.float)
        images_frontal = batch['image'].to(device=device1, dtype=torch.float)
        labels_lateral = batch['target_label'].to(device=device2, dtype=torch.float)
        images_lateral = batch['imageOtherView'].to(device=device2, dtype=torch.float)

        optimizer_frontal.zero_grad()
        optimizer_lateral.zero_grad()

        with autocast_ctx(device1):
            output_frontal = model_frontal(images_frontal)
            loss_frontal = loss_function_frontal(output_frontal, labels_frontal)
        with autocast_ctx(device2):
            output_lateral = model_lateral(images_lateral)
            loss_lateral = loss_function_lateral(output_lateral, labels_lateral)

        scaler_frontal.scale(loss_frontal).backward()
        scaler_lateral.scale(loss_lateral).backward()

        scaler_frontal.step(optimizer_frontal)
        scaler_lateral.step(optimizer_lateral)

        scaler_frontal.update()
        scaler_lateral.update()

        running_loss_frontal += loss_frontal.detach().item()
        running_loss_lateral += loss_lateral.detach().item()
        epoch_running_loss_frontal += loss_frontal.detach().item()
        epoch_running_loss_lateral += loss_lateral.detach().item()

        train_bar.set_postfix(
            Loss_F=f"{epoch_running_loss_frontal / (step + 1):.4f}",
            Loss_L=f"{epoch_running_loss_lateral / (step + 1):.4f}",
            LR=optimizer_frontal.param_groups[0]['lr']
        )
        train_bar.refresh()

    avg_train_loss_frontal = running_loss_frontal / len(dataLoaderTrain)
    avg_train_loss_lateral = running_loss_lateral / len(dataLoaderTrain)
    return avg_train_loss_frontal, avg_train_loss_lateral


def validate(model_frontal, model_lateral, dataLoaderVal, loss_function_validation, modelEvaluationVal):
    model_frontal.eval()
    model_lateral.eval()
    modelEvaluationVal.reset()
    validation_loss_frontal = 0
    validation_loss_lateral = 0

    val_bar = tqdm(dataLoaderVal, desc="Validation", unit="batch")
    with torch.no_grad():
        for batch in val_bar:
            labels_frontal = batch['target_label'].to(device=device1, dtype=torch.float)
            images_frontal_val = batch['image'].to(device=device1, dtype=torch.float)
            labels_lateral_device = batch['target_label'].to(device=device2, dtype=torch.float)
            images_lateral_val = batch['imageOtherView'].to(device=device2, dtype=torch.float)

            output_frontal = model_frontal(images_frontal_val)
            output_lateral = model_lateral(images_lateral_val)

            current_val_loss_f = loss_function_validation(output_frontal, labels_frontal).item()
            current_val_loss_l = loss_function_validation(output_lateral, labels_lateral_device).item()

            validation_loss_frontal += current_val_loss_f
            validation_loss_lateral += current_val_loss_l

            estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= 0.5 else THROMBUS_YES
            estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= 0.5 else THROMBUS_YES

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
                Loss_L=f"{current_val_loss_l:.4f}"
            )
            val_bar.refresh()

    avg_val_loss_frontal = validation_loss_frontal / len(dataLoaderVal)
    avg_val_loss_lateral = validation_loss_lateral / len(dataLoaderVal)
    return (
        avg_val_loss_frontal,
        avg_val_loss_lateral,
        modelEvaluationVal.getAccuracyFrontal(),
        modelEvaluationVal.getPrecisionFrontal(),
        modelEvaluationVal.getRecallFrontal(),
        modelEvaluationVal.getMccFrontal(),
        modelEvaluationVal.getAccuracyLateral(),
        modelEvaluationVal.getPrecisionLateral(),
        modelEvaluationVal.getRecallLateral(),
        modelEvaluationVal.getMccLateral(),
    )


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Split data once using original ratios
    print("Splitting data sequences (70/15/15) with stratification...")
    train_ids, val_ids, test_ids = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    trainval_ids = train_ids + val_ids  # test_ids held out

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    best_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(trainval_ids), start=1):
        print(f"\n========== Fold {fold_idx}/{N_FOLDS} ==========")

        fold_train_ids = [trainval_ids[i] for i in tr_idx]
        fold_val_ids = [trainval_ids[i] for i in va_idx]

        data_set_train = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=fold_train_ids, training=True)
        data_set_val = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=fold_val_ids, training=False)

        dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        dataLoaderVal = DataLoader(dataset=data_set_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        model_frontal = load_and_configure_model(device1, CHECKPOINT_FRONTAL_NAME)
        model_lateral = load_and_configure_model(device2, CHECKPOINT_LATERAL_NAME)

        optimizer_frontal = optim.AdamW(model_frontal.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
        scheduler_frontal = optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, 'min', factor=0.1, patience=5)
        loss_function_frontal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

        optimizer_lateral = optim.AdamW(model_lateral.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
        scheduler_lateral = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lateral, 'min', factor=0.1, patience=5)
        loss_function_lateral = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

        loss_function_validation = nn.BCEWithLogitsLoss()

        scaler_frontal = torch.amp.GradScaler("cuda") if device1.type == "cuda" else torch.amp.GradScaler()
        scaler_lateral = torch.amp.GradScaler("cuda") if device2.type == "cuda" else torch.amp.GradScaler()

        modelEvaluationVal = ModelEvaluation()
        best_mcc_frontal = -1.0
        best_mcc_lateral = -1.0

        # Metrics CSV for this fold
        metrics_file = os.path.join(OUTPUT_PATH, f"metrics_fold_{fold_idx}.csv")
        with open(metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
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

        print("\nStarting Fine-Tuning for this fold...")
        for epoch in range(EPOCHS):
            avg_train_loss_frontal, avg_train_loss_lateral = train_one_epoch(
                model_frontal,
                model_lateral,
                dataLoaderTrain,
                optimizer_frontal,
                optimizer_lateral,
                loss_function_frontal,
                loss_function_lateral,
                scaler_frontal,
                scaler_lateral,
            )

            (
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
            ) = validate(model_frontal, model_lateral, dataLoaderVal, loss_function_validation, modelEvaluationVal)

            scheduler_frontal.step(avg_train_loss_frontal)
            scheduler_lateral.step(avg_train_loss_lateral)

            print(
                f"[Fold {fold_idx}] Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: Frontal={avg_train_loss_frontal:.4f}, Lateral={avg_train_loss_lateral:.4f} | "
                f"Val Loss: Frontal={avg_val_loss_frontal:.4f}, Lateral={avg_val_loss_lateral:.4f} | "
                f"ACC_F={acc_front:.4f}, PREC_F={prec_front:.4f}, REC_F={rec_front:.4f}, MCC_F={mcc_front:.4f} | "
                f"ACC_L={acc_lat:.4f}, PREC_L={prec_lat:.4f}, REC_L={rec_lat:.4f}, MCC_L={mcc_lat:.4f}"
            )

            with open(metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        fold_idx,
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

            current_mcc_frontal = mcc_front
            if current_mcc_frontal > best_mcc_frontal:
                best_mcc_frontal = current_mcc_frontal
                print(f"[Fold {fold_idx}] New best frontal MCC: {best_mcc_frontal:.4f}. Saving model.")
                torch.save(
                    {'fold': fold_idx, 'epoch': epoch, 'model_state_dict': model_frontal.state_dict(), 'mcc': best_mcc_frontal},
                    os.path.join(OUTPUT_PATH, f"frontal_fine_tuned_best_mcc_fold_{fold_idx}.pt"),
                )

            current_mcc_lateral = mcc_lat
            if current_mcc_lateral > best_mcc_lateral:
                best_mcc_lateral = current_mcc_lateral
                print(f"[Fold {fold_idx}] New best lateral MCC: {best_mcc_lateral:.4f}. Saving model.")
                torch.save(
                    {'fold': fold_idx, 'epoch': epoch, 'model_state_dict': model_lateral.state_dict(), 'mcc': best_mcc_lateral},
                    os.path.join(OUTPUT_PATH, f"lateral_fine_tuned_best_mcc_fold_{fold_idx}.pt"),
                )

        print(f"\n[Fold {fold_idx}] Finished. Best MCC: Frontal={best_mcc_frontal:.4f}, Lateral={best_mcc_lateral:.4f}")
        best_rows.append([fold_idx, best_mcc_frontal, best_mcc_lateral])

    # Save per-fold best MCC summary
    best_file = os.path.join(OUTPUT_PATH, "best_mcc_per_fold.csv")
    with open(best_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_mcc_frontal", "best_mcc_lateral"])
        writer.writerows(best_rows)
    print(f"\nBest MCC per fold written to {best_file}")


if __name__ == "__main__":
    main()


