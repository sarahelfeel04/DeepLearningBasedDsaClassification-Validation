#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.cuda.amp as amp
import os
from tqdm import tqdm

# --- Import modified Dataset and repository components (package-relative) ---
from .dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES
from .utils.CnnLstmModel import CnnLstmModel
from .evaluation.ModelEvaluation import ModelEvaluation
from .GermanDataUtils import (
    GermanFineTuneDataset,
    load_german_sequences,
    split_german_data_by_patient,
)

# --- Configuration (UPDATED with your paths) ---
# 1. Chinese Data Root Path
DATA_ROOT_PATH_CHINA = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/Channel0-DataTypeUnsignedShort-Values0to4000"

# 2. German Data Root Path
DATA_ROOT_PATH_GERMAN = "/media/nami/FastDataSpace/ThromboMap-Validation/dataClinic2024"
GERMAN_ANNOTATIONS_CSV = "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/DeepLearningBasedDsaClassification-Validation/final_annotations_deutsch_2024.csv"

# 3. Model Checkpoint Paths (initial pre-trained weights)
MODEL_BASE_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/Classificator/Models"
CHECKPOINT_FRONTAL_NAME = "frontal/final_model.pt"
CHECKPOINT_LATERAL_NAME = "lateral/final_model.pt"

# 4. Output Path for mixed fine-tuning
OUTPUT_PATH = "./fine_tuned_models/china_german_mixed_50epochs/"

# Hyperparameters
FINE_TUNE_LR = 5e-6     # Low Learning Rate for Fine-Tuning (recommended when unfreezing CNN)
EPOCHS = 50
BATCH_SIZE = 1
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2  # Midpoint between the two label encodings

# --- Device Setup ---
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
if device2.type == "cpu":
    print("Warning: Only one GPU or none detected. Both models will use device:", device1)
    device2 = device1


# --- Model Loading and Configuration (UNFREEZE CNN) ---
def load_and_configure_model(device, checkpoint_name):
    """Loads pre-trained weights and ensures all layers, including the CNN, are trainable."""
    # Model parameters (using EfficientNet-B1/V2-RW-S standard config as an example)
    model = CnnLstmModel(512, 3, 1, True, device)

    # Load pre-trained weights
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


# --- Main Fine-Tuning Loop (Mixed China + German) ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. Split CHINESE data into TRAIN, VAL, TEST IDs
    print("1. Splitting CHINESE data sequences (70/15/15) with stratification...")
    train_ids_ch, val_ids_ch, test_ids_ch = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH_CHINA, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 2. Load and split GERMAN data by patient
    print("2. Loading GERMAN annotations and building sequences...")
    german_sequences = load_german_sequences(DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV)
    print("3. Splitting GERMAN data sequences (70/15/15) by patient...")
    train_ids_de, val_ids_de, test_ids_de = split_german_data_by_patient(
        german_sequences, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 3. Initialize Datasets and DataLoaders (Mixed)
    print(
        f"4. Initializing Mixed DataLoaders:\n"
        f"   China   -> Train={len(train_ids_ch)}, Val={len(val_ids_ch)}, Test={len(test_ids_ch)}\n"
        f"   German  -> Train={len(train_ids_de)}, Val={len(val_ids_de)}, Test={len(test_ids_de)}"
    )

    # Chinese datasets
    data_set_train_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=train_ids_ch, training=True)
    data_set_val_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=val_ids_ch, training=False)
    data_set_test_ch = FineTuneDsaDataset(DATA_ROOT_PATH_CHINA, data_subset=test_ids_ch, training=False)

    # German datasets
    data_set_train_de = GermanFineTuneDataset(german_sequences, train_ids_de, DATA_ROOT_PATH_GERMAN, training=True)
    data_set_val_de = GermanFineTuneDataset(german_sequences, val_ids_de, DATA_ROOT_PATH_GERMAN, training=False)
    data_set_test_de = GermanFineTuneDataset(german_sequences, test_ids_de, DATA_ROOT_PATH_GERMAN, training=False)

    # Concatenate China + German for each split
    data_set_train = ConcatDataset([data_set_train_ch, data_set_train_de])
    data_set_val = ConcatDataset([data_set_val_ch, data_set_val_de])
    data_set_test = ConcatDataset([data_set_test_ch, data_set_test_de])

    dataLoaderTrain = DataLoader(
        dataset=data_set_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    dataLoaderVal = DataLoader(
        dataset=data_set_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )
    # dataLoaderTest will be used after training is complete

    # 4. Load Models (Unfrozen)
    model_frontal = load_and_configure_model(device1, CHECKPOINT_FRONTAL_NAME)
    model_lateral = load_and_configure_model(device2, CHECKPOINT_LATERAL_NAME)

    # 5. Optimizer and Loss Function (same as original)
    optimizer_frontal = optim.AdamW(model_frontal.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
    scheduler_frontal = optim.lr_scheduler.ReduceLROnPlateau(optimizer_frontal, "min", factor=0.1, patience=5)
    loss_function_frontal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

    optimizer_lateral = optim.AdamW(model_lateral.parameters(), lr=FINE_TUNE_LR, weight_decay=0.01)
    scheduler_lateral = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lateral, "min", factor=0.1, patience=5)
    loss_function_lateral = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))

    loss_function_validation = nn.BCEWithLogitsLoss()

    scaler_frontal = amp.GradScaler()
    scaler_lateral = amp.GradScaler()

    modelEvaluationVal = ModelEvaluation()
    best_mcc_frontal = -1.0
    best_mcc_lateral = -1.0

    # 6. Training Loop (unchanged logic, now using mixed dataset)
    print("\n5. Starting Mixed Fine-Tuning (China + German)...")
    for epoch in range(EPOCHS):
        model_frontal.train()
        model_lateral.train()
        running_loss_frontal = 0.0
        running_loss_lateral = 0.0
        epoch_running_loss_frontal = 0.0
        epoch_running_loss_lateral = 0.0

        train_bar = tqdm(dataLoaderTrain, desc=f"Epoch {epoch+1}/{EPOCHS} Training", unit="batch")

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

        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        print(f"Train Loss (Mixed): Frontal={avg_train_loss_frontal:.4f}, Lateral={avg_train_loss_lateral:.4f}")

        # --- Validation ---
        model_frontal.eval()
        model_lateral.eval()
        modelEvaluationVal.reset()
        validation_loss_frontal = 0.0
        validation_loss_lateral = 0.0

        val_bar = tqdm(dataLoaderVal, desc="Validation", unit="batch")

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
        print(f"Validation Loss (Mixed): Frontal={avg_val_loss_frontal:.4f}, Lateral={avg_val_loss_lateral:.4f}")
        modelEvaluationVal.printAllStats()

        # --- Save Best Model based on MCC ---
        current_mcc_frontal = modelEvaluationVal.getMccFrontal()
        if current_mcc_frontal > best_mcc_frontal:
            best_mcc_frontal = current_mcc_frontal
            print(f"New best frontal MCC: {best_mcc_frontal:.4f}. Saving model.")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_frontal.state_dict(),
                    "mcc": best_mcc_frontal,
                },
                os.path.join(OUTPUT_PATH, "frontal_fine_tuned_best_mcc.pt"),
            )

        current_mcc_lateral = modelEvaluationVal.getMccLateral()
        if current_mcc_lateral > best_mcc_lateral:
            best_mcc_lateral = current_mcc_lateral
            print(f"New best lateral MCC: {best_mcc_lateral:.4f}. Saving model.")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_lateral.state_dict(),
                    "mcc": best_mcc_lateral,
                },
                os.path.join(OUTPUT_PATH, "lateral_fine_tuned_best_mcc.pt"),
            )

    print("\nMixed fine-tuning process finished. The best models are saved in the output directory.")
    print(
        f"Final Mixed Test Set sizes: "
        f"China={len(test_ids_ch)}, German={len(test_ids_de)}. "
        f"Use your evaluation script to run on the combined test set."
    )