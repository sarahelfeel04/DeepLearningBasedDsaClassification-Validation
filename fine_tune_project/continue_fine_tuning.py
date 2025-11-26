#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import os
import sys
from tqdm import tqdm

# --- Setup local imports ---
sys.path.append(os.path.abspath('./'))
from dsa_data_prep import FineTuneDsaDataset, THROMBUS_NO, THROMBUS_YES 
from utils.CnnLstmModel import CnnLstmModel 
from evaluation.ModelEvaluation import ModelEvaluation 

# --- Configuration for Resumption ---
# Path to the data and model definition files
DATA_ROOT_PATH = "/media/nami/FastDataSpace/ThromboMap-Validation/datasets/Channel0-DataTypeUnsignedShort-Values0to4000" 
LABEL_THRESHOLD = (THROMBUS_NO + THROMBUS_YES) / 2

# PATHS TO PREVIOUS RUN'S CHECKPOINTS (Your 20-epoch run)
PREVIOUS_OUTPUT_PATH = "./fine_tuned_models/china_data_unfrozen_cnn_20_epochs/" 
RESUME_CHECKPOINT_FRONTAL = os.path.join(PREVIOUS_OUTPUT_PATH, "frontal_fine_tuned_best_mcc.pt")
RESUME_CHECKPOINT_LATERAL = os.path.join(PREVIOUS_OUTPUT_PATH, "lateral_fine_tuned_best_mcc.pt")

# NEW OUTPUT PATH FOR THIS 50-EPOCH RUN
OUTPUT_PATH = "./fine_tuned_models/china_data_unfrozen_cnn_50epochs/" 

# Hyperparameters (Total Run)
FINE_TUNE_LR = 5e-6     
TOTAL_EPOCHS = 50           # Target final epoch number
BATCH_SIZE = 1          
NUM_WORKERS = 4         
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# --- Device Setup ---
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
if device2.type == 'cpu':
    print("Warning: Only one GPU or none detected. Both models will use device:", device1)
    device2 = device1

# --- Resumption and Loading Logic ---
def load_model_and_state(device, resume_path, optimizer_class, lr, wd):
    """Initializes model/optimizer/scheduler and loads state from resume_path safely."""
    
    # Initialize Model (Model parameters from CnnLstmModel)
    model = CnnLstmModel(512, 3, 1, True, device)
    model.to(device)
    
    # Initialize Optimizer
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
    
    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    
    start_epoch = 0
    best_mcc = -1.0
    
    if os.path.exists(resume_path):
        print(f"RESUMING: Loading state from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # FIX: Safely load optimizer state, which caused the KeyError
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   -> Optimizer state loaded successfully.")
        else:
            print("   -> WARNING: Optimizer state NOT FOUND. Starting optimizer fresh.")
        
        # Retrieve resume point
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_mcc = checkpoint.get('mcc', -1.0)
        
        print(f"Loaded: Epoch {start_epoch-1}, Best MCC: {best_mcc:.4f}")
        
    else:
        print(f"ERROR: Resume checkpoint not found at {resume_path}")
        raise FileNotFoundError(f"Cannot resume training: Checkpoint not found at {resume_path}")

    # Ensure ALL layers are unfrozen for continued fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    return model, optimizer, scheduler, start_epoch, best_mcc


# --- Main Fine-Tuning Loop ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 1. Split data (must use same split as previous run)
    print("1. Splitting data sequences (70/15/15) with stratification...")
    train_ids, val_ids, test_ids = FineTuneDsaDataset.split_data(
        DATA_ROOT_PATH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    # 2. Initialize Datasets and DataLoaders
    print(f"2. Initializing DataLoaders (Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)})...")
    data_set_train = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=train_ids, training=True)
    data_set_val = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=val_ids, training=False)
    data_set_test = FineTuneDsaDataset(DATA_ROOT_PATH, data_subset=test_ids, training=False)
    
    dataLoaderTrain = DataLoader(dataset=data_set_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    dataLoaderVal = DataLoader(dataset=data_set_val, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Load Models (Resuming from previous run)
    # The models will now load states safely, or raise FileNotFoundError if the path is wrong.
    model_frontal, optimizer_frontal, scheduler_frontal, START_EPOCH_F, best_mcc_frontal = \
        load_model_and_state(device1, RESUME_CHECKPOINT_FRONTAL, optim.AdamW, FINE_TUNE_LR, 0.01)

    model_lateral, optimizer_lateral, scheduler_lateral, START_EPOCH_L, best_mcc_lateral = \
        load_model_and_state(device2, RESUME_CHECKPOINT_LATERAL, optim.AdamW, FINE_TUNE_LR, 0.01)

    # Use the highest starting epoch to ensure consistency
    START_EPOCH = max(START_EPOCH_F, START_EPOCH_L)
    
    # 4. Loss Functions
    loss_function_frontal = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_lateral = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    loss_function_validation = nn.BCEWithLogitsLoss()
    
    scaler_frontal = amp.GradScaler()
    scaler_lateral = amp.GradScaler()
    
    modelEvaluationVal = ModelEvaluation()
    
    # 5. Training Loop
    print(f"\n3. Starting Fine-Tuning. Resuming from Epoch {START_EPOCH} (Target: {TOTAL_EPOCHS})...")
    
    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        model_frontal.train()
        model_lateral.train()
        
        epoch_running_loss_frontal = 0.0
        epoch_running_loss_lateral = 0.0

        # --- PROGRESS BAR FOR TRAINING ---
        train_bar = tqdm(dataLoaderTrain, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} Training", unit="batch")
        total_train_steps = len(dataLoaderTrain)

        for step, batch in enumerate(train_bar):
            
            # Use the single target label for both views
            labels_frontal = batch['target_label'].to(device=device1, dtype=torch.float)
            images_frontal = batch['image'].to(device=device1, dtype=torch.float)
            
            labels_lateral = batch['target_label'].to(device=device2, dtype=torch.float)
            images_lateral = batch['imageOtherView'].to(device=device2, dtype=torch.float)
            
            # Training Step
            optimizer_frontal.zero_grad()
            optimizer_lateral.zero_grad()
            
            # Use torch.autocast for non-deprecated, correct mixed precision
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
            
            # Loss accumulation
            step_loss_f = loss_frontal.detach().item()
            step_loss_l = loss_lateral.detach().item()
            
            epoch_running_loss_frontal += step_loss_f
            epoch_running_loss_lateral += step_loss_l
            
            # Update TQDM postfix with current average loss
            train_bar.set_postfix(
                Loss_F=f"{epoch_running_loss_frontal / (step + 1):.4f}", 
                Loss_L=f"{epoch_running_loss_lateral / (step + 1):.4f}",
                LR=optimizer_frontal.param_groups[0]['lr']
            )
            train_bar.refresh()


        # 6. End of Epoch Validation and Saving
        avg_train_loss_frontal = epoch_running_loss_frontal / total_train_steps
        avg_train_loss_lateral = epoch_running_loss_lateral / total_train_steps
        scheduler_frontal.step(avg_train_loss_frontal)
        scheduler_lateral.step(avg_train_loss_lateral)

        print(f"\n--- Epoch {epoch+1}/{TOTAL_EPOCHS} Complete ---")
        print(f"Train Loss (Full Epoch Avg): Frontal={avg_train_loss_frontal:.4f}, Lateral={avg_train_loss_lateral:.4f}")
        
        # --- Validation ---
        model_frontal.eval()
        model_lateral.eval()
        modelEvaluationVal.reset()
        validation_loss_frontal = 0
        validation_loss_lateral = 0
        
        # --- PROGRESS BAR FOR VALIDATION ---
        val_bar = tqdm(dataLoaderVal, desc=f"Validation", unit="batch")

        with torch.no_grad():
            for batch in val_bar:
                
                labels_frontal = batch['target_label'].to(device=device1, dtype=torch.float)
                
                output_frontal = model_frontal(batch['image'])
                output_lateral = model_lateral(batch['imageOtherView'])

                # Move label to lateral device (device2) for lateral loss calculation
                labels_lateral_device = batch['target_label'].to(device=device2, dtype=torch.float)

                current_val_loss_f = loss_function_validation(output_frontal, labels_frontal).item()
                current_val_loss_l = loss_function_validation(output_lateral, labels_lateral_device).item()

                validation_loss_frontal += current_val_loss_f
                validation_loss_lateral += current_val_loss_l
                
                # Update Validation Metrics
                estimate_frontal = THROMBUS_NO if torch.sigmoid(output_frontal).item() <= LABEL_THRESHOLD else THROMBUS_YES
                estimate_lateral = THROMBUS_NO if torch.sigmoid(output_lateral).item() <= LABEL_THRESHOLD else THROMBUS_YES
                
                label_value = labels_frontal.item()
                is_thrombus_free = label_value <= LABEL_THRESHOLD

                if is_thrombus_free:
                    modelEvaluationVal.increaseTNfrontal() if estimate_frontal == THROMBUS_NO else modelEvaluationVal.increaseFPfrontal()
                    modelEvaluationVal.increaseTNlateral() if estimate_lateral == THROMBUS_NO else modelEvaluationVal.increaseFPlateral()
                else:
                    modelEvaluationVal.increaseTPfrontal() if estimate_frontal == THROMBUS_YES else modelEvaluationVal.increaseFNfrontal()
                    modelEvaluationVal.increaseTPlateral() if estimate_lateral == THROMBUS_YES else modelEvaluationVal.increaseFNlateral()

                # Update TQDM postfix for Validation
                val_bar.set_postfix(
                    Loss_F=f"{current_val_loss_f:.4f}", 
                    Loss_L=f"{current_val_loss_l:.4f}"
                )
                val_bar.refresh()


        # Calculate and print validation stats
        avg_val_loss_frontal = validation_loss_frontal / len(dataLoaderVal)
        avg_val_loss_lateral = validation_loss_lateral / len(dataLoaderVal)
        print(f"Validation Loss (Avg): Frontal={avg_val_loss_frontal:.4f}, Lateral={avg_val_loss_lateral:.4f}")
        modelEvaluationVal.printAllStats()

        # --- Save Best Model based on MCC (Save Model, Optimizer, and Epoch) ---
        current_mcc_frontal = modelEvaluationVal.getMccFrontal()
        if current_mcc_frontal > best_mcc_frontal:
             best_mcc_frontal = current_mcc_frontal
             print(f"New best frontal MCC: {best_mcc_frontal:.4f}. Saving model.")
             # Save to the new 50-epoch output path
             torch.save({
                 'epoch': epoch, 
                 'model_state_dict': model_frontal.state_dict(),
                 'optimizer_state_dict': optimizer_frontal.state_dict(),
                 'mcc': best_mcc_frontal
                 }, 
                 os.path.join(OUTPUT_PATH, "frontal_fine_tuned_best_mcc.pt"))

        current_mcc_lateral = modelEvaluationVal.getMccLateral()
        if current_mcc_lateral > best_mcc_lateral:
             best_mcc_lateral = current_mcc_lateral
             print(f"New best lateral MCC: {best_mcc_lateral:.4f}. Saving model.")
             # Save to the new 50-epoch output path
             torch.save({
                 'epoch': epoch, 
                 'model_state_dict': model_lateral.state_dict(), 
                 'optimizer_state_dict': optimizer_lateral.state_dict(),
                 'mcc': best_mcc_lateral
                 }, 
                 os.path.join(OUTPUT_PATH, "lateral_fine_tuned_best_mcc.pt"))

    # 7. Final Step
    print("\nFine-tuning process finished. The best models are saved in the output directory.")
    print(f"Final Test Set size is {len(data_set_test)}. It is ready for final, unbiased evaluation using a script like evaluateModels.py.")