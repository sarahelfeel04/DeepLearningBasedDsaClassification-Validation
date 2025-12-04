#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick sanity-check for the German data utils:

- Loads German sequences using the annotation CSV
- Prints total number of sequences and patient counts
- Splits by patient into train/val/test (70/15/15)
- Verifies that no patient appears in more than one split
"""

import os

# Import the module from the same directory as this script
import GermanDataUtils

# Paths (must match what you use in fine_tune_mix.py)
DATA_ROOT_PATH_GERMAN = "/media/nami/FastDataSpace/ThromboMap-Validation/dataClinic2024"
GERMAN_ANNOTATIONS_CSV = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/final_annotations_deutsch_2024.csv"
)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def main():
    print("=== German Data Utils Sanity Check ===")
    print(f"Data root: {DATA_ROOT_PATH_GERMAN}")
    print(f"Annotations CSV: {GERMAN_ANNOTATIONS_CSV}")

    if not os.path.isdir(DATA_ROOT_PATH_GERMAN):
        print("\n[ERROR] German data root directory does not exist.")
        return
    if not os.path.isfile(GERMAN_ANNOTATIONS_CSV):
        print("\n[ERROR] German annotations CSV does not exist.")
        return

    # 1. Load sequences
    sequences = GermanDataUtils.load_german_sequences(
        DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV
    )
    n_sequences = len(sequences)
    print(f"\nTotal German sequences found (with both frontal & lateral and in CSV): {n_sequences}")

    if n_sequences == 0:
        print("[WARNING] No usable German sequences found. Check paths / folder names / CSV formatting.")
        return

    # 2. Basic stats
    patients = {}
    pos_count = 0
    neg_count = 0
    for seq_id, info in sequences.items():
        # Group by patient_id (the patient name), so all pre/post for a patient stay together
        pid = info["patient_id"]
        patients.setdefault(pid, []).append(seq_id)
        if info["label"] == GermanDataUtils.THROMBUS_YES:
            pos_count += 1
        else:
            neg_count += 1

    print(f"Unique patients in German data: {len(patients)}")
    print(f"Sequences with thrombus (label==THROMBUS_YES): {pos_count}")
    print(f"Sequences without thrombus (label==THROMBUS_NO): {neg_count}")

    # 3. Split by patient
    print("\nPerforming 70/15/15 split by patient...")
    train_ids, val_ids, test_ids = GermanDataUtils.split_german_data_by_patient(
        sequences,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )

    # 4. Verify that no patient appears in more than one split
    def collect_patients(seq_ids):
        s = set()
        for sid in seq_ids:
            s.add(sequences[sid]["patient_id"])
        return s

    train_patients = collect_patients(train_ids)
    val_patients = collect_patients(val_ids)
    test_patients = collect_patients(test_ids)

    inter_train_val = train_patients & val_patients
    inter_train_test = train_patients & test_patients
    inter_val_test = val_patients & test_patients

    print("\n=== Split Summary (German only) ===")
    print(f"Train sequences: {len(train_ids)} from {len(train_patients)} patients")
    print(f"Val   sequences: {len(val_ids)} from {len(val_patients)} patients")
    print(f"Test  sequences: {len(test_ids)} from {len(test_patients)} patients")

    if inter_train_val or inter_train_test or inter_val_test:
        print("\n[WARNING] Some patients appear in multiple splits!")
        if inter_train_val:
            print(f"  Patients in both TRAIN and VAL: {sorted(inter_train_val)}")
        if inter_train_test:
            print(f"  Patients in both TRAIN and TEST: {sorted(inter_train_test)}")
        if inter_val_test:
            print(f"  Patients in both VAL and TEST: {sorted(inter_val_test)}")
    else:
        print("\nPatient-level split check: OK (no patient appears in more than one split).")

    print("\nSanity check complete.")


if __name__ == "__main__":
    main()


