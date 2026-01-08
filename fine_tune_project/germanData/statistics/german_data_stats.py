#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to summarize German DSA data:
- Loads sequences via GermanDataUtils.load_german_sequences
- Computes per-sequence metadata (dtype, HxWxZ, frame counts)
- Counts thrombus vs no-thrombus
- Saves per-sequence CSV and a small summary CSV
"""

import os
import csv
import nibabel as nib
from tqdm import tqdm

from .GermanDataUtils import load_german_sequences, _parse_case_id

# Default paths (adjust if needed)
DATA_ROOT_PATH_GERMAN = "/media/nami/Volume/ThromboMap/dataClinic2024"
GERMAN_ANNOTATIONS_CSV = (
    "/media/nami/FastDataSpace/ThromboMap-Validation/original-train-repo/"
    "DeepLearningBasedDsaClassification-Validation/final_annotations_deutsch_2024.csv"
)

OUTPUT_DIR = "./fine_tune_project/german_stats"
PER_SEQ_CSV = "german_sequences_stats.csv"
SUMMARY_CSV = "german_sequences_summary.csv"
THROMBUS_COUNTS_CSV = "german_thrombus_value_counts.csv"


def _build_gt_map(annotations_csv):
    """
    Build:
      - mapping (patient_name, phase) -> is_thrombus (True/False) from the
        'Ground truth' column in the German annotations CSV.
      - dictionary of raw 'Thrombus' column values -> count
    """
    gt_map = {}
    thrombus_counts = {}
    with open(annotations_csv, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("CaseID", "").strip()
            if not case_id:
                continue
            try:
                patient_name, phase = _parse_case_id(case_id)
            except ValueError:
                # Unexpected CaseID format; skip
                continue
            gt_str = row.get("Ground truth", "").strip().lower()
            is_thrombus = gt_str == "true"
            gt_map[(patient_name, phase.lower())] = is_thrombus

            thrombus_str = row.get("Thrombus", "")
            thrombus_str = thrombus_str.strip()
            thrombus_counts[thrombus_str] = thrombus_counts.get(thrombus_str, 0) + 1
    return gt_map, thrombus_counts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load sequences as used in training
    sequences = load_german_sequences(DATA_ROOT_PATH_GERMAN, GERMAN_ANNOTATIONS_CSV)
    if not sequences:
        print("No German sequences loaded. Check paths/annotations.")
        return

    # Ground-truth map from CSV (patient_name, phase) -> True/False
    # and distribution of raw 'Thrombus' column values
    gt_map, thrombus_counts = _build_gt_map(GERMAN_ANNOTATIONS_CSV)

    per_seq_rows = []
    num_thrombus = 0
    num_no_thrombus = 0

    patients = set()

    min_frames_f = None
    max_frames_f = None
    min_frames_l = None
    max_frames_l = None

    print(f"Loaded {len(sequences)} German sequences. Computing stats...")
    for seq_id, info in tqdm(sequences.items(), desc="Processing", unit="seq"):
        # Derive is_thrombus from CSV 'Ground truth' via patient_name + phase
        patient_name = info.get("patient_name", "")
        phase = info.get("phase", "").lower()
        is_thrombus = gt_map.get((patient_name, phase), None)

        pid = info.get("patient_id", patient_name)
        if pid:
            patients.add(pid)

        if is_thrombus is True:
            num_thrombus += 1
        elif is_thrombus is False:
            num_no_thrombus += 1
        if is_thrombus is True:
            num_thrombus += 1
        elif is_thrombus is False:
            num_no_thrombus += 1

        frontal_rel = info.get("frontal_path", "")
        lateral_rel = info.get("lateral_path", "")
        frontal_path = os.path.join(DATA_ROOT_PATH_GERMAN, frontal_rel)
        lateral_path = os.path.join(DATA_ROOT_PATH_GERMAN, lateral_rel)

        shape_f = shape_l = ()
        dtype_f = dtype_l = ""
        frames_f = frames_l = None

        # Frontal
        try:
            img_f = nib.load(frontal_path)
            data_f = img_f.get_fdata()
            shape_f = data_f.shape
            dtype_f = str(data_f.dtype)
            frames_f = shape_f[2] if len(shape_f) >= 3 else None
        except Exception as e:
            print(f"[Warn] Could not read frontal {frontal_path}: {e}")

        # Lateral
        try:
            img_l = nib.load(lateral_path)
            data_l = img_l.get_fdata()
            shape_l = data_l.shape
            dtype_l = str(data_l.dtype)
            frames_l = shape_l[2] if len(shape_l) >= 3 else None
        except Exception as e:
            print(f"[Warn] Could not read lateral {lateral_path}: {e}")

        # Update min/max frames
        if frames_f is not None:
            min_frames_f = frames_f if min_frames_f is None else min(min_frames_f, frames_f)
            max_frames_f = frames_f if max_frames_f is None else max(max_frames_f, frames_f)
        if frames_l is not None:
            min_frames_l = frames_l if min_frames_l is None else min(min_frames_l, frames_l)
            max_frames_l = frames_l if max_frames_l is None else max(max_frames_l, frames_l)

        per_seq_rows.append(
            {
                "seq_id": seq_id,
                "patient_id": info.get("patient_id", ""),
                "patient_name": patient_name,
                "phase": phase,
                "label_numeric": info.get("label", None),
                "is_thrombus": is_thrombus,
                "frontal_path": frontal_rel,
                "lateral_path": lateral_rel,
                "shape_frontal": "x".join(map(str, shape_f)) if shape_f else "",
                "shape_lateral": "x".join(map(str, shape_l)) if shape_l else "",
                "frames_frontal": frames_f,
                "frames_lateral": frames_l,
                "dtype_frontal": dtype_f,
                "dtype_lateral": dtype_l,
            }
        )

    # Write per-sequence CSV
    per_seq_path = os.path.join(OUTPUT_DIR, PER_SEQ_CSV)
    with open(per_seq_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seq_id",
                "patient_id",
                "patient_name",
                "phase",
                "label_numeric",
                "is_thrombus",
                "frontal_path",
                "lateral_path",
                "shape_frontal",
                "shape_lateral",
                "frames_frontal",
                "frames_lateral",
                "dtype_frontal",
                "dtype_lateral",
            ],
        )
        writer.writeheader()
        writer.writerows(per_seq_rows)

    # Summary CSV
    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_CSV)
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["total_sequences", len(per_seq_rows)])
        writer.writerow(["num_patients", len(patients)])
        writer.writerow(["thrombus", num_thrombus])
        writer.writerow(["no_thrombus", num_no_thrombus])
        writer.writerow(["frames_frontal_min", min_frames_f])
        writer.writerow(["frames_frontal_max", max_frames_f])
        writer.writerow(["frames_lateral_min", min_frames_l])
        writer.writerow(["frames_lateral_max", max_frames_l])

    # Thrombus value distribution CSV
    thrombus_counts_path = os.path.join(OUTPUT_DIR, THROMBUS_COUNTS_CSV)
    with open(thrombus_counts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["thrombus_value", "count"])
        for key, count in sorted(thrombus_counts.items(), key=lambda kv: (kv[0] is None, str(kv[0]))):
            writer.writerow([key, count])

    print(f"Per-sequence stats saved to {per_seq_path}")
    print(f"Summary saved to {summary_path}")
    print(f"Thrombus value counts saved to {thrombus_counts_path}")


if __name__ == "__main__":
    main()


