import os
import re
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import nibabel
import torch
from torch.utils.data import Dataset

from .utils.ImageUtils import ImageUtils
from .utils.DataAugmentation import DataAugmentation

# Use the same label encoding as the Chinese data
THROMBUS_NO = 0.214   # Negative class
THROMBUS_YES = 0.786  # Positive class


class GermanFineTuneDataset(Dataset):
    """
    Dataset for German DSA data, structured to match FineTuneDsaDataset output.

    Each item is a paired frontal+lateral sequence with:
      - 'image'            : frontal volume (after augmentation, as tensor)
      - 'imageOtherView'   : lateral volume (after augmentation, as tensor)
      - 'target_label'     : scalar tensor [1, 1] with THROMBUS_NO / THROMBUS_YES
      - plus the keys required by DataAugmentation and filenames for bookkeeping.
    """

    def __init__(self, sequences: Dict[str, dict], sequence_ids: List[str], data_root: str, training: bool = True):
        """
        sequences: dict[sequence_id] -> {
            'patient_no': str,
            'patient_name': str,
            'phase': 'pre' or 'post',
            'frontal_path': str (absolute or relative to data_root),
            'lateral_path': str,
            'label': float (THROMBUS_NO/YES)
        }
        sequence_ids: list of keys into sequences used by this dataset split.
        data_root: root directory of German data (e.g. /media/.../dataClinic2024)
        training: True for train (uses training augmentation), False for val/test.
        """
        self.sequences_all = sequences
        self.sequence_ids = sequence_ids
        self.data_root = data_root
        self.training = training

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, index):
        seq_id = self.sequence_ids[index]
        seq = self.sequences_all[seq_id]

        frontal_path = seq["frontal_path"]
        lateral_path = seq["lateral_path"]

        # Make absolute if needed
        if not os.path.isabs(frontal_path):
            frontal_path = os.path.join(self.data_root, frontal_path)
        if not os.path.isabs(lateral_path):
            lateral_path = os.path.join(self.data_root, lateral_path)

        # Load NIfTI volumes, mimic Chinese pipeline (scaling, uint8, fill borders)
        image_data_f = nibabel.load(frontal_path).get_fdata(dtype=np.float32) * 0.062271062
        image_data_f = image_data_f.astype(np.uint8)
        image_data_f = ImageUtils.fillBlackBorderWithRandomNoise(image_data_f, 193)

        image_data_l = nibabel.load(lateral_path).get_fdata(dtype=np.float32) * 0.062271062
        image_data_l = image_data_l.astype(np.uint8)
        image_data_l = ImageUtils.fillBlackBorderWithRandomNoise(image_data_l, 193)

        # Dynamic sequence padding (match Chinese logic)
        z_f = image_data_f.shape[2]
        z_l = image_data_l.shape[2]

        if z_f != z_l:
            max_len = max(z_f, z_l)
            fill_value = 193

            if z_f < max_len:
                slices_to_add = max_len - z_f
                zeros = np.full(
                    (image_data_f.shape[0], image_data_f.shape[1], slices_to_add),
                    fill_value,
                    dtype=np.uint8,
                )
                image_data_f = np.append(image_data_f, zeros, axis=2)
            elif z_l < max_len:
                slices_to_add = max_len - z_l
                zeros = np.full(
                    (image_data_l.shape[0], image_data_l.shape[1], slices_to_add),
                    fill_value,
                    dtype=np.uint8,
                )
                image_data_l = np.append(image_data_l, zeros, axis=2)

        # Ensure frontal and lateral have the same HxW before Albumentations Compose
        h_f, w_f, _ = image_data_f.shape
        h_l, w_l, _ = image_data_l.shape
        if (h_f != h_l) or (w_f != w_l):
            max_h = max(h_f, h_l)
            max_w = max(w_f, w_l)
            fill_value = 193

            if h_f != max_h or w_f != max_w:
                padded_f = np.full((max_h, max_w, image_data_f.shape[2]), fill_value, dtype=np.uint8)
                padded_f[:h_f, :w_f, :] = image_data_f
                image_data_f = padded_f

            if h_l != max_h or w_l != max_w:
                padded_l = np.full((max_h, max_w, image_data_l.shape[2]), fill_value, dtype=np.uint8)
                padded_l[:h_l, :w_l, :] = image_data_l
                image_data_l = padded_l

        # Keypoints placeholder to satisfy DataAugmentation
        keypoint_placeholder = [(1, 1)]

        data = {
            "image": image_data_f,
            "keypoints": keypoint_placeholder,
            "imageOtherView": image_data_l,
            "keypointsOtherView": keypoint_placeholder,
            "frontalAndLateralView": True,
            "imageMean": 0,
            "imageOtherViewMean": 0,
            "imageStd": 1.0,
            "imageOtherViewStd": 1.0,
        }

        augmentation = DataAugmentation(data=data)

        if self.training:
            augmentation.createTransformTraining()
        else:
            augmentation.createTransformValidation()

        augmentation.applyTransform()
        augmentation.zeroPaddingEqualLength()
        augmentation.normalizeData()

        tensor_data = augmentation.convertToTensor()
        tensor_data["target_label"] = torch.tensor(seq["label"], dtype=torch.float).unsqueeze(0)

        # Add filenames for bookkeeping/evaluation
        tensor_data["filename"] = os.path.relpath(frontal_path, self.data_root)
        tensor_data["filenameOtherView"] = os.path.relpath(lateral_path, self.data_root)

        return tensor_data


def _parse_case_id(case_id: str) -> Tuple[str, str]:
    """
    From 'wen_ro_m50 (pre)' -> ('wen_ro_m50', 'pre')
    From 'buc-jo_m39 (post)' -> ('buc-jo_m39', 'post')
    """
    m = re.match(r"\s*([^)]+)\s*\(\s*(pre|post)\s*\)\s*$", case_id)
    if not m:
        raise ValueError(f"Unexpected CaseID format: {case_id}")
    name = m.group(1).strip()
    phase = m.group(2).strip().lower()  # 'pre' or 'post'
    return name, phase


def _find_view_folder(patient_dir: str, phase: str, view_keyword: str) -> str:
    """
    In patient_dir, find subfolder matching phase ('pre' or 'post') and view ('frontal' or 'lateral').
    Returns subfolder path or None if not found.
    """
    if not os.path.isdir(patient_dir):
        return None

    phase = phase.lower()
    view_keyword = view_keyword.lower()

    for entry in os.listdir(patient_dir):
        sub = os.path.join(patient_dir, entry)
        if not os.path.isdir(sub):
            continue
        name_lower = entry.lower()
        if phase in name_lower and view_keyword in name_lower:
            return sub

    return None


def _find_nii_in_folder(folder: str) -> str:
    """
    Return path to the first .nii/.nii.gz file in folder, or None if not found.
    """
    if not folder or not os.path.isdir(folder):
        return None
    for fname in os.listdir(folder):
        if fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz"):
            return os.path.join(folder, fname)
    return None


def load_german_sequences(
    data_root: str,
    annotations_csv: str,
) -> Dict[str, dict]:
    """
    Build a sequence dictionary for German data using the final_annotations_deutsch_2024.csv.

    Folder layout (German clinic data):
      data_root/
        April 2024/
          032-wen_ro_m50/
            pre frontal/
            pre lateral/
            post frontal/
            post lateral/
        May 2024/
          ...

    CaseID in CSV: 'wen_ro_m50 (pre)' or 'wen_ro_m50 (post)'
      - We parse out patient_name='wen_ro_m50' and phase='pre'/'post'
      - We then search ALL month folders for directories whose name ends with '-wen_ro_m50'
      - Within each matching patient directory, we look for 'pre/post' + 'frontal/lateral' subfolders.

    Returns dict[sequence_id] -> {
        'patient_id': str,        # patient_name, used for grouping in splits
        'patient_name': str,
        'phase': 'pre' or 'post',
        'frontal_path': str (relative to data_root),
        'lateral_path': str (relative to data_root),
        'label': float (THROMBUS_NO/YES)
    }

    Only sequences listed in the CSV are used. If a sequence is missing a frontal or
    lateral view, it is skipped.
    """
    sequences: Dict[str, dict] = {}

    # CSV is in German, likely encoded with latin-1 / ISO-8859-1
    with open(annotations_csv, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("CaseID", "").strip()
            gt_str = row.get("Ground truth", "").strip().lower()

            if not case_id:
                continue

            try:
                patient_name, phase = _parse_case_id(case_id)
            except ValueError:
                print(f"[GermanData] Skipping row with unexpected CaseID format: {case_id}")
                continue

            label = THROMBUS_YES if gt_str == "true" else THROMBUS_NO

            # Find all patient directories matching '*-patient_name' inside each month folder
            matching_patient_dirs: List[Tuple[str, str]] = []  # (month_name, full_patient_dir)
            for month_entry in os.listdir(data_root):
                month_path = os.path.join(data_root, month_entry)
                if not os.path.isdir(month_path):
                    continue

                for pdir in os.listdir(month_path):
                    full_pdir = os.path.join(month_path, pdir)
                    if not os.path.isdir(full_pdir):
                        continue
                    if "-" not in pdir:
                        continue
                    _, name_part = pdir.split("-", 1)
                    if name_part == patient_name:
                        matching_patient_dirs.append((month_entry, full_pdir))

            if not matching_patient_dirs:
                print(f"[GermanData] Skipping {patient_name} ({phase}): no matching patient folder found.")
                continue

            # For each matching patient directory (could be multiple months and/or multiple folders for same patient),
            # try to build a separate sequence. If a patient has multiple pre/post sequences, we take ALL of them.
            for month_name, patient_dir in matching_patient_dirs:
                frontal_folder = _find_view_folder(patient_dir, phase, "frontal")
                lateral_folder = _find_view_folder(patient_dir, phase, "lateral")

                if frontal_folder is None or lateral_folder is None:
                    print(
                        f"[GermanData] Skipping {os.path.basename(patient_dir)} ({phase}) "
                        f"in {month_name}: missing frontal or lateral folder."
                    )
                    continue

                frontal_nii = _find_nii_in_folder(frontal_folder)
                lateral_nii = _find_nii_in_folder(lateral_folder)

                if frontal_nii is None or lateral_nii is None:
                    print(
                        f"[GermanData] Skipping {os.path.basename(patient_dir)} ({phase}) "
                        f"in {month_name}: missing .nii in view folder."
                    )
                    continue

                # Use full folder name (e.g. '174-kli_hi_w39') to distinguish multiple sequences
                folder_id = os.path.basename(patient_dir)
                seq_id = f"{folder_id}-{phase}-{month_name}"
                sequences[seq_id] = {
                    "patient_id": patient_name,  # used for grouping in splits
                    "patient_name": patient_name,
                    "phase": phase,
                    "frontal_path": os.path.relpath(frontal_nii, data_root),
                    "lateral_path": os.path.relpath(lateral_nii, data_root),
                    "label": label,
                }

    print(f"[GermanData] Loaded {len(sequences)} sequences from annotations.")
    return sequences


def split_german_data_by_patient(
    sequences: Dict[str, dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split German sequences into train/val/test by patient, keeping all sequences
    (pre/post) for a patient in the same split. Approximate stratified split
    by patient-level label (any thrombus -> positive).
    """
    # Group sequences by patient (by patient_id so all pre/post for a name stay together)
    patients: Dict[str, List[str]] = {}
    for seq_id, info in sequences.items():
        pid = info["patient_id"]
        patients.setdefault(pid, []).append(seq_id)

    # Determine patient-level label: positive if any of their sequences is positive
    patient_labels: Dict[str, int] = {}
    for pid, seq_ids in patients.items():
        labels = [sequences[s]["label"] for s in seq_ids]
        is_pos = any(lbl == THROMBUS_YES for lbl in labels)
        patient_labels[pid] = 1 if is_pos else 0

    pos_patients = [pid for pid, lab in patient_labels.items() if lab == 1]
    neg_patients = [pid for pid, lab in patient_labels.items() if lab == 0]

    rng = random.Random(random_seed)
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def _split_patient_list(p_list: List[str]):
        n = len(p_list)
        n_test = int(round(test_ratio * n))
        n_train = int(round(train_ratio * n))
        n_val = max(0, n - n_train - n_test)

        train_p = p_list[:n_train]
        val_p = p_list[n_train:n_train + n_val]
        test_p = p_list[n_train + n_val:]
        return train_p, val_p, test_p

    pos_train, pos_val, pos_test = _split_patient_list(pos_patients)
    neg_train, neg_val, neg_test = _split_patient_list(neg_patients)

    train_patients = pos_train + neg_train
    val_patients = pos_val + neg_val
    test_patients = pos_test + neg_test

    rng.shuffle(train_patients)
    rng.shuffle(val_patients)
    rng.shuffle(test_patients)

    def _collect_seq_ids(patient_list: List[str]) -> List[str]:
        seqs: List[str] = []
        for pid in patient_list:
            seqs.extend(patients[pid])
        return seqs

    train_ids = _collect_seq_ids(train_patients)
    val_ids = _collect_seq_ids(val_patients)
    test_ids = _collect_seq_ids(test_patients)

    print(
        f"[GermanData] Patients -> train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}"
    )
    print(
        f"[GermanData] Sequences -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}"
    )

    return train_ids, val_ids, test_ids


