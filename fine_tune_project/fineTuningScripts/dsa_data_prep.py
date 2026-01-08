import os
import re
import numpy as np
import nibabel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
# Assumed dependencies from the repository (use package-relative imports within package)
from .utils.ImageUtils import ImageUtils
from .utils.DataAugmentation import DataAugmentation

# Define labels based on original paper values (THROMBUS_NO < 0.5, THROMBUS_YES > 0.5)
THROMBUS_NO = 0.214  # T3 -> Thrombus Free (Negative class)
THROMBUS_YES = 0.786 # T0 -> Thrombus (Positive class)

class FineTuneDsaDataset(Dataset):
    """
    Custom Dataset for fine-tuning using file naming convention for labels and views.
    Handles dynamic padding of unequal sequences.
    """
    def __init__(self, data_path, data_subset=None, training=True):
        self.dataPath = data_path
        self.training = training
        
        all_sequences = self._group_files_by_id()
        
        if data_subset is None:
            self.datasetDict = self._create_dataset_dict(all_sequences)
        else:
            self.datasetDict = self._create_dataset_dict({k: all_sequences[k] for k in data_subset})

    
    # =====================================================================
    # --- Data Loading and Grouping (UPDATED with flexible regex) ---
    # =====================================================================
    def _group_files_by_id(self):
        """Groups NIfTI files by RAW_id, handles C/S view, and T0/T3 labels."""
        files = os.listdir(self.dataPath)
        sequences = {} # {RAW_ID: {'C': frontal_file, 'S': lateral_file, 'Label': value}}
        
        # Regex: Captures RAW_ID, T0/T3, and C/S, allowing any characters (.*) afterwards.
        # Group 1: RAW_ID (e.g., RAW_0000)
        # Group 2: T0 or T3 (The label tag)
        # Group 5: C or S (The view tag)
        file_pattern = re.compile(r'(RAW_\d+)_((T0)|(T3))_(C|S).*?\.nii') 
        
        for filename in files:
            match = file_pattern.match(filename)
            if match:
                raw_id = match.group(1)   
                label_tag = match.group(2) 
                view = match.group(5)      
                
                if raw_id not in sequences:
                    sequences[raw_id] = {'C': None, 'S': None, 'Label': None}
                
                # Determine the common label for the pair
                label_value = THROMBUS_YES if label_tag == 'T0' else THROMBUS_NO
                sequences[raw_id]['Label'] = label_value 
                
                # Store by view type
                if view == 'C': # 'C' for Coronal/Frontal
                    sequences[raw_id]['C'] = filename
                elif view == 'S': # 'S' for Sagittal/Lateral
                    sequences[raw_id]['S'] = filename
        
        # Filter: Keep only complete pairs (C and S)
        final_sequences = {}
        for raw_id, data in sequences.items():
             if data['C'] and data['S']:
                final_sequences[raw_id] = {
                    'F_file': data['C'],
                    'L_file': data['S'],
                    'Label': data['Label']
                }
        
        return final_sequences # Returns dictionary keyed by RAW_ID

    def _create_dataset_dict(self, sequences):
        """Creates the final list of dictionaries for the DataLoader."""
        dataset_list = []
        for _, data in sequences.items():
            # Keypoints are placeholders but required by DataAugmentation.py
            keypoint_placeholder = [(1, 1)] 
            
            entry = {
                'filename': data['F_file'], 
                'keypoints': keypoint_placeholder, 
                'filenameOtherView': data['L_file'],
                'keypointsOtherView': keypoint_placeholder,
                'frontalAndLateralView': True,
                'target_label': data['Label']
            }
            dataset_list.append(entry)
        return dataset_list
    
    # =====================================================================
    # --- Core Dataset Methods ---
    # =====================================================================
    def __len__(self):
        return len(self.datasetDict)

    def __getitem__(self, index):
        entry = self.datasetDict[index]
        
        # 1. Load data
        image_data_f = nibabel.load(os.path.join(self.dataPath, entry['filename'])).get_fdata(dtype=np.float32) * 0.062271062
        image_data_f = image_data_f.astype(np.uint8)
        image_data_f = ImageUtils.fillBlackBorderWithRandomNoise(image_data_f, 193)
        
        image_data_l = nibabel.load(os.path.join(self.dataPath, entry['filenameOtherView'])).get_fdata(dtype=np.float32) * 0.062271062 
        image_data_l = image_data_l.astype(np.uint8)
        image_data_l = ImageUtils.fillBlackBorderWithRandomNoise(image_data_l, 193)
        
        # 2. Dynamic Sequence Padding (Pad the smaller sequence to match the larger)
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

        # 2b. Ensure frontal and lateral have the same HxW before Albumentations Compose
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
        
        # 3. Create data dictionary for DataAugmentation
        data = {
            'image': image_data_f,
            'keypoints': entry['keypoints'],
            'imageOtherView': image_data_l,
            'keypointsOtherView': entry['keypointsOtherView'],
            'frontalAndLateralView': True,
            'imageMean': 0, 'imageOtherViewMean': 0,
            'imageStd': 1.0, 'imageOtherViewStd': 1.0
        }

        # 4. Apply transformations, padding, and normalization
        augmentation = DataAugmentation(data=data)
        
        if self.training:
            augmentation.createTransformTraining()
        else:
            augmentation.createTransformValidation()
            
        augmentation.applyTransform()
        augmentation.zeroPaddingEqualLength()
        augmentation.normalizeData()

        # 5. Final conversion and label return
        tensor_data = augmentation.convertToTensor()
        tensor_data['target_label'] = torch.tensor(entry['target_label'], dtype=torch.float).unsqueeze(0)
        # Include file paths for evaluation tracking
        tensor_data['filename'] = entry['filename']
        tensor_data['filenameOtherView'] = entry['filenameOtherView']
        
        return tensor_data

    @staticmethod
    def split_data(full_data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """Performs stratified split of sequence IDs into train, val, and test sets."""
        temp_dataset = FineTuneDsaDataset(full_data_path)
        all_sequences = temp_dataset._group_files_by_id()
        all_ids = list(all_sequences.keys())
        labels = [all_sequences[id]['Label'] for id in all_ids]

        # 1. Split off Test set (approx. 15%)
        n_sequences = len(all_ids)
        n_test = int(test_ratio * n_sequences)
        train_val_ids, test_ids, train_val_labels, _ = train_test_split(
            all_ids, labels, test_size=n_test, random_state=random_state, shuffle=True, stratify=labels
        )
        
        # 2. Split Train/Val from the remainder (approx. 70%/15%)
        n_train = int(train_ratio * n_sequences)
        val_size_ratio = (n_sequences - n_train - n_test) / len(train_val_ids)
        train_ids, val_ids, _, _ = train_test_split(
            train_val_ids, train_val_labels, test_size=val_size_ratio, random_state=random_state, shuffle=True, stratify=train_val_labels
        )
        
        return train_ids, val_ids, test_ids