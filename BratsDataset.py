import torch.utils.data as data
import nibabel as nib
import os
import numpy as np
import pandas as pd
import glob
from skimage.transform import resize


class BratsDataset(data.Dataset):
    def __init__(self, folder, modal='all', transform=None):
        self.df = pd.read_csv(os.path.join(folder, "name_mapping.csv"))
        self.files, self.files_id = self.get_file_list(folder)
        self.transform = transform
        self.modal = modal
        assert self.modal in ['all', 't1', 't1ce', 'flair', 't2']
        self.n_modals = 4 if self.modal == 'all' else 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        X, y = self.read_and_preprocess(file)
        # test for single modal and multimodal
        if self.modal == 't1':
            X = X[0, :, :, :]
        elif self.modal == 't1ce':
            X = X[1, :, :, :]
        elif self.modal == 'flair':
            X = X[2, :, :, :]
        elif self.modal == 't2':
            X = X[3, :, :, :]
        return {"image": X,
                "mask": y,
                "id": self.files_id[item]}

    def get_file_list(self, folder):
        grade = self.df["Grade"]
        ids = self.df["BraTS_2019_subject_ID"]
        files_list = []
        for idx in range(len(grade)):
            files_list.append(os.path.join(os.getcwd(), folder, grade[idx], ids[idx]))
        return files_list, ids

    def _read_nii(self, path):
        data = nib.load(path)
        data = np.asarray(data.dataobj)
        return data

    def read_and_preprocess(self, file):
        X_path = [glob.glob(os.path.join(file, r"*" + x + r".nii.gz"))[0]
                  for x in ["t1", "t1ce", "flair", "t2"]]
        X = []
        for modal_path in X_path:
            im = self._read_nii(modal_path)
            im = self.normalize(im)
            im = self.resize(im)
            X.append(im)

        X = np.array(X)
        X = np.moveaxis(X, (0, 1, 2, 3), (0, 3, 2, 1))
        y_path = glob.glob(os.path.join(file, r"*seg.nii.gz"))[0]
        y = self._read_nii(y_path)
        y = self.resize(y)
        y = np.clip(y.astype(np.uint8), 0, 1).astype(np.float32)
        y = np.clip(y, 0, 1)
        y = self.preprocess_mask_labels(y)

        if self.transform:
            augmented = self.transform(image=X.astype(np.float32),
                                       mask=y.astype(np.float32))

            X = augmented['image']
            y = augmented['mask']

        return X, y

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


if __name__ == "__main__":
    dataset = BratsDataset('brats19')
    print(len(dataset))
    sample = dataset[0]
    print(sample.get("image").shape, sample.get("mask").shape, sample.get("id"))
    print(type(sample.get("image")))
