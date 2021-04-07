import torch.utils.data as data
import nibabel as nib
import os
import numpy as np
import pandas as pd
import glob
from skimage.transform import resize
from transforms import get_augmentations
from progressbar import ProgressBar
import pickle
from utils import minmax_normalize


class BratsDataset(data.Dataset):
    def __init__(self, folder, fileidx, stat_file="train_ds.pkl", modal='all', phase="train",
                 offset=0.1, mul_factor=100):
        self.df = pd.read_csv(os.path.join(folder, "name_mapping.csv"))
        # get files path and related ids
        self.files, self.files_id = self.get_file_list(folder)
        assert modal in ['all', 't1', 't1ce', 'flair', 't2']
        assert phase in ["train", "test"]
        self.phase = phase
        self.modal = ['t1', 't1ce', 'flair', 't2'] if modal == 'all' else [modal]
        self.n_modals = len(self.modal)
        # get file from file idx
        self.files, self.files_id = self.files[fileidx], self.files_id[fileidx]
        # get the stat for normalization
        self.avg_std_values = self.get_ds_stat(stat_file)
        # augmentation
        self.transform = get_augmentations(self.phase, self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        X, y = self.read_and_preprocess(file)
        # test for single modal and multimodal
        # if self.modal == 't1':
        #     X = X[0, :, :, :]
        # elif self.modal == 't1ce':
        #     X = X[1, :, :, :]
        # elif self.modal == 'flair':
        #     X = X[2, :, :, :]
        # elif self.modal == 't2':
        #     X = X[3, :, :, :]
        return {"image": X,
                "mask": y,
                "id": self.files_id[item]}

    def get_file_list(self, folder):
        grade = self.df["Grade"]
        ids = self.df["BraTS_2019_subject_ID"]
        files_list = []
        for idx in range(len(grade)):
            files_list.append(os.path.join(os.getcwd(), folder, grade[idx], ids[idx]))
        return pd.array(files_list), ids.values

    def _read_nii(self, path):
        data = nib.load(path)
        data = np.asarray(data.dataobj)
        return data

    def read_and_preprocess(self, file):
        X = []
        for x in self.modal:
            path = glob.glob(os.path.join(file, r"*" + x + r".nii.gz"))[0]
            im = self._read_nii(path)
            im = self.normalize(im, modal=x)
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

        if self.transform and self.phase == "train":
            augmented = self.transform(image=X.astype(np.float32),
                                       mask=y.astype(np.float32))

            X = augmented['image']
            y = augmented['mask']

        return X, y

    def normalize(self, data: np.ndarray, modal, offset=0.1, mul_factor=100):
        avg, std = self.avg_std_values[modal + "_avg"], self.avg_std_values[modal + "_std"]
        brain_index = np.nonzero(data)
        norm_data = np.copy(data)
        norm_data[brain_index] = mul_factor * \
                                 (minmax_normalize((data[brain_index] - avg) / std) + offset)
        return norm_data

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

    def get_ds_stat(self, stat_file):
        if os.path.exists(stat_file):
            print("stat file found, load from pickle.")
            with open(stat_file, "rb") as f:
                mean_std_values = pickle.load(f)
            return mean_std_values
        if not os.path.exists(stat_file) and self.phase == "test":
            raise FileNotFoundError("Constructing test dataloader, normalize stat file not found.")

        print('Getting training set statistics...')
        mean_std_values = {}
        modals = ["t1", "t1ce", "flair", "t2"] if self.modal == "all" else self.modal
        for x in modals:
            im_tot, im_tot2 = [], []
            print(f"Calculate mean and std for modal: {x}")

            pbar = ProgressBar().start()
            n_files = len(self.files)
            for i, file in enumerate(self.files):
                modal_path = glob.glob(os.path.join(file, r"*" + x + r".nii.gz"))[0]
                im = self._read_nii(modal_path)
                # some overflow problem here
                im = im.astype(np.int32)
                im_tot.append(np.mean(np.ravel(im)[np.flatnonzero(im)]))
                im_tot2.append(np.mean(np.ravel(im)[np.flatnonzero(im)] ** 2))
                pbar.update(int(i * 100 / (n_files - 1)))
            pbar.finish()

            im_avg = np.mean(np.array(im_tot))
            im_std = np.sqrt(np.mean(np.array(im_tot2)) - im_avg ** 2)
            mean_std_values[x + '_avg'] = im_avg
            mean_std_values[x + '_std'] = im_std
            print(f"mean={im_avg}, std={im_std}")
        with open(stat_file, 'wb') as f:
            pickle.dump(mean_std_values, f)
        return mean_std_values


if __name__ == "__main__":
    dataset = BratsDataset('brats19')
    print(len(dataset))
    sample = dataset[0]
    print(sample.get("image").shape, sample.get("mask").shape, sample.get("id"))
    print(type(sample.get("image")))
