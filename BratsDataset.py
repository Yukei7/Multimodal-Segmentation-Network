import torch.utils.data as data
import nibabel as nib
import os
import numpy as np
import pandas as pd
import glob
from skimage.transform import resize
from transforms import get_augmentations3d
from progressbar import ProgressBar
import pickle
from utils import minmax_normalize


class BratsDataset(data.Dataset):
    def __init__(self, folder, fileidx, stat_file="train_ds.pkl", modal='all', phase="train",
                 offset=0.1, mul_factor=100, patch_size=(78, 120, 120)):
        self.folder = folder
        self.df = pd.read_csv(os.path.join(folder, "name_mapping.csv"))
        # get files path and related ids
        self.files, self.files_id, self.grades = self.get_file_list(folder)
        assert modal in ['all', 't1', 't1ce', 'flair', 't2']
        assert phase in ["train", "test"]
        self.phase = phase
        self.modal = ['t1', 't1ce', 'flair', 't2'] if modal == 'all' else [modal]
        self.n_modals = len(self.modal)
        # get file from file idx
        self.files = self.files[fileidx]
        self.files_id = self.files_id[fileidx]
        self.grades = self.grades[fileidx]
        # get the stat for normalization
        self.avg_std_values = self.get_ds_stat(stat_file)
        self.offset = offset
        self.mul_factor = mul_factor
        # augmentation
        self.transform = get_augmentations3d(self.phase, patch_size=patch_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        X, y = self.read_and_preprocess(file)
        return {"image": X,
                "mask": y,
                "id": self.files_id[item]}

    def get_file_list(self, folder):
        grade = self.df["Grade"]
        ids = self.df["BraTS_2019_subject_ID"]
        files_list = []
        for idx in range(len(grade)):
            files_list.append(os.path.join(os.getcwd(), folder, grade[idx], ids[idx]))
        return pd.array(files_list), ids.values, grade.values

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
            X.append(im)

        X = np.array(X)
        # [c, x, y, z]
        X = np.moveaxis(X, (0, 1, 2, 3), (0, 3, 2, 1))
        y_path = glob.glob(os.path.join(file, r"*seg.nii.gz"))[0]
        y = self._read_nii(y_path)
        y = np.clip(y.astype(np.uint8), 0, 1).astype(np.float32)
        y = np.clip(y, 0, 1)
        y = self.preprocess_mask_labels(y)
        y = np.moveaxis(y, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.transform and self.phase == "train":
            data = {'image': X, 'mask': y}
            aug_data = self.transform(**data)
            X, y = aug_data['image'], aug_data['mask']

        return X, y

    def normalize(self, data: np.ndarray, modal):
        avg, std = self.avg_std_values[modal + "_avg"], self.avg_std_values[modal + "_std"]
        brain_index = np.nonzero(data)
        norm_data = np.copy(data)
        norm_data[brain_index] = self.mul_factor * \
                                 (minmax_normalize((data[brain_index] - avg) / std) + self.offset)
        return norm_data

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
        return mask

    def get_ds_stat(self, stat_file):
        pkl_path = os.path.join(self.folder, stat_file)
        if os.path.exists(pkl_path):
            print(f"{self.phase}: stat file found, load from pickle.")
            with open(pkl_path, "rb") as f:
                mean_std_values = pickle.load(f)
            return mean_std_values
        if not os.path.exists(pkl_path) and self.phase == "test":
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


# TODO: 2d dataset class
class BratsDataset2d(BratsDataset):
    def __init__(self, folder, fileidx, stat_file="train_ds.pkl", modal='all', phase="train",
                 offset=0.1, mul_factor=100, final_patch=(78, 120, 120)):
        super(BratsDataset2d, self).__init__(folder, fileidx, stat_file, modal, phase, offset, mul_factor, final_patch)
        self.imgs, self.labels = self.read_2d(self.files, self.modal)

    def __len__(self):
        return self.imgs.shape[0]

    def read_wd(self, files, modal):
        pass


if __name__ == "__main__":
    pass
