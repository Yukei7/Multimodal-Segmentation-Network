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
import h5py
from utils import minmax_normalize
import multiprocessing

N_CLASSES = 3


class BratsDataset(data.Dataset):
    def __init__(self, folder, fileidx, stat_file="train_ds.pkl", modal='all', phase="train",
                 offset=0.1, mul_factor=100, reduce=1.0, patch_size=64, patch_file="patch.h5"):
        self.folder = folder
        self.df = pd.read_csv(os.path.join(folder, "name_mapping.csv"))
        # get files path and related ids
        self.files, self.files_id, self.files_grades = self.get_file_list(folder)
        assert modal in ['all', 't1', 't1ce', 'flair', 't2']
        assert phase in ["train", "test"]
        self.phase = phase
        self.modal = ['t1', 't1ce', 'flair', 't2'] if modal == 'all' else [modal]
        self.n_modals = len(self.modal)
        # get file from file idx
        self.files = self.files[fileidx]
        self.files_id = self.files_id[fileidx]
        self.files_grades = self.files_grades[fileidx]
        # get the stat for normalization
        self.avg_std_values = self.get_ds_stat(stat_file)
        self.offset = offset
        self.mul_factor = mul_factor
        # augmentation
        self.transform = get_augmentations3d(self.phase)
        # crop into patches and form a new dataset
        # shape: imgs=[n, c, x, y, z], masks=[n, n_classes, x, y, z]
        # grade=[n] for stratified
        self.reduce = reduce
        self.patch_size = patch_size

        # read from pkl if cached
        patch_path = os.path.join(folder, self.phase + "_ps" + str(self.patch_size) + "_" + patch_file)
        if os.path.exists(patch_path):
            f = h5py.File(patch_path, "r")
            self.imgs = np.array(f[self.phase + "_imgs"][:])
            self.masks = np.array(f[self.phase + "_masks"][:])
            self.grades = np.array(f[self.phase + "_grades"][:])
            f.close()
        else:
            self.imgs, self.masks, self.grades = self.patches_generator()
            f = h5py.File(patch_path, "w")
            f.create_dataset(self.phase + "_imgs", data=self.imgs, dtype=np.float32)
            f.create_dataset(self.phase + "_masks", data=self.masks, dtype=np.float32)
            f.create_dataset(self.phase + "_grades", data=self.grades, dtype=str)
            f.close()

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        return {"image": self.imgs[item],
                "mask": self.masks[item],
                "grade": self.grades[item]}

    def get_file_list(self, folder):
        grade = self.df["Grade"]
        ids = self.df["BraTS_2019_subject_ID"]
        files_list = []
        for idx in range(len(grade)):
            files_list.append(os.path.join(os.getcwd(), folder, grade[idx], ids[idx]))
        return pd.array(files_list), ids.values, grade.values

    def _read_nii(self, path):
        x = nib.load(path)
        x = np.asarray(x.dataobj)
        return x

    def patches_generator(self):
        imgs, masks, grades = [], [], []
        print(f"Generating {self.phase} set patches")
        pbar = ProgressBar().start()
        n_files = len(self.files)
        for i, file in enumerate(self.files):
            X, y = self.read_patient_files(file)
            # get patches [c, x, y, z] -> [num_patches, c, x, y, z]
            patches_X, patches_y = self.get_patches_from_patient(X, y)

            if self.transform is not None and self.phase == "train":
                augmented = self.transform(image=patches_X.astype(np.float32),
                                           mask=patches_y.astype(np.float32))
                patches_X = augmented['image']
                patches_y = augmented['mask']
            grade = [self.files_grades[i]] * patches_X.shape[0]
            imgs.append(patches_X)
            masks.append(patches_y)
            grades.append(grade)
            pbar.update(int(i * 100 / (n_files - 1)))

        imgs = np.array(imgs).reshape((-1, self.n_modals, self.patch_size, self.patch_size, self.patch_size))
        masks = np.array(masks).reshape((-1, 1, self.patch_size, self.patch_size, self.patch_size))
        grades = np.array(grades).flatten()
        return imgs, masks, grades

    def read_patient_files(self, file):
        X = []
        for x in self.modal:
            path = glob.glob(os.path.join(file, r"*" + x + r".nii.gz"))[0]
            im = self._read_nii(path)
            im = self.normalize(im, modal=x)
            X.append(im)
        # [c, x, y, z]
        X = np.array(X)
        X = np.moveaxis(X, (0, 1, 2, 3), (0, 3, 2, 1))

        y_path = glob.glob(os.path.join(file, r"*seg.nii.gz"))[0]
        y = self._read_nii(y_path)
        y = self.preprocess_mask_labels(y)
        y = np.moveaxis(y, (0, 1, 2, 3), (0, 3, 2, 1))
        return X, y

    def get_patches_from_patient(self, img, mask):
        # img: [c, x, y, z], convert to [x, y, z, c]
        img = np.transpose(img, (1, 2, 3, 0))
        mask = np.transpose(mask, (1, 2, 3, 0))

        # add padding to make image dividable into tiles
        shape = img.shape
        nums = round(self.reduce * self.patch_size)
        # split into small block, then reduce the scale
        pad0 = (nums - shape[0] % nums) % nums
        pad1 = (nums - shape[1] % nums) % nums
        pad2 = (nums - shape[2] % nums) % nums

        img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2],
                           [pad1 // 2, pad1 - pad1 // 2],
                           [pad2 // 2, pad2 - pad2 // 2],
                           [0, 0]],
                     constant_values=0)
        mask = np.pad(mask, [[pad0 // 2, pad0 - pad0 // 2],
                             [pad1 // 2, pad1 - pad1 // 2],
                             [pad2 // 2, pad2 - pad2 // 2],
                             [0, 0]],
                      constant_values=0)

        # TODO: Reduce size, use skimage instead here to handle 3d images
        # img = cv2.resize(img, (img.shape[1] // self.reduce, img.shape[0] // self.reduce),
        #                  interpolation=cv2.INTER_AREA)
        # mask = cv2.resize(mask, (mask.shape[1] // self.reduce, mask.shape[0] // self.reduce),
        #                   interpolation=cv2.INTER_NEAREST)

        # Transform into small block
        # [n, x, y, z, c]
        patches_X = self.get_patches(img, is_label=False)
        patches_y = self.get_patches(mask, is_label=True)

        # convert from [n, x, y, z, c] to [n, c, x, y, z]
        patches_X = np.transpose(patches_X, (0, 4, 1, 2, 3))
        patches_y = np.transpose(patches_y, (0, 4, 1, 2, 3))

        return patches_X, patches_y

    def get_patches(self, x: np.ndarray, is_label=True):
        # x,x_s,y,y_s,z,z_s,C
        if is_label:
            n_channels = N_CLASSES
        else:
            n_channels = self.n_modals
        x = np.reshape(x, (x.shape[0] // self.patch_size,
                           self.patch_size,
                           x.shape[1] // self.patch_size,
                           self.patch_size,
                           x.shape[2] // self.patch_size,
                           self.patch_size,
                           n_channels))
        x = np.transpose(x, (0, 2, 4, 1, 3, 5, 6))
        x = np.reshape(x, (-1, self.patch_size, self.patch_size, self.patch_size, n_channels))
        return x

    def normalize(self, data: np.ndarray, modal):
        avg, std = self.avg_std_values[modal + "_avg"], self.avg_std_values[modal + "_std"]
        brain_index = np.nonzero(data)
        norm_data = np.copy(data)
        norm_data[brain_index] = self.mul_factor * \
                                 (minmax_normalize((data[brain_index] - avg) / std) + self.offset)
        return norm_data

    def resize(self, data: np.ndarray):
        data = resize(data, self.final_patch, preserve_range=True)
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
        return mask

    def get_ds_stat(self, stat_file):
        path = os.path.join(self.folder, stat_file)
        if os.path.exists(path):
            print(f"{self.phase}: stat file found, load from pickle.")
            with open(path, "rb") as f:
                mean_std_values = pickle.load(f)
            return mean_std_values
        if not os.path.exists(path) and self.phase == "test":
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
        with open(path, 'wb') as f:
            pickle.dump(mean_std_values, f)
        return mean_std_values


# TODO: 2d dataset class
# class BratsDataset2d(BratsDataset)
if __name__ == "__main__":
    pass
