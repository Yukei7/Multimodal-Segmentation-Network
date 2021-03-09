import torch.utils.data as data
import h5py
import os
import torch
import numpy as np


class BratsDataset(data.Dataset):
    def __init__(self, h5folder, transform=None):
        self.h5list = os.listdir(h5folder)
        self.h5files = [os.path.join(h5folder, x) for x in self.h5list]
        self.transform = transform

    def __len__(self):
        return len(self.h5list)

    def __getitem__(self, item):
        h5file = self.h5files[item]
        f = h5py.File(h5file, 'r')
        X, y = f['X'][:], f['y'][:]
        f.close()
        X, y = self.preprocess(X, y)
        return torch.from_numpy(X / 1.0), torch.from_numpy(y / 1.0)

    def preprocess(self, data, label):
        c, h, w = data.shape
        h_gt, w_gt = label.shape
        # make sure that the data and label have same size
        assert h == h_gt, "Error"
        assert w == w_gt, "Error"

        if self.transform:
            data, label = self.transform(data, label)
        label = label[np.newaxis, :, :]
        return data, label


if __name__ == "__main__":
    dataset = BratsDataset('brats19_h5')
    print(len(dataset))
    X_sample, y_sample = dataset[0]
    print(X_sample.shape, y_sample.shape)