import torch.utils.data as data
import h5py
import os
import torch


class BratsDataset(data.Dataset):
    def __init__(self, h5folder, list_ids):
        self.h5list = os.listdir(h5folder)
        self.list_ids = list_ids
        self.h5files = [os.path.join(h5folder, self.h5list[idx]) for idx in list_ids]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, item):
        index = self.list_ids[item]
        h5file = self.h5files[index]
        f = h5py.File(h5file)
        X, y = f['X'][:], f['y'][:]
        f.close()
        return torch.from_numpy(X), torch.from_numpy(y)


if __name__ == "__main__":
    dataset = BratsDataset('brats19_h5', [1, 2, 3, 4])
    print(len(dataset))
    X_sample, y_sample = dataset[0]
    print(X_sample.shape, y_sample.shape)
