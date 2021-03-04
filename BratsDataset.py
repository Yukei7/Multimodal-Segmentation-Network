import glob
import torch.utils.data as data
import h5py


class BratsDataset(data.Dataset):
    def __init__(self, h5folder, shuffle=True):
        self.h5list = glob.glob(h5folder)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.h5list)

    def __getitem__(self, index):
        h5file = self.h5list[index]
        f = h5py.File(h5file)
        X, y = f.get('X'), f.get('y')
        f.close()
        return X, y
