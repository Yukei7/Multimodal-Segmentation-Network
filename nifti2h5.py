import h5py
import numpy as np
import pandas as pd
import argparse
import os
import SimpleITK as sitk
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="brats19")
parser.add_argument("--output", type=str, default="brats19_h5")
args = parser.parse_args()


def nift2array(nift_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(nift_path))


def main():
    # create output directory if not exists
    if not os.path.exists(os.path.join(os.getcwd(), args.output)):
        os.mkdir(os.path.join(os.getcwd(), args.output))

    # read path from the mapping csv file
    df = pd.read_csv(os.path.join(os.getcwd(), args.input, "name_mapping.csv"))
    grade = df["Grade"]
    id = df["BraTS_2019_subject_ID"]

    X_all, y_all = [], []
    # create h5 file for each sample
    for idx in tqdm(range(id.shape[0])):
        path = os.path.join(os.getcwd(), args.input, grade[idx], id[idx])
        y_path = glob.glob(os.path.join(path, r"*seg.nii.gz"))[0]
        y = nift2array(y_path)
        X_path = [glob.glob(os.path.join(path, r"*" + x + r".nii.gz"))[0]
                  for x in ["t1", "t1ce", "flair", "t2"]]
        X = np.array([nift2array(nift_path) for nift_path in X_path])
        X_all.append(X)
        y_all.append(y[np.newaxis, :, :])

    # write in h5 file
    h5_name = 'brats19.h5'
    f = h5py.File(os.path.join(os.getcwd(), args.output, h5_name), 'w')
    f['X'] = X_all
    f['y'] = y_all
    f.close()


if __name__ == '__main__':
    main()
