from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from monai.data import Dataset


def get_brain_dataet(args, transform):
    data = []

    data_path = "Z:/Users/kimyw/data/sample"

    feat_df = pd.read_csv("nfeats_global.csv", index_col="subject")
    loc_df = pd.read_csv("nfeats_local.csv", index_col="subject")
    rad_df = pd.read_csv("radiomics_texture.csv", index_col="subject")

    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        image = os.path.join(sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(sub_path, "mri/aparc+aseg.nii.gz")
        if not os.path.isfile(image): continue
        if subject not in feat_df.index or subject not in rad_df.index: continue
        features = np.concatenate([feat_df.loc[subject].values, loc_df.loc[subject].values]).reshape(1, -1)
        radiomics = rad_df.loc[subject].values.reshape(1, -1)
        data.append({"image": image, "label": atlas, "features": features, "radiomics": radiomics})

    print("num of subject:", len(data))

    return Dataset(data=data, transform=transform)


if __name__ == "__main__":
    get_brain_dataet()