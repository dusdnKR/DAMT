import os
import numpy as np
import pandas as pd
from monai.data import Dataset


def _load_subjects_from_csv(csv_path):
    if not csv_path or not os.path.isfile(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return set()

    for col in ["subject", "subjects", "subj", "participant_id", "id"]:
        if col in df.columns:
            return set(df[col].dropna().astype(str).str.strip())

    return set(df.iloc[:, 0].dropna().astype(str).str.strip())


def get_brain_dataet(args, transform):
    data = []

    data_path = args.data_path
    selected_subjects = _load_subjects_from_csv(getattr(args, "outlier_path", None))
    all_subjects = [
        s for s in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, s))
    ]
    filtered_subjects = [
        s for s in all_subjects
        if selected_subjects is None or str(s) in selected_subjects
    ]

    print(f"subjects before csv filter: {len(all_subjects)}")
    if selected_subjects is None:
        print(f"subjects after csv filter: {len(filtered_subjects)} (csv not applied)")
    else:
        print(f"subjects listed in csv: {len(selected_subjects)}")
        print(f"subjects after csv filter: {len(filtered_subjects)}")

    feat_df = pd.read_csv(os.path.join(data_path, "nfeats_global.csv"), index_col="subject").fillna(0)
    loc_df = pd.read_csv(os.path.join(data_path, "nfeats_local.csv"), index_col="subject").fillna(0)
    rad_df = pd.read_csv(os.path.join(data_path, "radiomics_texture.csv"), index_col="subject").fillna(0)

    # Z-score standardise radiomics so texture_loss has the same scale as other tasks
    rad_mean = rad_df.mean()
    rad_std  = rad_df.std().replace(0, 1)   # avoid division by zero
    rad_df   = ((rad_df - rad_mean) / rad_std).fillna(0)

    for subject in filtered_subjects:
        sub_path = os.path.join(data_path, subject)
        image = os.path.join(sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(sub_path, "mri/aparc+aseg.nii.gz")
        if not os.path.isfile(image) or not os.path.isfile(atlas): continue
        if subject not in feat_df.index or subject not in loc_df.index or subject not in rad_df.index: continue
        features = np.concatenate([feat_df.loc[subject].values, loc_df.loc[subject].values]).reshape(1, -1)
        radiomics = rad_df.loc[subject].values.reshape(1, -1)
        data.append({"image": image, "label": atlas, "features": features, "radiomics": radiomics})

    print("subjects after data validity checks:", len(data))

    return Dataset(data=data, transform=transform)


if __name__ == "__main__":
    get_brain_dataet()