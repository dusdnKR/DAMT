import os
import numpy as np
import pandas as pd
import scipy.io as sio
from monai.data import Dataset


def _load_subjects_from_txt(data_path, data_name):
    if not data_name:
        return None, None

    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repo_root, "data", f"{data_name}.txt"),
        os.path.join(data_path, f"{data_name}.txt"),
    ]

    txt_path = None
    for path in candidates:
        if os.path.isfile(path):
            txt_path = path
            break

    if txt_path is None:
        return None, candidates[0]

    subjects = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            first = line.split(",", 1)[0].strip()
            subject = first.split()[0].strip()
            if subject:
                subjects.add(subject)
    return subjects, txt_path


def _load_msn(mat_path, msn_dim):
    """Return upper-triangle vector of the MSN connectivity matrix.

    Falls back to a zero vector if the file is missing or unreadable.
    Zero vectors are recognised downstream by remove_zerotensor and skipped.
    """
    try:
        conn = sio.loadmat(mat_path)["connectivity"]   # (n_regions, n_regions)
        idx = np.triu_indices(conn.shape[0], k=1)
        vec = conn[idx].astype(np.float32)
        if vec.shape[0] != msn_dim:
            return np.zeros(msn_dim, dtype=np.float32)
        return vec
    except Exception:
        return np.zeros(msn_dim, dtype=np.float32)


def get_brain_dataet(args, transform):
    data = []

    data_path = args.data_path
    selected_subjects, txt_file = _load_subjects_from_txt(data_path, getattr(args, "data", None))
    all_subjects = [
        s for s in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, s))
    ]
    filtered_subjects = [
        s for s in all_subjects
        if selected_subjects is None or str(s) in selected_subjects
    ]

    print(f"subjects before txt filter: {len(all_subjects)}")
    if selected_subjects is None:
        print(f"subjects after txt filter: {len(filtered_subjects)} (txt not applied: {txt_file})")
    else:
        print(f"txt applied: {txt_file}")
        print(f"subjects listed in txt: {len(selected_subjects)}")
        print(f"subjects after txt filter: {len(filtered_subjects)}")

    feat_df = pd.read_csv(os.path.join(data_path, "results/nfeats_global.csv"), index_col="subject").fillna(0)
    loc_df = pd.read_csv(os.path.join(data_path, "results/nfeats_local.csv"), index_col="subject").fillna(0)
    rad_df = pd.read_csv(os.path.join(data_path, "results/radiomics_texture.csv"), index_col="subject").fillna(0)

    # Z-score standardise radiomics so texture_loss has the same scale as other tasks
    rad_mean = rad_df.mean()
    rad_std  = rad_df.std().replace(0, 1)   # avoid division by zero
    rad_df   = ((rad_df - rad_mean) / rad_std).fillna(0)

    # MSN: optional. If msn_dir is set, load per-subject .mat files.
    msn_dir = getattr(args, "msn_dir", None)
    msn_n = getattr(args, "msn_n_regions", 62)
    msn_dim = msn_n * (msn_n - 1) // 2
    if msn_dir:
        print(f"MSN dir : {msn_dir}  (n_regions={msn_n}, vec_dim={msn_dim})")
    else:
        print("MSN dir : not set — msn_loss will be skipped")

    for subject in filtered_subjects:
        sub_path = os.path.join(data_path, subject)
        image = os.path.join(sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(sub_path, "mri/aparc+aseg.nii.gz")
        if not os.path.isfile(image) or not os.path.isfile(atlas): continue
        if subject not in feat_df.index or subject not in loc_df.index or subject not in rad_df.index: continue

        features = np.concatenate([feat_df.loc[subject].values, loc_df.loc[subject].values]).reshape(1, -1)
        radiomics = rad_df.loc[subject].values.reshape(1, -1)

        if msn_dir:
            msn = _load_msn(os.path.join(msn_dir, f"{subject}.mat"), msn_dim)
        else:
            msn = np.zeros(msn_dim, dtype=np.float32)

        data.append({"image": image, "label": atlas,
                     "features": features, "radiomics": radiomics, "msn": msn})

    print("subjects after data validity checks:", len(data))

    return Dataset(data=data, transform=transform)


if __name__ == "__main__":
    get_brain_dataet()
