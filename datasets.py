import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from monai.data import Dataset


# ── Subject list from txt ─────────────────────────────────────────────────────

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


# ── participants.tsv (age / sex) ──────────────────────────────────────────────

def _parse_subject_dir(dirname):
    """Extract (dataset, participant_id, session_id) from a subject directory name.

    Handles optional _run-X suffix that appears between session and _T1w:
      PT028_OASIS1_sub-001_ses-01_run-1_T1w  →  (PT028_OASIS1, sub-001, ses-01)
      PT021_IXI_sub-001_ses-01_T1w           →  (PT021_IXI,    sub-001, ses-01)
    Returns None if the pattern is not recognised.
    """
    name = re.sub(r"_T1w$", "", dirname)
    name = re.sub(r"_run-\d+", "", name)
    m = re.match(r"^(.+?)_(sub-[^_]+)_(ses-[^_]+)$", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _load_participants(data_path):
    """Load participants.tsv and return per-subject age/sex lookup.

    Returns
    -------
    tsv_lookup : dict  (dataset, participant_id, session_id) -> (z_age, sex_int)
                 z_age  : float or np.nan  (z-scored across all valid ages in TSV)
                 sex_int: 0=F, 1=M, -1=unknown
    age_mean, age_std : floats used for z-scoring (also stored on args for linear_probe)
    """
    tsv_path = os.path.join(data_path, "participants.tsv")
    if not os.path.isfile(tsv_path):
        print("participants.tsv : not found — age/sex tasks disabled")
        return {}, 0.0, 1.0

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)

    raw_ages = []
    raw = {}
    for _, row in df.iterrows():
        dataset = str(row.get("dataset", "")).strip()
        sub     = str(row.get("participant_id", "")).strip()
        ses     = str(row.get("session_id", "")).strip()
        if not dataset or not sub or not ses:
            continue

        age_str = str(row.get("age", "")).strip()
        try:
            age = float(age_str) if age_str and age_str.lower() not in ("nan", "na", "") else np.nan
        except ValueError:
            age = np.nan

        sex_str = str(row.get("sex", "")).strip().upper()
        sex = 1 if sex_str == "M" else (0 if sex_str == "F" else -1)

        raw[(dataset, sub, ses)] = (age, sex)
        if not np.isnan(age):
            raw_ages.append(age)

    age_mean = float(np.mean(raw_ages)) if raw_ages else 0.0
    age_std  = float(np.std(raw_ages))  if raw_ages else 1.0
    if age_std < 1e-6:
        age_std = 1.0

    tsv_lookup = {}
    for key, (age, sex) in raw.items():
        z_age = (age - age_mean) / age_std if not np.isnan(age) else np.nan
        tsv_lookup[key] = (z_age, sex)

    n_age = sum(1 for (z, _) in tsv_lookup.values() if not np.isnan(z))
    n_sex = sum(1 for (_, s) in tsv_lookup.values() if s >= 0)
    print(f"participants.tsv : {len(tsv_lookup)} entries  |  age={n_age}  sex={n_sex}  "
          f"(age mean={age_mean:.1f}, std={age_std:.1f})")
    return tsv_lookup, age_mean, age_std


# ── MSN loader ────────────────────────────────────────────────────────────────

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


# ── Dataset builder ───────────────────────────────────────────────────────────

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

    # ── Tabular feature CSVs ──────────────────────────────────────────────────
    feat_df = pd.read_csv(os.path.join(data_path, "results/nfeats_global.csv"), index_col="subject").fillna(0)
    loc_df  = pd.read_csv(os.path.join(data_path, "results/nfeats_local.csv"),  index_col="subject").fillna(0)
    rad_df  = pd.read_csv(os.path.join(data_path, "results/radiomics_texture.csv"), index_col="subject").fillna(0)

    # Z-score standardise radiomics so texture_loss has the same scale as other tasks
    rad_mean = rad_df.mean()
    rad_std  = rad_df.std().replace(0, 1)
    rad_df   = ((rad_df - rad_mean) / rad_std).fillna(0)

    # ── MSN ───────────────────────────────────────────────────────────────────
    msn_dir = getattr(args, "msn_dir", None)
    msn_n   = getattr(args, "msn_n_regions", 62)
    msn_dim = msn_n * (msn_n - 1) // 2
    if msn_dir:
        print(f"MSN dir : {msn_dir}  (n_regions={msn_n}, vec_dim={msn_dim})")
    else:
        print("MSN dir : not set — msn_loss will be skipped")

    # ── Hemisphere asymmetry ──────────────────────────────────────────────────
    asym_csv = getattr(args, "asym_csv", None)
    if asym_csv and os.path.isfile(asym_csv):
        asym_df  = pd.read_csv(asym_csv, index_col="subject").fillna(0)
        asym_dim = len(asym_df.columns)
        args.asym_dim = asym_dim   # expose for SSLHead_Swin init
        print(f"Asym CSV : {asym_csv}  ({len(asym_df)} subjects, dim={asym_dim})")
    else:
        asym_df  = None
        args.asym_dim = getattr(args, "asym_dim", 0)
        print("Asym CSV : not set — asym_loss will be skipped")

    # ── participants.tsv (age / sex) ──────────────────────────────────────────
    tsv_lookup, age_mean, age_std = _load_participants(data_path)
    args.age_mean = age_mean   # expose for linear_probe normalisation
    args.age_std  = age_std

    # ── Per-subject loop ──────────────────────────────────────────────────────
    for subject in filtered_subjects:
        sub_path = os.path.join(data_path, subject)
        image = os.path.join(sub_path, "mri/brainmask.nii.gz")
        atlas = os.path.join(sub_path, "mri/aparc+aseg.nii.gz")
        if not os.path.isfile(image) or not os.path.isfile(atlas):
            continue
        if subject not in feat_df.index or subject not in loc_df.index or subject not in rad_df.index:
            continue

        features  = np.concatenate([feat_df.loc[subject].values,
                                     loc_df.loc[subject].values]).reshape(1, -1)
        radiomics = rad_df.loc[subject].values.reshape(1, -1)

        # MSN
        if msn_dir:
            msn = _load_msn(os.path.join(msn_dir, f"{subject}.mat"), msn_dim)
        else:
            msn = np.zeros(msn_dim, dtype=np.float32)

        # Asymmetry (zero-vector sentinel → remove_zerotensor skips it if disabled)
        if asym_df is not None and subject in asym_df.index:
            asym = asym_df.loc[subject].values.astype(np.float32)
        else:
            asym = np.zeros(args.asym_dim, dtype=np.float32)

        # Age / sex from participants.tsv (NaN / -1 when missing)
        parsed = _parse_subject_dir(subject)
        if parsed and parsed in tsv_lookup:
            z_age, sex = tsv_lookup[parsed]
        else:
            z_age = np.nan
            sex   = -1

        data.append({
            "image":    image,
            "label":    atlas,
            "features": features,
            "radiomics": radiomics,
            "msn":      msn,
            "asym":     asym,
            "age":      np.float32(z_age),   # NaN → missing; used by age_loss
            "sex":      np.int32(sex),        # -1  → missing; used by sex_loss
        })

    print("subjects after data validity checks:", len(data))
    return Dataset(data=data, transform=transform)


if __name__ == "__main__":
    get_brain_dataet()
