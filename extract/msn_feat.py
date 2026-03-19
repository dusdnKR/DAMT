"""Generate per-subject Morphological Similarity Networks (MSN) from FreeSurfer stats.

For each subject, per-region morphological feature values are read directly from
lh/rh.aparc.DKTatlas.mapped.stats files.  The MSN is the (n_regions × n_regions)
Pearson correlation matrix computed over each region's feature profile.

Output:  {data_path}/results/msn_{abbrs}/{subject}.mat
         Each .mat contains:
           - connectivity : (n_regions, n_regions) Pearson correlation matrix (MSN)
           - value        : (n_regions, n_features) z-scored feature matrix
           - regions      : (n_regions,) region name array
           - features     : (n_features,) feature name array

Available features and their folder abbreviations:
  SurfArea → sa   GrayVol  → gv   ThickAvg → ta   ThickStd → ts
  MeanCurv → mc   GausCurv → gc   FoldInd  → fi   CurvInd  → ci

Usage:
    python msn_feat.py --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz
    python msn_feat.py --data-path /path/to/data --features SurfArea GrayVol ThickAvg
    python msn_feat.py --data-path /path/to/data --data train_subjects
"""
import os
import argparse
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ── Feature metadata ─────────────────────────────────────────────────────────
# Key: feature name as it appears in FreeSurfer # ColHeaders line
# abbr: folder name abbreviation
FEAT_INFO = {
    "SurfArea": "sa",
    "GrayVol":  "gv",
    "ThickAvg": "ta",
    "ThickStd": "ts",
    "MeanCurv": "mc",
    "GausCurv": "gc",
    "FoldInd":  "fi",
    "CurvInd":  "ci",
}

DEFAULT_FEATURES = ["SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "GausCurv"]


# ── Stats file parser ─────────────────────────────────────────────────────────
def parse_stats_file(path):
    """Parse a FreeSurfer aparc stats file.

    Returns
    -------
    dict  {region_name: {feature_name: float}}
    """
    regions = {}
    col_names = None

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# ColHeaders"):
                # e.g. "# ColHeaders  StructName NumVert SurfArea GrayVol ..."
                col_names = line.split()[2:]  # ["StructName", "NumVert", "SurfArea", ...]
                continue
            if col_names is None:
                continue
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            region = parts[0]
            regions[region] = {}
            for i, col in enumerate(col_names[1:], start=1):  # skip "StructName"
                if i < len(parts):
                    try:
                        regions[region][col] = float(parts[i])
                    except ValueError:
                        pass

    return regions


# ── MSN computation ───────────────────────────────────────────────────────────
def compute_msn(lh_path, rh_path, features):
    """Compute MSN for one subject.

    Parameters
    ----------
    lh_path, rh_path : str  Paths to lh/rh .stats files.
    features         : list[str]  Feature names to include.

    Returns
    -------
    regions     : list[str]   Region labels (lh_* then rh_*)
    feat_z      : ndarray     (n_regions, n_features) z-scored feature matrix
    msn         : ndarray     (n_regions, n_regions) Pearson correlation matrix
    """
    lh_data = parse_stats_file(lh_path)
    rh_data = parse_stats_file(rh_path)

    lh_regions = sorted(lh_data.keys())
    rh_regions = sorted(rh_data.keys())
    region_labels = [f"lh_{r}" for r in lh_regions] + [f"rh_{r}" for r in rh_regions]
    n_regions = len(region_labels)
    n_feats = len(features)

    # Build (n_regions × n_features) raw feature matrix
    feat_matrix = np.zeros((n_regions, n_feats), dtype=np.float64)
    for i, label in enumerate(region_labels):
        hemi, rname = label.split("_", 1)
        src = lh_data if hemi == "lh" else rh_data
        for j, feat in enumerate(features):
            feat_matrix[i, j] = src[rname].get(feat, 0.0)

    # Z-score each feature column across regions (within this subject)
    scaler = StandardScaler()
    feat_z = scaler.fit_transform(feat_matrix)  # (n_regions, n_features)

    # Pearson correlation between region feature profiles → MSN
    msn = np.corrcoef(feat_z)  # (n_regions, n_regions)

    return region_labels, feat_z, msn


# ── Subject list loader ───────────────────────────────────────────────────────
def load_subject_list(data_path, data_name):
    """Load subject IDs from {data_path}/{data_name}.txt, or return None."""
    if not data_name:
        return None
    txt_path = os.path.join(data_path, f"{data_name}.txt")
    if not os.path.isfile(txt_path):
        print(f"[WARN] txt file not found: {txt_path} — processing all subjects")
        return None
    subjects = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            subject = line.split(",", 1)[0].strip().split()[0]
            if subject:
                subjects.add(subject)
    return subjects


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate MSN from FreeSurfer per-region features")
    parser.add_argument("--data-path", type=str, default="/NFS/Users/kimyw/data/fomo60k_wo_scz",
                        help="Root directory of the dataset (contains per-subject subdirs)")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                        choices=list(FEAT_INFO.keys()),
                        metavar="FEAT",
                        help=(
                            "Feature(s) to use for MSN. "
                            f"Choices: {list(FEAT_INFO.keys())}. "
                            f"Default: {DEFAULT_FEATURES}"
                        ))
    parser.add_argument("--data", type=str, default=None,
                        help="Subject list txt file name without extension (e.g. train_subjects)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-compute and overwrite existing .mat files")
    args = parser.parse_args()

    features = args.features
    abbrs = [FEAT_INFO[f] for f in features]
    folder_name = "msn_" + "_".join(abbrs)
    save_path = os.path.join(args.data_path, "results", folder_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Features : {features}")
    print(f"Abbrs    : {abbrs}")
    print(f"Save dir : {save_path}")

    selected_subjects = load_subject_list(args.data_path, args.data)
    all_subjects = sorted([
        s for s in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, s))
    ])

    if selected_subjects is not None:
        subjects = [s for s in all_subjects if s in selected_subjects]
        print(f"Subjects : {len(subjects)} (filtered from txt)")
    else:
        subjects = all_subjects
        print(f"Subjects : {len(subjects)} (all dirs)")

    n_done = n_skip = n_fail = 0
    for subject in tqdm(subjects, desc="MSN"):
        lh_path = os.path.join(args.data_path, subject, "stats/lh.aparc.DKTatlas.mapped.stats")
        rh_path = os.path.join(args.data_path, subject, "stats/rh.aparc.DKTatlas.mapped.stats")
        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            n_skip += 1
            continue

        out_file = os.path.join(save_path, f"{subject}.mat")
        if os.path.exists(out_file) and not args.overwrite:
            n_done += 1
            continue

        try:
            regions, feat_z, msn = compute_msn(lh_path, rh_path, features)
            sio.savemat(out_file, {
                "connectivity": msn,
                "value":        feat_z,
                "regions":      np.array(regions, dtype=object),
                "features":     np.array(features, dtype=object),
            })
            n_done += 1
        except Exception as e:
            print(f"[WARN] {subject}: {e}")
            n_fail += 1

    print(f"\nDone.  saved={n_done}  skipped(no stats)={n_skip}  failed={n_fail}")
    print(f"Output: {save_path}")


if __name__ == "__main__":
    main()
