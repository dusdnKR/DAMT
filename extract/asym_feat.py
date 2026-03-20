"""Generate hemisphere asymmetry index (AI) vectors from FreeSurfer stats.

For each subject, per-region morphological feature values are read from
lh/rh.aparc.DKTatlas.mapped.stats.  For each region present in both
hemispheres, the Asymmetry Index is computed:

    AI = (lh_val - rh_val) / (0.5 * (|lh_val| + |rh_val|) + eps)

The result is a flat vector of shape (n_bilateral_regions × n_features,)
saved into a single CSV file for the whole dataset.

Output: {data_path}/results/asym_{abbrs}/asym.csv
        Index  : subject directory name (e.g. PT021_IXI_sub-001_ses-01_T1w)
        Columns: {region}_{feature}  for each (region, feature) pair

Available features and abbreviations:
  SurfArea→sa  GrayVol→gv  ThickAvg→ta  ThickStd→ts
  MeanCurv→mc  GausCurv→gc  FoldInd→fi  CurvInd→ci

Usage:
    python extract/asym_feat.py \\
        --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz \\
        --data fomo60k_wo_scz \\
        --features SurfArea GrayVol ThickAvg MeanCurv GausCurv
"""
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


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


def parse_stats_file(path):
    """Parse a FreeSurfer aparc stats file.

    Returns {region_name: {feature_name: float}}
    """
    regions = {}
    col_names = None
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("# ColHeaders"):
                    col_names = line.split()[2:]
                    continue
                if col_names is None or line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                region = parts[0]
                regions[region] = {}
                for i, col in enumerate(col_names[1:], start=1):
                    if i < len(parts):
                        try:
                            regions[region][col] = float(parts[i])
                        except ValueError:
                            pass
    except Exception:
        pass
    return regions


def compute_asym(lh_path, rh_path, features):
    """Compute hemisphere asymmetry index vector for one subject.

    Returns
    -------
    vec     : ndarray shape (n_bilateral × n_features,) of float32
    columns : list[str] of column names '{region}_{feature}'
    """
    lh_data = parse_stats_file(lh_path)
    rh_data = parse_stats_file(rh_path)

    bilateral_regions = sorted(set(lh_data.keys()) & set(rh_data.keys()))
    if not bilateral_regions:
        return None, []

    rows = []
    col_names = []
    for region in bilateral_regions:
        for feat in features:
            lh_val = lh_data[region].get(feat, 0.0)
            rh_val = rh_data[region].get(feat, 0.0)
            denom = 0.5 * (abs(lh_val) + abs(rh_val)) + 1e-6
            ai = (lh_val - rh_val) / denom
            rows.append(ai)
            col_names.append(f"{region}_{feat}")

    return np.array(rows, dtype=np.float32), col_names


def load_subject_list(data_path, data_name):
    if not data_name:
        return None
    txt_path = os.path.join(data_path, f"{data_name}.txt")
    if not os.path.isfile(txt_path):
        print(f"[WARN] txt not found: {txt_path} — processing all subjects")
        return None
    subjects = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split(",", 1)[0].strip().split()[0]
            if s:
                subjects.add(s)
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Generate hemisphere asymmetry index CSV from FreeSurfer stats"
    )
    parser.add_argument("--data-path", type=str,
                        default="/NFS/Users/kimyw/data/fomo60k_wo_scz")
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                        choices=list(FEAT_INFO.keys()), metavar="FEAT")
    parser.add_argument("--data", type=str, default=None,
                        help="Subject list txt file name without extension")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    abbrs = "_".join(FEAT_INFO[f] for f in args.features)
    out_dir = os.path.join(args.data_path, "results", f"asym_{abbrs}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "asym.csv")

    if os.path.isfile(out_csv) and not args.overwrite:
        print(f"Already exists: {out_csv}  (use --overwrite to regenerate)")
        return

    print(f"Features : {args.features}")
    print(f"Output   : {out_csv}")

    selected = load_subject_list(args.data_path, args.data)
    all_dirs = sorted(
        d for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d)) and d != "results"
    )
    subjects = [s for s in all_dirs if selected is None or s in selected]
    print(f"Subjects : {len(subjects)}")

    records = {}
    col_names_ref = None
    n_skip = n_fail = 0

    for subject in tqdm(subjects, desc="asym"):
        stats_dir = os.path.join(args.data_path, subject, "stats")
        lh_path = os.path.join(stats_dir, "lh.aparc.DKTatlas.mapped.stats")
        rh_path = os.path.join(stats_dir, "rh.aparc.DKTatlas.mapped.stats")
        if not os.path.isfile(lh_path) or not os.path.isfile(rh_path):
            n_skip += 1
            continue

        vec, col_names = compute_asym(lh_path, rh_path, args.features)
        if vec is None:
            n_fail += 1
            continue

        if col_names_ref is None:
            col_names_ref = col_names
        records[subject] = vec

    print(f"Success: {len(records)},  no-stats: {n_skip},  failed: {n_fail}")

    if records:
        df = pd.DataFrame.from_dict(records, orient="index", columns=col_names_ref)
        df.index.name = "subject"
        df.to_csv(out_csv)
        print(f"Saved {len(df)} rows × {len(df.columns)} cols → {out_csv}")
        print(f"asym_dim = {len(df.columns)}  ({len(col_names_ref) // len(args.features)} bilateral regions × {len(args.features)} features)")


if __name__ == "__main__":
    main()
