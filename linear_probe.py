"""Linear probing evaluation for DAMT pretrained encoder.

Extracts CLS token features from a pretrained checkpoint (frozen encoder),
then trains a lightweight linear head on labeled downstream data.

Supported tasks
---------------
age_regression      : predict z-scored chronological age from participants.tsv
                      Metrics: MAE (years), R²
sex_classification  : predict sex (0=F, 1=M) from participants.tsv
                      Metrics: Accuracy, balanced Accuracy, AUC

Evaluation protocol
-------------------
5-fold cross-validation on subjects that have valid labels AND valid MRI+atlas files.
Features are extracted once from the frozen encoder, then CV runs on the cached features.

Usage
-----
# Age regression on GPU 2
python linear_probe.py \\
    --checkpoint runs_test_msn/checkpoint.pth \\
    --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz \\
    --data fomo60k_wo_scz_0319 \\
    --task age_regression \\
    --gpu 2 \\
    --epochs 200 --lr 1e-3 --batch-size 256

# Sex classification on GPU 3
python linear_probe.py \\
    --checkpoint runs_test_msn/checkpoint.pth \\
    --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz \\
    --data fomo60k_wo_scz_0319 \\
    --task sex_classification \\
    --gpu 3 \\
    --epochs 200 --lr 1e-3

Notes
-----
- Single GPU only (no DDP). Use --gpu to select the device index.
- The encoder is always frozen; only the linear head trains.
- MRI files are loaded with the same preprocessing as pretraining (1.25mm isotropic,
  128³ crop, intensity normalisation).
"""
import os
import re
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, balanced_accuracy_score, roc_auc_score
import pandas as pd
from monai import transforms
import warnings
warnings.filterwarnings("ignore")


# ── Arg parser ────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Linear probing for DAMT pretrained encoder")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to pretrained checkpoint (.pth)")
    p.add_argument("--data-path",   type=str,
                   default="/NFS/Users/kimyw/data/fomo60k_wo_scz")
    p.add_argument("--data",        type=str, default="fomo60k_wo_scz",
                   help="Subject list txt filename (without .txt)")
    p.add_argument("--task",        type=str, default="age_regression",
                   choices=["age_regression", "sex_classification"])
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--folds",       type=int, default=5)
    p.add_argument("--gpu",         type=int, default=0,
                   help="GPU index to use (default: 0). Ignored if CUDA is unavailable.")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--output-dir",  type=str, default="./linear_probe_results")
    # MSN / asym dims for model init (must match pretraining)
    p.add_argument("--msn-n-regions", type=int, default=62)
    p.add_argument("--asym-dim",    type=int, default=0)
    return p.parse_args()


# ── TSV helpers (shared with datasets.py logic) ───────────────────────────────

def parse_subject_dir(dirname):
    name = re.sub(r"_T1w$", "", dirname)
    name = re.sub(r"_run-\d+", "", name)
    m = re.match(r"^(.+?)_(sub-[^_]+)_(ses-[^_]+)$", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def load_participants(data_path):
    tsv_path = os.path.join(data_path, "participants.tsv")
    if not os.path.isfile(tsv_path):
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
    return raw, age_mean, age_std


# ── MRI preprocessing (same as DataAugmentation.load_image in main.py) ───────

def build_preprocess():
    from main import remap_atlas_labels
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        transforms.Lambdad(keys=["image"], func=lambda x: x[0:1]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.CropForegroundd(keys=["image"], source_key="image"),
        transforms.Spacingd(keys=["image"], pixdim=(1.25, 1.25, 1.25), mode="nearest"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(128, 128, 128)),
        transforms.ScaleIntensityRangePercentilesd(
            keys="image", lower=0.05, upper=99.95, b_min=0, b_max=1),
        transforms.RandSpatialCropd(
            keys=["image"], roi_size=(128, 128, 128), random_size=False),
    ])


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path, args):
    """Load pretrained SSLHead_Swin, freeze all parameters, return model."""
    from models import SSLHead_Swin

    # Build a minimal args namespace for model init
    model_args = argparse.Namespace(
        in_channels=args.in_channels,
        device=args.device,
        msn_n_regions=args.msn_n_regions,
        asym_dim=args.asym_dim,
    )

    model = SSLHead_Swin(model_args).to(args.device)

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    # Support both raw state_dict and wrapped checkpoints
    state_dict = ckpt.get("model", ckpt)

    # Strip DDP "module." prefix if present
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1)] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {len(missing)}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, subjects_with_labels, data_path, device, preprocess):
    """Extract CLS token for each subject.

    Parameters
    ----------
    subjects_with_labels : list of (subject_dir, label)
    Returns (features_np, labels_np) arrays.
    """
    feats, labels = [], []
    for subject, label in subjects_with_labels:
        img_path = os.path.join(data_path, subject, "mri/brainmask.nii.gz")
        if not os.path.isfile(img_path):
            continue
        try:
            data = preprocess({"image": img_path})
            x = data["image"].unsqueeze(0).float().to(device)  # (1, 1, 128, 128, 128)
            _, cls_token = model.encode(x)                      # (1, 768)
            feats.append(cls_token.squeeze(0).cpu().numpy())
            labels.append(label)
        except Exception as e:
            print(f"[WARN] {subject}: {e}")
            continue

    if not feats:
        raise RuntimeError("No features extracted — check data path and checkpoint.")

    return np.stack(feats, axis=0), np.array(labels)


# ── Linear head training ──────────────────────────────────────────────────────

def train_linear_head(X_train, y_train, X_val, y_val, args, task):
    """Train a linear head on extracted features (in-memory, no MRI loading)."""
    device = args.device
    feat_dim = X_train.shape[1]

    if task == "age_regression":
        head = nn.Linear(feat_dim, 1).to(device)
        criterion = nn.MSELoss()
    else:
        head = nn.Linear(feat_dim, 2).to(device)
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    X_vl = torch.tensor(X_val,   dtype=torch.float32)

    if task == "age_regression":
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_vl = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)
    else:
        y_tr = torch.tensor(y_train, dtype=torch.long)
        y_vl = torch.tensor(y_val,   dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        preds = head(X_vl.to(device)).cpu()

    if task == "age_regression":
        preds_np = preds.squeeze(1).numpy()
        return preds_np, y_val
    else:
        probs    = torch.softmax(preds, dim=1)[:, 1].numpy()
        pred_cls = preds.argmax(dim=1).numpy()
        return pred_cls, probs, y_val


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_age(all_preds, all_labels, age_std, age_mean):
    preds_np  = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)
    # Back-convert from z-score to years for human-readable MAE
    preds_yr  = preds_np  * age_std + age_mean
    labels_yr = labels_np * age_std + age_mean
    mae = float(np.mean(np.abs(preds_yr - labels_yr)))
    r2  = float(r2_score(labels_yr, preds_yr))
    return {"MAE_years": mae, "R2": r2}


def evaluate_sex(all_pred_cls, all_probs, all_labels):
    pred_cls  = np.concatenate(all_pred_cls)
    probs     = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    acc  = float((pred_cls == labels_np).mean())
    bacc = float(balanced_accuracy_score(labels_np, pred_cls))
    try:
        auc = float(roc_auc_score(labels_np, probs))
    except ValueError:
        auc = float("nan")
    return {"Accuracy": acc, "BalancedAcc": bacc, "AUC": auc}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve device from --gpu flag
    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu}"
        torch.cuda.set_device(args.gpu)
    else:
        args.device = "cpu"
        print("[INFO] CUDA not available, running on CPU.")
    print(f"Using device: {args.device}")

    # ── Load participants.tsv ──────────────────────────────────────────────
    raw_lookup, age_mean, age_std = load_participants(args.data_path)
    print(f"TSV entries: {len(raw_lookup)},  age_mean={age_mean:.1f}, age_std={age_std:.1f}")

    # ── Enumerate valid subject directories ───────────────────────────────
    txt_path = os.path.join(args.data_path, f"{args.data}.txt") if args.data else None
    if txt_path and os.path.isfile(txt_path):
        with open(txt_path) as f:
            all_dirs = [l.strip().split(",")[0].strip().split()[0]
                        for l in f if l.strip()]
    else:
        all_dirs = [
            d for d in os.listdir(args.data_path)
            if os.path.isdir(os.path.join(args.data_path, d)) and d != "results"
        ]

    # ── Build (subject, label) pairs ──────────────────────────────────────
    subjects_with_labels = []
    for subject in all_dirs:
        parsed = parse_subject_dir(subject)
        if parsed not in raw_lookup:
            continue
        age_raw, sex = raw_lookup[parsed]

        if args.task == "age_regression":
            if np.isnan(age_raw):
                continue
            z_age = (age_raw - age_mean) / age_std
            subjects_with_labels.append((subject, z_age))
        else:  # sex_classification
            if sex < 0:
                continue
            subjects_with_labels.append((subject, sex))

    print(f"Subjects with valid labels: {len(subjects_with_labels)}")
    if len(subjects_with_labels) < args.folds * 2:
        raise RuntimeError(f"Too few labeled subjects ({len(subjects_with_labels)}) for {args.folds}-fold CV")

    # ── Load encoder ──────────────────────────────────────────────────────
    print(f"Loading encoder from: {args.checkpoint}")
    model = load_encoder(args.checkpoint, args)
    preprocess = build_preprocess()
    print("Extracting CLS token features ...")
    feats, labels = extract_features(model, subjects_with_labels, args.data_path,
                                     args.device, preprocess)
    print(f"Features extracted: {feats.shape}  Labels: {labels.shape}")

    # ── Cross-validation ──────────────────────────────────────────────────
    if args.task == "sex_classification":
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        splits = list(splitter.split(feats, labels))
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        splits = list(splitter.split(feats))

    all_preds, all_labels = [], []
    all_probs = []  # for AUC (sex only)

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"  Fold {fold+1}/{args.folds} ...")
        X_tr, X_vl = feats[train_idx], feats[val_idx]
        y_tr, y_vl = labels[train_idx], labels[val_idx]

        if args.task == "age_regression":
            preds, lbls = train_linear_head(X_tr, y_tr, X_vl, y_vl, args, args.task)
            all_preds.append(preds)
            all_labels.append(lbls)
        else:
            pred_cls, probs, lbls = train_linear_head(X_tr, y_tr, X_vl, y_vl, args, args.task)
            all_preds.append(pred_cls)
            all_probs.append(probs)
            all_labels.append(lbls)

    # ── Report ────────────────────────────────────────────────────────────
    if args.task == "age_regression":
        metrics = evaluate_age(all_preds, all_labels, age_std, age_mean)
    else:
        metrics = evaluate_sex(all_preds, all_probs, all_labels)

    print("\n── Linear Probe Results ──────────────────────────────")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Task       : {args.task}")
    print(f"  Subjects   : {len(labels)}")
    print(f"  Folds      : {args.folds}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("─────────────────────────────────────────────────────\n")

    # Save results
    result = {
        "checkpoint": args.checkpoint,
        "task": args.task,
        "n_subjects": int(len(labels)),
        "folds": args.folds,
        "metrics": metrics,
    }
    out_name = os.path.basename(args.checkpoint).replace(".pth", "")
    out_path = os.path.join(args.output_dir, f"{out_name}_{args.task}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
