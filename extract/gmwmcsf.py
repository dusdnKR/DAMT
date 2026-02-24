import SimpleITK as sitk
import numpy as np
import os

def create_gmwmcsf_mask(input_path, output_path):
    seg_img = sitk.ReadImage(input_path)
    seg_array = sitk.GetArrayFromImage(seg_img)
    
    mask_array = np.zeros_like(seg_array)

    # FreeSurfer LUT based label
    gm_labels = [3, 42, 10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
    wm_labels = [2, 41, 7, 16, 28, 46, 60, 77, 251, 252, 253, 254, 255]
    csf_labels = [4, 43, 5, 44, 14, 15, 24]

    # GM=1, WM=2, CSF=3
    for label in gm_labels:
        mask_array[seg_array == label] = 1
    for label in wm_labels:
        mask_array[seg_array == label] = 2
    for label in csf_labels:
        mask_array[seg_array == label] = 3

    new_mask_img = sitk.GetImageFromArray(mask_array)
    new_mask_img.CopyInformation(seg_img)

    sitk.WriteImage(new_mask_img, output_path)
    print(f"Processed: {output_path}")

# create_gmwmcsf_mask("aparc+aseg.nii.gz", "gmwmcsf.nii.gz")

from pathlib import Path

SOURCE_ROOT = Path(r"Z:/Users/kimyw/data/sample")

def main():
    if not SOURCE_ROOT.exists():
        print(f"Directory Not Found: {SOURCE_ROOT}")
        return

    subject_folders = [f for f in SOURCE_ROOT.iterdir() if f.is_dir()]
    total = len(subject_folders)
    print(f"A total of {total} subject folders were found.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for idx, subj_dir in enumerate(subject_folders, 1):
        input_nii = subj_dir / "mri" / "aparc+aseg.mgz"
        output_nii = subj_dir / "mri" / "gmwmcsf.nii.gz"

        if output_nii.exists():
            skipped_count += 1
            continue

        if not input_nii.exists():
            print(f"[{idx}/{total}] File Not Found: {subj_dir.name}")
            error_count += 1
            continue

        print(f"[{idx}/{total}] Processing: {subj_dir.name}...", end="\r")
        if create_gmwmcsf_mask(input_nii, output_nii):
            processed_count += 1
        else:
            error_count += 1

    print(f"\n" + "="*50)
    print(f"Completion Report")
    print(f" - Newly created: {processed_count}")
    print(f" - Skipped (already exists): {skipped_count}")
    print(f" - Failed (file not found/error): {error_count}")
    print(f" - Total: {total}")
    print("="*50)

if __name__ == "__main__":

    # say hello
    print("")
    print("-----------------------------")
    print("gmwmcsf")
    print("-----------------------------")
    print("")

    main()