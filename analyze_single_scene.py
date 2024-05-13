from glob import glob
import os
from PIL import Image
import numpy as np

from evaluation_utils import calculate_metrics
from tiff_utils import reassemble_patches_to_tiff


# def extract_unique_sceneids(preds_folder_root, preds_folder):
#     files = glob(os.path.join(preds_folder_root, preds_folder, '*.TIF'))
#     unique_sceneids = set(os.path.basename(file).split('_')[4] for file in files)
#     return list(unique_sceneids)

# def get_patches_for_sceneid(preds_folder_root, preds_folder, sceneid):
#     return glob(os.path.join(preds_folder_root, preds_folder, f'*{sceneid}*.TIF'))

def extract_rowcol_each_patch(name):
    parts = name.split('_')
    row = int(parts[2])
    col = int(parts[4])
    return row, col

def imbinarize(image, threshold):
    array = np.array(image)
    binarized_array = array > threshold
    binarized_img = Image.fromarray(binarized_array)
    return binarized_img

def reassemble_patches(patches, patch_size, original_size):
    stitched_image = Image.new('F', original_size)
    for patch_path in patches:
        patch = Image.open(patch_path)
        if patch.size != patch_size:
            patch = patch.resize(patch_size)
        patch_row, patch_col = extract_rowcol_each_patch(os.path.basename(patch_path))
        stitched_image.paste(patch, ((patch_col-1) * patch_size[0], (patch_row-1) * patch_size[1]))
    return stitched_image

# Example workflow
# preds_folder_root = './'
# preds_folder = 'sample'
# # gt_folder_path = 'path_to_38-Cloud_test_set_ground_truths'
# thresh = 12 / 255

# all_uniq_sceneid = extract_unique_sceneids(preds_folder_root, preds_folder)
# results = []
# print(all_uniq_sceneid)

# sceneid = "LC08_L1TP_064015_20160420_20170223_01_T1"
# for sceneid in all_uniq_sceneid:
# gt_path = "edited_corrected_gts_LC08_L1TP_050024_20160520_20170324_01_T1.TIF"
# gt_image = Image.open(gt_path)
# gt2_path = "scene2_pred_combined.tiff"
# gt2_image = Image.open(gt2_path)


# patches = get_patches_for_sceneid(preds_folder_root, preds_folder, sceneid)
# [print(patch) for patch in patches]

SCENE_ID = "LC08_L1TP_050024_20160520_20170324_01_T1"

ROOT = "/mnt/netstore1_home/jakub.suran/Cloud-Net_data"
GTS_PATH = os.path.join(ROOT, "Entire_scene_gts", f"edited_corrected_gts_{SCENE_ID}.TIF")
PRED_PATH = glob(os.path.join(ROOT, "Predictions", "CN_9201_original_patches", f"*{SCENE_ID}.TIF"))
OUT_PATH = os.path.join(ROOT, "Predictions", "FS_CN_9201_original_patches", f"{SCENE_ID}.TIF")
OUT_PATH2 = os.path.join(ROOT, "Predictions", "FS_CN_9201_original_patches", f"second_approach.TIF")
# band_dirs = ["test_blue", "test_green", "test_red", "test_nir"]
thresh = 12 / 255

gt = Image.open(GTS_PATH)

predicted_image = reassemble_patches(PRED_PATH, (384, 384), gt.size)
print("Binarizing")
predicted_image = imbinarize(predicted_image, thresh)
predicted_image.save(OUT_PATH)



reassemble_patches_to_tiff(PRED_PATH, 384, 384, gt.size, OUT_PATH2, thresh=thresh)
out = Image.open(OUT_PATH)

print("Calculating metrics")
precision, recall, f_score, accuracy, jaccard = calculate_metrics(out, gt)
print(f"Scene: {SCENE_ID}, Precision: {precision}, Recall: {recall}, F-score: {f_score}, Accuracy: {accuracy}, Jaccard: {jaccard}")
