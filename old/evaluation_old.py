import csv
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from evaluation_utils import calculate_metrics, cfmatrix
from glob import glob
from tiff_utils import reassemble_patches_to_tiff

def extract_unique_sceneids(preds_folder):
    file_paths = glob(os.path.join(preds_folder, '*.TIF'))
    scene_ids = set()
    for path in file_paths:
        filename = os.path.basename(path)
        scene_id = '_'.join(filename.split('_')[5:])  # Assuming scene ID is before the first underscore
        scene_id = scene_id.replace('.TIF', '')
        scene_ids.add(scene_id)
    return list(scene_ids)

def extract_rowcol_each_patch(name):
    # Extracting row and column based on the naming convention
    parts = name.replace('.TIF', '').split('_')
    row = int(parts[-2][1:])  # Assuming the row indicator is just before the file extension
    col = int(parts[-1][1:])  # Assuming the column indicator is the last part of the filename
    return row, col

def get_patches_for_sceneid(preds_folder, sceneid):
    file_paths = glob(os.path.join(preds_folder, f'*{sceneid}*.TIF'))
    return file_paths

def unzeropad(pred_mask, gt):
    gt_height, gt_width = gt.size
    pred_mask_cropped = pred_mask.crop((0, 0, gt_width, gt_height))
    return pred_mask_cropped


def stitch_all_patches():
    total_pathes = 0
    # Loop through each scene ID to stitch patches
    for scene_id in tqdm(all_uniq_sceneid, "Combining pathes..."):
        gt_path = os.path.join(gt_folder_path, f'edited_corrected_gts_{scene_id}.TIF')
        gt = Image.open(gt_path)
        patches = get_patches_for_sceneid(preds_patch_folder, scene_id)
        total_pathes += len(patches)
        
        # Reassemble patches to a single image
        reassemble_patches_to_tiff(patches, 384, 384, gt.size, f'{preds_full_scene_folder}/pred_{scene_id}.TIF', thresh)

    print(f"Processed {total_pathes} patches.")
 

def calculate_metrics_for_all_scenes(scene_list):
    # Create a CSV file to store the metrics
    csv_file = open('metrics2.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the header row
    header = ['Scene ID', 'Precision', 'Recall', 'F-1', 'Accuracy', 'Jaccard Index']  
    csv_writer.writerow(header)

    # Loop through each scene ID to calculate and write the metrics
    for scene_id in tqdm(scene_list, "Calculating metrics..."):
        gt_path = os.path.join(gt_folder_path, f'edited_corrected_gts_{scene_id}.TIF')
        pred_path = os.path.join(preds_full_scene_folder, f'pred_{scene_id}.TIF')
        
        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        
        labels = np.unique(np.concatenate((gt, pred)))
        
        # metrics = cfmatrix(gt, pred, labels, printout=False)
        precision, recall, f_score, accuracy, jaccard = calculate_metrics(pred, gt)
        row = [scene_id, precision, recall, f_score, accuracy, jaccard]
        csv_writer.writerow(row)

    # Close the CSV file
    csv_file.close()
    # for scene_id in tqdm(scene_list, "Calculating metrics..."):
    #     gt_path = os.path.join(gt_folder_path, f'edited_corrected_gts_{scene_id}.TIF')
    #     pred_path = os.path.join(preds_full_scene_folder, f'pred_{scene_id}.TIF')
        
    #     gt = np.array(Image.open(gt_path))
    #     pred = np.array(Image.open(pred_path))
        
    #     labels = np.unique(np.concatenate((gt, pred)))
        
        
        # Print or store the metrics as needed
        # print(f"Metrics for scene ID {scene_id}:")
        # print(metrics)
        

# def QE_calcul(predict, gt, labels, conf_print=False):
#     y_true = np.array(gt).flatten()
#     y_pred = np.array(predict).flatten()
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     # Calculate evaluators here, similar to cfmatrix in MATLAB
#     # This part will depend on the implementation of cfmatrix in Python
#     return evaluators

if __name__ == "__main__":
    # Example usage
    GLOBAL_PATH = '/mnt/netstore1_home/jakub.suran/Cloud-Net_data'
    gt_folder_path = os.path.join(GLOBAL_PATH, "Entire_scene_gts")
    preds_patch_folder = os.path.join(GLOBAL_PATH, "Predictions", 'Cloud-Net_predictions')
    preds_full_scene_folder = os.path.join(GLOBAL_PATH, "Predictions", 'Cloud-Net_full_scene_predictions2')
    all_uniq_sceneid = extract_unique_sceneids(preds_patch_folder)
    thresh = 12 / 255

    if not os.path.exists(preds_full_scene_folder):
        os.mkdir(preds_full_scene_folder)


    stitch_all_patches()
    scene_list = ["LC08_L1TP_003052_20160120_20170405_01_T1",
    "LC08_L1TP_018008_20160520_20170324_01_T1",
    "LC08_L1TP_029032_20160720_20170222_01_T1",
    "LC08_L1TP_029041_20160720_20170222_01_T1",
    "LC08_L1TP_029044_20160720_20170222_01_T1",
    "LC08_L1TP_032030_20160420_20170223_01_T1",
    "LC08_L1TP_032035_20160420_20170223_01_T1",
    "LC08_L1TP_032037_20160420_20170223_01_T1",
    "LC08_L1TP_034029_20160520_20170223_01_T1",
    "LC08_L1TP_034033_20160520_20170223_01_T1",
    "LC08_L1TP_034037_20160520_20170223_01_T1",
    "LC08_L1TP_035029_20160120_20170224_01_T1",
    "LC08_L1TP_035035_20160120_20170224_01_T1",
    "LC08_L1TP_039035_20160320_20170224_01_T1",
    "LC08_L1TP_050024_20160520_20170324_01_T1",
    "LC08_L1TP_063012_20160920_20170221_01_T1",
    "LC08_L1TP_063013_20160920_20170221_01_T1",
    "LC08_L1TP_064012_20160420_20170223_01_T1",
    "LC08_L1TP_064015_20160420_20170223_01_T1",
    "LC08_L1TP_066014_20160520_20170223_01_T1",
    ]
    calculate_metrics_for_all_scenes(scene_list)

