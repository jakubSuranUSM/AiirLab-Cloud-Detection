import csv
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from evaluation_utils import calculate_metrics
from glob import glob
from tiff_utils import reassemble_patches_to_tiff


def extract_unique_sceneids(preds_folder):
    file_paths = glob(os.path.join(preds_folder, '*.TIF'))
    scene_ids = set()
    for path in file_paths:
        filename = os.path.basename(path)
        scene_id = '_'.join(filename.split('_')[5:])
        scene_id = scene_id.replace('.TIF', '')
        scene_ids.add(scene_id)
    return list(scene_ids)


def get_patches_for_sceneid(preds_folder, sceneid):
    file_paths = glob(os.path.join(preds_folder, f'*{sceneid}*.TIF'))
    return file_paths


def stitch_all_patches(scene_ids, gt_dir, pred_dir, target_dir, thresh):
    total_pathes = 0
    # Loop through each scene ID to stitch patches
    for scene_id in tqdm(scene_ids, "Combining pathes..."):
        gt_path = os.path.join(gt_dir, f'edited_corrected_gts_{scene_id}.TIF')
        gt = Image.open(gt_path)
        patches = get_patches_for_sceneid(pred_dir, scene_id)
        total_pathes += len(patches)
        
        # Reassemble patches to a single image
        reassemble_patches_to_tiff(patches, 384, 384, gt.size, f'{target_dir}/pred_{scene_id}.TIF', thresh)

    print(f"Processed {total_pathes} patches.")
 

def calculate_metrics_for_all_scenes(scene_list, target_path, gt_dir, pred_dir, thresh):
    # Create a CSV file to store the metrics
    csv_file = open(target_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the header row
    header = ['Scene ID', 'Threshold', 'Precision', 'Recall', 'F-1', 'Accuracy', 'Jaccard Index']  
    csv_writer.writerow(header)

    # Loop through each scene ID to calculate and write the metrics
    for scene_id in tqdm(scene_list, "Calculating metrics..."):
        gt_path = os.path.join(gt_dir, f'edited_corrected_gts_{scene_id}.TIF')
        pred_path = os.path.join(pred_dir, f'pred_{scene_id}.TIF')
        
        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        
        precision, recall, f_score, accuracy, jaccard = calculate_metrics(pred, gt)
        
        row = [scene_id, thresh, precision, recall, f_score, accuracy, jaccard]
        csv_writer.writerow(row)

    # Close the CSV file
    csv_file.close()
