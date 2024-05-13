# import numpy as np
# from PIL import Image
# import os
# from glob import glob
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, jaccard_score

# from evaluation import extract_unique_sceneids, get_patches_for_sceneid
# from tiff_utils import reassemble_patches_to_tiff

# def reassemble_patches(patches, patch_size, original_size):
#     stitched_image = Image.new('F', original_size)
#     for patch_path in patches:
#         patch = Image.open(patch_path)
#         if patch.size != patch_size:
#             patch = patch.resize(patch_size)
#         patch_row, patch_col = extract_rowcol_each_patch(os.path.basename(patch_path))
#         stitched_image.paste(patch, ((patch_col-1) * patch_size[0], (patch_row-1) * patch_size[1]))
#     return stitched_image


# # # Example workflow
# # preds_folder_root = './'
# # preds_folder = 'sample'
# # # gt_folder_path = 'path_to_38-Cloud_test_set_ground_truths'
# # thresh = 12 / 255

# # all_uniq_sceneid = extract_unique_sceneids(preds_folder_root, preds_folder)
# # results = []
# # print(all_uniq_sceneid)

# # sceneid = "LC08_L1TP_064015_20160420_20170223_01_T1"
# # # for sceneid in all_uniq_sceneid:
# # gt_path = "edited_corrected_gts_LC08_L1TP_050024_20160520_20170324_01_T1.TIF"
# # gt_image = Image.open(gt_path)
# # gt2_path = "scene2_pred_combined.tiff"
# # gt2_image = Image.open(gt2_path)

# # # patches = get_patches_for_sceneid(preds_folder_root, preds_folder, sceneid)
# # # # [print(patch) for patch in patches]
# # # predicted_image = reassemble_patches(patches, (384, 384), gt_image.size)
# # # print("Binarizing")
# # # predicted_image = imbinarize(predicted_image, thresh)
# # # predicted_image.save('sample_combined.tiff')

# # print("Calculating metrics")
# # precision, recall, f_score, accuracy, jaccard = calculate_metrics(gt2_image, gt_image)
# # print(f"Scene: {sceneid}, Precision: {precision}, Recall: {recall}, F-score: {f_score}, Accuracy: {accuracy}, Jaccard: {jaccard}")
# # # results.append((sceneid, precision, recall, f_score, accuracy, jaccard))
# # # results now contains the evaluation metrics for each scene

# GLOBAL_PATH = '/mnt/netstore1_home/jakub.suran/Cloud-Net_data'
# gt_folder_path = os.path.join(GLOBAL_PATH, "Entire_scene_gts")
# preds_patch_folder = os.path.join(GLOBAL_PATH, "Predictions", 'Cloud-Net_predictions')
# preds_full_scene_folder = os.path.join(GLOBAL_PATH, "Predictions", 'Cloud-Net_full_scene_predictions')
# all_uniq_sceneid = extract_unique_sceneids(preds_patch_folder)
# thresh = 12 / 255

# pred = os.path.join(GLOBAL_PATH, "Predictions", 'Cloud-Net_full_scene_predictions', "pred_LC82320072014226LGN00.TIF")
# img = Image.open(pred)
# print("Image size:", img.size)
# print("Image mode:", img.mode)
# print("Data type:", np.asarray(img).dtype)
# print("Minimum value:", np.min(img))
# print("Maximum value:", np.max(img))
# true_values = np.count_nonzero(np.array(img))
# false_values = np.prod(img.size) - true_values
# print("Number of true values:", true_values)
# print("Number of false values:", false_values)

# scene_id = "LC82320072014226LGN00"
# gt_path = os.path.join(gt_folder_path, f'edited_corrected_gts_{scene_id}.TIF')
# gt = Image.open(gt_path)
# patches = get_patches_for_sceneid(preds_patch_folder, scene_id)
# for patch in sorted(patches):
#     print(patch)

# # Reassemble patches to a single image
# reassemble_patches_to_tiff(patches, 384, 384, gt.size, f'pred_{scene_id}.TIF', thresh)


# img = Image.open(f'pred_{scene_id}.TIF')
# print("Image size:", img.size)
# print("Image mode:", img.mode)

import pandas as pd

# df = pd.read_csv('metrics.csv')
# df = df[['Scene ID','Threshold','Precision','Recall','F-1','Accuracy','Jaccard Index']]
# df['Precision'] = (df['Precision'] * 100).round(4)
# df['Recall'] = (df['Recall'] * 100).round(4)
# df['F-1'] = (df['F-1'] * 100).round(4)
# df['Jaccard Index'] = (df['Jaccard Index'] * 100).round(4)
# df['Accuracy'] = (df['Accuracy'] * 100).round(4)
# df['Threshold'] = df['Threshold'].round(6)

# avg_metrics = df.mean()
# print(avg_metrics)

# avg_metrics['Scene ID'] = 'Average'
# df = df.append(avg_metrics, ignore_index=True)

# df.to_csv('metrics.tsv', index=False, sep='\t')

df = pd.read_csv('metrics.tsv', sep='\t')
min_f1 = df['F-1'].min()
min_accuracy = df['Accuracy'].min()
min_jaccard = df['Jaccard Index'].min()

min_f1_results = df[df['F-1'] == min_f1]
min_accuracy_results = df[df['Accuracy'] == min_accuracy]
min_jaccard_results = df[df['Jaccard Index'] == min_jaccard]

print("Results with minimal F1:")
print(min_f1_results)

print("Results with minimal Accuracy:")
print(min_accuracy_results)

print("Results with minimal Jaccard Index:")
print(min_jaccard_results)

max_f1 = df['F-1'].max()
max_accuracy = df['Accuracy'].max()
max_jaccard = df['Jaccard Index'].max()

max_f1_results = df[df['F-1'] == max_f1]
max_accuracy_results = df[df['Accuracy'] == max_accuracy]
max_jaccard_results = df[df['Jaccard Index'] == max_jaccard]

print("Results with maximal F1:")
print(max_f1_results)

print("Results with maximal Accuracy:")
print(max_accuracy_results)

print("Results with maximal Jaccard Index:")
print(max_jaccard_results)

# best f1: LC08_L1TP_066014_20160520_20170223_01_T1
# best accuracy:  LC80980762014216LGN00 

# worst f1: LC80980762014216LGN00, 
#           LC81660432014020LGN00, 
#           LC81460162014168LGN00, 
#           LC80180082014215LGN00, 
#           LC80420082013220LGN00, 
#           LC80530022014156LGN00, 
#           LC81750432013144LGN00, 
#           LC81030162014107LGN00

# worst accuracy: LC81321192014054LGN00,


