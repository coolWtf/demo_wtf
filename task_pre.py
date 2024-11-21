import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
def read_and_resize_images(folder_path, target_size):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, target_size)
            images.append(resized_img)
    return images
def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou
def calculate_dice_coefficient(pred_mask, true_mask):
    intersection = np.sum(pred_mask * true_mask)
    dice_coefficient = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(true_mask))
    return dice_coefficient


def calculate_precision_recall_f1(pred_mask, true_mask):
    tp = np.sum(np.logical_and(pred_mask, true_mask))
    fp = np.sum(np.logical_and(pred_mask, np.logical_not(true_mask)))
    fn = np.sum(np.logical_and(np.logical_not(pred_mask), true_mask))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def calculate_metrics(pred_folder, true_folder):
    target_size = (256, 256)  # 设置目标大小，例如 (256, 256)

    pred_images = read_and_resize_images(pred_folder, target_size)
    true_images = read_and_resize_images(true_folder, target_size)

    iou_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    auroc_list = []

    for pred, true in zip(pred_images, true_images):
        pred_mask = (pred > 0).astype(np.uint8)  # Assuming binary mask
        true_mask = (true > 0).astype(np.uint8)
        pred_mask[pred_mask>0] = 1
        true_mask[true_mask>0] = 1
        iou = calculate_iou(pred_mask, true_mask)
        dice_coefficient = calculate_dice_coefficient(pred_mask, true_mask)
        precision, recall, f1_score = calculate_precision_recall_f1(pred_mask, true_mask)
        accuracy = accuracy_score(pred_mask, true_mask)
        auroc = roc_auc_score(pred_mask, true_mask)
        print("iou",iou)
        print("pre",precision)
        print("auroc",auroc)
        iou_list.append(iou)
        dice_list.append(dice_coefficient)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        acc_list.append(accuracy)
        auroc_list.append(auroc)

    avg_iou = np.mean(iou_list)
    avg_dice = np.mean(dice_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_acc = np.mean(acc_list)
    #avg_auroc = np.mean(auroc_list)

    print(f"Avg IoU: {avg_iou:.4f}")
    print(f"Avg Dice Coefficient: {avg_dice:.4f}")
    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Avg Recall: {avg_recall:.4f}")
    print(f"Avg F1 Score: {avg_f1:.4f}")
    print(f"Avg avg_acc: {avg_acc:.4f}")
    #print(f"Avg avg_auroc: {avg_auroc:.4f}")

# Example usage
calculate_metrics("E:/CAROTID/our_res/leather_fold", "E:/CAROTID/unet/Datasets/LEA/predict/label")
