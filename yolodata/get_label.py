import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convert_labels_to_yolo_and_find_bounding_box(gt_image):
    # 对标签图像进行连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_image.astype(np.uint8))
    yolo_labels = []
    bounding_boxes = []

    for label in range(1, num_labels):  # 跳过背景（标签 0）
        # 获取当前连通区域的信息
        x, y, w, h, area = stats[label]
        # 获取当前连通区域的像素值
        pixel_value = np.unique(gt_image[labels == label])[0]
        class_id = pixel_value  # 直接使用像素值作为类别 ID

        # 计算框的中心、宽度和高度的归一化值
        x_center = (x + w / 2) / gt_image.shape[1]
        y_center = (y + h / 2) / gt_image.shape[0]
        w_norm = w / gt_image.shape[1]
        h_norm = h / gt_image.shape[0]

        yolo_labels.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")
        bounding_boxes.append((class_id, x_center, y_center, w_norm, h_norm))

    # 可视化原始标签图像和框
    '''plt.imshow(gt_image)
    for box in bounding_boxes:
        class_id, x_center, y_center, w_norm, h_norm = box
        x = int(x_center * gt_image.shape[1] - w_norm * gt_image.shape[1] / 2)
        y = int(y_center * gt_image.shape[0] - h_norm * gt_image.shape[0] / 2)
        w = int(w_norm * gt_image.shape[1])
        h = int(h_norm * gt_image.shape[0])
        rect = plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
    plt.title('Original Label Image with Bounding Boxes')
    plt.colorbar()
    plt.show()'''

    return yolo_labels

# 示例用法
image_folder = r'E:\QuqXian\gangcaibiaomianquexianfenge\NEU_Seg-main\NEU_Seg-main\annotations\test'
label_folder = r'E:\QuqXian\gangcaibiaomianquexianfenge\NEU_Seg-main\NEU_Seg-main\labels\test'
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        gt_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        yolo_labels = convert_labels_to_yolo_and_find_bounding_box(gt_image)
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_folder, label_filename)
        with open(label_path, 'w') as f:
            for label in yolo_labels:
                print(label)
                f.write(label + '\n')
