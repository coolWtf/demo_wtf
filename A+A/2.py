import cv2
import numpy as np

image1 = cv2.imread(r"E:\Defect_detection\GANdehaze\dataset\ziran_mask_to_mask\model_out\003.png")
image2 = cv2.imread(r"E:\Defect_detection\GANdehaze\dataset\squee-qxjc-nu\GT\003.png")
combined_image = np.zeros_like(image1)
for i in range(3):  # 遍历三个通道
    combined_image[:, :, i] = image1[:, :, i]*0.4 + image2[:, :, i]*0.6
cv2.imwrite("00ronghe.png",combined_image)

