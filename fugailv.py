import cv2
import numpy as np

gt_img = cv2.imread("D:/project-icedetect/204-yz-3/000159-vb.bmp", 0)
gtimg_array = np.array(gt_img)

sum = 0
shape = gtimg_array.shape

for i in range(0, shape[0]):
    for j in range(0, shape[1]):
        if gtimg_array[i, j] > 0:
            sum = sum + 1
print(sum)
print(shape)
print(sum / (shape[0] * shape[1]))
