import sklearn as sk
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix

gt_img = cv2.imread("D:/project-icedetect/250cut_masks.png", 0)
res_img = cv2.imread("D:/project-icedetect/250cut-out.jpg", 0)
# img.show()
gtimg_array = np.array(gt_img)  # 把图像转成数组格式img = np.asarray(image)
resimg_array = np.array(res_img)  # 把图像转成数组格式img = np.asarray(image)

# shape = gtimg_array.shape
# #print(img_array.shape)
# for i in range(0,shape[0]):
#     for j in range(0,shape[1]):
#         #if resimg_array[i, j] > 0:
#          #   resimg_array[i, j] = 1
#         if gtimg_array[i, j] > 0:
#             gtimg_array[i, j] = 255
# cv2.imwrite('E:/wtf/yz25-ez2.jpg', gtimg_array)
y_true = np.array(gtimg_array).flatten()
y_pred = np.array(resimg_array).flatten()
print("f1_score", sk.metrics.f1_score(y_true, y_pred, average='micro'))
