import numpy as np
import cv2
img = cv2.imread("D:/ice-detect/yz25.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret0, thresh0 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh0,cv2.MORPH_OPEN,kernel, iterations = 2)
# 确定背景区域
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# 确定前景区域
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret1, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# 查找未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# 标记标签
ret2, markers1 = cv2.connectedComponents(sure_fg)
markers = markers1+1
markers[unknown==255] = 0
markers3 = cv2.watershed(img,markers)
img[markers3 == -1] = [0,255,0]
cv2.imshow('thresh0',thresh0)
cv2.imwrite('D:/ice-detect/yz25-thresh0.jpg', thresh0)
cv2.imshow('sure_bg', sure_bg)
cv2.imwrite('D:/ice-detect/yz25-sure_bg.jpg', sure_bg)
cv2.imshow('sure_fg', sure_fg)
cv2.imwrite('D:/ice-detect/yz25-sure_fg.jpg', sure_fg)
cv2.imshow('result_img', img)
cv2.imwrite('D:/ice-detect/yz25-img.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()