import cv2
import numpy as np
# 二值化
img = cv2.imread('D:/ice-detect/zq25.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = imgGray.shape[:2]
for row in range(rows):
    for col in range(cols):
        if imgGray[row, col] > 40:
            imgGray[row, col] = 255
        else:
            imgGray[row, col] = 0
cv2.imshow('zq-laplacian', imgGray)
cv2.waitKey()
cv2.imwrite('D:/ice-detect/yz25-lap-zq-40.jpg', imgGray)