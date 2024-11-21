import cv2
import numpy as np
color_folder = r"C:\Users\T.f\Desktop\out4.jpg"
p1 = cv2.imread(color_folder, cv2.IMREAD_GRAYSCALE)
mask = p1 > 150
feimask = p1<=150
p1[mask] = 255
p1[feimask] = 0
cv2.imwrite('04t_out.jpg',p1)
cv2.imshow('p1',p1)
cv2.waitKey(0)