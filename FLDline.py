# coding=utf-8
import cv2
import numpy as np


# 读取输入图片
img0 = cv2.imread("D:/ice-detect/yz25.jpg")
# 将彩色图片转换为灰度图片
img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)

# 创建一个LSD对象
fld = cv2.ximgproc.createFastLineDetector()
# 执行检测结果
dlines = fld.detect(img)
# 绘制检测结果
# drawn_img = fld.drawSegments(img0,dlines, )
for dline in dlines:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    cv2.line(img0, (x0, y0), (x1,y1), (0,255,0), 1, cv2.LINE_AA)

# 显示并保存结果
cv2.imwrite('test3_r.jpg', img0)
cv2.imshow("LSD", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
