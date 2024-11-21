# coding=utf-8
import math
import os
import sys

import cv2
import random
import numpy as np
from PIL import Image
from numpy import pi

file = r"D:\project-icedetect\test.txt"
if (os.path.exists(file)):
    print("file exists")
    f = open(file, mode='r')
    a = f.readline()
    a=a[:-1]
    b = f.readline()
    print(a, b)

f1 = open(file,mode='w')
f1.write('12\n')
f1.write('18')

# 读取输入图片
img0 = cv2.imread("D:/project-icedetect/JB204yz.jpg")
# 将彩色图片转换为灰度图片
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# 创建一个LSD对象
lsd = cv2.createLineSegmentDetector(0)
# 执行检测结果
dlines = lsd.detect(img)
# 声明最终需要的线的数组
need_lines = []
# 绘制检测结果
for dline in dlines[0]:
    x0 = int(round(dline[0][0]))  # round 四舍五入
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))
    if ((abs(y1 - y0) / (1 + abs(x1 - x0))) > 1 and abs(y1 - y0) > 50):  # 增加角度及长度筛选
        need_lines.append([x0, y0, x1, y1])  # 保存筛选后的线段
        # cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)

# 显示并保存结果
# cv2.imwrite('D:/ice-detect/yz25-LSd-xs-50-1.jpg', img0)
# cv2.imshow("LSD", img0)
# cv2.waitKey(0)
# 图像二值化
erzhi = img.copy()
first_horizon = []
second_horizon = []
n = 0
while True and n < 1000:
    n = n + 1  # 避免死循环
    i = random.randint(1, int(erzhi.shape[0] / 2))  # 生成一个n到m之间的随机数
    j = random.randint(int(erzhi.shape[0] / 2), erzhi.shape[0])
    for line in need_lines:
        if (line[1] <= i and line[3] >= i):  # 线段可能是两个像素点大小顺序不一致
            x = int((line[3] - i) * (line[2] - line[0]) / (line[1] - line[3]) + line[2])
            first_horizon.append(x)
        if (line[1] >= i and line[3] <= i):
            x = int((line[3] - i) * (line[2] - line[0]) / (line[1] - line[3]) + line[2])
            first_horizon.append(x)
        if (line[1] <= j and line[3] >= j):
            x = int((line[3] - j) * (line[2] - line[0]) / (line[1] - line[3]) + line[2])
            second_horizon.append(x)
        if (line[1] >= j and line[3] <= j):
            x = int((line[3] - j) * (line[2] - line[0]) / (line[1] - line[3]) + line[2])
            second_horizon.append(x)
    if (len(first_horizon) >= 2 and len(second_horizon) >= 2):
        x0 = min(first_horizon)  # x0,i为左上点 x1,j为左下点
        x2 = max(first_horizon)
        x1 = min(second_horizon)
        x3 = max(second_horizon)
        if (abs(abs(math.atan((i - j) / (x0 - x1 + 0.00001))) - abs(
                math.atan((i - j) / (x2 - x3 + 0.00001)))) < pi / 130):  # 找到两条线准备二值化
            leftup = int(x0 + (x0 - x1) * (1 - i) / (i - j))  # 切割左上像素点（leftup,1)-
            if (leftup < 0):
                leftup = 0
            leftdown = int(x0 + (x0 - x1) * (erzhi.shape[0] - i) / (i - j))
            if (leftdown < 0):
                leftdown = 0
            rightup = int(x2 + (x2 - x3) * (1 - i) / (i - j))
            if (rightup > erzhi.shape[1]):
                rightup = erzhi.shape[1] - 1
            rightdown = int(x2 + (x2 - x3) * (erzhi.shape[0] - i) / (i - j))
            if (rightdown > erzhi.shape[1]):
                rightdown = erzhi.shape[1] - 1
            left_cj = min(leftup, leftdown)
            right_cj = max(rightup, rightdown)
            ##在原图显示确定好的机翼边缘线
            cv2.line(img0, (leftup, 1), (leftdown, erzhi.shape[0]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(img0, (rightup, 1), (rightdown, erzhi.shape[0]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("lihou", img0)
            cv2.waitKey(0)
            # cv2.imwrite('D:/project-icedetect/yz25-cut2.jpg', img0)
            box = (left_cj, 1, right_cj, erzhi.shape[0])
            # imgx = Image.open("D:/ice-detect/yz25.jpg")
            # a_crop = imgx.crop(box)
            a_crop = img0[1:erzhi.shape[0], left_cj:right_cj]  # 确定图片裁剪坐标
            cv2.line(a_crop, (leftup - left_cj, 1), (leftdown - left_cj, erzhi.shape[0]), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(a_crop, (rightup - left_cj, 1), (rightdown - left_cj, erzhi.shape[0]), (0, 255, 0), 1, cv2.LINE_AA)
            # a_crop.show()
            # cv2.imwrite('D:/ice-detect/yz25-cut2.jpg', a_crop)  #输出包含机翼的最小矩形
            # cv2.imshow("fwhou", a_crop)

            break

# 选取两条线内区域 用y+kx=b 左边线的右边部分y+kx>b 右边线的左侧y+kx<b 不行 竖直的话k接近于无穷大 用ky+x=b
# 左边直线
left_k = (leftup - leftdown) / (erzhi.shape[0] - 1)
left_b = left_k + leftup
# 右边直线
right_k = (rightup - rightdown) / (erzhi.shape[0] - 1)
right_b = right_k + rightup
# 二值化
rows, cols = erzhi.shape[:2]
for row in range(rows):  # 行 即 y
    for col in range(cols):
        if (row * left_k + col) >= left_b and (row * right_k + col) <= right_b:
            erzhi[row, col] = 255
        else:
            erzhi[row, col] = 0
cv2.imshow("erzhi", erzhi)
cv2.imshow("lihou", img0)
cv2.imwrite('D:/project-icedetect/28200-302zq-cut-out-line.jpg', img0)  # 原图二值化机翼部分
cv2.imwrite('D:/project-icedetect/28200-302zq-cut-out.jpg', erzhi)  # 原图二值化机翼部分
cv2.waitKey(0)
cv2.destroyAllWindows()
