import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建一个全局变量用于存储选定的矩形区域
selected_region = []
drawing = False  # 用于标记是否正在绘制框

# 复制图像以进行绘制
image_copy = None

# 鼠标事件回调函数
def select_region(event, x, y, flags, param):
    global selected_region, drawing, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        # 如果之前已经绘制了框，清除之前的框
        if len(selected_region) == 2:
            selected_region = []
            image_copy = image.copy()

        # 记录起始点
        selected_region = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        # 记录结束点
        selected_region.append((x, y))
        drawing = False

        # 提取所选区域并计算平均灰度值
        if len(selected_region) == 2:
            x1, y1 = selected_region[0]
            x2, y2 = selected_region[1]
            roi = gray[y1:y2, x1:x2]
            mean_gray = np.mean(roi)

            # 绘制红色边界线
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 打印平均灰度值
            print("所选区域的平均灰度值:", mean_gray)

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # 鼠标左键双击事件，在双击位置打印灰度值
        gray_value = gray[y, x]
        print("鼠标点击点的灰度值:", gray_value)

# 读取图像
image = cv2.imread(r"E:\crack_detect\img\liefeng0.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 计算灰度直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# 将直方图展平为一维数组
hist = hist.flatten()

# 绘制灰度分布直方图
plt.figure(figsize=(8, 5))
plt.title('gray-hist')
plt.xlabel('gray')
plt.ylabel('piex')
plt.xlim([0, 256])
plt.bar(range(256), hist, color='gray')
plt.show()

# 批量获得直方图
# 指明被遍历的文件夹
rootdir = r'.\car_cut'
n=0
# for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
#     for filename in filenames:
#         currentPath = os.path.join(parent, filename)
#         # 读取图像
#         image = cv2.imread(currentPath)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         gray[gray<65] = 0
#         gray[gray>=65] = 255
#         cv2.imwrite('gray'+str(n)+'.png',gray)
#         # # 计算灰度直方图
#         # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#         #
#         # # 将直方图展平为一维数组
#         # hist = hist.flatten()
#         #
#         # # 绘制灰度分布直方图
#         # plt.figure(figsize=(8, 5))
#         # plt.title('gray-hist'+ str(n) +'')
#         # plt.xlabel('gray')
#         # plt.ylabel('piex')
#         # plt.xlim([0, 256])
#         # plt.bar(range(256), hist, color='gray')
#         # # 保存直方图到本地文件
#         # plt.savefig('histogram'+str(n)+'.png')
#         n= n+1
image_copy = image.copy()  # 复制图像

# 创建一个窗口并绑定鼠标事件
cv2.namedWindow('Select Region')
cv2.setMouseCallback('Select Region', select_region)

while True:
    cv2.imshow('Select Region', image_copy)
    key = cv2.waitKey(1) & 0xFF

    # 按ESC键退出
    if key == 27:
        break

# 清理资源
cv2.destroyAllWindows()
