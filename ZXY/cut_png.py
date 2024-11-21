'''
作者  WTF
功能  裁剪图像，按照给定像素值
输入  png图像以及裁剪像素点

'''
import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

def read_and_combine_first_images(folder_paths):
    combined_image = None
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        first_image_path = os.path.join(folder_path, file_names[0])
        image = cv2.imread(first_image_path)
        if combined_image is None:
            combined_image = image
        else:
            combined_image = (combined_image + image) // 2
    return combined_image

def crop_images_in_folders(folder_paths, top_left, bottom_right):
    for folder_path in folder_paths:
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if file_name.endswith('.png'):  # 新增判定后缀名为.png
                image_path = os.path.join(folder_path, file_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"无法读取图像: {image_path}")
                    continue
                cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                new_image_path = os.path.splitext(image_path)[0] + '_cut' + os.path.splitext(image_path)[1]
                cv2.imwrite(new_image_path, cropped_image)
folder_paths = [r"E:\ZXY\visualization\models\intra\modify815"]#, r"E:\ZXY\visualization\models\ad_new\pdfs\189", r"E:\ZXY\visualization\models\ad_new\pdfs\c020", r"E:\ZXY\visualization\models\ad_new\pdfs\c028",r"E:\ZXY\visualization\models\ad_new\pdfs\c0085",r"E:\ZXY\visualization\models\ad_new\pdfs\M146"]  # 替换为您的四个文件夹路径
top_left = (320, 130)  # 替换为左上角坐标
bottom_right = (1300, 960)  # 替换为右下角坐标

crop_images_in_folders(folder_paths, top_left, bottom_right)
'''combined_image = read_and_combine_first_images(folder_paths)

roi = [0, 0, 0, 0]
selecting = False'''

def on_mouse(event, x, y, flags, param):
    global roi, selecting
    if event == cv2.EVENT_LBUTTONDOWN:
        roi[0] = x
        roi[1] = y
        selecting = True
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        roi[2] = x - roi[0]
        roi[3] = y - roi[1]
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        print(f"左上角坐标: ({roi[0]}, {roi[1]})")
        print(f"右下角坐标: ({roi[0] + roi[2]}, {roi[1] + roi[3]})")

'''cv2.namedWindow('Combined Image')
cv2.setMouseCallback('Combined Image', on_mouse)

while True:
    display_image = combined_image.copy()
    if selecting:
        cv2.rectangle(display_image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
    cv2.imshow('Combined Image', display_image)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 键退出
        break

cv2.destroyAllWindows()'''


#folder_paths = [r"E:\ZXY\visualization\models\ad_new\pdfs\187", r"E:\ZXY\visualization\models\ad_new\pdfs\189", r"E:\ZXY\visualization\models\ad_new\pdfs\c020", r"E:\ZXY\visualization\models\ad_new\pdfs\c0085"]  # 替换为您的四个文件夹路径
