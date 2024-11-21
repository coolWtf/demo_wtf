import os
import cv2
import numpy as np

import cv2
import numpy as np


# 截取最小矩阵包含mask区域
def mask_region(img_gray, img_normal, img_defective):
    # 计算掩码区域的最小矩形
    while 1:
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img_gray, contours, -1, (0, 0, 255), -1)
        # cv2.imshow("img", img_gray)
        # cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(contours[0])
        if w < 10 and h < 10:
            for i in range(h):
                img_gray[y+i][x:x+w] = 0
            continue
        else:
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        break


    # 创建掩码，只保留最小矩形内的区域
    mask = np.zeros(img_gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, (255), -1)

    # 获取掩码之内的区域
    roi = img_normal[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]

    # 将掩码之外的区域全部置为0  这里只是将掩码所在矩阵之外区域置为0
    img_normal = cv2.bitwise_and(roi, roi, mask=roi_mask)
    roi = img_defective[y:y + h, x:x + w]
    img_defective = cv2.bitwise_and(roi, roi, mask=roi_mask)

    return img_normal, img_defective


def mask_and_cut(img_gray, img_normal, img_defective):
    # 创建一个与p2尺寸相同的空图像，用于存储修改后的图像
    img_normal_out = np.zeros_like(img_normal)
    img_defective_out = np.zeros_like(img_defective)

    # 对于p1中的每个像素，检查其是否为白色（像素值为255）
    # 如果是白色，将对应的p2区域复制到结果图像中
    mask = img_gray == 255
    img_normal_out[mask] = img_normal[mask]
    img_defective_out[mask] = img_defective[mask]

    # 计算掩码区域的最小矩形
    while 1:
        contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img_gray, contours, -1, (0, 0, 255), -1)
        # cv2.imshow("img", img_gray)
        # cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(contours[0])
        if w < 10 and h < 10:
            for i in range(h):
                img_gray[y + i][x:x + w] = 0
            continue
        else:
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        break

    img_normal_out = img_normal_out[y:y + h, x:x + w]
    img_defective_out = img_defective_out[y:y + h, x:x + w]
    return img_normal_out, img_defective_out


# 将mask区域保留 其他区域置0
def apply_mask(p1, p2):
    if p1.shape != p2.shape[:2]:
        raise ValueError("两张图像尺寸不匹配，请输入相同尺寸的图像")

    # 创建一个与p2尺寸相同的空图像，用于存储修改后的图像
    result = np.zeros_like(p2)

    # 对于p1中的每个像素，检查其是否为白色（像素值为255）
    # 如果是白色，将对应的p2区域复制到结果图像中
    mask = p1 == 255
    result[mask] = p2[mask]

    return result

def guiwei(p1,p2,p3):
    while 1:
        contours, _ = cv2.findContours(p1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img_gray, contours, -1, (0, 0, 255), -1)
        # cv2.imshow("img", img_gray)
        # cv2.waitKey(0)
        x, y, w, h = cv2.boundingRect(contours[0])
        if w < 10 and h < 10:
            for i in range(h):
                p1[y+i][x:x+w] = 0
            continue
        break
    mask = p1 == 255
    p3[mask] = 0
    for i in range(h):
        for j in range(w):
            p3[y + i][x + j] = p2[i][j]+p3[y + i][x + j]
    return p3

# 指定图像文件夹路径
gray_folder = r"E:\Defect_detection\GANdehaze\dataset\ziran_mask_to_mask\GT_mask"
color_folder = r"E:\Defect_detection\GANdehaze\dataset\ziran_mask_to_mask\out"
normal_folder = r"E:\Defect_detection\GANdehaze\dataset\ziran_mask_to_mask\yuantu"

# 输出文件夹路径
normal_output_folder = r"E:\Defect_detection\GANdehaze\dataset\ziran_mask_to_mask\out_1"
defective_output_folder = r"E:\Defect_detection\GANdehaze\dataset\squee-mask-to-mask\GT"
os.makedirs(normal_output_folder, exist_ok=True)
os.makedirs(defective_output_folder, exist_ok=True)

# 读取两个文件夹中的图像文件名
gray_files = sorted(os.listdir(gray_folder))
color_files = sorted(os.listdir(color_folder))
normal_files = sorted(os.listdir(normal_folder))

# 检查文件数量是否相同
if len(gray_files) != len(color_files):
    raise ValueError("两个文件夹中的图像数量不匹配，请确保它们具有相同数量的图像")

# 遍历文件夹中的图像文件
for gray_file, normal_file, color_file in zip(gray_files, normal_files, color_files):
    gray_path = os.path.join(gray_folder, gray_file)
    color_path = os.path.join(color_folder, color_file)
    normal_path = os.path.join(normal_folder, normal_file)
    # 读取图像
    p1 = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
    p2 = cv2.imread(color_path, cv2.IMREAD_COLOR)
    p3 = cv2.imread(normal_path, cv2.IMREAD_COLOR)

    # 应用掩码
    # result1 = apply_mask(p1, p3)
    # result1, result2 = mask_region(p1, p3, p2)
    # result1, result2 = mask_and_cut(p1, p3, p2)
    # 保存结果图像
    mask = p1 == 0
    p2[mask] = p3[mask]
    result1 = p2
    # result1 = guiwei(p1,p2,p3)
    normal_output_path = os.path.join(normal_output_folder, normal_file)
    cv2.imwrite(normal_output_path, result1)
    defective_output_path = os.path.join(defective_output_folder, color_file)
    #cv2.imwrite(defective_output_path, result2)

print("处理完成！")
