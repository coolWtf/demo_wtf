import os

import numpy as np
from PIL import Image, ImageDraw

def generate_mask(image):
    """
    在原图中随机找一个点，以这个点为中心，绘制一个封闭的不规则曲线，区域大小约为20*50像素，
    输出灰度图，封闭曲线内区域为mask，值为255，其余部分为0。

    Args:
        image: 原图像，PIL.Image对象

    Returns:
        mask: 灰度掩膜，numpy.ndarray数组，值为0或255，形状为原图像大小
    """

    # 生成随机的封闭曲线
    x, y = np.random.randint(20, image.width - 20), np.random.randint(50, image.height - 50)
    r = 20
    theta = np.linspace(0, 2*np.pi, 200)
    xs = x + r * np.cos(theta)
    ys = y + r * np.sin(theta)
    xs += np.random.normal(scale=5, size=xs.shape)
    ys += np.random.normal(scale=5, size=ys.shape)
    xs = np.clip(xs, 0, image.width - 1).astype(int)
    ys = np.clip(ys, 0, image.height - 1).astype(int)
    polygon = list(zip(xs, ys))

    # 绘制封闭曲线
    mask = Image.new(mode='L', size=image.size, color=0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)


    # 转换为numpy数组
    mask = np.array(mask)

    return mask

# 指定图像文件夹路径
gray_folder = r"E:\QuqXian\mvtec_anomaly_detection\leather_r\train\good"
color_folder = r"E:\QuqXian\mvtec_anomaly_detection\leather_r\train\good"
normal_folder = r"E:\QuqXian\mvtec_anomaly_detection\leather_r\train\good"

# 输出文件夹路径
normal_output_folder = r"E:\Defect_detection\GANdehaze\dataset\squee-mask-to-mask\mask"
defective_output_folder = r"E:\Defect_detection\GANdehaze\dataset\squee-qxjc-nu\GT"
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
for gray_file,normal_file, color_file in zip(gray_files,normal_files, color_files):
    gray_path = os.path.join(gray_folder, gray_file)
    color_path = os.path.join(color_folder, color_file)
    normal_path = os.path.join(normal_folder, normal_file)
    # 读取图像
    image = Image.open(normal_path)
    mask = generate_mask(image)
    out = Image.fromarray(mask)
    normal_output_path = os.path.join(normal_output_folder, normal_file)
    out.save(normal_output_path)

print("处理完成！")

