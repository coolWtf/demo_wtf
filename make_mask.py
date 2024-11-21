import cv2
import numpy as np


def generate_mask(image):
    # 创建与输入图像相同大小的灰度图
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 生成起点和终点
    start_point = np.random.randint(0, width, size=(2))
    end_point = np.random.randint(0, width, size=(2))

    # 计算曲线点
    num_points = 10
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)

    # 生成平滑曲线
    for i in range(num_points - 1):
        cv2.line(mask, (int(x[i]), int(y[i])), (int(x[i + 1]), int(y[i + 1])), 255, 1)

    # 创建封闭区域的mask
    contours = np.column_stack((x, y)).astype(np.int32)
    cv2.fillPoly(mask, [contours], 255)

    # 检查曲线的角度是否大于10度
    angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    angle_deg = np.degrees(angle)
    if np.abs(angle_deg) <= 10:
        return generate_mask(image)

    # 调整mask大小
    resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 创建与输入图像相同大小的二值图像
    binary_image = np.zeros_like(image)

    # 将调整后的mask放置在二值图像中
    binary_image[resized_mask > 0] = 255

    return binary_image


# 示例用法
input_image = cv2.imread('000.png', 0)  # 读取输入图像，假设为灰度图
output_image = generate_mask(input_image)

cv2.imwrite('output_image4  00.jpg', output_image)  # 保存输出图像
