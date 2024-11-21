import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_txt_file(file_path):
    """读取txt文件中的点云数据"""
    data = np.loadtxt(file_path)
    return data

def plot_point_cloud_and_save(data, file_name, save_path):
    """绘制点云数据并保存"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标和标签
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    labels = data[:, 4]

    # 根据标签绘制不同颜色的点
    ax.scatter(x[labels == 0], y[labels == 0], z[labels == 0], c='blue', label='Class 0', s=1)
    ax.scatter(x[labels == 1], y[labels == 1], z[labels == 1], c='red', label='Class 1', s=1)

    ax.set_title(file_name)
    ax.legend()
    ax.grid(False)
    ax.set_axis_off()

    # 保存图像
    plt.savefig(os.path.join(save_path, file_name.replace('.ad', '.png')))
    plt.close()  # 关闭图形以释放资源

def traverse_folders(folder_path):
    """遍历文件夹及子文件夹"""
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.ad'):
                file_path = os.path.join(root, file_name)
                data = read_txt_file(file_path)
                plot_point_cloud_and_save(data, file_name, root)

# 设置文件夹路径
folder_path = r"E:\ZXY\visualization\models\intra\pointnet++originsampling\point.shape"

# 调用遍历函数
traverse_folders(folder_path)