
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_txt_file(file_path):
    """读取txt文件中的点云数据"""
    data = np.loadtxt(file_path)
    return data

def plot_point_cloud(data, file_name):
    """绘制点云数据"""
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

    plt.show()

def main(folder_path):
    """读取文件夹中的所有txt文件并绘制点云"""
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = read_txt_file(file_path)
            plot_point_cloud(data, file_name)

# 设置txt文件所在的文件夹路径
folder_path = 'E:/ZXY/visualization/models/intra/adaptconv'

# 调用主函数
main(folder_path)
