import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_txt_file(file_path):
    """读取txt文件中的点云数据"""
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    return data


def plot_point_cloud(data, file_name):
    """绘制点云数据并保存为PDF文件"""
    if data is None:
        print("No data to plot.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标和标签（假设最后一列是标签）
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    if data.shape[1] > 4:  # 确保有标签列
        labels = data[:, 4]  # 注意：这里假设标签在第四列，如果不是请调整

        # 根据标签绘制不同颜色的点
        if np.any(labels == 0):
            ax.scatter(x[labels == 0], y[labels == 0], z[labels == 0], c='blue', label='Class 0', s=1)
        if np.any(labels == 1):
            ax.scatter(x[labels == 1], y[labels == 1], z[labels == 1], c='red', label='Class 1', s=1)

    ax.grid(False)
    ax.set_axis_off()

    # 保存为PDF文件
    plt.savefig(file_name + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭图形，释放资源


def main():
    base_file_names = [
        'ept_sample_3_2_modify',
        'ept_sample_3_2',
        'adaptconv_fold_0_12',
        'paconv_fold_0_12',
        'pointnet_plus_plus_originsampling_AN185'
        # 'pointnet_plus_plus_sample_11_sample_3'
    ]
    file_paths = [
        'E:/ZXY/visualization/models/intra/ept/fold_0_sample_3_2_modify.txt',
        'E:/ZXY/visualization/models/intra/ept/fold_0_sample_3_2.txt',
        'E:/ZXY/visualization/models/intra/adaptconv/fold_0_12.txt',
        'E:/ZXY/visualization/models/intra/paconv/fold_0_12.txt',
        r'E:\ZXY\visualization\models\intra\pointnet++originsampling\point.shape\AN185-_norm.ad'
        # r'E:\ZXY\visualization\models\intra\pointnet++\sample_11/sample_3.txt'
    ]

    for file_name, file_path in zip(base_file_names, file_paths):
        data = read_txt_file(file_path)
        if data is not None:
            plot_point_cloud(data, file_name)

        # 调用主函数


main()