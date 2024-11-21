import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_txt_file(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    return data

def plot_3d_subplots(data_list):
    fig = plt.figure(figsize=(18, 9))

    for i, data in enumerate(data_list, 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        ax.grid(False)
        ax.set_axis_off()
        x = data[0]
        y = data[1]
        z = data[2]
        labels = data[6]  # 最后一列作为标签
        # 根据标签设置颜色
        scatter = ax.scatter(x, y, z, c=labels,cmap='coolwarm')
        ax.set_title(f'Sample {i}')

    # 链接子图的视角
    ax1 = fig.get_axes()[0]
    for ax in fig.get_axes()[1:]:
        ax.azim = ax1.azim
        ax.elev = ax1.elev
        ax.dist = ax1.dist

    def on_move(event):
        if event.inaxes == ax1:
            for ax in fig.get_axes()[1:]:
                ax.azim = ax1.azim
                ax.elev = ax1.elev
                ax.dist = ax1.dist
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()

# 读取六个 txt 文件
file_paths = ['AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad']
data_list = [read_txt_file(file_path) for file_path in file_paths]

plot_3d_subplots(data_list)