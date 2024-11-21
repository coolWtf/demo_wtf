import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def read_txt_file(file_path):
    data = np.loadtxt(file_path)
    return data


class PointCloudVisualizer:
    def __init__(self, data_list, labels_list, ground_truth):
        self.data_list = data_list
        self.labels_list = labels_list
        self.ground_truth = ground_truth
        self.fig = plt.figure(figsize=(18, 9))
        self.ax_list = []
        self.scatter_list = []
        self.point_size = 3
        self.plot_3d_subplots()

    def plot_3d_subplots(self):
        self.fig.clear()
        self.ax_list = []
        self.scatter_list = []

        ax_gt = self.fig.add_subplot(2, 3, 1, projection='3d')
        self.ax_list.append(ax_gt)
        ax_gt.grid(False)
        ax_gt.set_axis_off()
        x_gt = self.ground_truth[:, 0]
        y_gt = self.ground_truth[:, 1]
        z_gt = self.ground_truth[:, 2]
        labels_gt = self.ground_truth[:, 3]
        scatter_gt = ax_gt.scatter(x_gt, y_gt, z_gt, c=labels_gt, cmap='coolwarm', s=self.point_size)
        ax_gt.set_title('Ground Truth')

        ax_gt.set_xlim([x_gt.min(), x_gt.max()])
        ax_gt.set_ylim([y_gt.min(), y_gt.max()])
        ax_gt.set_zlim([z_gt.min(), z_gt.max()])
        ax_gt.view_init(elev=20., azim=120)

        for i, (data, labels) in enumerate(zip(self.data_list, self.labels_list), 2):
            ax = self.fig.add_subplot(2, 3, i, projection='3d')
            self.ax_list.append(ax)
            ax.grid(False)
            ax.set_axis_off()
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            scatter = ax.scatter(x, y, z, c=labels, cmap='coolwarm', s=self.point_size, picker=True)
            self.scatter_list.append(scatter)
            ax.set_title(f'Sample {i - 1}')

            ax.set_xlim([x.min(), x.max()])
            ax.set_ylim([y.min(), y.max()])
            ax.set_zlim([z.min(), z.max()])
            ax.view_init(elev=20., azim=120)

        ax1 = self.ax_list[0]
        for ax in self.ax_list[1:]:
            ax.azim = ax1.azim
            ax.elev = ax1.elev

        def on_move(event):
            if event.inaxes == ax1:
                for ax in self.ax_list[1:]:
                    ax.azim = ax1.azim
                    ax.elev = ax1.elev
                self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('motion_notify_event', on_move)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        plt.show()

    def on_pick(self, event):
        if event.artist not in self.scatter_list:
            return
        ind = event.ind[0]
        scatter = event.artist
        ax = scatter.axes
        ax_idx = self.ax_list.index(ax)
        labels = self.labels_list[ax_idx - 1]
        labels[ind] = 1 - labels[ind]
        self.redraw()

    def redraw(self):
        self.plot_3d_subplots()


file_gd = ['E:/ZXY/visualization/models/intra/ept/fold_3_sample_5_0.txt']
file_paths = [
    'E:/ZXY/visualization/models/intra/JGAnet_modify/fold_3_sample_5_0.txt',
    'E:/ZXY/visualization/models/intra/ept/fold_3_sample_5_0.txt',
    'E:/ZXY/visualization/models/intra/adaptconv/fold_3_19.txt',
    'E:/ZXY/visualization/models/intra/paconv/fold_3_19.txt',
    'E:/ZXY/visualization/models/intra/pointnet++/sample_24/sample_1.txt',
]

data_list = [read_txt_file(file_path) for file_path in file_paths]
gd_data = read_txt_file(file_gd[0])
original_labels = [data[:, 4].tolist() for data in data_list]

visualizer = PointCloudVisualizer(data_list, original_labels, gd_data)
