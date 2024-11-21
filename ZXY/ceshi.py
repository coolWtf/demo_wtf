import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
import os

def read_txt_file(file_path):
    """读取 TXT 文件中的点云数据"""
    data = np.loadtxt(file_path)
    return data

class PointCloudVisualizer:
    def __init__(self, data, file_path):
        self.data = data
        self.file_path = file_path
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter = None
        self.selected_points = []

    def plot_point_cloud(self):
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        labels = self.data[:, 3]

        self.scatter = self.ax.scatter(x, y, z, c=labels, cmap='bwr', s=7, picker=True)
        self.ax.set_title("3D Point Cloud")
        self.ax.grid(False)
        self.ax.set_axis_off()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.add_save_button()

        plt.show()

    def on_pick(self, event):
        ind = event.ind
        for i in ind:
            self.selected_points.append(i)

    def on_select(self, verts):
        path = Path(verts)
        x = self.data[:, 0]
        y = self.data[:, 1]
        z = self.data[:, 2]
        pts = np.column_stack((x, y, z))

        ind = np.where(path.contains_points(pts[:, :2]))[0]  # 只检查 X 和 Y 坐标

        self.selected_points.extend(ind)
        self.update_labels()

    def update_labels(self):
        for i in self.selected_points:
            self.data[i, 3] = 1 - self.data[i, 3]

        self.selected_points = []
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.plot_point_cloud()

    def add_save_button(self):
        ax_save = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_data)

    def save_data(self, event):
        print("Save button clicked!")
        base, ext = os.path.splitext(self.file_path)
        modified_file_path = f"{base}_modify{ext}"
        np.savetxt(modified_file_path, self.data)
        print(f"Data saved to {modified_file_path}")

# 示例用法
file_path = r"E:\ZXY\visualization\models\ad_new\label_187_long.txt"  # 替换为您的 TXT 文件路径
data = read_txt_file(file_path)
visualizer = PointCloudVisualizer(data, file_path)
visualizer.plot_point_cloud()
