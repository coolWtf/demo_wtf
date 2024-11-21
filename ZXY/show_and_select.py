import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

class PointCloudVisualizer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.datas = [self.read_txt_file(file_path) for file_path in file_paths]
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot(231 + i, projection='3d') for i in range(6)]
        self.scatter_plots = []

        self.plot_all_point_clouds()
        self.sync_view()
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.add_save_button()

    def read_txt_file(self, file_path):
        """读取 TXT 文件中的点云数据"""
        data = np.loadtxt(file_path)
        return data

    def plot_point_cloud(self, ax, data, index):
        ax.clear()
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        labels = data[:, 3]

        scatter = ax.scatter(x, y, z, c=labels, cmap='bwr', s=1, picker=True)
        self.scatter_plots[index] = scatter

        ax.set_title(f"Point Cloud {index + 1}")
        ax.grid(False)
        ax.set_axis_off()

    def plot_all_point_clouds(self):
        for i, (ax, data) in enumerate(zip(self.axes, self.datas)):
            self.plot_point_cloud(ax, data, i)
            self.scatter_plots.append(None)

    def sync_view(self):
        def on_rotate(event):
            for ax in self.axes[1:]:
                ax.view_init(elev=self.axes[0].elev, azim=self.axes[0].azim)
            plt.draw()

        def on_zoom(event):
            for ax in self.axes[1:]:
                ax.set_xlim(self.axes[0].get_xlim())
                ax.set_ylim(self.axes[0].get_ylim())
                ax.set_zlim(self.axes[0].get_zlim())
            plt.draw()

        self.fig.canvas.mpl_connect('motion_notify_event', on_rotate)
        self.fig.canvas.mpl_connect('scroll_event', on_zoom)

    def on_pick(self, event):
        ind = event.ind
        artist = event.artist
        for i, scatter in enumerate(self.scatter_plots):
            if artist == scatter:
                for j in ind:
                    self.datas[i][j, 3] = 1 - self.datas[i][j, 3]
                self.plot_point_cloud(self.axes[i], self.datas[i], i)
                break
        plt.draw()

    def add_save_button(self):
        ax_save = plt.axes([0.8, 0.05, 0.1, 0.075])
        btn_save = Button(ax_save, 'Save')
        btn_save.on_clicked(self.save_data)

    def save_data(self, event):
        for i, data in enumerate(self.datas):
            np.savetxt(f'output_{i}.txt', data)
        print("Data saved!")

# 示例用法
file_paths = [f"fold_0_{i}.txt" for i in range(6)]  # 替换为您的 6 个 TXT 文件路径
visualizer = PointCloudVisualizer(file_paths)
plt.show()
