import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages


def read_txt_file(file_path):
    data = np.loadtxt(file_path)
    return data


def save_data(file_path, data):
    np.savetxt(file_path, data, fmt='%f')
    print(f"Data saved to {file_path}")


class PointCloudVisualizer:
    def __init__(self, data_list, labels_list, ground_truth, file_paths):
        self.data_list = data_list
        self.labels_list = labels_list
        self.ground_truth = ground_truth
        self.file_paths = file_paths
        self.fig = plt.figure(figsize=(18, 9))
        self.ax_list = []
        self.scatter_list = []
        self.point_size = 1
        self.plot_3d_subplots()

    def plot_3d_subplots(self):
        self.fig.clear()
        self.ax_list = []
        self.scatter_list = []

        # Create subplots in a single row
        num_subplots = len(self.data_list) + 1
        for i in range(num_subplots):
            ax = self.fig.add_subplot(1, num_subplots, i + 1, projection='3d')
            self.ax_list.append(ax)
            ax.grid(False)
            ax.set_axis_off()

        ax_gt = self.ax_list[0]
        x_gt = self.ground_truth[:, 0]
        y_gt = self.ground_truth[:, 1]
        z_gt = self.ground_truth[:, 2]
        labels_gt = self.ground_truth[:, 3]
        scatter_gt = ax_gt.scatter(x_gt, y_gt, z_gt, c=labels_gt, cmap='coolwarm', s=self.point_size)

        ax_gt.set_xlim([x_gt.min(), x_gt.max()])
        ax_gt.set_ylim([y_gt.min(), y_gt.max()])
        ax_gt.set_zlim([z_gt.min(), z_gt.max()])
        ax_gt.view_init(elev=20., azim=120)

        for i, (data, labels) in enumerate(zip(self.data_list, self.labels_list)):
            ax = self.ax_list[i + 1]
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            scatter = ax.scatter(x, y, z, c=labels, cmap='coolwarm', s=self.point_size, picker=True)
            self.scatter_list.append(scatter)

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

        # Add save buttons
        ax_save_data = plt.axes([0.35, 0.01, 0.1, 0.05])
        btn_save_data = Button(ax_save_data, 'Save Data')
        btn_save_data.on_clicked(self.save_all_data)

        ax_save_pdf = plt.axes([0.55, 0.01, 0.1, 0.05])
        btn_save_pdf = Button(ax_save_pdf, 'Save PDFs')
        btn_save_pdf.on_clicked(self.save_all_pdfs)

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
        self.update_single_subplot(ax_idx, labels)

    def update_single_subplot(self, ax_idx, labels):
        ax = self.ax_list[ax_idx]
        ax.clear()
        data = self.data_list[ax_idx - 1]
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        scatter = ax.scatter(x, y, z, c=labels, cmap='coolwarm', s=self.point_size, picker=True)
        self.scatter_list[ax_idx - 1] = scatter
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_zlim([z.min(), z.max()])
        ax.view_init(elev=ax.elev, azim=ax.azim)
        ax.grid(False)
        ax.set_axis_off()
        plt.draw()

    def save_all_data(self, event):
        for data, labels, file_path in zip(self.data_list, self.labels_list, self.file_paths):
            data[:, 4] = labels
            save_data(file_path, data)

    def save_all_pdfs(self, event):
        for i, ax in enumerate(self.ax_list):
            with PdfPages(f'output_subplot_{i + 1}.pdf') as pdf:
                fig_single, single_ax = plt.subplots(subplot_kw={'projection': '3d'})
                scatter = ax.collections[0]

                # Use scatter._offsets3d for 3D scatter data
                if hasattr(scatter, '_offsets3d'):
                    x, y, z = scatter._offsets3d
                    single_ax.scatter(x, y, z, c=scatter.get_facecolors(), s=scatter.get_sizes())
                else:
                    print(f'No 3D offsets found for subplot {i + 1}')
                    continue

                single_ax.set_xlim(ax.get_xlim())
                single_ax.set_ylim(ax.get_ylim())
                single_ax.set_zlim(ax.get_zlim())
                single_ax.view_init(elev=ax.elev, azim=ax.azim)
                single_ax.grid(False)
                single_ax.set_axis_off()
                pdf.savefig(fig_single, bbox_inches='tight')
                plt.close(fig_single)
                print(f'Saved output_subplot_{i + 1}.pdf')


file_gd = ['E:/ZXY/visualization/models/intra/ept/fold_1_sample_5_0.txt']
file_paths = [
    'E:/ZXY/visualization/models/intra/ept/fold_1_sample_5_0_modify808_modify.txt',

    'E:/ZXY/visualization/models/intra/adaptconv/fold_1_22_modify.txt',
'E:/ZXY/visualization/models/intra/ept/fold_1_sample_5_0_modify_modify_modify.txt',
    'E:/ZXY/visualization/models/intra/paconv/fold_1_22.txt',
    r'E:\ZXY\visualization\models\intra\pointnet++originsampling\point.shape\AN9-2-_norm_modify.ad'
    # r'E:\ZXY\visualization\models\intra\pointnet++\sample_11/sample_3.txt'
]

data_list = [read_txt_file(file_path) for file_path in file_paths]
gd_data = read_txt_file(file_gd[0])
original_labels = [data[:, 4].tolist() for data in data_list]

visualizer = PointCloudVisualizer(data_list, original_labels, gd_data, file_paths)
