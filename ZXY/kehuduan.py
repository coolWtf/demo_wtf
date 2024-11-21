import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import scrolledtext

def read_txt_file(file_path):
    data = pd.read_csv(file_path, sep=' ', header=None)
    return data

def plot_3d_subplots(data_list, labels_list):
    fig = plt.figure(figsize=(18, 9))

    for i, (data, labels) in enumerate(zip(data_list, labels_list), 1):
        ax = fig.add_subplot(2, 3, i, projection='3d')
        ax.grid(False)
        ax.set_axis_off()
        x = data[0]
        y = data[1]
        z = data[2]
        # 使用新的标签
        scatter = ax.scatter(x, y, z, c=labels, cmap='coolwarm')
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

def update_labels():
    new_labels = [text_widgets[i].get("1.0", tk.END).split() for i in range(len(file_paths))]
    new_labels = [[label for label in labels] for labels in new_labels]  # 转换为列表列表
    plot_3d_subplots(data_list, new_labels)

# 读取文件并获取数据和初始标签
file_paths = ['AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad', 'AN27-_norm.ad']
data_list = [read_txt_file(file_path) for file_path in file_paths]
original_labels = [data[6].tolist() for data in data_list]  # 将数据列转换为列表

root = tk.Tk()
root.title("3D Plot Label Editor")

text_widgets = []  # 用于存储多个文本框

for i, labels in enumerate(original_labels):
    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
    for label in labels:  # 直接插入标签值，不添加序号
        text_widget.insert(tk.END, str(label) + " ")
    text_widget.pack()
    text_widgets.append(text_widget)  # 将文本框添加到列表

update_button = tk.Button(root, text="Update", command=update_labels)
update_button.pack()

root.mainloop()