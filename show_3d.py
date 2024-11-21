import matplotlib.pyplot as plt
import numpy as np

# 假设 batch_size 为 1
batch_size = 1

# 生成示例的三维点数据
points = np.random.rand(100, 3)  # 100 个三维点

# 生成示例的真实值和预测值（假设在 0 到 1 之间）
ground_truth = np.random.rand(100)
predictions = np.random.rand(100)

# 将生成的数据放入列表或数组中，以模拟 batch
points_batch = [points]
ground_truth_batch = [ground_truth]
predictions_batch = [predictions]

for i in range(batch_size):
    points = points_batch[i]
    ground_truth = ground_truth_batch[i]
    predictions = predictions_batch[i]

    # Check if the number of points matches the number of ground truths and predictions
    if points.shape[0]!= ground_truth.shape[0] or points.shape[0]!= predictions.shape[0]:
        raise ValueError(f"The number of points does not match the number of ground truths or predictions for batch item {i}")

    fig = plt.figure(figsize=(12, 6))

    # Plot ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.grid(False)  # Disable the grid
    ax1.set_axis_off()  # Hide the axis
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=ground_truth, cmap='coolwarm')  # Use red-blue color map
    ax1.set_title(f'Ground Truth - Sample {i}')

    # Plot predictions
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.grid(False)  # Disable the grid
    ax2.set_axis_off()  # Hide the axis
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=predictions, cmap='coolwarm')  # Use red-blue color map
    ax2.set_title(f'EPT - Sample {i}')

    # 链接两个子图的视角
    ax1.azim = ax2.azim
    ax1.elev = ax2.elev
    ax1.dist = ax2.dist

    def on_move(event):
        if event.inaxes == ax1:
            ax2.azim = ax1.azim
            ax2.elev = ax1.elev
            ax2.dist = ax1.dist
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()