import numpy as np
from PIL import Image

def get_diff(img_arr):
    diff = []
    for i in range(len(img_arr)):
        row_diff = []
        for j in range(len(img_arr[0])):
            if j == 0:
                row_diff.append(0)
            else:
                row_diff.append(abs(int(img_arr[i][j]) - int(img_arr[i][j-1])))
        diff.append(row_diff)

    # 将差异矩阵进行归一化处理
    max_val = np.amax(np.absolute(np.diff(img_arr)))
    diff = (diff / max_val) * 255
    return diff.astype(np.uint8)

def dynamic_programming(diff):
    n = len(diff)
    m = len(diff[0])

    # 初始化最小代价矩阵和路径矩阵
    min_cost = [[0] * m for _ in range(n)]
    path = [[0] * m for _ in range(n)]

    # 初始化第一列
    for i in range(n):
        min_cost[i][0] = diff[i][0]
        path[i][0] = -1

    # 动态规划计算最小代价矩阵和路径矩阵
    for j in range(1, m):
        # 使用斜率优化技巧
        q = [i for i in range(n)]
        head, tail = 0, 0
        for i in range(n):
            while head < tail and (min_cost[q[head+1]][j-1] - min_cost[q[head]][j-1]) <= (q[head+1] - q[head]) * diff[i][j]:
                head += 1
            k = q[head]
            if k > 0:
                ub = min(k + 1, n)
            else:
                ub = min(k + 2, n)
            lb = max(k - 1, 0)
            min_val = min([min_cost[x][j-1] + abs(x-i)*diff[i][j] for x in range(lb, ub)])
            min_idx = np.argmin([min_cost[x][j-1] + abs(x-i)*diff[i][j] for x in range(lb, ub)])
            min_cost[i][j] = min_val
            path[i][j] = lb + min_idx
            while head < tail and (min_cost[i][j-1]-min_cost[q[tail-1]][j-1])*(q[tail]-q[tail-1]) <= (min_cost[q[tail]][j-1]-min_cost[q[tail-1]][j-1])*(i-q[tail-1]):
                tail -= 1
            q.append(i)
            tail += 1

    # 从最后一列中选择一个最小的元素作为路径起点
    start = 0
    for i in range(n):
        if min_cost[i][m-1] < min_cost[start][m-1]:
            start = i

    # 构建最小代价路径
    p = []
    for j in range(m-1, -1, -1):
        p.append(start)
        start = path[start][j]

    return p[::-1]


# 将灰度图像转化为矩阵
img = Image.open('000.png').convert('L')
img_arr = np.array(img)

# 计算相邻像素点之间的差异值并进行归一化处理
diff = get_diff(img_arr)

# 使用动态规划计算最小代价路径
min_cost_path = dynamic_programming(diff)

# 压缩图像
compressed_img_arr = []
for i in range(len(min_cost_path)):
    compressed_img_arr.append(img_arr[min_cost_path[i]][i])

# 将压缩后的图像转化为PIL格式，并保存原图像和压缩后的图像
compressed_img_arr = np.array(compressed_img_arr).reshape((-1, len(min_cost_path)))
compressed_img = Image.fromarray(compressed_img_arr.astype(np.uint8))
Image.fromarray(img_arr).save('original_image.jpg')
compressed_img.save('compressed_image.jpg')
