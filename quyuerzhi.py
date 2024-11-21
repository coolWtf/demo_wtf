import glob
import os

from PIL import Image, ImageDraw
import numpy as np

folder_path = 'E:\颈动脉斑块\shujvji-gongkai\SEGMENTATIONS\Computerized-CNR_IT'
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
i = 1
for txt_file in txt_files:
    if i % 6 != 1:
        i = i + 1
        continue
    # 读取文件
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    points = []
    for line in lines:
        # 将每行数据按空格分割为两个字符串，并将字符串转换为浮点数
        x, y = line.split()
        points.append((float(x), float(y)))

    print(points)

    # 创建一个空白图像
    img = Image.new('L', (618, 464), 0)

    # 将区域内的点填充为白色
    draw = ImageDraw.Draw(img)
    draw.polygon(points, fill=255, outline=255)

    # 将图像转换为Numpy数组
    img_array = np.array(img)

    # 存储图像
    Image.fromarray(img_array).save(
        'E:/颈动脉斑块/shujvji-gongkai/label/Computerized-CNR_IT/' + str((int((i - 1) / 6 + 1))).zfill(4) + '.jpg')
    i = i + 1
