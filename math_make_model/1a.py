from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans


def read_excel_file(file_path):
    try:
        # 使用 pandas 读取 Excel 文件
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return None


ex4 = r"E:\数学建模\E题\数据\竞赛发布数据\表4-答案文件.xlsx"  # 将文件路径替换为你的 Excel 文件路径


# 读入所有excel表
def read_all():
    ex1 = r"E:\数学建模\E题\数据\竞赛发布数据\表1-患者列表及临床信息.xlsx"  # 将文件路径替换为你的 Excel 文件路径
    dataframe = read_excel_file(ex1)
    ex_f1 = r"E:\数学建模\E题\数据\竞赛发布数据\附表1-检索表格-流水号vs时间.xlsx"  # 将文件路径替换为你的 Excel 文件路径
    dataframe_exf1 = read_excel_file(ex_f1)
    ex2 = r"E:\数学建模\E题\数据\竞赛发布数据\表2-患者影像信息血肿及水肿的体积及位置.xlsx"  # 将文件路径替换为你的 Excel 文件路径
    dataframe_ex2 = read_excel_file(ex2)
    ex3 = r"E:\数学建模\E题\数据\竞赛发布数据\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx"  # 将文件路径替换为你的 Excel 文件路径
    dataframe_ex3 = read_excel_file(ex3)
    ex4 = r"E:\数学建模\E题\数据\竞赛发布数据\表4-答案文件.xlsx"  # 将文件路径替换为你的 Excel 文件路径
    dataframe_ex4 = read_excel_file(ex4)
    return dataframe, dataframe_exf1, dataframe_ex2, dataframe_ex3, dataframe_ex4


dataframe, dataframe_exf1, dataframe_ex2, dataframe_ex3, dataframe_ex4 = read_all()

# 获取具体参数值
first_jc_id = dataframe['入院首次影像检查流水号'][:100]
time_fir_jc = dataframe['发病到首次影像检查时间间隔'][:100]
fir_ry_time = dataframe_exf1['入院首次检查时间点'][:100]
## 题目1a
# 获得首次发病时间
fir_fb_time = []


def get_fitst_fb_time():
    for t, y in zip(time_fir_jc, fir_ry_time):
        minutes_to_subtract = t * 60  # 例如，减去30分钟
        # 使用 timedelta 减去分钟数
        new_datetime_obj = y - pd.to_timedelta(minutes_to_subtract, unit='m')
        fir_fb_time.append(new_datetime_obj)
    return fir_fb_time


fir_fb_time = get_fitst_fb_time()

# 用数组记录病人 以及是否血肿变大，以及血肿变大时间
sub_xz_time = []


def get_sub_xz_time():
    # 遍历sub001到sub100号病人判定他们是否在发病后四十八小时内是否发生血肿扩张事件   表2
    for index, row in dataframe_ex2.iloc[:100].iterrows():
        # 这里的 row 是一个包含每一行数据的 Series 对象 i代表首次长度，j代表第二次血仲体积
        i = 2
        j = i + 23
        fir_hm = row[2]
        sec_hm = row[j]
        if sec_hm / fir_hm > 1.33 or sec_hm - fir_hm > 6000:
            selected_row = dataframe_exf1[dataframe_exf1['随访1流水号'] == row[j - 1]]
            xz_time = selected_row['随访1时间点']
            hour_difference = (xz_time - fir_fb_time[index]).dt.total_seconds() / 3600
            hour_difference = float(hour_difference)
            sub_xz_time.append([1, hour_difference])
        else:
            j = j + 23
            sec_hm = row[j]
            selected_row = dataframe_exf1[dataframe_exf1['随访' + str(int((j - 2) / 23)) + '流水号'] == row[j - 1]]
            xz_time = selected_row['随访' + str(int((j - 2) / 23)) + '时间点']
            # 计算时间戳之间的小时差
            hour_difference = (xz_time - fir_fb_time[index]).dt.total_seconds() / 3600
            hour_difference = float(hour_difference)
            while hour_difference <= 48.0:
                if sec_hm / fir_hm > 1.33 or sec_hm - fir_hm > 6000:
                    selected_row = dataframe_exf1[
                        dataframe_exf1['随访' + str(int((j - 2) / 23)) + '流水号'] == row[j - 1]]
                    xz_time = selected_row['随访' + str(int((j - 2) / 23)) + '时间点']
                    hour_difference = (xz_time - fir_fb_time[index]).dt.total_seconds() / 3600
                    hour_difference = float(hour_difference)
                    sub_xz_time.append([1, hour_difference])
                    break
                j = j + 23
                if pd.isna(row[j - 1]):
                    sub_xz_time.append([0, hour_difference])
                    break
                selected_row = dataframe_exf1[dataframe_exf1['随访' + str(int((j - 2) / 23)) + '流水号'] == row[j - 1]]
                xz_time = selected_row['随访' + str(int((j - 2) / 23)) + '时间点']
                # 计算时间戳之间的小时差
                hour_difference = (xz_time - fir_fb_time[index]).dt.total_seconds() / 3600
                hour_difference = float(hour_difference)
            if hour_difference > 48.0:
                sub_xz_time.append([0, hour_difference])
    return sub_xz_time


sub_xz_time = get_sub_xz_time()


# 将sub_xz_time写入到excel表中
def write_xztime_to_ex4():
    for index, row in dataframe_ex4.iloc[2:102].iterrows():
        print(index)
        if sub_xz_time[index - 2][0] == 0:
            row[2] = 0
        else:
            row[2] = 1
            row[3] = sub_xz_time[index - 2][1]
        print(row[0], row[2], row[3])
    # dataframe_ex4.to_excel(ex4, index=False)


## 题目1b
# 数据分析 画出各个特征的分布以及肿大的占整体的分布

# 将目标值 （肿大的人群特征找出来） 示例 表一
def draw_t1_f1():
    target_bs = []
    target_bs_sum = []
    all_bs_sum = []
    for index, row in dataframe[:100].iterrows():
        all_bs_sum.append(int(row[6] + row[7] + row[8] + row[9] + row[10] + row[11] + row[12] + row[13]))
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[6:14])
            target_bs_sum.append(int(row[6] + row[7] + row[8] + row[9] + row[10] + row[11] + row[12] + row[13]))
    # 绘制直方图
    # 创建3x3子图
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    # 绘制每种属性直方图
    qishi_i = 6
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if qishi_i < 14:
                column = dataframe[dataframe.columns[qishi_i]][:100]
                # 计算两组数据的分箱边界
                bin_edges = np.histogram_bin_edges(column, bins=5)
                ax.hist(column, bins=bin_edges, color='blue', alpha=0.5,
                        label='All Samples')
                # 使用转置获取第一列的所有值
                column_values = list(zip(*target_bs))[qishi_i - 6]
                ax.hist(column_values, bins=bin_edges, color='red', alpha=0.5, label='Target Samples')
                ax.set_title('Histogram of ' + str(qishi_i - 5), fontsize=6)
                ax.set_xlabel('Value', fontsize=6)
                ax.set_ylabel('Frequency', fontsize=6)
            else:
                ax.hist(all_bs_sum, bins=10, color='blue', alpha=0.5, label='All Samples')
                ax.hist(target_bs_sum, bins=10, color='red', alpha=0.5, label='Target Samples')
                ax.set_title('Histogram of sum', fontsize=6)
                ax.set_xlabel('Value', fontsize=6)
                ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    legend.get_texts()[0].set_fontsize(6)  # 设置图例文字字体大小
    legend.get_texts()[1].set_fontsize(6)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t1_f1()

# 表一 图2
# 将目标值 （肿大的人群特征找出来） 示例 表一
def draw_t1_f2():
    target_bs = []
    target_bs_sum = []
    all_bs_sum = []
    for index, row in dataframe[:100].iterrows():
        all_bs_sum.append(int(row[16] + row[17] + row[18] + row[19] + row[20] + row[21] + row[22]))
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[16:23])
            target_bs_sum.append(int(row[16] + row[17] + row[18] + row[19] + row[20] + row[21] + row[22]))

    # 绘制直方图
    # 创建4x2子图
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    qishi_i = 16
    for i in range(2):
        for j in range(4):
            ax = axes[i, j]
            if qishi_i < 23:
                column = dataframe[dataframe.columns[qishi_i]][:100]
                # 计算两组数据的分箱边界
                bin_edges = np.histogram_bin_edges(column, bins=5)
                ax.hist(column, bins=bin_edges, color='blue', alpha=0.5,
                        label='All Samples')
                # 使用转置获取第一列的所有值
                column_values = list(zip(*target_bs))[qishi_i - 16]
                ax.hist(column_values, bins=bin_edges, color='red', alpha=0.5, label='Target Samples')
                ax.set_title('Histogram of ' + str(qishi_i - 15), fontsize=6)
                ax.set_xlabel('Value', fontsize=6)
                ax.set_ylabel('Frequency', fontsize=6)
            else:
                ax.hist(all_bs_sum, bins=10, color='blue', alpha=0.5, label='All Samples')
                ax.hist(target_bs_sum, bins=10, color='red', alpha=0.5, label='Target Samples')
                ax.set_title('Histogram of sum', fontsize=6)
                ax.set_xlabel('Value', fontsize=6)
                ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    legend.get_texts()[0].set_fontsize(6)  # 设置图例文字字体大小
    legend.get_texts()[1].set_fontsize(6)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t1_f2()

## 绘制表格2 两个总体积
def draw_t2_f0():
    target_bs = []
    target_bs1 = []
    for index, row in dataframe_ex2[:100].iterrows():
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[2])
            target_bs1.append(row[13])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    ax = axes[0]
    column = dataframe_ex2[dataframe_ex2.columns[2]][:100]
    # 计算两组数据的分箱边界
    bin_edges = np.histogram_bin_edges(column, bins=20)
    ax.hist(column, bins=bin_edges, color='blue', alpha=0.5, label='All Samples')
    ax.hist(target_bs, bins=bin_edges, color='red', alpha=0.5, label='Target Samples')
    ax.set_title('Histogram of ' + str(dataframe_ex2.columns[2]), fontsize=6)
    ax.set_xlabel('Value', fontsize=6)
    ax.set_ylabel('Frequency', fontsize=6)

    ax = axes[1]
    column1 = dataframe_ex2[dataframe_ex2.columns[13]][:100]
    # 计算两组数据的分箱边界
    bin_edges1 = np.histogram_bin_edges(column1, bins=20)
    ax.hist(column1, bins=bin_edges1, color='blue', alpha=0.5, label='All Samples')
    ax.hist(target_bs1, bins=bin_edges1, color='red', alpha=0.5, label='Target Samples')
    ax.set_title('Histogram of ' + str(dataframe_ex2.columns[13]), fontsize=6)
    ax.set_xlabel('Value', fontsize=6)
    ax.set_ylabel('Frequency', fontsize=6)

    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    legend.get_texts()[0].set_fontsize(6)  # 设置图例文字字体大小
    legend.get_texts()[1].set_fontsize(6)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t2_f0()
# 绘制表格2 十个分区血肿图
def draw_t2_f1():
    target_bs = []
    for index, row in dataframe_ex2[:100].iterrows():
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[3:13])

    # 绘制直方图
    # 创建5x2子图
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    qishi_i = 3
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            column = dataframe_ex2[dataframe_ex2.columns[qishi_i]][:100]
            # 计算两组数据的分箱边界
            bin_edges = np.histogram_bin_edges(column, bins=20)
            ax.hist(column, bins=bin_edges, color='blue', alpha=0.5,
                    label='All Samples')
            # 使用转置获取第一列的所有值
            column_values = list(zip(*target_bs))[qishi_i - 3]
            ax.hist(column_values, bins=bin_edges, color='red', alpha=0.5, label='Target Samples')
            ax.set_title('Histogram of ' + str(qishi_i - 2), fontsize=6)
            ax.set_xlabel('Value', fontsize=6)
            ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    legend.get_texts()[0].set_fontsize(6)  # 设置图例文字字体大小
    legend.get_texts()[1].set_fontsize(6)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t2_f1()
# 绘制表格2  十个分区水肿图
def draw_t2_f2():
    target_bs = []
    for index, row in dataframe_ex2[:100].iterrows():
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[14:24])

    # 绘制直方图
    # 创建5x2子图
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
    qishi_i = 14
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            column = dataframe_ex2[dataframe_ex2.columns[qishi_i]][:100]
            # 计算两组数据的分箱边界
            bin_edges = np.histogram_bin_edges(column, bins=20)
            ax.hist(column, bins=bin_edges, color='blue', alpha=0.5,
                    label='All Samples')
            # 使用转置获取第一列的所有值
            column_values = list(zip(*target_bs))[qishi_i - 14]
            ax.hist(column_values, bins=bin_edges, color='red', alpha=0.5, label='Target Samples')
            ax.set_title('Histogram of ' + str(qishi_i - 13), fontsize=6)
            ax.set_xlabel('Value', fontsize=6)
            ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    legend.get_texts()[0].set_fontsize(6)  # 设置图例文字字体大小
    legend.get_texts()[1].set_fontsize(6)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t2_f2()

# 构建表3 得到sub1-100的初诊信息
tabel_5 = []


def write_tb3_to_tb5by_sub():
    ex5 = r"E:\数学建模\E题\数据\竞赛发布数据\tabel5.xlsx"
    for index, row in dataframe_ex2[:].iterrows():
        selected_row = dataframe_ex3[dataframe_ex3['流水号'] == row[1]]
        if not selected_row.empty and len(selected_row) > 0:
            row_list = selected_row.values.tolist()[0]
            row_list[0] = 'sub' + str(index + 1)
            tabel_5.append(row_list)
        else:
            # 当没有找到匹配行时，补充每个元素为0的占位符列表
            placeholder_row = [0] * len(dataframe_ex3.columns)
            placeholder_row[0] = 'sub' + str(index + 1)
            tabel_5.append(placeholder_row)
            print('sub' + str(index + 1) + 'mess_loss,default =0 ')
    # 将列表转换为DataFrame，使用dataframe_ex3的列名作为列名
    columns = dataframe_ex3.columns.tolist()  # 获取dataframe_ex3的列名
    tabel_5_df = pd.DataFrame(tabel_5, columns=columns)
    # 将DataFrame写入Excel文件，不写入行索引
    # tabel_5_df.to_excel(ex5, index=False)
    return tabel_5, tabel_5_df


tabel_5, tabel_5_df = write_tb3_to_tb5by_sub()


# 画图 表3 2d 14个字段 2*6+5分布
def draw_t3_f1():
    target_bs = []
    for index, row in enumerate(tabel_5):
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[2:16])
    list1 = target_bs[:][0]
    # 绘制直方图
    # 创建4x2子图
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(15, 9))
    qishi_i = 2
    for i in range(3):
        for j in range(5):
            ax = axes[i, j]
            # 获得第几列的值 先转置
            column_values = list(zip(*tabel_5))[qishi_i]
            ax.hist(column_values, bins=20, color='blue', alpha=0.5, label='All Samples')
            column_values_tar = list(zip(*target_bs))[qishi_i - 2]
            ax.hist(column_values_tar, bins=20, color='red', alpha=0.5, label='Target Samples')
            ax.set_title('Histogram of ' + str(qishi_i - 1), fontsize=6)
            ax.set_xlabel('Value', fontsize=6)
            ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
            if qishi_i > 15:
                break
    # 调整子图的布局
    plt.tight_layout()
    legend = plt.legend()
    # legend.get_frame().set_linewidth(0.1)  # 设置图例边框宽度
    # legend.get_texts()[0].set_fontsize(5)  # 设置图例文字字体大小
    # legend.get_texts()[1].set_fontsize(5)  # 设置图例文字字体大小
    # 显示图形
    plt.show()


# draw_t3_f1()

# 画图，表3 后17个字段
def draw_t3_f2():
    target_bs = []
    for index, row in enumerate(tabel_5):
        if sub_xz_time[index][0] == 1:
            target_bs.append(row[16:33])
    list1 = target_bs[:][0]
    # 绘制直方图
    # 创建4x2子图
    fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18, 9))
    qishi_i = 16
    for i in range(3):
        for j in range(6):
            ax = axes[i, j]
            # 获得第几列的值 先转置
            column_values = list(zip(*tabel_5))[qishi_i]
            ax.hist(column_values, bins=20, color='blue', alpha=0.5, label='All Samples')
            column_values_tar = list(zip(*target_bs))[qishi_i - 16]
            ax.hist(column_values_tar, bins=20, color='red', alpha=0.5, label='Target Samples')
            ax.set_title('Histogram of ' + str(qishi_i - 1), fontsize=6)
            ax.set_xlabel('Value', fontsize=6)
            ax.set_ylabel('Frequency', fontsize=6)
            qishi_i = qishi_i + 1
            if qishi_i > 32:
                break
    # 调整子图的布局
    plt.tight_layout()
    plt.legend()
    # 显示图形
    plt.show()


# draw_t3_f2()


# 主成分分析法加线性回归预测
def _1b_xxhg():
    # 构建原始特征集 100* 8+8+2+10+10+14+17=70
    t1_1 = dataframe.iloc[:, 6:14]
    t1_2 = dataframe.iloc[:, 16:23]
    t2_1 = dataframe_ex2.iloc[:, 2:24]
    t3_1 = tabel_5_df.iloc[:, 2:]
    # 合并这些特征列成一个特征矩阵
    feature_matrix = pd.concat([t1_1, t1_2, t2_1, t3_1], axis=1)

    print(feature_matrix.shape)

    # 创建一个PCA对象，指定主成分数量
    n_components = 5  # 假设选择前5个主成分
    pca = PCA(n_components=n_components)

    # 对特征矩阵进行PCA变换
    pca_result = pca.fit_transform(feature_matrix)

    test_fea = pca_result[100:160]
    pca_result = pca_result[:100]

    y_train = list(zip(*sub_xz_time))[0]
    model = LinearRegression()
    model.fit(pca_result, y_train)
    train_pred = model.predict(pca_result)
    test_pred = model.predict(test_fea)

    y_pred = np.concatenate((train_pred, test_pred))
    print(y_pred)
    print("归一化后")
    # 计算最小值和最大值
    min_value = min(y_pred)
    max_value = max(y_pred)

    # 对预测结果进行归一化
    normalized_predictions = [(x - min_value) / (max_value - min_value) for x in y_pred]
    print(normalized_predictions)

    normalized_predictions = np.array(normalized_predictions)
    y_pred = np.where(normalized_predictions > 0.655, 1, 0)
    y_train = np.array(y_train)
    y_pred = y_pred[:100]
    acc = np.mean(y_pred == y_train)
    print("Accuracy:", acc)
    # # 将结果写入表中
    # for index, row in dataframe_ex4.iloc[2:].iterrows():
    #     row[4] = normalized_predictions[index-2]
    # dataframe_ex4.to_excel(ex4, index=False)
# _1b_xxhg()

# 2.a
# 1 画出数据散点图？ 数据结构 [time,value] 先不画 确定使用每个患者的前三次随访记录
# 2 构建训练数据集，三个时间点以及时间点对应的体积，时间选用距离发病时间
# 定义存储数据从结构time[] value[](target) time是指距离发病时间
all_times = []
all_volumes = []

# 构建分类数据集
feature_sub_sz = []

# 构建1.c [sub_id,水肿增长速率]
sub_volumes_s = []
# 构建1.d [sub_id,血肿增长速率]
sub_volumes_x = []

# 首先遍历附表1，前一百个病人
for index, row in dataframe_exf1.iloc[:100].iterrows():
    # 第一次时间等于发病时间,找到表1距离发病时间
    fir_t = dataframe.iloc[index, 14]
    fit_v_x = dataframe_ex2.iloc[index, 2]
    fir_v = dataframe_ex2.iloc[index, 13]
    # 第二次时间等于减去第一次时间加上fir_t
    sec_t = float((row[4] - row[2]).total_seconds() / 3600) + fir_t
    sec_v = dataframe_ex2.iloc[index, 36]
    # 第三次时间等于减去第二次加上sec_t
    tir_t = float((row[6] - row[4]).total_seconds() / 3600) + sec_t
    tir_v_x = dataframe_ex2.iloc[index, 48]
    tir_v = dataframe_ex2.iloc[index, 59]
    all_times.append(fir_t)
    all_volumes.append(fir_v)
    all_times.append(sec_t)
    all_volumes.append(sec_v)
    all_times.append(tir_t)
    all_volumes.append(tir_v)
    # 获得第一阶段增长速率以及第二阶段增长速率
    speed0 = (sec_v - fir_v) / (sec_t - fir_t)
    speed1 = (tir_v - sec_v) / (tir_t - sec_t)
    feature_sub_sz.append([fir_v, speed0, speed1, tir_v])
    speed_s = (tir_v - fir_v) / (tir_t - fir_t)
    sub_volumes_s.append([index, speed_s])
    speed_x = (tir_v_x - fit_v_x) / (tir_t - fir_t)
    sub_volumes_x.append([index, speed_x])

# 定义拟合函数，这里使用指数函数作为示例
def exponential_func(t, a, b, c):
    return a * np.exp(b * t) + c


# 转换为NumPy数组
all_times = np.array(all_times)
all_volumes = np.array(all_volumes)


# 使用 curve_fit 进行拟合
def nihe_zhishu():
    params, covariance = curve_fit(exponential_func, all_times, all_volumes)

    # 获取拟合参数
    a, b, c = params

    # 计算拟合曲线上的点
    fit_time = np.linspace(min(all_times), max(all_times), 100)
    fit_volume = exponential_func(all_times, a, b, c)

    # 计算残差
    residuals = np.array(all_volumes) - fit_volume

    max_residual = np.max(np.abs(residuals))
    y_limit = max_residual * 2
    # 绘制残差图
    plt.scatter(all_times, residuals, color='green')
    plt.axhline(y=0, color='red', linestyle='--')  # 添加水平线表示残差为0
    plt.ylim(-y_limit, y_limit)  # 设置纵坐标范围
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.show()

    # 输出拟合参数
    print(f'a: {a}, b: {b}, c: {c}')


# 进行多项式拟合，这里选择3次多项式
def nihe_duoxiang():
    degree = 5
    coeffs = np.polyfit(all_times, all_volumes, degree)

    # 构建多项式函数
    poly_func = np.poly1d(coeffs)

    # 计算拟合曲线上的点
    fit_time = np.linspace(min(all_times), max(all_times), 100)
    fit_volume = poly_func(fit_time)

    # 绘制拟合曲线
    plt.scatter(all_times, all_volumes, label='Data', color='blue')
    plt.plot(fit_time, fit_volume, label='Fit', color='red')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()


# 聚类
def jvlei_fenzu(feature_sub_sz):
    feature_sub_sz = np.array(feature_sub_sz)
    print(feature_sub_sz.shape)
    # 创建K均值聚类模型，将病人分为5个亚组
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(feature_sub_sz)

    # 获取每个病人所属的亚组标签
    cluster_labels = kmeans.labels_
    print(cluster_labels)
    # 现在 cluster_labels 中包含了每个病人所属的亚组标签
    # 您可以将这些标签添加到原始数据中，以便进一步分析或可视化

    # 打印每个亚组的统计信息
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        print(f'亚组 {cluster_id} 包含 {len(cluster_indices)} 个病人')
    # 构建不同组数据集
    yazu_time = [[] for _ in range(5)]
    yazu_volumes = [[] for _ in range(5)]
    # 遍历亚组标签
    for i in range(len(cluster_labels)):
        # 获取当前病人所属的亚组标签
        cluster_id = cluster_labels[i]
        # 将时间数据和水肿体积数据添加到对应亚组的列表中
        yazu_time[cluster_id].append(all_times[i * 3])
        yazu_time[cluster_id].append(all_times[i * 3 + 1])
        yazu_time[cluster_id].append(all_times[i * 3 + 2])
        yazu_volumes[cluster_id].append(all_volumes[i * 3])
        yazu_volumes[cluster_id].append(all_volumes[i * 3 + 1])
        yazu_volumes[cluster_id].append(all_volumes[i * 3 + 2])
    return yazu_time,yazu_volumes

yazu_time,yazu_volumes = jvlei_fenzu(feature_sub_sz)

# 分组进行多项式拟合
def fz_i_nh():
    # 先获取list[i]后再转化为数组
    fz_time = yazu_time[0]
    fz_time = np.array(fz_time)
    fz_volumes = yazu_volumes[0]
    fz_volumes = np.array(fz_volumes)
    degree = 4
    coeffs = np.polyfit(fz_time, fz_volumes, degree)

    # 构建多项式函数
    poly_func = np.poly1d(coeffs)

    # 计算拟合曲线上的点
    fit_time = np.linspace(min(fz_time), max(fz_time), 100)
    fit_volume = poly_func(fit_time)

    # 绘制拟合曲线
    plt.scatter(fz_time, fz_volumes, label='Data', color='blue')
    plt.plot(fit_time, fit_volume, label='Fit', color='red')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()



# 1.c 将增长速率从小到大进行排序，筛选出来前十个的治疗方法
sub_volumes_s = np.array(sub_volumes_s)
# 使用argsort对第二列进行排序，并返回排序后的索引
sorted_indices = np.argsort(sub_volumes_s[:, 1])

# 使用sorted_indices来重新排列数组
sorted_array = sub_volumes_s[sorted_indices]
# 获得前二十个增长速率较小的病人的治疗信息
t1_2 = dataframe.iloc[:, 16:23].values
res_t1 = []
for row in sorted_array[:20]:
    #得到表一
    res_t1.append(t1_2[int(row[0])])
res_t1 = np.array(res_t1)
print(res_t1.shape)
# 1.d 血肿增长速率
sub_volumes_x = np.array(sub_volumes_x)
# 获得第几列的值 先转置
sub_volumes_x_oi = list(zip(*sub_volumes_x))[1]
sub_volumes_s_oi = list(zip(*sub_volumes_s))[1]
# 计算皮尔逊相关系数
correlation_coefficient = np.corrcoef(sub_volumes_x_oi, sub_volumes_s_oi)[0, 1]

print("皮尔逊相关系数:", correlation_coefficient)

# 3.1
# 处理数据 100 个患者 (sub001 至 sub100) 个人史、疾病史、发病相关(“表1”字段E至W)及首次影像结果(表 2,表 3 中相关字段)
# 处理性别以及血压 性别女 0 男 1
dataframe['性别'] = dataframe['性别'].apply(lambda x: 1 if x == '男' else 0)
# 提取血压列
dataframe['高压'], dataframe['低压'] = zip(*dataframe['血压'].str.split('/').tolist())
# 将高压和低压转换为数值类型
dataframe['高压'] = dataframe['高压'].astype(int)
dataframe['低压'] = dataframe['低压'].astype(int)
# 删除原始的血压列
dataframe.drop(['血压'], axis=1, inplace=True)
print(dataframe.columns)

