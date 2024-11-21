#  # 导入相关包
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
# from sklearn import metrics
# # 一、加载数据集：
# iris = datasets.load_iris()  # 导入鸢尾花数据集
# # 标签重置
# iris.target[iris.target == 2] = 3
# iris.target[iris.target == 1] = 2
# iris.target[iris.target == 0] = 1
# iris.target[iris.target == 3] = 0
# # 数据拆分
# iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris.data, iris.target, test_size=0.2,
#                                                                         random_state=23)
# # 这里是从sklearn中直接导入的数据集。
# # 二、配置模型
# Kmeans = KMeans(n_clusters=3, random_state=42)  # K-Means算法模型，3类标签
# # 三、训练模型
# kmeans_fit = Kmeans.fit(iris_X_train)  # 模型训练
# # 四、模型预测
# y_predict = Kmeans.predict(iris_X_train)
# # 五、模型评估
# score = metrics.accuracy_score(iris_y_train, Kmeans.predict(iris_X_train))
# print('准确率:{0:f}'.format(score))  # 显示准确率
# # 六、结果可视化
# # 因为图形只有两个维度X和Y，所以该程序只有将特征值的第一个和第二个分别当成表格中X和Y的位置，第三个和第四个特征值虽然在计算时会使用，但显示图片的时候就不使用。
# x1 = iris_X_train[:, 0]  # 鸢尾花花萼长度
# y1 = iris_X_train[:, 1]  # 鸢尾花花萼宽度
# plt.scatter(x1, y1, c=y_predict, cmap='viridis')  # 画每一条的位置
# centers = Kmeans.cluster_centers_  # 每个分类的中心点
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);  # 中心点
# plt.show()  # 显示图像

# #导入相关包
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# # 一、数据获取
# # 1.1.导入数据
# df = pd.read_csv('China_cities.csv',encoding='GB18030') # 导入数据集
# print(df.shape)    # 输出数据维度
# print(df.head())  # 展示前5行数据
# # 二、数据预处理
# # 2.1.提取经纬度
# x = df.drop('省级行政区', axis=1) # 删除 省级行政区 这一列
# x = x.drop("城市", axis=1) # 删除 城市 这一列
# x_np = np.array(x)        # 将x转化为numpy数组
# # 三、模型构建与训练
# #3.1.构造K-Means聚类器
# n_clusters = 7                # 类簇的数量
# estimator = KMeans(n_clusters)  # 构建聚类器
# # 3.2.训练K-Means聚类器
# estimator.fit(x)
# # 四、数据可视化
# markers = ['*', 'v', '+', '^', 's', 'x', 'o']      # 标记样式列表
# colors = ['r', 'g', 'm', 'c', 'y', 'b', 'orange']  # 标记颜色列表
# labels = estimator.labels_      # 获取聚类标签
# plt.figure(figsize=(9, 6))
# plt.title("china cities", fontsize=25)
# plt.xlabel('East Longitude', fontsize=18)
# plt.ylabel('North Longitude', fontsize=18)
# for i in range(n_clusters):
#     members = labels == i      # members是一个布尔型数组
#     plt.scatter(
#         x_np[members, 1],      # 城市经度数组
#         x_np[members, 0],      # 城市纬度数组
#         marker = markers[i],   # 标记样式
#         c = colors[i]          # 标记颜色
#     )   # 绘制散点图
# plt.grid()
# plt.show()
#导入相关包
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 一、数据获取
# ------ 1.1.导入数据 ------
df = pd.read_csv('beverage.csv',encoding='GB18030') # 导入数据集
print(df.shape)    # 输出数据维度
print(df.head())  # 展示前5行数据
# 二、数据可视化
def showOrgData(dataMat):
    df=np.array(dataMat)
    print(type(df))
    plt.scatter(df[:, 0], df[:, 1],color='m', marker='o', label='Org_data')
    plt.xlabel('juice')
    plt.ylabel('sweet')
    plt.legend(loc=2) # 把说明放在左上角，具体请参考官方文档
    plt.show()

#四、训练不同k值下的kmeans、画出不同k值下的结果散点图、记录不同k值下聚类的CH评价指标的结果
score_all=[]
list1=range(2,8)
for i in range(2,8):
    estimator = KMeans(n_clusters=i)
    estimator.fit(df)
    y_pred = estimator.fit_predict(df)
    plt.scatter(df[:, 0], df[:, 1], c=y_pred,label=i)
    plt.legend(loc=2)  # 把说明放在左上角，具体请参考官方文档
    plt.xlabel('juice')
    plt.ylabel('sweet')
    # 重要属性cluster_centers_，查看质心
    centroid = estimator.cluster_centers_
    print("k=%d:" % i)
    print("centroid:\n",centroid)
    # 各类簇中心点的可视化
    plt.scatter(
        centroid[:, 0],
        centroid[:, 1],
        marker="x",
        c="black",
        s=48
    )
    score = metrics.calinski_harabasz_score(df, y_pred)
    score_all.append(score)
    print("score=",score)
    print('------------------------------')
    plt.show()
#七、画出不同k值对应的聚类效果（折线）
plt.plot(list1,score_all)
plt.xlabel('k')
plt.ylabel('CH')
plt.show()




