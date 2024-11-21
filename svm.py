# # 导入需要使用的包
import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

sp = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
data = pd.read_csv("E:\\课程资料\\机器学习\\pima-indians-diabetes.csv", encoding='gb2312')
x, y = np.split(data, (8, ), axis=1) # 数据分组 第9列开始往后为y 代表纵向分割按列分割
x_train, x_test, y_train, y_test=model_selection.train_test_split(x, y, random_state=1, test_size=0.3)

print("pima_data有%d条数据", data.shape[0])
print("训练集有%d条数据", x_train.shape[0])
print("测试集有%d条数据", x_test.shape[0])


# SVM分类器构建
def classifierLinear():
    clf = svm.SVC(C=0.8,  # 误差项惩罚系数
                  kernel='linear',  # 线性
                  decision_function_shape='ovr')  # 决策函数
    return clf


def classifierKernel():
    clf = svm.SVC(C=0.8,  # 误差项惩罚系数
                  kernel='rbf',  # 高斯核 rbf
                  decision_function_shape='ovo')  # 决策函数
    return clf


# 训练模型
def train(clf, x_train, y_train):
    clf.fit(x_train, y_train.values.ravel())  # 训练集特征向量和 训练集目标值


# 2 实例化SVM模型
clf1 = classifierLinear()
clf2 = classifierKernel()

# 3 训练
train(clf1, x_train, y_train)
train(clf2, x_train, y_train)


# 分别打印训练集和测试集的准确率 score(x_train, y_train)表示输出 x_train,y_train在模型上的准确率
def print_accuracy(clf, x_train, y_train, x_test, y_test):
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))


# 4 模型评估
print('-------- eval(Linear) ----------')
print_accuracy(clf1, x_train, y_train, x_test, y_test)
print('-------- eval(Kernel) ----------')
print_accuracy(clf2, x_train, y_train, x_test, y_test)
