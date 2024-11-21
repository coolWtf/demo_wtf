# 导入相关包及加载数据
# from sklearn.datasets import load_iris # 导入鸢尾花数据集
# import numpy as np
# from sklearn import tree # 导入决策树包
# from sklearn.metrics import accuracy_score # 导入准确率评价指标
# import matplotlib.pylab as plt
# iris = load_iris() # 载入数据集
# # 打乱数据集
# index = [i for i in range(len(iris.data))]
# np.random.shuffle(index)
# iris.data = iris.data[index]
# iris.target = iris.target[index]
#
#
# #构建决策树
# criterion = ['gini','entropy']
# clf = tree.DecisionTreeClassifier(criterion=criterion[0],max_depth=4) #加载决策树模型
#
# #训练
# clf.fit(iris.data[:120], iris.target[:120]) #模型训练，取前五分之四作训练集
#
# #预测
# predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
# predictions[:]
#
# #打印结果
# print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))
# print((iris.target[120:] == predictions).sum() / predictions.size)
# print(iris.target[120:])
# print(predictions)


# from sklearn import tree
# from sklearn.metrics import accuracy_score
# import numpy as np
# import matplotlib.pylab as plt
# from sklearn.datasets import load_breast_cancer
#
# breast_cancer = load_breast_cancer()  # 载入数据集
# print(breast_cancer.feature_names)
# # print(breast_cancer.data.shape,breast_cancer.target.size)
# clf = tree.DecisionTreeClassifier(max_depth=2, criterion='gini')
# clf.fit(breast_cancer.data[:500], breast_cancer.target[:500])
# # 打印决策树图
# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(
#     clf,
#     feature_names=breast_cancer.feature_names,
#     class_names=breast_cancer.target_names,
#     filled=True
# )
#
# predictions = clf.predict(breast_cancer.data[500:])
# print('Accuracy:%s' % accuracy_score(breast_cancer.target[500:], predictions))


# 导入相关包

from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pylab as plt

#加载数据
data = np.loadtxt('3_buy.csv',
delimiter=',',
skiprows= 1)

index = [i for i in range(len(data))]
train_size = int(len(data)*0.7)
np.random.shuffle(index)
data = {'x':data[index][:,:4], 'y':data[index][:,4:]}

#训练
clf_ID3 = tree.DecisionTreeClassifier(criterion = 'entropy',max_depth=4)# 加载决策树
clf_CART = tree.DecisionTreeClassifier(criterion = 'gini',max_depth=4)

clf_ID3.fit(data['x'][:train_size],data['y'][:train_size])
clf_CART.fit(data['x'][:train_size],data['y'][:train_size])

#预测
predictions_ID3 = clf_ID3.predict(data['x'][train_size:])
predictions_CART = clf_CART.predict(data['x'][train_size:])
print('ID3 Accuracy:%s'% accuracy_score(data['y'][train_size:], predictions_ID3))
print('CART Accuracy:%s'% accuracy_score(data['y'][train_size:], predictions_CART))


# 打印结果
buy_fea = ['review','discount','needed','shipping']
buy_class = ['no','yes']
#ID3
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    clf_ID3,
    feature_names = buy_fea,
    class_names = buy_class,
    filled=True
)
#CART
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(
    clf_CART,
    feature_names = buy_fea,
    class_names = buy_class,
    filled=True
)

