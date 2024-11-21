#作   者：Asita
#开发时间：2021/11/11 16:11

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
datMat = np.matrix([
        [ 0.33,1.8,0],
        [ -0.75,-0.47,1],
        [ -0.94,-3.79,1],
        [ -0.87,-1.9,1],
        [ 0.95,4.34,0],
        [ 0.36,4.27,0],
        [ -0.83,-1.32,1],
        [ 0.28,2.13,0],
        [ -0.9,-1.84,1],
        [ -0.76,-3.47,1],
        [ 0.01,4.0,0],
        ])


def sigmoid(x):
    sigmoid=1.0/(1.0+np.exp(-x))
    return sigmoid

#损失：
# y=classLabels
def loss(h,y):
    loss=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()

    return loss

# 计算梯度: log(p/(1-p))=wx
def gradient(X,h,y):
    gradient=np.dot(X.T,(y-h))
    # print("gradient",gradient)
    return gradient


# 逻辑回归迭代过程：批量梯度下降法：
def Logistic_Regression(X, y, stepsize, max_iters):
    intercept = np.ones((X.shape[0], 1))  # 初始化截距为1
    X = np.concatenate(( X,intercept), axis=1)# 数组拼接
    # print(X)
    m, n = X.shape
    w = np.zeros((n,1))  # 初始化参数为0
    J = pd.Series(np.arange(max_iters, dtype=float))  # 损失函数
    # print("w", w)
    # print("x",X)
    # iter_count = 0  # 当前迭代次数
    # print("y", y)
    # print("y", type(y))

    count=0
    # \sum_i(Sigmoid(wx)-y)*x
    for i in range(max_iters):  # 梯度下降迭代

        z = np.dot(X, w)  # 线性函数
        # print("z\n",z)
        h = sigmoid(z)
        # print("h",type(h))
        g = gradient(X, h, y)  # 计算梯度

        # print("g\n",g)
        w -= stepsize * g # 更新参数

        # l = loss(h, y)  # 计算更新后的损失
        J[i] = -stepsize*np.sum(y.T*np.log(h)+(1-y).T*np.log(1-h)) #计算损失函数值
        count+=1

    return J, w,count  # 返回迭代后的损失和参数

#逻辑回归预测函数：
def Logistic_Regression_predict(test_X,test_label,w):
    intercept=np.ones((test_X.shape[0],1))
    test_X=np.concatenate((intercept,test_X),axis=1)
    predict_z=np.dot(test_X,w)
    predict_label=sigmoid(-predict_z)
    predict_label[predict_label<0.5]=0
    predict_label[predict_label>0.5]=1
    return predict_label





def plotBestFit(J,dataMat,labelMat):
    """
    绘制图形
    :param wb: 回归系数
    :param dataMat: 输入数据集
    :param labelMat: 类别标签
    :return: 无
    """
    import matplotlib.pyplot as plt
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]      #获取数据集的行数
    x1=[];y1=[];x2=[];y2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            x1.append(dataArr[i,0])
            y1.append(dataArr[i,1])
        else:
            x2.append(dataArr[i, 0])
            y2.append(dataArr[i, 1])

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(x1,y1,s=5,c="green")
    ax.scatter(x2,y2,s=5,c="blue")


    J.plot(c="red")
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')  # 绘制label
    plt.show()



if __name__ == '__main__':

    train_x=datMat[0:5,0:2]
    train_y=datMat[0:5,2]
    test_x=datMat[6:11:,0:2]

    test_y=datMat[6:11,2]
    datMat_all=datMat[:,0:2]
    class_all=datMat[:,2]

    # print(train_x)
    # print("----------------------")
    # print(train_y)
    # print("----------------------")

    l,w,count=Logistic_Regression(train_x,train_y,0.05,10)
    # print(w)
    # 图像：
    plotBestFit(l,datMat_all,class_all)
    print("迭代次数：",count)
    predict=Logistic_Regression_predict(test_x,test_y,w)
    print("w:\n",w)
    print("测试集标签test_y\n",test_y)
    print("predict\n",predict)
    accuracy=(predict==test_y).mean()
    print("accuracy:",accuracy)

