#作   者：Asita
#开发时间：2021/11/10 23:11

# 给定一维数据集：
import time
import numpy as np
import matplotlib.pyplot as plt

datMat = np.matrix([
        [ 1.,],
        [ 2.,],
        [ 3.,],
        [ 3.,],
        [ 5.,],
        [ 6.,],
        [ 7.,],
        [ 8.,],
        [ 9.,],
        [ 10.,],
        [ 11.,],
        [ 12.,],
        ])

classLabels = [3.09, 5.06, 6.03, 9.12, 10.96,6.4,12.8,14.6,15.7,17.8,18.9,10.8]

def average(data):
    sum = 0
    num = len(data)
    for i in range(num):
        sum += data[i]
    return sum / num

# 泛化的式子
def linearRegressionModule(x, a, b):
    """
    线性回归模块
    """
    y_hat = a * x + b
    return y_hat

def lossFunction(y, y_hat):
    '''
    损失函数模块
    '''
    n = len(y)
    sum = 0
    for i in range(n):
        sum += pow((y[i] - y_hat[i]), 2)
    # 损失函数定义为 MSE/2
    L = (sum) / (2 * n)
    return L

# 公式1：一元回归
def Linear_Regression1(dataArr,classLabels):

    Denominator = 0.0 # 分母
    molecular = 0.0 # 分子

    for i in range(len(dataArr)):
        molecular += classLabels[i]* (dataArr[i] - average(dataArr))
        Denominator += (dataArr[i]-average(dataArr))**2

    w=molecular/Denominator
    b=average(classLabels)-w*average(dataArr)
    return w,b


# 公式2：多元回归
def Linear_Regression2(dataArr,classLabels):


    # print(a)
    # 在原始数据矩阵上加一列1(法1):
    # 建立一列为1的矩阵
    # a=np.matrix(np.ones((len(classLabels),1)))
    # datMat=np.c_[dataArr,a]
    # print(datMat)X

    # 法2：
    intercept = np.ones((dataArr.shape[0], 1))  # 初始化截距为1
    X = np.concatenate((intercept, dataArr), axis=1)  # 数组拼接

    classLabels = np.asmatrix(classLabels).reshape(-1, 1)# 把数组转化为1列矩阵

    # print(classLabels)
    # w=np.linalg.inv((dataArr.T*dataArr))*dataArr.T*classLabels
    w = (X.T * X).I * X.T * classLabels # .I 表示矩阵的逆
    w_0=w[0]
    w_1=w[1]
    return w_0,w_1

def showDataSet(dataMat,classLabels,w,b):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_x = []
    data_y = []
    line_y=[]
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    for i in range(len(dataMat)):
        data_x.append(dataMat[i])
        line_y.append(w*datMat[i]+b)

    for j in range(len(classLabels)):
        data_y.append(classLabels[j])

    data_x1 = np.array(data_x)
    data_y1 = np.array(data_y)
    line_y1=np.array(line_y)
    plt.scatter(data_x1, data_y1, c='r', marker='o')

    plt.plot(data_x1[:,0],line_y1[:,0],c='b')
    plt.show()



if __name__ == '__main__':

    start_time = time.time()  # 记录程序开始运行时间
    for i in range(1000):
        w, b = Linear_Regression1(datMat, classLabels)
    end_time = time.time()  # 记录程序结束运行时间
    time1=(end_time-start_time)/1000
    print('first algorithm  Took %f seconds' % time1)

    if b>0:
        print('the line is y=%f*x+%f' % (w, b))
    else:
        print('the line is y=%f*x%f' % (w, b))

    showDataSet(datMat,classLabels,w,b)

    print("---------------------------------------")
    start_time1 = time.time()  # 记录程序开始运行时间
    for i in range(1000):
        w_0,w_1= Linear_Regression2(datMat, classLabels)
    end_time1 = time.time()  # 记录程序结束运行时间
    time2=(end_time1 - start_time1)/1000
    print('second algorithm  Took %f seconds' % time2)
    print(w_0)
    print(w_1)

    # 计算损失：
    y_hat = np.ones(len(datMat))
    for i in range(len(datMat)):
        y_hat[i] = datMat[i] * w + b

    l1 = lossFunction(y_hat, classLabels)

    print('loss is %f'%l1)
