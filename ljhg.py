import time
import numpy as np

#sigmoid
def sigmoid(z):
    return 1 / (1+np.exp(-z))
#假设函数
def hypothesis(X, theta):
    z=np.dot(X, theta)
    return sigmoid(z)
#损失函数
def computeCost(X, y, theta) :
    m =X.shape[0]
    l = -1*y*np.log(hypothesis(X, theta))-(1-y)*np.log(1-hypothesis(X, theta))
    return np.sum(l)/m
def gradientDescent(X, y, theta, iterations, alpha):
    #取数据条数
    start = time.time()
    m = X.shape[0]
    theta_list = []
    J_list = []
    step_list = []
    X = np.hstack([np.ones((m,1)),X])
    for i in range(iterations):
        for j in range (len(theta)):
            theta[j]=theta[j]- (alpha/m) *np.sum((hypothesis(X, theta)-y)*X[:, j].reshape(-1, 1))
        if(i%10000==0):
            print('第',i,'次迭代，当前损失为:', computeCost(X, y, theta),' theta=', theta)
        J = computeCost(X, y, theta)
        J_list.append(J)
        step_list.append(i + 1)
    end = time.time()
    print("The total time cost is {}".format(end-start))
    return theta_list, J_list, step_list, theta
#目标函数梯度
def grad_matrix(theta, x, y):
    y = np.squeeze(y)
    error = sigmoid(np.inner(theta, x)) - y
    G = np.inner(error, x.T)
    return G
#海森矩阵
def hessian_matrix(theta, x):
    h = []
    for i in range(x.shape[0]):
        h += [sigmoid(np.inner(theta, x[i])) * (1 - sigmoid(np.inner(theta, x[i]))) * x[i]]
    H = np.dot(x.T, h)
    return H

