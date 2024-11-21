import numpy as np

def G_S(a, b, x, g):   # a为系数矩阵  b增广的一列  x迭代初始值  g计算精度
    x = x.astype(float)  #设置x的精度，让x计算中能显示多位小数
    m, n = a.shape
    times = 0          #迭代次数
    if (m < n):
        print("There is a 解空间。")  # 保证方程个数大于未知数个数
    else:
        while True:
            for i in range(n):
                s1 = 0
                tempx = x.copy()    #记录上一次的迭代答案
                for j in range(n):
                    if i != j:
                        s1 += x[j] * a[i][j]
                x[i] = (b[i] - s1) / a[i][i]
                times += 1                  #迭代次数加一
            gap = max(abs(x - tempx))       #与上一次答案模的差
            print(times)
            print(x)

            if gap < g:             #精度满足要求，结束
                break

            elif times > 10000:     #如果迭代超过10000次，结束
                break
                print("10000次迭代仍不收敛")



if __name__ == '__main__':      #当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
    a = np.array([[5, 2, 1], [-1, 4, 2], [2, -3, 10]])
    b = np.array([-12, 20, 3])
    x = np.array([0, 0, 0])     #迭代初始值
    g = 1e-6                      #精度为0.000001
    G_S(a, b, x, g)

