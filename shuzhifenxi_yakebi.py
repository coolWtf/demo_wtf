import numpy as np

A = np.array([[5.000, 0, 0, -3, -1], [-1, 4, 0, 0, -1], [0, 0, 2, -1, 0], [-1, 0, 0, 4, -2], [0, 0, 0, -1, 2]])
B = np.array([2.000, 3, -1, 0, -1])
x0 = np.array([1.000, 1, 1, 1, 1])
x = np.array([0.000, 0, 0, 0, 0])

times = 0

while True:
    for i in range(5):
        temp = 0
        for j in range(5):
            if i != j:
                temp += x0[j] * A[i][j]
        x[i] = (B[i] - temp) / A[i][i]
    calTemp = max(abs(x - x0))
    times += 1
    if calTemp < 1e-6:
        break
    else:
        x0 = x.copy()
    print(times)
    print(x)

