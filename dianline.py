import matplotlib.pyplot as plt

# 定义数据
x = [1, 2, 3, 4, 5, 1]
y = [5, 3, 4, 2, 6, 5]

# 绘制图形
plt.plot(x, y, 'bo-', linewidth=2)
plt.fill(x, y, color='gray', alpha=0.2)

# 设置图形属性
plt.xlim(min(x)-1, max(x)+1)
plt.ylim(min(y)-1, max(y)+1)
plt.grid(True)

# 显示图形
plt.show()

