import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

# 读取数据
data = pd.read_excel('华为杯b题数据归一化.xlsx')

# 分割自变量和目标变量
X = data.iloc[0:100, 0:73]
y = data.iloc[0:100, 73]

# 使用SMOTE进行过采样
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 训练模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 预测结果
y_pred = rf.predict(X_test)

# 计算MSE和R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出模型评估结果和目标方程
print('MSE:', mse)
print('R-squared:', r2)
# 输出目标方程
print("目标方程：")
for i, feature in enumerate(X.columns):
    print("{} * {} +".format(rf.feature_importances_[i], feature), end=' ')

# 绘制特征重要性条形图
feature_importance = rf.feature_importances_
feature_names = X.columns.tolist()
sorted_idx = feature_importance.argsort()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [feature_names[i] for i in sorted_idx], fontsize=5)
plt.xlabel('特征重要性')
plt.ylabel('特征名称')
plt.title('')
plt.savefig('', dpi=1000)
# 绘制预测拟合折线图
plt.figure()
plt.plot(np.arange(len(y_test)), y_test, 'go-', label='True Values')
plt.plot(np.arange(len(y_test)), y_pred, 'ro-', label='Predicted Values')
plt.title(f'RandomForestRegression R^2')
plt.legend()
plt.show()


# 继续执行后续代码
# 提取第2到第160行的前73列作为新数据
new_data = data.iloc[0:160, 0:73]

# 使用训练好的模型进行预测
new_data_predictions = rf.predict(new_data)

# 输出新数据的预测结果
print("新数据的预测结果：", new_data_predictions)
result_df = pd.DataFrame({'Predicted Values': new_data_predictions})

# 保存DataFrame到Excel文件
result_df.to_excel('predicted_results.xlsx', index=False)  # 保存为Excel文件，不包含索引列
