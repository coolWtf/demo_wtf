import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier





#读取数据，八个参数以及结果，将结果A，B替换为0，1.
data = pd.read_excel(r"C:\Users\T.f\Desktop\wtf\万象光形-面试\测试数据.xlsx")
data['结局'] = data['结局'].replace({'A': 0, 'B': 1})
X = data.iloc[:, 1:9]  # 2到9列属性
y = data.iloc[:, 9]   # 结果列

#进行参数筛选，计算皮尔逊相关性系数，选择绝对值前4个参数。
corr_matrix = X.join(y).corr()
correlation_with_result = corr_matrix[y.name].drop(y.name)
print(correlation_with_result)

abs_correlation = correlation_with_result.abs()
sorted_correlation = abs_correlation.sort_values(ascending=False)

top_four_attributes = sorted_correlation.head(4)
selected_columns = top_four_attributes.index
X_selected = X[selected_columns]

#划分数据集 7：3
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# model采用逻辑回归
model = LogisticRegression()
model.fit(X_train, y_train)

# model 采用支持向量机
#model = SVC()
#model.fit(X_train, y_train)

#model 采用决策树
#model = DecisionTreeClassifier()
#model.fit(X_train, y_train)

#model 采用随机森林
#model = RandomForestClassifier()
#model.fit(X_train, y_train)

#model Knn
#model = KNeighborsClassifier()
#model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy}, Recall: {recall}, F1 - score: {f1}")