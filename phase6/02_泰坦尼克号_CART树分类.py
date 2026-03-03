import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 加载数据
data = pd.read_csv('./data/train.csv')
# data.info()
# print(data.head())

# 数据的预处理
# 提取特征和标签

x = data[['Pclass', 'Sex', 'Age']]
y = data['Survived']
# print(x.head(5))
# print(y.head(5))
# age列有缺失，用该列的平均值做填充
# x['Age'].fillna(x['Age'].mean(), inplace=True)    # 报警告
# x['Age'] = x['Age'].fillna(x['Age'].mean())         # 报警告，因为修改源数据
# copy之后再改
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())
# print(x.info())
# 对于 Sex 列 进行one-hot编码
x = pd.get_dummies(x, columns=['Sex'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# 特征工程

# 模型训练
# 参数：max_depth=10 绘制的决策树最多十层
estimator = DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train, y_train)

# 模型预测
y_pred = estimator.predict(x_test)
print(f'预测值为:{y_pred}')

# 模型评估
print(f'分类评估报告：\n{classification_report(y_test, y_pred)}')

# 绘制决策树图
plt.figure(figsize=(30, 20))
# 参1：模型对象 参2：是否用颜色填充 参3：绘制的决策树结构，最多10层
plot_tree(estimator,filled=True,max_depth=10)
plt.savefig('./data/my_titanic.png')
plt.show()

