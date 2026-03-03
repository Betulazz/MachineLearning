"""
案例：
    演示集成学习之Bagging思想随机森林算法代码.
集成学习：
    概述：
        把多个弱学习器组成1个强学习器的过程→集成学习。
    思想：
    Bagging思想：
        1．有放回的随机抽样。
        2. 平权投票。
        3．可以并行执行.
    Boosting思想:
        1．每次训练都会使用全部样本。
        2. 加权投票→预测正确：权重降低，预测错误：权重增加。
        3．只能串行执行.
    Bagging思想代表：
        随机森林算法
随机森林算法：
    1．每个弱学习器都是CART树(必须是二叉树）
    2，有放回的随机抽样，平权投票，并行执行.
"""
import pandas as pd
from pydantic.v1 import DecimalIsNotFiniteError
from sklearn.model_selection import train_test_split  # 切分训练集和测试集
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.ensemble import RandomForestClassifier  # 随机森林算法（分类器）
from sklearn.model_selection import GridSearchCV  # 网格搜索

# 加载数据
df = pd.read_csv('./data/train.csv')
# df.info()

# 数据预处理
# 提取特征和标签
x = df[['Pclass', 'Sex', 'Age']].copy()
y = df['Survived']
# 空值处理，用Age列的平均值填充Age列的空值
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 热编码处理
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
# 特征工程

# 模型训练 预测 评估
estimator1 = DecisionTreeClassifier()
estimator1.fit(x_train, y_train)
y_pred = estimator1.predict(x_test)
print(f'预测值为{y_pred}')
print(f'决策树模型的准确率为：{estimator1.score(x_test, y_test)}')
print('-' * 50)

# 场景2：随机森林算法 采用默认参数
# 1. 创建随机森林对象，演示：多个的决策树（Bagging思想）效果
estimator2 = RandomForestClassifier()  # n_estimators=100 max_depth=None
# 模型训练
estimator2.fit(x_train, y_train)
# 模型预测
y_pred2 = estimator2.predict(x_test)
print(f'预测值为：{y_pred2}')
# 模型评估
print(f'随机森林模型的准确率为：{estimator2.score(x_test, y_test)}')
print('-' * 50)

# 场景3：随机森林算法 采用网格搜素
# 创建随机森林对象，演示：多个的决策树（Bagging思想）效果
estimator3 = RandomForestClassifier()
estimator3.fit(x_train, y_train)
# 参数准备
params = {'n_estimators': [30, 50, 60, 90, 110], 'max_depth': [2, 3, 5, 7]}
# 创建网格搜索对象，结合交叉验证
gs_estimator = GridSearchCV(estimator3, param_grid=params, cv=3)
# 模型训练
gs_estimator.fit(x_train, y_train)
# 模型预测
y_pred3 = gs_estimator.predict(x_test)
print(f'预测值为：{y_pred3}')
# 模型评估
print(f'随机森林模型的准确率为：{gs_estimator.score(x_test, y_test)}')
# 获取最佳参数
print(f'最佳参数：{gs_estimator.best_estimator_}')
