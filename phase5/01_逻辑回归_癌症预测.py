"""
案例：
    演示逻辑回归模型实现癌症预测
逻辑回归模型介绍：
    概述：
        属于有监督学习，即：有特征，有标签，且表示是离散的
    原理：
        把线性回归后的预测值 -> 通过Sigmod激活函数，映射到(0,1)概率->基于自定义的阈值，结合概率来分类
    损失函数：
        极大似然估计函数的负数形式
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()

# 数据预处理
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True, axis=0)  # axis=0 表示行，删除含有空值的行
# data.info()

# 特征工程
# 特征提取值提取特征和标签
x = data.iloc[:, 1:-1]  # ：表示所有行，1：-1表示从第1列到最后一列
# y = data.iloc[:, -1]
# y = data['Class']
y = data.Class
print(x[:5])
print(y[:5])
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

# 模型训练
estimator=LogisticRegression()
estimator.fit(x_train,y_train)

# 模型预测
y_pre=estimator.predict(x_test)
print(f'predict:{y_pre}')

# 模型评估
# 正确率
print(f'预测前评估，正确率：{estimator.score(x_test,y_test)}')
print(f'预测后评估，正确率：{accuracy_score(y_test,y_pre)}')

# 逻辑回归模型能用准确率来评估吗？
# 可以，但是不精准，因为逻辑回归模型主要用于二分类，即：A类还是B类，不能说97%的A类 3%的B类
# 所以要通过混淆矩阵来评测，即：精确率，召回率，F1值（F1-Score），ROC曲线，AUC值