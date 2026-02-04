"""
线性回归介绍（Linear Regressor）：
    目的：
        用线性公式来描述多个自变量（特征）和 1个因变量（标签）之间关系的，对其关系进行建模，基于特征预测标签
        线性回归属于：有监督学习，即：有特征，有标签，且标签是连续的
    分类：
        一元线性回归：1个特征列 + 1个标签列
        多元线性回归：多个特征列 + 1个标签列
    公式：
        一元线性回归：
            y = wx + b
                w : weight 权重
                b ：bias 偏置
        多元线性回归：
            y = w1x1 + w2x2 + w3x3 + ... + wnxn +b
                = w的转置 * x +b

    误差 = 预测值 - 真实值
    损失函数（Loss Function）
        用于描述每个样本点和其预测值之间关系的，让损失函数最小，就是让误差和小，线性回归效率，评估就越高
    问题：如何让损失函数最小？
        思路 1：正规方程法，
        思路 2：梯度下降法
    损失函数分类：
        最小二乘：每个样本点误差的平方和
        MSE(Mean Square Error,均方误差)：每个样本点误差的平方和/样本个数
        RMSE(Root Mean Square Error,均方根误差)：均方误差开平方根
        MAE(Mean Absolute Error,均绝对误差)：每个样本点误差的绝对值/样本个数
"""
from sklearn.linear_model import LinearRegression

# 案例：演示线性回归API入门

# 1. 准备数据
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]

# 2. 数据预处理
# 3. 特征工程（特征提取，特征预处理）

# 4. 创建模型对象
estimator = LinearRegression()

estimator.fit(x_train,y_train)

print(f'weight:{estimator.coef_}')
print(f'bias:{estimator.intercept_}')

y_pre=estimator.predict(x_test)
print(f'predict:{y_pre}')