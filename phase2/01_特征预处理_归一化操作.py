"""
案例：演示特征预处理之归一化操作

特征工程的目的和步骤
    目的：
        利用专业的背景只是和技巧处理数据，用于提升模型的性能
    步骤
        1. 特征提取
        2. 特征预处理（归一化，标准化）
        3. 特征降维
        4. 特征选择
        5. 特征组合
特征预处理之归一化操作：
    目的：
        防止因为量纲问题，导致特征列的方差值较大，影响模型的最终结果
        所以通过公式将把各列的值映射到 [0,1]之间
    公式：
        x' = （当前值-该列最小值）/（该列最大值-该列最小值）
        x'' = x' * （mx-mi） + mi
    弊端：
        容易受到最大值和最小值的影响，所以一般用于处理小数据集
"""
from sklearn.preprocessing import MinMaxScaler  # 归一化对象

x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 参数feature_range 表示生成范围，默认为(0,1)
# transfer = MinMaxScaler(feature_range=(0, 1))
transfer = MinMaxScaler()

x_train_new = transfer.fit_transform(x_train)

print(x_train_new)
