import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

# 忽略警告
import warnings
warnings.filterwarnings('ignore',module='sklearn')

def show_digit(idx):
    df = pd.read_csv('./data/手写数字识别.csv')
    # print(df)  # (42000行 * 785列)

    # 判断传入的索引是否越界
    if idx < 0 or idx > len(df) - 1:
        print('索引越界')
        return

    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    print(f'该图片对应的数字是：{y.iloc[idx]}')
    # print(f'所有的标签的分布情况：{Counter(y)}')

    # print(x.iloc[idx].shape) # (784,) 想办法把(784,)转化为(28,28)
    # print(x.iloc[idx].values)  # 具体的784个像素点数据

    # 把(784,)转化为(28,28)
    x = x.iloc[idx].values.reshape(28, 28)
    # print(x)

    plt.imshow(x, cmap='gray')  # 灰度图
    plt.axis('off')
    plt.show()


def train_model():
    df = pd.read_csv('./data/手写数字识别.csv')
    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    print(f'x的形状:{x.shape}')  # (42000,784)
    print(f'x的形状:{y.shape}')  # (42000,)
    print(f'所有的标签的分布情况：{Counter(y)}')
    # 归一化操作
    x = x / 255

    # 参考y值进行抽取，保持标签的比例（数据均衡）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22, stratify=y)

    estimator = KNeighborsClassifier(n_neighbors=3)

    estimator.fit(x_train, y_train)

    print(f'准确率：{estimator.score(x_test, y_test)}')
    print(f'准确率：{accuracy_score(y_test, estimator.predict(x_test))}')

    # 保存模型
    joblib.dump(estimator, './my_model/手写数字识别.pkl')  # pickle文件：Pandas独有的文件类型
    print('模型保存成功')


def use_model():
    # 1. 加载图片
    x = plt.imread('./data/demo.png')  # 28 * 28 图片
    # 2. 绘制图片
    # plt.imshow(x, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # 3. 加载模型
    estimator = joblib.load('./my_model/手写数字识别.pkl')

    # 4. 模型预测
    # print(x.shape)                  # 28 * 28
    # print(x.reshape(1,784).shape)   # 1 * 784
    print(x.reshape(1, -1).shape)  # 语法糖

    # 4.2 具体的转换动作 imread已经进行了归一化操作
    x = x.reshape(1, -1)

    # 4.3 模型预测
    y_pre = estimator.predict(x)

    print(f'预测值为：{y_pre}')

if __name__ == '__main__':
    # 绘制数字
    # show_digit(23)

    # 训练模型，保存模型
    train_model()

    # 模型预测（使用模型）
    use_model()
