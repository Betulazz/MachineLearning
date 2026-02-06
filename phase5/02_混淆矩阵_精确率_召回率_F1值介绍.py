"""
案例：
    演示混淆矩阵，召回率，F1值
回顾：逻辑回归
    概率：
        属于有监督学习，即：有特征，有标签，且标签是离散的
    评估：
        精确率，召回率，F1值
混淆矩阵：
    概率：
        用来描述真实值和预测值关系的
    图解：
                          预测标签（正例）      预测标签（反例）
        真实标签（正例）      真正例（TP）         伪反例（FN）
        真实标签（反例）      伪正例（FP）         真反例（TN）
    结论：
        1. 模型使用分类少的充当正例
        2. 精确率 = 真正例在预测正例中的占比，tp/(tp+fp)
        3. 召回率 = 真正例在真正例中的占比，tp/(tp+fn)
        4. F1值 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
"""
import pandas
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 需求：已知10个样本，6个恶性肿瘤（正例），4个良性肿瘤（反例）
# 模型A预测结果为：预测对了3个恶性肿瘤，预测对了4个良性肿瘤
# 模型B预测结果为：预测对了6个恶性肿瘤，预测对了1个良性肿瘤
# 请针对于上述的数据集，搭建混淆矩阵，并分别计算模型A，模型B的精确率，召回率，F1值

# 定义变量，记录样本数据
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']

# 定义变量，记录模型A的预测结果
y_pred_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']

# 定义变量，记录模型B的预测结果
y_pred_B = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '恶性', '恶性', '恶性']

# 用标签标记正例反例
label = ['恶性', '良性']
df_label = ['恶性（正例）', '良性（反例）']

# 针对于真实值（y_train）和模型A的预测结果(y_pred_A)，搭建混淆矩阵
cm_A = confusion_matrix(y_train, y_pred_A, labels=label)
print(f'混淆矩阵A：\n{cm_A}')

# 为了测试结果更好看，把上述的混淆矩阵转换为df对象
df_A = pandas.DataFrame(cm_A, index=df_label, columns=df_label)
print(f'混淆矩阵A：\n{df_A}')

# 针对于真实值（y_train）和模型B的预测结果(y_pred_B)，搭建混淆矩阵
cm_B = confusion_matrix(y_train, y_pred_B, labels=label)
print(f'混淆矩阵B：\n{cm_B}')

# 为了测试结果更好看，把上述的混淆矩阵转换为df对象
df_B = pandas.DataFrame(cm_B, index=df_label, columns=df_label)
print(f'混淆矩阵B：\n{df_B}')

# 计算模型A的精确率，召回率，F1值
print(f'模型A精确率：{precision_score(y_train, y_pred_A, pos_label='恶性')}')
print(f'模型A召回率：{recall_score(y_train, y_pred_A, pos_label='恶性')}')
print(f'模型A F1值：{f1_score(y_train, y_pred_A, pos_label='恶性')}')

# 计算模型B的精确率，召回率，F1值
print(f'模型B精确率：{precision_score(y_train, y_pred_B, pos_label='恶性')}')
print(f'模型B召回率：{recall_score(y_train, y_pred_B, pos_label='恶性')}')
print(f'模型B F1值：{f1_score(y_train, y_pred_B ,pos_label='恶性')}')