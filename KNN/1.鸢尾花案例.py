

#0.导包
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#1.加载数据集
iris_data = load_iris()
# print(iris_data)  #1.data是150个样本的特征  2.target，每个样本对应一个类别标签，0: 山鸢尾（Iris setosa）1: 变色鸢尾（Iris versicolor）2: 维吉尼亚鸢尾（Iris virginica）
# print(iris_data.data)
# print(iris_data.target)
# print(iris_data.feature_names)



#2.数据展示
iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

iris_df['label'] = iris_data.target
# print(iris_df.tail(10))
"""
画出带有回归线的分类型散点图
用于分析不同鸢尾花种类（标签0/1/2）的花萼长度（sepal length）与花萼宽度（sepal width）之间的线性关系
"""
sns.lmplot(data = iris_df,x = 'sepal length (cm)',y = 'sepal width (cm)',hue = 'label')
# plt.show()


#3.特征工程（预处理--标准化）
#3.1数据集划分
"""
按比例划分成7:3
x_train：data的训练集（7）
x_test：data的测试集（3）
y_train：Target的训练集（7）
y_test：Target的测试集（3）
"""
x_train, x_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target,test_size = 0.3,random_state = 22)
# print(iris_data.data.shape)
# print(x_train.shape)

#3.2特征预处理--标准化
"""
标准化（或归一化）
仅对特征（X）进行处理，不处理标签（Y）
即只对x_train和x_test进行处理
"""
process = StandardScaler()
x_train = process.fit_transform(x_train)    #对x_train进行两个步骤，fit_transform() = fit() + transform()，1.学习参数：计算训练集的统计量 （均值 μ、标准差 σ） 2.转换数据：用这些参数对训练集进行标准化
x_test = process.transform(x_test)   #对x_test进行一个步骤，应用已有参数：直接使用训练集学到的 μ 和 σ 对测试集进行标准化


#4.模型训练
"""
模型训练（Model Training）是指通过训练数据(x_train和y_train)，让模型学习数据中的规律，从而能够对新数据进行预测
fit() 方法的具体操作：
1.数据存储：将 x_train（特征）和 y_train（标签）存入模型对象中。
2.参数初始化：根据 n_neighbors=3 等超参数配置模型行为。
"""
#4.1模型实例化
model = KNeighborsClassifier(n_neighbors = 3)  # 创建K近邻分类器，设定k=3

#4.2调用fit方法
model.fit(x_train, y_train)  # 用训练集数据训练模型

#5.模型预测
x = [[5.1,3.5,1.4,0.2]]
x = process.transform(x)
model_predict = model.predict(x)
print(model_predict)
print(model.predict_proba(x))


#6.模型评估（准确率）
 #6.1 使用预测结果
x_predict = model.predict(x_test)
acc = accuracy_score(y_test,x_predict)
print(acc)
