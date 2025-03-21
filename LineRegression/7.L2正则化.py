#0.导入工具包


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

"""
相当于x是data  y是target
"""
#1.准备数据
np.random.seed(22)  # 设置随机种子(保证可重复性)
x = np.random.uniform(-3,3,size=100)   # 生成100个在[-3,3)区间均匀分布的随机数作为特征x
print(x)
y = 0.5*x**2+x+2+np.random.normal(0,1,size=100)    # 根据二次函数关系生成目标值y，并添加正态分布噪声 # 真实关系式：y = 0.5x² + x + 2 + 噪声
print(y)


#2.模型训练
model = Ridge(alpha=0.1)
X =x.reshape(-1,1)   # 将一维数组x转换为二维数组(100行1列)，符合sklearn输入要求
X3 = np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7])
model.fit(X3,y)


#3.模型预测
y_predict = model.predict(X3)
print(y_predict)

#4.模型误差
print(mean_squared_error(y_true=y,y_pred=y_predict))

#4.展示
plt.scatter(x,y,label='原始数据')
#
plt.plot(
    np.sort(x),                  # X轴：排序后的x值（从小到大排列）
    y_predict[np.argsort(x)],    # Y轴：按x的排序索引重新排列预测值
    c='red',                     # 线条颜色设置为红色
    linewidth=2,                 # 线宽为2像素
    label='线性回归拟合'           # 图例标签
)   #为什么需要这样处理？原始问题：当x值无序时，直接绘图会导致折线来回交叉,排序后x和y_predict保持对应关系，形成平滑曲线
# plt.plot(x,y_predict,label='预测数据',c='red')
plt.legend()
plt.show()
