#0.导入工具包


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


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
model = LinearRegression()
X =x.reshape(-1,1)   # 将一维数组x转换为二维数组(100行1列)，符合sklearn输入要求
model.fit(X,y)


#3.模型预测
y_predict = model.predict(X)
print(y_predict)


#4.展示
plt.scatter(x,y,label='原始数据')
plt.plot(np.sort(x), y_predict[np.argsort(x)],  # 按x顺序排列
         c='red',
         linewidth=2,
         label='线性回归拟合')
# plt.plot(x,y_predict,label='预测数据',c='red')
plt.legend()
plt.show()
