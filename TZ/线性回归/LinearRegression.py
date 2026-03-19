import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成一些随机数据
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)


# 可视化数
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Generated Data From Runoob')
# plt.show(block=True)



########################################################################################################


# # 创建线性回归模型
# model = LinearRegression()
# # 拟合模型
# model.fit(x, y)
# # 输出模型的参数
# print(f"斜率 (w): {model.coef_[0][0]}")
# print(f"截距 (b): {model.intercept_[0]}")
# # 预测
# y_pred = model.predict(x)
# # 可视化拟合结果
# plt.scatter(x, y)
# plt.plot(x, y_pred, color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Linear Regression Fit')
# plt.show(block=True)

########################################################################################################


# 手动实现梯度下降法
# 初始化参数
w = 0
b = 0
learning_rate = 0.1
n_iterations = 1000

# 梯度下降
for i in range(n_iterations):
    y_pred = w * x + b
    dw = -(2/len(x)) * np.sum(x * (y - y_pred))
    db = -(2/len(x)) * np.sum(y - y_pred)
    w = w - learning_rate * dw
    b = b - learning_rate * db

# 输出最终参数
print(f"手动实现的斜率 (w): {w}")
print(f"手动实现的截距 (b): {b}")

# 可视化手动实现的拟合结果
y_pred_manual = w * x + b
plt.scatter(x, y)
plt.plot(x, y_pred_manual, color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Manual Gradient Descent Fit')
plt.show()