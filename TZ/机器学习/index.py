# 完整的机器学习流程示例
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. 数据准备
np.random.seed(42)
n_samples = 200
# 生成特征数据
area = np.random.normal(100, 30, n_samples)  # 面积
bedrooms = np.random.randint(1, 5, n_samples)  # 卧室数
age = np.random.randint(0, 20, n_samples)  # 房龄
location_score = np.random.randint(1, 10, n_samples)  # 地段评分
# 生成标签（房价）- 基于特征的线性组合加噪声
price = (area * 2.5 + bedrooms * 20 + age * -2 + location_score * 15 + 
        np.random.normal(0, 50, n_samples))
# 创建数据框
data = pd.DataFrame({
    '面积': area,
    '卧室数': bedrooms,
    '房龄': age,
    '地段评分': location_score,
    '价格': price
})
print("数据示例：")
print(data.head())
# 2. 划分训练集和测试集
features = ['面积', '卧室数', '房龄', '地段评分']
X = data[features]
y = data['价格']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集大小：{X_train.shape[0]}")
print(f"测试集大小：{X_test.shape[0]}")
# 3. 训练模型
model = LinearRegression()
model.fit(X_train, y_train)
print(f"\n模型参数：")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"截距: {model.intercept_:.2f}")
# 4. 评估模型
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"\n模型评估：")
print(f"训练集 MSE: {train_mse:.2f}, R²: {train_r2:.2f}")
print(f"测试集 MSE: {test_mse:.2f}, R²: {test_r2:.2f}")
# 5. 推理（预测新数据）
new_houses = pd.DataFrame({
    '面积': [85, 120, 65],
    '卧室数': [2, 3, 1],
    '房龄': [3, 1, 8],
    '地段评分': [7, 9, 5]
})
predictions = model.predict(new_houses)
print(f"\n新房价预测：")
for i, price in enumerate(predictions, 1):
    print(f"房子{i}: {price:.2f} 万元")