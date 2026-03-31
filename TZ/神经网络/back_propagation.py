import numpy as np

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

# 初始化网络参数（通常随机初始化，这里为演示指定值）
w1, b1 = 2.0, -1.0  # 隐藏层参数
w2, b2 = 1.5, 0.5   # 输出层参数

def forward_pass(x):
    """执行一次前向传播"""
    # 隐藏层计算
    z1 = w1 * x + b1
    a1 = sigmoid(z1)  # 应用激活函数
    
    # 输出层计算
    y_pred = w2 * a1 + b2  # 线性输出
    
    # 返回中间结果和最终预测，便于后续理解
    return {'z1': z1, 'a1': a1, 'y_pred': y_pred}

# 假设房屋面积为 3（单位：百平方米）
x_input = 3.0
result = forward_pass(x_input)

print(f"输入 x = {x_input}")
print(f"隐藏层线性输出 z1 = w1*x + b1 = {result['z1']:.4f}")
print(f"隐藏层激活输出 a1 = sigmoid(z1) = {result['a1']:.4f}")
print(f"最终预测房价 y_pred = w2*a1 + b2 = {result['y_pred']:.4f}")



# 接续前向传播的代码和结果
y_true = 2.5
y_pred = result['y_pred']
a1 = result['a1']
z1 = result['z1']
x = x_input

print(f"真实值 y_true = {y_true}")
print(f"预测值 y_pred = {y_pred:.4f}")
print(f"初始损失 Loss = {(y_true - y_pred)**2:.4f}")
print("\n--- 开始反向传播计算梯度 ---")

# 1. 计算损失对y_pred的梯度
dL_dy_pred = -2 * (y_true - y_pred)
print(f"梯度 ∂L/∂y_pred = -2*(y_true - y_pred) = {dL_dy_pred:.4f}")

# 2. 计算输出层参数 w2, b2 的梯度
dL_dw2 = dL_dy_pred * a1
dL_db2 = dL_dy_pred * 1
print(f"梯度 ∂L/∂w2 = (∂L/∂y_pred) * a1 = {dL_dw2:.4f}")
print(f"梯度 ∂L/∂b2 = (∂L/∂y_pred) * 1 = {dL_db2:.4f}")

# 3. 计算损失对隐藏层输出a1的梯度
dL_da1 = dL_dy_pred * w2
print(f"梯度 ∂L/∂a1 = (∂L/∂y_pred) * w2 = {dL_da1:.4f}")

# 4. 计算Sigmoid函数的导数在z1处的值
def sigmoid_derivative(x):
    """Sigmoid函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

sigma_prime_z1 = sigmoid_derivative(z1)
print(f"Sigmoid导数 σ'(z1) = σ(z1)*(1-σ(z1)) = {sigma_prime_z1:.4f}")

# 5. 计算隐藏层参数 w1, b1 的梯度
dL_dw1 = dL_da1 * sigma_prime_z1 * x
dL_db1 = dL_da1 * sigma_prime_z1 * 1
print(f"梯度 ∂L/∂w1 = (∂L/∂a1) * σ'(z1) * x = {dL_dw1:.4f}")
print(f"梯度 ∂L/∂b1 = (∂L/∂a1) * σ'(z1) * 1 = {dL_db1:.4f}")


# 学习率
learning_rate = 0.1

# 更新参数
w1_new = w1 - learning_rate * dL_dw1
b1_new = b1 - learning_rate * dL_db1
w2_new = w2 - learning_rate * dL_dw2
b2_new = b2 - learning_rate * dL_db2

print("--- 更新后的参数 ---")
print(f"w1: {w1:.4f} -> {w1_new:.4f}")
print(f"b1: {b1:.4f} -> {b1_new:.4f}")
print(f"w2: {w2:.4f} -> {w2_new:.4f}")
print(f"b2: {b2:.4f} -> {b2_new:.4f}")

# 用新参数做一次前向传播，验证损失是否减小
def forward_pass_with_params(x, w1, b1, w2, b2):
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    y_pred = w2 * a1 + b2
    return y_pred

y_pred_new = forward_pass_with_params(x_input, w1_new, b1_new, w2_new, b2_new)
loss_new = (y_true - y_pred_new)**2
print(f"\n用新参数预测: y_pred_new = {y_pred_new:.4f}")
print(f"更新后的损失 New Loss = {loss_new:.4f}")
print(f"损失变化: {loss_new - (y_true-y_pred)**2:.4f} (负值表示损失减小)")