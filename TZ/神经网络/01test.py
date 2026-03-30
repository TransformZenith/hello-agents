import numpy as np

# 定义激活函数
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

# 初始化一个简单的神经网络
def initialize_network(input_size, hidden_size, output_size):
    """
    初始化网络权重和偏置。
    参数:
        input_size: 输入层神经元数
        hidden_size: 隐藏层神经元数
        output_size: 输出层神经元数
    返回:
        network: 包含各层参数的字典
    """
    np.random.seed(42)  # 设置随机种子，确保每次运行结果一致
    network = {}
    # 初始化 输入层->隐藏层 的参数
    # 权重矩阵形状: (下一层神经元数， 上一层神经元数)
    network['W1'] = np.random.randn(hidden_size, input_size) * 0.01
    network['b1'] = np.zeros((hidden_size, 1))  # 偏置是列向量
    # 初始化 隐藏层->输出层 的参数
    network['W2'] = np.random.randn(output_size, hidden_size) * 0.01
    network['b2'] = np.zeros((output_size, 1))
    return network

# 前向传播函数
def forward_propagation(network, X):
    """
    执行前向传播，计算网络输出。
    参数:
        network: 包含权重和偏置的字典
        X: 输入数据，形状为 (特征数, 样本数)
    返回:
        y_pred: 网络预测输出
        cache: 缓存中间结果（用于后续的反向传播）
    """
    # 获取参数
    W1, b1, W2, b2 = network['W1'], network['b1'], network['W2'], network['b2']

    # 第1层计算: 输入层 -> 隐藏层
    Z1 = np.dot(W1, X) + b1  # 加权和
    A1 = relu(Z1)            # 通过ReLU激活函数

    # 第2层计算: 隐藏层 -> 输出层
    Z2 = np.dot(W2, A1) + b2 # 加权和
    A2 = sigmoid(Z2)         # 通过Sigmoid激活函数（假设是二分类）

    # 缓存中间结果，反向传播时会用到
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

# --- 让我们来运行它！---
# 1. 定义网络结构：2个输入特征，3个隐藏神经元，1个输出（二分类）
input_size = 2
hidden_size = 3
output_size = 1

# 2. 初始化网络
my_network = initialize_network(input_size, hidden_size, output_size)
print("权重 W1 的形状（隐藏层 x 输入层）:", my_network['W1'].shape)
print("偏置 b1 的形状:", my_network['b1'].shape)
print("权重 W2 的形状（输出层 x 隐藏层）:", my_network['W2'].shape)

# 3. 创建一个样本输入数据（2个特征，1个样本）
# X 的列代表样本，行代表特征
X_sample = np.array([[1.5], [-0.5]])  # 形状 (2, 1)
print("\n输入数据 X:", X_sample.T) # .T 是为了转置打印，便于观看

# 4. 执行前向传播
y_pred, cache = forward_propagation(my_network, X_sample)
print("\n神经网络预测输出 (A2):", y_pred)
# 如果输出 > 0.5，我们可以认为是类别1，否则是类别0
predicted_class = 1 if y_pred > 0.5 else 0
print(f"预测类别: {predicted_class}")