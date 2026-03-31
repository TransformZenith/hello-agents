import numpy as np

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate  # 丢弃概率
        self.mask = None                  # 掩码

    # 训练阶段前向传播
    def forward_train(self, x):
        # 生成掩码：和x形状一样，dropout_rate概率置0，否则1
        self.mask = np.random.rand(*x.shape) > self.dropout_rate
        
        a = self.mask / (1 - self.dropout_rate)  # 缩放因子，保持期望不变
        print("缩放因子：", a)
        # 丢弃 + 缩放
        out = x * a
        return out

    # 推理阶段：不丢弃
    def forward_eval(self, x):
        return x

# ------------------- 测试 -------------------
dropout = Dropout(dropout_rate=0.5)
x = np.array([2, 4, 1, 5])

# 训练
out_train = dropout.forward_train(x)
print("训练输出（带dropout）：", out_train)

# 推理
out_eval = dropout.forward_eval(x)
print("推理输出（无dropout）：", out_eval)