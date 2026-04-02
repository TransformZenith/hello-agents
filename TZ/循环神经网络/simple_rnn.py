import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, seq_length):
        # 超参数
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        
        # 模型参数初始化 (Xavier 初始化思想)
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def lossFun(self, inputs, targets, hprev):
        """
        inputs, targets 都是整数列表
        hprev 是初始隐藏状态 (hidden_size, 1)
        Returns: loss, 梯度, 最后的隐藏状态
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # 前向传播
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1)) 
            xs[t][inputs[t]] = 1 # One-hot 编码
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # Softmax
            loss += -np.log(ps[t][targets[t], 0]) # 交叉熵损失

        # 反向传播 (BPTT)
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # Softmax + CrossEntropy 的导数
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # 损失对 h 的梯度
            dhraw = (1 - hs[t] * hs[t]) * dh # tanh 的导数
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # 防止梯度爆炸的裁剪
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(self, h, seed_ix, n):
        """生成 n 个字符的预测序列"""
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes
    





# --- 新增数据预处理部分 ---
# 1. 准备数据（你可以换成读取文件：data = open('input.txt', 'r').read()）
data = "hello world, this is a simple rnn text generation example built with numpy."

# 2. 提取唯一字符并创建映射表
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print(f'数据量: {data_size} 字符, 唯一字符数: {vocab_size}')

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# 超参数
hidden_size = 100 
seq_length = 25 
learning_rate = 1e-1

# 初始化模型
rnn = SimpleRNN(vocab_size, hidden_size, seq_length)

# 优化器记忆变量 (用于 Adagrad)
mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

n, p = 0, 0
hprev = np.zeros((hidden_size, 1)) # 初始隐藏状态

while n < 10001:
    # 1. 准备输入序列 (当前偏移量 p 开始的 seq_length 个字符)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) # 重置隐藏状态
        p = 0 

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # 2. 核心：前向+反向传播计算梯度
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = rnn.lossFun(inputs, targets, hprev)

    # 3. 每 1000 次迭代进行一次推理演示
    if n % 1000 == 0:
        print(f'\n---- 迭代次数: {n}, 损失: {loss:.4f} ----')
        # 推理：给定当前 hprev 和当前第一个字符，生成 50 个字符
        sample_ix = rnn.sample(hprev, inputs[0], 50)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f'生成预测: \n "{txt}"')
        print('-' * 40)

    # 4. 执行 Adagrad 参数更新
    for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # 这里的 1e-8 防止除以0

    p += seq_length # 移动指针
    n += 1
