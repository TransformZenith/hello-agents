# Hello Aegent

## 1.初识智能体
智能体定义：任何能够通过==传感器Sensors==感知==环境Environment==，并自主通过==执行器Actuators==采取==行动Action==
### 1.1.智能体运行机制
1. 感知Perception：智能体通过传感器（API监听端口、用户输入端口）接受环境的输入信息。即==观察Observation==
2. 思考Thought：接收信息后，智能体进入核心决策阶段。
    - 规划Planning：基于当前观察和内部记忆，更新对任务和环境理解，制定一个行动计划
    - 工具选择Tool Selection：智能体选取工具执行下一步，并确定调用参数
3. 行动Acion：改变环境状态
### 1.2.五分钟实现第一个智能体
>  [见代码](code/chapter1/FirstAgentTest.py)

## 2.智能体发展史
- 符号主义：物理符号+规则推理
- 心智社会：去中心化控制+涌现式计算
- 学习范式：神经网络+强化学习
- LLM智能体:LLM核心+工具调用

## 3.大语言模型基础


## n.笔记
### n.1.线性回归
线性回归 (Linear Regression) 是一种用于预测连续值的最基本的机器学习算法，它假设目标变量 y 和特征变量 x 之间存在线性关系，并试图找到一条最佳拟合直线来描述这种关系。
$$
y = w * x + b
$$

线性回归的目标是找到最佳的 w 和 b，使得预测值 y 与真实值之间的误差最小。常用的误差函数是均方误差 (MSE)：
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - y_{\text{pred}_i})^2
$$
我们的目标是通过调整 w 和 b ，使得 MSE 最小化。
#### n.1.1.求解线性回归
##### n.1.1.1.最小二乘法
最小二乘法是一种常用的求解线性回归的方法，它通过求解以下方程来找到最佳的 ( w ) 和 ( b )。
最小二乘法的目标是最小化残差平方和（RSS），其公式为：
$$
\text{RSS} = \sum_{i=1}^n \left( y_i - (wx_i + b) \right)^2
$$
对 w 和 b分别求导获得
$$
\begin{cases} w\sum x_i^2 + b\sum x_i = \sum x_i y_i \\ w\sum x_i + bn = \sum y_i \end{cases}
$$
用现代表达
$$\begin{bmatrix}\sum x_i^2 & \sum x_i \\\sum x_i & n\end{bmatrix}\begin{bmatrix}w\\b\end{bmatrix}=\begin{bmatrix}\sum x_i y_i \\\sum y_i\end{bmatrix}$$
换一下元获得最终
$$\begin{bmatrix}w\\b\end{bmatrix}=\begin{bmatrix}\sum x_i^2 & \sum x_i \\\sum x_i & n\end{bmatrix}^{-1}\begin{bmatrix}\sum x_i y_i \\\sum y_i\end{bmatrix}$$
##### n.1.1.2.梯度下降法
梯度下降法的目标是最小化损失函数 $J(w,b)$。对于线性回归问题，通常使用均方误差（MSE）作为损失函数：

$$
J(w,b) = \frac{1}{2m}\sum_{i=1}^m (y_i - \hat{y}_i)^2
$$

其中：
- $m$ 是样本数量
- $y_i$ 是实际值
- $\hat{y}_i$ 是预测值，由线性回归模型 $\hat{y}_i = wx_i + b$ 计算得到

梯度是损失函数对参数的偏导数，表示损失函数在参数空间中的变化方向。对于线性回归，梯度计算如下：

$$
\frac{\partial J}{\partial w} = -\frac{1}{m}\sum_{i=1}^m x_i(y_i - \hat{y}_i)
$$

$$
\frac{\partial J}{\partial b} = -\frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)
$$

参数更新规则
梯度下降法通过以下规则更新参数 $w$ 和 $b$：

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

其中：
- $\alpha$ 是学习率（learning rate），控制每次更新的步长

梯度下降法的步骤
1. 初始化参数：初始化 $w$ 和 $b$ 的值（通常设为 0 或随机值）
2. 计算损失函数：计算当前参数下的损失函数值 $J(w,b)$
3. 计算梯度：计算损失函数对 $w$ 和 $b$ 的偏导数
4. 更新参数：根据梯度更新 $w$ 和 $b$
5. 重复迭代：重复步骤 2 到 4，直到损失函数收敛或达到最大迭代次数

#### n.1.2.使用 Python 实现线性回归
> [见代码](TZ\线性回归\LinearRegression.py)

### n.2.机器学习
#### n.2.1.机器学习类型
1. 监督学习（Supervised Learning）
    - 定义： 监督学习是指使用带标签的数据进行训练，模型通过学习输入数据与标签之间的关系，来做出预测或分类。
    - 应用： 分类（如垃圾邮件识别）、回归（如房价预测）。
    - 例子： 线性回归、决策树、支持向量机（SVM）。
2. 无监督学习（Unsupervised Learning）
    - 定义： 无监督学习使用没有标签的数据，模型试图在数据中发现潜在的结构或模式。
    - 应用： 聚类（如客户分群）、降维（如数据可视化）。
    - 例子： K-means 聚类、主成分分析（PCA）。
3. 强化学习（Reinforcement Learning）
    - 定义： 强化学习通过与环境互动，智能体在试错中学习最佳策略，以最大化长期回报。每次行动后，系统会收到奖励或惩罚，来指导行为的改进。
    - 应用： 游戏AI（如AlphaGo）、自动驾驶、机器人控制。
    - 例子： Q-learning、深度Q网络（DQN）。
#### n.2.2.机器学习项目生命周期
1. 收集数据：准备包含特征和标签的数据。
2. 选择模型：根据任务选择合适的机器学习算法。
3. 训练模型：让模型通过数据学习模式，最小化误差。
4. 评估与验证：通过测试集评估模型性能，并进行优化。
5. 部署模型：将训练好的模型应用到实际场景中进行预测。
6. 持续改进：随着新数据的产生，模型需要定期更新和优化。

### n.3.神经网络
#### n.3.1.神经网络的基本组成单元：神经元
