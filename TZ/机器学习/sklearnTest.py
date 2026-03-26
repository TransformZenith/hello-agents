# Scikit-learn 核心功能示例
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def scikit_learn_basics():
#     """演示 Scikit-learn 的核心功能"""
    
#     print("=== Scikit-learn 核心功能示例 ===")
    
#     # 1. 数据生成
#     X, y = make_classification(
#         n_samples=1000, 
#         n_features=20, 
#         n_classes=3, 
#         n_informative=15,
#         random_state=42
#     )
    
#     print(f"数据形状：X={X.shape}, y={y.shape}")
#     print(f"类别分布：{np.bincount(y)}")
    
#     # 2. 数据划分
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"训练集大小：{X_train.shape[0]}")
#     print(f"测试集大小：{X_test.shape[0]}")
    
#     # 3. 数据预处理
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     print("数据标准化完成")
    
#     # 4. 模型训练和比较
#     models = {
#         '逻辑回归': LogisticRegression(random_state=42),
#         '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
#         '支持向量机': SVC(random_state=42)
#     }
    
#     results = {}
    
#     for name, model in models.items():
#         print(f"\n训练 {name}...")
        
#         # 训练模型
#         model.fit(X_train_scaled, y_train)
        
#         # 预测
#         y_pred = model.predict(X_test_scaled)
        
#         # 评估
#         accuracy = accuracy_score(y_test, y_pred)
#         results[name] = accuracy
        
#         print(f"{name} 准确率：{accuracy:.4f}")
#         print(f"分类报告：\n{classification_report(y_test, y_pred)}")
    
#     # 5. 结果比较
#     print("\n=== 模型比较 ===")
#     for name, accuracy in results.items():
#         print(f"{name}: {accuracy:.4f}")
    
#     best_model = max(results, key=results.get)
#     print(f"\n最佳模型：{best_model}")
    
#     return models[best_model]

# # 运行示例
# best_model = scikit_learn_basics()


# 完整的机器学习流程示例
def complete_ml_pipeline():
    """演示完整的机器学习流程"""
    
    print("=== 完整机器学习流程 ===")
    
    # 1. 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"数据集：{iris.DESCR.split('\n')[0]}")
    print(f"特征数量：{len(feature_names)}")
    print(f"类别数量：{len(target_names)}")
    
    # 2. 数据探索
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print("\n数据预览：")
    print(df.head())
    
    print("\n数据统计：")
    print(df.describe())
    
    # 3. 数据可视化
    plt.figure(figsize=(12, 4))
    # 解决中文 + 关闭警告
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.subplot(1, 2, 1)
    for i, target_name in enumerate(target_names):
        plt.scatter(
            df[df['target'] == i]['sepal length (cm)'],
            df[df['target'] == i]['sepal width (cm)'],
            label=target_name
        )
    plt.xlabel('花萼长度')
    plt.ylabel('花萼宽度')
    plt.title('花萼尺寸分布')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i, target_name in enumerate(target_names):
        plt.scatter(
            df[df['target'] == i]['petal length (cm)'],
            df[df['target'] == i]['petal width (cm)'],
            label=target_name
        )
    plt.xlabel('花瓣长度')
    plt.ylabel('花瓣宽度')
    plt.title('花瓣尺寸分布')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. 数据准备
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 5. 模型训练
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. 模型评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n模型准确率：{accuracy:.4f}")
    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 7. 特征重要性
    feature_importance = model.feature_importances_
    feature_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importance
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性：")
    print(feature_df)
    
    # 8. 特征重要性可视化
    plt.figure(figsize=(8, 4))
    plt.bar(feature_df['特征'], feature_df['重要性'])
    plt.title('特征重要性')
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return model, feature_df

# 运行示例
trained_model, feature_importance = complete_ml_pipeline()