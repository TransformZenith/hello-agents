# 灰度图像示例
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def grayscale_image_example():
    """灰度图像示例"""
    
    print("\n=== 灰度图像示例 ===")
    
    # 创建简单的灰度图像
    # 创建一个 100x100 的灰度图像
    height, width = 100, 100
    
    # 创建渐变图像
    gradient = np.zeros((height, width))
    for i in range(height):
        gradient[i, :] = i  # 垂直渐变
    
    # 创建棋盘图案
    checkerboard = np.zeros((height, width))
    for i in range(0, height, 10):
        for j in range(0, width, 10):
            if (i // 10 + j // 10) % 2 == 0:
                checkerboard[i:i+10, j:j+10] = 255
    
    # 创建圆形图案
    circle = np.zeros((height, width))
    center_x, center_y = width // 2, height // 2
    radius = 30
    for i in range(height):
        for j in range(width):
            if (i - center_y) ** 2 + (j - center_x) ** 2 <= radius ** 2:
                circle[i, j] = 255
    
    # 显示图像
    plt.figure(figsize=(12, 4))
    
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.subplot(1, 3, 1)
    plt.imshow(gradient, cmap='gray')
    plt.title('渐变图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(checkerboard, cmap='gray')
    plt.title('棋盘图案')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(circle, cmap='gray')
    plt.title('圆形图案')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 图像数据信息
    print(f"渐变图像形状：{gradient.shape}")
    print(f"数据类型：{gradient.dtype}")
    print(f"像素值范围：{gradient.min()} - {gradient.max()}")
    
    return gradient, checkerboard, circle

# 运行示例
gradient_img, checkerboard_img, circle_img = grayscale_image_example()