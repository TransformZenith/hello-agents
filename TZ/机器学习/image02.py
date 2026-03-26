# 彩色图像示例
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def color_image_example():
    """彩色图像示例"""
    
    print("\n=== 彩色图像示例 ===")
    
    height, width = 100, 100
    
    # 创建 RGB 彩色图像
    # 红色渐变
    red_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    red_gradient[:, :, 0] = np.linspace(0, 255, width)  # 红色通道渐变
    
    # 绿色渐变
    green_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    green_gradient[:, :, 1] = np.linspace(0, 255, width)  # 绿色通道渐变
    
    # 蓝色渐变
    blue_gradient = np.zeros((height, width, 3), dtype=np.uint8)
    blue_gradient[:, :, 2] = np.linspace(0, 255, width)  # 蓝色通道渐变
    
    # 彩虹图案
    rainbow = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        hue = i / width
        # 简化的 HSV 到 RGB 转换
        if hue < 1/3:
            rainbow[:, i] = [255 * (1 - 3*hue), 255 * 3*hue, 0]
        elif hue < 2/3:
            rainbow[:, i] = [0, 255 * (2 - 3*hue), 255 * (3*hue - 1)]
        else:
            rainbow[:, i] = [255 * (3*hue - 2), 0, 255 * (3 - 3*hue)]
    
    # 显示图像
    plt.figure(figsize=(12, 8))
    
    plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    
    plt.subplot(2, 2, 1)
    plt.imshow(red_gradient)
    plt.title('红色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(green_gradient)
    plt.title('绿色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(blue_gradient)
    plt.title('蓝色渐变')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(rainbow)
    plt.title('彩虹图案')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 图像通道信息
    print(f"彩色图像形状：{rainbow.shape}")
    print(f"数据类型：{rainbow.dtype}")
    print(f"像素值范围：{rainbow.min()} - {rainbow.max()}")
    
    return red_gradient, green_gradient, blue_gradient, rainbow

# 运行示例
red_img, green_img, blue_img, rainbow_img = color_image_example()