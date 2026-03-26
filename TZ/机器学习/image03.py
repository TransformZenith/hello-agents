import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, measure, transform, data
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

# 设置中文显示（防止 Matplotlib 乱码）
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class ImageDataProcessor:
    def __init__(self):
        pass
    
    def resize_image(self, image, target_size):
        """调整图像大小"""
        return transform.resize(image, target_size, anti_aliasing=True)
    
    def normalize_image(self, image):
        """图像归一化"""
        return (image - image.min()) / (image.max() - image.min())
    
    def extract_color_features(self, image):
        """提取颜色特征"""
        # 如果图像是浮点数(0-1)，转为 uint8(0-255) 以便计算直方图
        if image.dtype != np.uint8:
            proc_img = img_as_ubyte(image)
        else:
            proc_img = image

        if len(proc_img.shape) == 3:  # 彩色图像
            features = {}
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = proc_img[:, :, i]
                features[f'{channel}_mean'] = np.mean(channel_data)
                features[f'{channel}_std'] = np.std(channel_data)
                features[f'{channel}_min'] = np.min(channel_data)
                features[f'{channel}_max'] = np.max(channel_data)
            
            # 计算颜色直方图特征
            hist_r, _ = np.histogram(proc_img[:, :, 0], bins=256, range=(0, 256))
            hist_g, _ = np.histogram(proc_img[:, :, 1], bins=256, range=(0, 256))
            hist_b, _ = np.histogram(proc_img[:, :, 2], bins=256, range=(0, 256))
            
            features.update({
                'hist_r_peak': np.argmax(hist_r),
                'hist_g_peak': np.argmax(hist_g),
                'hist_b_peak': np.argmax(hist_b)
            })
            return features
        else:  # 灰度图像
            return {
                'mean': np.mean(proc_img),
                'std': np.std(proc_img),
                'min': np.min(proc_img),
                'max': np.max(proc_img)
            }
    
    def extract_texture_features(self, image):
        """提取纹理特征"""
        if len(image.shape) == 3:
            image = rgb2gray(image)
        
        # 计算边缘特征
        edges = filters.sobel(image)
        edge_density = np.sum(edges > 0.05) / edges.size # 设定微小阈值过滤噪声
        
        # 计算局部二值模式 (LBP)
        lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
        
        return {
            'edge_density': edge_density,
            'lbp_hist': lbp_hist.tolist()
        }
    
    def augment_image(self, image):
        """图像增强"""
        augmented = []
        augmented.append(image) # 原图
        augmented.append(np.fliplr(image)) # 水平翻转
        augmented.append(np.flipud(image)) # 垂直翻转
        augmented.append(np.rot90(image))  # 旋转90度
        
        # 亮度调整
        brightened = np.clip(image * 1.2, 0, 1 if image.dtype != np.uint8 else 255)
        augmented.append(brightened)
        
        return augmented
    
    def visualize_image_channels(self, image):
        """可视化图像通道"""
        if len(image.shape) == 3:
            plt.figure(figsize=(12, 3))
            
            titles = ['原始图像', '红色通道', '绿色通道', '蓝色通道']
            cmaps = [None, 'Reds', 'Greens', 'Blues']
            
            for i in range(4):
                plt.subplot(1, 4, i+1)
                if i == 0:
                    plt.imshow(image)
                else:
                    plt.imshow(image[:, :, i-1], cmap=cmaps[i])
                plt.title(titles[i])
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

# --- 准备测试数据 ---
# 使用 skimage 内置的测试图像
rainbow_img = data.astronaut()        # 彩色图
checkerboard_img = data.checkerboard() # 纹理图
circle_img = data.coins()             # 灰度图（硬币）

# --- 执行示例 ---
image_processor = ImageDataProcessor()

# 1. 提取颜色特征
color_features = image_processor.extract_color_features(rainbow_img)
print("\n[颜色特征提取结果]：")
for key, value in color_features.items():
    if not key.startswith('hist'):
        print(f"{key}: {value:.2f}")

# 2. 提取纹理特征
texture_features = image_processor.extract_texture_features(checkerboard_img)
print("\n[纹理特征提取结果]：")
for key, value in texture_features.items():
    if key != 'lbp_hist':
        print(f"{key}: {value:.4f}")

# 3. 图像增强
augmented_images = image_processor.augment_image(circle_img)
print(f"\n[图像增强]：已生成 {len(augmented_images)} 个变体（翻转、旋转、亮度）")

# 4. 可视化通道
print("\n正在显示通道可视化窗口...")
image_processor.visualize_image_channels(rainbow_img)