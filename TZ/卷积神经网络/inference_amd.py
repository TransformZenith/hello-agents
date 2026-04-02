import cv2
import numpy as np
import torch
import torch_directml  # 必须加
from torchvision import transforms
from PIL import Image
# 确保你的 train.py 就在当前目录下
from train import DigitCNN 

# --- 全局变量设定 ---
# 定义一个统一的窗口名称，防止因为名称不匹配导致回调失效
WINDOW_NAME = "Digit Canvas (L:Draw, R:Clear, Enter:Predict, ESC:Exit)"
canvas = np.zeros((400, 400), dtype="uint8") 
drawing = False 

def draw_event(event, x, y, flags, param):
    global drawing, canvas

    # 左键按下：开始绘画
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(canvas, (x, y), 15, 255, -1)

    # 鼠标移动：如果左键按住，则持续画圆
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 15, 255, -1)

    # 左键抬起：停止绘画
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    # 右键按下：立即清空画布 (修复清除没效果的问题)
    elif event == cv2.EVENT_RBUTTONDOWN:
        canvas.fill(0)
        print("画布已清空")

def predict_from_canvas(model, canvas_data, device):
    # 将画布转为 PIL Image 进行缩放和预处理
    img = Image.fromarray(canvas_data)
    img = img.resize((28, 28), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(img).unsqueeze(0)
    tensor = tensor.to(device)  # 🔥 关键：把输入数据放到和模型一样的设备上

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1).item()
    return prediction

if __name__ == "__main__":
    # 自动判断是否有 GPU
    device = torch_directml.device()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 加载模型
    model = DigitCNN()

    try:
        # --- 修改这里 ---
        # 先以 CPU 方式加载权重文件（最稳妥，不会触发设备对比报错）
        state_dict = torch.load('digit_cnn.pth', map_location='cpu', weights_only=True)
        # 将权重加载进模型
        model.load_state_dict(state_dict)
        # 最后统一将模型搬运到 AMD 显卡 (DirectML)
        model.to(device) 
        # ----------------
        print("模型加载成功！已切换至 AMD GPU (DirectML)")
    except Exception as e:
        print(f"加载失败: {e}")
        exit()


    # 初始化窗口
    cv2.namedWindow(WINDOW_NAME)
    # 绑定回调函数
    cv2.setMouseCallback(WINDOW_NAME, draw_event)

    print("\n交互说明：")
    print("- 鼠标左键：绘画")
    print("- 鼠标右键：清空画布（清除效果在这里）")
    print("- 回车键 (Enter)：识别数字")
    print("- 按 'C' 键：也可以清空画布")
    print("- 按 ESC 键：退出程序")

    while True:
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        # 按 'c' 键手动清除
        if key == ord('c'):
            canvas.fill(0)
            print("画布已清空 (C键)")

        # 按回车键 (13) 识别
        elif key == 13: 
            # 如果画布全是黑的就不识别
            if np.sum(canvas) == 0:
                print("请先在画布上写字")
            else:
                digit = predict_from_canvas(model, canvas,device)
                print(f"--- 识别结果: {digit} ---")

        # 按 ESC (27) 退出
        elif key == 27:
            break

    cv2.destroyAllWindows()