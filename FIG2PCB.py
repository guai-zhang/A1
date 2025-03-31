import cv2
import numpy as np
from collections import Counter
import pandas as pd

def hex_to_BGR(hex_color):
    # 去掉 '#' 符号并解析RGB值
    hex_color = hex_color.lstrip('#')
    bgr_color = np.array([int(hex_color[m:m + 2], 16) for m in (4, 2, 0)])
    return bgr_color
def save_color_masks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 将图像形状调整为2D数组
    pixels = image_rgb.reshape((-1, 3)) # 统计颜色出现次数
    color_counter = Counter(map(tuple, pixels))
    # 保存每种颜色的遮罩图像
    for color,_ in color_counter.items():
        # 创建遮罩图像，将非目标颜色的像素设置为黑色
        mask=np.all(image_rgb==color,axis=-1)
        mask_image=np.zeros_like(image_rgb)
        mask_image[mask]=color
        color_name=''.join(f'{c:02x}' for c in color)
        outn=Colors.columns[(Colors == "#"+color_name.upper()).any()].values[0]
        cv2.imwrite(f"out/{outn}.png",mask_image)# 保存遮罩图像

# 这里需要自己改!!!运行中会弹出图像简化结果，叉掉继续运行，要修改则结束程序，pytharm默认为Ctrl+f2
# 深红浅红。深白浅白。深黄浅黄。深黑浅黑。深紫浅紫和深铜浅铜..白色丝印层，深绿背阻层# 黑色其实是银色
hexColors={"DBlue":"#4F1C12","Blue":"#72371E","DGreen":"#9D6741","Green":"#D7BA88","Black":"#080202","White":"#F2E5B7"}
Colors=pd.DataFrame([hexColors])
image = cv2.imread("d.png")

colors = {color: hex_to_BGR(value) for color, value in hexColors.items()}
scale_factor = 1  # 设置缩放倍数
target_width = int(image.shape[1] * scale_factor)# 计算目标图像的宽度和高度
target_height = int(image.shape[0] * scale_factor)
image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)# 缩放图像
# 将像素设置为该颜色
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i, j]
        distances = {color: np.linalg.norm(pixel - value) for color, value in colors.items()}
        min_distance_color = min(distances, key=distances.get)
        image[i, j] = colors[min_distance_color]

# cv2.imshow("Processed Image", image)
# cv2.waitKey(0)# 显示处理后的图像
# cv2.destroyAllWindows()

save_color_masks(image) # 拆分图像
print("遮罩图像已保存")

# 正面阻焊层：深蓝+浅蓝+白（即浅绿+深绿后取反）
# 正面铜皮层：浅蓝+黑
# 正面丝印层：白
# 背面阻焊层：深绿是背面有阻焊层，浅绿是背面没阻焊层
# 背面铜皮层：有了更有金属的冷峻感，没有的绿更浅
def hebing(image1path,image2path,outpath):
    image1 = cv2.imread(image1path)
    image2 = cv2.imread(image2path)
    assert image1.shape==image2.shape,"图片尺寸不匹配"
    output_image=np.zeros_like(image1)# 创建一个与输入图像尺寸相同的空白图像
    # 遍历输入图像的每个像素
    for y in range(image1.shape[0]):
        for x in range(image1.shape[1]):
            # 如果任意一个像素不是黑色，则将其复制到输出图像中
            if (not np.all(image1[y, x] == 0)
                    or not np.all(image2[y, x] == 0)):
                output_image[y, x] = np.array([255, 255, 255])
    # 保存输出图像
    cv2.imwrite(outpath, output_image)

hebing(r"out/DBlue.png",r"out/Blue.png",r"out/DBlueBlue.png")
hebing(r"out/Black.png",r"out/DBlueBlue.png",r"out/Zzu.png")
hebing(r"out/Black.png",r"out/Blue.png",r"out/Z.png")

print("合并图像已保存")
