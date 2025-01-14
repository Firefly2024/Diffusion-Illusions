# 通过手动赋值矩阵坐标 用于生成像素映射所需的uvmap
import numpy as np
from PIL import Image

# 七巧板的UV映射生成 需要注意uint8支持的是0-255的值 因此所有运算值需要mod256
res = np.zeros((256, 256, 3)).astype(np.uint8)
# 大等腰三角形1
for i in range(0, 256):
    for j in range(i, 256 - i):
        res[i, j, 0] = (256 - j) % 256
        res[i, j, 1] = i % 256
# 大等腰三角形2
for i in range(0, 256):
    for j in range(0, 128 - abs(128 - i)):
        res[i, j, 0] = (256 - j) % 256
        res[i, j, 1] = (256 - i) % 256
# 平行四边形
for i in range(192, 256):
    for j in range(256 - i, 384 - i):
        res[i, j, 0] = (256 - i) % 256
        res[i, j, 1] = j % 256
# 中等腰三角形
for i in range(128, 256):
    for j in range(384 - i, 256):
        res[i, j, 0] = (256 - j) % 256
        res[i, j, 1] = i % 256
# 小等腰三角形1
for i in range(128, 192):
    for j in range(256 - i, i):
        res[i, j, 0] = (j + 64) % 256
        res[i, j, 1] = (i + 64) % 256
# 小等腰三角形2
for i in range(0, 128):
    for j in range(192 + abs(64 - i), 256):
        res[i, j, 0] = (i + 64) % 256
        res[i, j, 1] = (j - 64) % 256
# 正方形
for i in range(64, 192):
    for j in range(128 + abs(i - 128), 256 - abs(i - 128)):
        res[i, j, 0] = (256 - i) % 256
        res[i, j, 1] = j % 256

image = Image.fromarray(res, mode="RGB")
image.save("Tangram.png")

# 华容道的UV 映射生成 由于尺寸不为256*256 因此在计算时应进行比例放缩
res = np.zeros((300, 240, 3)).astype(np.uint8)

# 第一个2*1长方块
for i in range(0, 120):
    for j in range(0, 60):
        res[i, j, 0] = (240 - j) / 240 * 255
        res[i, j, 1] = (180 - i) / 300 * 255

# 2*2正方块
for i in range(0, 120):
    for j in range(60, 180):
        res[i, j, 0] = (240 - j) / 240 * 255
        res[i, j, 1] = (300 - i) / 300 * 255

# 第二个2*1长方块
for i in range(0, 120):
    for j in range(180, 240):
        res[i, j, 0] = (240 - j) / 240 * 255
        res[i, j, 1] = (i + 60) / 300 * 255

# 第三个2*1长方块
for i in range(120, 240):
    for j in range(0, 60):
        res[i, j, 0] = (i - 60) / 240 * 255
        res[i, j, 1] = j / 300 * 255

# 第一个1*2长方块
for i in range(120, 180):
    for j in range(60, 180):
        res[i, j, 0] = (360 - i) / 240 * 255
        res[i, j, 1] = (360 - j) / 300 * 255

# 第一个1*1方块
for i in range(180, 240):
    for j in range(60, 120):
        res[i, j, 0] = (j - 60) / 240 * 255
        res[i, j, 1] = (i - 180) / 300 * 255

# 第二个1*1方块
for i in range(180, 240):
    for j in range(120, 180):
        res[i, j, 0] = (360 - j) / 240 * 255
        res[i, j, 1] = (i - 180) / 300 * 255

# 第四个2*1方块
for i in range(120, 240):
    for j in range(180, 240):
        res[i, j, 0] = (i - 60) / 240 * 255
        res[i, j, 1] = (300 - j) / 300 * 255

# 第三个1*1方块
for i in range(240, 300):
    for j in range(0, 60):
        res[i, j, 0] = (i - 180) / 240 * 255
        res[i, j, 1] = (180 - j) / 300 * 255

# 第二个1*2方块
for i in range(240, 300):
    for j in range(60, 180):
        res[i, j, 0] = (300 - i) / 240 * 255
        res[i, j, 1] = (360 - j) / 300 * 255

# 第四个1*1方块
for i in range(240, 300):
    for j in range(180, 240):
        res[i, j, 0] = (360 - j) / 240 * 255
        res[i, j, 1] = (i - 120) / 300 * 255

image = Image.fromarray(res, mode="RGB")
image.show()
image.save("Klotski.png")
