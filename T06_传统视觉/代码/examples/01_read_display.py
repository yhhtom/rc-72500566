# -*- coding: utf-8 -*-
"""
例程 01：读图 + 显示 + 转灰度/HSV（传统视觉第 1 课）
========================================
目标：理解"图片在 OpenCV 里 = numpy 三维数组"，分清 BGR/灰度/HSV

怎么运行：
    python examples/01_read_display.py        # 终端打印形状信息，弹出 3 个窗口

看什么：
    终端里 img.shape / img[0,0] 的输出，理解"高宽通道"和像素的 BGR 三数；
    窗口里对比 原图(BGR) / 灰度 / HSV-H 通道 的样子。

学习重点：
    - 图片是 [高, 宽, 通道] 的数组
    - OpenCV 是 BGR 顺序（不是 RGB）
    - cvtColor 三种转换：灰度 / HSV / RGB（下面有示例）
"""
import os
import cv2
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")

img = imread(IMG)
if img is None:
    print("图片读不到：", IMG)
    raise SystemExit(1)

# 1. 看图片的形状
print("img.shape =", img.shape, "  (高 H, 宽 W, 通道 C=3)")
print("img.dtype =", img.dtype)

# 2. 读一个像素（左上角）
print("img[0, 0] =", img[0, 0], "  → BGR 三个数")

# 3. 把 BGR 彩色图 转成 灰度图（单通道，只有亮度）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray.shape =", gray.shape, "  (灰度只有 1 个通道)")

# 4. 把 BGR 彩色图 转成 HSV（H色相/S饱和/V明度，便于按颜色识别）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("hsv.shape  =", hsv.shape, "  (HSV 也是 3 通道)")

# 5. RGB 互换：OpenCV 是 BGR，matplotlib 等要 RGB，用 cvtColor 换
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR → RGB
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # RGB → BGR（可再换回来）

# 6. 怎么检验图片当前是什么格式 / 通道顺序
#    - 看通道数：shape 最后一位 =3 是彩色，=1 是灰度
#    - 分辨 BGR/RGB：读一个"纯红"像素，看三个数里谁是最大
print("img.shape =", img.shape, " → 最后一位 3 表示彩色图")
print("img.dtype =", img.dtype, " → uint8 表示每个数 0~255")
print("img[0,0]  =", img[0, 0], " → 若 [0,0,255] 是 BGR；若 [255,0,0] 是 RGB")

# 7. 显示
cv2.imshow("BGR (original)", img)
cv2.imshow("GRAY", gray)
cv2.imshow("HSV-H channel", hsv[:, :, 0])  # 只看 H 通道
cv2.waitKey(0)
cv2.destroyAllWindows()
