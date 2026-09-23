# -*- coding: utf-8 -*-
"""
例程 02：HSV 颜色阈值提取（传统视觉三板斧·第 1 板）
========================================
目标：把指定颜色（比如红、蓝）从图里"抠出来"，得到黑白掩膜 mask
关键：HSV 阈值调参（H=色相 决定颜色种类，S=饱和 过滤灰，V=明度 过滤暗）

怎么运行：
    python examples/02_hsv_threshold.py        # 弹 5 个窗口：原图/红mask/蓝mask/抠出来的红/蓝

看什么：
    红色掩膜里白色 = 图上所有红色像素；"只保留红色"窗口能看到抠出来的效果。

学习重点：
    - cv2.inRange 的三个参数（图, 下限[H,S,V], 上限[H,S,V]）
    - 红色 H 是环形要分两段再按位或（色环跨 0 和 179）
    - 这就是"颜色阈值"，后面所有识别都靠它
"""
import os
import cv2
import numpy as np
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")

img = imread(IMG)
# 把 BGR 彩色图 转成 HSV（H色相/S饱和/V明度），才能按颜色阈值找东西
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 红色的 HSV 范围（注意红色 H 在 HSV 是个环，要用两段）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

# 蓝色的 HSV 范围（H 100-130 左右）
lower_blue = np.array([100, 120, 80])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 用 mask 把原图里对应颜色抠出来
red_out = cv2.bitwise_and(img, img, mask=red_mask)
blue_out = cv2.bitwise_and(img, img, mask=blue_mask)

cv2.imshow("Original", img)
cv2.imshow("Red mask", red_mask)
cv2.imshow("Blue mask", blue_mask)
cv2.imshow("Red only", red_out)
cv2.imshow("Blue only", blue_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 小测 1：换张图试试
# 用 quiz_01.png 测一下，提示：这图里只有一个红色圆
