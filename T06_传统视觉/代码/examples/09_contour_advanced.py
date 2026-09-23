# -*- coding: utf-8 -*-
"""
进阶例程 09：连通域分析 + 外接矩形
========================================
目标：一次拿到图上所有物体的 面积/外接框/质心，并判断是不是"方块"
  - connectedComponentsWithStats 比 findContours 更快，还自带统计信息

怎么运行：
    python examples/09_contour_advanced.py        # 弹窗口，给每个物体画红框并标注

看什么：
    demo_roi.png 里有红圆/蓝圆/绿方块/黄方块/小三角，看每个都被框出来。

学习重点：
    - connectedComponentsWithStats 返回的 4 个东西分别是啥
    - 用"外接矩形宽高比"粗判方块(≈1)还是长条
    - 想更准判圆：用 圆度 = 4π*面积/周长²（下一个例程 10 会讲）
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_roi.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化：把"非白色"物体变白(255)、白背景变黑(0)（阈值240，比240暗的都算物体）
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# 连通域分析：一次性标记所有连在一起的白色区域
# cv2.connectedComponentsWithStats(二值图, 连通方式8, 数据类型)
# 返回4个：num=物体个数, labels=每个像素的编号, stats=统计表, centroids=质心表
num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)

out = img.copy()
for i in range(1, num):  # i=物体编号，0 号是背景，跳过
    area = stats[i, cv2.CC_STAT_AREA]   # 该物体的面积（像素数）
    if area < 300:                      # 面积 < 300 的算小噪点，跳过
        continue
    x = stats[i, cv2.CC_STAT_LEFT]      # 外接框左上角 x
    y = stats[i, cv2.CC_STAT_TOP]       # 外接框左上角 y
    w = stats[i, cv2.CC_STAT_WIDTH]     # 外接框宽
    h = stats[i, cv2.CC_STAT_HEIGHT]    # 外接框高
    cx, cy = int(centroids[i][0]), int(centroids[i][1])  # 质心(中心点)

    # 在图上画外接矩形（红色框）
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 用"宽高比"粗判形状：接近 1 = 方块，否则 = 长方形/椭圆
    ratio = w / h if h != 0 else 0
    label = "square" if 0.9 < ratio < 1.1 else "rect/ellipse"
    cv2.putText(out, f"{label} a={area}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)  # 黑色描边
    cv2.putText(out, f"{label} a={area}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f"检测到 {num-1} 个连通域（含小噪点）")
cv2.imshow("Connected Components", out)
finish()

# 关键点：
# - connectedComponentsWithStats 比 findContours 更快，直接给面积/外接框/质心
# - 外接矩形宽高比可粗判 方块(≈1) vs 长条
# - 想更精确判圆：算 4π*area/perimeter²，接近 1 是圆
