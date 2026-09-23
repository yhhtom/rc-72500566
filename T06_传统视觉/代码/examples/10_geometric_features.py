# -*- coding: utf-8 -*-
"""
进阶例程 10：几何特征精算（圆度/最小外接矩形/凸包）
========================================
目标：用数值精确描述轮廓，把"圆 vs 方块 vs 细长条"分清楚
重点公式：圆度 = 4π × 面积 ÷ 周长²
   圆≈1.0   正方形≈0.785   三角形≈0.60   细长条≈更小

怎么运行：
    python examples/10_geometric_features.py        # 弹窗口，红框=最小外接矩形，绿=凸包

看什么：
    窗口里每个物体被画了 3 样东西 + 一段文字：
      - 红色框 = 最小外接矩形（带旋转，能看出物体朝向）
      - 绿色框 = 凸包（包住轮廓的最小凸多边形）
      - 蓝色文字 = 形状分类 + 圆度值，如 "circle circ=0.89"
        其中 circ 是圆度（越接近1越像圆）

    终端会打印每个物体的 面积/周长/圆度/分类，对照窗口看。

学习重点：
    - 圆度 = 4π*面积/周长²：圆≈1.0、正方≈0.785、三角≈0.60、细长条≈更小
    - 圆度是"形状分类"最稳的指标，比数顶点更抗噪
    - minAreaRect 带旋转，能看出物体倾斜角度
    - 凸包 = 包住轮廓的最小凸多边形
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_roi.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            # 转灰度
_, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 二值化（物体=白）

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = img.copy()

for cnt in contours:                 # 遍历每个轮廓
    area = cv2.contourArea(cnt)      # 面积（像素数）
    if area < 300:                   # 过滤小噪点
        continue
    perim = cv2.arcLength(cnt, True) # 周长（True=轮廓是闭合的）

    # 圆度 = 4π*面积/周长²：面积相同时圆周长最短 → 圆最接近 1
    circularity = 4 * np.pi * area / (perim * perim) if perim > 0 else 0

    # 最小外接矩形（带旋转角）：能看出物体朝向
    # 返回 ((中心x,中心y),(宽,高),旋转角)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)  # 把矩形转成 4 个顶点坐标
    cv2.drawContours(out, [box], -1, (0, 0, 255), 2)  # 画红色最小外接矩形

    # 凸包：包住轮廓的最小凸多边形，凸缺陷分析用
    hull = cv2.convexHull(cnt)
    cv2.drawContours(out, [hull], -1, (0, 255, 0), 2)  # 画绿色凸包

    # 按圆度分类
    if circularity > 0.85:
        kind = "circle"              # 接近圆
    elif circularity > 0.6:
        kind = "square/rect"         # 接近方形
    else:
        kind = "elongated"           # 细长条
    # 在物体旁标注文字，如 "circle circ=0.89"：
    #   kind=形状分类(circle/square/rect/elongated)，circ=圆度(保留2位小数)
    cv2.putText(out, f"{kind} circ={circularity:.2f}",
                (int(rect[0][0]) - 40, int(rect[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 5)  # 黑色描边
    cv2.putText(out, f"{kind} circ={circularity:.2f}",
                (int(rect[0][0]) - 40, int(rect[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
    print(f"面积={area} 周长={perim:.0f} 圆度={circularity:.2f} -> {kind}")

cv2.imshow("Geometric Features", out)
finish()

# 圆度参考值：圆≈1.0，正方形≈0.785，三角形≈0.60，细长条≈更小
# 这是最常用、最稳的"形状分类"指标
