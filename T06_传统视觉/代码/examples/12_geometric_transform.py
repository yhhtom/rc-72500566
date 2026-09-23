# -*- coding: utf-8 -*-
"""
进阶例程 12：图像几何变换（缩放/平移/旋转/仿射/翻转）
========================================
目标：掌握改变"图像位置/大小/角度"的各种操作
  - 缩放 resize、平移、旋转、仿射（三点映射）、翻转

怎么运行：
    python examples/12_geometric_transform.py        # 弹 6 个窗口对比

看什么：
    demo_shapes.png 的每种变换效果，重点看"平移/旋转"方向对不对。

学习重点：
    - warpAffine 靠"变换矩阵 M"驱动，先理解 M 是什么
    - 仿射变换用 3 个点对，可做倾斜矫正
    - 应用：训练数据扩增（把图转一转、翻一翻，模型更抗变化）
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")
img = imread(IMG)
h, w = img.shape[:2]  # 取原图的高 h、宽 w

# 1. 缩放：cv2.resize(图, (目标宽, 目标高))
resized = cv2.resize(img, (w // 2, h // 2))  # 目标尺寸=宽高各一半

# 2. 平移：变换矩阵 M 里 tx=向右50, ty=向下30
#    M = [[1,0,tx],[0,1,ty]] 意思是 x方向平移tx、y方向平移ty
M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(img, M_trans, (w, h))  # warpAffine(图, 矩阵M, 输出尺寸)

# 3. 旋转 45 度：cv2.getRotationMatrix2D(旋转中心, 角度, 缩放比例)
M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)  # 绕中心转45°，比例1不变形
rotated = cv2.warpAffine(img, M_rot, (w, h))

# 4. 仿射变换：用 3 个"对应点对"算出变换矩阵（可矫正倾斜/透视）
pts_src = np.float32([[50, 50], [200, 50], [50, 200]])    # 原图的 3 个点
pts_dst = np.float32([[10, 100], [200, 50], [100, 250]])  # 变换后这 3 个点去哪
M_affine = cv2.getAffineTransform(pts_src, pts_dst)       # 由点对算出矩阵
affined = cv2.warpAffine(img, M_affine, (w, h))

# 5. 翻转：cv2.flip(图, 方向)  1=水平, 0=垂直, -1=水平+垂直都翻
flip_h = cv2.flip(img, 1)

cv2.imshow("Original", img)
cv2.imshow("Resize (half)", resized)
cv2.imshow("Translate", translated)
cv2.imshow("Rotate 45deg", rotated)
cv2.imshow("Affine", affined)
cv2.imshow("Flip H", flip_h)
finish(["Original", "Resize (half)", "Translate", "Rotate 45deg", "Affine", "Flip H"])
