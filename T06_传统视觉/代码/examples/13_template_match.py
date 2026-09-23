# -*- coding: utf-8 -*-
"""
进阶例程 13：模板匹配 matchTemplate
========================================
目标：拿一个小"模板图"，在大图里滑动找"最像"的位置
  - 适合：固定图案查找、UI 图标定位、简单字符识别
  - 局限：对旋转/缩放很敏感，模板要和目标基本一致才准

怎么运行：
    python examples/13_template_match.py        # 弹窗口，红框标出匹配到的位置

看什么：
    程序先裁剪 demo_shapes.png 左上角的红圆当模板，再全图找它，
    应该能在红圆原本的位置框出来（相似度 1.0）。

学习重点：
    - matchTemplate 输出"相似度图" res，再用 minMaxLoc 找最高点
    - TM_CCOEFF_NORMED = 归一化相关，值越接近 1 越像
    - 多目标匹配：一次只返回最像的，要多目标得自己写循环+抑制
"""
import os
import cv2
import numpy as np
from util import setup, imread, finish

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "demo_shapes.png")
img = imread(IMG)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 从原图裁剪一个"模板"（手工指定矩形区域）
# 语法 图[起始行:结束行, 起始列:结束列] = 裁剪
# 红圆在 (150,150) 附近、半径60，所以裁 [90:210, 90:210]
template = gray[90:210, 90:210]
th, tw = template.shape  # 模板的高 th、宽 tw

# 模板匹配：把模板在大图上滑动，算每个位置"像不像"
# cv2.matchTemplate(大图, 模板, 匹配方法TM_CCOEFF_NORMED) → 相似度图 res
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
# 在相似度图里找最大值位置：minMaxLoc 返回(最小,最大,最小位置,最大位置)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
print(f"最佳匹配：位置 {max_loc}，相似度 {max_val:.2f}（1.0 最像）")

# 在匹配位置画红框
out = img.copy()
cv2.rectangle(out, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 0, 255), 3)
cv2.imshow("Template (red circle)", template)  # 显示模板
cv2.imshow("Matched Location", out)            # 显示匹配结果
finish()

# 局限：
# - 模板不能旋转/缩放，否则匹配度急剧下降
# - 多目标：用 cv2.matchTemplate 一次只给最像的一个，多目标要写循环+非极大抑制
