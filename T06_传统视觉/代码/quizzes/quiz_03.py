# -*- coding: utf-8 -*-
"""
课后题 03：识别 quiz_03.png 中的绿色三角形（图中还有红色圆和蓝色方块干扰）
要求：
  1. 读入 ../images/quiz_03.png
  2. HSV 提取绿色 → 形态学去噪 → 找所有轮廓
  3. 用 approxPolyDP 多边形逼近，按顶点数判断形状（3=三角,4=方块,其他=圆）
  4. 只保留形状为"triangle"的轮廓
  5. 在原图上用红色画轮廓边框 + 质心标注
  6. 终端打印："找到 N 个绿色三角形，质心：[(x1,y1), (x2,y2)...]"
提示：
  - 绿色 H 范围大概 35~85
  - approxPolyDP 的 epsilon 常用 0.04 * 周长，可以微调
  - cv2.drawContours(img, [cnt], -1, (0,0,255), 3) 画红色边框
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import cv2
import numpy as np
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "quiz_03.png")


def detect_green_triangles(img):
    """输入 BGR 图，返回所有绿色三角形的质心列表 [(cx1,cy1), (cx2,cy2)...]
    提示：
      - 转 HSV → 绿色 inRange → 形态学去噪 → findContours
      - 用 approxPolyDP 逼近，顶点数==3 才是三角形
      - 遍历每个三角形轮廓，cv2.moments 算质心
    返回: 质心坐标的列表（list）
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    return []  # 写完把 [] 换成你的质心列表
    # ↑↑↑ 你的代码写在这里 ↑↑↑


if __name__ == "__main__":
    img = imread(IMG)
    triangles = detect_green_triangles(img)
    if not triangles:
        print("还没实现 detect_green_triangles，先填代码！")
        raise SystemExit(0)

    # 在原图上用红色画质心标注
    out = img.copy()
    for cx, cy in triangles:
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    print(f"找到 {len(triangles)} 个绿色三角形，质心：{triangles}")
    cv2.imshow("Quiz03", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
