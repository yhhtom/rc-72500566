# -*- coding: utf-8 -*-
"""
课后题 02：识别 quiz_02.png 中的蓝色正方形，画出轮廓 + 输出边长
要求：
  1. 读入 ../images/quiz_02.png
  2. HSV 提取蓝色 → 形态学去噪 → 找最大轮廓
  3. 用 cv2.boundingRect 或 cv2.minAreaRect 算外接矩形的宽和高
  4. 在原图上画外接矩形（绿框） + 质心（红点）
  5. 终端打印："正方形中心：(x, y)  边长约：L 像素"
提示：
  - 蓝色 H 范围大概 100~130
  - 质心公式：(m["m10"]/m["m00"], m["m01"]/m["m00"])
  - 边长：取外接矩形的 width 和 height 平均（正方形应该接近）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import cv2
import numpy as np
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "quiz_02.png")


def detect_blue_square(img):
    """输入 BGR 图，返回蓝色方块的中心 (cx, cy)、边长 side、外接框 (x,y,w,h)
    提示：
      - 转 HSV → 蓝色 inRange → 形态学去噪 → findContours
      - 找最大轮廓，cv2.boundingRect 算外接框，中心 = (x+w//2, y+h//2)
      - 边长 side 取宽高平均 (w+h)//2（正方形宽高接近）
    返回: (cx, cy, side, x, y, w, h)
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    return None  # 写完后把 None 换成 (cx, cy, side, x, y, w, h)
    # ↑↑↑ 你的代码写在这里 ↑↑↑


if __name__ == "__main__":
    img = imread(IMG)
    result = detect_blue_square(img)
    if result is None:
        print("还没实现 detect_blue_square，先填代码！")
        raise SystemExit(0)
    cx, cy, side, x, y, w, h = result

    # 画外接矩形（绿框）+ 质心（红点）
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    print(f"正方形中心：({cx}, {cy})  边长约：{side} 像素")
    cv2.imshow("Quiz02", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
