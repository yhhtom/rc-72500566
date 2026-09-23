# -*- coding: utf-8 -*-
"""
课后题 01：识别 quiz_01.png 中的红色圆形，并输出质心坐标
要求：
  1. 读入 ../images/quiz_01.png
  2. 用 HSV 颜色阈值提取红色区域
  3. 用形态学去噪（开运算+闭运算）
  4. 找最大轮廓，算质心
  5. 在原图上画一个十字标注质心，按 q 退出
  6. 终端打印："质心坐标：(x, y)  面积: S"
提示：
  - 例程 03 已经把整套流程写好了，照着改
  - 红色 H 在 HSV 是个环，要两段 inRange 然后按位或
  - 用 cv2.moments 算质心，m["m00"] 不能等于 0
"""
import os
import cv2
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "quiz_01.png")





if __name__ == "__main__":
    img = imread(IMG)
    result = detect_red_circle(img)
    if result is None:
        print("还没实现 detect_red_circle，先填代码！")
        raise SystemExit(0)
    cx, cy, area = result

    # 在原图上画十字标注质心
    out = img.copy()
    cv2.line(out, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
    cv2.line(out, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
    print(f"质心坐标：({cx}, {cy})  面积: {area}")
    cv2.imshow("Quiz01", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
