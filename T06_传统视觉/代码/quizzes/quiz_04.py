# -*- coding: utf-8 -*-
"""
课后题 04（综合题）：识别 quiz_04.png 中所有"红色"图案，输出形状 + 质心，按面积从大到小排序
提示：
  - quiz_04.png 里有红圆、红三角、橙圆、蓝色和绿色形状、还撒了噪点
  - 颜色用 HSV 红色（两段 inRange 然后 or）
  - 形状判断：approxPolyDP 顶点数（3=三角,4=方块,其他=圆），但要小心区分橙圆和红圆
  - 排序：sorted(results, key=lambda x: -x["area"])
  - 鼓励自己封装成函数 detect_red_shapes(img) -> List[Dict]
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import cv2
import numpy as np
from util import setup, imread

setup()  # 让控制台中文不乱码

IMG = os.path.join(os.path.dirname(__file__), "..", "images", "quiz_04.png")


def detect_red_shapes(img):
    """输入 BGR 图，输出所有红色图案，按面积从大到小排序
    返回: [{"shape": "circle"/"triangle"/"rectangle", "center": (x,y), "area": int}, ...]
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    # 提示：先转 HSV → inRange 提红色(两段合并) → 形态学去噪 → findContours
    #      遍历轮廓，用 approxPolyDP 判形状，moments 算质心，contourArea 算面积
    #      最后 return results（list 类型）
    return []  # 先返回空列表，写完代码后替换成你的 results
    # ↑↑↑ 你的代码写在这里 ↑↑↑


if __name__ == "__main__":
    img = imread(IMG)
    out = img.copy()
    results = detect_red_shapes(img)
    for r in results:
        cx, cy = r["center"]
        cv2.circle(out, (cx, cy), 5, (0, 0, 0), -1)
        cv2.putText(out, f"{r['shape']} area={r['area']}",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)  # 黑色描边
        cv2.putText(out, f"{r['shape']} area={r['area']}",
                    (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(f"共找到 {len(results)} 个红色图案（按面积从大到小）：")
    for r in results:
        print(" ", r)
    cv2.imshow("Quiz04 Red Detection", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
