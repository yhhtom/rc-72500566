#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M07 · 图片推理（三行跑通 YOLO 的第一份代码）
============================================
运行（在 yolo 环境下）：
    cd 07_YOLO推理
    python detect.py                  # 默认检测 assets/bus.jpg
    python detect.py 你的图片.jpg      # 检测指定图片

前置：yolov8n.pt 权重（约 6 MB）。脚本会依次在这些地方找：
    当前目录 → ~/yolov8n.pt → assets/
找不到就提示你去哪拿，不会偷偷联网下载。

对照课件：第七章 · 7.2「三行跑通」/ 7.3「解析结果」
"""
import os
import sys

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from vision_util import setup, find_asset                          # noqa: E402

setup()

WEIGHT_NAMES = ("yolov8n.pt",)


def find_weight():
    """在 当前目录 / 用户主目录 / assets 里找权重。"""
    for name in WEIGHT_NAMES:
        if os.path.exists(name):
            return name
        home = os.path.join(os.path.expanduser("~"), name)
        if os.path.exists(home):
            return home
        p = find_asset(name)
        if p:
            return p
    return None


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else None
    if image is None:
        image = find_asset("bus.jpg") or "bus.jpg"

    weight = find_weight()
    if not weight:
        print("未找到 yolov8n.pt，请先执行（在 yolo 环境里）： cp ~/yolov8n.pt .")
        sys.exit(1)
    if not os.path.exists(image):
        print("找不到图片：%s" % image)
        print("用法：python detect.py 图片路径（不传参数则用 assets/bus.jpg）")
        sys.exit(1)

    model = YOLO(weight)
    results = model(image)          # 推理
    results[0].show()               # 弹窗显示带框结果（按任意键关闭）

    # 顺手把结果存一份，方便写作业时截图
    out = os.path.join(HERE, "out")
    os.makedirs(out, exist_ok=True)
    results[0].save(os.path.join(out, "detect_result.jpg"))
    print("结果已保存到 out/detect_result.jpg")
    print("共检测到 %d 个目标" % len(results[0].boxes))


if __name__ == "__main__":
    main()
