# -*- coding: utf-8 -*-
"""
T08 YOLO 推理 —— 作业骨架（学生版）

目标：学会加载模型、跑推理、把结果解析成"能用的数据"。
训练是 M08 的事，这里只用官方预训练权重 yolov8n.pt（80 类通用目标）。

自检（本教案不含自动校验脚本）：跑一遍 assets/bus.jpg，看能不能检出 person 与 bus
"""
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------- 1
def load_model(weight_path: str):
    """加载 YOLO 模型并返回模型对象。

    from ultralytics import YOLO
    return YOLO(weight_path)
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: from ultralytics import YOLO")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 2
def detect_objects(model, img, conf: float = 0.25) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """对一张 BGR 图跑推理，返回检测列表。

    返回格式: [(类别名, 置信度, (x1, y1, x2, y2)), ...]
    按置信度从高到低排序。

    提示：
        results = model.predict(source=img, conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes
        - boxes.cls  是类别 id 的 tensor
        - boxes.conf 是置信度
        - boxes.xyxy 是左上角/右下角坐标
        - r.names    是 {id: 类别名} 的字典
    """
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: model.predict -> 解析 boxes -> 排序")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 3
def count_by_class(dets: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> Dict[str, int]:
    """按类别名计数：{"person": 4, "bus": 1}"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 4
def max_conf_target(dets, name: str):
    """返回指定类别中置信度最高的那一条；没有该类返回 None。"""
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO")
    # ↑↑↑ 你的代码写在这里 ↑↑↑


# ---------------------------------------------------------------- 5
def draw_dets(img, dets, copy_img: bool = True) -> "np.ndarray":
    """把检测结果画到图上：绿框 + 顶部文字 "name conf"。

    返回画好的图；copy_img=True 时不改动原图。
    """
    import cv2
    # ↓↓↓ 在这里写你的代码 ↓↓↓
    raise NotImplementedError("TODO: cv2.rectangle + cv2.putText")
    # ↑↑↑ 你的代码写在这里 ↑↑↑
