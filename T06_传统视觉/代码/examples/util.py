# -*- coding: utf-8 -*-
"""
工具：中文路径安全的 imread / imwrite / 控制台输出
原因：
  1. OpenCV 的 cv2.imread / imwrite 在 Windows 上对含中文路径支持不好
     → 用 np.fromfile + cv2.imdecode 读，cv2.imencode + 文件写
  2. Python 在 Windows 控制台默认用 GBK 输出，现代终端(UTF-8)会乱码
     → 强制 stdout 用 UTF-8
"""
import os
import sys
import cv2
import numpy as np


def setup():
    """在脚本开头调用一次，让控制台中文不乱码"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass  # 非 UTF-8 终端或旧版本 Python 时静默跳过


def imread(path):
    """读取图片，支持中文路径"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def imwrite(path, img):
    """写入图片，支持中文路径"""
    # 扩展名（含点号），如 ".png"
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("imwrite failed: " + path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def show_image(window_name, img):
    """统一显示图片，按 q 退出"""
    cv2.imshow(window_name, img)
    key = cv2.waitKey(0)
    if key == ord("q"):
        return False
    return True


def arrange_windows(names=None, step=45, offset=24):
    """把多个窗口按斜线错开排列，避免叠在同一个位置（让人误以为"逐个出现"）。
    names：窗口名列表。若不传，则尝试从当前打开的所有窗口取（需支持的 cv2 版本）。
    在 cv2.imshow 全部调用之后、waitKey 之前调用，窗口就会一次错开显示出来。
    """
    if os.environ.get("HEADLESS") == "1":
        return
    if not names:
        # 尝试自动枚举（新版本 cv2 支持 windowNames）
        try:
            names = list(cv2.windowNames)
        except AttributeError:
            return  # 无法枚举则跳过
    for i, name in enumerate(names):
        try:
            cv2.moveWindow(name, offset + (i % 3) * step, offset + (i % 3) * step)
        except cv2.error:
            pass  # 窗口已关闭则跳过


def finish(names=None):
    """一次弹所有窗口（已由 imshow 弹出）→ 错开排列 → 按任意键全部关闭。
    names：窗口名列表，用于把多窗口错开排列避免重叠。
    HEADLESS=1 时跳过，用于批量/无头测试。
    """
    arrange_windows(names)
    if os.environ.get("HEADLESS") != "1":
        cv2.waitKey(0)
    cv2.destroyAllWindows()
