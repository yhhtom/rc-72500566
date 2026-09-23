# -*- coding: utf-8 -*-
"""
Day3 例图生成脚本：用 OpenCV 绘制规则图案测试图
生成：
  images/demo_shapes.png   入门演示图（干净背景，多形状多颜色）
  images/quiz_01.png       小测1：红色圆形
  images/quiz_02.png       小测2：蓝色正方形（含噪点干扰）
  images/quiz_03.png       小测3：识别三角形 + 输出质心
  images/quiz_04.png       小测4：多形状多颜色，识别指定目标
"""
import cv2
import numpy as np
import os

IMG_W, IMG_H = 640, 480
OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)


def save_img(img, filename):
    """用 imencode 写入，解决 OpenCV 对中文路径不支持的问题"""
    path = os.path.join(OUT, filename)
    ext = "." + filename.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("imencode 失败: " + filename)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def blank():
    return np.full((IMG_H, IMG_W, 3), 255, np.uint8)


def add_noise(img, density=0.02):
    """随机撒小白点模拟噪点干扰"""
    rng = np.random.default_rng(42)
    n = int(IMG_W * IMG_H * density)
    for _ in range(n):
        x, y = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        img[y, x] = (rng.integers(0, 256), rng.integers(0, 256), rng.integers(0, 256))
    return img


# 颜色定义 (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)


def demo_shapes():
    img = blank()
    cv2.circle(img, (150, 150), 60, RED, -1)      # 红圆
    cv2.rectangle(img, (320, 90), (450, 220), BLUE, -1)  # 蓝方块
    pts = np.array([[570, 250], [490, 100], [650, 100]], np.int32)
    cv2.fillPoly(img, [pts], GREEN)                # 绿三角
    cv2.line(img, (80, 380), (560, 380), YELLOW, 20)     # 黄线
    cv2.circle(img, (180, 400), 25, ORANGE, -1)    # 橙圆
    save_img(img, "demo_shapes.png")


def quiz_01():
    img = blank()
    cv2.circle(img, (320, 240), 90, RED, -1)       # 一个红色大圆
    add_noise(img)
    save_img(img, "quiz_01.png")


def quiz_02():
    img = blank()
    cv2.rectangle(img, (240, 160), (400, 320), BLUE, -1)  # 蓝色正方形
    add_noise(img)
    save_img(img, "quiz_02.png")


def quiz_03():
    img = blank()
    pts = np.array([[320, 80], [180, 400], [460, 400]], np.int32)
    cv2.fillPoly(img, [pts], GREEN)                # 一个绿色大三角形
    cv2.circle(img, (120, 100), 20, RED, -1)       # 干扰小红圆
    cv2.rectangle(img, (500, 50), (560, 110), BLUE, -1)  # 干扰小蓝块
    add_noise(img)
    save_img(img, "quiz_03.png")


def quiz_04():
    img = blank()
    cv2.circle(img, (120, 130), 55, RED, -1)
    cv2.circle(img, (520, 130), 55, BLUE, -1)
    cv2.rectangle(img, (180, 320), (320, 460), GREEN, -1)
    pts = np.array([[420, 460], [380, 320], [560, 340]], np.int32)
    cv2.fillPoly(img, [pts], RED)                  # 红色三角形
    cv2.circle(img, (420, 120), 30, ORANGE, -1)
    add_noise(img)
    save_img(img, "quiz_04.png")


def demo_gray():
    """灰度图：均匀浅背景 + 实心深色物体 + 一条细线（光照均匀）
    目的：清楚演示三种二值化的区别
      - 固定阈值 / OTSU：能把实心深色物体完整分离出来（实心）
      - 自适应：只认边缘，实心物体会被"吃空"，但【细线】能完整识别
         → 直观展示"自适应适合线/边缘（车道线、巡迹线），不适合实心"
    构图：背景200 + 实心圆30 + 实心方块60 + 底部一条细黑线(宽4)
    """
    img = np.full((IMG_H, IMG_W), 200, np.uint8)   # 均匀浅灰背景(亮度200)
    cv2.circle(img, (150, 150), 70, 30, -1)        # 实心深圆(亮度30)
    cv2.rectangle(img, (430, 130), (560, 260), 60, -1)  # 实心深方块(亮度60)
    cv2.line(img, (40, 380), (600, 380), 30, 4)    # 细黑线(宽4)，模拟车道线
    save_img(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), "demo_gray.png")


def demo_lowcontrast():
    """低对比度图：整体灰蒙蒙、亮暗区分度低
    目的：给例程 11（直方图均衡化）演示用
      低对比度图均衡化效果最明显：原本挤在中间一段的亮度被拉开，
      对比度提升、细节变清楚。（demo_gray 对比太强，不适合演示均衡化）
    构图：背景亮度在 110~150 之间缓慢变化（灰蒙蒙），
          加几个亮暗差异很小的圆形物体。
    """
    img = np.full((IMG_H, IMG_W), 130, np.uint8)   # 灰蒙蒙背景(亮度130)
    # 让背景有轻微亮度起伏（模拟雾/低对比场景）
    for y in range(IMG_H):
        for x in range(IMG_W):
            v = 130 + int(20 * np.sin(x / 50.0)) + int(10 * np.sin(y / 40.0))
            img[y, x] = np.clip(v, 0, 255)
    # 加几个亮度差异小的圆（和背景接近，均衡化前几乎看不清）
    cv2.circle(img, (140, 150), 70, 105, -1)    # 亮度105（略暗于背景130）
    cv2.circle(img, (430, 160), 60, 155, -1)    # 亮度155（略亮于背景130）
    cv2.rectangle(img, (180, 320), (340, 430), 118, -1)  # 亮度118
    save_img(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), "demo_lowcontrast.png")


def demo_lines_circles():
    """几何图：直线 + 圆，演示霍夫直线/圆检测"""
    img = blank()
    cv2.line(img, (50, 120), (590, 120), (0, 0, 0), 5)   # 横线
    cv2.line(img, (120, 40), (300, 420), (0, 0, 0), 5)   # 斜线
    cv2.line(img, (450, 30), (450, 430), (0, 0, 0), 5)   # 竖线
    cv2.circle(img, (150, 320), 50, (0, 0, 255), 3)      # 圆
    cv2.circle(img, (480, 300), 80, (0, 255, 0), 3)
    cv2.circle(img, (320, 150), 35, (255, 0, 0), 3)
    save_img(img, "demo_lines_circles.png")


def demo_noise_text():
    """带盐噪点的图：演示形态学开/闭运算去噪"""
    img = blank()
    cv2.circle(img, (160, 150), 90, (0, 0, 0), 4)
    cv2.rectangle(img, (320, 90), (520, 300), (0, 0, 0), 4)
    # 大量随机黑白噪点（盐噪）
    rng = np.random.default_rng(7)
    for _ in range(3000):
        x, y = rng.integers(0, IMG_W), rng.integers(0, IMG_H)
        v = 255 if rng.random() > 0.5 else 0
        img[y, x] = (v, v, v)
    save_img(img, "demo_noise_text.png")


def demo_roi():
    """彩色大图：多物体，演示直方图/ROI/几何变换/连通域"""
    img = blank()
    cv2.circle(img, (150, 130), 60, RED, -1)
    cv2.circle(img, (500, 130), 60, BLUE, -1)
    cv2.rectangle(img, (150, 320), (300, 460), GREEN, -1)
    cv2.rectangle(img, (430, 330), (540, 440), YELLOW, -1)
    pts = np.array([[320, 90], [270, 60], [370, 60]], np.int32)
    cv2.fillPoly(img, [pts], ORANGE)                 # 小三角
    cv2.circle(img, (450, 220), 30, (200, 200, 200), -1)
    add_noise(img, density=0.008)
    save_img(img, "demo_roi.png")


def patterns():
    """彩色多图案图（A4 打印用）：红圆/蓝方块/绿三角/黄圆
    目的：给 track_target.py（实时追踪）当"识别对象"——打印出来举着
          对着摄像头，程序实时算出指定图案的中心坐标。
    设计：纯白背景 + 4 个独立大图案，间距大、颜色纯，摄像头识别清晰。
    """
    W, H = 1920, 1080          # 高清，打印清晰
    img = np.full((H, W, 3), 255, np.uint8)   # 纯白背景
    # 4 个图案分布（左上/右上/左下/右下，间距大）
    cv2.circle(img, (480, 270), 150, RED, -1)          # 红圆（左上）
    cv2.rectangle(img, (1290, 120), (1650, 420), BLUE, -1)   # 蓝方块（右上）
    pts = np.array([[480, 810], [330, 560], [630, 560]], np.int32)
    cv2.fillPoly(img, [pts], GREEN)                    # 绿三角（左下）
    cv2.circle(img, (1440, 800), 140, YELLOW, -1)      # 黄圆（右下）
    # 给每个图案加一个灰色数字标签（1-4），方便区分
    cv2.putText(img, "1", (480, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 6)
    cv2.putText(img, "2", (1470, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 6)
    cv2.putText(img, "3", (480, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 6)
    cv2.putText(img, "4", (1440, 800), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 6)
    save_img(img, "patterns.png")


if __name__ == "__main__":
    demo_shapes()
    quiz_01()
    quiz_02()
    quiz_03()
    quiz_04()
    demo_gray()
    demo_lines_circles()
    demo_noise_text()
    demo_roi()
    demo_lowcontrast()
    patterns()
    print("已生成：")
    for f in sorted(os.listdir(OUT)):
        print("  -", f)
