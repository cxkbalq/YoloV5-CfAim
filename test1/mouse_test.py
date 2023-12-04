import csv
import os
import sys
import time
import pyautogui
from pathlib import Path
from pynput import mouse
mouse_left_click = False
mouse_right_click = False
mouses = mouse.Controller()
# mouses_test=mouses.click()
def xywh(box):
    # 将第一个张量转换为Python数字
    x = box[0].item()
    y = box[1].item()
    w = box[2].item()
    h = box[3].item()
    # 计算框的中心点坐标
    w_half = (w - x) / 2
    h_half = (h - y) / 2

    # 计算目标位置的坐标
    x1 = x + w_half + 480
    y1 = y + h_half + 270

    # 是否设置安全移动，防止死循环
    pyautogui.FAILSAFE = False
    # 获取当前鼠标位置
    current_position = pyautogui.position()
    # 移动鼠标到目标位置
    pyautogui.moveTo(x1, y1)
    # 这是鼠标移动以及单机
def on_click(x, y, button, pressed):
    global mouse_left_click, mouse_right_click
    if pressed:
        if button == mouse.Button.left:
            # 按下鼠标左键
            mouse_left_click = True
            print("调用b1")
        elif button == mouse.Button.right:
            # 按下鼠标右键
            mouse_right_click = True
            print("调用b2")
    else:
        # 无论鼠标哪一个键松开，都会执行下面的东西
        mouse_left_click = False
        print("调用a1")
        mouse_right_click = False
        # a.move(100,100)
        print("调用a2")
        # 获取鼠标当前的位置
        current_position = pyautogui.position()
        # 输出鼠标当前的位置
        print("鼠标当前位置：", current_position)

def on_move(self, x, y):
    print('鼠标移动至坐标：({0}, {1})'.format(x, y))
for i in range(0,3):
    on_click(1, 1, mouse.Button.left, False)
    time.sleep(1)




