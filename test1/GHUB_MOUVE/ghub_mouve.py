import ctypes
import os
import time
import pyautogui
import pynput
import winsound
import threading
from pynput import mouse
mouse_left_click = False
mouse_right_click = False
from ctypes import CDLL
mouses = mouse.Controller()
try:
    print("正在加载罗技驱动")
    driver = ctypes.CDLL(r'D:\python a\yolov5-master\test1\GHUB_MOUVE\MouseControl.dll')
    print("驱动加载完成，快去奔放")
except FileNotFoundError:
    print("驱动调用失败，请检查原因")
    #鼠标按钮按下的回调函数
# try:
#     root = os.path.abspath(os.path.dirname(__file__))
#     driver = ctypes.CDLL(f'{root}/mouse.dll')
#     print('罗技驱动正在加载')
#     ok = driver.device_open() == 1
#     if not ok:
#         print('Error, GHUB or LGS driver not found')
# except FileNotFoundError:
#     print(f'Error, DLL file not found')

def on_click(x, y, button, pressed):
    global mouse_left_click, mouse_right_click
    if pressed:
        if button == mouse.Button.left:
            mouse_left_click = True
            print("左键按下")
        elif button == mouse.Button.right:
            mouse_right_click = True
            print("右键按下")
    else:
        mouse_left_click = False
        mouse_right_click = False
        print("按键松开"

)
class mouse_test:

    def release(key):
        if key == pynput.keyboard.Key.end:  # 结束程序 End 键
            winsound.Beep(400, 200)
            return False
        elif key == pynput.keyboard.Key.home:  # 移动鼠标 Home 键
            winsound.Beep(600, 200)


    # 绝对平滑移动num_steps越大移动慢，delay为睡眠时间和前面同理
    def linear_interpolation(self,x_end, y_end, num_steps, delay):
        start_x, start_y = pyautogui.position()
        dx = (x_end - start_x) / num_steps
        dy = (y_end - start_y) / num_steps

        for i in range(1, num_steps + 1):
            next_x = int(start_x + dx * i)
            next_y = int(start_y + dy * i)
            driver.move_Abs(int(next_x), int(next_y))
            time.sleep(delay)


    # 相对平滑移动num_steps越大移动慢，delay为睡眠时间和前面同理
    def r_linear_interpolation(self,r_x, r_y, num_steps, delay):
        r_y = 0 - r_y
        dx = r_x / num_steps
        dy = r_y / num_steps
        for i in range(1, num_steps + 1):
            next_x, next_y = (dx), (dy)
            driver.move_R(int(next_x), int(next_y))
            time.sleep(delay)

    def jiance(self):
        # 创建鼠标监听器
        listener = mouse.Listener(on_click=on_click)
        # 启动监听器
        listener.start()
        #保持主线程运行，以便监听鼠标事件
        # try:
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #listener.stop()
            # 用户按下 Ctrl+C止程序时，停止监听器终

    @staticmethod
    def move(x, y):
        if not ok:
            return
        if x == 0 and y == 0:
            return
        driver.moveR(x, y, True)
    def mouse_aim_controller(self,xywh_list, left, top, width, height):
        # 获取鼠标相对于屏幕的XY坐标
        mouse_x, mouse_y = mouse.position
        # 能获取到检测区域的大小以及位置
        best_xy = None
        for xywh in xywh_list:
            x, y, _, _ = xywh
            # 还原相对于监测区域的 x y
            x *= width
            y *= height
            # 转换坐标系，使得坐标系一致，统一为相对于屏幕的 x y 值
            x += left
            y += top
            dist = ((x - mouse_x) ** 2 + (y - mouse_y) ** 2) ** .5
            if not best_xy:
                best_xy = ((x, y), dist)
            else:
                _, old_dist = best_xy
                if dist < old_dist:
                    best_xy = ((x, y), dist)

        x, y = best_xy[0]
        sub_x, sub_y = x - mouse_x, y - mouse_y
        self.move(sub_x,sub_y)




test=mouse_test()
# # 创建两个线程来执行鼠标事件检测函数
# thread1 = threading.Thread(target=test.jiance)
# # thread2 = threading.Thread(target=test.linear_interpolation(30,30, num_steps=10, delay=0.01))
#
# # 启动两个线程
# thread1.start()
# # thread2.start()
# test.jiance()
# test.linear_interpolation(30,30, num_steps=10, delay=0.01)
# driver.click_Right_down();