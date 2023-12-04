import sys
import subprocess
import tkinter.messagebox
from tkinter import filedialog
import tkinter
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
#import numpy as np

pt_path="CF.pt"
conf=0.6
screen="screen 480 270 960 540"
command1= [
    "python",
    "detect.py",
    "--source",
    screen,
    "--save-txt",
    "--conf-thres",
    str(conf),
    "--weights",
    pt_path
]

#commda="python detect.py --source "+screen+" --save-txt"+" --conf-thres"+conf
#print(commda)
# 创建主窗口
window = tk.Tk()

def pt1():
    global wjlj
    global pt_path  # 声明为全局变量
    wjlj = tkinter.filedialog.askopenfilename()  # 打开文件对话框，获取绝对路径
    if wjlj==None:
        print("程序异常,选择为空,自动退出")
        sys.exit()
    else:
        print("权限选择成功:"+wjlj)
        command1[8] = wjlj
        print("当前运行配置：" +str(command1))
        return wjlj
def conf1():
    global conf
    # 创建主窗口
    root = tk.Tk()
    root.title("置信度选择")
    root.geometry("300x200")
    # 创建滑块控件
    scale = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
    scale.pack()
    def on_confirm():
        conf = scale.get()  # 获取滑块选择的值
        command1[6] = conf
        print("当前运行配置：" +str(command1))
        root.destroy()  # 关闭主窗口
    # 创建确认按钮
    confirm_button = tk.Button(root, text="确定", command=on_confirm)
    confirm_button.pack()
def on_button_click():
    # 创建菜单
    menu = tk.Menu(window, tearoff=0)
    # 添加选项
    menu.add_command(label="1920*1080", command=lambda: handle_option(1))
    menu.add_command(label="960*540", command=lambda: handle_option(2))
    menu.add_command(label="1060*610", command=lambda: handle_option(3))
    # 显示菜单
    menu.post(btn3.winfo_rootx(), btn3.winfo_rooty() + btn3.winfo_height())
def handle_option(t):
    global screen
    if(t==1):
        screen = "screen 0 0 1920 1080"
        print("当前识别区域为:" + screen)
        command1[3] = screen  # 更新命令中的识别区域参数
        print("当前运行配置：" +str(command1))
    if(t==2):
        screen = "screen 480 270 960 540"
        command1[3] = screen
        print("当前运行配置：" +str(command1))
    if(t==3):
        screen = "screen 0 0 1050 610"
        command1[3] = screen
        print("当前运行配置：" +str(command1))
def pm():
    pass
# 执行命令行命令
# 设置窗口大小和标题
window.geometry("528x350")
window.title("cxkbalq")
def kaishi():
    global command1
    # 将 "0.5" 转换为整数
    conf_thres = float(float(command1[command1.index("--conf-thres") + 1]))
    # 更新命令列表中的值
    command1[command1.index("--conf-thres") + 1] = str(conf_thres)
    print(command1)
    print("开始启动,请耐心等待")
    subprocess.run(command1)
# 加载背景图片1
image = Image.open("1.png")
background_image = ImageTk.PhotoImage(image)

# # 创建标签并设置背景图片
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
btn1 = Button(window, text="权重选择", height=1, width=10,command=pt1)
btn1.grid(row=0, column=0, pady=300, padx=20)
btn2 = Button(window, text="置信度选择", height=1, width=10,command=conf1)
btn2.grid(row=0, column=1, pady=300, padx=20)
btn3 = Button(window, text="识别区域", height=1, width=10,command=on_button_click)
btn3.grid(row=0, column=2, pady=300, padx=20)
btn4 = Button(window, text="开始运行", height=1, width=10,command=kaishi)
btn4.grid(row=0, column=3, pady=300, padx=20)
# 运行窗口
window.mainloop()