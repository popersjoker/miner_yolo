import tkinter as tk
import sys
import threading
from pystray import Icon, MenuItem, Menu
from PIL import Image, ImageDraw
import time

root = None
label_output = None


# 创建托盘图标
class TkinterHandler:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)  # 插入日志消息
        self.text_widget.yview(tk.END)  # 滚动到最新的日志

    def flush(self):  # 这个方法是为了兼容 Python 标准库的文件处理器
        pass


# 创建图标
def create_image():
    width = 64
    height = 64
    image = Image.new("RGB", (width, height), (255, 255, 255))
    # draw = ImageDraw.Draw(image)
    #
    # eye_width = 40
    # eye_height = 24
    # eye_left = (width - eye_width) // 2
    # eye_top = (height - eye_height) // 2
    # eye_right = eye_left + eye_width
    # eye_bottom = eye_top + eye_height
    # draw.ellipse(
    #     (eye_left, eye_top, eye_right, eye_bottom),
    #     fill="white",
    #     outline="black"
    # )
    #
    # pupil_diameter = 10
    # pupil_left = (width - pupil_diameter) // 2
    # pupil_top = (height - pupil_diameter) // 2
    # draw.ellipse(
    #     (pupil_left, pupil_top, pupil_left + pupil_diameter, pupil_top + pupil_diameter),
    #     fill="black"
    # )
    #
    # eyelid_top = eye_top - 2
    # eyelid_bottom = eye_top + 6
    # draw.arc(
    #     (eye_left, eyelid_top, eye_right, eyelid_bottom),
    #     start=180,
    #     end=0,
    #     fill="black"
    # )

    return image


# 托盘菜单项
def quit_action():
    # icon.stop()
    try:
        root.quit()
        context.close=True
        root.destroy()
        sys.exit()

    except SystemExit as e:
        print("退出图形化界面", e)
def exit_action(icon, item):
    icon.stop()
    try:
        root.quit()
        context.close=True
        root.destroy()
        sys.exit()

    except SystemExit as e:
        print("退出图形化界面", e)


def show_window(icon, item):
    icon.stop()  # 停止托盘图标
    root.deiconify()  # 恢复窗口


# 托盘菜单
def create_tray():
    icon_image = create_image()
    menu = Menu(MenuItem("退出", exit_action), MenuItem("恢复窗口", show_window))
    icon = Icon("Anti-collisionDetector", icon_image, menu=menu)
    icon.run()


# 监听窗口最小化事件，将窗口隐藏
def minimize_window(event=None):
    print("最小化到托盘")
    root.withdraw()  # 隐藏窗口
    threading.Thread(target=quit_action, daemon=True).start()  # 启动托盘线程


# 切换按钮文本
def toggle_button_text(button):
    if button["text"] == "井上":
        button.config(text="井下")
        # context.toggle()
    else:
        button.config(text="井上")
    context.toggle()

def create_normal_window(event,con,log_queue):
    global root, label_output,context
    context=con
    root = tk.Tk()

    root.title("Anti-collision Detector")
    root.geometry("400x300")  # 调整窗口尺寸

    text_widget = tk.Text(root, height=15, width=150)  # 扩宽 text_widget
    text_widget.pack(pady=10, padx=20)  # 添加左右内边距

    def update_log():
        while not log_queue.empty():
            log_message = log_queue.get()
            text_widget.insert(tk.END, log_message)
            text_widget.see(tk.END)  # 滚动到最新日志
        root.after(500, update_log)
    root.after(500,update_log)
    # 将 TkinterHandler 添加到 logger

    from loguru import logger

    # if not stdout is None:
    # sys.stdout=handler
    # print("测试东定向")
    # 当窗口最小化时调用 minimize_window
    root.protocol("WM_DELETE_WINDOW", minimize_window)  # 窗口关闭时也执行最小化操作

    # 输出标签（用于显示main函数的输出）
    # label_output = tk.Label(root, text="程序输出", justify="left", anchor="w", padx=10, pady=10)
    # label_output.pack(padx=20, pady=20, fill=tk.X)

    # 创建按钮并绑定切换函数，添加样式
    # toggle_button = tk.Button(
    #     root,
    #     text="井上",
    #     command=lambda: toggle_button_text(toggle_button),
    #     bg="#4CAF50",  # 背景色
    #     fg="white",  # 字体颜色
    #     font=("Arial", 12, "bold"),  # 字体
    #     relief="raised",  # 按钮外观
    #     bd=4,  # 边框宽度
    #     padx=20,  # 按钮内边距
    #     pady=10  # 按钮内边距
    # )
    # toggle_button.pack(pady=20)  # 增加按钮的上下间距

    # def check_and_trigger_mode_change():
    #     last_mode = context.mode
    #     logger.info(f"checking mode change{context.mode}")
    #     root.after(1000, check_and_trigger_mode_change)
    #     # 如果mode改变了，手动触发按钮点击
    #     if context.mode != last_mode:
    #         logger.info('触发文本')
    #         toggle_button_text(toggle_button)

    # check_and_trigger_mode_change()
    # 启动 tkinter 窗口
    root.after(0, event.set)  # 设置事件，告诉主线程窗口已经创建完成
    root.mainloop()
def create_main_window(event, logger,con):
    global root, label_output,context
    context=con
    root = tk.Tk()

    root.title("Anti-collision Detector")
    root.geometry("400x300")  # 调整窗口尺寸

    text_widget = tk.Text(root, height=15, width=150)  # 扩宽 text_widget
    text_widget.pack(pady=10, padx=20)  # 添加左右内边距

    # 将 TkinterHandler 添加到 logger
    if logger is  None:
        from loguru import logger
    logger.remove()  # 移除默认的日志输出

    handler = TkinterHandler(text_widget)
    logger.add(handler, level="INFO")
    # if not stdout is None:
    # sys.stdout=handler
    # print("测试东定向")
    # 当窗口最小化时调用 minimize_window
    root.protocol("WM_DELETE_WINDOW", minimize_window)  # 窗口关闭时也执行最小化操作

    # 输出标签（用于显示main函数的输出）
    # label_output = tk.Label(root, text="程序输出", justify="left", anchor="w", padx=10, pady=10)
    # label_output.pack(padx=20, pady=20, fill=tk.X)

    # 创建按钮并绑定切换函数，添加样式
    # toggle_button = tk.Button(
    #     root,
    #     text="井上",
    #     command=lambda: toggle_button_text(toggle_button),
    #     bg="#4CAF50",  # 背景色
    #     fg="white",  # 字体颜色
    #     font=("Arial", 12, "bold"),  # 字体
    #     relief="raised",  # 按钮外观
    #     bd=4,  # 边框宽度
    #     padx=20,  # 按钮内边距
    #     pady=10  # 按钮内边距
    # )
    # toggle_button.pack(pady=20)  # 增加按钮的上下间距

    # def check_and_trigger_mode_change():
    #     last_mode = context.mode
    #     logger.info(f"checking mode change{context.mode}")
    #     root.after(1000, check_and_trigger_mode_change)
    #     # 如果mode改变了，手动触发按钮点击
    #     if context.mode != last_mode:
    #         logger.info('触发文本')
    #         toggle_button_text(toggle_button)

    # check_and_trigger_mode_change()
    # 启动 tkinter 窗口
    root.after(0, event.set)  # 设置事件，告诉主线程窗口已经创建完成
    root.mainloop()
    # print("窗口推出")
