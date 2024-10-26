import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk
import ttkbootstrap as ttkb
from multiprocessing import Pipe, Process
from ttkbootstrap import Style
from multiprocessing import freeze_support

# style = Style(theme='vapor')

# Image processing and color transfer functions
cpu_count = os.cpu_count()


def get_color_checker_corners(img):
    height, width, _ = img.shape
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    return corners


def get_color_block_centers(corners, rows=4, cols=6):
    centers = []
    x_step = (corners[1][0] - corners[0][0]) / cols
    y_step = (corners[2][1] - corners[0][1]) / rows
    for i in range(rows):
        for j in range(cols):
            x = corners[0][0] + (j + 0.5) * x_step
            y = corners[0][1] + (i + 0.5) * y_step
            centers.append((int(x), int(y)))
    return centers


def read_color_checker_image(path):
    return np.array(Image.open(path).convert("RGB"))


def sample_color_from_region(img, center, region_size=10):
    x, y = center
    region = img[max(0, y - region_size):min(img.shape[0], y + region_size),
             max(0, x - region_size):min(img.shape[1], x + region_size)]
    return np.mean(region, axis=(0, 1))


def calculate_nonlinear_color_matrix(img1, img2, centers, degree=2, conn=None):
    log_message(conn, "计算颜色转换矩阵...")
    log_message(conn, "从指定中心点提取颜色样本...")
    colors1 = np.array([sample_color_from_region(img1, center) / 255.0 for center in centers])
    colors2 = np.array([sample_color_from_region(img2, center) / 255.0 for center in centers])
    log_message(conn, "颜色样本被提取了。")
    poly = PolynomialFeatures(degree=degree)
    A = poly.fit_transform(colors1)
    log_message(conn, "训练线性回归模型以建立颜色映射...")
    model = LinearRegression()
    model.fit(A, colors2)
    log_message(conn, "线性回归模型训练完成了。")
    log_message(conn, "颜色转换矩阵计算完成了。")
    return model, poly


def transform_row(y, img, model, poly):
    row_transformed = np.zeros((img.shape[1], 3), dtype=np.float32)
    for x in range(img.shape[1]):
        color = img[y, x, :3] / 255.0
        color_poly = poly.transform([color])
        transformed_color = model.predict(color_poly)
        row_transformed[x, :3] = np.clip(transformed_color, 0, 1) * 255
    return y, row_transformed


def transform_image_nonlinear(img, model, poly, conn=None):
    log_message(conn, "开始转换图像，请稍候...")
    img_transformed = np.zeros_like(img, dtype=np.float32)
    height = img.shape[0]
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(transform_row, y, img, model, poly): y for y in range(height)}
        for future in futures:
            y, row_transformed = future.result()
            img_transformed[y] = row_transformed
            progress = y / height * 100
            log_message(conn, f"已处理第 {y} / {height} 行, 当前完成了 {progress:.2f}% ")
    log_message(conn, "图像转换完成!")
    return img_transformed.astype(np.uint8)


# def generate_lut_chunk(model, poly, grid_size, start, end):
#     lut_chunk = []
#     for idx in range(start, end):
#         r = (idx // (grid_size ** 2)) / (grid_size - 1)
#         g = ((idx // grid_size) % grid_size) / (grid_size - 1)
#         b = (idx % grid_size) / (grid_size - 1)
#         color = np.array([r, g, b])
#         color_poly = poly.transform([color])
#         transformed_color = model.predict(color_poly)
#         lut_chunk.append(transformed_color[0])
#     return lut_chunk
#
#
# # def generate_lut_nonlinear(model, poly, grid_size=33, conn=None):
# #     log_message(conn, "生成 LUT，请稍候...")
# #     total_steps = grid_size ** 3
# #     step_size = total_steps // 16
# #
# #     with ProcessPoolExecutor(max_workers=16) as executor:
# #         futures = {executor.submit(generate_lut_chunk, model, poly, grid_size, i, min(i + step_size, total_steps)): i
# #                    for i in range(0, total_steps, step_size)}
# #         lut = []
# #         for future in futures:
# #             lut.extend(future.result())
# #             step_count = len(lut)
# #             log_message(conn, f"LUT 进度: {step_count}/{total_steps} ({(step_count / total_steps) * 100:.2f}%)")
# #
# #     lut = np.array(lut).reshape((grid_size, grid_size, grid_size, 3))
# #     log_message(conn, "LUT 生成完成!")
# #     return lut
# def generate_lut_chunk(model, poly, grid_size, start, end):
#     lut_chunk = []
#     for idx in range(start, end):
#         r = (idx // (grid_size ** 2)) / (grid_size - 1)
#         g = ((idx // grid_size) % grid_size) / (grid_size - 1)
#         b = (idx % grid_size) / (grid_size - 1)
#         color = np.array([r, g, b])
#         color_poly = poly.transform([color])
#         transformed_color = model.predict(color_poly)
#         lut_chunk.append(transformed_color[0])
#     return lut_chunk
#
# def generate_lut_nonlinear(model, poly, grid_size=33,conn = None):
#     log_message(conn, "生成 LUT，请稍候...")
#     total_steps = grid_size ** 3
#     step_size = total_steps // 16  # 每个进程处理的步数
#
#     with ProcessPoolExecutor(max_workers=16) as executor:
#         futures = {executor.submit(generate_lut_chunk, model, poly, grid_size, i, min(i + step_size, total_steps)): i for i in range(0, total_steps, step_size)}
#         lut = []
#         for future in futures:
#             lut.extend(future.result())
#             step_count = len(lut)
#             log_message(conn, f"LUT 进度: {step_count}/{total_steps} ({(step_count / total_steps) * 100:.2f}%)")
#     log_message(conn, f"LUT 进度: {step_count}/{total_steps} ({(step_count / total_steps) * 100:.2f}%)")
#     lut = np.array(lut).reshape((grid_size, grid_size, grid_size, 3))
#     log_message(conn,"LUT 生成完成!")
#     return lut
#
#
# def save_cube_lut(lut, file_path, conn=None):
#     log_message(conn, f"保存 LUT 文件: {file_path}")
#     grid_size = lut.shape[0]
#     with open(file_path, 'w') as f:
#         f.write("TITLE \"Generated LUT\"\n")
#         f.write(f"LUT_3D_SIZE {grid_size}\n")
#         f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
#         f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
#         for b in range(grid_size):
#             for g in range(grid_size):
#                 for r in range(grid_size):
#                     color = lut[r, g, b]
#                     f.write(f"{color[0]} {color[1]} {color[2]}\n")
#     log_message(conn, "LUT 文件保存成功!")
def generate_lut_chunk(model, poly, grid_size, start, end):
    chunk = []
    for i in range(start, end):
        # 计算对应的输入值
        input_color = np.array([i // (grid_size ** 2), (i // grid_size) % grid_size, i % grid_size]) / (grid_size - 1)
        input_poly = poly.transform(input_color.reshape(1, -1))
        output_color = model.predict(input_poly)[0]
        # 确保输出值在 0 到 1 的范围内
        output_color = np.clip(output_color, 0, 1)
        chunk.append(output_color)
    return chunk


def generate_lut_nonlinear(model, poly, grid_size=65, conn=None):
    log_message(conn, "生成 LUT，请稍候...")
    total_steps = grid_size ** 3
    step_size = total_steps // 16  # 每个进程处理的步数
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(generate_lut_chunk, model, poly, grid_size, i, min(i + step_size, total_steps)): i
                   for i in range(0, total_steps, step_size)}
        lut = []
        for future in futures:
            lut.extend(future.result())
            step_count = len(lut)
            log_message(conn, f"LUT 进度: {step_count}/{total_steps} ({(step_count / total_steps) * 100:.2f}%)")
    log_message(conn, f"LUT 进度: {step_count}/{total_steps} ({(step_count / total_steps) * 100:.2f}%)")
    lut = np.array(lut).reshape((grid_size, grid_size, grid_size, 3))
    log_message(conn, "LUT 生成完成!")
    return lut


def save_cube_lut(lut, file_path, conn=None):
    log_message(conn, f"保存 LUT 文件: {file_path}")
    grid_size = lut.shape[0]
    with open(file_path, 'w') as f:
        f.write("TITLE \"Generated LUT\"\n")
        f.write(f"LUT_3D_SIZE {grid_size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        for b in range(grid_size):
            for g in range(grid_size):
                for r in range(grid_size):
                    color = lut[r, g, b]
                    f.write(f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n")  # 保持精度
    log_message(conn, "LUT 文件保存成功!")


def log_message(conn, message):
    print(message)
    if conn:
        conn.send(('log', message))


def on_exit():
    os._exit(0)


# GUI application

class ColorTransferApp:
    def __init__(self, root, conn):
        self.root = root
        self.conn = conn
        self.root.iconbitmap("icon.ico")
        self.root.title("Tunnel")
        self.root.protocol("WM_DELETE_WINDOW", on_exit)
        # Canvas for image display
        self.canvas1 = tk.Canvas(root, width=450, height=300, bg='lightgray')
        self.canvas1.grid(row=0, column=0, padx=10, pady=10)

        self.canvas2 = tk.Canvas(root, width=450, height=300, bg='lightgray')
        self.canvas2.grid(row=0, column=1, padx=10, pady=10)

        # Text entries for camera names
        self.camera_a_name = ttkb.Entry(root, width=20)
        self.camera_a_name.grid(row=1, column=0, padx=5, pady=5)
        self.camera_a_name.insert(0, "相机A")

        self.camera_b_name = ttkb.Entry(root, width=20)
        self.camera_b_name.grid(row=1, column=1, padx=5, pady=5)
        self.camera_b_name.insert(0, "相机B")

        # Frame for buttons
        button_frame = ttkb.Frame(root)
        button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        # Checkbox for previewing color conversion effect
        self.preview_var = tk.BooleanVar()  # 创建布尔变量以存储复选框状态
        self.preview_checkbox = ttkb.Checkbutton(
            button_frame,
            text="预览颜色转换效果（很慢）",
            variable=self.preview_var
        )
        self.preview_checkbox.grid(row=0, column=1, columnspan=2, sticky='w', padx=5)  # 复选框保留 columnspan=2

        # Buttons for selecting images

        self.about_btn = ttkb.Button(button_frame, text="关于Tunnel", command=self.about)
        self.about_btn.grid(row=1, column=0, padx=5)

        self.select_img1_btn = ttkb.Button(button_frame, text="选择主图像", command=self.load_image1)
        self.select_img1_btn.grid(row=1, column=1, padx=5)

        self.select_img2_btn = ttkb.Button(button_frame, text="选择目标图像", command=self.load_image2)
        self.select_img2_btn.grid(row=1, column=2, padx=5)

        # Button to start processing
        self.process_btn = ttkb.Button(button_frame, text="开始转换", command=self.start_processing)
        self.process_btn.grid(row=1, column=3, padx=5)

        # Add precision selection combobox
        precision_options = {
            "33点精度": 33,
            "65点精度": 65
        }
        self.precision_var = tk.IntVar(value=33)  # 默认值设为33
        self.precision_combo = ttkb.Combobox(
            button_frame,
            values=list(precision_options.keys()),
            width=10,
            state="readonly",
        )
        self.precision_combo.set("33点精度")  # 设置默认显示文本
        self.precision_combo.grid(row=1, column=4, padx=5)

        # 绑定选择改变事件
        def on_precision_select(event):
            selected_text = self.precision_combo.get()
            self.precision_var.set(precision_options[selected_text])
        self.precision_combo.bind("<<ComboboxSelected>>", on_precision_select)
        # Text box for logs
        self.log_text = tk.Text(root, width=100, height=10, bg='lightyellow')
        self.log_text.grid(row=3, column=0, columnspan=3, pady=10)
        self.log_text.insert(tk.END,
                             "Tunnel旨在提供一种提取色卡图片色彩特征并进行转换的方式。\n该程序主要由人工智能生成。\n")
        # Images
        self.img1 = None
        self.img2 = None

        self.root.after(100, self.check_for_messages)

    def load_image1(self):
        path = filedialog.askopenfilename()
        if path:
            self.img1 = read_color_checker_image(path)
            self.display_image(self.img1, self.canvas1)

    def load_image2(self):
        path = filedialog.askopenfilename()
        if path:
            self.img2 = read_color_checker_image(path)
            self.display_image(self.img2, self.canvas2)

    def display_image(self, img, canvas):
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((450, 300), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def start_processing(self):
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("错误", "请先选择两张图像")
            return
        self.conn.send(('process', self.img1, self.img2, self.camera_a_name.get(), self.camera_b_name.get(),
                        self.preview_var.get(), self.precision_var.get()))

    def check_for_messages(self):
        while self.conn.poll():
            msg_type, msg_content = self.conn.recv()
            if msg_type == 'log':
                self.log(msg_content)
        self.root.after(100, self.check_for_messages)

    def about(self):
        messagebox.showinfo("关于Tunnel", "Tunnel可以用于处理色卡图片之间的色彩风格迁移。\n开源软件，以MIT许可证分发。\n本软件按“原样”提供，不包含任何形式的保证，无论是明示的还是暗示的，包括但不限于适销性、特定用途的适用性和不侵权的保证。在任何情况下，开发者均不对因使用本软件而产生的任何形式的损害或其他责任负责，无论这些损害是基于合同、侵权或其他法律理论的。\n本项目开源代码仓库：https://github.com/iZhuzzyi/Tunnel\n用于贡献代码、分支其他版本或提交意见与功能要求。")

def process_images(conn):
    while True:
        msg = conn.recv()
        if msg[0] == 'process':
            test_img = read_color_checker_image("TDTI.jpeg")
            img1, img2, camera_a, camera_b, preview_enabled, precision_var = msg[1], msg[2], msg[3], msg[4], msg[5], msg[6]
            log_message(conn, str(preview_enabled)+"   "+str(precision_var))
            corners = get_color_checker_corners(img1)
            centers = get_color_block_centers(corners)
            if img1.shape != img2.shape:
                log_message(conn, "调整目标图像大小以匹配主图像...")
                img2 = np.array(Image.fromarray(img2).resize((img1.shape[1], img1.shape[0])), dtype=np.uint8)
                test_img = np.array(Image.fromarray(test_img).resize((img1.shape[1], img1.shape[0])), dtype=np.uint8)
                img1 = np.array(img1, dtype=np.uint8)
            model1, poly1 = calculate_nonlinear_color_matrix(img1, img2, centers, degree=2, conn=conn)
            model2, poly2 = calculate_nonlinear_color_matrix(img2, img1, centers, degree=2, conn=conn)
            if preview_enabled:
                log_message(conn, "图像会被转换并用于预览。")
                img1_transformed = transform_image_nonlinear(img1, model1, poly1, conn=conn)
                img2_transformed = transform_image_nonlinear(img2, model2, poly2, conn=conn)
                testimg_transformed_1 = transform_image_nonlinear(test_img, model1, poly1, conn=conn)
                # Display transformed images using Matplotlib
                plt.figure(figsize=(18, 6))  # 调整画布大小以适应三张图片
                # 第1张图片
                plt.subplot(1, 3, 1)
                plt.title("Transformed Image 1")
                plt.imshow(img1_transformed)

                # 第2张图片
                plt.subplot(1, 3, 2)
                plt.title("Transformed Image 2")
                plt.imshow(img2_transformed)

                # 第3张图片，新的图片
                plt.subplot(1, 3, 3)
                plt.title("Transformed Test Image")
                plt.imshow(testimg_transformed_1)

                # 显示图像
                plt.show(block=False)
            else:
                log_message(conn, "图像转换和预览被跳过了。")
            root = tk.Tk()
            root.withdraw()
            # Ask for LUT file paths
            lut1_file = filedialog.asksaveasfilename(defaultextension=".cube",
                                                     initialfile=f"{camera_a}_to_{camera_b}.cube")
            lut2_file = filedialog.asksaveasfilename(defaultextension=".cube",
                                                     initialfile=f"{camera_b}_to_{camera_a}.cube")

            # Start LUT generation in parallel
            lut1 = generate_lut_nonlinear(model1, poly1, grid_size=precision_var, conn=conn)
            lut2 = generate_lut_nonlinear(model2, poly2, grid_size=precision_var, conn=conn)

            save_cube_lut(lut1, lut1_file, conn=conn)
            save_cube_lut(lut2, lut2_file, conn=conn)

            log_message(conn, f"LUT 文件已生成并保存为 {lut1_file} 和 {lut2_file}.")


def run_gui(conn):
    root = ttkb.Window(themename='vapor')
    app = ColorTransferApp(root, conn)
    root.mainloop()


if __name__ == "__main__":
    freeze_support()
    multiprocessing.set_start_method('spawn')
    parent_conn, child_conn = Pipe()

    gui_process = Process(target=run_gui, args=(parent_conn,))
    gui_process.start()

    process_images(child_conn)
