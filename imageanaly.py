import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class ImageAnalysisTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis Tool")

        self.image_path = None
        self.image = None
        self.photo = None

        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_rectangle)

        menu_bar = tk.Menu(root)
        root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)

        self.load_button = tk.Button(root, text="Load Image", command=self.open_image)
        self.load_button.pack()

        self.histogram_button = tk.Button(root, text="Show Histogram", command=self.show_histogram)
        self.histogram_button.pack()

        self.info_label = tk.Label(root, text="")
        self.info_label.pack()

        self.rectangles = []
        self.rectangle_start = None
        self.rectangle_end = None

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.load_image()

    def load_image(self):
        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.reset_rectangles()

    def show_histogram(self):
        if self.image is not None:
            grayscale_image = self.image.convert("L")
            histogram_data = np.histogram(np.array(grayscale_image).flatten(), bins=256, range=[0, 256])[0]
            plt.plot(histogram_data, color="black")
            plt.title("Grayscale Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.show()

    def start_rectangle(self, event):
        self.rectangle_start = (event.x, event.y)

    def draw_rectangle(self, event):
        if self.image is not None and self.rectangle_start is not None:
            x, y = self.rectangle_start
            self.rectangle_end = (event.x, event.y)
            self.reset_rectangles()
            self.create_rectangle(x, y, event.x, event.y)

    def end_rectangle(self, event):
        if self.image is not None and self.rectangle_start is not None:
            x, y = self.rectangle_start
            self.create_rectangle(x, y, event.x, event.y)
            self.rectangle_start = None
            self.rectangle_end = None

    def create_rectangle(self, x1, y1, x2, y2):
        color = "#{:02x}{:02x}{:02x}".format(np.random.randint(256), np.random.randint(256), np.random.randint(256))
        rectangle = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color)
        self.rectangles.append(rectangle)

        self.calculate_average_grayscale(x1, y1, x2, y2)

    def calculate_average_grayscale(self, x1, y1, x2, y2):
        if self.image is not None:
            selected_region = self.image.crop((x1, y1, x2, y2))
            average_grayscale = np.mean(np.array(selected_region).flatten())

            self.info_label.config(text=f"Average Grayscale: {average_grayscale:.2f}")

    def reset_rectangles(self):
        for rectangle in self.rectangles:
            self.canvas.delete(rectangle)
        self.rectangles = []

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalysisTool(root)
    app.run()
