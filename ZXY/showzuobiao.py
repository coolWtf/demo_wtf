import fitz  # PyMuPDF
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog


class PDFViewer(tk.Tk):
    def __init__(self, pdf_file, page_number):
        super().__init__()
        self.pdf_file = pdf_file
        self.page_number = page_number
        self.canvas = tk.Canvas(self, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.load_pdf_page()

    def load_pdf_page(self):
        pdf_document = fitz.open(self.pdf_file)
        page = pdf_document.load_page(self.page_number - 1)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        self.img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.img, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        pdf_document.close()

    def on_button_press(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        print(f"Clicked at: ({x}, {y})")


def main():
    pdf_file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_file:
        return
    app = PDFViewer(pdf_file, 1)
    app.mainloop()


if __name__ == "__main__":
    main()
