import fitz  # PyMuPDF
from tkinter import filedialog

def extract_and_save_region(pdf_file, page_number, region, output_file):
    pdf_document = fitz.open(pdf_file)
    page = pdf_document.load_page(page_number - 1)
    x1, y1, x2, y2 = map(int, region)
    rect = fitz.Rect(x1, y1, x2, y2)
    new_pdf = fitz.open()
    new_page = new_pdf.new_page(width=rect.width, height=rect.height)
    new_page.show_pdf_page(new_page.rect, pdf_document, page_number - 1, clip=rect)
    new_pdf.save(output_file)
    pdf_document.close()
    new_pdf.close()

def main():
    # 选择PDF文件
    pdf_file = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not pdf_file:
        return

    # 手动输入六条垂直线的x坐标和四条水平线的y坐标
    vert_lines = [100, 200, 300, 400, 500, 600]  # 例：六条垂直线的x坐标
    horiz_lines = [100, 200, 300, 400]           # 例：四条水平线的y坐标

    # 计算九个矩形区域
    rectangles = []
    vert_lines.sort()
    horiz_lines.sort()
    for i in range(len(horiz_lines) - 1):
        for j in range(len(vert_lines) - 1):
            x1 = vert_lines[j]
            x2 = vert_lines[j + 1]
            y1 = horiz_lines[i]
            y2 = horiz_lines[i + 1]
            rectangles.append((x1, y1, x2, y2))

    # 提取并保存每个矩形区域
    for idx, rect in enumerate(rectangles):
        output_file = f"output_file_{idx + 1}.pdf"
        extract_and_save_region(pdf_file, 1, rect, output_file)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
