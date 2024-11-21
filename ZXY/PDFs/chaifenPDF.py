import fitz  # PyMuPDF


def save_each_page_as_pdf(input_pdf, output_folder):
    # 打开原始 PDF 文件
    pdf_document = fitz.open(input_pdf)

    # 遍历每一页
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # 创建一个新的 PDF 文件
        new_pdf = fitz.open()

        # 计算页面的矩形区域
        page_rect = page.rect
        new_page = new_pdf.new_page(width=page_rect.width, height=page_rect.height)

        # 将当前页面的内容复制到新的页面中
        new_page.show_pdf_page(new_page.rect, pdf_document, page_number)

        # 保存新的 PDF 文件
        output_pdf_path = f"{output_folder}/page_{page_number + 1}.pdf"
        new_pdf.save(output_pdf_path)
        new_pdf.close()
        print(f"Saved page {page_number + 1} as {output_pdf_path}")

    # 关闭原始 PDF 文件
    pdf_document.close()


# 示例用法
input_pdf_path = "组合 1.pdf"  # 输入 PDF 文件路径
output_folder = "./test/"  # 输出文件夹路径

# 确保输出文件夹存在
import os

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

save_each_page_as_pdf(input_pdf_path, output_folder)
