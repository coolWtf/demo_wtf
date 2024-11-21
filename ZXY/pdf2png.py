import fitz  # PyMuPDF
from PIL import Image

# 打开PDF文件
doc = fitz.open("PDFs/page_1_1.pdf")

# 选择要转换的页面（例如，第一页）
page = doc.load_page(0)

# 获取页面的边界矩形
rect = page.rect

# 设置缩放比例
zoom_x = 2.0
zoom_y = 2.0

# 使用缩放矩阵获取高分辨率的Pixmap
matrix = fitz.Matrix(zoom_x, zoom_y)
pix = page.get_pixmap(matrix=matrix, alpha=False)

# 将Pixmap对象转换为PIL图像
image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# 保存为PNG
image.save("F_bt.png")

# 清理资源
pix = None
doc.close()