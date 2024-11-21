import cv2
import numpy as np

def compute_texture_features(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 计算GLCM矩阵
    glcm = cv2.calcGLCM(image, [5], 0, [256], symmetric=True, normed=True)

    # 计算纹理特征
    contrast = cv2.compareHist(glcm[0], glcm[1], cv2.HISTCMP_CONTRAST)
    dissimilarity = cv2.compareHist(glcm[0], glcm[1], cv2.HISTCMP_DISSEMINATION)
    homogeneity = cv2.compareHist(glcm[0], glcm[1], cv2.HISTCMP_HOMOGENEITY)
    energy = cv2.compareHist(glcm[0], glcm[1], cv2.HISTCMP_ENERGY)

    return {
        "Contrast": contrast,
        "Dissimilarity": dissimilarity,
        "Homogeneity": homogeneity,
        "Energy": energy
    }

if __name__ == "__main__":
    # 图像文件路径
    image_path = "000.png"

    # 提取纹理特征
    texture_features = compute_texture_features(image_path)

    # 打印纹理特征
    for key, value in texture_features.items():
        print(f"{key}: {value}")
