import os
import shutil


def copy_files_from_list(file_list_path, source_folder, destination_folder):
    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 读取文件列表
    with open(file_list_path, 'r') as file_list:
        file_names = [line.strip() for line in file_list if line.strip()]

    # 复制文件
    for file_name in file_names:
        file_name = file_name[10:-2]
        file_name = file_name + 'pts'
        print(file_name)
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name[:-3]+'seg')

        if os.path.exists(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"文件 {file_name} 已复制到 {destination_folder}")
        else:
            print(f"文件 {file_name} 不存在于源文件夹 {source_folder}")


# 示例使用
file_list_path = r'E:\ZXY\Data_new\annSplit_0.txt'  # 包含文件名的txt文件路径
source_folder = r'E:\ZXY\Data_new\label'  # 源文件夹路径
destination_folder = r'E:\ZXY\Data_new\label_train_0'  # 目标文件夹路径

copy_files_from_list(file_list_path, source_folder, destination_folder)
