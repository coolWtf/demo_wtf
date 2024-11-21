import os
import shutil


def copy_files_to_new_folder(folders, extra_folder, output_base_folder):
    for i, test_folder in enumerate(folders):
        # 创建新的目标文件夹名称
        output_folder = os.path.join(output_base_folder, f'train_{i}_point_lc')
        extra_folder_dir = os.path.join(extra_folder, f'test_{i}_point_lc')
        # 创建目标文件夹（如果不存在）
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(extra_folder_dir, exist_ok=True)
        # 复制其他四个文件夹的内容
        for j, folder in enumerate(folders):
            if j != i:
              copy_folder_contents(folder, output_folder)

        # 复制额外文件夹的内容
        copy_folder_contents(test_folder, extra_folder_dir)

        print(f"已将文件夹 {test_folder} 作为测试文件夹，其他文件夹内容已复制到 {output_folder}")


def copy_folder_contents(source_folder, destination_folder):
    # 复制文件夹中的所有文件到目标文件夹
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


# 示例使用
folders = [
    r'E:\ZXY\Data_new\point_lc_train_0',  # 第一个文件夹路径
    r'E:\ZXY\Data_new\point_lc_train_1',  # 第二个文件夹路径
    r'E:\ZXY\Data_new\point_lc_train_2',  # 第三个文件夹路径
    r'E:\ZXY\Data_new\point_lc_train_3',  # 第四个文件夹路径
    r'E:\ZXY\Data_new\point_lc_train_4'  # 第五个文件夹路径
]
extra_folder = r'E:\ZXY\Data_new'  # 额外文件夹路径
output_base_folder = r'E:\ZXY\Data_new'  # 目标基础文件夹路径

copy_files_to_new_folder(folders, extra_folder, output_base_folder)
