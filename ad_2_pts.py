import os


def process_ad_files(input_folder, output_folder_3cols, output_folder_1col):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_3cols, exist_ok=True)
    os.makedirs(output_folder_1col, exist_ok=True)

    # 获取所有.ad文件
    ad_files = [f for f in os.listdir(input_folder) if f.endswith('.ad')]

    for ad_file in ad_files:
        input_path = os.path.join(input_folder, ad_file)
        output_path_3cols = os.path.join(output_folder_3cols, f'{os.path.splitext(ad_file)[0]}.pts')
        output_path_1col = os.path.join(output_folder_1col, f'{os.path.splitext(ad_file)[0]}.pts')

        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"文件 {input_path} 不存在")
            continue

        # 打开输入文件和输出文件
        with open(input_path, 'r') as infile, \
                open(output_path_3cols, 'w') as outfile_3cols, \
                open(output_path_1col, 'w') as outfile_1col:

            for line in infile:
                # 解析每一行的数据
                columns = line.split()

                # 提取前三列和最后一列
                if len(columns) >= 7:
                    first_three_cols = columns[:3]
                    last_col = columns[-1]

                    # 写入输出文件
                    outfile_3cols.write(' '.join(first_three_cols) + '\n')
                    outfile_1col.write(last_col + '\n')
                else:
                    print(f"警告：跳过行，因为它没有7列: {line.strip()}")

        print(f"处理完成。文件 {ad_file} 的前三列数据保存在 {output_path_3cols}，最后一列数据保存在 {output_path_1col}")


# 示例使用
input_folder = r'E:\ZXY\Data_new\intrA'  # 输入文件夹路径
output_folder_3cols = r'E:\ZXY\Data_new\point_lc'  # 输出前三列数据的文件夹路径
output_folder_1col = r'E:\ZXY\Data_new\label'  # 输出最后一列数据的文件夹路径

process_ad_files(input_folder, output_folder_3cols, output_folder_1col)
