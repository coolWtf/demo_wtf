import os
import re


def detect_language(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.java':
        return 'Java'
    elif file_extension == '.php':
        return 'PHP'
    elif file_extension == '.py':
        return 'Python'
    elif file_extension == '.cpp':
        return 'C++'
    # 添加其他语言的判断条件

    # 或者根据文件内容判断
    # with open(file_path, 'r') as file:
    #     file_content = file.read()
    #     if 'import java' in file_content:
    #         return 'Java'
    #     elif '<?php' in file_content:
    #         return 'PHP'
    #     elif 'import' in file_content or 'from' in file_content:
    #         return 'Python'
    #     # 添加其他语言的判断条件

    return None  # 如果无法判断语言，则返回None

def scan_folder(folder_path):
    folder_tree = {}

    for root, dirs, files in os.walk(folder_path):
        folder_name = os.path.basename(root)
        folder_tree[folder_name] = {}

        for file in files:
            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(root, file)
            language = detect_language(file_path)

            if language:
                folder_tree[folder_name][file_name] = language

    return folder_tree

def remove_blank_lines(code):
    code_lines = code.split('\n')
    non_blank_lines = [line for line in code_lines if line.strip()]
    return '\n'.join(non_blank_lines)
def remove_comments(code):
    # 使用正则表达式删除注释
    pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'
    code_without_comments = re.sub(pattern, '', code, flags=re.S | re.MULTILINE)
    return code_without_comments

def remove_inconsistent_version(code, version_number):
    # 删除注释中与版本号不一致的说明
    pattern = r'@version\s+[^*\n\r]+'
    code_without_inconsistent_version = re.sub(pattern, '', code, flags=re.MULTILINE)
    return code_without_inconsistent_version


if __name__ == '__main__':
    folder_path = r'E:\qs_by\ramile(1)'
    folder_tree = scan_folder(folder_path)
    print(folder_tree)
