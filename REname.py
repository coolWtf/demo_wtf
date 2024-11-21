import os

def main (path):
    filename_list = os.listdir(path)
    """os.listdir(path) 扫描路径的文件，将文件名存入存入列表"""

    a = 0
    b = 55
    for i in filename_list:
        used_name = path + filename_list[a]
        new_name = path + str(b).zfill(3)+'_mask.png' # 保留原后缀
        os.rename(used_name, new_name)
        print("文件%s重命名成功，新的文件名为%s" %(used_name,new_name))
        a += 1
        b += 1

if __name__=='__main__':
    path=r"E:\QuqXian\mvtec_anomaly_detection\leather_r\ground_truth\poke/" # 目标路径
    main(path)
