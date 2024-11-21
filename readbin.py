import numpy as np
b = np.fromfile("C:\\Users\\T.f\\Desktop\\TBb.bin", dtype=np.float64, count=-1,offset=0,sep=' ')  #sep代表隔几个读取，count代表读多少个字节，-1代表读完整个文件
print('\n', b)
print('\n',b.size)