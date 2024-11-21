import cv2
import ctypes
from ctypes import *
import numpy as np
from pip._internal.resolution.resolvelib.factory import C

# dll = ctypes.cdll.LoadLibrary('cpb train2 dll.dll')
# dll1 = ctypes.cdll.LoadLibrary('cpbdetect-dll.dll')
dll3 = ctypes.cdll.LoadLibrary('Dll1.dll')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filePath = "D:\\WTF\\实验数据\\CPB实验数据\\jb302youzhong-cut\\"
    print(filePath.encode())
    resultPath = "..\\ "
    num = 20
    f = c_char_p(filePath.encode())
    r = c_char_p(resultPath.encode())
    # 1.传入实数调用demo
    img = cv2.imread('test.jpg')
    #img = cv2.imdecode(np.fromfile(r'D:\WTF\实验数据\CPB实验数据\jb302youzhong-cut\011600.jpg',dtype=np.uint8),-1)
    cols = img.shape[1]
    rows = img.shape[0]
    channels = 3
    dt = img.ctypes.data_as(POINTER(ctypes.c_ubyte))
    dll3.testmat(dt, rows, cols, channels)
    #dll1.detect.argtypes = (POINTER(ctypes.c_ubyte), c_int, c_int, c_int, c_char_p)
    # dll1.detect.restype = ctypes.c_void_p
    print(dt)
   # dll1.detect(dt, int(1200), int(1065), int(3), r)

    #result1 = dll.train(f,int(20),r)