import ctypes as C

dll = C.cdll.LoadLibrary('cpb train2 dll.dll')

filePath = "C:\\Users\\T.f\\Desktop\\项目汇报\\原始帧\\youzhong\\"
resultPath = "..\\ "
#1.传入实数调用demo
result1 = dll.train(filePath, 20, resultPath)
#print("传入实数调用demo:")
#print(result1)
