#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<iostream>
using namespace cv;
using namespace std; 

#ifdef  __cplusplus
extern "C"
{
#endif //  __cplusplus
#define DLL_EXPORT_API __declspec(dllexport)
	DLL_EXPORT_API	void testmat(uchar* dt, int rows, int cols, int channles);
#ifdef __cplusplus
}
#endif