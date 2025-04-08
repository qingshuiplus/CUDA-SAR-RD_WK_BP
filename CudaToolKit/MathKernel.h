#pragma once
#include "CudaHeader.h"

void DLL_EXPORT cuda_fft(cuDoubleComplex* data, int rows, int cols, int dim);

void DLL_EXPORT cuda_ifft(cuDoubleComplex* data, int rows, int cols, int dim);

DLL_EXPORT __global__ void LineSpace(double* x, double start, double end, int count);

DLL_EXPORT __device__ int PCHIP_Index(double* x, double v, int len);

DLL_EXPORT __device__ int PCHIP_Index_X(double* x, double v, int len, int ir, int start, int row);

DLL_EXPORT __device__ int PCHIP_Index_Y_Dec(double* x, double v, int len, int ic, int start, int row);

DLL_EXPORT __device__ double PCHIP_Diff(double* h, double* delta, int n, int k);

DLL_EXPORT __global__ void PCHIP_SLOPE_Dec(double* x, double* y, double* h, double* delta, int len);

DLL_EXPORT __global__ void PCHIP_SLOPE(double* x, double* y, double* h, double* delta, int len);

DLL_EXPORT __global__ void PCHIP_SLOPE_CX(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int len);

DLL_EXPORT __global__ void PCHIP_INTERP(double* x, double* y, double* h, double* delta, double* xq, double* yq, int n, int m);

DLL_EXPORT __global__ void PCHIP_INTERP_Dec(double* x, double* y, double* h, double* delta, double* xq, double* yq, int n, int m);

DLL_EXPORT __global__ void PCHIP_INTERP_CX(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq, int n, int m);

DLL_EXPORT __global__ void PCHIP_SLOPE_X(double* x, double* y, double* h, double* delta, int ir, int start, int count, int row, int col);

DLL_EXPORT __global__ void PCHIP_INTERP_X(double* x, double* y, double* h, double* delta, double* xq, double* yq, int ir, int start, int n, int m, int row, int col);

DLL_EXPORT __global__ void PCHIP_SLOPE_CX_X(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int ir, int start, int count, int row, int col);

DLL_EXPORT __global__ void PCHIP_INTERP_CX_X(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq, int ir, int start, int n, int m, int row, int col);

DLL_EXPORT __global__ void PCHIP_SLOPE_CX_Y(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int ic, int start, int count, int row, int col);

DLL_EXPORT __global__ void PCHIP_INTERP_CX_Y(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq,
	int ic, int start, int n, int m, int row, int col, int row2, int col2);