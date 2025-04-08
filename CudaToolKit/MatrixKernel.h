#pragma once
#include "CudaHeader.h"
//Cuda Kernel
__global__ void DLL_EXPORT mat_plus(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols);
__global__ void DLL_EXPORT mat_dot_muiltple(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols);
__global__ void DLL_EXPORT mat_dot_div(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols);
__global__ void DLL_EXPORT mat_muiltple(
	cuDoubleComplex* mat1, int matRows1, int matCols1,
	cuDoubleComplex* mat2, int matRows2, int matCols2,
	cuDoubleComplex* matRes);
__global__ void DLL_EXPORT mat_shift(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols, int shift, int dim);
__global__ void DLL_EXPORT mat_shift_double(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols, int row_shift, int col_shift);
__global__ void DLL_EXPORT mat_fftshift(
	cuDoubleComplex* input, cuDoubleComplex* out, int rows, int cols, int dim=-1);
__global__ void DLL_EXPORT mat_ifftshift(
	cuDoubleComplex* input, cuDoubleComplex* out, int rows, int cols, int dim=-1);
__global__ void DLL_EXPORT mat_col2row(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols);
__global__ void DLL_EXPORT mat_row2col(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols);
__global__ void DLL_EXPORT mat_conj(
	cuDoubleComplex* mat, int rows, int cols);
__global__ void DLL_EXPORT mat_mean(
	cuDoubleComplex* mat, cuDoubleComplex* out, int rows, int cols, int dim);
__global__ void DLL_EXPORT mat_cell_plus(
	cuDoubleComplex* mat, cuDoubleComplex v, int rows, int cols);
__global__ void DLL_EXPORT mat_cell_muiltple(
	cuDoubleComplex* mat, double v, int rows, int cols);
__global__ void DLL_EXPORT mat_cell_div(
	cuDoubleComplex* mat, double v, int rows, int cols);
__global__ void DLL_EXPORT cxmat_initial(
	cuDoubleComplex* mat, double r, double i, int rows, int cols);
__global__ void DLL_EXPORT mat_initial(
	double* mat, double r, int rows, int cols);

__global__ void DLL_EXPORT mat_slice_row(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int start, int end);
__global__ void DLL_EXPORT mat_slice_col(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int start, int end);
__global__ void DLL_EXPORT mat_set_row(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int irow);
__global__ void DLL_EXPORT mat_set_col(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int icol);