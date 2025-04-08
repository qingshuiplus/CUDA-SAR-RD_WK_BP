#include "MatrixKernel.h"
//Cuda Kernel
__global__ void DLL_EXPORT mat_plus(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < matRows && y < matCols)
	{
		int pos = x + y * matRows;
		matRes[pos].x = mat1[pos].x + mat2[pos].x;
		matRes[pos].y = mat1[pos].y + mat2[pos].y;
	}
}
__global__ void DLL_EXPORT mat_dot_muiltple(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < matRows && y < matCols)
	{
		int pos = x + y * matRows;
		matRes[pos] = cuCmul(mat1[pos], mat2[pos]);
	}
}
__global__ void DLL_EXPORT mat_dot_div(
	cuDoubleComplex* mat1, cuDoubleComplex* mat2, cuDoubleComplex* matRes,
	int matRows, int matCols)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < matRows && y < matCols)
	{
		int pos = x + y * matRows;
		matRes[pos] = cuCdiv(mat1[pos], mat2[pos]);
	}
}
__global__ void DLL_EXPORT mat_muiltple(
	cuDoubleComplex* mat1, int matRows1, int matCols1,
	cuDoubleComplex* mat2, int matRows2, int matCols2,
	cuDoubleComplex* matRes)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < matRows1 && y < matCols2) {
		cuDoubleComplex sum;
		sum.x = 0;
		sum.y = 0;
		for (int i = 0; i < matCols1; i++)
		{
			sum = cuCadd(cuCmul(mat1[x + i * matRows1], mat2[i + y * matRows2]), sum);
		}
		matRes[x + y * matRows1] = sum;
	}
}
__global__ void DLL_EXPORT mat_shift(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols, int shift, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		if (dim == 0) {
			int ir = (x + shift) % rows;
			out[x + y * rows].x = input[ir + y * rows].x;
			out[x + y * rows].y = input[ir + y * rows].y;
		}
		else {
			int il = (y + shift) % cols;
			out[x + y * rows].x = input[x + il * rows].x;
			out[x + y * rows].y = input[x + il * rows].y;
		}
	}
}
__global__ void DLL_EXPORT mat_shift_double(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols, int row_shift, int col_shift) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		int ir = (x + row_shift) % rows;
		int il = (y + col_shift) % cols;
		out[x + y * rows].x = input[ir + il * rows].x;
		out[x + y * rows].y = input[ir + il * rows].y;
	}
}
__global__ void DLL_EXPORT mat_fftshift(
	cuDoubleComplex* input, cuDoubleComplex* out, int rows, int cols, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		if (dim == -1) {
			int ir = (x + (int)ceil(rows / 2.0)) % rows;
			int il = (y + (int)ceil(cols / 2.0)) % cols;
			out[x + y * rows].x = input[ir + il * rows].x;
			out[x + y * rows].y = input[ir + il * rows].y;
		}
		else if (dim == 0) {
			int ir = (x + (int)ceil(rows / 2.0)) % rows;
			out[x + y * rows].x = input[ir + y * rows].x;
			out[x + y * rows].y = input[ir + y * rows].y;
		}
		else if (dim == 1) {
			int il = (y + (int)ceil(cols / 2.0)) % cols;
			out[x + y * rows].x = input[x + il * rows].x;
			out[x + y * rows].y = input[x + il * rows].y;
		}
	}
}
__global__ void DLL_EXPORT mat_ifftshift(
	cuDoubleComplex* input, cuDoubleComplex* out, int rows, int cols, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		if (dim == -1) {
			int ir = (x + (int)floor(rows / 2.0)) % rows;
			int il = (y + (int)floor(cols / 2.0)) % cols;
			out[x + y * rows].x = input[ir + il * rows].x;
			out[x + y * rows].y = input[ir + il * rows].y;
		}
		else if (dim == 0) {
			int ir = (x + (int)floor(rows / 2.0)) % rows;
			out[x + y * rows].x = input[ir + y * rows].x;
			out[x + y * rows].y = input[ir + y * rows].y;
		}
		else if (dim == 1) {
			int il = (y + (int)floor(cols / 2.0)) % cols;
			out[x + y * rows].x = input[x + il * rows].x;
			out[x + y * rows].y = input[x + il * rows].y;
		}
	}
}
__global__ void DLL_EXPORT mat_col2row(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		out[y + x * cols].x = input[x + y * rows].x;
		out[y + x * cols].y = input[x + y * rows].y;
	}
}
__global__ void DLL_EXPORT mat_row2col(
	cuDoubleComplex* input, cuDoubleComplex* out,
	int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		out[x + y * rows].x = input[y + x * cols].x;
		out[x + y * rows].y = input[y + x * cols].y;
	}
}
__global__ void DLL_EXPORT mat_conj(
	cuDoubleComplex* mat, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows].y *= -1;
	}
}
__global__ void DLL_EXPORT mat_mean(
	cuDoubleComplex* mat, cuDoubleComplex* out, int rows, int cols, int dim) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		if (dim == -1) {
			int cnt = rows * cols;
			atomicAdd(&(out[0].x), mat[x + y * rows].x / cnt);
			atomicAdd(&(out[0].y), mat[x + y * rows].y / cnt);
		}
		else if (dim == 0) {
			int cnt = rows;
			atomicAdd(&(out[y].x), mat[x + y * rows].x / cnt);
			atomicAdd(&(out[y].y), mat[x + y * rows].y / cnt);
		}
		else
		{
			int cnt = cols;
			atomicAdd(&(out[x].x), mat[x + y * rows].x / cnt);
			atomicAdd(&(out[x].y), mat[x + y * rows].y / cnt);
		}
	}
}
__global__ void DLL_EXPORT mat_cell_plus(
	cuDoubleComplex* mat, cuDoubleComplex v, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows] = cuCadd(mat[x + y * rows], v);
	}
}
__global__ void DLL_EXPORT mat_cell_muiltple(
	cuDoubleComplex* mat, double v, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows].x *= v;
		mat[x + y * rows].y *= v;
	}
}
__global__ void DLL_EXPORT mat_cell_muiltple(
	cuDoubleComplex* mat, cuDoubleComplex v, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows] = cuCsub(mat[x + y * rows], v);
	}
}
__global__ void DLL_EXPORT mat_cell_div(
	cuDoubleComplex* mat, double v, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows].x /= v;
		mat[x + y * rows].y /= v;
	}
}
__global__ void DLL_EXPORT mat_cell_div(
	cuDoubleComplex* mat, cuDoubleComplex v, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows] = cuCdiv(mat[x + y * rows], v);
	}
}
__global__ void DLL_EXPORT cxmat_initial(
	cuDoubleComplex* mat, double r, double i, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows].x = r;
		mat[x + y * rows].y = i;
	}
}
__global__ void DLL_EXPORT mat_initial(
	double* mat, double r, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < rows && y < cols) {
		mat[x + y * rows]= r;
	}
}
__global__ void DLL_EXPORT mat_slice_row(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int start, int end) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= start && x <= end && y < col) {
		slice[x - start + y * (end - start + 1)] = mat[x + y * row];
	}
}
__global__ void DLL_EXPORT mat_slice_col(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int start, int end) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y >= start && y <= end) {
		slice[x + (y - start) * row] = mat[x + y * row];
	}
}
__global__ void DLL_EXPORT mat_set_row(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int irow) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 1 && y < col) {
		mat[irow + y * row]= slice[y];
	}
}
__global__ void DLL_EXPORT mat_set_col(cuDoubleComplex* mat, cuDoubleComplex* slice, int row, int col, int icol) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < 1) {
		mat[x + icol * row] = slice[x];
	}
}