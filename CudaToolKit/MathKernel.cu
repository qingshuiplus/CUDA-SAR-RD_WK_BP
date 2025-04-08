#include "MathKernel.h"
#include "MatrixKernel.h"
#include "stdio.h"
using namespace std;

void DLL_EXPORT cuda_fft(cuDoubleComplex* data, int rows, int cols, int dim)
{
	cufftHandle plan;
	if (dim == 0)
		cufftPlan1d(&plan, rows, CUFFT_Z2Z, cols);
	else
		cufftPlan1d(&plan, cols, CUFFT_Z2Z, rows);
	cufftExecZ2Z(plan, data, data, CUFFT_FORWARD);
	cufftDestroy(plan);
}

void DLL_EXPORT cuda_ifft(cuDoubleComplex* data, int rows, int cols, int dim)
{
	cufftHandle plan;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(rows / threadsPerBlock.x + 1, cols / threadsPerBlock.y + 1);
	if (dim == 0)
		cufftPlan1d(&plan, rows, CUFFT_Z2Z, cols);
	else
		cufftPlan1d(&plan, cols, CUFFT_Z2Z, rows);
	cufftExecZ2Z(plan, data, data, CUFFT_INVERSE);
	cufftDestroy(plan);
	if (dim == 0)
		mat_cell_div <<<numBlocks, threadsPerBlock >>> (data, rows, rows, cols);
	else
		mat_cell_div <<<numBlocks, threadsPerBlock >>> (data, cols, rows, cols);
}

__global__ void LineSpace(double* x, double start, double end, int count) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count) {
		x[idx] = (end - start) / (count - 1) * idx + start;
	}
}

//PCHIP 三次样条插值
__device__ int PCHIP_Index_X(double* x, double v, int len, int ir, int start, int row)
{
	int index = 0;
	int iBeg = 0;
	if (v <= x[ir + (start + 1) * row]) {
		index = 0;
	}
	else if (v >= x[ir + (start + len - 2) * row]) {
		index = len - 2;
	}
	else {
		iBeg = floor((len - 1) / (x[ir + (start + len - 1) * row] - x[ir + start * row]) * (v - x[ir + start * row]));
		if (v >= x[ir + (start + iBeg) * row]) {
			for (int i = iBeg; i <= len - 3; i++) {
				if (v >= x[ir + (start + i) * row] && v <= x[ir + (start + i + 1) * row]) {
					index = i;
					break;
				}
			}
		}
		else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (v >= x[ir + (start + i) * row] && v <= x[ir + (start + i + 1) * row]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

__device__ int PCHIP_Index_Y_Dec(double* x, double v, int len, int ic, int start, int row)
{
	int index = 0;
	int iBeg = 0;
	if (v >= x[start + 1 + ic * row]) {
		index = 0;
	}
	else if (v <= x[start + len - 2 + ic * row]) {
		index = len - 2;
	}
	else {
		iBeg = floor((len - 1) / (x[start + ic * row] - x[start + len - 1 + ic * row]) * (x[start + ic * row] - v));
		if (v <= x[(start + iBeg) + ic * row]) {
			for (int i = iBeg; i <= len - 3; i++) {
				if (v <= x[start + i + ic * row] && v >= x[start + i + 1 + ic * row]) {
					index = i;
					break;
				}
			}
		}
		else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (v <= x[start + i + ic * row] && v >= x[start + i + 1 + ic * row]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

__device__ int PCHIP_Index(double* x, double v, int len)
{
	int index = 0;
	int iBeg = 0;
	if (v <= x[1]) {
		index = 0;
	}
	else if (v >= x[len - 2]) {
		index = len - 2;
	}
	else {
		iBeg = floor((len - 1) / (x[len - 1] - x[0]) * (v - x[0]));
		if (v >= x[iBeg]) {
			for (int i = iBeg; i <= len - 3; i++) {
				if (v >= x[i] && v <= x[i + 1]) {
					index = i;
					break;
				}
			}
		}
		else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (v >= x[i] && v <= x[i + 1]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

__device__ int PCHIP_Index_Dec(double* x, double v, int len)
{
	int index = 0;
	int iBeg = 0;
	if (v >= x[1]) {
		index = 0;
	}
	else if (v <= x[len - 2]) {
		index = len - 2;
	}
	else {
		iBeg = floor((len - 1) / (x[0] - x[len - 1]) * (x[0]-v));
		if (v <= x[iBeg]) {
			for (int i = iBeg; i <= len - 3; i++) {
				if (v <= x[i] && v >= x[i + 1]) {
					index = i;
					break;
				}
			}
		}
		else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (v <= x[i] && v >= x[i + 1]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

__device__ double PCHIP_Diff(double* h, double* delta, int n, int k)
{
	double diff = 0;
	if (k == 0) {
		double t = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
		if (t * delta[0] <= 0)
			diff = 0;
		else if (delta[0] * delta[1] < 0 && abs(t) > abs(3 * delta[0]))
			diff = 3 * delta[0];
		else
			diff = t;
	}
	else if (k == n - 1) {
		double t = ((2 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3]);
		if (t * delta[n - 2] <= 0)
			diff = 0;
		else if (delta[n - 2] * delta[n - 3] < 0 && abs(t) > abs(3 * delta[n - 2]))
			diff = 3 * delta[n - 2];
		else
			diff = t;
	}
	else {
		if (delta[k] * delta[k - 1] <= 0)
			diff = 0;
		else if (delta[k] * delta[k - 1] > 0 && abs(h[k] - h[k - 1]) < 1e-12)
			diff = 2 * delta[k] * delta[k - 1] / (delta[k] + delta[k - 1]);
		else {
			double w1 = 2 * h[k] + h[k - 1];
			double w2 = h[k] + 2 * h[k - 1];
			diff = delta[k] * delta[k - 1] / (w1 * delta[k] + w2 * delta[k - 1]) * (w1 + w2);
		}
	}
	return diff;
}

__global__ void PCHIP_SLOPE(double* x, double* y, double* h, double* delta, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len - 1) {
		h[idx] = x[idx + 1] - x[idx];
		delta[idx] = (y[idx + 1] - y[idx]) / h[idx];
	}
}

__global__ void PCHIP_SLOPE_CX(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int len)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len - 1) {
		h[idx] = x[idx + 1] - x[idx];
		deltar[idx] = (y[idx + 1].x - y[idx].x) / h[idx];
		deltai[idx] = (y[idx + 1].y - y[idx].y) / h[idx];
	}
}

__global__ void PCHIP_INTERP(double* x,double*y,double*h,double* delta,double* xq,double *yq,int n,int m)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index(x, xq[idx], n);
		double diff1 = PCHIP_Diff(h, delta, n, k);
		double diff2 = PCHIP_Diff(h, delta, n, k + 1);
		double a = (diff2 + diff1 - 2 * delta[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * delta[k]) / h[k];
		double c = diff1;
		double d = y[k];
		yq[idx] = a * pow(xq[idx] - x[k], 3) + b * pow(xq[idx] - x[k], 2) + c * (xq[idx] - x[k]) + d;
	}
}

__global__ void PCHIP_INTERP_Dec(double* x, double* y, double* h, double* delta, double* xq, double* yq, int n, int m)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index_Dec(x, xq[idx], n);
		double diff1 = PCHIP_Diff(h, delta, n, k);
		double diff2 = PCHIP_Diff(h, delta, n, k + 1);
		double a = (diff2 + diff1 - 2 * delta[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * delta[k]) / h[k];
		double c = diff1;
		double d = y[k];
		yq[idx] = a * pow(xq[idx] - x[k], 3) + b * pow(xq[idx] - x[k], 2) + c * (xq[idx] - x[k]) + d;
	}
}

__global__ void PCHIP_INTERP_CX(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq, int n, int m)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index(x, xq[idx], n);
		double diff1 = PCHIP_Diff(h, deltar, n, k);
		double diff2 = PCHIP_Diff(h, deltar, n, k + 1);
		double a = (diff2 + diff1 - 2 * deltar[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * deltar[k]) / h[k];
		double c = diff1;
		double d = y[k].x;
		yq[idx].x = a * pow(xq[idx] - x[k], 3) + b * pow(xq[idx] - x[k], 2) + c * (xq[idx] - x[k]) + d;
		diff1 = PCHIP_Diff(h, deltai, n, k);
		diff2 = PCHIP_Diff(h, deltai, n, k + 1);
		a = (diff2 + diff1 - 2 * deltai[k]) / h[k] / h[k];
		b = (-diff2 - 2 * diff1 + 3 * deltai[k]) / h[k];
		c = diff1;
		d = y[k].y;
		yq[idx].y = a * pow(xq[idx] - x[k], 3) + b * pow(xq[idx] - x[k], 2) + c * (xq[idx] - x[k]) + d;
	}
}

__global__ void PCHIP_SLOPE_X(double* x, double* y, double* h, double* delta, int ir, int start, int count, int row, int col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count - 1) {
		h[idx] = x[ir + (start + idx + 1) * row] - x[ir + (start + idx) * row];
		delta[idx] = (y[ir + (start + idx + 1) * row] - y[ir + (start + idx) * row]) / h[idx];
	}
}

__global__ void PCHIP_INTERP_X(double* x, double* y, double* h, double* delta, double* xq, double* yq,
	int ir, int start, int n, int m, int row, int col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index_X(x, xq[idx], n, ir, start, row);
		double diff1 = PCHIP_Diff(h, delta, n, k);
		double diff2 = PCHIP_Diff(h, delta, n, k + 1);
		double a = (diff2 + diff1 - 2 * delta[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * delta[k]) / h[k];
		double c = diff1;
		double d = y[ir + (start + k) * row];
		yq[ir + idx * row] = a * pow(xq[idx] - x[ir + (start + k) * row], 3) + b * pow(xq[idx] - x[ir + (start + k) * row], 2) + c * (xq[idx] - x[ir + (start + k) * row]) + d;
	}
}

__global__ void PCHIP_SLOPE_CX_X(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int ir, int start, int count, int row, int col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count - 1) {
		h[idx] = x[ir + (start + idx + 1) * row] - x[ir + (start + idx) * row];
		deltar[idx] = (y[ir + (start + idx + 1) * row].x - y[ir + (start + idx) * row].x) / h[idx];
		deltai[idx] = (y[ir + (start + idx + 1) * row].y - y[ir + (start + idx) * row].y) / h[idx];
	}
}

__global__ void PCHIP_INTERP_CX_X(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq,
	int ir, int start, int n, int m, int row, int col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index_X(x, xq[idx], n, ir, start, row);
		double diff1 = PCHIP_Diff(h, deltar, n, k);
		double diff2 = PCHIP_Diff(h, deltar, n, k + 1);
		double a = (diff2 + diff1 - 2 * deltar[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * deltar[k]) / h[k];
		double c = diff1;
		double d = y[ir + (start + k) * row].x;
		yq[ir + idx * row].x = a * pow(xq[idx] - x[ir + (start + k) * row], 3) + b * pow(xq[idx] - x[ir + (start + k) * row], 2) + c * (xq[idx] - x[ir + (start + k) * row]) + d;
		diff1 = PCHIP_Diff(h, deltai, n, k);
		diff2 = PCHIP_Diff(h, deltai, n, k + 1);
		a = (diff2 + diff1 - 2 * deltai[k]) / h[k] / h[k];
		b = (-diff2 - 2 * diff1 + 3 * deltai[k]) / h[k];
		c = diff1;
		d = y[ir + (start + k) * row].y;
		yq[ir + idx * row].y = a * pow(xq[idx] - x[ir + (start + k) * row], 3) + b * pow(xq[idx] - x[ir + (start + k) * row], 2) + c * (xq[idx] - x[ir + (start + k) * row]) + d;
		//printf("idx=%d,\tk=%d,\tvalue.x=%f,\tvalue.y=%f \n ", idx, k, yq[ir + idx * row].x, yq[ir + idx * row].y);
	}
}

__global__ void PCHIP_SLOPE_CX_Y(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, int ic, int start, int count, int row, int col)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count - 1) {
		h[idx] = x[(start + idx + 1) + ic * row] - x[(start + idx) + ic * row];
		deltar[idx] = (y[(start + idx + 1) + ic * row].x - y[(start + idx) + ic * row].x) / h[idx];
		deltai[idx] = (y[(start + idx + 1) + ic * row].y - y[(start + idx) + ic * row].y) / h[idx];
	}
}

__global__ void PCHIP_INTERP_CX_Y(double* x, cuDoubleComplex* y, double* h, double* deltar, double* deltai, double* xq, cuDoubleComplex* yq,
	int ic, int start, int n, int m, int row, int col, int row2, int col2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		int k = PCHIP_Index_Y_Dec(x, xq[idx], n, ic, start, row);
		double diff1 = PCHIP_Diff(h, deltar, n, k);
		double diff2 = PCHIP_Diff(h, deltar, n, k + 1);
		double a = (diff2 + diff1 - 2 * deltar[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * deltar[k]) / h[k];
		double c = diff1;
		double d = y[(start + k)+ ic * row].x;
		yq[idx + ic * row2].x = a * pow(xq[idx] - x[(start + k) + ic * row], 3) + b * pow(xq[idx] - x[(start + k) + ic * row], 2) + c * (xq[idx] - x[(start + k) + ic * row]) + d;
		diff1 = PCHIP_Diff(h, deltai, n, k);
		diff2 = PCHIP_Diff(h, deltai, n, k + 1);
		a = (diff2 + diff1 - 2 * deltai[k]) / h[k] / h[k];
		b = (-diff2 - 2 * diff1 + 3 * deltai[k]) / h[k];
		c = diff1;
		d = y[(start + k) + ic * row].y;
		yq[idx + ic * row2].y = a * pow(xq[idx] - x[(start + k) + ic * row], 3) + b * pow(xq[idx] - x[(start + k) + ic * row], 2) + c * (xq[idx] - x[(start + k) + ic * row]) + d;
		//printf("idx=%d,\tk=%d,\tvalue.x=%f,\tvalue.y=%f \n ", idx, k, yq[idx + ic * row2].x, yq[idx + ic * row2].y);
	}
}

//线性插值
__global__ void LINEAR_INTERP(double* x, double* y, double* xq, double* yq, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		float xqi = xq[idx];
		int i = 0;
		while (i < n - 1 && x[i + 1] < xqi)
			i++;
		float x0 = x[i];
		float x1 = x[i + 1];
		float y0 = y[i];
		float y1 = y[i + 1];
		yq[idx] = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
	}
}
__global__ void LINEAR_INTERP_CX(double* x, cuDoubleComplex* y, double* xq, cuDoubleComplex* yq, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		float xqi = xq[idx];
		int i = 0;
		while (i < n - 1 && x[i + 1] < xqi)
			i++;
		float x0 = x[i];
		float x1 = x[i + 1];
		float y0 = y[i].x;
		float y1 = y[i + 1].x;
		yq[idx].x = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
		y0 = y[i].y;
		y1 = y[i + 1].y;
		yq[idx].y = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
	}
}
__global__ void LINEAR_INTERP_CX_MAT(double* x, cuDoubleComplex* y, double* xq, cuDoubleComplex* yq, int n, int m, int icol, int row, int col) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < m) {
		float xqi = xq[idx];
		int i = 0;
		while (i < n - 1 && x[i + 1] < xqi)
			i++;
		float x0 = x[i];
		float x1 = x[i + 1];
		float y0 = y[i + icol * row].x;
		float y1 = y[i + icol * row + 1].x;
		yq[idx].x = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
		y0 = y[i + icol * row].y;
		y1 = y[i + icol * row + 1].y;
		yq[idx].y = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
	}
}
