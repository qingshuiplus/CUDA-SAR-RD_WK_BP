#pragma once
#include <CudaHeader.h>
#include <MatrixKernel.h>
#define ARMA_NO_FFTW


#include <MathKernel.h>
#include <armadillo>
#include <vector>

#define _Pi 3.14159265358979323846
using namespace arma;




struct ProcessPara {
	double Lambda;
	double PRF;
	double Fsr;
	double Rs;
	double Kr;
	double Tp;
};
struct NaciData {
	mat X;
	mat Y;
	mat Z;
};


//测试通过，计算结果一致，相比Matlab效率提升5-6倍
DLL_EXPORT void RD_Processing(cx_mat& echo, double lambda, double va, double prf, double r0, double rangeBin, double kr, double select, double ls, double fdc, double fs);
DLL_EXPORT void RD_Processing_CPU(cx_mat& echo, double lambda, double va, double prf, double r0, double rangeBin, double kr, double select, double ls, double fdc, double fs);

DLL_EXPORT void WK_Processing(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs);
DLL_EXPORT void WK_Processing_CPU(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs);

//插值是CUDA
DLL_EXPORT void WK_Processing_CPU_2(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs);

//插值是CPU
DLL_EXPORT void WK_Processing_CPU_3(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs);

DLL_EXPORT void BP_Processing(cx_mat& echo, double fs,double kr,double tp,double prf,double lambda,double va,double r0,double rangeBin,double theta,double widthRange, double widthAzi, double deltaimr, double deltaima,double aziSection);
DLL_EXPORT void BP_Processing_CPU(cx_mat& echo, double fs, double kr, double tp, double prf, double lambda, double va, double r0, double rangeBin, double theta, double widthRange, double widthAzi, double deltaimr, double deltaima, double aziSection);

DLL_EXPORT vec PCHIP(vec x, vec y, vec xq);
DLL_EXPORT vec PCHIP_DEC(vec x, vec y, vec xq);
DLL_EXPORT cx_vec PCHIP_CX(vec x, cx_vec y, vec xq);
DLL_EXPORT vec Linear(vec x, vec y, vec xq);
// 计算斜率函数
DLL_EXPORT vec compute_slopes(const vec& x, const vec& y);
// PCHIP插值函数
DLL_EXPORT vec pchip_interpolate(const vec& x, const vec& y, const vec& xi);
DLL_EXPORT cx_vec complex_pchip_interpolate(const vec& x, const cx_vec& y, const vec& xi);

class CPchip
{
public:
	~CPchip();
	vec CPchip_double(vec X, vec Y, const arma::vec xi);
	cx_vec CPchip_Complex(vec X, cx_vec Y, const arma::vec xi);
	int FindIndex(vec X, const int n, const double x);
	//int FindIndexDec(vec X, const int n, const double x);
	double ComputeDiff(const double* h, const double* delta, const int n, const int k);
private:
	double y;
	double* h;
	double* delta;
};