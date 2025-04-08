
#include <iostream>
#include "ImageProcessing.h"
#define ARMA_NO_FFTW
#include <armadillo>
#include <matio.h>

using namespace std;
using namespace arma;

//std::chrono::time_point<std::chrono::system_clock> timeStart;
//std::chrono::time_point<std::chrono::system_clock> timeEnd;
//std::chrono::duration<double> elapsed_seconds;
cx_mat readEchoData(const char* file, const char* varname)
{
    mat_t* matfp;
    mat_complex_split_t* complex;
    int rows, cols;
    double* realData, * imgData;
    cx_mat data;
    matfp = Mat_Open(file, MAT_ACC_RDONLY);
    matvar_t* matData = Mat_VarRead(matfp, varname);
    if (matData == NULL) {
        fprintf(stderr, "Error reading variable from MAT file %s\n", file);
        Mat_Close(matfp);
        return data;
    }
    complex = (mat_complex_split_t*)(matData->data);
    rows = matData->dims[0];
    cols = matData->dims[1];
    data = cx_mat(rows, cols);
    realData = (double*)complex->Re;
    imgData = (double*)complex->Im;
    for (size_t irow = 0; irow < rows; irow++)
    {
        for (size_t icol = 0; icol < cols; icol++)
        {
            data(irow, icol) = { realData[icol * rows + irow] ,imgData[icol * rows + irow] };
        }
    }
    return data;
}
int testImagePorcessing(cx_mat input, int method)
{
    const double c = 299792458;
    const double fc = 5.4e9;
    const double PRF = 2588.568766;
    const double prf_real = PRF;
    const double Fs = 66.66e6;
    const double B = 60e6;
    const double Tp = 24.99e-6;
    const double wave_start = 0.0059;
    const int D = 15;
    const double lookangle = 30.77 / 180 * _Pi;
    const double Kr = -B / Tp;
    const double lambda = c / fc;
    const double Re = 6371e3;
    const double He = 7127266.845909;
    const double nearRange = 882850.962378;
    const double R_ref = 895439.506726;
    const double Vs = 7569.510962;
    const double Vg = 6748.647298;
    const double Va = sqrt(Vs * Vg) - 7;
    const double R0 = wave_start * c / 2;
    const double range_bin = c / 2 / Fs;
    const double azi_bin = Va / PRF;

    const double fdc1 = 133;
    const int Np = static_cast<int>(Tp * Fs);
    const int start_R = 3712 * 1;
    const int Nr = input.n_rows;// 4096 + Np; //5761 14848
    const int start_A = 4096;
    const int Na = input.n_cols; //5120 16384 8192
    const double R0_part = R0 + start_R * range_bin;
    const double R0_center = R0_part + Nr / 2 * range_bin;
    const double Ls = 2 * R0_center * tan(0.886 * lambda / D * _Pi * 180 / 2);
    cx_mat cpu(input), cuda(input);


    //存储结果中间变量
    mat_t* matfp;
    matvar_t* matvar;
    int nrn;
    int nan;

    size_t dims[2];
    struct mat_complex_split_t image_data_mat;
    mat Echoreal;
    mat Echoimag;
    struct mat_complex_split_t EchoData;

    std::chrono::time_point<std::chrono::system_clock> timeStart;
    std::chrono::time_point<std::chrono::system_clock> timeEnd;
    std::chrono::duration<double> elapsed_seconds;

    switch (method)
    {
    case 0: // 0.678764
        timeStart = std::chrono::system_clock::now();
        RD_Processing_CPU(cpu, lambda, Va, PRF, R0, range_bin, Kr, 1, Ls, fdc1, Fs);
        timeEnd = std::chrono::system_clock::now();
        elapsed_seconds = timeEnd - timeStart;
        cout << "RD_Processing CPU Running Seconds:" << elapsed_seconds.count() << endl;

        //结果保存CPUmat文件
        nrn = cpu.n_rows;
        nan = cpu.n_cols;
        dims[0] = nrn;
        dims[1] = nan;
        Echoreal = real(cpu);
        Echoimag = imag(cpu);

        EchoData = { Echoreal.memptr(),Echoimag.memptr() };
        matfp = Mat_CreateVer("cpuRDEcho.mat", NULL, MAT_FT_MAT73);
        if (NULL == matfp) {
            fprintf(stderr, "Error creating MAT file \"cpuRDEcho.mat\"\n");
            return EXIT_FAILURE;
        }
        matvar = Mat_VarCreate("RDEchoData", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
            MAT_F_COMPLEX);
        if (NULL == matvar) {
            fprintf(stderr, "Error creating variable for ’RDEchoData’\n");
            Mat_VarFree(matvar);
        }
        else {
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        timeStart = std::chrono::system_clock::now();
        RD_Processing(cuda, lambda, Va, PRF, R0, range_bin, Kr, 1, Ls, fdc1, Fs);
        timeEnd = std::chrono::system_clock::now();
        elapsed_seconds = timeEnd - timeStart;
        cout << "RD_Processing CUDA Running Seconds:" << elapsed_seconds.count() << endl;
        //结果保存mat文件
        nrn = cuda.n_rows;
        nan = cuda.n_cols;
        dims[0] = nrn;
        dims[1] = nan;
        Echoreal = real(cuda);
        Echoimag = imag(cuda);
        EchoData = { Echoreal.memptr(),Echoimag.memptr()};
        matfp = Mat_CreateVer("RDEcho.mat", NULL, MAT_FT_MAT73);
        if (NULL == matfp) {
            fprintf(stderr, "Error creating MAT file \"RDEcho.mat\"\n");
            return EXIT_FAILURE;
        }
        matvar = Mat_VarCreate("RDEchoData", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
            MAT_F_COMPLEX);
        if (NULL == matvar) {
            fprintf(stderr, "Error creating variable for ’RDEchoData’\n");
            Mat_VarFree(matvar);
        }
        else {
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        break;
    case 1: // 0.965689
        timeStart = std::chrono::system_clock::now();
        //WK_Processing_CPU(cpu, fc, Va, Kr, PRF, B, 3, 0.886 * lambda / D, R0_part, lambda, range_bin, Fs);
        WK_Processing_CPU_3(cpu, fc, Va, Kr, PRF, B, 3, 0.886 * lambda / D, R0_part, lambda, range_bin, Fs);
        timeEnd = std::chrono::system_clock::now();
        elapsed_seconds = timeEnd - timeStart;
        cout << "WK_Processing CPU Running Seconds:" << elapsed_seconds.count() << endl;
        //结果保存CPUmat文件
        nrn = cpu.n_rows;
        nan = cpu.n_cols;
        dims[0] = nrn;
        dims[1] = nan;
        Echoreal = real(cpu);
        Echoimag = imag(cpu);
        EchoData = { Echoreal.memptr(),Echoimag.memptr() };
        matfp = Mat_CreateVer("cpuWKEcho.mat", NULL, MAT_FT_MAT73);
        if (NULL == matfp) {
            fprintf(stderr, "Error creating MAT file \"cpuWKEcho.mat\"\n");
            return EXIT_FAILURE;
        }
        matvar = Mat_VarCreate("WKEchoData", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
            MAT_F_COMPLEX);
        if (NULL == matvar) {
            fprintf(stderr, "Error creating variable for ’RDEchoData’\n");
            Mat_VarFree(matvar);
        }
        else {
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        timeStart = std::chrono::system_clock::now();
        WK_Processing(cuda, fc, Va, Kr, PRF, B, 3, 0.886 * lambda / D, R0_part, lambda, range_bin, Fs); //注意此时对应方位长度应该调整为8才是全场景的
        timeEnd = std::chrono::system_clock::now();
        elapsed_seconds = timeEnd - timeStart;
        cout << "WK_Processing CUDA Running Seconds:" << elapsed_seconds.count() << endl;
        //结果保存mat文件
        nrn = cuda.n_rows;
        nan = cuda.n_cols;
        dims[0] = nrn;
        dims[1] = nan;
        Echoreal = real(cuda);
        Echoimag = imag(cuda);
        EchoData = { Echoreal.memptr(),Echoimag.memptr() };
        matfp = Mat_CreateVer("WKEcho.mat", NULL, MAT_FT_MAT73);
        if (NULL == matfp) {
            fprintf(stderr, "Error creating MAT file \"WKEcho.mat\"\n");
            return EXIT_FAILURE;
        }
        matvar = Mat_VarCreate("WKEchoData", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
            MAT_F_COMPLEX);
        if (NULL == matvar) {
            fprintf(stderr, "Error creating variable for ’WKEchoData’\n");
            Mat_VarFree(matvar);
        }
        else {
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        break;
    case 2: // 8.72704
        //cout << "BP_Processing start"<< endl;
        //timeStart = std::chrono::system_clock::now();
        //BP_Processing_CPU(cpu, Fs, Kr, Tp, PRF, lambda, Va, R0_part, range_bin, 0.886 * lambda / D/20, 20, 20, c / 2 / Fs / 2, Va / PRF / 2, 0);
        //timeEnd = std::chrono::system_clock::now();
        //elapsed_seconds = timeEnd - timeStart;
        //cout << "BP_Processing CPU Running Seconds:" << elapsed_seconds.count() << endl;
        ////结果保存CPUmat文件
        //nrn = cpu.n_rows;
        //nan = cpu.n_cols;
        //dims[0] = nrn;
        //dims[1] = nan;
        //Echoreal = real(cpu);
        //Echoimag = imag(cpu);
        //EchoData = { Echoreal.memptr(),Echoimag.memptr() };
        //matfp = Mat_CreateVer("cpuBPEcho.mat", NULL, MAT_FT_MAT73);
        //if (NULL == matfp) {
        //    fprintf(stderr, "Error creating MAT file \"cpuWKEcho.mat\"\n");
        //    return EXIT_FAILURE;
        //}
        //matvar = Mat_VarCreate("BPEchoData", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
        //    MAT_F_COMPLEX);
        //if (NULL == matvar) {
        //    fprintf(stderr, "Error creating variable for ’RDEchoData’\n");
        //    Mat_VarFree(matvar);
        //}
        //else {
        //    Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
        //    Mat_VarFree(matvar);
        //}
        //Mat_Close(matfp);


        timeStart = std::chrono::system_clock::now();
        BP_Processing(cuda, Fs, Kr, Tp, PRF, lambda, Va, R0_part, range_bin, 0.886 * lambda / D, 2000, 2000, c / 2 / Fs / 2, Va / PRF / 2, 0);
        timeEnd = std::chrono::system_clock::now();
        elapsed_seconds = timeEnd - timeStart;
        cout << "BP_Processing CUDA Running Seconds:" << elapsed_seconds.count() << endl;
        //结果保存mat文件
        nrn = cuda.n_rows;
        nan = cuda.n_cols;
        dims[0] = nrn;
        dims[1] = nan;
        Echoreal = real(cuda);
        Echoimag = imag(cuda);
        EchoData = { Echoreal.memptr(),Echoimag.memptr() };
        matfp = Mat_CreateVer("BPEcho_CUDA.mat", NULL, MAT_FT_MAT73);
        if (NULL == matfp) {
            fprintf(stderr, "Error creating MAT file \"BPEcho.mat\"\n");
            return EXIT_FAILURE;
        }
        matvar = Mat_VarCreate("BPEchoData_CUDA", MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
            MAT_F_COMPLEX);
        if (NULL == matvar) {
            fprintf(stderr, "Error creating variable for ’BPEchoData’\n");
            Mat_VarFree(matvar);
        }
        else {
            Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
        }
        Mat_Close(matfp);
        break;
    default:
        break;
    }
    return 0;
}
int main()
{
    vec index = { 0,1,2 };//遍历算法使用的

    //下面需要改成计算机里回波数据的存储位置
    //cx_mat input = readEchoData("D:\\pxy\\CUDASAR\\project-xd_20240410\\Matlab\\Data2\\Echo2_r14848_a16384.mat","Echo2");
    cx_mat input = readEchoData("E:\\CUDA-SAR-GMTI\\HalfEcho.mat", "EchoData");
    //cx_mat input = readEchoData("D:\\pxy\\CUDASAR\\project-xd_20240410\\Matlab\\Data\\GF3_Echo_5760_5120.mat", "ECHO33");
    //cx_mat input = readEchoData("D:\\pxy\\CUDASAR\\project-xd_20240410\\Matlab\\Data\\Echo_8192_8192.mat");
    testImagePorcessing(input, 0);
    //for (size_t i = 0; i < index.n_elem; i++)
    //{
    //    if (testImagePorcessing(input, index(i)) > 0)
    //    {
    //        cout << "Processing Method %d Test Passed" << index;
    //    }
    //}
}