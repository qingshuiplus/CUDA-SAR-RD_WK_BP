#include <matio.h>
#include <armadillo>
using namespace std;
using namespace arma;
cx_mat readEchoData(const char* file, const char* varname)
{
    mat_t* matfp;
    mat_complex_split_t* complex;
    int rows, cols;
    double* realData, * imgData;
    cx_mat data;
    matfp = Mat_Open(file, MAT_ACC_RDONLY);
    matvar_t* matData = Mat_VarRead(matfp, varname);
    if (matData== NULL) {
        fprintf(stderr, "Error open file %s\n", file);
        return data;
    }
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

int main()
{
    cx_mat input = readEchoData("E:\\CUDA-SAR-GMTI\\HalfEcho.mat", "EchoData");
    if (input.size() == NULL) {
        fprintf(stderr, "Error reading matio\n");
        return -1;
    }

}