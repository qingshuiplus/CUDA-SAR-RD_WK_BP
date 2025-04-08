#include "ImageProcessing.h"
#include "matio.h"


std::chrono::time_point<std::chrono::system_clock> timeStart;
std::chrono::time_point<std::chrono::system_clock> timeEnd;
std::chrono::duration<double> elapsed_seconds;
//RD Kernel Functions
inline int fix(double x) {
	return (x < 0) ? std::ceil(x) : std::floor(x);
}
int Savecx_MatByMatio(cx_mat data, char* filename, char* valname)
{
	//结果保存mat文件
	mat_t* matfp;
	matvar_t* matvar;
	int nrn = data.n_rows;
	int nan = data.n_cols;
	size_t dims[2];
	mat Echoreal;
	mat Echoimag;
	dims[0] = nrn;
	dims[1] = nan;
	Echoreal = real(data);
	Echoimag = imag(data);
	struct mat_complex_split_t EchoData;
	EchoData = { Echoreal.memptr(),Echoimag.memptr() };
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT73);
	if (NULL == matfp) {
		fprintf(stderr, "Error creating MAT file \"RDEcho.mat\"\n");
		return EXIT_FAILURE;
	}
	matvar = Mat_VarCreate(valname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, &EchoData,
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
}

int SavedatByMatio(dmat data, char* filename, char* valname)
{
	//结果保存mat文件
	mat_t* matfp;
	matvar_t* matvar;
	int nrn = data.n_rows;
	int nan = data.n_cols;
	size_t dims[2];
	dims[0] = nrn;
	dims[1] = nan;
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT73);
	if (NULL == matfp) {
		fprintf(stderr, "Error creating MAT file \"RDEcho.mat\"\n");
		return EXIT_FAILURE;
	}
	matvar = Mat_VarCreate(valname, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, data.memptr(),0);
	if (NULL == matvar) {
		fprintf(stderr, "Error creating variable for ’RDEchoData’\n");
		Mat_VarFree(matvar);
	}
	else {
		Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
		Mat_VarFree(matvar);
	}
	Mat_Close(matfp);
}
cx_mat readcx_matData(const char* file, const char* val)
{
	mat_t* matfp;
	mat_complex_split_t* complex;
	int rows, cols;
	double* realData, * imgData;
	cx_mat data;
	matfp = Mat_Open(file, MAT_ACC_RDONLY);
	matvar_t* matData = Mat_VarRead(matfp, val);
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

dmat readdmatData(const char* file, const char* val)
{
	mat_t* matfp;
	double* _double;
	int rows, cols;
	dmat data;
	matfp = Mat_Open(file, MAT_ACC_RDONLY);
	matvar_t* matData = Mat_VarRead(matfp, val);
	if (matData == NULL) {
		fprintf(stderr, "Error reading variable from MAT file %s\n", file);
		Mat_Close(matfp);
		return data;
	}
	_double = (double*)(matData->data);
	rows = matData->dims[0];
	cols = matData->dims[1];
	data = dmat(rows, cols);
	for (size_t irow = 0; irow < rows; irow++)
	{
		for (size_t icol = 0; icol < cols; icol++)
		{
			data(irow, icol) = _double[icol * rows + irow];
		}
	}
	return data;
}

__global__ void F_Initial(cuDoubleComplex* mat, int row, int col, int dim, int count, double start, double end,double subValue)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		if (dim == 0) {
			mat[x].x = (end - start) / (count - 1) * x + start;
			mat[x].x *= subValue;
			mat[x].y = 0;
		}
		else {
			mat[y].x = (end - start) / (count - 1) * y + start;
			mat[y].x *= subValue;
			mat[y].y = 0;
		}
	}
}
__global__ void T_Initial(cuDoubleComplex* mat, int row, int col, int dim, int count, double start, double end,double subValue,double plusValue)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		if (dim == 0) {
			mat[x].x = (end - start) / (count - 1) * x + start;
			mat[x].x *= subValue;
			mat[x].x += plusValue;
			mat[x].y = 0;
		}
		else {
			mat[y].x = (end - start) / (count - 1) * y + start;
			mat[y].x *= subValue;
			mat[y].x += plusValue;
			mat[y].y = 0;
		}
	}
}
__global__ void SinCosInitial(cuDoubleComplex* mat, int row, int col, double div, cuDoubleComplex* sin, cuDoubleComplex* cos)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		sin[x+y*row].x = mat[x + y * row].x / div;
		sin[x + y * row].y = 0;
		cos[x + y * row].x = sqrt(1 - sin[x + y * row].x * sin[x + y * row].x);
	}
}
__global__ void GamaInitial(cuDoubleComplex* sin, cuDoubleComplex* cos, int row, int col,double kr,double rs,double lambda,double c,cuDoubleComplex* gama) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		double t1 = sin[x + y * row].x;
		double t2 = cos[x + y * row].x;
		gama[x + y * row].x = 1 / kr - rs * 2 * lambda * t1 * t1 / (c * c) / (t2 * t2 * t2);
	}
}
__global__ void H21Initial(cuDoubleComplex* fast, cuDoubleComplex* slow, int row,int col,double c, double fa_M, double rs,cuDoubleComplex* h21) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		cuDoubleComplex tfast = make_cuDoubleComplex(0, 2 * _Pi * rs);
		tfast = cuCmul(fast[x], tfast);
		tfast.x /= c; tfast.y /= c;
		cuDoubleComplex tslow = slow[y];
		tslow.x /= fa_M; tslow.y /= fa_M;
		tslow = cuCmul(tslow, tslow);
		cuDoubleComplex val = cuCmul(tfast, tslow);
		double exp_real = exp(val.x) * cos(val.y);
		double exp_imag = exp(val.x) * sin(val.y);
		h21[x + y * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void H22Initial(cuDoubleComplex* fast, cuDoubleComplex* gama, int row, int col,cuDoubleComplex* h22) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		cuDoubleComplex tfast = cuCmul(fast[x], fast[x]);
		tfast = cuCmul(make_cuDoubleComplex(0, _Pi), tfast);
		cuDoubleComplex val = cuCmul(tfast, gama[y]);
		double exp_real = exp(val.x) * cos(val.y);
		double exp_imag = exp(val.x) * sin(val.y);
		h22[x + y * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void H3Initial(cuDoubleComplex* r0, cuDoubleComplex* slow, int row, int col, double va,double fa_M,cuDoubleComplex* h3) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		cuDoubleComplex t1 = make_cuDoubleComplex(0, 2*_Pi/va);
		t1 = cuCmul(r0[x], t1);
		cuDoubleComplex t2 = make_cuDoubleComplex(sqrt(fa_M * fa_M - slow[y].x * slow[y].x), 0);
		cuDoubleComplex val = cuCmul(t1, t2);
		double exp_real = exp(val.x) * cos(val.y);
		double exp_imag = exp(val.x) * sin(val.y);
		h3[x + y * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void EchoInitial(cuDoubleComplex* echo, cuDoubleComplex* tslow,int row, int col, double fdc) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		cuDoubleComplex t1 = make_cuDoubleComplex(0, -2*_Pi*fdc);
		cuDoubleComplex val = cuCmul(t1, tslow[y]);
		double exp_real = exp(val.x) * cos(val.y);
		double exp_imag = exp(val.x) * sin(val.y);
		echo[x + y * row] = cuCmul(echo[x + y * row],make_cuDoubleComplex(exp_real, exp_imag));
	}
}
void RD_Processing(cx_mat& echo, double lambda, double va, double prf, double r0, double rangeBin, double kr, double select, double ls, double fdc, double fs) {
	timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double fa_M = 2 * va / lambda;
	double Rs = r0 + ceil(Nr / 2.0) * rangeBin;
	cuDoubleComplex *Echo,*R0, * Fslow, * Ffast, * Tfast, * Tslow, * sin_theta, * cos_theta, * gama_e, * H21, * H22, * H3;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	size_t nr_na_size = Nr * Na * sizeof(double2);
	size_t nr_size = Nr * sizeof(double2);
	size_t na_size = Na * sizeof(double2);
	cudaMalloc((void**)&Echo, nr_na_size);
	cudaMalloc((void**)&H21, nr_na_size);
	cudaMalloc((void**)&H22, nr_na_size);
	cudaMalloc((void**)&H3, nr_na_size);

	cudaMalloc((void**)&R0, nr_size);
	cudaMalloc((void**)&Ffast, nr_size);
	cudaMalloc((void**)&Tfast, nr_size);

	cudaMalloc((void**)&Fslow, na_size);
	cudaMalloc((void**)&Tslow, na_size);
	cudaMalloc((void**)&sin_theta, na_size);
	cudaMalloc((void**)&cos_theta, na_size);
	cudaMalloc((void**)&gama_e, na_size);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cudaMemcpyAsync(Echo, echo.mem, nr_na_size, cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	T_Initial <<<numBlocks, threadsPerBlock >>> (R0, Nr, 1, 0, Nr, 0, Nr - 1, rangeBin, r0);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Fslow, 1, Na, 1, Na, -Na / 2.0, Na / 2.0 - 1, prf / Na);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Ffast, Nr, 1, 0, Nr, -Nr / 2.0, Nr / 2.0 - 1, fs / Nr);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Tslow, 1, Na, 1, Na, -Na / 2.0, Na / 2.0 - 1, 1 / prf);
	T_Initial <<<numBlocks, threadsPerBlock >>> (Tfast, Nr, 1, 0, Nr, 0, Nr - 1, 4 * rangeBin / c, 2 * r0 / c);
	SinCosInitial <<<numBlocks, threadsPerBlock >>> (Fslow, 1, Na, fa_M, sin_theta, cos_theta);
	GamaInitial <<<numBlocks, threadsPerBlock >>> (sin_theta, cos_theta, 1, Na, kr, Rs, lambda, c, gama_e);
	H21Initial <<<numBlocks, threadsPerBlock >>> (Ffast, Fslow, Nr, Na, c, fa_M, Rs, H21);
	H22Initial <<<numBlocks, threadsPerBlock >>> (Ffast, gama_e, Nr, Na, H22);
	H3Initial <<<numBlocks, threadsPerBlock >>> (R0, Fslow, Nr, Na, va, fa_M, H3);
	EchoInitial <<<numBlocks, threadsPerBlock >>> (Echo, Tslow , Nr, Na, fdc);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	if (select == 1) {
		cuDoubleComplex* temp, * singnal_rf_a, * singnal_rf_af;
		cudaMalloc((void**)&temp, nr_na_size);
		cudaMalloc((void**)&singnal_rf_a, nr_na_size);
		cudaMalloc((void**)&singnal_rf_af, nr_na_size);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (Echo, temp, Nr, Na);
		cuda_fft(temp, Nr, Na, 0);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		mat_col2row <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		cuda_fft(singnal_rf_af, Nr, Na, 1);
		mat_row2col <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, temp, Nr, Na);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, H21, temp, Nr, Na);
		mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (temp, H22, temp, Nr, Na);
		timeEnd = std::chrono::system_clock::now();
		elapsed_seconds = timeEnd - timeStart;
		printf("\"FFT\":\t%f\n", elapsed_seconds);
		timeStart = std::chrono::system_clock::now();
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		cuda_ifft(singnal_rf_af, Nr, Na, 0);
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, temp, Nr, Na);
		mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (temp, H3, singnal_rf_a, Nr, Na);
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		mat_col2row <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		cuda_ifft(singnal_rf_a, Nr, Na, 1);
		mat_row2col <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		timeEnd = std::chrono::system_clock::now();
		elapsed_seconds = timeEnd - timeStart;
		printf("\"IFFT\":\t%f\n", elapsed_seconds);
		timeStart = std::chrono::system_clock::now();
		cudaMemcpyAsync(echo.memptr(), singnal_rf_a, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		timeEnd = std::chrono::system_clock::now();
		elapsed_seconds = timeEnd - timeStart;
		printf("\"数据传输\":\t%f\n", elapsed_seconds);
		timeStart = std::chrono::system_clock::now();
		cudaFree(temp); cudaFree(singnal_rf_a); cudaFree(singnal_rf_af);
	}
	else if(select==2){
		//待定
	}
	cudaFree(Echo); cudaFree(R0); cudaFree(Fslow); cudaFree(Ffast); cudaFree(Tslow); cudaFree(Tfast);
	cudaFree(sin_theta); cudaFree(cos_theta); cudaFree(gama_e); cudaFree(H21); cudaFree(H22); cudaFree(H3);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);
}
void RD_Processing_CPU(cx_mat& echo, double lambda, double va, double prf, double r0, double rangeBin, double kr, double select, double ls, double fdc, double fs) {
	timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double fa_M = 2 * va / lambda;
	double Rs = r0 + ceil(Nr / 2.0) * rangeBin;
	cuDoubleComplex* Echo, * R0, * Fslow, * Ffast, * Tfast, * Tslow, * sin_theta, * cos_theta, * gama_e, * H21, * H22, * H3;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	size_t nr_na_size = Nr * Na * sizeof(double2);
	size_t nr_size = Nr * sizeof(double2);
	size_t na_size = Na * sizeof(double2);
	cx_mat cx_temp(Nr, Na);
	cudaMalloc((void**)&Echo, nr_na_size);
	cudaMalloc((void**)&H21, nr_na_size);
	cudaMalloc((void**)&H22, nr_na_size);
	cudaMalloc((void**)&H3, nr_na_size);

	cudaMalloc((void**)&R0, nr_size);
	cudaMalloc((void**)&Ffast, nr_size);
	cudaMalloc((void**)&Tfast, nr_size);

	cudaMalloc((void**)&Fslow, na_size);
	cudaMalloc((void**)&Tslow, na_size);
	cudaMalloc((void**)&sin_theta, na_size);
	cudaMalloc((void**)&cos_theta, na_size);
	cudaMalloc((void**)&gama_e, na_size);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cudaMemcpyAsync(Echo, echo.mem, nr_na_size, cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	T_Initial <<<numBlocks, threadsPerBlock >>> (R0, Nr, 1, 0, Nr, 0, Nr - 1, rangeBin, r0);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Fslow, 1, Na, 1, Na, -Na / 2.0, Na / 2.0 - 1, prf / Na);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Ffast, Nr, 1, 0, Nr, -Nr / 2.0, Nr / 2.0 - 1, fs / Nr);
	F_Initial <<<numBlocks, threadsPerBlock >>> (Tslow, 1, Na, 1, Na, -Na / 2.0, Na / 2.0 - 1, 1 / prf);
	T_Initial <<<numBlocks, threadsPerBlock >>> (Tfast, Nr, 1, 0, Nr, 0, Nr - 1, 4 * rangeBin / c, 2 * r0 / c);
	SinCosInitial <<<numBlocks, threadsPerBlock >>> (Fslow, 1, Na, fa_M, sin_theta, cos_theta);
	GamaInitial <<<numBlocks, threadsPerBlock >>> (sin_theta, cos_theta, 1, Na, kr, Rs, lambda, c, gama_e);
	H21Initial <<<numBlocks, threadsPerBlock >>> (Ffast, Fslow, Nr, Na, c, fa_M, Rs, H21);
	H22Initial <<<numBlocks, threadsPerBlock >>> (Ffast, gama_e, Nr, Na, H22);
	H3Initial <<<numBlocks, threadsPerBlock >>> (R0, Fslow, Nr, Na, va, fa_M, H3);
	EchoInitial <<<numBlocks, threadsPerBlock >>> (Echo, Tslow, Nr, Na, fdc);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	if (select == 1) {
		//cuDoubleComplex* temp, * singnal_rf_a, * singnal_rf_af;
		//cudaMalloc((void**)&temp, nr_na_size);
		//cudaMalloc((void**)&singnal_rf_a, nr_na_size);
		//cudaMalloc((void**)&singnal_rf_af, nr_na_size);
		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (Echo, temp, Nr, Na);
		//cuda_fft(temp, Nr, Na, 0);
		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		//mat_col2row <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		//cuda_fft(singnal_rf_af, Nr, Na, 1);
		//mat_row2col <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, temp, Nr, Na);
		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, H21, temp, Nr, Na);
		//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (temp, H22, temp, Nr, Na);
		cudaMemcpy(echo.memptr(), Echo, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		echo = arma::fft(echo);
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		echo = arma::fft(echo.t()).t();
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		cudaMemcpy(cx_temp.memptr(), H21, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		echo = echo % cx_temp;
		cudaMemcpy(cx_temp.memptr(), H22, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		echo = echo % cx_temp;
		timeEnd = std::chrono::system_clock::now();
		elapsed_seconds = timeEnd - timeStart;
		printf("\"FFT\":\t%f\n", elapsed_seconds);
		timeStart = std::chrono::system_clock::now();
		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_af, Nr, Na);
		//cuda_ifft(singnal_rf_af, Nr, Na, 0);
		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_af, temp, Nr, Na);
		//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (temp, H3, singnal_rf_a, Nr, Na);
		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		//mat_col2row <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		//cuda_ifft(singnal_rf_a, Nr, Na, 1);
		//mat_row2col <<<numBlocks, threadsPerBlock >>> (singnal_rf_a, temp, Nr, Na);
		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (temp, singnal_rf_a, Nr, Na);
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		echo = arma::ifft(echo);
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		cudaMemcpy(cx_temp.memptr(), H3, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		echo = echo % cx_temp;
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		echo = arma::ifft(echo.t()).t();
		echo = arma::shift(echo, Nr / 2, 0);
		echo = arma::shift(echo, Na / 2, 1);
		timeEnd = std::chrono::system_clock::now();
		elapsed_seconds = timeEnd - timeStart;
		printf("\"IFFT\":\t%f\n", elapsed_seconds);
		timeStart = std::chrono::system_clock::now();
		//cudaMemcpyAsync(echo.memptr(), singnal_rf_a, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
		//timeEnd = std::chrono::system_clock::now();
		//elapsed_seconds = timeEnd - timeStart;
		//printf("\"数据传输\":\t%f\n", elapsed_seconds);
		//timeStart = std::chrono::system_clock::now();
		//cudaFree(temp); cudaFree(singnal_rf_a); cudaFree(singnal_rf_af);
	}
	else if (select == 2) {
		//待定
	}
	cudaFree(Echo); cudaFree(R0); cudaFree(Fslow); cudaFree(Ffast); cudaFree(Tslow); cudaFree(Tfast);
	cudaFree(sin_theta); cudaFree(cos_theta); cudaFree(gama_e); cudaFree(H21); cudaFree(H22); cudaFree(H3);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);
}
//WK Kernel Functions
__global__ void FrInitial(cuDoubleComplex* mat, int row, int col, int dim, int count, double start, double end, double dec,double nr,double fs)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		if (dim == 0) {
			mat[x].x = (end - start) / (count - 1) * x + start;
			mat[x].x -= dec;
			mat[x].x = mat[x].x/nr*fs;
			mat[x].y = 0;
		}
		else {
			mat[y].x = (end - start) / (count - 1) * y + start;
			mat[y].x -= dec;
			mat[y].x = mat[y].x / nr * fs;
			mat[y].y = 0;
		}
	}
}
__global__ void TfastInitial(cuDoubleComplex* fast, cuDoubleComplex* rb, int row, int col,double c) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		fast[x + y * row].x = 2 * rb[x].x / c;
	}
}
__global__ void TslowInitial(cuDoubleComplex* slow, int row, int col, double start, double end, int count, double prf) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		slow[x + y * row].x = ((end - start) / (count - 1) * y + start) / prf;
	}
}
__global__ void RrefInitial(cuDoubleComplex* rref, cuDoubleComplex* xt, double yt, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		rref[pos].x = sqrt(xt[pos].x * xt[pos].x + yt * yt);
	}
}
__global__ void TauhInitial(cuDoubleComplex* rref, cuDoubleComplex* xt, double yt, double c, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		rref[pos].x = 2 * sqrt(xt[pos].x * xt[pos].x + yt * yt) / c;
	}
}
__global__ void YredInitial(cuDoubleComplex* yref, cuDoubleComplex* tauh, cuDoubleComplex* tfast, double fc, double kr, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		double t1 = -2 * _Pi * fc * tauh[pos].x + _Pi * kr * (tfast[pos].x - tauh[pos].x) * (tfast[pos].x - tauh[pos].x);
		double exp_real = cos(t1);
		double exp_imag = sin(t1);
		yref[x + y * row] = make_cuDoubleComplex(exp_real, -1 * exp_imag);
	}
}
__global__ void RVPInitial(cuDoubleComplex* rvp, cuDoubleComplex* fr, int row, int col, double kr) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		double t1 = -1 * _Pi * fr[x].x * fr[x].x / kr;
		double exp_real = cos(t1);
		double exp_imag = sin(t1);
		rvp[x + y * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void KrInitial(cuDoubleComplex* kr, cuDoubleComplex* fast, cuDoubleComplex* tauh, int row, int col, double c,double fc,double Kr){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		kr[pos].x = 4 * _Pi / c * (fc + Kr * (fast[pos].x - tauh[pos].x));
		kr[pos].y = 0;
	}
}
__global__ void KyInitial(double* ky, cuDoubleComplex* kr, cuDoubleComplex* radar, double yt, int row, int col)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		ky[pos] = yt * kr[pos].x / radar[pos].x;
	}
}
__global__ void KxInitial(double* kx, cuDoubleComplex* xt, cuDoubleComplex* kr, cuDoubleComplex* radar, int row, int col)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		kx[pos] = xt[pos].x * kr[pos].x / radar[pos].x;
	}
}
__global__ void KxTempInitial(cuDoubleComplex* kxTemp, cuDoubleComplex* kr, int index, double kyDown, int row, int col)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		kxTemp[pos].x = sqrt(kr[pos].x * kr[pos].x - kyDown * kyDown);
		kxTemp[pos].y = 0;
	}
}
__global__ void kyResampleInitial(double* kyResample, double start, double step, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		kyResample[x + y * row] = start + step * x;
	}
}
__global__ void kxResampleInitial(double* kxResample, double start, double end, double dkx, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		kxResample[x + y * row] = ((end - start) / (col - 1) * y + start) * dkx;
	}
}
__global__ void RangeInitial(double* res, cuDoubleComplex* mat, double down, double up, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		if (mat[pos].x >= down && mat[pos].x <= up)
			res[pos] = 1;
		else
			res[pos] = 0;
	}
}
__global__ void RangeInitial(double* res, double* mat, double down, double up, int row, int col) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row && y < col)
	{
		int pos = x + y * row;
		if (mat[pos] >= down && mat[pos] <= up)
			res[pos] = 1;
		else
			res[pos] = 0;
	}
}
__global__ void FillReSample(cuDoubleComplex* mat, cuDoubleComplex* out, int left, int top, int row1, int col1, int row2, int col2) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < row2 && y < col2)
	{
		if (x>=top&&x<(row1+top)&&y>=left&&y<(col1+left)) {
			out[x + y * row2].x = mat[x - top + (y - left) * row1].x;
			out[x + y * row2].y = mat[x - top + (y - left) * row1].y;
		}
		else {
			out[x + y * row2].x = 0;
			out[x + y * row2].y = 0;
		}
	}
}
vec compute_slopes(const arma::vec& x, const arma::vec& y) {
	int n = x.n_elem;
	arma::vec h = arma::diff(x);
	arma::vec delta = arma::diff(y) / h;
	arma::vec m(n);

	// 内部点的斜率
	for (int i = 1; i < n - 1; ++i) {
		if (delta(i - 1) * delta(i) > 0) {
			m(i) = (delta(i - 1) + delta(i)) / 2;
		}
		else {
			m(i) = 0.0;
		}
	}

	// 边界斜率
	m(0) = delta(0);
	m(n - 1) = delta(n - 2);

	return m;
}
vec pchip_interpolate(const arma::vec& x, const arma::vec& y, const arma::vec& xi) {
	int n = x.n_elem;
	arma::vec yi(xi.n_elem);
	arma::vec m = compute_slopes(x, y);

	for (size_t k = 0; k < xi.n_elem; ++k) {
		if (xi(k) <= x(0)) {
			yi(k) = y(0);
		}
		else if (xi(k) >= x(n - 1)) {
			yi(k) = y(n - 1);
		}
		else {
			// 找到区间
			int i = arma::as_scalar(arma::find(x <= xi(k), 1, "last"));
			double h = x(i + 1) - x(i);
			double t = (xi(k) - x(i)) / h;
			double h00 = (1 + 2 * t) * std::pow(1 - t, 2);
			double h10 = t * std::pow(1 - t, 2);
			double h01 = t * t * (3 - 2 * t);
			double h11 = t * t * (t - 1);

			yi(k) = h00 * y(i) + h10 * h * m(i) + h01 * y(i + 1) + h11 * h * m(i + 1);
		}
	}

	return yi;
}
// 对复数进行PCHIP插值
cx_vec complex_pchip_interpolate(const arma::vec& x, const arma::cx_vec& y, const arma::vec& xi) {
	arma::vec y_real = arma::real(y);
	arma::vec y_imag = arma::imag(y);

	arma::vec yi_real = pchip_interpolate(x, y_real, xi);
	arma::vec yi_imag = pchip_interpolate(x, y_imag, xi);

	arma::cx_vec yi(xi.n_elem);
	for (int i = 0; i < xi.n_elem; ++i) {
		yi(i) = std::complex<double>(yi_real(i), yi_imag(i));
	}
	return yi;
}

vec CPchip::CPchip_double(vec X, vec Y, const arma::vec xi)
{
	int n = X.n_elem;
	arma::vec yi(xi.n_elem);
	for (int j = 0; j < xi.n_elem; j++)
	{
		int k = FindIndex(X, n, xi[j]);
		h = new double[n] {};
		delta = new double[n] {};
		for (int i = 0; i <= n - 2; i++) {
			h[i] = X[i + 1] - X[i]; //节点上增量
			if (h[i] < 1e-12) {
				//cout << "X[" << i + 1 << "]=" << X[i + 1] << "不递增" << endl;
				abort();
			}
			delta[i] = (Y[i + 1] - Y[i]) / h[i]; //节点上差商，向后差分法, 注意点除
		}
		double diff1 = ComputeDiff(h, delta, n, k);
		double diff2 = ComputeDiff(h, delta, n, k + 1);

		double a = (diff2 + diff1 - 2 * delta[k]) / h[k] / h[k];
		double b = (-diff2 - 2 * diff1 + 3 * delta[k]) / h[k];
		double c = diff1;
		double d = Y[k];
		y = a * pow(xi[j] - X[k], 3) + b * pow(xi[j] - X[k], 2) + c * (xi[j] - X[k]) + d;
		yi[j] = y;
		delete[] h;
		h = nullptr;
		delete[] delta;
		delta = nullptr;
	}

	return yi;
}
cx_vec CPchip::CPchip_Complex(vec X, cx_vec Y, const arma::vec xi)
{
	arma::vec y_real = arma::real(Y);
	arma::vec y_imag = arma::imag(Y);

	arma::vec yi_real = CPchip_double(X, y_real, xi);
	arma::vec yi_imag = CPchip_double(X, y_imag, xi);

	arma::cx_vec yi(yi_real, yi_imag);
	//for (int i = 0; i < xi.n_elem; ++i) {
	//	yi(i) = std::complex<double>(yi_real(i), yi_imag(i));
	//}
	return yi;
}
CPchip::~CPchip()
{
	delete[] h;
	h = nullptr;
	delete[] delta;
	delta = nullptr;
}
int CPchip::FindIndex(vec X, const int n, const double x)
{
	/* 找到插值位置 */
	if (n < 3) {
		cout << "pchip要求三个及以上节点!" << endl;
		abort();
	}
	int index = 0;
	int iBeg = 0;
	if (x <= X[1]) {
		index = 0;
		if (x < X[0]) {
			//cout << "x=" << x << ",pchip外插，继续但不保证正确性" << endl;
		}
	}
	else if (x >= X[n - 2]) {
		index = n - 2;
		if (x > X[n + 1]) {
			//cout << "x=" << x << ",pchip外插，继续但不保证正确性" << endl;
		}
	}
	else {
		iBeg = floor((n - 1) / (X[n - 1] - X[0]) * (x - X[0]));
		if (x >= X[iBeg]) {
			for (int i = iBeg; i <= n - 3; i++) {
				if (x >= X[i] && x <= X[i + 1]) {
					index = i;
					break;
				}
			}
		}
		else {
			for (int i = iBeg - 1; i >= 1; i--) {
				if (x >= X[i] && x <= X[i + 1]) {
					index = i;
					break;
				}
			}
		}
	}
	return index;
}

double CPchip::ComputeDiff(const double* h, const double* delta, const int n, const int k)
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

void WK_Processing(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs) {
	//timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double R0_center = r0 + Nr / 2.0 * rangeBin;
	double x_scence_center = 0;
	double Ls = 2 * R0_center * tan(theta / 2);
	int Na_Ls = fix(Ls / va / 2 * prf) * 2;
	double Na_select = aziWidth * Na_Ls + Na_Ls;
	echo = echo(span::all, span(Na / 2 - Na_select / 2 - 1, Na / 2 + Na_select / 2 - 2));
	Na = Na_select;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	cuDoubleComplex* Rb, * fr, * Tfast, * Tslow, * Tauh, * Yref, * Ydechirp, * echoPart, * cxtemp, * RVP;
	size_t nr_na_size = Nr * Na * sizeof(double2);
	size_t nr_size = Nr * sizeof(double2);
	size_t na_size = Na * sizeof(double2);

	cudaMalloc((void**)&Rb, nr_size);
	cudaMalloc((void**)&fr, nr_size);
	cudaMalloc((void**)&Tfast, nr_na_size);
	cudaMalloc((void**)&Tslow, nr_na_size);
	cudaMalloc((void**)&Tauh, nr_na_size);
	cudaMalloc((void**)&Yref, nr_na_size);
	cudaMalloc((void**)&Ydechirp, nr_na_size);
	cudaMalloc((void**)&echoPart, nr_na_size);
	cudaMalloc((void**)&cxtemp, nr_na_size);
	cudaMalloc((void**)&RVP, nr_na_size);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//cudaMalloc((void**)&Rb, Nr * sizeof(double2));
	//cudaMalloc((void**)&fr, Nr * sizeof(double2));
	//cudaMalloc((void**)&Tfast, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Tslow, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Tauh, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Yref, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Ydechirp, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&echoPart, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&cxtemp, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&RVP, Nr * Na * sizeof(double2));
	T_Initial <<<numBlocks, threadsPerBlock >>> (Rb, Nr, 1, 0, Nr, -Nr / 2.0 + 1, Nr / 2.0, rangeBin, R0_center);
	FrInitial <<<numBlocks, threadsPerBlock >>> (fr, Nr, 1, 0, Nr, 0, Nr - 1, fix(Nr / 2.0), Nr, fs);
	TfastInitial <<<numBlocks, threadsPerBlock >>> (Tfast, Rb, Nr, Na, c);
	TslowInitial <<<numBlocks, threadsPerBlock >>> (Tslow, Nr, Na, -Na / 2.0, Na / 2.0 - 1, Na, prf);
	mat_cell_muiltple <<<numBlocks, threadsPerBlock >>> (Tslow, va, Nr, Na);
	mat_cell_plus <<<numBlocks, threadsPerBlock >>> (Tslow, make_cuDoubleComplex(-1 * x_scence_center, 0), Nr, Na);//tslow=>xt
	TauhInitial <<<numBlocks, threadsPerBlock >>> (Tauh, Tslow, R0_center, c, Nr, Na);
	YredInitial <<<numBlocks, threadsPerBlock >>> (Yref, Tauh, Tfast, fc, kr, Nr, Na);
	//cudaMemcpyAsync(echoPart, echo.mem, Nr * Na * sizeof(double2), cudaMemcpyHostToDevice);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	cudaMemcpyAsync(echoPart, echo.mem, nr_na_size, cudaMemcpyHostToDevice);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (echoPart, Yref, Ydechirp, Nr, Na);
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (Ydechirp, cxtemp, Nr, Na, 0);
	cuda_fft(cxtemp, Nr, Na, 0);
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, Ydechirp, Nr, Na, 0);
	RVPInitial <<<numBlocks, threadsPerBlock >>> (RVP, fr, Nr, Na, kr);
	mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (RVP, Ydechirp, RVP, Nr, Na);
	mat_ifftshift <<<numBlocks, threadsPerBlock >>> (RVP, cxtemp, Nr, Na, 0);
	cuda_ifft(cxtemp, Nr, Na, 0);
	mat_ifftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, RVP, Nr, Na, 0);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	cuDoubleComplex* kR, * R_radar;
	double*kx, * ky;
	size_t nr_na_double = Nr * Na * sizeof(double);
	cudaMalloc((void**)&kR, nr_na_size);
	cudaMalloc((void**)&R_radar, nr_na_size);
	cudaMalloc((void**)&kx, nr_na_double);
	cudaMalloc((void**)&ky, nr_na_double);
	//cudaMalloc((void**)&kR, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&R_radar, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&kx, Nr * Na * sizeof(double));
	//cudaMalloc((void**)&ky, Nr * Na * sizeof(double));
	RrefInitial <<<numBlocks, threadsPerBlock >>> (R_radar, Tslow, R0_center, Nr, Na);
	KrInitial <<<numBlocks, threadsPerBlock >>> (kR, Tfast, Tauh, Nr, Na, c, fc, kr);
	KyInitial <<<numBlocks, threadsPerBlock >>> (ky, kR, R_radar, R0_center, Nr, Na);
	KxInitial <<<numBlocks, threadsPerBlock >>> (kx, Tslow, kR, R_radar, Nr, Na);
	//以上计算结果已验证
	double kyDown = 4 * _Pi / c * (fc - b / 2);
	double kxUp = kyDown * tan((aziWidth + 1) * theta / 2);
	double kyUp_internal = 4 * _Pi / c * (fc + b / 2);
	double kyUp = sqrt(kyUp_internal * kyUp_internal - kxUp * kxUp);
	//double kyUp = sqrt((4 * _Pi / c * (fc + b / 2)) * (4 * _Pi / c * (fc + b / 2)) - kxUp * kxUp);
	double kxDown = -kxUp;
	//cudaMemcpyAsync(echo.memptr(), kR, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(echo.memptr(), kR, nr_na_size, cudaMemcpyDeviceToHost);
	double dkr = abs(echo(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1) - echo(fix(Nr / 2.0) - 2, fix(Na / 2.0) - 1));
	arma::mat kxMat(Nr, Na);
	cudaMemcpyAsync(kxMat.memptr(), kx, nr_na_double, cudaMemcpyDeviceToHost);
	double dkx = kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0)) - kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1);
	int Na1 = floor(kxUp / dkx);
	double* kyResample, * kxResample;
	int Nr1 = (int)((kyUp - kyDown) / dkr) + 1;
	cudaMalloc((void**)&kyResample, Nr1 * sizeof(double));
	kyResampleInitial <<<numBlocks, threadsPerBlock >>> (kyResample, kyDown, dkr, Nr1, 1);
	cudaMalloc((void**)&kxResample, (2 * Na1 + 1) * sizeof(double));
	kxResampleInitial <<<numBlocks, threadsPerBlock >>> (kxResample, -1 * Na1, Na1, dkx, 1, 2 * Na1 + 1);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值范围初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//插值
	int thread = 32;
	int blocks = Na / thread + 1;
	double* X, * Y, * tempky, * h, * deltar, * deltai;
	cuDoubleComplex* s, * temp;
	arma::mat xRange(Nr, Na), yRange(Nr, (2 * Na1 + 1));
	(cudaMalloc((void**)&s, Nr1 * (2 * Na1 + 1) * sizeof(double2)));
	(cudaMalloc((void**)&temp, Nr * (2 * Na1 + 1) * sizeof(double2)));
	(cudaMalloc((void**)&tempky, Nr * (2 * Na1 + 1) * sizeof(double)));
	(cudaMalloc((void**)&X, nr_na_double));
	(cudaMalloc((void**)&Y, Nr * (2 * Na1 + 1) * sizeof(double)));
	(cudaMalloc((void**)&h, Nr * sizeof(double)));
	(cudaMalloc((void**)&deltar, Nr * sizeof(double)));
	(cudaMalloc((void**)&deltai, Nr * sizeof(double)));
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值数据内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	RangeInitial <<<numBlocks, threadsPerBlock >>> (X, kx, kxDown, kxUp, Nr, Na);
	cudaMemcpyAsync(xRange.memptr(), X, Nr * Na * sizeof(double), cudaMemcpyDeviceToHost);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X插值范围计算\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	mat tempads = arma::sum(xRange);
	for (size_t i = 0; i < Nr; i++)
	{
		uvec data = arma::find(xRange.submat(span(i, i), span::all));
		int start = data.min();
		int end = data.max();
		int count = end - start + 1;
		PCHIP_SLOPE_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, i, start, count, Nr, Na);
		PCHIP_INTERP_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, kxResample, temp, i, start, count, (2 * Na1 + 1), Nr, Na);
		PCHIP_SLOPE_X <<<blocks, thread >>> (kx, ky, h, deltar, i, start, count, Nr, Na);
		PCHIP_INTERP_X <<<blocks, thread >>> (kx, ky, h, deltar, kxResample, tempky, i, start, count, (2 * Na1 + 1), Nr, Na);
	}	
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	RangeInitial <<<numBlocks, threadsPerBlock >>> (Y, tempky, kyDown, kyUp, Nr, (2 * Na1 + 1));
	cudaMemcpyAsync(yRange.memptr(), Y, Nr * (2 * Na1 + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y插值范围计算\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	for (size_t i = 0; i < (2 * Na1 + 1); i++)
	{
		uvec data = arma::find(yRange.submat(span::all,span(i, i)));
		int start = data.min() - 1;
		int end = data.max() + 1;
		int count = end - start + 1;
		PCHIP_SLOPE_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, i, start, count, Nr, (2 * Na1 + 1));
		PCHIP_INTERP_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, kyResample, s, i, start, count, Nr1, Nr, (2 * Na1 + 1), Nr1, (2 * Na1 + 1));
	}
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//升采样
	FillReSample <<<numBlocks, threadsPerBlock >>> (s, cxtemp, fix((Na - Na1 * 2 - 1) / 2), fix((Nr - Nr1) / 2), Nr1, Na1 * 2 + 1, Nr, Na);
	//fft & ifft
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na, 0);
	cuda_fft(echoPart, Nr, Na, 0);
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na, 0);
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na, 1);
	mat_col2row <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na);
	cuda_fft(cxtemp, Nr, Na, 1);
	mat_row2col <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na);
	mat_fftshift <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na, 1);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"升采样&FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	cudaMemcpyAsync(echo.memptr(), cxtemp, nr_na_size, cudaMemcpyDeviceToHost);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//cuda free
	cudaFree(Rb); cudaFree(fr); cudaFree(Tfast); cudaFree(Tslow);cudaFree(Tauh);
	cudaFree(Yref); cudaFree(Ydechirp); cudaFree(echoPart); cudaFree(cxtemp); cudaFree(RVP); cudaFree(kR);
	cudaFree(R_radar); cudaFree(kx); cudaFree(ky); cudaFree(X); cudaFree(Y); cudaFree(tempky);
	cudaFree(h); cudaFree(deltar); cudaFree(deltai); cudaFree(s); cudaFree(temp);
	cudaFree(kyResample); cudaFree(kxResample);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);*/
}
void WK_Processing_CPU(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs) {
	//timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double R0_center = r0 + Nr / 2.0 * rangeBin;
	double x_scence_center = 0;
	double Ls = 2 * R0_center * tan(theta / 2);
	int Na_Ls = fix(Ls / va / 2 * prf) * 2;
	double Na_select = aziWidth * Na_Ls + Na_Ls;
	echo = echo(span::all, span(Na / 2 - Na_select / 2 - 1, Na / 2 + Na_select / 2 - 2));
	Na = Na_select;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	cuDoubleComplex* Rb, * fr, * Tfast, * Tslow, * Tauh, * Yref, * Ydechirp, * echoPart, * cxtemp, * RVP;
	size_t nr_na_size = Nr * Na * sizeof(double2);
	size_t nr_size = Nr * sizeof(double2);
	size_t na_size = Na * sizeof(double2);

	cudaMalloc((void**)&Rb, nr_size);
	cudaMalloc((void**)&fr, nr_size);
	cudaMalloc((void**)&Tfast, nr_na_size);
	cudaMalloc((void**)&Tslow, nr_na_size);
	cudaMalloc((void**)&Tauh, nr_na_size);
	cudaMalloc((void**)&Yref, nr_na_size);
	cudaMalloc((void**)&Ydechirp, nr_na_size);
	cudaMalloc((void**)&echoPart, nr_na_size);
	cudaMalloc((void**)&cxtemp, nr_na_size);
	cudaMalloc((void**)&RVP, nr_na_size);
	//timeEnd = std::chrono::system_clock::now();
	//elapsed_seconds = timeEnd - timeStart;
	//printf("\"内存分配\":\t%f\n", elapsed_seconds);
	//timeStart = std::chrono::system_clock::now();
	
	//cudaMalloc((void**)&Rb, Nr * sizeof(double2));
	//cudaMalloc((void**)&fr, Nr * sizeof(double2));
	//cudaMalloc((void**)&Tfast, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Tslow, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Tauh, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Yref, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&Ydechirp, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&echoPart, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&cxtemp, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&RVP, Nr * Na * sizeof(double2));
	T_Initial <<<numBlocks, threadsPerBlock >>> (Rb, Nr, 1, 0, Nr, -Nr / 2.0 + 1, Nr / 2.0, rangeBin, R0_center);
	FrInitial <<<numBlocks, threadsPerBlock >>> (fr, Nr, 1, 0, Nr, 0, Nr - 1, fix(Nr / 2.0), Nr, fs);
	TfastInitial <<<numBlocks, threadsPerBlock >>> (Tfast, Rb, Nr, Na, c);
	TslowInitial <<<numBlocks, threadsPerBlock >>> (Tslow, Nr, Na, -Na / 2.0, Na / 2.0 - 1, Na, prf);
	mat_cell_muiltple <<<numBlocks, threadsPerBlock >>> (Tslow, va, Nr, Na);
	mat_cell_plus <<<numBlocks, threadsPerBlock >>> (Tslow, make_cuDoubleComplex(-1 * x_scence_center, 0), Nr, Na);//tslow=>xt
	TauhInitial <<<numBlocks, threadsPerBlock >>> (Tauh, Tslow, R0_center, c, Nr, Na);
	YredInitial <<<numBlocks, threadsPerBlock >>> (Yref, Tauh, Tfast, fc, kr, Nr, Na);
	//cudaMemcpyAsync(echoPart, echo.mem, Nr * Na * sizeof(double2), cudaMemcpyHostToDevice);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	cudaMemcpy(echoPart, echo.mem, nr_na_size, cudaMemcpyHostToDevice);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (echoPart, Yref, Ydechirp, Nr, Na);
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (Ydechirp, cxtemp, Nr, Na, 0);
	//cuda_fft(cxtemp, Nr, Na, 0);
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, Ydechirp, Nr, Na, 0);
	//RVPInitial <<<numBlocks, threadsPerBlock >>> (RVP, fr, Nr, Na, kr);
	//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (RVP, Ydechirp, RVP, Nr, Na);
	//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (RVP, cxtemp, Nr, Na, 0);
	//cuda_ifft(cxtemp, Nr, Na, 0);
	//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, RVP, Nr, Na, 0);
	cx_mat t1(Nr, Na), t2(Nr, Na);
	cudaMemcpy(t1.memptr(), Yref, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = echo % t1;
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::fft(t1);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	RVPInitial <<<numBlocks, threadsPerBlock >>> (RVP, fr, Nr, Na, kr);
	cudaMemcpy(t2.memptr(), RVP, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = t1 % t2;
	t1 = arma::shift(t1, ceil(Nr / 2.0), 0);
	t1 = arma::ifft(t1);
	t1 = arma::shift(t1, ceil(Nr / 2.0), 0);
	cudaMemcpy(RVP, t1.memptr(), Nr * Na * sizeof(double2), cudaMemcpyHostToDevice);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	cuDoubleComplex* kR, * R_radar;
	double* kx, * ky;
	size_t nr_na_double = Nr * Na * sizeof(double);
	cudaMalloc((void**)&kR, nr_na_size);
	cudaMalloc((void**)&R_radar, nr_na_size);
	cudaMalloc((void**)&kx, nr_na_double);
	cudaMalloc((void**)&ky, nr_na_double);
	//cudaMalloc((void**)&kR, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&R_radar, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&kx, Nr * Na * sizeof(double));
	//cudaMalloc((void**)&ky, Nr * Na * sizeof(double));
	RrefInitial <<<numBlocks, threadsPerBlock >>> (R_radar, Tslow, R0_center, Nr, Na);
	KrInitial <<<numBlocks, threadsPerBlock >>> (kR, Tfast, Tauh, Nr, Na, c, fc, kr);
	KyInitial <<<numBlocks, threadsPerBlock >>> (ky, kR, R_radar, R0_center, Nr, Na);
	KxInitial <<<numBlocks, threadsPerBlock >>> (kx, Tslow, kR, R_radar, Nr, Na);
	//以上计算结果已验证
	double kyDown = 4 * _Pi / c * (fc - b / 2);
	double kxUp = kyDown * tan((aziWidth + 1) * theta / 2);
	double kyUp_internal = 4 * _Pi / c * (fc + b / 2);
	double kyUp = sqrt(kyUp_internal * kyUp_internal - kxUp * kxUp);
	//double kyUp = sqrt((4 * _Pi / c * (fc + b / 2)) * (4 * _Pi / c * (fc + b / 2)) - kxUp * kxUp);
	double kxDown = -kxUp;
	//cudaMemcpyAsync(echo.memptr(), kR, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpy(echo.memptr(), kR, nr_na_size, cudaMemcpyDeviceToHost);
	double dkr = abs(echo(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1) - echo(fix(Nr / 2.0) - 2, fix(Na / 2.0) - 1));
	arma::mat kxMat(Nr, Na);
	cudaMemcpy(kxMat.memptr(), kx, nr_na_double, cudaMemcpyDeviceToHost);
	double dkx = kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0)) - kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1);
	int Na1 = floor(kxUp / dkx);
	double* kyResample, * kxResample;
	int Nr1 = (int)((kyUp - kyDown) / dkr) + 1;
	cudaMalloc((void**)&kyResample, Nr1 * sizeof(double));
	kyResampleInitial <<<numBlocks, threadsPerBlock >>> (kyResample, kyDown, dkr, Nr1, 1);
	cudaMalloc((void**)&kxResample, (2 * Na1 + 1) * sizeof(double));
	kxResampleInitial <<<numBlocks, threadsPerBlock >>> (kxResample, -1 * Na1, Na1, dkx, 1, 2 * Na1 + 1);
	//timeEnd = std::chrono::system_clock::now();
	//elapsed_seconds = timeEnd - timeStart;
	//printf("\"插值范围初始化\":\t%f\n", elapsed_seconds);
	//timeStart = std::chrono::system_clock::now();
	//插值
	int thread = 32;
	int blocks = Na / thread + 1;
	double* X, * Y, * tempky, * h, * deltar, * deltai;
	cuDoubleComplex* s, * temp;
	arma::mat xRange(Nr, Na), yRange(Nr, (2 * Na1 + 1));
	cudaMalloc((void**)&s, Nr1 * (2 * Na1 + 1) * sizeof(double2));
	cudaMalloc((void**)&temp, Nr * (2 * Na1 + 1) * sizeof(double2));
	cudaMalloc((void**)&tempky, Nr * (2 * Na1 + 1) * sizeof(double));
	cudaMalloc((void**)&X, nr_na_double);
	cudaMalloc((void**)&Y, Nr * (2 * Na1 + 1) * sizeof(double));
	cudaMalloc((void**)&h, Nr * sizeof(double));
	cudaMalloc((void**)&deltar, Nr * sizeof(double));
	cudaMalloc((void**)&deltai, Nr * sizeof(double));
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值数据内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	RangeInitial <<<numBlocks, threadsPerBlock >>> (X, kx, kxDown, kxUp, Nr, Na);
	cudaMemcpyAsync(xRange.memptr(), X, Nr * Na * sizeof(double), cudaMemcpyDeviceToHost);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X插值范围计算\":\t%f\n", elapsed_seconds);*/
	//timeStart = std::chrono::system_clock::now();

	cuDoubleComplex* temp_cuda_test;
	temp_cuda_test = (cuDoubleComplex*)malloc(Nr * (2 * Na1 + 1) * sizeof(double2));
	for (size_t i = 0; i < Nr; i++)
	{
		uvec data = arma::find(xRange.submat(span(i, i), span::all));
		int start = data.min();
		int end = data.max();
		int count = end - start + 1;
		PCHIP_SLOPE_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, i, start, count, Nr, Na);
		PCHIP_INTERP_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, kxResample, temp, i, start, count, (2 * Na1 + 1), Nr, Na);
		//cudaMemcpy(temp_cuda_test, temp_cuda, Nr * (2 * Na1 + 1) * sizeof(double2), cudaMemcpyDeviceToHost);
		PCHIP_SLOPE_X <<<blocks, thread >>> (kx, ky, h, deltar, i, start, count, Nr, Na);
		PCHIP_INTERP_X <<<blocks, thread >>> (kx, ky, h, deltar, kxResample, tempky, i, start, count, (2 * Na1 + 1), Nr, Na);
	}
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	RangeInitial <<<numBlocks, threadsPerBlock >>> (Y, tempky, kyDown, kyUp, Nr, (2 * Na1 + 1));
	cudaMemcpy(yRange.memptr(), Y, Nr * (2 * Na1 + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y插值范围计算\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	for (size_t i = 0; i < (2 * Na1 + 1); i++)
	{
		uvec data = arma::find(yRange.submat(span::all, span(i, i)));
		int start = data.min() - 1;
		int end = data.max() + 1;
		int count = end - start + 1;
		PCHIP_SLOPE_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, i, start, count, Nr, (2 * Na1 + 1));
		PCHIP_INTERP_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, kyResample, s, i, start, count, Nr1, Nr, (2 * Na1 + 1), Nr1, (2 * Na1 + 1));
	}
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();*/
	//升采样
	FillReSample <<<numBlocks, threadsPerBlock >>> (s, cxtemp, fix((Na - Na1 * 2 - 1) / 2), fix((Nr - Nr1) / 2), Nr1, Na1 * 2 + 1, Nr, Na);
	//fft & ifft
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na, 0);
	//cuda_fft(echoPart, Nr, Na, 0);
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na, 0);
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na, 1);
	//mat_col2row <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na);
	//cuda_fft(cxtemp, Nr, Na, 1);
	//mat_row2col <<<numBlocks, threadsPerBlock >>> (cxtemp, echoPart, Nr, Na);
	//mat_fftshift <<<numBlocks, threadsPerBlock >>> (echoPart, cxtemp, Nr, Na, 1);

	cudaMemcpy(t1.memptr(), cxtemp, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::fft(t1);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	t1 = arma::fft(t1.t()).t();
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	echo = cx_mat(t1);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"升采样&FFT\":\t%f\n", elapsed_seconds);*/
	//timeStart = std::chrono::system_clock::now();
	//cudaMemcpyAsync(echo.memptr(), cxtemp, nr_na_size, cudaMemcpyDeviceToHost);
	//timeEnd = std::chrono::system_clock::now();
	//elapsed_seconds = timeEnd - timeStart;
	//printf("\"数据传输\":\t%f\n", elapsed_seconds);
	//timeStart = std::chrono::system_clock::now();
	//cuda free
	cudaFree(Rb); cudaFree(fr); cudaFree(Tfast); cudaFree(Tslow); cudaFree(Tauh);
	cudaFree(Yref); cudaFree(Ydechirp); cudaFree(echoPart); cudaFree(cxtemp); cudaFree(RVP); cudaFree(kR);
	cudaFree(R_radar); cudaFree(kx); cudaFree(ky); cudaFree(X); cudaFree(Y); cudaFree(tempky);
	cudaFree(h); cudaFree(deltar); cudaFree(deltai); cudaFree(s); cudaFree(temp);
	cudaFree(kyResample); cudaFree(kxResample);
	/*timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);*/
}
void WK_Processing_CPU_2(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs) {
	timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double R0_center = r0 + Nr / 2.0 * rangeBin;
	double x_scence_center = 0;
	double Ls = 2 * R0_center * tan(theta / 2);
	int Na_Ls = fix(Ls / va / 2 * prf) * 2;
	double Na_select = aziWidth * Na_Ls + Na_Ls;
	echo = echo(span::all, span(Na / 2 - Na_select / 2 - 1, Na / 2 + Na_select / 2 - 2));
	Na = Na_select;


	dmat Rb = (R0_center + rangeBin * (regspace<vec>((int)(-Nr / 2 + 1), (int)(Nr / 2)))) * ones(1, Na);
	dmat fr = ((regspace<vec>(0, Nr - 1) - fix(Nr / 2)) / Nr * fs) * ones(1, Na);
	dmat tr = 2 * Rb / c;
	dmat Tslow = (regspace<vec>(0, Na - 1) - fix(Na / 2)) / prf * ones(1, Nr);
	dmat xt = va * Tslow.t() - x_scence_center;
	double yt = R0_center;
	dmat Rref = sqrt(xt % xt + yt * yt);
	dmat tauh = 2 * Rref / c;
	dmat realtemp1 = cos(-2 * _Pi * fc * tauh + _Pi * kr * ((tr - tauh) % (tr - tauh)));
	dmat imagtemp1 = sin(-2 * _Pi * fc * tauh + _Pi * kr * ((tr - tauh) % (tr - tauh)));//

	cx_dmat Yref(realtemp1, imagtemp1);
	cx_dmat Ydechirp = echo % conj(Yref);

	// 补偿RVP
	Ydechirp = arma::shift(Ydechirp, floor(Nr / 2.0), 0);
	Ydechirp = arma::fft(Ydechirp);
	cx_dmat echo_dechirp_range = arma::shift(Ydechirp, floor(Nr / 2.0), 0);
	dmat realtemp2 = cos(_Pi * fr % fr / kr);
	dmat imagtemp2 = -sin(_Pi * fr % fr / kr);
	cx_dmat HRPV(realtemp2, imagtemp2);
	cx_dmat temp_liu = echo_dechirp_range % HRPV;
	temp_liu = arma::shift(temp_liu, ceil(Nr / 2.0), 0);
	temp_liu = arma::ifft(temp_liu);
	cx_dmat Y_RVP = arma::shift(temp_liu, ceil(Nr / 2.0), 0);
	
	dmat kR = 4 * _Pi / c * (fc + kr*(tr - tauh));
	dmat ky = yt * kR % (1 / Rref);
	dmat kx = xt % kR % (1 / Rref);
	//以上计算结果已验证
	double kyDown = 4 * _Pi / c * (fc - b / 2);
	double kxUp = kyDown * tan((aziWidth + 1) * theta / 2);
	double kyUp_internal = 4 * _Pi / c * (fc + b / 2);
	double kyUp = sqrt(kyUp_internal * kyUp_internal - kxUp * kxUp);
	double kxDown = -kxUp;
	


	// 重采样位置
	//cudaMemcpyAsync(echo.memptr(), kR, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	double dkr = abs(kR(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1) - kR(fix(Nr / 2.0) - 2, fix(Na / 2.0) - 1));

	double dkx = kx(fix(Nr / 2.0) - 1, fix(Na / 2.0)) - kx(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1);
	int Na1 = floor(kxUp / dkx);
	dmat kyResample = regspace(kyDown, dkr, kyUp);
	int Nr1 = kyResample.n_rows;
	dmat kxResample = regspace(-Na1, Na1) * dkx;

	// 重采样数据预设

	cx_mat s(zeros(Nr1, Na1 * 2 + 1), zeros(Nr1, Na1 * 2 + 1));
	cx_mat temp(zeros(Nr, Na1 * 2 + 1), zeros(Nr, Na1 * 2 + 1));
	dmat tempky=zeros(Nr, Na1 * 2 + 1);
	dmat x_range1, x_range2;
	x_range1 = kx;
	x_range2 = kx;
	x_range1.transform([kxDown](double val) {return val >= kxDown ? 1.0 : 0; });
	x_range2.transform([kxUp](double val) {return val <= kxUp ? 1.0 : 0; });
	dmat xRange = x_range1 % x_range2;

	// 插值采用cuda版本
	int thread = 32;
	int blocks = Na / thread + 1;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);

	double* kx_cuda, * ky_cuda;
	size_t nr_na_double = Nr * Na * sizeof(double);
	cudaMalloc((void**)&kx_cuda, nr_na_double);
	cudaMalloc((void**)&ky_cuda, nr_na_double);
	cudaMemcpy(kx_cuda, kx.mem, nr_na_double, cudaMemcpyHostToDevice);
	cudaMemcpy(ky_cuda, ky.mem, nr_na_double, cudaMemcpyHostToDevice);

	cuDoubleComplex* RVP_cuda, * cxtemp_cuda;
	size_t nr_na_size = Nr * Na * sizeof(double2);
	cudaMalloc((void**)&RVP_cuda, nr_na_size);
	cudaMalloc((void**)&cxtemp_cuda, nr_na_size);
	cudaMemcpy(RVP_cuda, Y_RVP.mem, nr_na_size, cudaMemcpyHostToDevice);

	double * tempky_cuda, * h_cuda, * deltar_cuda, * deltai_cuda;
	cuDoubleComplex* s_cuda, * temp_cuda;
	cudaMalloc((void**)&s_cuda, Nr1 * (2 * Na1 + 1) * sizeof(double2));
	

	cudaMalloc((void**)&temp_cuda, Nr * (2 * Na1 + 1) * sizeof(double2));
	//cudaMemcpy(temp_cuda, temp.mem, Nr * (2 * Na1 + 1) * sizeof(double2), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&tempky_cuda, Nr * (2 * Na1 + 1) * sizeof(double));
	//cudaMemcpy(temp_cuda, temp.mem, Nr * (2 * Na1 + 1) * sizeof(double2), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&h_cuda, Nr * sizeof(double));
	cudaMalloc((void**)&deltar_cuda, Nr * sizeof(double));
	cudaMalloc((void**)&deltai_cuda, Nr * sizeof(double));

	double* kyResample_cuda, * kxResample_cuda;
	cudaMalloc((void**)&kxResample_cuda, (2 * Na1 + 1) * sizeof(double));
	cudaMemcpy(kxResample_cuda, kxResample.mem, (2 * Na1 + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&kyResample_cuda, Nr1 * sizeof(double));
	cudaMemcpy(kyResample_cuda, kyResample.mem, Nr1 * sizeof(double), cudaMemcpyHostToDevice);

	//for (size_t i = 0; i < Nr; i++)
	//{
	//	uvec data = arma::find(xRange.submat(span(i, i), span::all));
	//	int start = data.min();
	//	int end = data.max();
	//	int count = end - start + 1;
	//	PCHIP_SLOPE_CX_X <<<blocks, thread >>> (kx_cuda, RVP_cuda, h_cuda, deltar_cuda, deltai_cuda, i, start, count, Nr, Na);
	//	PCHIP_INTERP_CX_X <<<blocks, thread >>> (kx_cuda, RVP_cuda, h_cuda, deltar_cuda, deltai_cuda, kxResample_cuda, temp_cuda, i, start, count, (2 * Na1 + 1), Nr, Na);
	//	
	//	PCHIP_SLOPE_X <<<blocks, thread >>> (kx_cuda, ky_cuda, h_cuda, deltar_cuda, i, start, count, Nr, Na);
	//	PCHIP_INTERP_X <<<blocks, thread >>> (kx_cuda, ky_cuda, h_cuda, deltar_cuda, kxResample_cuda, tempky_cuda, i, start, count, (2 * Na1 + 1), Nr, Na);
	//}
	CPchip PCHIP_INTERP;
	for (size_t i = 0; i < Nr; i++)
	{
		uvec data = arma::find(xRange.submat(span(i, i), span::all));
		int start = data.min();
		int end = data.max();
		int count = end - start + 1;
		//vec kxTemp = (kx.submat(span(i, i), span(start, end))).t();
		//temp.row(i) = (complex_pchip_interpolate((kx.submat(span(i,i),span(start, end))).t(), (Y_RVP.submat(span(i, i), span(start, end))).t(), kxResample)).t();
		//tempky.row(i) = (pchip_interpolate((kx.submat(span(i, i), span(start, end))).t(), (ky.submat(span(i, i), span(start, end))).t(), kxResample)).t();
		temp.row(i) = (PCHIP_INTERP.CPchip_Complex((kx.submat(span(i, i), span(start, end))).t(), (Y_RVP.submat(span(i, i), span(start, end))).t(), kxResample)).t();
		tempky.row(i) = (PCHIP_INTERP.CPchip_double((kx.submat(span(i, i), span(start, end))).t(), (ky.submat(span(i, i), span(start, end))).t(), kxResample)).t();

	}
	//temp = readcx_matData("temp.mat", "temp");
	//tempky = readdmatData("tempky.mat", "tempky");
	//PCHIP_INTERP.~CPchip();
	cout << "success" << endl;
	//cudaMemcpy(tempky.memptr(), tempky_cuda, tempky.n_elem * sizeof(double), cudaMemcpyDeviceToHost);
	dmat y_range1, y_range2;
	y_range1 = tempky;
	y_range2 = tempky;
	y_range1.transform([kyDown](double val) {return val >= kyDown ? 1.0 : 0; });
	y_range2.transform([kyUp](double val) {return val <= kyUp ? 1.0 : 0; });
	dmat yRange = y_range1 % y_range2;
	kyResample = flipud(kyResample);
	for (size_t i = 0; i < (2 * Na1 + 1); i++)
	{
		uvec data = arma::find(yRange.submat(span::all, span(i, i)));
		int start = data.min() - 1;
		int end = data.max() + 1;
		int count = end - start + 1;
		//s.col(i) = complex_pchip_interpolate((tempky.submat(span(start - 1, end + 1), span(i, i))), (temp.submat(span(start - 1, end + 1), span(i, i))), kyResample);
		vec tempkyinput = flipud((tempky.submat(span(start, end), span(i, i))));
		cx_vec tempinput = flipud(temp.submat(span(start, end), span(i, i)));
		cx_vec scol_i = PCHIP_INTERP.CPchip_Complex(tempkyinput, tempinput, kyResample);
		scol_i = flipud(scol_i);
		s.col(i) = scol_i;
	}
	cudaMemcpy(s_cuda, s.mem, Nr1 * (2 * Na1 + 1) * sizeof(double2), cudaMemcpyHostToDevice);
	//升采样
	FillReSample <<<numBlocks, threadsPerBlock >>> (s_cuda, cxtemp_cuda, fix((Na - Na1 * 2 - 1) / 2), fix((Nr - Nr1) / 2), Nr1, Na1 * 2 + 1, Nr, Na);
	cx_mat t1(Nr, Na), t2(Nr, Na);
	cudaMemcpy(t1.memptr(), cxtemp_cuda, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::fft(t1);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	t1 = arma::fft(t1.t()).t();
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	echo = cx_mat(t1);

	//cuda free
	cudaFree(cxtemp_cuda); cudaFree(RVP_cuda);
	cudaFree(kx_cuda); cudaFree(ky_cuda);cudaFree(tempky_cuda);
	cudaFree(h_cuda); cudaFree(deltar_cuda); cudaFree(deltai_cuda); cudaFree(s_cuda); cudaFree(temp_cuda);
	cudaFree(kyResample_cuda); cudaFree(kxResample_cuda);
	
}
void WK_Processing_CPU_3(cx_mat& echo, double fc, double va, double kr, double prf, double b, double aziWidth, double theta, double r0, double lambda, double rangeBin, double fs) {
	timeStart = std::chrono::system_clock::now();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double R0_center = r0 + Nr / 2.0 * rangeBin;
	double x_scence_center = 0;
	double Ls = 2 * R0_center * tan(theta / 2);
	int Na_Ls = fix(Ls / va / 2 * prf) * 2;
	double Na_select = aziWidth * Na_Ls + Na_Ls;
	echo = echo(span::all, span(Na / 2 - Na_select / 2 - 1, Na / 2 + Na_select / 2 - 2));
	Na = Na_select;
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	cuDoubleComplex* Rb, * fr, * Tfast, * Tslow, * Tauh, * Yref, * Ydechirp, * echoPart, * cxtemp, * RVP;
	size_t nr_na_size = Nr * Na * sizeof(double2);
	size_t nr_size = Nr * sizeof(double2);
	size_t na_size = Na * sizeof(double2);

	dmat Rb_cpu = (R0_center + rangeBin * (regspace<vec>((int)(-Nr / 2 + 1), (int)(Nr / 2)))) * ones(1, Na);
	dmat fr_cpu = ((regspace<vec>(0, Nr - 1) - fix(Nr / 2)) / Nr * fs) * ones(1, Na);
	dmat tr = 2 * Rb_cpu / c;
	dmat Tslow_cpu = (regspace<vec>(0, Na - 1) - fix(Na / 2)) / prf * ones(1, Nr);
	dmat xt = va * Tslow_cpu.t() - x_scence_center;
	double yt = R0_center;
	dmat Rref = sqrt(xt % xt + yt * yt);
	dmat tauh = 2 * Rref / c;
	dmat realtemp1 = cos(2 * _Pi * fc * tauh + _Pi * kr * ((tr - tauh) % (tr - tauh)));
	dmat imagtemp1 = sin(2 * _Pi * fc * tauh + _Pi * kr * ((tr - tauh) % (tr - tauh)));
	cx_dmat Yref_cpu(realtemp1, imagtemp1);
	cx_dmat Ydechirp_cpu = echo % Yref_cpu;

	cudaMalloc((void**)&Rb, nr_size);
	cudaMalloc((void**)&fr, nr_size);
	cudaMalloc((void**)&Tfast, nr_na_size);
	cudaMalloc((void**)&Tslow, nr_na_size);
	cudaMalloc((void**)&Tauh, nr_na_size);
	cudaMalloc((void**)&Yref, nr_na_size);
	cudaMalloc((void**)&Ydechirp, nr_na_size);
	cudaMalloc((void**)&echoPart, nr_na_size);
	cudaMalloc((void**)&cxtemp, nr_na_size);
	cudaMalloc((void**)&RVP, nr_na_size);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	  
	T_Initial <<<numBlocks, threadsPerBlock >>> (Rb, Nr, 1, 0, Nr, -Nr / 2.0 + 1, Nr / 2.0, rangeBin, R0_center);
	FrInitial <<<numBlocks, threadsPerBlock >>> (fr, Nr, 1, 0, Nr, 0, Nr - 1, fix(Nr / 2.0), Nr, fs);
	TfastInitial <<<numBlocks, threadsPerBlock >>> (Tfast, Rb, Nr, Na, c);
	TslowInitial <<<numBlocks, threadsPerBlock >>> (Tslow, Nr, Na, -Na / 2.0, Na / 2.0 - 1, Na, prf);
	mat_cell_muiltple <<<numBlocks, threadsPerBlock >>> (Tslow, va, Nr, Na);
	mat_cell_plus <<<numBlocks, threadsPerBlock >>> (Tslow, make_cuDoubleComplex(-1 * x_scence_center, 0), Nr, Na);//tslow=>xt
	TauhInitial <<<numBlocks, threadsPerBlock >>> (Tauh, Tslow, R0_center, c, Nr, Na);
	YredInitial <<<numBlocks, threadsPerBlock >>> (Yref, Tauh, Tfast, fc, kr, Nr, Na);
	cudaMemcpyAsync(echoPart, echo.mem, Nr * Na * sizeof(double2), cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cudaMemcpy(echoPart, echo.mem, nr_na_size, cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();

	cx_mat t1(Nr, Na), t2(Nr, Na), t3(Nr, Na);
	cudaMemcpy(t1.memptr(), Yref, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = echo % t1;
	t3 = echo % t3;
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t3 = arma::shift(t3, floor(Nr / 2.0), 0);
	t1 = arma::fft(t1);
	t3 = arma::fft(t3);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t3 = arma::shift(t3, floor(Nr / 2.0), 0);
	RVPInitial <<<numBlocks, threadsPerBlock >>> (RVP, fr, Nr, Na, kr);
	cudaMemcpy(t2.memptr(), RVP, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = t1 % t2;
	t1 = arma::shift(t1, ceil(Nr / 2.0), 0);
	t3 = arma::shift(t3, ceil(Nr / 2.0), 0);
	t1 = arma::ifft(t1);
	t3 = arma::ifft(t3);
	t1 = arma::shift(t1, ceil(Nr / 2.0), 0);
	t3 = arma::shift(t3, ceil(Nr / 2.0), 0);
	cudaMemcpy(RVP, t1.memptr(), Nr * Na * sizeof(double2), cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cuDoubleComplex* kR, * R_radar;
	double* kx, * ky;
	size_t nr_na_double = Nr * Na * sizeof(double);
	cudaMalloc((void**)&kR, nr_na_size);
	cudaMalloc((void**)&R_radar, nr_na_size);
	cudaMalloc((void**)&kx, nr_na_double);
	cudaMalloc((void**)&ky, nr_na_double);
	//cudaMalloc((void**)&kR, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&R_radar, Nr * Na * sizeof(double2));
	//cudaMalloc((void**)&kx, Nr * Na * sizeof(double));
	//cudaMalloc((void**)&ky, Nr * Na * sizeof(double));
	RrefInitial <<<numBlocks, threadsPerBlock >>> (R_radar, Tslow, R0_center, Nr, Na);
	KrInitial <<<numBlocks, threadsPerBlock >>> (kR, Tfast, Tauh, Nr, Na, c, fc, kr);
	KyInitial <<<numBlocks, threadsPerBlock >>> (ky, kR, R_radar, R0_center, Nr, Na);
	KxInitial <<<numBlocks, threadsPerBlock >>> (kx, Tslow, kR, R_radar, Nr, Na);
	//以上计算结果已验证
	double kyDown = 4 * _Pi / c * (fc - b / 2);
	double kxUp = kyDown * tan((aziWidth + 1) * theta / 2);
	double kyUp_internal = 4 * _Pi / c * (fc + b / 2);
	double kyUp = sqrt(kyUp_internal * kyUp_internal - kxUp * kxUp);
	//double kyUp = sqrt((4 * _Pi / c * (fc + b / 2)) * (4 * _Pi / c * (fc + b / 2)) - kxUp * kxUp);
	double kxDown = -kxUp;
	//cudaMemcpyAsync(echo.memptr(), kR, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpy(echo.memptr(), kR, nr_na_size, cudaMemcpyDeviceToHost);
	double dkr = abs(echo(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1) - echo(fix(Nr / 2.0) - 2, fix(Na / 2.0) - 1));
	arma::mat kxMat(Nr, Na);
	cudaMemcpy(kxMat.memptr(), kx, nr_na_double, cudaMemcpyDeviceToHost);
	double dkx = kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0)) - kxMat(fix(Nr / 2.0) - 1, fix(Na / 2.0) - 1);
	int Na1 = floor(kxUp / dkx);
	double* kyResample, * kxResample;
	int Nr1 = (int)((kyUp - kyDown) / dkr) + 1;
	cudaMalloc((void**)&kyResample, Nr1 * sizeof(double));
	kyResampleInitial <<<numBlocks, threadsPerBlock >>> (kyResample, kyDown, dkr, Nr1, 1);
	cudaMalloc((void**)&kxResample, (2 * Na1 + 1) * sizeof(double));
	kxResampleInitial <<<numBlocks, threadsPerBlock >>> (kxResample, -1 * Na1, Na1, dkx, 1, 2 * Na1 + 1);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值范围初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	//插值
	int thread = 32;
	int blocks = Na / thread + 1;
	double* X, * Y, * tempky, * h, * deltar, * deltai;
	cuDoubleComplex* s, * temp;
	arma::mat xRange(Nr, Na), yRange(Nr, (2 * Na1 + 1));
	cudaMalloc((void**)&s, Nr1 * (2 * Na1 + 1) * sizeof(double2));
	cudaMalloc((void**)&temp, Nr * (2 * Na1 + 1) * sizeof(double2));
	cudaMalloc((void**)&tempky, Nr * (2 * Na1 + 1) * sizeof(double));
	cudaMalloc((void**)&X, nr_na_double);
	cudaMalloc((void**)&Y, Nr * (2 * Na1 + 1) * sizeof(double));
	cudaMalloc((void**)&h, Nr * sizeof(double));
	cudaMalloc((void**)&deltar, Nr * sizeof(double));
	cudaMalloc((void**)&deltai, Nr * sizeof(double));
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值数据内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	RangeInitial <<<numBlocks, threadsPerBlock >>> (X, kx, kxDown, kxUp, Nr, Na);
	cudaMemcpyAsync(xRange.memptr(), X, Nr * Na * sizeof(double), cudaMemcpyDeviceToHost);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X插值范围计算\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	for (size_t i = 0; i < Nr; i++)
	{
		uvec data = arma::find(xRange.submat(span(i, i), span::all));
		int start = data.min();
		int end = data.max();
		int count = end - start + 1;
		PCHIP_SLOPE_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, i, start, count, Nr, Na);
		PCHIP_INTERP_CX_X <<<blocks, thread >>> (kx, RVP, h, deltar, deltai, kxResample, temp, i, start, count, (2 * Na1 + 1), Nr, Na);
		PCHIP_SLOPE_X <<<blocks, thread >>> (kx, ky, h, deltar, i, start, count, Nr, Na);
		PCHIP_INTERP_X <<<blocks, thread >>> (kx, ky, h, deltar, kxResample, tempky, i, start, count, (2 * Na1 + 1), Nr, Na);
	}
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"X方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	RangeInitial <<<numBlocks, threadsPerBlock >>> (Y, tempky, kyDown, kyUp, Nr, (2 * Na1 + 1));
	cudaMemcpy(yRange.memptr(), Y, Nr * (2 * Na1 + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y插值范围计算\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	for (size_t i = 0; i < (2 * Na1 + 1); i++)
	{
		uvec data = arma::find(yRange.submat(span::all, span(i, i)));
		int start = data.min() - 1;
		int end = data.max() + 1;
		int count = end - start + 1;
		PCHIP_SLOPE_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, i, start, count, Nr, (2 * Na1 + 1));
		PCHIP_INTERP_CX_Y <<<blocks, thread >>> (tempky, temp, h, deltar, deltai, kyResample, s, i, start, count, Nr1, Nr, (2 * Na1 + 1), Nr1, (2 * Na1 + 1));
	}
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"Y方向插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	//升采样
	FillReSample <<<numBlocks, threadsPerBlock >>> (s, cxtemp, fix((Na - Na1 * 2 - 1) / 2), fix((Nr - Nr1) / 2), Nr1, Na1 * 2 + 1, Nr, Na);

	cudaMemcpy(t1.memptr(), cxtemp, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::fft(t1);
	t1 = arma::shift(t1, floor(Nr / 2.0), 0);
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	t1 = arma::fft(t1.t()).t();
	t1 = arma::shift(t1, floor(Na / 2.0), 1);
	echo = cx_mat(t1);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"升采样&FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	//cuda free
	cudaFree(Rb); cudaFree(fr); cudaFree(Tfast); cudaFree(Tslow); cudaFree(Tauh);
	cudaFree(Yref); cudaFree(Ydechirp); cudaFree(echoPart); cudaFree(cxtemp); cudaFree(RVP); cudaFree(kR);
	cudaFree(R_radar); cudaFree(kx); cudaFree(ky); cudaFree(X); cudaFree(Y); cudaFree(tempky);
	cudaFree(h); cudaFree(deltar); cudaFree(deltai); cudaFree(s); cudaFree(temp);
	cudaFree(kyResample); cudaFree(kxResample);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);
}
//BP Kernel Functions
__global__ void AntInitial(double* x, double* y, double* z, double start, double end, int count, double va, double prf)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count) {
		x[idx] = ((end - start) / (count - 1) * idx + start)*va/prf;
		y[idx] = 0;
		z[idx] = 0;
		//printf("x=%d,\tantx=%f\n", idx, x[idx]);
	}
}
__global__ void MatInitial(double* x, double* y, double* z, int row, int col,double deltaima,double deltaimr,double r0center) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		int pos = ix + iy * row;
		x[pos] = (iy - col / 2.0) * deltaima;
		y[pos] = (ix - row / 2.0 + 1) * deltaimr + r0center;
		z[pos] = 0;
	}
}
__global__ void FastInitial(cuDoubleComplex* fast, int row, int col, double fs,double nr) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		fast[ix + iy * row].x = ((nr - 1.0) / (row - 1) * ix - nr / 2.0) / nr * fs;
	}
}
__global__ void H2Initial(cuDoubleComplex* h2, cuDoubleComplex* fast, int row, int col, double kr)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		cuDoubleComplex t1 = make_cuDoubleComplex(0, _Pi*fast[ix].x*fast[ix].x/kr);
		double exp_real = exp(t1.x) * cos(t1.y);
		double exp_imag = exp(t1.x) * sin(t1.y);
		h2[ix + iy * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void HShiftInitial(cuDoubleComplex* hshift, cuDoubleComplex* fast, int row, int col, double tp) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		cuDoubleComplex t1 = make_cuDoubleComplex(0, 2 * _Pi * fast[ix].x * tp / 2);
		double exp_real = exp(t1.x) * cos(t1.y);
		double exp_imag = exp(t1.x) * sin(t1.y);
		hshift[ix + iy * row] = make_cuDoubleComplex(exp_real, exp_imag);
	}
}
__global__ void RvecInitial(double* vec, double start, double end, double rangebin, double rs, int count)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < count && iy < 1)
	{
		vec[ix] = ((end - start) / (count - 1) * ix + start) * rangebin + rs;
	}
}
__global__ void EchoReduceRow(cuDoubleComplex* echo, cuDoubleComplex* dec, int row, int col, int rowDec)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < (row-rowDec) && iy < col)
	{
		dec[ix + iy * (row - rowDec)] = echo[ix + iy * row];
	}
}
__global__ void CaldR_Corr(cuDoubleComplex* coor,double* dr, double* matx, double* maty, double* matz, double* antx, double* anty, double* antz, int i, double lambda, int row, int col)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		int pos = ix + iy * row;
		double t1 = antx[i] - matx[pos];
		double t2 = anty[i] - maty[pos];
		double t3 = antz[i] - matz[pos];
		dr[pos] = sqrt(t1 * t1 + t2 * t2 + t3 * t3);
		double t = dr[pos] * 4 * _Pi / lambda;
		coor[pos].x = cos(t);
		coor[pos].y = sin(t);
		//printf("x=%d,\ty=%d,\ti=%d,\tantx=%f,\tdr=%f\n", ix, iy, i, antx[i], dr[ix + iy * row]);
	}
}
__global__ void GetVecRange(double* vec, double* index, double min, double max, int row, int col)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		int pos = ix + iy * row;
		double t = vec[pos];
		if (t > min && t < max) {
			index[pos] = 1;
		}
		else
		{
			index[pos] = -1;
		}
	}
}
__global__ void AddImage_Interp(cuDoubleComplex* img, double* index, double* vec, cuDoubleComplex* echo, double* dr, cuDoubleComplex* corr, int icol, int n, int row, int col)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < row && iy < col)
	{
		int iii = ix + iy * row;
		int offset = icol * n;
		if (index[iii] > 0) {
			double xqi = dr[iii];
			int i = 0;
			if (xqi < vec[0])
				i = 0;
			else if (xqi > vec[n - 1])
				i = n - 1;
			else
			{
				//while (left < right) {
				//	printf("mid=%d\n", mid);
				//	mid = left + (int)((right - left) / 2.0);
				//	if (vec[mid] > xqi) {
				//		right = mid; // 继续在左侧查找更小的索引位置
				//	}
				//	else {
				//		left = mid; // 在右侧继续查找
				//	}
				//	i = mid - 1;
				//}

				// 
				//while (i < n - 1 && vec[i + 1] < xqi)
				//	i++;
				// 
				i = n - 1;
				while (i >0 && vec[i] > xqi)
					i--;
				//printf("ix=%d,\tiy=%d,\tindex=%d\n", ix, iy, i);
			}			
			double x0, x1, y0, y1, y2, y3;
			if (i == n - 1) {
				x0 = vec[i - 1];
				x1 = vec[i];
				y0 = echo[i - 1 + offset].x;
				y1 = echo[i + offset].x;
				y2 = echo[i - 1 + offset].y;
				y3 = echo[i + offset].y;
			}
			else {
				x0 = vec[i];
				x1 = vec[i + 1];
				y0 = echo[i + offset].x;
				y1 = echo[i + 1 + offset].x;
				y2 = echo[i + offset].y;
				y3 = echo[i + 1 + offset].y;
			}
			cuDoubleComplex t1 = make_cuDoubleComplex(y0 + (y1 - y0) * (xqi - x0) / (x1 - x0), y2 + (y3 - y2) * (xqi - x0) / (x1 - x0));
			cuDoubleComplex t2 = cuCmul(t1, corr[iii]);
			img[iii] = cuCadd(img[iii], t2);
			//printf("X=%d,\tY=%d,\tvalue.x=%f,\tvalue.y=%f\n", ix, iy, t1.x, t1.y);
		}
	}
}
void BP_Processing(cx_mat& echo, double fs, double kr, double tp, double prf, double lambda, double va, double r0, double rangeBin, double theta, double widthRange, double widthAzi, double deltaimr, double deltaima, double aziSection) {
	timeStart = std::chrono::system_clock::now();
	//echo = echo.t();
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double r0Center = r0 + Nr / 2.0 * rangeBin;
	double Ls = 2 * r0Center * tan(theta / 2);
	int naLs = fix(Ls / va / 2 * prf) * 2;
	int naSelect = fix(widthAzi / va * prf / 2) * 2 + naLs;
	int imnan = fix(widthAzi / deltaima / 2) * 2 ;
	int imnrn = fix(widthRange / deltaimr / 2) * 2;
	int t = Na / 2 + fix(aziSection / va * prf);
	int np = fix(tp * fs);
	rangeBin = c / 2 / fs;
	double thetaSq = 0, thetaBw = 0;
	echo = echo.submat(span::all, span(t - naSelect / 2 - 1, t + naSelect / 2 - 2));

	//cx_mat echo_cputest(echo.n_rows, echo.n_cols);//CPU测试矩阵

	Na = naSelect;
	double* antX, * antY, * antZ, * matX, * matY, * matZ;
	size_t nas_double = naSelect * sizeof(double);
	size_t imna_imnr_double = imnan * imnrn * sizeof(double);
	(cudaMalloc((void**)&antX, nas_double));
	(cudaMalloc((void**)&antY, nas_double));
	(cudaMalloc((void**)&antZ, nas_double));
	(cudaMalloc((void**)&matX, imna_imnr_double));
	(cudaMalloc((void**)&matY, imna_imnr_double));
	(cudaMalloc((void**)&matZ, imna_imnr_double));
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存分配\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	AntInitial <<<naSelect / 64 + 1, 64 >>> (antX, antY, antZ, -naSelect / 2.0, naSelect / 2.0 - 1, naSelect, va, prf);
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
	MatInitial <<<numBlocks, threadsPerBlock >>> (matX, matY, matZ, imnrn, imnan, deltaima, deltaimr, r0Center);
	cuDoubleComplex* Echo, * Fast, * H2, * HShift, * Temp;
	size_t nr_na = Nr * Na * sizeof(double2);
	cudaMalloc((void**)&Echo, nr_na);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cudaMemcpy(Echo, echo.mem, nr_na, cudaMemcpyHostToDevice);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	(cudaMalloc((void**)&Fast, Nr * sizeof(double2)));
	(cudaMalloc((void**)&H2, Nr * sizeof(double2)));
	(cudaMalloc((void**)&HShift, Nr * sizeof(double2)));
	(cudaMalloc((void**)&Temp, Nr * sizeof(double2)));
	FastInitial <<<numBlocks, threadsPerBlock >>> (Fast, Nr, 1, fs, Nr);
	H2Initial <<<numBlocks, threadsPerBlock >>>(H2, Fast, Nr, 1, kr);
	HShiftInitial <<<numBlocks, threadsPerBlock >>> (HShift, Fast, Nr, 1, tp);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"矩阵初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	for (size_t i = 0; i < Na; i++)
	{
		mat_slice_col <<<numBlocks, threadsPerBlock >>> (Echo, Fast, Nr, Na, i, i);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (Fast, Temp, Nr, 1);
		//data gpu to cpu cudamemecopy
		//arma::fft()
		//data cpu to gpu
		cuda_fft(Temp, Nr, 1, 0);
		mat_fftshift <<<numBlocks, threadsPerBlock >>> (Temp, Fast, Nr, 1);
		mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (Fast, HShift, Fast, Nr, 1);
		mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (Fast, H2, Fast, Nr, 1);
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (Fast, Temp, Nr, 1);
		cuda_ifft(Temp, Nr, 1, 0);
		mat_ifftshift <<<numBlocks, threadsPerBlock >>> (Temp, Fast, Nr, 1);
		mat_set_col <<<numBlocks, threadsPerBlock >>> (Echo, Fast, Nr, Na, i);
	}
	//cudaMemcpy(echo_cputest.memptr(), Echo, nr_na, cudaMemcpyDeviceToHost);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"逐行FFT\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	double* rVec, * T, * dR, * I;
	cuDoubleComplex* echoRec, * Corr, * Img, * Img_temp;
	cx_mat Ima_cputest(imnrn, imnan);
	cudaMalloc((void**)&echoRec, (Nr-np) * Na * sizeof(double2));
	EchoReduceRow <<<numBlocks, threadsPerBlock >>> (Echo, echoRec, Nr, Na, np);
	Nr -= np;
	//echo_cputest.resize(Nr, Na);
	//cudaMemcpy(echo_cputest.memptr(), echoRec, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMalloc((void**)&rVec, Nr * sizeof(double));
	size_t imna_imnr_double2 = imnan * imnrn * sizeof(double2);
	cudaMalloc((void**)&Img, imna_imnr_double2);
	cudaMalloc((void**)&Img_temp, imna_imnr_double2);
	cudaMalloc((void**)&dR, imna_imnr_double);
	cudaMalloc((void**)&I, imna_imnr_double);
	cudaMalloc((void**)&T, Na * sizeof(double));
	cudaMalloc((void**)&Corr, imna_imnr_double2);
	RvecInitial <<<numBlocks, threadsPerBlock >>> (rVec, 0, Nr - 1, rangeBin, r0, Nr);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"插值数据初始化\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	double rMin = r0, rMax = (Nr - 1) * rangeBin + r0;
	for (size_t i = 0; i < Na; i++)
	{
		CaldR_Corr <<<numBlocks, threadsPerBlock >>> (Corr, dR, matX, matY, matZ, antX, antY, antZ, i, lambda, imnrn, imnan);
		GetVecRange <<<numBlocks, threadsPerBlock >>> (dR, I, rMin, rMax, imnrn, imnan);
		AddImage_Interp <<<numBlocks, threadsPerBlock >>> (Img, I, rVec, echoRec, dR, Corr, i, Nr, imnrn, imnan);
		//cudaMemcpy(Ima_cputest.memptr(), Img, imna_imnr_double2, cudaMemcpyDeviceToHost);

	}
	//mat_row2col <<< numBlocks, threadsPerBlock >>> (Img, Img_temp, imnrn, imnan);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"线性插值\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	echo = echo.submat(span(0, imnrn - 1), span(0, imnan - 1));
	cudaMemcpyAsync(echo.memptr(), Img, imna_imnr_double2, cudaMemcpyDeviceToHost);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"数据传输\":\t%f\n", elapsed_seconds);
	timeStart = std::chrono::system_clock::now();
	cudaFree(antX); cudaFree(antY); cudaFree(antZ);
	cudaFree(matX); cudaFree(matY); cudaFree(matZ);
	cudaFree(Echo); cudaFree(Fast); cudaFree(H2); cudaFree(HShift);cudaFree(Temp); 
	cudaFree(rVec); cudaFree(T); cudaFree(dR);
	cudaFree(I); cudaFree(Corr); cudaFree(Img); cudaFree(echoRec);
	timeEnd = std::chrono::system_clock::now();
	elapsed_seconds = timeEnd - timeStart;
	printf("\"内存释放\":\t%f\n", elapsed_seconds);
}
void BP_Processing_CPU(cx_mat& echo, double fs, double kr, double tp, double prf, double lambda, double va, double r0, double rangeBin, double theta, double widthRange, double widthAzi, double deltaimr, double deltaima, double aziSection) {
	int Nr = echo.n_rows;
	int Na = echo.n_cols;
	double c = 3e8;
	double r0Center = r0 + Nr / 2.0 * rangeBin;
	double Ls = 2 * r0Center * tan(theta / 2);
	int naLs = fix(Ls / va / 2 * prf) * 2;
	int naSelect = fix(widthAzi / va * prf / 2) * 2 + naLs;
	int imnan = fix(widthAzi / deltaima / 2) * 2;
	int imnrn = fix(widthRange / deltaimr / 2) * 2;
	int itemp = Na / 2 + fix(aziSection / va * prf);
	int np = fix(tp * fs);
	rangeBin = c / 2 / fs;
	double thetaSq = 0, thetaBw = 0;
	echo = echo.submat(span::all, span(itemp - 1 * naSelect / 2 - 1, itemp + naSelect / 2 - 2));
	Na = naSelect;
	mat antx = arma::linspace(-naSelect / 2.0, naSelect / 2.0 - 1, naSelect) * (va / prf);
	mat anty = arma::linspace(-naSelect / 2.0, naSelect / 2.0 - 1, naSelect) * (0);
	mat antz = arma::linspace(-naSelect / 2.0, naSelect / 2.0 - 1, naSelect) * (0);
	mat rowTemp = arma::ones(imnrn, 1);
	mat colTemp = arma::linspace<rowvec>(-imnan / 2.0, imnan / 2.0 - 1, imnan);
	mat matx = (rowTemp * colTemp) * deltaima;
	rowTemp = arma::linspace(-imnrn / 2.0 + 1, imnrn / 2.0, imnrn);
	rowTemp = rowTemp * deltaimr + r0Center;
	colTemp = arma::ones(1, imnan);
	mat maty = rowTemp * colTemp;
	mat matz = arma::zeros(imnrn, imnan);
	mat fast = (arma::linspace(0, Nr - 1, Nr) - (Nr / 2.0)) * fs / Nr;
	mat matTemp = fast % fast * _Pi / kr;
	cx_mat H2 = cx_mat(arma::cos(matTemp), arma::sin(matTemp));
	matTemp = fast * _Pi * 2 * tp / 2;
	cx_mat HShift = cx_mat(arma::cos(matTemp), arma::sin(matTemp));
	cout << "run fft" << endl;
	for (size_t i = 0; i < Na; i++)
	{
		cx_mat f = echo.submat(span::all, span(i, i));
		f = arma::shift(f, floor(Nr / 2.0), 0);
		f = arma::fft(f);
		f = arma::shift(f, floor(Nr / 2.0), 0);
		f = f % HShift % H2;
		f = arma::shift(f, ceil(Nr / 2.0), 0);
		f = arma::ifft(f);
		f = arma::shift(f, ceil(Nr / 2.0), 0);
		echo.col(i) = f.col(0);
		cout << "run fft times" << i << endl;
	}
	cout << "run fft over" << endl;
	cx_mat echoDec = echo.submat(span(0, Nr - np - 1), span::all);
	Nr -= np;
	vec rvec = arma::linspace(0, Nr - 1, Nr) * rangeBin + r0;
	cx_mat img = cx_mat(arma::zeros(imnrn, imnan), arma::zeros(imnrn, imnan));
	cout << "run here" << endl;
	for (size_t i = 0; i < Na; i++)
	{
		mat dr = arma::sqrt((antx(i) - matx) % (antx(i) - matx)
			+ (anty(i) - maty) % (anty(i) - maty)
			+ (antz(i) - matz) % (antz(i) - matz));
		matTemp = dr * 4 * _Pi / lambda;
		cx_mat phCorr = cx_mat(arma::cos(matTemp), arma::sin(matTemp));
		uvec index = arma::find(dr > rvec.min() && dr < rvec.max());
		vec tr(Nr), ti(Nr), yr(index.n_elem), yi(index.n_elem);
		for (size_t j = 0; j < Nr; j++)
		{
			tr[j] = echoDec(j, i).real();
			ti[j] = echoDec(j, i).imag();
		}
		vec xx = dr(index);
		arma::interp1(rvec, tr, xx, yr);
		arma::interp1(rvec, ti, xx, yi);
		for (size_t ii = 0; ii < img.n_elem; ii++)
		{
			img(index(ii)) = {
				img(index(ii)).real() + yr(ii) * phCorr(index(ii)).real() - yi(ii) * phCorr(index(ii)).imag(),
				img(index(ii)).imag() + yr(ii) * phCorr(index(ii)).imag() + yi(ii) * phCorr(index(ii)).real()
			};
		}
		cout << "run here"<<i << endl;
	}
	cout << "run over" << endl;
	echo = img.resize(imnrn, imnan);
}
//void BP_Processing_CPU(cx_mat& echo, double fs, double kr, double tp, double prf, double lambda, double va, double r0, double rangeBin, double theta, double widthRange, double widthAzi, double deltaimr, double deltaima, double aziSection) {
//	timeStart = std::chrono::system_clock::now();
//	int Nr = echo.n_rows;
//	int Na = echo.n_cols;
//	double c = 3e8;
//	double r0Center = r0 + Nr / 2.0 * rangeBin;
//	double Ls = 2 * r0Center * tan(theta / 2);
//	int naLs = fix(Ls / va / 2 * prf) * 2;
//	int naSelect = fix(widthAzi / va * prf / 2) * 2 + naLs;
//	int imnan = fix(widthAzi / deltaima / 2) * 2;
//	int imnrn = fix(widthRange / deltaimr / 2) * 2;
//	int t = Na / 2 + fix(aziSection / va * prf);
//	int np = fix(tp * fs);
//	rangeBin = c / 2 / fs;
//	double thetaSq = 0, thetaBw = 0;
//	echo = echo.submat(span::all, span(t - 1 * naSelect / 2 - 1, t + naSelect / 2 - 2));
//	Na = naSelect;
//	double* antX, * antY, * antZ, * matX, * matY, * matZ;
//	size_t nas_double = naSelect * sizeof(double);
//	size_t imna_imnr_double = imnan * imnrn * sizeof(double);
//	cudaMalloc((void**)&antX, nas_double);
//	cudaMalloc((void**)&antY, nas_double);
//	cudaMalloc((void**)&antZ, nas_double);
//	cudaMalloc((void**)&matX, imna_imnr_double);
//	cudaMalloc((void**)&matY, imna_imnr_double);
//	cudaMalloc((void**)&matZ, imna_imnr_double);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"内存分配\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	AntInitial <<<naSelect / 64 + 1, 64 >>> (antX, antY, antZ, -naSelect / 2.0, naSelect / 2.0 - 1, naSelect, va, prf);
//	dim3 threadsPerBlock(32, 32);
//	dim3 numBlocks(Nr / threadsPerBlock.x + 1, Na / threadsPerBlock.y + 1);
//	MatInitial <<<numBlocks, threadsPerBlock >>> (matX, matY, matZ, imnrn, imnan, deltaima, deltaimr, r0Center);
//	cuDoubleComplex* Echo, * Fast, * H2, * HShift, * Temp;
//	size_t nr_na = Nr * Na * sizeof(double2);
//	cudaMalloc((void**)&Echo, nr_na);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"数据初始化\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	cudaMemcpyAsync(Echo, echo.mem, nr_na, cudaMemcpyHostToDevice);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"数据传输\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	cudaMalloc((void**)&Fast, Nr * sizeof(double2));
//	cudaMalloc((void**)&H2, Nr * sizeof(double2));
//	cudaMalloc((void**)&HShift, Nr * sizeof(double2));
//	cudaMalloc((void**)&Temp, Nr * sizeof(double2));
//	FastInitial <<<numBlocks, threadsPerBlock >>> (Fast, Nr, 1, fs, Nr);
//	H2Initial <<<numBlocks, threadsPerBlock >>> (H2, Fast, Nr, 1, kr);
//	HShiftInitial <<<numBlocks, threadsPerBlock >>> (HShift, Fast, Nr, 1, tp);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"矩阵初始化\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	for (size_t i = 0; i < Na; i++)
//	{
//		cx_mat ttt(Nr, 1);
//		cx_mat f = echo.submat(span::all, span(i, i));
//		f = arma::shift(f, floor(Nr / 2.0), 0);
//		f = arma::fft(f);
//		f = arma::shift(f, floor(Nr / 2.0), 0);
//		cudaMemcpy(ttt.memptr(), HShift, Nr * sizeof(double2), cudaMemcpyDeviceToHost);
//		f = f % ttt;
//		cudaMemcpy(ttt.memptr(), H2, Nr * sizeof(double2), cudaMemcpyDeviceToHost);
//		f = f % ttt;
//		f = arma::shift(f, ceil(Nr / 2.0), 0);
//		f = arma::ifft(f);
//		f = arma::shift(f, ceil(Nr / 2.0), 0);
//		echo.col(i) = f.col(0);
//		//mat_slice_col <<<numBlocks, threadsPerBlock >>> (Echo, Fast, Nr, Na, i, i);
//		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (Fast, Temp, Nr, 1);
//		//cuda_fft(Temp, Nr, 1, 0);
//		//mat_fftshift <<<numBlocks, threadsPerBlock >>> (Temp, Fast, Nr, 1);
//		//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (Fast, HShift, Fast, Nr, 1);
//		//mat_dot_muiltple <<<numBlocks, threadsPerBlock >>> (Fast, H2, Fast, Nr, 1);
//		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (Fast, Temp, Nr, 1);
//		//cuda_ifft(Temp, Nr, 1, 0);
//		//mat_ifftshift <<<numBlocks, threadsPerBlock >>> (Temp, Fast, Nr, 1);
//		//mat_set_col <<<numBlocks, threadsPerBlock >>> (Echo, Fast, Nr, Na, i);
//	}
//	cudaMemcpy(Echo, echo.memptr(), nr_na, cudaMemcpyHostToDevice);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"逐行FFT\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	double* rVec, * T, * dR, * I;
//	cuDoubleComplex* echoRec, * Corr, * Img;
//	cudaMalloc((void**)&echoRec, (Nr - np) * Na * sizeof(double2));
//	EchoReduceRow <<<numBlocks, threadsPerBlock >>> (Echo, echoRec, Nr, Na, np);
//	Nr -= np;
//	cudaMalloc((void**)&rVec, Nr * sizeof(double));
//	size_t imna_imnr_double2 = imnan * imnrn * sizeof(double2);
//	cudaMalloc((void**)&Img, imna_imnr_double2);
//	cudaMalloc((void**)&dR, imna_imnr_double);
//	cudaMalloc((void**)&I, imna_imnr_double);
//	cudaMalloc((void**)&T, Na * sizeof(double));
//	cudaMalloc((void**)&Corr, imna_imnr_double2);
//	RvecInitial <<<numBlocks, threadsPerBlock >>> (rVec, 0, Nr - 1, rangeBin, r0, Nr);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"插值数据初始化\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	mat dRmat(imnrn, imnan), xmat(imnrn, imnan), ymat(imnrn, imnan), zmat(imnrn, imnan);
//	vec rVecMat(Nr), antx(Na), anty(Na), antz(Na);
//	cx_mat echoMat(Nr, Na), imageMat(imnrn, imnan);
//	cudaMemcpy(echoMat.memptr(), echoRec, Nr * Na * sizeof(double2), cudaMemcpyDeviceToHost);
//	cudaMemcpy(xmat.memptr(), matX, imnan * imnrn * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(ymat.memptr(), matY, imnan * imnrn * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(zmat.memptr(), matZ, imnan * imnrn * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(antx.memptr(), antX, Na * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(anty.memptr(), antY, Na * sizeof(double), cudaMemcpyDeviceToHost);
//	cudaMemcpy(antz.memptr(), antZ, Na * sizeof(double), cudaMemcpyDeviceToHost);
//	for (size_t i = 0; i < Na; i++)
//	{
//		dRmat = arma::sqrt(
//			(antx[i] - xmat) % (antx[i] - xmat) + 
//			(anty[i] - ymat) % (anty[i] - ymat) +
//			(antz[i] - zmat) % (antz[i] - zmat));
//		rVecMat = arma::linspace(0, Nr - 1, Nr);
//		rVecMat = rVecMat * rangeBin;
//		rVecMat = rVecMat + r0;
//		uvec index = arma::find(dRmat > rVecMat.min() && dRmat < rVecMat.max());
//		vec tr(Nr), ti(Nr), yr(index.n_elem), yi(index.n_elem);
//		cx_vec img(index.n_elem);
//		for (size_t j = 0; j < Nr; j++)
//		{
//			tr[j] = echoMat[j, i].real();
//			ti[j] = echoMat[j, i].imag();
//		}
//		vec xx = dRmat(index);
//		yr = Linear(rVecMat, tr, xx);
//		yi = Linear(rVecMat, ti, xx);
//		for (size_t iele = 0; iele < index.n_elem; iele++)
//		{
//			img[iele] = { yr[iele],yi[iele] };
//		}
//		//CaldR_Corr <<<numBlocks, threadsPerBlock >>> (Corr, dR, matX, matY, matZ, antX, antY, antZ, i, lambda, imnrn, imnan);
//		//GetVecRange <<<numBlocks, threadsPerBlock >>> (dR, I, rMin, rMax, imnrn, imnan);
//		//AddImage_Interp <<<numBlocks, threadsPerBlock >>> (Img, I, rVec, echoRec, dR, Corr, i, Nr, imnrn, imnan);
//		for (size_t ii = 0; ii < img.n_elem; ii++)
//		{
//			imageMat(index[ii]) = {
//				imageMat(index[ii]).real() + yr[ii],
//				imageMat(index[ii]).imag() + yi[ii]
//			};
//		}
//	}
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"线性插值\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	echo = echo.submat(span(0, imnrn - 1), span(0, imnan - 1));
//	echo = cx_mat(imageMat);
//	//cudaMemcpyAsync(echo.memptr(), Img, imna_imnr_double2, cudaMemcpyDeviceToHost);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"数据传输\":\t%f\n", elapsed_seconds);
//	timeStart = std::chrono::system_clock::now();
//	cudaFree(antX); cudaFree(antY); cudaFree(antZ);
//	cudaFree(matX); cudaFree(matY); cudaFree(matZ);
//	cudaFree(Echo); cudaFree(Fast); cudaFree(H2); cudaFree(HShift); cudaFree(Temp);
//	cudaFree(rVec); cudaFree(T); cudaFree(dR);
//	cudaFree(I); cudaFree(Corr); cudaFree(Img); cudaFree(echoRec);
//	timeEnd = std::chrono::system_clock::now();
//	elapsed_seconds = timeEnd - timeStart;
//	printf("\"内存释放\":\t%f\n", elapsed_seconds);
//}
//PCHIP Interp1 Function
vec PCHIP(vec x, vec y, vec xq)
{
	int n = x.n_elem;
	int m = xq.n_elem;
	vec yq(m);
	double* xp, * yp, * h, * delta, * xqp, * yqp;
	cudaMalloc((void**)&xp, n * sizeof(double));
	cudaMalloc((void**)&yp, n * sizeof(double));
	cudaMalloc((void**)&h, n * sizeof(double));
	cudaMalloc((void**)&delta, n * sizeof(double));
	cudaMalloc((void**)&xqp, m * sizeof(double));
	cudaMalloc((void**)&yqp, m * sizeof(double));
	cudaMemcpy(xp, x.mem, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(yp, y.mem, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xqp, xq.mem, m * sizeof(double), cudaMemcpyHostToDevice);
	int threadPerBlock = 32;
	int blocks = m / threadPerBlock + 1;
	PCHIP_SLOPE <<<blocks, threadPerBlock >>> (xp, yp, h, delta, n);
	PCHIP_INTERP <<<blocks, threadPerBlock >>> (xp, yp, h, delta, xqp, yqp, n, m);
	cudaMemcpy(yq.memptr(), yqp, m * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(xp); cudaFree(yp); cudaFree(h); cudaFree(delta); cudaFree(xqp); cudaFree(yqp);
	return yq;
}
vec PCHIP_DEC(vec x, vec y, vec xq)
{
	int n = x.n_elem;
	int m = xq.n_elem;
	vec yq(m);
	double* xp, * yp, * h, * delta, * xqp, * yqp;
	cudaMalloc((void**)&xp, n * sizeof(double));
	cudaMalloc((void**)&yp, n * sizeof(double));
	cudaMalloc((void**)&h, n * sizeof(double));
	cudaMalloc((void**)&delta, n * sizeof(double));
	cudaMalloc((void**)&xqp, m * sizeof(double));
	cudaMalloc((void**)&yqp, m * sizeof(double));
	cudaMemcpy(xp, x.mem, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(yp, y.mem, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xqp, xq.mem, m * sizeof(double), cudaMemcpyHostToDevice);
	int threadPerBlock = 32;
	int blocks = m / threadPerBlock + 1;
	PCHIP_SLOPE <<<blocks, threadPerBlock >>> (xp, yp, h, delta, n);
	PCHIP_INTERP_Dec <<<blocks, threadPerBlock >>> (xp, yp, h, delta, xqp, yqp, n, m);
	cudaMemcpy(yq.memptr(), yqp, m * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(xp); cudaFree(yp); cudaFree(h); cudaFree(delta); cudaFree(xqp); cudaFree(yqp);
	return yq;
}
cx_vec PCHIP_CX(vec x, cx_vec y, vec xq)
{
	int n = x.n_elem;
	int m = xq.n_elem;
	cx_vec yq(m);
	double* xp, * h, * deltar, * deltai, * xqp;
	cuDoubleComplex* yp, * yqp;
	cudaMalloc((void**)&xp, n * sizeof(double));
	cudaMalloc((void**)&yp, n * sizeof(double2));
	cudaMalloc((void**)&h, n * sizeof(double));
	cudaMalloc((void**)&deltar, n * sizeof(double));
	cudaMalloc((void**)&deltai, n * sizeof(double));
	cudaMalloc((void**)&xqp, m * sizeof(double));
	cudaMalloc((void**)&yqp, m * sizeof(double2));
	cudaMemcpy(xp, x.mem, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(yp, y.mem, n * sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpy(xqp, xq.mem, m * sizeof(double), cudaMemcpyHostToDevice);
	int threadPerBlock = 32;
	int blocks = m / threadPerBlock + 1;
	PCHIP_SLOPE_CX <<<blocks, threadPerBlock >>> (xp, yp, h, deltar, deltai, n);
	PCHIP_INTERP_CX <<<blocks, threadPerBlock >>> (xp, yp, h, deltar, deltai, xqp, yqp, n, m);
	cudaMemcpy(yq.memptr(), yqp, m * sizeof(double2), cudaMemcpyDeviceToHost);
	cudaFree(xp); cudaFree(yp); cudaFree(h); cudaFree(deltar); cudaFree(deltai); cudaFree(xqp); cudaFree(yqp);
	return yq;
}
vec Linear(vec x, vec y, vec xq)
{
	vec yq(xq.n_elem);
	int n = x.n_elem;
	for (size_t ix = 0; ix < xq.n_elem; ix++)
	{
		float xqi = xq[ix];
		int i = 0;
		while (i < n - 1 && x[i + 1] < xqi)
			i++;
		float x0 = x[i];
		float x1 = x[i + 1];
		float y0 = y[i];
		float y1 = y[i + 1];
		yq[ix] = y0 + (y1 - y0) * (xqi - x0) / (x1 - x0);
	}
	return yq;
}