#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "propDevice.h"  //Biblioteca externa

//20/01/18
//Modificações em 21/03/19
//Pequena correção de normalização da distribuição de velocidades em 27/05/19
//Mudança nas taxas de decaimento em 12/09/19 (concordando com Alexandre)
//Sistema de quatro níveis com efeito Doppler
//3 campos incidentes e um campo gerado
//Paralelização nos grupos de átomos
//Se tem efeito Doppler, considere que o átomo viaja na direção aposta ao laser de diodo
//24/05/2020: Gera gráfico tridimensional rho33 em função de delta e v

#define Pi 3.141592653589793

#define blocks 20
#define threads 32
#define gpu 1
#define arq 3		    //número de arquivo para salvar

#define kMax 200000
#define passoVelocidade 0.195
#define pontosDelta 150
#define passoDelta 1

#define varr 'f'
//  d/f (diodo ou femto varrendo)
#define dopp 's'
//  s/n (com doppler = sim ou não)
#define sinal -1
//  1 --> copropagante , -1 --> contrapropagante
#define niveis 4	
//  3 ou 4 níveis

const double Diodo = (2 * Pi) * 12e6;
const double Femto = (2 * Pi) * 0.6e6;
__constant__ double Ai = 1;
__constant__ double Aa = 0;    //em rad/s

__constant__ double Bd = 0;
__constant__ double Bf = 0;
__constant__ double Bi = 0;
__constant__ double Ba = 0;    //em rad/s

__constant__ double deltai = 0;

#define kb 1.38e-23
#define m 1.4195e-25
#define T 353

#define gama22 (2*Pi)*6.06e6
#define gama33 (2*Pi)*660e3
#define gama44 (2*Pi)*1.3e6

__constant__ double kd = (2 * Pi) / 780e-9;
__constant__ double kf = (2 * Pi) / 776e-9 * (sinal);
__constant__ double ka = (2 * Pi) / 420e-9;
__constant__ double ki = (2 * Pi) / 5300e-9;

__constant__ double h = 5e-12;    //*10000/kMax;

__constant__ double gama12 = 0.5 * gama22;
__constant__ double gama13 = 0.5 * gama33;
__constant__ double gama14 = 0.5 * gama44;
__constant__ double gama24 = 0.5 * (gama22 + gama44);
__constant__ double gama23 = 0.5 * (gama22 + gama33);
__constant__ double gama43 = 0.5 * (gama33 + gama44);

__constant__ int nucleos = blocks * threads;

__constant__ char d_dopp = dopp;
char h_dopp = dopp;
__constant__ char d_var = varr;
char h_var = varr;

__constant__ double a10 = 1;                     //população inicial do estado 1
__constant__ double a20 = 0;                     //população inicial do estado 2
__constant__ double a30 = 0;                     //população inicial do estado 3
__constant__ double a40 = 0;                     //população inicial do estado 4

#define CUDA_ERROR_CHECK
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		system("pause");
		exit(-1);
	}
#endif
	return;
}

__device__ double f(double a11, double a22, double a33, double a44, double a12, double b12, double a23, double b23, double a14, double b14,
	double a43, double b43, double a13, double b13, double a24, double b24, double deltad, double deltaf, double deltaa, int j,
	double v, double Ad, double Af)  //sistema de 4 níveis
{
	/*a11*/ if (j == 1)  return 2 * Ad * b12 - 2 * Bd * a12 + 2 * Aa * b14 - 2 * Ba * a14 + gama22 * a22 + gama44 * a44;				   //a11
	/*a22*/ if (j == 2)  return -2 * Ad * b12 + 2 * Bd * a12 + 2 * Af * b23 - 2 * Bf * a23 - gama22 * a22 + (2.05 - 0.35 * niveis) * gama33 * a33;        //a22
	/*a33*/ if (j == 3)  return -2 * Ai * b43 + 2 * Bi * a43 - 2 * Af * b23 + 2 * Bf * a23 - gama33 * a33;									   //a33
	/*a44*/ if (j == 4)  return 2 * Ai * b43 - 2 * Bi * a43 - 2 * Aa * b14 + 2 * Ba * a14 - gama44 * a44 + 0.35 * (niveis - 3) * gama33 * a33;		   //a44

	/*a12*/ if (j == 5)  return -gama12 * a12 - (deltad - kd * v) * b12 + Aa * b24 + Af * b13 - Ba * a24 - Bf * a13 - Bd * (a22 - a11); //a12
	/*b12*/ if (j == 6)  return -gama12 * b12 + (deltad - kd * v) * a12 + Aa * a24 - Af * a13 + Ba * b24 - Bf * b13 + Ad * (a22 - a11); //b12
	/*a23*/ if (j == 7)  return -gama23 * a23 - (deltaf - kf * v) * b23 - Ad * b13 + Ai * b24 + Bd * a13 + Bi * a24 - Bf * (a33 - a22); //a23
	/*b23*/ if (j == 8)  return -gama23 * b23 + (deltaf - kf * v) * a23 + Ad * a13 - Ai * a24 + Bd * b13 + Bi * b24 + Af * (a33 - a22); //b23
	/*a14*/ if (j == 9)  return -gama14 * a14 - (deltaa - ka * v) * b14 - Ad * b24 + Ai * b13 - Bd * a24 - Bi * a13 - Ba * (a44 - a11); //a14
	/*b14*/ if (j == 10) return -gama14 * b14 + (deltaa - ka * v) * a14 + Ad * a24 - Ai * a13 - Bd * b24 - Bi * b13 + Aa * (a44 - a11); //b14
	/*a43*/ if (j == 11) return -gama43 * a43 - (deltai - ki * v) * b43 - Af * b24 - Aa * b13 + Bf * a24 + Ba * a13 - Bi * (a33 - a44); //a43
	/*b43*/ if (j == 12) return -gama43 * b43 + (deltai - ki * v) * a43 - Af * a24 + Aa * a13 - Bf * b24 + Ba * b13 + Ai * (a33 - a44); //b43

	/*a13*/ if (j == 13) return -gama13 * a13 - (deltad + deltaf - (kd + kf) * v) * b13 - Aa * b43 + Af * b12 + Ai * b14 - Ad * b23 - Ba * a43 + Bf * a12 + Bi * a14 - Bd * a23; //a13
	/*b13*/ if (j == 14) return -gama13 * b13 + (deltad + deltaf - (kd + kf) * v) * a13 + Aa * a43 - Af * a12 - Ai * a14 + Ad * a23 - Ba * b43 + Bf * b12 + Bi * b14 - Bd * b23; //b13
	/*a24*/ if (j == 15) return -gama24 * a24 - (deltaa - deltad - (ka - kd) * v) * b24 - Ad * b14 + Af * b43 + Ai * b23 - Aa * b12 + Ba * a12 - Bf * a43 - Bi * a23 + Bd * a14; //a24
	/*b24*/ if (j == 16) return -gama24 * b24 + (deltaa - deltad - (ka - kd) * v) * a24 + Ad * a14 + Af * a43 - Ai * a23 - Aa * a12 - Ba * b12 + Bf * b43 - Bi * b23 + Bd * b14; //b24
}

__global__ void Kernel(double* a11, double* a22, double* a33, double* a44, double* a12, double* b12, double* a23, double* b23, double* a14, double* b14,
	double* a43, double* b43, double* a13, double* b13, double* a24, double* b24, double* variavel, double* Ad, double* Af)
{
	//Paralelização nos grupos de átomos (variável v)

	int j, k;
	double k1[17], k2[17], k3[17], k4[17];

	double deltad, deltaf, deltaa, v, Add, Aff;

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	Add = Ad[0]; Aff = Af[0];

	if (d_var == 'd')
	{
		deltaf = 0;
		deltad = variavel[0];
	}
	if (d_var == 'f')
	{
		deltad = 0;
		deltaf = variavel[0];
	}

	deltaa = deltad + deltaf - deltai;

	v = (i - nucleos / 2) * passoVelocidade;

	a11[i] = 1;

	for (k = 1; k <= kMax - 1; k++)    //abre loop de k (temporal)
	{
		for (j = 1; j <= 16; j++)
			k1[j] = f(a11[i], a22[i], a33[i], a44[i], a12[i], b12[i], a23[i], b23[i], a14[i], b14[i], a43[i], b43[i],
				a13[i], b13[i], a24[i], b24[i], deltad, deltaf, deltaa, j, v, Add, Aff);

		for (j = 1; j <= 16; j++)
			k2[j] = f(a11[i] + k1[1] * h / 2, a22[i] + k1[2] * h / 2, a33[i] + k1[3] * h / 2, a44[i] + k1[4] * h / 2, a12[i] + k1[5] * h / 2, b12[i] + k1[6] * h / 2, a23[i] + k1[7] * h / 2, b23[i] + k1[8] * h / 2,
				a14[i] + k1[9] * h / 2, b14[i] + k1[10] * h / 2, a43[i] + k1[11] * h / 2, b43[i] + k1[12] * h / 2, a13[i] + k1[13] * h / 2, b13[i] + k1[14] * h / 2,
				a24[i] + k1[15] * h / 2, b24[i] + k1[16] * h / 2, deltad, deltaf, deltaa, j, v, Add, Aff);

		for (j = 1; j <= 16; j++)
			k3[j] = f(a11[i] + k2[1] * h / 2, a22[i] + k2[2] * h / 2, a33[i] + k2[3] * h / 2, a44[i] + k2[4] * h / 2, a12[i] + k2[5] * h / 2, b12[i] + k2[6] * h / 2, a23[i] + k2[7] * h / 2, b23[i] + k2[8] * h / 2,
				a14[i] + k2[9] * h / 2, b14[i] + k2[10] * h / 2, a43[i] + k2[11] * h / 2, b43[i] + k2[12] * h / 2, a13[i] + k2[13] * h / 2, b13[i] + k2[14] * h / 2,
				a24[i] + k2[15] * h / 2, b24[i] + k2[16] * h / 2, deltad, deltaf, deltaa, j, v, Add, Aff);

		for (j = 1; j <= 16; j++)
			k4[j] = f(a11[i] + k3[1] * h, a22[i] + k3[2] * h, a33[i] + k3[3] * h, a44[i] + k3[4] * h, a12[i] + k3[5] * h, b12[i] + k3[6] * h, a23[i] + k3[7] * h, b23[i] + k3[8] * h, a14[i] + k3[9] * h, b14[i] + k3[10] * h,
				a43[i] + k3[11] * h, b43[i] + k3[12] * h, a13[i] + k3[13] * h, b13[i] + k3[14] * h, a24[i] + k3[15] * h, b24[i] + k3[16] * h, deltad, deltaf, deltaa, j, v, Add, Aff);

		a11[i] = a11[i] + h * (k1[1] / 6 + k2[1] / 3 + k3[1] / 3 + k4[1] / 6);	   a22[i] = a22[i] + h * (k1[2] / 6 + k2[2] / 3 + k3[2] / 3 + k4[2] / 6);
		a33[i] = a33[i] + h * (k1[3] / 6 + k2[3] / 3 + k3[3] / 3 + k4[3] / 6);	   a44[i] = a44[i] + h * (k1[4] / 6 + k2[4] / 3 + k3[4] / 3 + k4[4] / 6);
		a12[i] = a12[i] + h * (k1[5] / 6 + k2[5] / 3 + k3[5] / 3 + k4[5] / 6);     b12[i] = b12[i] + h * (k1[6] / 6 + k2[6] / 3 + k3[6] / 3 + k4[6] / 6);
		a23[i] = a23[i] + h * (k1[7] / 6 + k2[7] / 3 + k3[7] / 3 + k4[7] / 6);     b23[i] = b23[i] + h * (k1[8] / 6 + k2[8] / 3 + k3[8] / 3 + k4[8] / 6);
		a14[i] = a14[i] + h * (k1[9] / 6 + k2[9] / 3 + k3[9] / 3 + k4[9] / 6);     b14[i] = b14[i] + h * (k1[10] / 6 + k2[10] / 3 + k3[10] / 3 + k4[10] / 6);
		a43[i] = a43[i] + h * (k1[11] / 6 + k2[11] / 3 + k3[11] / 3 + k4[11] / 6); b43[i] = b43[i] + h * (k1[12] / 6 + k2[12] / 3 + k3[12] / 3 + k4[12] / 6);
		a13[i] = a13[i] + h * (k1[13] / 6 + k2[13] / 3 + k3[13] / 3 + k4[13] / 6); b13[i] = b13[i] + h * (k1[14] / 6 + k2[14] / 3 + k3[14] / 3 + k4[14] / 6);
		a24[i] = a24[i] + h * (k1[15] / 6 + k2[15] / 3 + k3[15] / 3 + k4[15] / 6); b24[i] = b24[i] + h * (k1[16] / 6 + k2[16] / 3 + k3[16] / 3 + k4[16] / 6);
	}  //loop tempo
}

int main()
{
	clock_t begin, end;
	double time_spent;
	begin = clock();

	const int nucleos = blocks * threads;
	FILE* arquivo[arq];
	arquivo[0] = fopen("dados-rho33.dat", "w");
	arquivo[1] = fopen("dados-Re-rho14.dat", "w");
	arquivo[2] = fopen("dados-Im-rho14.dat", "w");

	//fprintf(arquivo[0], "\\g(r)\\-(33)");    fprintf(arquivo[1], "l\\g(s)\\-(14)l\\+(2)");

	propriedades();	//Biblioteca externa

	printf("Blocks = %d\n", blocks);
	printf("Threads = %d\n\n", threads);

	printf("Calculando...\n");

	double rho14, rho13, rho33, Rerho14, Imrho14;
	double v, pesoDoppler;

	double a[17][nucleos];
	double variavel[1], Ad[1], Af[1];;
	int p, q;

	double* dev_a11[gpu], * dev_a22[gpu];
	double* dev_a33[gpu], * dev_a44[gpu];
	double* dev_a12[gpu], * dev_b12[gpu];
	double* dev_a23[gpu], * dev_b23[gpu];
	double* dev_a14[gpu], * dev_b14[gpu];
	double* dev_a43[gpu], * dev_b43[gpu];
	double* dev_a13[gpu], * dev_b13[gpu];
	double* dev_a24[gpu], * dev_b24[gpu];
	double* dev_Ad[gpu], * dev_Af[gpu];
	double* dev_variavel[gpu];

	int bytes = nucleos * sizeof(double);
	cudaStream_t stream[gpu];

	Ad[0] = Diodo;
	Af[0] = Femto;

	for (p = -pontosDelta; p <= pontosDelta; p++)  //loop diodo
	{
		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);

			variavel[0] = (2 * Pi) * (gpu * p + pp) * passoDelta * 1e6;
			printf("%d, ", gpu * p + pp);

			cudaMalloc((void**)&dev_a11[pp], bytes); cudaMalloc((void**)&dev_a22[pp], bytes);
			cudaMalloc((void**)&dev_a33[pp], bytes); cudaMalloc((void**)&dev_a44[pp], bytes);
			cudaMalloc((void**)&dev_a12[pp], bytes); cudaMalloc((void**)&dev_b12[pp], bytes);
			cudaMalloc((void**)&dev_a23[pp], bytes); cudaMalloc((void**)&dev_b23[pp], bytes);
			cudaMalloc((void**)&dev_a14[pp], bytes); cudaMalloc((void**)&dev_b14[pp], bytes);
			cudaMalloc((void**)&dev_a43[pp], bytes); cudaMalloc((void**)&dev_b43[pp], bytes);
			cudaMalloc((void**)&dev_a13[pp], bytes); cudaMalloc((void**)&dev_b13[pp], bytes);
			cudaMalloc((void**)&dev_a24[pp], bytes); cudaMalloc((void**)&dev_b24[pp], bytes);
			cudaMalloc((void**)&dev_Ad[pp], 1 * sizeof(double));
			cudaMalloc((void**)&dev_Af[pp], 1 * sizeof(double));
			cudaMalloc((void**)&dev_variavel[pp], 1 * sizeof(double));

			cudaStreamCreate(&stream[pp]);

			for (q = 0; q <= nucleos - 1; q++)
			{
				a[1][q] = a10; a[2][q] = a20;
				a[3][q] = a30; a[4][q] = a40;
				for (int k = 5; k <= 16; k++)
					a[k][q] = 0;
			}

			cudaMemcpyAsync(dev_a11[pp], a[1], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_a22[pp], a[2], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a33[pp], a[3], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_a44[pp], a[4], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a12[pp], a[5], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b12[pp], a[6], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a23[pp], a[7], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b23[pp], a[8], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a14[pp], a[9], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b14[pp], a[10], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a43[pp], a[11], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b43[pp], a[12], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a13[pp], a[13], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b13[pp], a[14], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_a24[pp], a[15], bytes, cudaMemcpyHostToDevice, stream[pp]); cudaMemcpyAsync(dev_b24[pp], a[16], bytes, cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_variavel[pp], variavel, 1 * sizeof(double), cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_Ad[pp], Ad, 1 * sizeof(double), cudaMemcpyHostToDevice, stream[pp]);
			cudaMemcpyAsync(dev_Af[pp], Af, 1 * sizeof(double), cudaMemcpyHostToDevice, stream[pp]);
		}

		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);
			Kernel << < blocks, threads, 0, stream[pp] >> > (dev_a11[pp], dev_a22[pp], dev_a33[pp], dev_a44[pp], dev_a12[pp], dev_b12[pp], dev_a23[pp], dev_b23[pp], dev_a14[pp],
				dev_b14[pp], dev_a43[pp], dev_b43[pp], dev_a13[pp], dev_b13[pp], dev_a24[pp], dev_b24[pp], dev_variavel[pp], dev_Ad[pp], dev_Af[pp]);
		}

		//CudaCheckError();

		for (int pp = 0; pp <= gpu - 1; pp++)
		{
			cudaSetDevice(pp);
			cudaMemcpyAsync(a[1], dev_a11[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[2], dev_a22[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[3], dev_a33[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[4], dev_a44[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[5], dev_a12[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[6], dev_b12[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[7], dev_a23[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[8], dev_b23[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[9], dev_a14[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[10], dev_b14[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[11], dev_a43[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[12], dev_b43[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[13], dev_a13[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[14], dev_b13[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(a[15], dev_a24[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]); cudaMemcpyAsync(a[16], dev_b24[pp], bytes, cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(variavel, dev_variavel[pp], 1 * sizeof(double), cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(Ad, dev_Ad[pp], 1 * sizeof(double), cudaMemcpyDeviceToHost, stream[pp]);
			cudaMemcpyAsync(Af, dev_Af[pp], 1 * sizeof(double), cudaMemcpyDeviceToHost, stream[pp]);

			//cudaDeviceSynchronize();

			for (q = 0; q <= nucleos - 1; q++)
			{

				v = (q - nucleos / 2) * passoVelocidade;
				pesoDoppler = passoVelocidade * sqrt(m / (2 * Pi * kb * T)) * exp(-m * v * v / (2 * kb * T));

				//rho13 = (pow(a[13][q], 2) + pow(a[14][q], 2)) * pesoDoppler;
				//rho14 = (pow(a[9][q], 2) + pow(a[10][q], 2)) * pesoDoppler;
				rho33 = a[3][q] * pesoDoppler;
				Rerho14 = a[9][q] * pesoDoppler;
				Imrho14 = a[10][q] * pesoDoppler;
				fprintf(arquivo[0], "%g ", rho33);
				fprintf(arquivo[1], "%g ", Rerho14);
				fprintf(arquivo[2], "%g ", Imrho14);
			}
			for (int ss = 0; ss <= 2; ss++)
				fprintf(arquivo[ss], "\n");
		}
	} //loop diodo

	for (int pp = 0; pp <= gpu - 1; pp++)
	{
		cudaSetDevice(pp);
		cudaFree(dev_a11[pp]); cudaFree(dev_a22[pp]);
		cudaFree(dev_a33[pp]); cudaFree(dev_a44[pp]);
		cudaFree(dev_a12[pp]); cudaFree(dev_b12[pp]);
		cudaFree(dev_a23[pp]); cudaFree(dev_b23[pp]);
		cudaFree(dev_a14[pp]); cudaFree(dev_b14[pp]);
		cudaFree(dev_a43[pp]); cudaFree(dev_b43[pp]);
		cudaFree(dev_a13[pp]); cudaFree(dev_b13[pp]);
		cudaFree(dev_a24[pp]); cudaFree(dev_b24[pp]);
		cudaFree(dev_variavel[pp]);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	for (int ss = 0; ss <= 2; ss++)
		fclose(arquivo[ss]);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	if (time_spent <= 60) printf("\nTempo de execucao = %f s\n\n", time_spent);
	if (time_spent > 60 && time_spent <= 3600) printf("\nTempo de execucao = %f min\n\n", time_spent / 60);
	if (time_spent > 3600) printf("\nTempo de execucao = %f h\n\n", time_spent / 3600);

	printf("\a");
	//system("pause");
}