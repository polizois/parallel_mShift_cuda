/*
    Author: Polizois Siois 8535
*/
/*
		Faculty of Electrical and Computer Engineering AUTH
		3rd assignment at Parallel and Distributed Systems (7th semester)
*/
/*
		Parallel implementation of mean shift algorithm for running on nvidia GPUs using cuda.
		Give N number of points in a D-dimendional space, the program repeatedly makes NxN
		parallel calculations.In every step it finds vectors(mean shifts) that move the points to
		new positions which tend to be closer to the maxima of a predefined kernel function,
		the Gaussian.The repetitions stop when each point has moved close enough(depends on
		EPSILON) to the maxima.
*/
/*
		This iteration of mean shift uses the GPU's SHARED MEMORY to perfom reduction and
		speed up the process of calculating a sum of N doubles.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>


//set VAR=1 for the demo dataset (600x2)
//set VAR=0.1 for the dataset from the knn search exercise (60000x30) or the products of it
#define VAR 1 // σ^2
#define EPSILON 0.0001 // ε
#define THREADSPERBLOCK 128 // the number of threads in every block

//Used in main
double** alloc2d(int rows, int cols);
void loadFile(char name[65], double **x, int rows, int cols);
void showResults(double **x, int start, int end, int rows, int cols);
void exportResults(double **x, int rows, int cols, int threads, double dur, int iters);
double timeCalc(struct timeval start, struct timeval end);
void free2d(double **matrix);
int blockNum(int N, int thPerBlock);
int errors(double **y, int rows, int cols);


//Used inside kernels
__device__ double d_k_func(double x);
__device__ void d_reduce(double *sdata, double *out, int blockSize, int tid);

//Kernels
__global__ void tableCopy(double *from, double *to, int rows, int cols);
__global__ void colsToRow(double *from, double *to, int rows, int cols);
__global__ void rowToCols(double *from, double *to, int rows, int cols);
__global__ void yNext(double *x, double *y, double *out, int rows, int cols, int blocksForY);

struct timeval startwtime, endwtime; // Timer start and end value


int main(int argc, char *argv[])
{
	if(argc!=5) { printf("Wrong number of args\n"); return -1; }
	int ROWS = atoi(argv[1]);
	int COLS = atoi(argv[2]);
	char *FILENAME = argv[3];
	int EXPORT = atoi(argv[4]);

	int size2d = ROWS*COLS*sizeof(double);
	int bNum = blockNum(ROWS, THREADSPERBLOCK); // the number of blocks needed to store <<ROWS>>
	int tempRows = ROWS * bNum; // The number of blocks needed for ROWS*ROWS parallel calculations
	int i, j, k, con = 1, iters=0;
	double denom, dist=0;
  double nb1 = sqrt(tempRows);
	int nb = (int)nb1;
	if(nb1 > (double)(int)nb1) nb = (int)(nb1+1);
  //printf("Blocks per grid dimenson : %d\n", nb);
	double **x, // The points at their original positions
				 **y; // The points after they have moved towards the maxima
	double **temp; // stores the result of the reduction of every block
	double *num = (double*) malloc(COLS * sizeof(double));

	// Memory allocation in Host memory
	x = alloc2d(ROWS, COLS);
	y = alloc2d(ROWS, COLS);
	temp = alloc2d((COLS+1), tempRows);

  // Loading data from file to table
	loadFile(FILENAME, x, ROWS, COLS);

	// Memory allocation in Device memory
	double *dx;
	cudaMalloc(&dx, size2d);
	double *dy;
	cudaMalloc(&dy, size2d);
	double *dtemp;
	cudaMalloc(&dtemp, tempRows*(COLS+1)*sizeof(double));
	int *dcon;
	cudaMalloc(&dcon, 1*sizeof(int));

	// Copy points from host memory to device memory
	cudaMemcpy(dx, x[0], size2d, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(THREADSPERBLOCK, 1, 1); // Defining number of threads in a block (1d)
	dim3 numBlocks(blockNum(ROWS, threadsPerBlock.x), blockNum(ROWS, threadsPerBlock.x), 1); // Defining number of blocks in a grid (2d)

	//rearrange table data so that we have coalesced memory access
	colsToRow<<<numBlocks, threadsPerBlock>>>(dx, dy, ROWS, COLS); // Stores the transpose of dx to dy
	cudaThreadSynchronize(); // wait all threads to finish
	tableCopy<<<numBlocks, threadsPerBlock>>>(dy, dx, ROWS, COLS); // Copies dy to dx
	cudaThreadSynchronize(); // wait all threads to finish

	// Timer start
	gettimeofday( &startwtime, NULL ); // Starts timing the process (memory copies between device and host will be included)

	// Repeat until all mean shifts converge
	do
	{
		iters++;
		//printf("Iteration: %d\n", iters);
		dim3 threadsPerBlock(THREADSPERBLOCK, 1, 1);
		dim3 numBlocks(nb , nb, 1);

		//Reducing the sum parts for each new y from <<ROWS>> to <<bNum>> and storing them to dtemp
		yNext<<<numBlocks, threadsPerBlock, THREADSPERBLOCK*sizeof(double)>>>(dx, dy, dtemp, ROWS, COLS, blockNum(ROWS, THREADSPERBLOCK));
		cudaThreadSynchronize();

		con=1;
		cudaMemcpy(temp[0], dtemp, tempRows*(COLS+1)*sizeof(double), cudaMemcpyDeviceToHost);

		//Calculating every new y (using dtemp) and checking if the corresponding mean shift converges
		//printf("checking convergence\n");
		for(i=0;i<ROWS;i++)
		{
			dist=0;
      denom = 0; for(k=0;k<COLS;k++) num[k] = 0;
			for(j=0;j<bNum;j++)
			{
        for(k=0;k<COLS;k++) num[k] += temp[0][k*tempRows+i*bNum+j];
        denom += temp[0][COLS*tempRows+i*bNum+j];
			}
      for(k=0;k<COLS;k++) num[k] = num[k]/denom;
      for(k=0;k<COLS;k++) dist+=pow(y[0][k*ROWS+i]-num[k],2);
      dist = sqrt(dist);
      if (dist >= EPSILON) con=0;
      for(k=0;k<COLS;k++) y[0][k*ROWS+i] = num[k];
		}
		cudaMemcpy(dy, y[0], size2d, cudaMemcpyHostToDevice);
		//printf("done checking\n");

	}while(!con && iters <15);

	// Timer stop
	gettimeofday( &endwtime, NULL );

	// Test prints
	printf("Final positions\n");
	cudaMemcpy(y[0], dy, size2d, cudaMemcpyDeviceToHost);
	printf("first 5\n");
	showResults(y, 0, 5, ROWS, COLS);
	printf("last 5\n");
	showResults(y, ROWS-5, ROWS, ROWS, COLS);

	// Completion time show
	double duration = timeCalc(startwtime, endwtime);
	printf("Completed in %.3f sec !\n", duration);
	printf("Iteration num: %d\n", iters);

	// Exporting results
	if(EXPORT)
	{
		numBlocks.x = bNum; numBlocks.y = bNum; numBlocks.z = 1;
		rowToCols<<<numBlocks, threadsPerBlock>>>(dy, dx, ROWS, COLS);
		cudaMemcpy(x[0], dx, size2d, cudaMemcpyDeviceToHost);
		exportResults(x, ROWS, COLS, THREADSPERBLOCK, duration, iters);
	}

	// Checking for errors
	int errs = errors(y, ROWS, COLS);
	if(errs != -1) printf("Errors = %d\n", errs);

	// Freeing the allocated memory
	free2d(x); free2d(y); free2d(temp);
	cudaFree(dx); cudaFree(dy); cudaFree(dtemp);
}

// Calculates (and reduces to bNum) all parts of the sum of the new position of every point based on its former
// position and the inititial position of all the points.
// Reduction results stored in "out"
__global__ void yNext(double *x, double *y, double *out, int rows, int cols, int blocksForY)
{
	extern __shared__ double shared[]; //will be used for reuduction
	double *denom = (double*)shared;
	double *numer = (double*)shared;

  int bid = blockIdx.y*gridDim.x + blockIdx.x;
	int blockOfY = bid % blocksForY;
	int id = blockOfY*blockDim.x +threadIdx.x; // 0-ROWS

	int yRow = bid / blocksForY; // 0-ROWS
	double dist=0;
	int i, tempRows = rows*blocksForY;
	double inRows=0, inVar=0, gaus;

	int tid1d = threadIdx.x;

	if(id < rows && bid < tempRows) inRows=1;

	//Distance calculation and check
	for(i=0;i<cols;i++) dist+=pow(y[i*rows+yRow]-x[i*rows+id],2);
	dist = sqrt(dist);
	if(dist <= VAR) inVar=1;

	gaus = d_k_func(pow(dist, 2));

	// Every thread in a block(if in limits) fills the accornding place of denom[]
	if(bid < tempRows)
		denom[tid1d] = inRows * inVar * gaus;

	// When all threads are done filling, denom gets reduced to one sum and stored to the according place of out[][]
	__syncthreads();
	if(bid < tempRows)
		d_reduce(denom, &out[cols*tempRows+bid], blockDim.x, tid1d);

	// The exact same thing done here for every dimention(colum)
	for(i=0;i<cols;i++)
	{
		__syncthreads();

		if(bid < tempRows)
			numer[tid1d] = inRows * inVar * gaus * x[i*rows+id]; // rows x 1

		__syncthreads();
		if(bid < tempRows)
			d_reduce(numer, &out[i*tempRows+bid], blockDim.x, tid1d);
	}

}

//Gaussian kernel
__device__ double d_k_func(double x)
{
  return exp(-x/(2*VAR));
}

__device__ void d_reduce(double *sdata, double *out, int blockSize, int tid)
{
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

	if (tid < 32)
	{
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
	}

	if (tid == 0) *out = sdata[0];
}

__global__ void tableCopy(double *from, double *to, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
		int current = i*cols + j;
    if (i < rows && j < cols)
        to[current] = from[current];
}

__global__ void colsToRow(double *from, double *to, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
		int before = i*cols + j;
		int after = j*rows + i;
    if (i < rows && j < cols)
        to[after] = from[before];
}

__global__ void rowToCols(double *from, double *to, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // row
    int j = blockIdx.y * blockDim.y + threadIdx.y; // col
		int after = i*cols + j;
		int before = j*rows + i;
    if (i < rows && j < cols)
        to[after] = from[before];
}

int blockNum(int N, int thPerBlock)
{
	int num = N / thPerBlock,
		  mod = N % thPerBlock;

	if(N <= thPerBlock) num = 1;
	else if(mod) num += 1;

	return num;
}

// Allocates continuous memory for a 2d array of doubles
double** alloc2d(int rows, int cols)
{
	int i;

	double **matrix= (double**)malloc(rows * sizeof(*matrix));
	if(!matrix)
	{
   printf("Out of memory\n");
   exit(-1);
 }
	matrix[0] = (double*)malloc(rows * (cols) * sizeof(**matrix));
	if(!matrix[0])
	{
   printf("Out of memory\n");
   exit(-1);
 }
	for(i = 1; i < rows; i++)
		matrix[i] = matrix[0] + i * (cols);

	return matrix;
}

void free2d(double **matrix)
{
	free(matrix[0]);
	free(matrix);
}

void loadFile(char name[65], double **x, int rows, int cols)
{
	FILE *pointFile;
	int i;

	pointFile=fopen(name,"rb");
	if (!pointFile){ printf("Unable to open file!\n"); exit(1); }

	for (i=0; i < rows; i++)
		//Writing a row of coordinates
		if (!fread(&(x[i][0]),sizeof(double),cols,pointFile))
			{ printf("Unable to read from file!"); exit(1); }
	fclose(pointFile);
}


void showResults(double **x, int start, int end, int rows, int cols)
{
  int i,j;

  for(i=start;i<end;i++)
  {
    printf("%d:",i);
    for(j=0;j<cols;j++) printf(" %f ", x[0][j*rows+i]);
    printf("\n");
  }
}

void exportResults(double **x, int rows, int cols, int threads, double dur, int iters)
{
	FILE *out;
	int i;
	char name[65];
	//Generating the file name
	sprintf(name, "./results/y_(%d_%d)_(%d_%.3f_%d).bin", rows, cols, threads, dur, iters);

	out=fopen(name,"wb");
	if (!out){ printf("Unable to open file!\n"); exit(1); }

	for (i=0; i < rows; i++)
		//Writing a row of coordinates
		if (!fwrite(&(x[i][0]),sizeof(double),cols,out))
			{ printf("Unable to read from file!"); exit(1); }

	printf("Exported !\n");
	fclose(out);
}

double timeCalc(struct timeval start, struct timeval end)
{
	return (double)( ( end.tv_usec - start.tv_usec ) / 1.0e6 + end.tv_sec - start.tv_sec );
}

// Opens a binary file that has the results of a serial execution of mean shift for the same data
// Checks the points stored in y for errors, counts the errors and returns them
// The binary files used for comparisson should be stored in a folder called "compare" in the same
// directory with proggram
// For example if we want to test our result for the data set of 600 2-dimentional points we
// refer to the ./compare/600_2.bin file
int errors(double **y, int rows, int cols)
{
	FILE *data;
	double *tempLine;
	char fileName[650];
	int i,j, er=0;

	//Generating the file name
	sprintf(fileName, "./compare/%d_%d.bin", rows, cols);
	// Allocating space for the reading line
	tempLine = (double *) malloc(cols * sizeof(double));

	//Opening the label results binary file for reading
	data=fopen(fileName,"rb");
	if (!data){ printf("Unable to open file in order to compare results!\n"); return -1; }

	// Finding the correct place to start loadng
	//fseek(data, 0, SEEK_SET);

	// reading every line and checking if theres a difference between my results and those from matlab
	for (i=0; i < rows; i++)
	{
		//Loading a label
		if(!fread(tempLine, sizeof(double), cols, data))
			{ printf("Unable to read from file!\n"); return -1; }
		for(j=0;j<cols;j++)
			// comparing with 10 decimal percision
			if((int)(10000000000*tempLine[j]) != (int)(10000000000*y[0][j*rows+i])) { er++; break; }
	}
	//Closing the binary files
	fclose(data);

	return er;
}
