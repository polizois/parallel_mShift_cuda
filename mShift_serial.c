#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define VAR 1 // σ^2
#define EPSILON 0.0001 // ε

int meanShift(double **x, double **y, int rows, int cols);
void copyArr(double **from, double **to, int rows, int cols);
void copyVect(double *from, double *to, int size);
double distance(double *pointA,double  *pointB,int size);
void vectFill(double *v, double fill, int size);
void mult(double num, double *v, double *to, int size);
void vectSum(double *vA, double *vB,double *to, int size);
void yNext(double *y, double **x, double *nextY, int rows, int cols, int neigh);
double k_func(double x);
double** alloc2d(int rows, int cols);
void loadFile(char name[65], double **x, int rows, int cols);
void showResults(double **y_new, int start, int end, int cols);
void exportResults(double **x, int rows, int cols, double dur, int iters);
double timeCalc(struct timeval start, struct timeval end);
void free2d(double **matrix);

struct timeval startwtime, endwtime;


int main(int argc, char *argv[])
{
	if(argc!=5) { printf("Wrong number of args\n"); return -1; }
	int ROWS = atoi(argv[1]);
	int COLS = atoi(argv[2]);
	char *FILENAME = argv[3];
	int EXPORT = atoi(argv[4]);
	int i,j;
	double **x, // The points at their original positions
				 **y; // The points after they have moved towards the maxima

	// Memory allocation
	x = alloc2d(ROWS, COLS);
	y = alloc2d(ROWS, COLS);

  // Loading data from file to table
	loadFile(FILENAME, x, ROWS, COLS);

	// Timer start
	gettimeofday( &startwtime, NULL );
	// meanShift execution
	int iters = meanShift(x, y, ROWS, COLS);
	// Timer stop
	gettimeofday( &endwtime, NULL );
	// Completion time show
	double duration = timeCalc(startwtime, endwtime);
	printf("Completed in %.3f sec !\n", duration);

	printf("Final positions\n");
	printf("First 5\n");
	showResults(y, 0, 5, COLS);
	printf("Last 5\n");
	showResults(y, ROWS-5, ROWS, COLS);
	if(EXPORT) exportResults(y, ROWS, COLS, duration, iters);


	//free allocated memory
	free2d(x);
	free2d(y);
}

int meanShift(double **x, double **y, int rows, int cols)
{
	int i,j,con, iters=0;
  double *currY, *nextY;

	// Memory allocations
	currY = (double *) malloc(cols * sizeof(double));
	nextY = (double *) malloc(rows * sizeof(double));

	// Initialize y with the original positions
  copyArr(x, y, rows, cols);

  do
	{
		iters++;
		con = 1;
		for(i=0;i<rows;i++)
	  {
				//copyVect(y[i], nextY, cols);
				//copyVect(nextY, currY, cols);
	      yNext(y[i], x, nextY, rows, cols, rows);
	      if (distance(nextY, y[i], cols) >= EPSILON) con = 0;
	      copyVect(nextY, y[i], cols);
	  }
	}while(!con);

	free(currY);
	free(nextY);
	return iters;
}


void copyArr(double **from, double **to, int rows, int cols)
{
	int i;

	for(i=0;i<rows;i++)
		copyVect(from[i], to[i], cols);
}

void copyVect(double *from, double *to, int size)
{
		int i;
		for(i=0;i<size;i++)
			to[i] = from[i];
}

// Calculates and returns the euclidean distance between 2 points
double distance(double *pointA,double  *pointB,int size)
{
	int i;
	double sum=0;

	for(i=0;i<size;i++)
		sum += pow(pointA[i]-pointB[i], 2);

	return sqrt(sum);
}

void vectFill(double *v, double fill, int size)
{
	int i;

	for(i=0;i<size;i++)
		v[i] = fill;
}

void mult(double num, double *v, double *to, int size)
{
	int i;

	for(i=0;i<size;i++)
			to[i] = num * v[i];
}

void vectSum(double *vA, double *vB,double *to, int size)
{
	int i;

	for(i=0;i<size;i++)
		to[i] = vA[i] + vB[i];
}

// Calculates a new position for a point based on its former
//position and the inititial position of all the points
//(Weighted mean)
void yNext(double *y, double **x, double *nextY, int rows, int cols, int neigh)
{
	double *numerator, *tempVect;
	double temp, denominator = 0, dis;
	int i;

	// Memry allocation
	numerator = (double *) malloc(cols * sizeof(double));
	tempVect = (double *) malloc(cols * sizeof(double));
	// Initialize numerator with zeros
	vectFill(numerator, 0, cols);


	for(i=0;i<neigh;i++)
	{
    dis = distance(y,x[i], cols);
		if(dis <= VAR)
    {
      temp = k_func(pow(dis, 2));
      mult(temp, x[i], tempVect, cols);
  		vectSum(numerator, tempVect, numerator, cols);
  		denominator += temp;
    }
	}
  mult(1/denominator, numerator, nextY, cols);

	free(numerator);
	free(tempVect);
}

//Gaussian kernel
double k_func(double x)
{
  return exp(-x/(2*VAR));
}

// Allocates continuous memory for a 2d array of doubles
double** alloc2d(int rows, int cols)
{
	int i;

	double **matrix= malloc(rows * sizeof(*matrix));
	if(!matrix)
	{
   printf("Out of memory\n");
   exit(-1);
 }
	matrix[0] = malloc(rows * (cols) * sizeof(**matrix));
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

void showResults(double **y_new, int start, int end, int cols)
{
  int i,j;

  for(i=start; i<end; i++)
  {
		printf("%d: ", i);
    for (j=0; j<cols; j++)
      printf("%f ", y_new[i][j]);
    printf("\n");
  }
}

void exportResults(double **x, int rows, int cols, double dur, int iters)
{
	FILE *out;
	int i;
	char name[65];
	//Generating the file name
	sprintf(name, "./compare/%d_%d.bin", rows, cols);

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
