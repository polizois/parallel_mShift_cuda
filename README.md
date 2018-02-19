# parallel_mShift_cuda
Faculty of Electrical and Computer Engineering AUTH  
3rd assignment at Parallel and Distributed Systems (7th semester)
  
Three **versions** of the meanShift program provided:
* Parallel version using cuda with shared memory (*mShift_cuda_shared.cu*)
* Parallel version using cuda without shared memory (*mShift_cuda_non.cu*)
* Serial versiom (*mShift_serial.c*)
  
In order to **compile** the programs run the following commands
* For mShift_cuda_shared.cu run : *nvcc mShift_cuda_shared.cu -o mShift_cuda_shared.o -lm -O3*
* For mShift_cuda_non.cu run    : *nvcc mShift_cuda_non.cu -o mShift_cuda_non.o -lm -O3*
* For mShift_serial.c run       : *gcc mShift_serial.c -o mShift_serial.o -lm -O3*

Every program takes 4 arguements:
1. Number of points of the data set
2. Number of dimensions of every point
3. Dataset name with full path (ex. *./data/points_600_2.bin*)
4. 1 or 0 depending wether you want the results(new positions of the points) to be exported or not (1=yes, 0=no)

The 2 parallel programs export their results in a folder named *"results"*.
The serial program exports its results in s folder named *"compare"*.

In order to check if the parallel programs are correct, after they have run meanShift, their results are compared with
the results from the equivalent serial execution found in *"compare"* folder.Any results not provided in this folder, can be
created by running the serial program with the export flag set to 1.
