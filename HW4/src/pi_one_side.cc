#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int monte_carlo(long long int my_n_tosses, int &world_rank) {
    long long int my_number_in_circle = 0;
    unsigned int seed = world_rank;
    for (long long int toss = 0; toss < my_n_tosses; toss ++) {
        double x = ((double) rand_r(&seed)*2 / RAND_MAX) - 1.0;
        double y = ((double) rand_r(&seed)*2 / RAND_MAX) - 1.0;
        if ( x * x + y * y <= 1){
            my_number_in_circle++;
        }
    }

    return my_number_in_circle;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---
    // MPI_Comm nodecomm;                                                          
    // MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodecomm);    
    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int number_in_circle = monte_carlo(tosses / world_size, world_rank);
    
    // Allocate shared memory
    // long long int *number_in_circle_sum;
    // MPI_Win_allocate_shared(sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, nodecomm, &number_in_circle_sum, &win);

    // // Initialize the shared memory
    // if (world_rank == 0) {
    //     *number_in_circle_sum = 0;
    // }

    // // Start the access/exposure epoch
    // MPI_Win_fence(0, win);

    // // All processes, including the root, accumulate their results into the shared memory
    // MPI_Accumulate(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, 1, MPI_LONG_LONG_INT, MPI_SUM, win);

    // // End the access/exposure epoch
    // MPI_Win_fence(0, win);
    long long int *number_in_circle_sum;
    if (world_rank == 0) {
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &number_in_circle_sum);
        for (int i = 0; i < world_size; i++) {
            number_in_circle_sum[i] = 0;
        }
        MPI_Win_create(number_in_circle_sum, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    } else {
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }

    // Add a while loop to prevent deadlock
    int flag = 0;
    MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
    MPI_Put(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, world_rank, 1, MPI_LONG_LONG_INT, win);
    MPI_Win_flush(0, win);  // Ensure that the MPI_Put operation is complete
    MPI_Win_unlock(0, win);
    MPI_Win_fence(0, win);
 
    if (world_rank == 0)
    {
        // TODO: handle PI result
        long long int total_number_in_circle = 0;
        for (int i = 0; i < world_size; i++) {
            total_number_in_circle += number_in_circle_sum[i];
        }
        pi_result = 4 * total_number_in_circle / (double) tosses;
        // pi_result = double( 4 * (*number_in_circle_sum) / (double) tosses );
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
        MPI_Free_mem(number_in_circle_sum);
    }
    
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}