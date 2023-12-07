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

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    long long int number_in_circle = monte_carlo(tosses / world_size, world_rank);
    
    long long int* gather_data = nullptr;
    if (world_rank == 0) {
        gather_data = new long long int[world_size];
    }

    // Perform the gather operation
    // gather(send_data, send_count, send_datatype, recv_data, recv_count, recv_datatype, root, communicator)
    MPI_Gather(&number_in_circle, 1, MPI_LONG_LONG_INT, gather_data, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);


    // TODO: use MPI_Gather

    if (world_rank == 0)
    {
        long long int number_in_circle_sum = 0;
        for (int i = 0; i < world_size; i++) {
            number_in_circle_sum += gather_data[i];
        }
        // TODO: PI result
        pi_result = double( 4 * number_in_circle_sum / (double) tosses );
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
