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
    long long int number_in_circle_sum = 0;
    number_in_circle_sum += number_in_circle;

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Send(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request* requests = new MPI_Request[world_size - 1];
        long long int* recv_data = new long long int[world_size - 1];

        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&recv_data[i - 1], 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
        }

        // Wait for all receive operations to complete
        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < world_size - 1; i++) {
            number_in_circle_sum += recv_data[i];
        }
    }

    if (world_rank == 0)
    {
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
