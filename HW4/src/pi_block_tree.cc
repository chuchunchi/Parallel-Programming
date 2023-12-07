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
    int mul = 1;
    // TODO: binary tree redunction
    while(world_rank / mul != 0){
        if(world_rank / mul % 2 == 0){
            MPI_Recv(&number_in_circle, 1, MPI_LONG_LONG_INT, world_rank + 1 * mul, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Hello world from rank %d, recv %lld from %d, get total %lld\n", world_rank, number_in_circle, world_rank + 1 * mul, number_in_circle_sum);
            number_in_circle_sum += number_in_circle;
            mul *= 2;
        }
        else{
            // printf("Hello world from rank %d, send %lld to %d\n", world_rank, number_in_circle, world_rank - 1 * mul);
            MPI_Send(&number_in_circle_sum, 1, MPI_LONG_LONG_INT, world_rank - 1 * mul, 0, MPI_COMM_WORLD);
            break;
        }
        
    }
    
    if (world_rank == 0)
    {
        while(mul < world_size){
            // printf("Hello world from rank %d, recv %lld from %d, get total %lld\n", world_rank, number_in_circle, world_rank + 1 * mul, number_in_circle_sum);
            MPI_Recv(&number_in_circle, 1, MPI_LONG_LONG_INT, world_rank + 1 * mul, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            number_in_circle_sum += number_in_circle;
            mul *= 2;
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
