#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <random>

using namespace std;

long long int number_of_tosses;
int thread_count;
long long int* number_in_circle;

void* monte_carlo(void* rank) {
    long my_rank = (long) rank;
    long long int my_n_tosses = number_of_tosses / thread_count;
    long long int my_number_in_circle = 0;
    unsigned int seed = 5;
    // Create a thread-local random number generator
    // std::default_random_engine generator;
    // std::mt19937 generator;
    // std::minstd_rand generator;
    // std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (long long int toss = 0; toss < my_n_tosses; toss ++) {
        // long double x = (double) rand()*2 / RAND_MAX - 1;
        // long double y = (double) rand()*2 / RAND_MAX - 1;
        // long double x = distribution(generator);
        // long double y = distribution(generator);
        double x = ((double) rand_r(&seed)*2 / RAND_MAX) - 1.0;
        double y = ((double) rand_r(&seed)*2 / RAND_MAX) - 1.0;
        long double distance_squared = x * x + y * y;
        if ( distance_squared <= 1){
            my_number_in_circle++;
        }
    }

    number_in_circle[my_rank] = my_number_in_circle;

    return NULL;
}

int main(int argc, char* argv[]){
    /*if (argc != 3) {
        cout << "Usage: " << argv[0] << " <number of threads> <number of tosses>" << endl;
        return 1;
    }*/
    thread_count = atoi(argv[1]);
    number_of_tosses = atoll(argv[2]);
    long long int total_number_in_circle = 0;
    pthread_t* thread_handles;

    number_in_circle = new long long int[thread_count];
    thread_handles = new pthread_t[thread_count];

    for (long thread = 0; thread < thread_count; thread++) {
        pthread_create(&thread_handles[thread], NULL, monte_carlo, (void*) thread);
    }

    for (long thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
        total_number_in_circle += number_in_circle[thread];
    }

    cout << double( 4 * total_number_in_circle / (double) number_of_tosses) << endl;

    return 0;
}