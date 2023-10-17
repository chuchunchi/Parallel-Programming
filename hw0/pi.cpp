#include <cstdlib>
#include <iostream>
using namespace std;
/*
number_in_circle = 0;
for ( toss = 0; toss < number_of_tosses; toss ++) {
    x = random double between -1 and 1;
    y = random double between -1 and 1;
    distance_squared = x * x + y * y;
    if ( distance_squared <= 1)
        number_in_circle++;
}
pi_estimate = 4 * number_in_circle /(( double ) number_of_tosses);
*/
int main(){
    long long int number_in_circle = 0;
    long long int number_of_tosses = 5 * 10e6;
    for (long long int toss = 0; toss < number_of_tosses; toss ++) {
        long double x = (double) rand()*2 / RAND_MAX - 1;
        long double y = (double) rand()*2 / RAND_MAX - 1;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1){
            number_in_circle++;
        }
    }
    cout << double( 4 * number_in_circle / (double) number_of_tosses) << endl;
}
