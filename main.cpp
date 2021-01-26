#include <iostream>
#include "model.cuh"


int main(int argc, const char * argv[]) {
    std::cout << "Hello, World!" << std::endl;

    double dlt_t            = argc > 1 ? atof(argv[1]) : 0.02;
    int max_steps           = argc > 2 ? atoi(argv[2]) : 200000;

    int L                   = argc > 3 ? atoi(argv[3]) : 30;

    std::cout << "dlt_t = " << dlt_t << "\t max_steps = " << max_steps << std::endl;
    std::cout << "L = " << L << std::endl;

    Square_Lattice system(L);

    system.t1 = -1;

    system.kT = 0.000000001;
    system.filling = 0.25;

    system.simulate_dynamics(max_steps, dlt_t, 0.0);

    return 0;
}
