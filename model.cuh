//
// Created by Puhan Zhang on 1/12/21.
//

#ifndef NEUMANN_MODEL_CUH
#define NEUMANN_MODEL_CUH

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <random>
#include "cuComplex.h"
#include "kernel.cuh"

typedef std::mt19937 RNG;

class Square_Lattice {
public:
    int L, Ns;
    int dim;

    double t1;

    double filling;
    double mu;
    double kT;

    static constexpr int N_nn1 = 4;

    double time;

    class Site {
    public:
        int idx;
        int x, y;

        int sgn;

        Site *nn1[N_nn1];

    } *site;

    double *onsite_V = nullptr;

    int *neighboring = nullptr;
    cuDoubleComplex *Hamiltonian = nullptr;
    cuDoubleComplex *Density_Mat = nullptr;
    cuDoubleComplex *actor = nullptr;

    Square_Lattice(int linear_size);

    ~Square_Lattice() {
        delete [] site;
        site = nullptr;
    };

    inline int index(int x, int y) {
        return L * y + x;
    };

    void init_lattice();

    void build_Hamiltonian();
    void step(cuDoubleComplex *KD_sum, cuDoubleComplex *D2, double dt, int type);
    // void step_2(cuDoubleComplex *KD_sum, cuDoubleComplex *D2, double dt);
    // void step_3(cuDoubleComplex *KD_sum, cuDoubleComplex *D2, double dt);
    // void step_4(cuDoubleComplex *KD_sum, cuDoubleComplex *D2, double dt);

    void compute_fermi_level(double *);
    void compute_density_matrix();

    // ========================================

    void save_configuration(std::string const filename);
    void simulate_dynamics(int max_steps, double dt, double W);
    cuDoubleComplex calculate_trace();
    void init_quenched_disorder(double W);
    void integrate_EOM_RK4(double dt);
};

#endif //NEUMANN_MODEL_CUH
