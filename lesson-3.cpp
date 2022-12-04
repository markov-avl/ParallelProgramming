#include <chrono>
#include <thread>
#include <vector>
#include <type_traits>
#include <iostream>
#include <omp.h>

#define CACHE_LINE 64u
#define n (100000000)

static unsigned num_threads = std::thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    num_threads = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return num_threads;
}

struct PartialSumT {
    double value[CACHE_LINE / sizeof(double)];
};


double integrateArrAlign(double a, double b, double (*f)(double)) {
    unsigned T;
    double result = 0, dx = (b - a) / n;
    PartialSumT *accum;
#pragma omp parallel shared(accum, T)
    {
        auto t = omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = new PartialSumT[T];
        }
        for (unsigned i = t; i < n; i += T) {
            accum[t].value[0] += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].value[0];
    }

    delete[] accum;

    return result;
}