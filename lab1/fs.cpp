#include <vector>
#include <omp.h>
#include "../helper/partial_sum.h"


unsigned averageOmp(const unsigned *v, size_t n) {
    PartialSum<unsigned> *sums;
    double r;

#pragma omp parallel
    {
        unsigned T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
        unsigned localSum = 0;
#pragma omp single
        {
            sums = (PartialSum<unsigned> *) malloc(T * sizeof(PartialSum<unsigned>));
        }
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
            sums[t].value = localSum;
        }
        for (size_t i = 0; i < T; ++i) {
            free(sums);
            r += sums[i].value;
        }
    }
    return r;
}
