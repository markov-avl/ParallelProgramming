#include "average.h"

double averageStaticOmp(const unsigned *v, size_t n) {
    auto sum = .0;
#pragma omp parallel for reduction(+: sum)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (double) n;
}


double averageDynamicOmp(const unsigned *v, size_t n) {
    auto sum = .0;
#pragma omp parallel for reduction(+: sum) schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (double) n;
}
