#include "average.h"

unsigned averageStaticOmp(const unsigned *v, size_t n) {
    auto sum = 0U;
#pragma omp parallel for reduction(+: sum)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (unsigned) n;
}


unsigned averageDynamicOmp(const unsigned *v, size_t n) {
    auto sum = 0U;
#pragma omp parallel for reduction(+: sum) schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (unsigned) n;
}
