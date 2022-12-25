#include <iostream>
#include <thread>
#include "../helper/vector.h"
#include "../helper/tester.h"

unsigned averageDynamicParallel(const unsigned *v, size_t n) {
    unsigned sum = 0;
#pragma omp parallel for reduction(+: sum) schedule(dynamic)
    for (size_t i = 0; i < n; ++i) {
        sum += v[i];
    }

    return sum / (unsigned) n;
}

int main() {
    auto v = std::make_unique<unsigned[]>(N);
    fillVector<unsigned>(v.get(), 1);

    std::cout << "Average Dynamic Parallel:" << std::endl;
    measureScalability(averageDynamicParallel, v.get(), N);
}