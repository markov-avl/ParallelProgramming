#include "fs.h"

#include <vector>
#include <omp.h>

#include "../helper/threads.h"
#include "../helper/partial_sum.h"


double averageOmp(const unsigned *v, size_t n) {
    unsigned *sums;
    auto result = .0;
    unsigned T;

#pragma omp parallel shared(sums, T)
    {
        auto t = omp_get_thread_num();
        unsigned localSum = 0;

#pragma omp single
        {
            T = omp_get_num_threads();
            sums = (unsigned *) malloc(T * sizeof(unsigned));
        }
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
        }
        sums[t] = localSum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i];
    }
    free(sums);

    return result / (double) n;
}


double averageCpp(const unsigned *v, size_t n) {
    auto T = getThreadsNum();
    auto sums = std::make_unique<double[]>(T);

    auto worker = [&sums, &v, T, n](size_t t) {
        auto localSum = 0U;
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
        }
        sums[t] = localSum;
    };

    std::vector<std::thread> workers;
    for (size_t t = 1; t < T; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    for (size_t t = 1; t < T; ++t) {
        sums[0] += sums[t];
    }

    return sums[0] / (double) n;
}

double averageAlignedOmp(const unsigned *v, size_t n) {
    PartialSum<unsigned> *sums;
    auto result = .0;
    unsigned T;

#pragma omp parallel shared(sums, T)
    {
        auto t = omp_get_thread_num();
        unsigned localSum = 0;

#pragma omp single
        {
            T = omp_get_num_threads();
            sums = (PartialSum<unsigned> *) malloc(T * sizeof(PartialSum<unsigned>));
        }
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
        }
        sums[t].value = localSum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }
    free(sums);

    return result / (double) n;
}

double averageAlignedCpp(const unsigned *v, size_t n) {
    auto T = getThreadsNum();
    auto sums = std::make_unique<PartialSum<double>[]>(T);

    auto worker = [&sums, &v, T, n](size_t t) {
        auto localSum = 0U;
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
        }
        sums[t].value = localSum;
    };

    std::vector<std::thread> workers;
    for (size_t t = 1; t < T; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    for (size_t t = 1; t < T; ++t) {
        sums[0].value += sums[t].value;
    }

    return sums[0].value / (double) n;
}
