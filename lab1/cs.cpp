#include <thread>
#include <vector>
#include <omp.h>
#include <mutex>
#include "cs.h"
#include "../helper/threads.h"


unsigned checkSumOmp(const unsigned *v, size_t n) {
    unsigned totalSum = 0;

#pragma omp parallel
    {
        unsigned T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
        size_t nt, i0;

        if (t < n % T) {
            nt = n / T + 1;
            i0 = nt * t;
        } else {
            nt = n / T;
            i0 = nt * (n % T);
        }

        unsigned localSum = 0;
        for (size_t i = i0; i < nt + i0; ++i) {
            localSum ^= v[i];
        }

#pragma omp critical
        {
            totalSum ^= localSum;
        }
    }

    return totalSum;
}


unsigned checkSumCpp(const unsigned *v, size_t n) {
    unsigned totalSum = 0;
    std::mutex mutex;
    std::vector<std::thread> workers;

    auto worker = [&totalSum, &mutex, v, n](unsigned t) {
        auto T = getThreadsNum();
        unsigned localSum = 0;
        size_t nt, i0;

        if (t < n % T) {
            nt = n / T + 1;
            i0 = nt * t;
        } else {
            nt = n / T;
            i0 = nt * (n % T);
        }

        for (size_t i = i0; i < nt + i0; ++i) {
            localSum ^= v[i];
        }

        std::scoped_lock lock{mutex};
        totalSum ^= localSum;
    };

    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    return totalSum;
}
