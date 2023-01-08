#include <thread>
#include <vector>
#include <c++/10/atomic>
#include <omp.h>
#include "../helper/threads.h"

#include "atomic.h"

unsigned checkSumAtomicOmp(const unsigned *v, size_t n) {
    auto checkSum = 0U;

#pragma omp parallel shared(checkSum)
    {
        auto t = omp_get_thread_num();
        auto T = omp_get_num_threads();

        auto localSum = 0U;
        size_t nt, i0;

        if (t < n % T) {
            nt = n / T + 1;
            i0 = nt * t;
        } else {
            nt = n / T;
            i0 = nt * (n % T);
        }

        for (auto i = i0; i < nt + i0; ++i) {
            localSum ^= v[i];
        }
#pragma omp atomic
        checkSum ^= localSum;
    }

    return checkSum;
}

unsigned checkSumAtomicCpp(const unsigned *v, size_t n) {
    std::atomic<unsigned> checkSum{0};
    std::vector<std::thread> workers;

    auto worker = [&checkSum, v, n](unsigned t) {
        unsigned T = getThreadsNum();
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

        checkSum.fetch_xor(localSum, std::memory_order_relaxed);
    };

    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    return checkSum;
}
