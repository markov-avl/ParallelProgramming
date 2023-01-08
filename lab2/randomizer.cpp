#include <cstdint>
#include <vector>
#include <omp.h>
#include <condition_variable>
#include <thread>
#include "../helper/lut.h"
#include "../helper/partial_sum.h"
#include "../helper/threads.h"
#include "../helper/barrier.h"


double randomizedAverageSimple(uint64_t seed, unsigned *v, size_t n, unsigned a, unsigned b) {
    auto x = seed;
    uint64_t sum = 0;
    for (auto i = 0; i < n; ++i) {
        x = A * x + B;
        v[i] = a + int(x % (b - a + 1));
        sum += v[i];
    }
    return (double) sum / (double) n;
}


double randomizedAverageOmp(uint64_t seed, unsigned *v, size_t n, unsigned a, unsigned b) {
    auto P = omp_get_num_procs();
    auto partialSums = std::make_unique<PartialSum<uint64_t>[]>(P);
    static auto lut = getLut(P);

    unsigned T;

#pragma omp parallel shared(partialSums, T)
    {
        auto t = omp_get_thread_num();
        uint64_t St = lut[t].a * seed + lut[t].b;
        uint64_t localSum = 0;

#pragma omp single
        {
            T = omp_get_num_threads();
        }

        for (size_t k = t; k < n; k += T) {
            v[k] = a + St % (b - a + 1);
            St = lut[t].a * St + lut[t].b;
            localSum += v[k];
        }

        partialSums[t].value = localSum;
    }

    for (auto i = 1; i < T; ++i) {
        partialSums[0].value += partialSums[i].value;
    }

    return (double) partialSums[0].value / (double) n;
}


double randomizedAverageCpp(uint64_t seed, unsigned *v, size_t n, unsigned a, unsigned b) {
    auto T = getThreadsNum();
    auto P = omp_get_num_procs();
    auto partialSums = std::make_unique<PartialSum<uint64_t>[]>(T);
    static auto lut = getLut(P);
    std::vector<std::thread> workers;
    Barrier barrier(T);

    auto worker = [&barrier, &partialSums, &v, T, seed, n, a, b](unsigned t) {
        uint64_t St = lut[t].a * seed + lut[t].b;
        uint64_t localSum = 0;

        for (size_t k = t; k < n; k += T) {
            v[k] = a + St % (b - a + 1);
            St = lut[t].a * St + lut[t].b;
            localSum += v[k];
        }

        partialSums[t].value = localSum;
        barrier.arrive_and_wait();

        for (size_t step = 1, next_step = step * 2; step < T; step = next_step, next_step += next_step) {
            if (t % next_step == 0 && t + step < T) {
                partialSums[t].value += partialSums[t + step].value;
            }
            barrier.arrive_and_wait();
        }
    };

    for (auto t = 1; t < T; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    return (double) partialSums[0].value / (double) n;
}
