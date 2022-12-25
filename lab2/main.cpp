#include <cstdint>
#include <iostream>
#include <vector>
#include <omp.h>
#include <condition_variable>
#include <thread>
#include "lut.h"
#include "../helper/partial_sum.h"
#include "../helper/threads.h"

typedef uint64_t word;

auto randomizedAverage(word seed, int *v, size_t n, int a, int b) {
    auto x = seed;
    word sum = 0;
    for (auto i = 0; i < n; ++i) {
        x = A * x + B;
        v[i] = a + int(x % (b - a + 1));
        sum += v[i];
    }
    return sum / n;
}

auto randomizedAverageV2C(word seed, int *v, size_t n, int a, int b) {
    auto s0 = seed;
    auto T = omp_get_num_procs();
    auto partialSums = std::make_unique<PartialSum<word>[]>(T);
    static auto lut = getLut(T);

#pragma omp parallel
    {
        auto t = omp_get_thread_num();
        word St = lut[t].a * s0 + lut[t].b;
        word localSum = 0;

        for (auto k = t; k < n; k += T) {
            v[k] = a + int(St % (b - a + 1));
            St = lut[t].a * St + lut[t].b;
            localSum += v[k];
        }

        partialSums[t].value = localSum;
    }

    for (auto i = 1; i < T; ++i) {
        partialSums[0].value += partialSums[i].value;
    }

    return partialSums[0].value / n;
}

auto randomizedAverageV2Cpp(word seed, int *v, size_t n, int a, int b) {
    auto s0 = seed;
    auto T = getThreadsNum();
    auto partialSums = std::make_unique<PartialSum<word>[]>(T);
    static auto lut = getLut(T);
    std::vector<std::thread> workers;

    auto worker = [&partialSums, T, s0, &v, n, a, b](unsigned t) {
        word St = lut[t].a * s0 + lut[t].b;
        word localSum = 0;

        for (auto k = t; k < n; k += T) {
            v[k] = a + int(St % (b - a + 1));
            St = lut[t].a * St + lut[t].b;
            localSum += v[k];
        }

        partialSums[t].value = localSum;
    };

    for (auto t = 1; t < T; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    for (auto i = 1; i < T; ++i) {
        partialSums[0].value += partialSums[i].value;
    }

    return partialSums[0].value / n;
}

int main() {
    auto seed = 453453453534534545;
    auto a = 1;
    auto b = 1000000;
    auto n = 100000000;
    auto v = std::make_unique<int[]>(n);

    std::cout << randomizedAverage(seed, v.get(), n, a, b) << std::endl;
    std::cout << randomizedAverageV2C(seed, v.get(), n, a, b) << std::endl;
    std::cout << randomizedAverageV2Cpp(seed, v.get(), n, a, b) << std::endl;
}
