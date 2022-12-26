#include <cstdint>
#include <iostream>
#include <vector>
#include <omp.h>
#include <condition_variable>
#include <thread>
#include "../helper/lut.h"
#include "../helper/partial_sum.h"
#include "../helper/threads.h"
#include "../helper/tester.h"
#include "../helper/vector.h"


int randomizedAverageSimple(word seed, int *v, size_t n, int a, int b) {
    auto x = seed;
    word sum = 0;
    for (auto i = 0; i < n; ++i) {
        x = A * x + B;
        v[i] = a + int(x % (b - a + 1));
        sum += v[i];
    }
    return sum / n;
}

int randomizedAverageOmp(word seed, int *v, size_t n, int a, int b) {
    auto T = omp_get_num_procs();
    auto partialSums = std::make_unique<PartialSum<word>[]>(T);
    static auto lut = getLut(T);

#pragma omp parallel shared(partialSums, T)
    {
        auto t = omp_get_thread_num();
        word St = lut[t].a * seed + lut[t].b;
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


int randomizedAverageCpp(word seed, int *v, size_t n, int a, int b) {
    auto T = getThreadsNum();
    auto partialSums = std::make_unique<PartialSum<word>[]>(T);
    static auto lut = getLut(T);
    std::vector<std::thread> workers;

    auto worker = [&partialSums, &v, T, seed, n, a, b](unsigned t) {
        word St = lut[t].a * seed + lut[t].b;
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

    return (int) (partialSums[0].value / n);
}

int main() {
    word seed = 123456789;
    int a = 1;
    int b = 10;
    auto v = std::make_unique<int[]>(N);

    measureScalability("Randomized Average (OMP)", randomizedAverageOmp, seed, v.get(), N, a, b);
    measureScalability("Randomized Average (C++)", randomizedAverageCpp, seed, v.get(), N, a, b);
}
