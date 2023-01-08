#ifndef PARALLELPROGRAMMING_TESTER_H
#define PARALLELPROGRAMMING_TESTER_H

#include <iostream>
#include <chrono>
#include <omp.h>
#include <thread>
#include "threads.h"

template<typename T>
struct TestResult {
    T value;
    double milliseconds;
};

template<typename D, typename R>
auto runExperiment(R (*f)(const D *, size_t), const D *v, size_t n) {
    auto t0 = std::chrono::steady_clock::now();
    R value = f(v, n);
    auto t1 = std::chrono::steady_clock::now();
    auto time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return TestResult{value, time};
}

template<typename D, typename R>
auto runExperiment(R (*f)(uint64_t, D *, size_t, D, D), uint64_t seed, D *v, size_t n, D a, D b) {
    auto t0 = std::chrono::steady_clock::now();
    R value = f(seed, v, n, a, b);
    auto t1 = std::chrono::steady_clock::now();
    auto time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return TestResult{value, time};
}

template<typename D, typename R>
auto runExperiment(R (*f)(D), D n) {
    auto t0 = std::chrono::steady_clock::now();
    R value = f(n);
    auto t1 = std::chrono::steady_clock::now();
    auto time = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return TestResult{value, time};
}

template<typename D, typename R>
void measureScalability(R (*f)(const D *, size_t), const D *v, size_t n) {
    auto P = omp_get_num_procs();
    auto results = std::make_unique<TestResult<R>[]>(P);

    std::cout << "Количество потоков | Время | Значение | Ускорение" << std::endl;

    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        results[T - 1] = runExperiment(f, v, n);

        std::cout << T;
        std::cout << "\t" << results[T - 1].milliseconds;
        std::cout << "\t" << results[T - 1].value;
        std::cout << "\t" << results[0].milliseconds / results[T - 1].milliseconds;
        std::cout << std::endl;
    }
}

template<typename D, typename R>
void measureScalability(R (*f)(uint64_t, D *, size_t, D, D), uint64_t seed, D *v, size_t n, D a, D b) {
    auto P = omp_get_num_procs();
    auto results = std::make_unique<TestResult<R>[]>(P);

    std::cout << "Количество потоков | Время | Значение | Ускорение" << std::endl;

    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        results[T - 1] = runExperiment(f, seed, v, n, a, b);

        std::cout << T;
        std::cout << "\t" << results[T - 1].milliseconds;
        std::cout << "\t" << results[T - 1].value;
        std::cout << "\t" << results[0].milliseconds / results[T - 1].milliseconds;
        std::cout << std::endl;
    }
}


#endif
