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
    long milliseconds;
};

template<typename T>
TestResult<T> runExperiment(T (*f)(const T *, size_t), const T *v, size_t n) {
    auto t0 = std::chrono::steady_clock::now();
    auto value = f(v, n);
    auto t1 = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return TestResult{value, time};
}


template<typename T_>
void measureScalability(T_ (*f)(const T_ *, size_t), const T_ *v, size_t n) {
    auto P = omp_get_num_procs();
    auto partialResults = std::make_unique<TestResult<T_>[]>(P);
    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partialResults[T - 1] = runExperiment(f, v, n);

        auto speedup = partialResults[0].milliseconds / partialResults[T - 1].milliseconds;

        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partialResults[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partialResults[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl;
        std::cout << std::endl;
    }
}


#endif
