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

template<typename T>
TestResult<T> runExperiment(T (*f)(const T *, size_t), const T *v, size_t n) {
    auto t0 = std::chrono::steady_clock::now();
    auto value = f(v, n);
    auto t1 = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return TestResult<T>{value, time};
}


template<typename T_>
void measureScalability(T_ (*f)(const T_ *, size_t), const T_ *v, size_t n) {
    auto P = omp_get_num_procs();
    auto partialResults = std::make_unique<TestResult<T_>[]>(P);

    std::cout << "Количество потоков | Время | Значение | Ускорение" << std::endl;

    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partialResults[T - 1] = runExperiment(f, v, n);

        std::cout << T;
        std::cout << "\t" << partialResults[T - 1].milliseconds;
        std::cout << "\t" << partialResults[T - 1].value;
        std::cout << "\t" << partialResults[0].milliseconds / partialResults[T - 1].milliseconds;
        std::cout << std::endl;
    }
}


#endif
