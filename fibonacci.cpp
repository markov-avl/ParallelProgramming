#include <iostream>
#include <chrono>
#include <omp.h>
#include <memory>
#include <thread>

static unsigned threadsNum = std::thread::hardware_concurrency();

void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads(T);
}


unsigned getThreadsNum() {
    return threadsNum;
}

struct TestResult {
    unsigned value;
    double milliseconds;
};


TestResult
run_experiment(unsigned (*fib)(unsigned), unsigned n) {
    auto tm1 = std::chrono::steady_clock::now();
    auto value = fib(n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    return TestResult{value, (double) time};
}


void measure_scalability(unsigned (*fib)(unsigned)) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<TestResult[]>(P);
    unsigned n = 35;
    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partial_res[T - 1] = run_experiment(fib, n);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
}


unsigned fib(unsigned n) {
    if (n < 2) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}


unsigned fibTask(unsigned n) {
    if (n < 2) {
        return n;
    }

    unsigned fib1, fib2;

#pragma omp task shared(fib1)
    {
        fib1 = fibTask(n - 1);
    }
#pragma omp task shared(fib2)
    {
        fib2 = fibTask(n - 2);
    }

#pragma omp taskwait
    return fib1 + fib2;
}

int main() {
    std::cout << "No parallel tasks:" << std::endl;
    measureScalability(fib);
    std::cout << "With parallel tasks:" << std::endl;
    measureScalability(fibTask);
}