#include <chrono>
#include <thread>
#include <vector>
#include <type_traits>
#include <iostream>
#include <omp.h>

#define n (100000000)

static unsigned threadsNum = std::thread::hardware_concurrency();

struct TestResult {
    double value, milliseconds;
};


void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads(T);
}

unsigned getThreadsNum() {
    return threadsNum;
}


TestResult
run_experiment(double (*integrate)(double, double, double (*f)(double)), double a, double b, double (*f)(double)) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = integrate(a, b, f);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    TestResult res{value, (double) time};
    return res;
}


double integrate_seq(double a, double b, double (*f)(double)) {
    double dx = (b - a) / n;
    double sum = 0;
    for (unsigned i = 0; i < n; i++) {
        sum += f(a + i * dx);
    }
    return dx * sum;
}


double integrate_par(double a, double b, double (*f)(double)) {
    double delta = (b - a) / n;
    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += f(a + i * delta);
    return sum * delta;
}


double integrate_cpp_partial_sums(double a, double b, double (*f)(double)) {
    std::size_t T = std::thread::hardware_concurrency();
    auto partial_sums = std::make_unique<double[]>(T);
    auto thread_proc = [T, &partial_sums, a, b, f](std::size_t t) {
        partial_sums[t] = 0;
        auto dx = (b - a) / n;
        for (auto i = t; i < n; i += T) {
            partial_sums[t] += f(i * dx + a);
        }
        partial_sums[t] *= dx;
    };
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < T; ++t) {
        workers.emplace_back(thread_proc, t);
    }
    for (auto &worker: workers) {
        worker.join();
    }
    for (std::size_t t = 1; t < T; ++t) {
        partial_sums[0] += partial_sums[t];
    }

    return partial_sums[0];
}


template<class F>
requires std::is_invocable_r_v<double, F, double>
double integral_rr(double a, double b, F f) {
    double sum = 0.0, dx = (b - a) / n;
    unsigned P = omp_get_num_procs();
#pragma omp parallel
    {
        unsigned T = getThreadsNum();
        unsigned t = omp_get_thread_num();
        for (unsigned k = 0; t + k * T < n; ++k) {
            sum += f(a + (t + k * T) * dx);
        }
    }
    return sum * dx;
}


void measureScalability(auto integrate_fn) {
    auto f = [](double x) { return x * x; };
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<TestResult[]>(P);
    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partial_res[T - 1] = run_experiment(averageFunction, -1, 1, f);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
}


int main() {
    std::cout << "IntegrateSeq:" << std::endl;
    measureScalability(integrate_seq);
    std::cout << "IntegratePar:" << std::endl;
    measureScalability(integrate_par);
    std::cout << "IntegrateCppPartialSums:" << std::endl;
    measureScalability(integrate_cpp_partial_sums);
}