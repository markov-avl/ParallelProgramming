#include <chrono>
#include <thread>
#include <vector>
#include <type_traits>
#include <iostream>
#include <omp.h>

#define N (1000000)

static unsigned num_threads = std::thread::hardware_concurrency();
struct result_t {
    double value, milliseconds;
};

void set_num_threads(unsigned T) {
    num_threads = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return num_threads;
}


void fillVector(double *v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0;
    }
}


result_t
run_experiment(double (*average)(const double *, size_t), const double *v, size_t n) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = average(v, n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    result_t res{value, (double) time};
    return res;
}


void measure_scalability(auto averageFunction) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<result_t[]>(P);
    double v[N];
    fillVector(v, N);
    for (auto T = 1; T <= P; ++T) {
        set_num_threads(T);
        partial_res[T - 1] = run_experiment(averageFunction, v, N);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
}


double average_seq(const double *v, size_t n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += v[i] / (double)n;
        // Просто, чтобы медленее считал...
        for (int j = 0; j < 1000; ++j);
    }
    return sum;
}


double average_par(const double *v, size_t n) {
    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; ++i) {
        sum += v[i];
        // Просто, чтобы медленее считал...
        for (int j = 0; j < 1000; ++j);
    }
    return sum / (double)n;
}


double average_cpp_partial_sums(const double *v, size_t n) {
    std::size_t T = get_num_threads();
    auto partial_sums = std::make_unique<double[]>(T);
    auto thread_proc = [T, &partial_sums, v, n](std::size_t t) {
        partial_sums[t] = 0;
        for (auto i = t; i < n; i += T) {
            partial_sums[t] += v[i];
            // Просто, чтобы медленее считал...
            for (int j = 0; j < 1000; ++j);
        }
        partial_sums[t] /= (double)n;
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


int main() {
    std::cout << "AverageSeq:" << std::endl;
    measure_scalability(average_seq);
    std::cout << "AveragePar:" << std::endl;
    measure_scalability(average_par);
    std::cout << "AverageCppPartialSums:" << std::endl;
    measure_scalability(average_cpp_partial_sums);
}