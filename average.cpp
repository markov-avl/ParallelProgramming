#include <chrono>
#include <thread>
#include <vector>
#include <type_traits>
#include <iostream>
#include <omp.h>

#define CACHE_LINE 64u
#define N (10000000)


static unsigned threadsNum = std::thread::hardware_concurrency();
struct TestResult {
    double value, milliseconds;
};


union partial_sum_t {
    double value;
    alignas(double) char pd[64];
};


void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads(T);
}

unsigned getThreadsNum() {
    return threadsNum;
}


void fillVector(double *v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1.0;
    }
}


TestResult
run_experiment(double (*average)(const double *, size_t), const double *v, size_t n) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = average(v, n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    TestResult res{value, (double) time};
    return res;
}


void measureScalability(auto averageFunction) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<TestResult[]>(P);
    double v[N];
    fillVector(v, N);
    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
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
    std::size_t T = getThreadsNum();
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


double average_par_1(const double *v, size_t n) {
    unsigned T;
    partial_sum_t *sums;
    double result = 0;
#pragma omp parallel shared(T, sums)
    {
        unsigned t = omp_get_thread_num();
        double local_sum;
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
        }
        for (size_t i = t; i < n; i += T) {
            local_sum = v[i];
        }
        sums[t].value = local_sum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }

    free(sums);
    return result;
}


double average_par_2(const double *v, size_t n) {
    unsigned T;
    partial_sum_t *sums;
    double result = 0;
#pragma omp parallel shared(T, sums)
    {
        unsigned t = omp_get_thread_num();
        double local_sum;
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
        }

        size_t n_t, i_0;

        if (t < n % T) {
            n_t = n / T + 1;
            i_0 = n_t * t;
        } else {
            n_t = n / T;
            i_0 = t * (n / T) + (n % T);
        }

        for (size_t i = i_0; i < n_t; ++i) {
            local_sum = v[i];
            for (int j = 0; j < 1000; ++j);
        }
        sums[t].value = local_sum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }

    free(sums);
    return result;
}


struct PartialSumT {
    double value[CACHE_LINE / sizeof(double)];
};


double averageAlign(const double* v, size_t n) {
    unsigned T;
    double result = 0;
    PartialSumT *accum;
#pragma omp parallel shared(accum, T)
    {
        auto t = omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = new PartialSumT[T];
        }
        for (unsigned i = t; i < n; i += T) {
            accum[t].value[0] += v[i];
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].value[0];
    }

    delete[] accum;

    return result;
}


int main() {
//    std::cout << "AverageSeq:" << std::endl;
//    measureScalability(average_seq);
//    std::cout << "AveragePar:" << std::endl;
//    measureScalability(average_par);
    std::cout << "AveragePar1:" << std::endl;
    measureScalability(average_par_1);
    std::cout << "AveragePar2:" << std::endl;
    measureScalability(average_par_2);
//    std::cout << "AverageCppPartialSums:" << std::endl;
//    measureScalability(average_cpp_partial_sums);
}