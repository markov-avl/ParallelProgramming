#include <chrono>
#include <thread>
#include <vector>
#include <iostream>
#include <omp.h>
#include <mutex>

#define CACHE_LINE 64u
#define N (500000000)
#define load(N) for (auto i = N; i > 0; --i) std::cout << "Load\n"


static unsigned threadsNum = std::thread::hardware_concurrency();
struct TestResult {
    double value, milliseconds;
};


union partial_sum_t {
    double value;
    alignas(double) char pd[64];
};


double average_cs_omp(const double *v, size_t n) {
    double total_sum = 0;
#pragma omp parallel
    {
        unsigned T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
        size_t nt, i0;

        if (t < n % T) {
            nt = n / T + 1;
            i0 = nt * t;
        } else {
            nt = n / T;
            i0 = nt * (n % T);
        }

        double par_sum = 0;
        for (size_t i = i0; i < nt + i0; ++i) {
            par_sum += v[i];
        }

#pragma omp critical
        {
            total_sum += par_sum;
        }
    }

    return total_sum / n;
}


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
    auto v = std::make_unique<double[]>(N);
    fillVector(v.get(), N);
    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partial_res[T - 1] = run_experiment(averageFunction, v.get(), N);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
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
            local_sum += v[i];
        }
        sums[t].value = local_sum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }

    free(sums);
    return result / n;
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
        }
        sums[t].value = local_sum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }

    free(sums);
    return result;
}


double average_cs_cpp() {
    std::mutex mtx1, mtx2;
    int x1 = 0;
    int x2 = 0;
    auto tp1 = [&mtx1, &x1, &mtx2, &x2]() {
        mtx1.lock();
        ++x1;
        load(100);
        mtx2.lock();
        ++x2;
        mtx2.unlock();
        mtx1.unlock();
    };
    auto tp2 = [&mtx2, &x2, &mtx1, &x1]() {
        mtx2.lock();
        --x2;
        mtx1.lock();
        --x1;
        mtx1.unlock();
        mtx2.unlock();
    };
    auto th1 = std::thread(tp1), th2 = std::thread(tp2);
    th1.join(); th2.join();
    std::cout << "x1 = " << x1 << "; x2 = " << x2 << std::endl;

    return x1 + x2;
}


double average(const double *v, size_t n) {
    double result = 0;
    std::mutex mtx;
    auto worker = [&result, &mtx] (unsigned t) {
        unsigned T = omp_get_num_threads();
        size_t n_t = n / T;
        size_t i_0 = n % T;
        double local_sum = 0;

        if (t < i_0) {
            i_0 += ++n_t * t;
        } else {
            i_0 += t * n_t;
        }

        for (size_t i = i_0; i < n_t + i_0; ++i) {
            local_sum += v[i];
        }

        mtx.lock();
        result += local_sum;
        mtx.unlock();
    }

    std::vector<std::thread> workers;
    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);

    for (auto &w : workers) {
        w.join();
    }

    return result / n;
}


int main() {
    std::size_t n1 = 10;
    auto m = std::make_unique<double[]>(n1);
    std::generate_n(m.get(), n1, []() {
        static int i;
        return i++;
    });

    std::cout << "Average value: " << average(m.get(), n1) << "\n";
}


//int main() {
////    std::cout << "AveragePar1:" << std::endl;
////    measureScalability(average_par_1);
////    std::cout << "AveragePar2:" << std::endl;
////    measureScalability(average_par_2);
////    std::cout << "CriticalSection:" << std::endl;
////    measureScalability(average_cs_omp);
////    std::cout << "Mutex:" << std::endl;
////    measureScalability(average_cs_cpp);
//
////    average_cs_cpp();
//
//    // переделать прагма на мутекс лок и мутекс анлок
//    for (int i = 0; i < 1000; ++i) {
//        if (average_cs_cpp() != 0) {
//            std::cout << "СЛОМАЛОСЬ!\n";
//        }
//    }
//}