#include <thread>
#include <vector>
#include <iostream>
#include <omp.h>
#include <mutex>
#include <iomanip>


static unsigned threadsNum = std::thread::hardware_concurrency();

void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads(T);
}

unsigned getThreadsNum() {
    return threadsNum;
}


unsigned checkSumOmp(const unsigned *v, size_t n) {
    unsigned totalSum = 0;

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

        unsigned localSum = 0;
        for (size_t i = i0; i < nt + i0; ++i) {
            localSum ^= v[i];
        }

#pragma omp critical
        {
            totalSum ^= localSum;
        }
    }

    return totalSum;
}


unsigned checkSumCpp(const unsigned *v, size_t n) {
    unsigned totalSum = 0;
    std::mutex mtx;
    std::vector<std::thread> workers;

    auto worker = [&totalSum, &mtx, v, n](unsigned t) {
        unsigned T = getThreadsNum();
        unsigned localSum = 0;
        size_t nt, i0;

        if (t < n % T) {
            nt = n / T + 1;
            i0 = nt * t;
        } else {
            nt = n / T;
            i0 = nt * (n % T);
        }

        for (size_t i = i0; i < nt + i0; ++i) {
            localSum ^= v[i];
        }

        std::scoped_lock lock{mtx};
        totalSum ^= localSum;
    };

    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    return totalSum;
}


struct TestResult {
    unsigned value;
    double milliseconds;
};


TestResult
run_experiment(unsigned (*checkSum)(const unsigned *, size_t), const unsigned *v, size_t n) {
    auto tm1 = std::chrono::steady_clock::now();
    auto value = checkSum(v, n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    TestResult res{value, (double) time};
    return res;
}


void measure_scalability(auto checkSumFunction, unsigned *v, size_t n) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<TestResult[]>(P);

    for (auto T = 1; T <= P; ++T) {
        setThreadsNum(T);
        partial_res[T - 1] = run_experiment(checkSumFunction, v, n);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
}


int main() {
    std::size_t n = 1000000000;
    auto v = std::make_unique<unsigned[]>(n);
    for (size_t i = 0; i < n; ++i) {
        v.get()[i] = 0x12345678;
    }

    std::cout << "checkSumOmp:" << std::endl;
    measure_scalability(checkSumOmp, v.get(), n);
    std::cout << "checkSumCpp:" << std::endl;
    measure_scalability(checkSumCpp, v.get(), n);
}
