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


unsigned checkSumOMP(const unsigned* v, size_t n) {
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


unsigned checkSumMutex(const unsigned* v, size_t n) {
    unsigned totalSum = 0;
    std::mutex mtx;
    std::vector<std::thread> workers;

    auto worker = [&totalSum, &mtx, v, n] (unsigned t) {
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
    for (auto &w : workers) {
        w.join();
    }

    return totalSum;
}

unsigned checkSumAtomic(const unsigned* v, size_t n) {
    std::atomic<unsigned> totalSum{0};
    std::vector<std::thread> workers;

    auto worker = [&totalSum, v, n] (unsigned t) {
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

        totalSum.fetch_xor(localSum, std::memory_order_relaxed);
    };

    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w : workers) {
        w.join();
    }

    return totalSum;
}


int main() {
    std::size_t n1 = 23342321;
    auto m = std::make_unique<unsigned[]>(n1);
    for (size_t i = 0; i < n1; ++i) {
        m.get()[i] = 0x12345678;
    }
    std::cout << "CheckSum Mutex OMP value: " << std::hex << checkSumOMP(m.get(), n1) << std::endl;
    std::cout << "CheckSum Mutex C++ value: " << std::hex << checkSumMutex(m.get(), n1) << std::endl;
    std::cout << "CheckSum Atomic C++ value: " << std::hex << checkSumAtomic(m.get(), n1) << std::endl;
}
