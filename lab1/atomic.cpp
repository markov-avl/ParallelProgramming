#include <thread>
#include <vector>
#include <c++/10/atomic>
#include "../helper/threads.h"
#include "../helper/vector.h"
#include "../helper/tester.h"

unsigned checkSumAtomic(const unsigned *v, size_t n) {
    std::atomic<unsigned> globalSum{0};
    std::vector<std::thread> workers;

    auto worker = [&globalSum, v, n](unsigned t) {
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

        globalSum.fetch_xor(localSum, std::memory_order_relaxed);
    };

    for (unsigned t = 1; t < getThreadsNum(); ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    return globalSum;
}

int main() {
    auto v = std::make_unique<unsigned[]>(N);
    fillVector(v.get(), 0x12345678U);

    measureScalability("Check Sum (ATOMIC, C++)", checkSumAtomic, v.get(), N);
}
