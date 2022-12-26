#include <vector>
#include <omp.h>
#include <thread>
#include "../helper/partial_sum.h"
#include "../helper/vector.h"
#include "../helper/tester.h"


unsigned averageOmp(const unsigned *v, size_t n) {
    PartialSum<unsigned> *sums;
    unsigned result = 0;
    unsigned T;

#pragma omp parallel shared(sums, T)
    {
        auto t = omp_get_thread_num();
        unsigned localSum = 0;

#pragma omp single
        {
            T = omp_get_num_threads();
            sums = (PartialSum<unsigned> *) malloc(T * sizeof(PartialSum<unsigned>));
        }
        for (size_t i = t; i < n; i += T) {
            localSum += v[i];
        }
        sums[t].value = localSum;
    }

    for (size_t i = 0; i < T; ++i) {
        result += sums[i].value;
    }
    free(sums);

    return result / n;
}


unsigned averageCpp(const unsigned *v, size_t n) {
    auto T = getThreadsNum();
    auto partialSums = std::make_unique<unsigned[]>(T);

    auto worker = [&partialSums, &v, T, n](std::size_t t) {
        unsigned localSum = 0;
        for (auto i = t; i < n; i += T) {
            localSum += v[i];
        }
        partialSums[t] = localSum;
    };

    std::vector<std::thread> workers;
    for (std::size_t t = 1; t < T; ++t) {
        workers.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w: workers) {
        w.join();
    }

    for (std::size_t t = 1; t < T; ++t) {
        partialSums[0] += partialSums[t];
    }

    return partialSums[0] / n;
}


int main() {
    auto v = std::make_unique<unsigned[]>(N);
    fillVector<unsigned>(v.get(), 1);

    measureScalability("Average (FS, OMP)", averageOmp, v.get(), N);
    std::cout << std::endl;
    measureScalability("Average (FS, C++)", averageCpp, v.get(), N);
}
