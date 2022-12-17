#include <cstdint>
#include <iostream>
#include <vector>
#include <omp.h>
#include <condition_variable>
#include <thread>
// рандомизация массива
// линейно конгруентный генератор

void srand(int seed);

typedef uint64_t word;

word rand(word s);

static unsigned num_threads = std::thread::hardware_concurrency();

struct partial_sum_t {
    alignas(64) int value;
};

void set_num_threads(unsigned T) {
    num_threads = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return num_threads;
}


# define A UINT64_C(6364136223846793005)
# define B 1
// c = 1 << 64

void randomize(word seed, int *V, size_t n, int a, int b) {
    word x = seed;
    for (size_t i = 0; i < n; ++i) {
        x = A * x + b;
        V[i] = a + int(x % (b - a + 1));
    }
}

struct lut_row {
    size_t a;
    size_t b;
};

auto get_lut(unsigned T) {
    auto lut = std::make_unique<lut_row[]>(T + 1);
    lut[0].a = 1;
    lut[0].b = 0;
    for (size_t i = 1; i < T; ++i) {
        lut[i].a = lut[i - 1].a * A;
        lut[i].b = A * lut[i - 1].b + B;
    }
    return lut;
}

auto randomizeV2(word seed, int *V, size_t n, int a, int b) {
    word s0 = seed;
    size_t T = omp_get_num_procs();
    static auto lut = get_lut(T);
    auto average = 0;

#pragma omp parallel
    {
        auto t = omp_get_num_procs();
        size_t St = lut[t].a * s0 + lut[t].b;

        for (unsigned k = t; k < n; k += T) {
            V[k] = a + int(St % (b - a + 1));
            average += V[k];
            St = lut[T].a * St + lut[T].b;
        }
    }

    return average / T;
}

auto randomizeV2_cpp(word seed, int *V, size_t n, int a, int b) {
    std::vector<std::thread> thr;
    word s0 = seed;
    auto T = get_num_threads();

    static auto lut = get_lut(T);
    auto partial_sums = std::make_unique<partial_sum_t[]>(T);

    auto worker = [&partial_sums, T, s0, &V, n, a, b](unsigned t)
    {
        int local_sum = 0;
        size_t St = lut[T].a * s0 + lut[T].b;

        for (unsigned k = t; k < n; k += t) {
            V[k] = a + int(St % (b - a + 1));
            local_sum += V[k];
            St = lut[T].a * St + lut[T].b;
        }

        partial_sums[t].value = local_sum;
    };

    for (unsigned t = 1; t < get_num_threads(); ++t) {
        thr.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w : thr) {
        w.join();
    }

    return partial_sums[0].value / n;
}

// 1. реализовать рандомизатор, используя механизм std::thread
// 2. добавить расчёт среднего арифметического, чтобы понять работает или нет

int main() {
    size_t n = 10;
    int *V = new int[n];
//    randomize(456, V, n, 0, 1);
//    for (int i = 0; i < n; i++) {
//        std::cout << V[i];
//    }
    std::cout << randomizeV2(456, V, n, 1, 999) << std::endl;
}