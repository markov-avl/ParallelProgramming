#include <thread>
#include <vector>
#include <iostream>
#include <omp.h>
#include <mutex>
#include <iomanip>
#include <condition_variable>
#include <atomic>


static unsigned num_threads = std::thread::hardware_concurrency();

class my_barrier
{
    std::condition_variable cv;
    std::mutex mtx;
    const unsigned T;
    unsigned WT = 0;
    bool gen{};
public:
    my_barrier(unsigned threads): T(threads) {}
    void arrive_and_wait();
};
void my_barrier::arrive_and_wait() {
    std::unique_lock l {mtx};
    if (++WT < T)
    {
        bool my_gen = gen;
        while (my_gen == gen)
        {
            cv.wait(l);
        }
    }
    else
    {
        cv.notify_all();
        WT = 0;
        gen = !gen;
    }
}

void set_num_threads(unsigned T) {
    num_threads = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return num_threads;
}


struct partial_sum_t {
    alignas(64) double value;
};

struct result_t {
    double value;
    double milliseconds;
};


result_t
run_experiment(double (*checkSum)(const double *, size_t), const double *v, size_t n) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = checkSum(v, n);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    result_t res{value, (double) time};
    return res;
}


double average(const double *v, size_t n) {
    auto T = get_num_threads();
    auto partial_sums = std::make_unique<partial_sum_t[]>(T);

    std::vector<std::thread> thr;
    my_barrier bar(T);
    auto worker = [&partial_sums, T, &bar, v, n](unsigned t) {
        double local_sum = 0;
        for (size_t i = t; i < n; i += T) {
            local_sum += v[i];
        }
        partial_sums[t].value = local_sum;
        bar.arrive_and_wait();

        for (size_t step = 1, next_step = step * 2; step < T; step = next_step, next_step += next_step) {
            if (t % next_step == 0 && t + step < T) {
                partial_sums[t].value += partial_sums[t + step].value;
            }
            bar.arrive_and_wait();
        }
    };

    for (unsigned t = 1; t < get_num_threads(); ++t) {
        thr.emplace_back(worker, t);
    }
    worker(0);
    for (auto &w : thr) {
        w.join();
    }

    return partial_sums[0].value / (double) n;
}

void measure_scalability(auto average_func, double *v, size_t n) {
    auto P = omp_get_num_procs();
    auto partial_res = std::make_unique<result_t[]>(P);

    for (auto T = 1; T <= P; ++T) {
        set_num_threads(T);
        partial_res[T - 1] = run_experiment(average_func, v, n);
        auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
        std::cout << "Количество потоков: " << T << std::endl;
        std::cout << "Время: " << partial_res[T - 1].milliseconds << std::endl;
        std::cout << "Значение: " << partial_res[T - 1].value << std::endl;
        std::cout << "Ускорение: " << speedup << std::endl << std::endl;
    }
}


int main() {
    std::size_t n = 500000000;
    auto v = std::make_unique<double[]>(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 1;
    }

    std::cout << "Average Reduce:" << std::endl;
    measure_scalability(average, v.get(), n);
}