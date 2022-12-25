#include <iostream>
#include <chrono>
#include <omp.h>

#define N (100000000)

struct TestResult {
    double value, milliseconds;
};

TestResult
run_experiment(double (*integrate)(double, double, double (*f)(double)), double a, double b, double (*f)(double)) {
    auto tm1 = std::chrono::steady_clock::now();
    double value = integrate(a, b, f);
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
    TestResult res{value, (double) time};
    return res;
}

double
integrate_seq(double a, double b, double (*f)(double)) {
    double delta = (b - a) / N;
    double sum = 0;
    for (int i = 0; i < N; i++) {
        sum += f(a + i * delta);
    }
    return sum * delta;
}


double integrate_par(double a, double b, double (*f)(double)) {
    double delta = (b - a) / N;
    double sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < N; i++)
        sum += f(a + i * delta);
    return sum * delta;
}

template<class F>
requires std::is_invocable_r_v<double, F, double>
double integral_rr(double a, double b, F f) {
    double sum = 0.0, dx = (b - a) / N;
    unsigned P = omp_get_num_procs();
#pragma omp parallel
    {
        unsigned T = omp_get_num_threads();
        unsigned t = omp_get_thread_num();
        for (unsigned k = 0; t + k * T < N; ++k) {
            sum += f(a + (t + k * T) * dx);
        }
    }
    return sum * dx;
}


// g++ -fopenmp main.cpp --std=c++20 - чтобы запустить параллельное вычисление

int main() {
    auto f = [](double x) { return x * x; };
    auto tm1 = std::chrono::steady_clock::now();
    std::cout << "Sequential result: " << integrate_seq(-1, 1, f);
    std::cout << "     Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count()
              << '\n';

    auto g = [](double x) { return x * x; };
    auto r_seq = run_experiment(integrate_seq, -1, 1, g);
    auto r_par = run_experiment(integrate_par, -1, 1, g);
    std::cout << "Sequential result: " << r_seq.value << "     Time: " << r_seq.milliseconds << '\n';
    std::cout << "Parallel result: " << r_par.value << "     Time: " << r_par.milliseconds << '\n';

}