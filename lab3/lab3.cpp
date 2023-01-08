#include "fibonacci.h"

#include "../helper/tester.h"

int main() {
    TestResult<unsigned> result{};
    auto N = 24U;

    std::cout << "Fibonacci (OMP):" << std::endl;
    std::cout << "n   | Результат | Время" << std::endl;
    for (auto n = 1U; n <= N; ++n) {
        result = runExperiment(fibonacciAsyncOmp, n);
        std::cout << n;
        std::cout << "\t" << result.value;
        std::cout << "\t" << result.milliseconds;
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Fibonacci (C++):" << std::endl;
    std::cout << "n   | Результат | Время" << std::endl;
    for (auto n = 1U; n <= N; ++n) {
        result = runExperiment(fibonacciAsyncCpp, n);
        std::cout << n;
        std::cout << "\t" << result.value;
        std::cout << "\t" << result.milliseconds;
        std::cout << std::endl;
    }
}