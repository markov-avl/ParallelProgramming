#include <thread>

#include "randomizer.h"
#include "../helper/vector.h"
#include "../helper/tester.h"

int main() {
    uint64_t seed = 100;
    auto a = 1U;
    auto b = 100U;
    auto v = std::make_unique<unsigned[]>(N);

    std::cout << "Randomized Average (OMP):" << std::endl;
    measureScalability(randomizedAverageOmp, seed, v.get(), N, a, b);
    std::cout << std::endl;

    std::cout << "Randomized Average (C++):" << std::endl;
    measureScalability(randomizedAverageCpp, seed, v.get(), N, a, b);
    std::cout << std::endl;
}
