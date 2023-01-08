#include <thread>
#include "../helper/threads.h"
#include "../helper/tester.h"
#include "../helper/vector.h"

#include "cs.h"
#include "fs.h"
#include "average.h"
#include "atomic.h"


int main() {
    auto v = std::make_unique<unsigned[]>(N);
    fillVector(v.get(), 1U);

    std::cout << "Check Sum (CS, OMP):" << std::endl;
    measureScalability(checkSumOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Check Sum (CS, C++):" << std::endl;
    measureScalability(checkSumCpp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (STATIC REDUCTION, OMP):" << std::endl;
    measureScalability(averageStaticOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (DYNAMIC REDUCTION, OMP):" << std::endl;
    measureScalability(averageDynamicOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (FS, OMP):" << std::endl;
    measureScalability(averageOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (FS, C++):" << std::endl;
    measureScalability(averageCpp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (FS ALIGNED, OMP):" << std::endl;
    measureScalability(averageAlignedOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Average (FS ALIGNED, C++):" << std::endl;
    measureScalability(averageAlignedCpp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Check Sum (ATOMIC, OMP):" << std::endl;
    measureScalability(checkSumAtomicOmp, v.get(), N);
    std::cout << std::endl;

    std::cout << "Check Sum (ATOMIC, C++):" << std::endl;
    measureScalability(checkSumAtomicCpp, v.get(), N);
    std::cout << std::endl;

}