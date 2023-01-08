#include <future>
#include <omp.h>

#include "fibonacci.h"

unsigned fibonacciAsyncOmp(unsigned n) {
    if (n < 2) {
        return n;
    }

    unsigned fib1, fib2;

#pragma omp task shared(fib1)
    {
        fib1 = fibonacciAsyncOmp(n - 1);
    }
#pragma omp task shared(fib2)
    {
        fib2 = fibonacciAsyncOmp(n - 2);
    }

#pragma omp taskwait
    return fib1 + fib2;
}


unsigned fibonacciAsyncCpp(unsigned n) {
    if (n < 2) {
        return n;
    }

    auto fib1 = std::async(fibonacciAsyncCpp, n - 1);
    auto fib2 = std::async(fibonacciAsyncCpp, n - 2);

    return fib1.get() + fib2.get();
}
