#include <future>
#include <iostream>

unsigned fibAsync(unsigned n) {
    if (n < 2) {
        return n;
    }

    // if (free_threads > 0) {
    //     ...
    // }

    auto fib1 = std::async(fibAsync, n - 1);
    auto fib2 = std::async(fibAsync, n - 2);

    return fib1.get() + fib2.get();
}

// добавить счетчик поток:


int main() {
    std::cout << fibAsync(10);
}