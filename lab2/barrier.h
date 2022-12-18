#ifndef PARALLELPROGRAMMING_BARRIER_H
#define PARALLELPROGRAMMING_BARRIER_H

#include <mutex>
#include <condition_variable>


class Barrier {
private:
    std::condition_variable cv;
    std::mutex mtx;
    const unsigned T;
    unsigned WT = 0;
    bool gen{};

public:
    explicit Barrier(unsigned threads) : T(threads) {}

    void arrive_and_wait();
};

#endif
