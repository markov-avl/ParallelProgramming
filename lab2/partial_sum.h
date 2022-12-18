#ifndef PARALLELPROGRAMMING_PARTIAL_SUM_H
#define PARALLELPROGRAMMING_PARTIAL_SUM_H

template <typename T>
struct PartialSum {
    alignas(64) T value;
};

#endif
