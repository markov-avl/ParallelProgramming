#ifndef PARALLELPROGRAMMING_LUT_H
#define PARALLELPROGRAMMING_LUT_H

#include <iostream>

#define A UINT64_C(6364136223846793005)
#define B 1

struct LutRow {
    size_t a;
    size_t b;
};

std::unique_ptr<LutRow[]> getLut(unsigned T);

#endif
