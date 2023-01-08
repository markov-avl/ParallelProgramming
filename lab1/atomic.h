#ifndef PARALLELPROGRAMMING_ATOMIC_H
#define PARALLELPROGRAMMING_ATOMIC_H

#include <iostream>

unsigned checkSumAtomicOmp(const unsigned *, size_t);

unsigned checkSumAtomicCpp(const unsigned *, size_t);

#endif
