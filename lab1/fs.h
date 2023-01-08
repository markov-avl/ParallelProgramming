#ifndef PARALLELPROGRAMMING_FS_H
#define PARALLELPROGRAMMING_FS_H

#include <iostream>

double averageOmp(const unsigned *, size_t);

double averageCpp(const unsigned *, size_t);

double averageAlignedOmp(const unsigned *v, size_t n);

double averageAlignedCpp(const unsigned *v, size_t n);

#endif
