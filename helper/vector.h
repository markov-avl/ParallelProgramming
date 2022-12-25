#ifndef PARALLELPROGRAMMING_VECTOR_H
#define PARALLELPROGRAMMING_VECTOR_H

#include <iostream>

#define N 100000000

template<typename T>
void fillVector(T *v, T element) {
    for (size_t i = 0; i < N; ++i) {
        v[i] = element;
    }
}


#endif
