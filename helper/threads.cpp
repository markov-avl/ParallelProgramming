#include <omp.h>
#include "threads.h"

void setThreadsNum(unsigned T) {
    threadsNum = T;
    omp_set_num_threads((int) T);
}

unsigned getThreadsNum() {
    return threadsNum;
}
