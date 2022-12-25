#ifndef PARALLELPROGRAMMING_THREADS_H
#define PARALLELPROGRAMMING_THREADS_H

#include <thread>

static unsigned threadsNum = std::thread::hardware_concurrency();

void setThreadsNum(unsigned T);

unsigned getThreadsNum();


#endif
