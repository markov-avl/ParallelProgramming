#include "barrier.h"

void Barrier::arrive_and_wait() {
    std::unique_lock l{mtx};
    if (++WT < T) {
        auto my_gen = gen;
        while (my_gen == gen) {
            cv.wait(l);
        }
    } else {
        cv.notify_all();
        WT = 0;
        gen = !gen;
    }
}
