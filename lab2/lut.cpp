#include <cstdint>
#include <vector>
#include <condition_variable>
#include "lut.h"

std::unique_ptr<LutRow[]> getLut(unsigned T) {
    auto lut = std::make_unique<LutRow[]>(T + 1);
    lut[0].a = A;
    lut[0].b = B;
    for (auto i = 1; i < T; ++i) {
        lut[i].a = lut[i - 1].a * A;
        lut[i].b = A * lut[i - 1].b + B;
    }
    return lut;
}
