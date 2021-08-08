#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

#include <random>

namespace qdtsne {

template<int ndim = 2>
void fill_initial_values(double* Y, size_t N, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist(0, 1);
    for (size_t i = 0; i < N * ndim; ++i) {
        Y[i] = dist(rng);
    }
}

}

#endif
