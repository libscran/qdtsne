#ifndef QDTSNE_UTILS_HPP
#define QDTSNE_UTILS_HPP

#include <random>

namespace qdtsne {

template<int ndim = 2>
void initialize_random(double* Y, size_t N, int seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<> dist(0, 1);
    for (size_t i = 0; i < N * ndim; ++i) {
        Y[i] = dist(rng);
    }
}

template<int ndim = 2>
std::vector<double> initialize_random(size_t N, int seed = 42) {
    std::vector<double> Y(ndim * N);
    initialize_random<ndim>(Y.data(), N, seed);
    return Y;
}

}

#endif
