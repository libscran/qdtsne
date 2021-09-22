#include <cassert>
#include <random>
#include <vector>

#include <iostream>
#include <chrono>
//#define QDTSNE_BETA_BINARY_SEARCH_ONLY
#define PROGRESS_PRINTER(x, y) { std::cout << x << "\t" << y << std::endl; }

#include "qdtsne/qdtsne.hpp"

int main() {
    size_t dim = 5;
    size_t nobs = 50000;

    // Simulating some data.
    std::cout << "Simulating data... ";
    std::mt19937_64 rng;
    std::normal_distribution<> dist;
    std::vector<double> thing(dim*nobs);
    for (auto& t : thing) {
        t = dist(rng);
    }
    std::cout << "Done" << std::endl;

    // Initializing everything.
    qdtsne::Tsne tsne;
    tsne.set_interpolation(100);
    std::cout << "Building index... ";
    knncolle::AnnoyEuclidean<> searcher(dim, nobs, static_cast<const double*>(thing.data())); 
    auto init = tsne.initialize(&searcher);
    std::cout << "Done" << std::endl;

    auto Y = qdtsne::initialize_random(nobs);
    std::cout << "Iterating... ";
    auto t1 = std::chrono::high_resolution_clock::now();
    tsne.run(init, Y.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " seconds" << std::endl;


    return 0;
}
