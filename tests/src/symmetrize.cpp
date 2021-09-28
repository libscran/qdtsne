#include <gtest/gtest.h>

#include "qdtsne/symmetrize.hpp"
#include "knncolle/knncolle.hpp"

#include <map>
#include <random>
#include <cmath>

class SymmetrizeTest : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    template<class Param>
    auto generate_neighbors (Param p) { 
        size_t nobs = std::get<0>(p);
        int k = std::get<1>(p);
        int ndim = 5;

        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        std::vector<double> data(nobs * ndim);
        for (int r = 0; r < data.size(); ++r) {
            data[r] = dist(rng);
        }

        qdtsne::NeighborList<int, double> stored(nobs);
        knncolle::VpTreeEuclidean<> searcher(ndim, nobs, data.data());
        for (size_t i = 0; i < nobs; ++i) {
            stored[i] = searcher.find_nearest_neighbors(i, k);
            for (auto& x : stored.back()) {
                x.second = std::exp(-x.second);
            }
        }

        return stored;
    }
};

template<class Searched>
void slow_symmetrization(const Searched& original, const Searched& combined) {
    std::map<std::pair<int, int>, double> probs;

    // Filling 'probs'.
    double total = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        const auto& y = original[i];
        for (const auto& z : y) {
            EXPECT_NE(z.first, i);
            std::pair<int, int> target(z.first, i);
            std::pair<int, int> alt(i, z.first);

            // If it was already added, it would be when the first index is
            // less than the second, as those would be added at a prior 'i'.
            auto it = probs.find(target);
            if (it != probs.end()) {
                it->second += z.second;
                probs[alt] = it->second;
            } else {
                probs[target] = z.second;
                probs[alt] = z.second;
            }

            total += z.second;
        }
    }

    // Comparing to the combined results.
    std::map<std::pair<int, int>, int> found;
    for (size_t i = 0; i < combined.size(); ++i) {
        const auto& y = combined[i];
        for (const auto& z : y) {
            std::pair<int, int> target(i, z.first);
            auto it = probs.find(target);
            EXPECT_TRUE(it != probs.end());
            if (it != probs.end()) {
                EXPECT_FLOAT_EQ(it->second / total / 2, z.second);
            }
            ++found[target];
        }
    }

    EXPECT_EQ(probs.size(), found.size());
}

TEST_P(SymmetrizeTest, Combining) {
    auto stored = generate_neighbors(GetParam());

    auto copy = stored;
    qdtsne::symmetrize_matrix(copy);
    slow_symmetrization(stored, copy);

    // Comparing the number of edges.
    size_t total_s = 0, total_c = 0;
    for (size_t i = 0; i < stored.size(); ++i) {
        const auto& cvec = copy[i];
        total_c += cvec.size();
        const auto& svec = stored[i];
        total_s += svec.size();
    }

    EXPECT_TRUE(total_c > total_s);
}

INSTANTIATE_TEST_SUITE_P(
    Symmetrize,
    SymmetrizeTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors 
    )
);
