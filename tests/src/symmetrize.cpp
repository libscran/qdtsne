#include <gtest/gtest.h>

#include "qdtsne/symmetrize.hpp"
#include "knncolle/knncolle.hpp"

#include <map>
#include <random>
#include <cmath>

class SymmetrizeTest : public ::testing::TestWithParam<std::tuple<int, int> > {};

TEST_P(SymmetrizeTest, Combining) {
    auto p = GetParam();
    size_t nobs = std::get<0>(p);
    int k = std::get<1>(p);

    qdtsne::NeighborList<int, double> stored(nobs);
    {
        std::mt19937_64 rng(nobs * k); // for some variety
        std::normal_distribution<> dist(0, 1);

        int ndim = 5;
        std::vector<double> data(nobs * ndim);
        for (auto& d : data) {
            d = dist(rng);
        }

        knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
        auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, data.data()));
        auto searcher = index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;
        for (size_t i = 0; i < nobs; ++i) {
            searcher->search(i, k, &indices, &distances);

            auto& current = stored[i];
            int actual_k = indices.size();
            for (int x = 0; x < actual_k; ++x) {
                current.emplace_back(indices[x], std::exp(-distances[x]));
            }
        }
    }

    std::map<std::pair<int, int>, double> probs;
    double total = 0;
    {
        for (size_t i = 0; i < nobs; ++i) {
            const auto& y = stored[i];
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
    }

    // Checking that the number of edges actually increases after symmetrization.
    size_t total_before = 0;
    for (size_t i = 0; i < nobs; ++i) {
        const auto& svec = stored[i];
        total_before += svec.size();
    }

    qdtsne::internal::symmetrize_matrix(stored);
    EXPECT_EQ(stored.size(), nobs);

    size_t total_after = 0;
    for (size_t i = 0; i < stored.size(); ++i) {
        const auto& svec = stored[i];
        total_after += svec.size();
    }
    EXPECT_LT(total_before, total_after);

    // Checking the probabilities are as expected.
    std::map<std::pair<int, int>, int> found;
    for (size_t i = 0; i < nobs; ++i) {
        const auto& y = stored[i];
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

INSTANTIATE_TEST_SUITE_P(
    Symmetrize,
    SymmetrizeTest,
    ::testing::Combine(
        ::testing::Values(50, 100, 200), // number of observations
        ::testing::Values(5, 10, 15) // number of neighbors 
    )
);
