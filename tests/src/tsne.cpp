#include <gtest/gtest.h>

#ifdef CUSTOM_PARALLEL_TEST
// must be before any qdtsne includes.
#include "custom_parallel.h"
#endif

#include "knncolle/knncolle.hpp"
#include "qdtsne/tsne.hpp"
#include "qdtsne/utils.hpp"
#include <random>

class TsneTester : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    void assemble(int N, int D, int K) {
        X.resize(N * D);

        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : X) {
            y = dist(rng);
        }

        knncolle::VpTreeEuclidean<> searcher(D, N, X.data()); 
        for (size_t i = 0; i < N; ++i) {
            neighbors.push_back(searcher.find_nearest_neighbors(i, K));
        }

        return;
    }

    std::vector<double> X;
    qdtsne::NeighborList<int, double> neighbors;
};

TEST_P(TsneTester, Initialization) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    qdtsne::Tsne thing;
    auto status = thing.initialize(neighbors);

    // Checking probabilities are all between zero and 1.
    const auto& probs = status.neighbors;
    EXPECT_EQ(probs.size(), N);
    double total = 0;
    for (const auto& curp : probs) {
        EXPECT_TRUE(curp.size() >= K);
        for (const auto& p : curp) {
            EXPECT_TRUE(p.second < 1);
            EXPECT_TRUE(p.second > 0);
            total += p.second;
        }
    }
    EXPECT_FLOAT_EQ(total, 1);

    // Checking symmetry of the probabilities.
    std::map<std::pair<int, int>, std::tuple<double, bool, bool> > stuff;
    for (size_t n = 0; n < neighbors.size(); ++n) {
        const auto& current = status.neighbors[n];

        for (const auto& y : current) {
            auto neighbor = y.first;
            EXPECT_TRUE(neighbor != n);

            std::pair<int, int> key(std::min((int)n, neighbor), std::max((int)n, neighbor)); // only consider combinations
            auto it = stuff.lower_bound(key);

            if (it != stuff.end() && it->first == key) {
                EXPECT_EQ(std::get<0>(it->second), y.second);

                // Checking that this permutation doesn't already exist.
                if (n > neighbor) {
                    EXPECT_FALSE(std::get<1>(it->second));
                    std::get<1>(it->second) = true;
                } else {
                    EXPECT_FALSE(std::get<2>(it->second));
                    std::get<2>(it->second) = true;
                }
            } else {
                stuff.insert(it, std::make_pair(key, std::make_tuple(y.second, n > neighbor, n < neighbor)));
            }
        }
    }

    for (const auto& s : stuff) {
        EXPECT_TRUE(std::get<1>(s.second));
        EXPECT_TRUE(std::get<2>(s.second));
    }
}

TEST_P(TsneTester, Runner) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    qdtsne::Tsne thing;
    auto Y = qdtsne::initialize_random<>(N);
    auto old = Y;

    auto status = thing.run(neighbors, Y.data());
    EXPECT_NE(old, Y); // there was some effect...
    EXPECT_EQ(status.nobs(), N); 
    EXPECT_EQ(status.iteration(), 1000); // actually ran through the specified iterations

    // Check that coordinates are zero-mean.
    for (int d = 0; d < 2; ++d) {
        double total = 0;
        for (size_t i = 0; i < N; ++i){
            total += Y[2*i + d];
        }
        EXPECT_TRUE(std::abs(total/N) < 1e-10);
    }

    // Same results when run in parallel.
    thing.set_num_threads(3);
    auto copy = old;
    auto pstatus = thing.run(neighbors, copy.data());
    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, StopStart) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    auto Y = qdtsne::initialize_random<>(N);
    auto copy = Y;

    qdtsne::Tsne thing;
    auto ref = thing.run(neighbors, Y.data());

    auto status = thing.initialize(neighbors);
    status.run(copy.data(), 500);
    status.run(copy.data(), 1000);

    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, EasyStart) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    auto original = qdtsne::initialize_random<>(N);
    qdtsne::Tsne thing;
    thing.set_max_iter(10); // don't need that many iterations for this...

    auto Y = original;
    auto ref = thing.run(neighbors, Y.data());

    auto copy = original;
    auto easy = thing.set_perplexity(K/3).run(X.data(), D, N, copy.data());
    EXPECT_EQ(copy, Y);

    EXPECT_EQ(ref.neighbors[0], easy.neighbors[0]);
    EXPECT_EQ(ref.neighbors[10], easy.neighbors[10]);
    EXPECT_EQ(ref.neighbors[100], easy.neighbors[100]);
}

INSTANTIATE_TEST_CASE_P(
    TsneTests,
    TsneTester,
    ::testing::Combine(
        ::testing::Values(200), // number of observations
        ::testing::Values(5), // input dimensions, doesn't really matter
        ::testing::Values(30, 60, 90) // number of neighbors
    )
);
