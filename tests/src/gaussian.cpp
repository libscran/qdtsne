#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "knncolle/knncolle.hpp"
#include "qdtsne/gaussian.hpp"

class GaussianTest : public ::testing::TestWithParam<std::tuple<int, double> > {
protected:
    inline static int N = 200;
    inline static int D = 5;
    inline static std::vector<double> X;

    static void SetUpTestSuite() {
        X.resize(N * D);
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : X) {
            y = dist(rng);
        }
        return;
    }
};

TEST_P(GaussianTest, Gaussian) {
    auto PARAM = GetParam();
    size_t K = std::get<0>(PARAM);
    double P = std::get<1>(PARAM);

    auto index = knncolle::VptreeBuilder().build_unique(knncolle::SimpleMatrix(D, N, X.data()));
    auto neighbors = knncolle::find_nearest_neighbors(*index, K, /* num_threads = */ 1);

    auto copy = neighbors;
    qdtsne::internal::compute_gaussian_perplexity(neighbors, P, 1);
    const double expected = std::log(P);

    // Checking that the entropy is within range.
    for (int i = 0; i < N; ++i) {
        double entropy = 0;
        double sum = 0;
        for (const auto& x : neighbors[i].second) {
            entropy += x * std::log(x);
            sum += x;
        }
        entropy *= -1;
        
        EXPECT_TRUE(std::abs(expected - entropy) < 1e-5);
        EXPECT_TRUE(std::abs(sum - 1) < 1e-8);
    }

    // Same result in parallel.
    qdtsne::internal::compute_gaussian_perplexity(copy, P, 3);
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(neighbors[i], copy[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Gaussian,
    GaussianTest,
    ::testing::Combine(
        ::testing::Values(30, 60, 90), // number of neighbors
        ::testing::Values(10.0, 20.0, 30.0) // perplexity
    )
);

TEST(GaussianTest, Overflow) {
    {
        knncolle::NeighborList<int, float> neighbors(1);
        auto& first = neighbors.front();

        // Lots of ties causes the beta search to overflow.
        for (size_t i = 0; i < 90; ++i) {
            first.first.emplace_back(i);
            first.second.emplace_back(1);
        }

        qdtsne::internal::compute_gaussian_perplexity(neighbors, static_cast<float>(30), 1);

        // Expect finite probabilities.
        for (auto& x : neighbors.front().second) {
            EXPECT_EQ(x, neighbors.front().second.front());
        }
    }

    {
        knncolle::NeighborList<int, float> neighbors(1);
        auto& first = neighbors.front();

        first.first.emplace_back(0);
        first.second.emplace_back(1);
        for (size_t i = 1; i < 90; ++i) {
            first.first.emplace_back(i);
            first.second.emplace_back(1.0000001);
        }

        // Really cranking down the perplexity (and thus forcing the beta
        // search to try to get an unreachable entropy).
        qdtsne::internal::compute_gaussian_perplexity(neighbors, static_cast<float>(1), 1);

        EXPECT_TRUE(first.second.front() > 0);
        for (size_t i = 1; i < 90; ++i) {
            EXPECT_EQ(first.second[i], 0);
        }
    }
}
