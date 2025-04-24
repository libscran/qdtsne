#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "knncolle/knncolle.hpp"
#include "qdtsne/gaussian.hpp"

class GaussianTest : public ::testing::TestWithParam<std::tuple<int, double> > {
protected:
    inline static std::unique_ptr<knncolle::Prebuilt<int, double, double> > index;
    inline static int N = 200;

    static void SetUpTestSuite() {
        int D = 5;
        std::vector<double> X(N * D);
        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : X) {
            y = dist(rng);
        }

        knncolle::VptreeBuilder<int, double, double> builder(std::make_shared<knncolle::EuclideanDistance<double, double> >());
        index = builder.build_unique(knncolle::SimpleMatrix(D, N, X.data()));
        return;
    }

    static qdtsne::NeighborList<int, double> get_neighbors(int K) {
        qdtsne::NeighborList<int, double> neighbors(N);
        std::vector<int> indices;
        std::vector<double> distances;
        auto searcher = index->initialize();
        for (int i = 0; i < N; ++i) {
            searcher->search(i, K, &indices, &distances);
            int actual_k = indices.size();
            for (int k = 0; k < actual_k; ++k) {
                neighbors[i].emplace_back(indices[k], distances[k]);
            }
        }
        return neighbors;
    }
};

TEST_P(GaussianTest, Newton) {
    auto PARAM = GetParam();
    size_t K = std::get<0>(PARAM);
    double P = std::get<1>(PARAM);

    auto neighbors = get_neighbors(K);
    auto copy = neighbors;
    qdtsne::internal::compute_gaussian_perplexity(neighbors, P, 1);
    const double expected = std::log(P);

    // Checking that the entropy is within range.
    for (int i = 0; i < N; ++i) {
        double entropy = 0;
        double sum = 0;
        for (const auto& x : neighbors[i]) {
            entropy += x.second * std::log(x.second);
            sum += x.second;
        }
        entropy *= -1;

        EXPECT_LT(std::abs(expected - entropy), 1e-5);
        EXPECT_LT(std::abs(sum - 1), 1e-8);
    }

    // Same result in parallel.
    qdtsne::internal::compute_gaussian_perplexity(copy, P, 3);
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(neighbors[i], copy[i]);
    }
}

TEST_P(GaussianTest, BinaryFallback) {
    // We need to test the binary fallback explicitly because I can't figure
    // out a scenario where Newton's fails...  though I can't prove that it
    // won't, hence the need for a fallback at all.
    auto PARAM = GetParam();
    size_t K = std::get<0>(PARAM);
    double P = std::get<1>(PARAM);

    auto neighbors = get_neighbors(K);
    auto copy = neighbors;
    qdtsne::internal::compute_gaussian_perplexity<false>(neighbors, P, 1);
    const double expected = std::log(P);

    // Checking that the entropy is within range.
    for (int i = 0; i < N; ++i) {
        double entropy = 0;
        double sum = 0;
        for (const auto& x : neighbors[i]) {
            entropy += x.second * std::log(x.second);
            sum += x.second;
        }
        entropy *= -1;

        EXPECT_LT(std::abs(expected - entropy), 1e-5);
        EXPECT_LT(std::abs(sum - 1), 1e-8);
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

TEST(GaussianTest, Empty) {
    qdtsne::NeighborList<int, float> neighbors(1);
    qdtsne::internal::compute_gaussian_perplexity(neighbors, static_cast<float>(30), 1);
}

TEST(GaussianTest, AllEqualDistances) {
    qdtsne::NeighborList<int, float> neighbors(1);
    auto& first = neighbors.front();

    // Lots of ties causes the beta search to overflow.
    for (size_t i = 0; i < 90; ++i) {
        first.emplace_back(i, 1);
    }

    qdtsne::internal::compute_gaussian_perplexity(neighbors, static_cast<float>(30), 1);

    // Expect finite probabilities.
    for (auto& x : neighbors.front()) {
        EXPECT_FLOAT_EQ(x.second, 1.0 / 90.0);
    }
}

TEST(GaussianTest, ConvergenceFailure) {
    // Really cranking down the perplexity (and thus forcing the beta
    // search to try to get an impossible entropy). We test multiple
    // 'leads' to ensure that we handle ties on the first distance.
    for (int leads = 1; leads < 10; leads += 5) {
        qdtsne::NeighborList<int, float> neighbors(1);
        auto& first = neighbors.front();

        for (int i = 0; i < leads; ++i) {
            first.emplace_back(0, 1);
        }
        for (int i = leads; i < 90; ++i) {
            first.emplace_back(i, 1.0000001);
        }

        qdtsne::internal::compute_gaussian_perplexity(neighbors, static_cast<float>(0.5), 1);

        for (int i = 0; i < leads; ++i) {
            EXPECT_FLOAT_EQ(first[i].second, 1.0/leads);
        }
        for (int i = leads; i < 90; ++i) {
            EXPECT_EQ(first[i].second, 0);
        }
    }
}
