#include <gtest/gtest.h>

#ifdef CUSTOM_PARALLEL_TEST
// must be before any qdtsne includes.
#include "custom_parallel.h"
#endif

#include <random>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <array>

#include "knncolle/knncolle.hpp"

#include "qdtsne/initialize.hpp"

class TsneTester : public ::testing::TestWithParam<int> {
protected:
    inline static int ndim = 5;
    inline static int nobs = 200;
    inline static std::vector<double> X;
    inline static std::shared_ptr<knncolle::Builder<int, double, double> > builder;

    static void SetUpTestSuite() {
        X.resize(ndim * nobs);

        std::mt19937_64 rng(42);
        std::uniform_real_distribution<> dist(0, 1);
        for (auto& y : X) {
            y = dist(rng);
        }

        // Shift every second observation by 10 in the first dimension.
        for (int o = 1; o < nobs; o += 2) {
            X[o * ndim] += 10;
        }

        builder.reset(new knncolle::VptreeBuilder<int, double, double>(
            std::make_shared<knncolle::EuclideanDistance<double, double> >()
        ));
    }
};

TEST_P(TsneTester, Initialization) {
    int K = GetParam();

    qdtsne::Options opt;
    opt.perplexity = K / 3.0;
    auto status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    // Checking probabilities are all between zero and 1.
    const auto& probs = status.get_neighbors();
    EXPECT_EQ(probs.size(), nobs);
    double total = 0;
    for (const auto& curp : probs) {
        EXPECT_GE(curp.size(), K);
        for (const auto& p : curp) {
            EXPECT_TRUE(p.second < 1);
            EXPECT_TRUE(p.second > 0);
            total += p.second;
        }
    }
    EXPECT_FLOAT_EQ(total, 1);

    // Checking symmetry of the probabilities.
    std::map<std::pair<int, int>, std::tuple<double, bool, bool> > stuff;
    for (int n = 0; n < nobs; ++n) {
        const auto& current = probs[n];

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
    int K = GetParam();

    qdtsne::Options opt;
    opt.perplexity = K / 3.0;
    auto status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto Y = qdtsne::initialize_random<2>(nobs);
    auto old = Y;

    status.run(Y.data());
    EXPECT_NE(old, Y); // there was some effect...
    EXPECT_EQ(status.num_observations(), nobs); 
    EXPECT_EQ(status.iteration(), 1000); // actually ran through the specified iterations

    // Check that coordinates are zero-mean.
    for (int d = 0; d < 2; ++d) {
        double total = 0;
        for (int i = 0; i < nobs; ++i){
            total += Y[2*i + d];
        }
        EXPECT_TRUE(std::abs(total/nobs) < 1e-10);
    }

    // Sanity check for separation between the two groups of observations, at least on one dimension.
    std::array<int, 2> odd_above_zero{}, even_above_zero{};
    for (int i = 0; i < nobs; ++i) {
        auto& counter = (i % 2 == 0 ? even_above_zero : odd_above_zero);
        for (int d = 0; d < 2; ++d) {
            counter[d] += (Y[2 * i + d] > 0);
        }
    }
    EXPECT_TRUE(
        (odd_above_zero[0] == 0 && even_above_zero[0] == nobs/2) ||
        (odd_above_zero[1] == 0 && even_above_zero[1] == nobs/2) ||
        (odd_above_zero[0] == nobs/2 && even_above_zero[0] == 0) ||
        (odd_above_zero[1] == nobs/2 && even_above_zero[1] == 0)
    );

    // Same results when run in parallel.
    opt.num_threads = 3;
    auto pstatus = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto copy = old;
    pstatus.run(copy.data());
    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, StopStart) {
    int K = GetParam();

    qdtsne::Options opt;
    opt.perplexity = K / 3.0;
    auto status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto Y = qdtsne::initialize_random<2>(nobs);
    auto copy = Y;
    status.run(Y.data());

    auto restatus = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);
    restatus.run(copy.data(), 500);
    restatus.run(copy.data(), 1000);

    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, AltStart) {
    int K = GetParam();

    qdtsne::NeighborList<int, double> neighbors(nobs);
    {
        auto index = builder->build_unique(knncolle::SimpleMatrix(ndim, nobs, X.data()));
        auto searcher = index->initialize();
        std::vector<int> indices;
        std::vector<double> distances;
        for (int o = 0; o < nobs; ++o) {
            searcher->search(o, K, &indices, &distances);
            int actual_k = indices.size();
            for (int x = 0; x < actual_k; ++x) {
                neighbors[o].emplace_back(indices[x], distances[x]);
            }
        }
    }

    auto original = qdtsne::initialize_random<2>(nobs);
    qdtsne::Options options;

    auto Y = original;
    auto status = qdtsne::initialize<2>(std::move(neighbors), options);
    status.run(Y.data());

    options.perplexity = K / 3.0;
    auto ref_status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, options);
    auto copy = original;
    ref_status.run(copy.data());

    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, LeafApproximation) {
    int K = GetParam();

    qdtsne::Options opt;
    opt.perplexity = K / 3.0;
    opt.max_depth = 4;
    opt.leaf_approximation = true;
    auto status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto Y = qdtsne::initialize_random<2>(nobs);
    auto old = Y;

    status.run(Y.data());
    EXPECT_NE(old, Y); // there was some effect...
    EXPECT_EQ(status.num_observations(), nobs); 
    EXPECT_EQ(status.iteration(), 1000); // actually ran through the specified iterations

    // Check that coordinates are zero-mean.
    for (int d = 0; d < 2; ++d) {
        double total = 0;
        for (int i = 0; i < nobs; ++i){
            total += Y[2*i + d];
        }
        EXPECT_TRUE(std::abs(total/nobs) < 1e-10);
    }

    // Sanity check for separation between the means of the two groups of observations on at least one dimension.
    // The approximation does degrade the quality of the global embedding so we can't do a more precise check.
    std::array<double, 2> mean_odd{};
    std::array<double, 2> mean_even{};
    auto inf = std::numeric_limits<double>::infinity();
    std::array<double, 2> min_odd{ inf, inf }, max_odd{ -inf, -inf };
    std::array<double, 2> min_even{ inf, inf }, max_even{ -inf, -inf };
    for (int i = 0; i < nobs; ++i) {
        auto& mean = (i % 2 == 0 ? mean_even : mean_odd);
        auto& min = (i % 2 == 0 ? min_even : min_odd);
        auto& max = (i % 2 == 0 ? max_even : max_odd);
        for (int d = 0; d < 2; ++d) {
            auto val = Y[2 * i + d];
            mean[d] += val;
            min[d] = std::min(min[d], val);
            max[d] = std::max(max[d], val);
        }
    }
    for (int d = 0; d < 2; ++d) {
        mean_odd[d] /= nobs/2;
        mean_even[d] /= nobs/2;
    }
    EXPECT_TRUE( // just checking that the mean of each group is beyond the range of the other subpopulation.
        (mean_odd[0] < min_even[0] && mean_even[0] > max_odd[0]) ||
        (mean_odd[1] < min_even[1] && mean_even[1] > max_odd[1]) ||
        (mean_even[0] < min_odd[0] && mean_odd[0] > max_even[0]) ||
        (mean_even[1] < min_odd[1] && mean_odd[1] > max_even[1])
    );

    // Same results when run in parallel.
    opt.num_threads = 3;
    auto pstatus = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto copy = old;
    pstatus.run(copy.data());
    EXPECT_EQ(copy, Y);
}

TEST_P(TsneTester, Float32) {
    int K = GetParam();
    qdtsne::Options opt;
    opt.perplexity = K / 3.0;

    knncolle::VptreeBuilder<int, double, float> builder(std::make_shared<knncolle::EuclideanDistance<double, float> >());
    auto index = builder.build_unique(knncolle::SimpleMatrix<int, double>(ndim, nobs, X.data()));

    auto status = qdtsne::initialize<2>(*index, opt);
    auto Y = qdtsne::initialize_random<2, float>(nobs);
    auto old = Y;

    status.run(Y.data());
    EXPECT_NE(old, Y); // there was some effect...
    EXPECT_EQ(status.num_observations(), nobs); 
    EXPECT_EQ(status.iteration(), 1000); // actually ran through the specified iterations
}

TEST_P(TsneTester, Cost) {
    int K = GetParam();

    qdtsne::Options opt;
    opt.perplexity = K / 3.0;
    auto status = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);

    auto Y = qdtsne::initialize_random<2>(nobs);
    auto copy = Y;
    status.run(Y.data());
    auto cost = status.cost(Y.data());

    // Checking that cost_norun doesn't affect the results.
    auto restatus = qdtsne::initialize<2>(ndim, nobs, X.data(), *builder, opt);
    EXPECT_EQ(cost, restatus.cost(Y.data()));
    restatus.run(copy.data(), 500);
    EXPECT_EQ(cost, restatus.cost(Y.data()));
    restatus.run(copy.data(), 1000);

    EXPECT_EQ(copy, Y);
    EXPECT_EQ(cost, restatus.cost(Y.data()));
}

INSTANTIATE_TEST_SUITE_P(
    TsneTests,
    TsneTester,
    ::testing::Values(30, 60, 90) // number of neighbors
);
