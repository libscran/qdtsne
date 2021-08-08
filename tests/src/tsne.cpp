#include <gtest/gtest.h>

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
        indices.resize(N * K);
        distances.resize(N * K);

        for (size_t i = 0; i < N; ++i) {
            auto out = searcher.find_nearest_neighbors(i, K);
            for (size_t k = 0; k < out.size(); ++k) {
                indices[k + i *K] = out[k].first;
                distances[k + i *K] = out[k].second;
            }
            nn_index.push_back(indices.data() + i * K);
            nn_dist.push_back(distances.data() + i * K);
        }

        return;
    }

    std::vector<double> X;
    std::vector<int> indices;
    std::vector<double> distances;
    std::vector<const int*> nn_index;
    std::vector<const double*> nn_dist;
};

TEST_P(TsneTester, Initialization) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    qdtsne::Tsne thing;
    auto status = thing.initialize(nn_index, nn_dist, K);

    // Checking probabilities are all between zero and 1.
    const auto& probs = status.probabilities;
    EXPECT_EQ(probs.size(), nn_index.size());
    double total = 0;
    for (const auto& curp : probs) {
        EXPECT_TRUE(curp.size() >= K);
        for (const auto& p : curp) {
            EXPECT_TRUE(p < 1);
            EXPECT_TRUE(p > 0);
            total += p;
        }
    }
    EXPECT_FLOAT_EQ(total, 1);

    // Checking symmetry of the probabilities.
    std::map<std::pair<int, int>, std::tuple<double, bool, bool> > stuff;
    for (size_t n = 0; n < nn_index.size(); ++n) {
        const auto& curp = status.probabilities[n];
        const auto& curi = status.neighbors[n];
        EXPECT_EQ(curi.size(), curp.size());

        for (size_t x = 0; x < curp.size(); ++x) {
            auto neighbor = curi[x];
            EXPECT_TRUE(neighbor != n);

            std::pair<int, int> key(std::min((int)n, neighbor), std::max((int)n, neighbor)); // only consider combinations
            auto it = stuff.lower_bound(key);

            if (it != stuff.end() && it->first == key) {
                EXPECT_EQ(std::get<0>(it->second), curp[x]);

                // Checking that this permutation doesn't already exist.
                if (n > neighbor) {
                    EXPECT_FALSE(std::get<1>(it->second));
                    std::get<1>(it->second) = true;
                } else {
                    EXPECT_FALSE(std::get<2>(it->second));
                    std::get<2>(it->second) = true;
                }
            } else {
                stuff.insert(it, std::make_pair(key, std::make_tuple(curp[x], n > neighbor, n < neighbor)));
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
    std::vector<double> Y(N * 2);
    qdtsne::fill_initial_values(Y.data(), N);
    auto old = Y;
    auto status = thing.run(nn_index, nn_dist, K, Y.data());

    EXPECT_NE(old, Y); // there was some effect...
    EXPECT_EQ(status.iteration, 1000); // actually ran through the specified iterations

    // Check that coordinates are zero-mean.
    for (int d = 0; d < 2; ++d) {
        double total = 0;
        for (size_t i = 0; i < N; ++i){
            total += Y[2*i + d];
        }
        EXPECT_TRUE(std::abs(total/N) < 1e-10);
    }
}

TEST_P(TsneTester, StopStart) {
    auto PARAM = GetParam();
    size_t N = std::get<0>(PARAM);
    size_t D = std::get<1>(PARAM);
    size_t K = std::get<2>(PARAM);
    assemble(N, D, K);

    std::vector<double> Y(N * 2);
    qdtsne::fill_initial_values(Y.data(), N);
    auto copy = Y;

    qdtsne::Tsne thing;
    auto ref = thing.run(nn_index, nn_dist, K, Y.data());

    auto status = thing.initialize(nn_index, nn_dist, K);
    thing.set_max_iter(500).run(status, copy.data());
    thing.set_max_iter(1000).run(status, copy.data());

    for (size_t i =0 ; i < N; ++i) {
        std::cout << Y[i*2] << "\t" << Y[i*2+1] << std::endl;
    }

    EXPECT_EQ(copy, Y);
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
