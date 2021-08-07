#include <gtest/gtest.h>

#include "qdtsne/sptree.hpp"
#include <random>

class SPTreeTester : public ::testing::TestWithParam<int> {
protected:
    void assemble(int n) {
        Y.resize(n * 2);

        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : Y) {
            y = dist(rng);
        }
    }

    std::vector<double> Y;
};

TEST_P(SPTreeTester, BuildWithLowCap) {
    size_t N = GetParam();
    assemble(N);

    qdtsne::SPTree<2, 3> tree(N);
    tree.set(Y.data());
}

TEST_P(SPTreeTester, BuildWithDefaultCap) {
    size_t N = GetParam();
    assemble(N);

    qdtsne::SPTree<2, 7> tree(N);
    tree.set(Y.data());
}

TEST_P(SPTreeTester, BuildUncapped) {
    size_t N = GetParam();
    assemble(N);

    qdtsne::SPTree<2, 100> tree(N);
    tree.set(Y.data());
}

INSTANTIATE_TEST_CASE_P(
    SPTree,
    SPTreeTester,
    ::testing::Values(10, 500, 1000) // number of observations
);
