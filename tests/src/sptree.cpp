#include <gtest/gtest.h>

#include "qdtsne/sptree.hpp"
#include <random>

class SPTreeTester : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    static constexpr int ndim = 2;

    void assemble(int n) {
        Y.resize(n * ndim);

        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : Y) {
            y = dist(rng);
        }
    }

    std::vector<double> Y;

protected:
    template<class V>
    void validate_store(const V& store, const double* data, size_t N, int maxdepth) {
        std::vector<int> covered(store.size());
        int leaf_count = 0, depth = 0;

        validate_store(0, store, covered, leaf_count, depth);
        for (auto c : covered) {
            EXPECT_EQ(c, 1); // checking that we hit every node of the tree.
        }

        EXPECT_EQ(N, leaf_count); // checking that the counts match up.

        for (size_t n = 0; n < N; ++n, data += ndim) { 
            // Checking all points are within range of the global thing.
            for (int d = 0; d < ndim; ++d) {
                EXPECT_TRUE(data[d] < store[0].midpoint[d] + store[0].halfwidth[d]);
                EXPECT_TRUE(data[d] > store[0].midpoint[d] - store[0].halfwidth[d]);
            }
        }

        // Checking the max depth is not exceeded.
        for (const auto& s : store) {
            EXPECT_TRUE(s.depth <= maxdepth);
        }

        return;
    }

    template<class V>
    void validate_store(int position, const V& store, std::vector<int>& covered, int& leaf_count, int depth) {
        const auto& node = store[position];
        covered[position] = 1;
        EXPECT_EQ(depth, node.depth);
            
        // Check that halfwidth, midpoint and center of mass are all non-zero.
        for (int d = 0; d < ndim; ++d) {
            EXPECT_TRUE(node.midpoint[d] != 0);
            EXPECT_TRUE(node.halfwidth[d] > 0);

            if (position != 0) { // ... except the first, for which we don't bother computing the center of mass.
                EXPECT_TRUE(node.center_of_mass[d] != 0);
                EXPECT_TRUE(node.center_of_mass[d] >= node.midpoint[d] - node.halfwidth[d]);
                EXPECT_TRUE(node.center_of_mass[d] <= node.midpoint[d] + node.halfwidth[d]);
            }
        }

        const auto& kids = node.children;
        if (node.is_leaf) {
            leaf_count += node.number;
            for (auto k : kids) {
                EXPECT_TRUE(k == 0);
            }
            return;
        }

        int child_counts = 0;
        for (size_t k = 0; k < kids.size(); ++k) {
            if (kids[k] == 0) {
                continue;
            }

            const auto& child = store[kids[k]];
            child_counts += child.number;

            auto copy = k;
            for (int d = 0; d < ndim; ++d, copy >>= 1) {
                if (copy & 1) {
                    EXPECT_TRUE(node.midpoint[d] < child.midpoint[d]);
                    EXPECT_TRUE(node.midpoint[d] + node.halfwidth[d] > child.midpoint[d]);
                } else {
                    EXPECT_TRUE(node.midpoint[d] > child.midpoint[d]);
                    EXPECT_TRUE(node.midpoint[d] - node.halfwidth[d] < child.midpoint[d]);
                }
                EXPECT_EQ(node.halfwidth[d] / 2, child.halfwidth[d]);
            }

            validate_store(kids[k], store, covered, leaf_count, depth + 1);
        }

        // Verifying that the number here is the sum of the counts in the children.
        EXPECT_EQ(child_counts, node.number);
        EXPECT_TRUE(node.number > 0);
    }

protected:
    double reference_non_edge_forces(const double* point, const double* data, size_t N, double* neg_f) const {
        double resultSum = 0;
        for (size_t n = 0; n < N; ++n, data += ndim) {
            if (point == data) {
                continue;
            }

            double sqdist = 0;
            for(int d = 0; d < ndim; d++) {
                sqdist += (point[d] - data[d]) * (point[d] - data[d]);
            }

            sqdist = 1.0 / (1.0 + sqdist);
            double mult = sqdist;
            resultSum += mult;
            mult *= sqdist;

            for (int d = 0; d < ndim; ++d) {
                neg_f[d] += mult * (point[d] - data[d]);
            }
        }
        return resultSum;
    }
};

TEST_P(SPTreeTester, CheckTree2) {
    auto param = GetParam();
    size_t N = std::get<0>(param);
    assemble(N);

    size_t maxd = std::get<1>(param);
    qdtsne::SPTree<2> tree(N, maxd);
    tree.set(Y.data());

    // Validating the tree.
    validate_store(tree.get_store(), Y.data(), N, maxd);

    std::vector<double> neg_f(2);
    auto output = tree.compute_non_edge_forces(0, 0.5, neg_f.data());

    EXPECT_TRUE(output > 0);
    for (int d = 0; d < neg_f.size(); ++d) {
        EXPECT_TRUE(neg_f[d] != 0);
    }

    // Checking against a reference, if the tree is exact.
    if (std::pow(2.0, maxd) > N) {
        double no_theta = tree.compute_non_edge_forces(0, 0, neg_f.data());

        std::vector<double> neg_f_ref(2);
        double ref = reference_non_edge_forces(Y.data(), Y.data(), N, neg_f_ref.data());

        EXPECT_FLOAT_EQ(neg_f_ref[0], neg_f[0]);
        EXPECT_FLOAT_EQ(neg_f_ref[1], neg_f[1]);
        EXPECT_FLOAT_EQ(no_theta, ref);
    }
}

INSTANTIATE_TEST_CASE_P(
    SPTree,
    SPTreeTester,
    ::testing::Combine(
        ::testing::Values(10, 500, 1000), // number of observations
        ::testing::Values(3, 7, 100) // max depth
    )
);
