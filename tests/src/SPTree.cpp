#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "qdtsne/SPTree.hpp"

class SPTreeTester : public ::testing::TestWithParam<std::tuple<int, int, bool> > {
protected:
    static constexpr int ndim = 2;

protected:
    template<class Store_>
    void validate_store(int position, const Store_& store, std::vector<int>& covered, int& leaf_count, int maxdepth, int depth = 0) {
        const auto& node = store[position];
        covered[position] = 1;

        // Checking the max depth is not exceeded.
        EXPECT_TRUE(depth <= maxdepth);

        // Check that halfwidth, midpoint and center of mass are all non-zero for non-leaf nodes.
        const auto& kids = node.children;
        if (node.is_leaf) {
            leaf_count += node.number;
            for (auto k : kids) {
                EXPECT_TRUE(k == 0);
            }
            return;
        }

        for (int d = 0; d < ndim; ++d) {
            EXPECT_TRUE(node.midpoint[d] != 0);
            EXPECT_TRUE(node.halfwidth[d] > 0);
            if (position != 0) { // ... except the first, for which we don't bother computing the center of mass.
                EXPECT_TRUE(node.center_of_mass[d] != 0);
                EXPECT_GE(node.center_of_mass[d], node.midpoint[d] - node.halfwidth[d]);
                EXPECT_LE(node.center_of_mass[d], node.midpoint[d] + node.halfwidth[d]);
            }
        }

        int child_counts = 0;
        for (size_t k = 0; k < kids.size(); ++k) {
            if (kids[k] == 0) {
                continue;
            }

            const auto& child = store[kids[k]];
            child_counts += child.number;

            if (!child.is_leaf) {
                auto copy = k;
                for (int d = 0; d < ndim; ++d, copy >>= 1) {
                    if (copy & 1) {
                        EXPECT_LT(node.midpoint[d], child.midpoint[d]);
                        EXPECT_GT(node.midpoint[d] + node.halfwidth[d], child.midpoint[d]);
                        EXPECT_LE(node.midpoint[d], child.center_of_mass[d]);
                        EXPECT_GE(node.midpoint[d] + node.halfwidth[d], child.center_of_mass[d]);
                    } else {
                        EXPECT_GT(node.midpoint[d], child.midpoint[d]);
                        EXPECT_LT(node.midpoint[d] - node.halfwidth[d], child.midpoint[d]);
                        EXPECT_GE(node.midpoint[d], child.center_of_mass[d]);
                        EXPECT_LE(node.midpoint[d] - node.halfwidth[d], child.center_of_mass[d]);
                    }
                    EXPECT_EQ(node.halfwidth[d] / 2, child.halfwidth[d]);
                }
            }

            validate_store(kids[k], store, covered, leaf_count, maxdepth, depth + 1);
        }

        // Verifying that the number here is the sum of the counts in the children.
        EXPECT_EQ(child_counts, node.number);
        EXPECT_GT(node.number, 0);
    }

    template<class Tree_>
    void validate_tree(const std::vector<double>& Y, const Tree_& tree, size_t N, int maxd) {
        const auto& store = tree.get_store();

        // Checking all points are within the root's box.
        for (size_t n = 0; n < N; ++n) {
            for (int d = 0; d < ndim; ++d) {
                double pos = Y[n * ndim + d];
                EXPECT_LT(pos, store[0].midpoint[d] + store[0].halfwidth[d]);
                EXPECT_GT(pos, store[0].midpoint[d] - store[0].halfwidth[d]);
            }
        }

        std::vector<int> covered(store.size());
        int leaf_count = 0;
        validate_store(0, store, covered, leaf_count, maxd);
        EXPECT_EQ(covered, std::vector<int>(covered.size(), 1)); // checking that we hit every node of the tree.
        EXPECT_EQ(N, leaf_count); // checking that the counts match up.

        // Checking that the locations are correct.
        const auto& locations = tree.get_locations();
        EXPECT_EQ(locations.size(), N);
        std::vector<std::array<double, ndim> > accumulated_sum(store.size());
        std::vector<int> accumulated_count(store.size());
        for (size_t n = 0; n < N; ++n) {
            EXPECT_GT(locations[n], 0);
            const auto& locale = store[locations[n]];
            EXPECT_TRUE(locale.is_leaf);

            if (locale.number == 1) {
                for (int d = 0; d < ndim; ++d) {
                    EXPECT_EQ(Y[n * ndim + d], locale.center_of_mass[d]);
                }
            } else {
                auto& current = accumulated_sum[locations[n]];
                for (int d = 0; d < ndim; ++d) {
                    current[d] += Y[n * ndim + d];
                }
                ++accumulated_count[locations[n]];
            }
        }

        for (size_t n = 0; n < N; ++n) {
            const auto& locale = store[locations[n]];
            if (locale.number > 1) {
                const auto& current = accumulated_sum[locations[n]];
                double count = accumulated_count[locations[n]];
                for (int d = 0; d < ndim; ++d) {
                    EXPECT_FLOAT_EQ(locale.center_of_mass[d], current[d] / count);
                }
            }
        }
    }

protected:
    double reference_non_edge_forces(const double* point, const double* data, size_t N, double* neg_f) const {
        double resultSum = 0;
        std::fill_n(neg_f, ndim, 0);

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

    template<class Tree_>
    double reference_non_edge_forces_approx(size_t self, const double* point, const Tree_& tree, double* neg_f) const {
        double resultSum = 0;
        std::fill_n(neg_f, ndim, 0);

        const auto& store = tree.get_store();
        const auto& locations = tree.get_locations();
        size_t N = locations.size();

        std::array<double, ndim> recenter;
        auto self_location = locations[self];
        const auto& self_store = store[self_location];
        if (self_store.number > 1) {
            for (int d = 0; d < ndim; ++d) {
                recenter[d] = (self_store.center_of_mass[d] * self_store.number - point[d]) / (self_store.number - 1);
            }
        }

        for (size_t n = 0; n < N; ++n) {
            if (n == self) {
                continue;
            }

            auto loc = locations[n];
            const auto& data = (loc == self_location ? recenter : store[loc].center_of_mass);

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

TEST_P(SPTreeTester, CheckTree) {
    auto param = GetParam();
    size_t N = std::get<0>(param);
    size_t maxd = std::get<1>(param);
    size_t dup = std::get<2>(param);

    std::vector<double> Y(N * ndim);
    {
        std::mt19937_64 rng(N + maxd);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : Y) {
            y = dist(rng);
        }
    }

    if (dup) {
        auto copy = Y;
        Y.insert(Y.end(), copy.begin(), copy.end());
        N *= 2;
    }

    qdtsne::internal::SPTree<2, double> tree(N, maxd);
    tree.set(Y.data());
    validate_tree(Y, tree, N, maxd);

    // Cursory check for non-zero theta.
    int top = std::min(static_cast<int>(N), 20); // computing just the top set for simplicity.
    for (int i = 0; i < top; ++i) {
        std::vector<double> neg_f(2);
        auto output = tree.compute_non_edge_forces(i, 0.5, neg_f.data());

        EXPECT_TRUE(output > 0);
        for (size_t d = 0; d < neg_f.size(); ++d) {
            EXPECT_TRUE(neg_f[d] != 0);
        }
    }

    // Checking if the tree has 1:1 mappings from points to leaf nodes.
    bool is_one_to_one = true;
    const auto& store = tree.get_store();
    for (const auto& s : store) {
        if (s.is_leaf && s.number > 1) {
            is_one_to_one = false;
            break;
        }
    }

    if (dup) {
        EXPECT_FALSE(is_one_to_one);
    } else if (maxd == 20) {
        EXPECT_TRUE(is_one_to_one);
    }

    if (is_one_to_one || (dup && maxd == 20)) { 
        std::vector<double> neg_f(2);
        std::vector<double> neg_f_ref(2);

        int top = std::min(static_cast<int>(N), 20); // computing just the top set for simplicity.
        for (int i = 0; i < top; ++i) {
            double no_theta = tree.compute_non_edge_forces(i, 0, neg_f.data()); // set theta=0 for exact calculation.
            double ref = reference_non_edge_forces(Y.data() + i * ndim, Y.data(), N, neg_f_ref.data());

            EXPECT_FLOAT_EQ(neg_f_ref[0], neg_f[0]);
            EXPECT_FLOAT_EQ(neg_f_ref[1], neg_f[1]);
            EXPECT_FLOAT_EQ(no_theta, ref);
        }

        const auto& locations = tree.get_locations();
        if (is_one_to_one) {
            for (auto l : locations) {
                EXPECT_EQ(store[l].number, 1);
            }
        } else {
            for (auto l : locations) {
                EXPECT_EQ(store[l].number, 2);
            }
        }

    } else {
        std::vector<double> neg_f(2);
        std::vector<double> neg_f_ref(2);

        int top = std::min(static_cast<int>(N), 20); // computing just the top set for simplicity.
        for (int i = 0; i < top; ++i) {
            double no_theta = tree.compute_non_edge_forces(i, 0, neg_f.data()); // set theta=0 for exact calculation.
            double ref = reference_non_edge_forces_approx(i, Y.data() + i * ndim, tree, neg_f_ref.data());

            EXPECT_FLOAT_EQ(neg_f_ref[0], neg_f[0]);
            EXPECT_FLOAT_EQ(neg_f_ref[1], neg_f[1]);
            EXPECT_FLOAT_EQ(no_theta, ref);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SPTree,
    SPTreeTester,
    ::testing::Combine(
        ::testing::Values(10, 100, 1000), // number of observations
        ::testing::Values(3, 7, 20), // max depth
        ::testing::Values(false, true) // duplicated
    )
);
