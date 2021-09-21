#include <gtest/gtest.h>

#include "qdtsne/interpolate.hpp"

TEST(InterpolateUtils, Encode) {
    std::array<double, 2> thing { 1.1, 2.5 };
    std::array<double, 2> mins { 1, 2 };
    std::array<double, 2> steps { 0.07, 0.15 };

    std::array<size_t, 2> expected { 1, 3 };
    EXPECT_EQ(qdtsne::interpolator::encode<2>(thing.data(), mins, steps, 10), expected);
}

TEST(InterpolateUtils, Hash) {
    std::array<size_t, 2> expected { 1, 3 };
    EXPECT_EQ(qdtsne::interpolator::hash<2>(expected, 10), 14); // remember, we +1 the intervals.
    EXPECT_EQ(qdtsne::interpolator::unhash<2>(14, 10), expected);

    std::array<size_t, 2> expected2 { 9, 9 };
    EXPECT_EQ(qdtsne::interpolator::hash<2>(expected2, 10), 108);
    EXPECT_EQ(qdtsne::interpolator::unhash<2>(108, 10), expected2);
}

TEST(InterpolateUtils, Corners) {
    std::array<size_t, 2> thing { 1, 5 };
    int intervals = 10;

    std::unordered_map<size_t, size_t> collected;
    collected[qdtsne::interpolator::hash<2>(thing, intervals)] = 0;
    qdtsne::interpolator::populate_corners<2>(collected, thing, intervals);

    EXPECT_EQ(collected.size(), 4);
    ++thing[0];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing, intervals)], -1);
    ++thing[1];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing, intervals)], -1);
    --thing[0];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing, intervals)], -1);

    // Doesn't overwrite existing 0's.
    std::array<size_t, 2> thing2 { 0, 5 };
    collected[qdtsne::interpolator::hash<2>(thing2, intervals)] = 0;
    qdtsne::interpolator::populate_corners<2>(collected, thing2, intervals);

    EXPECT_EQ(collected.size(), 6);
    ++thing2[0];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing2, intervals)], 0);
    ++thing2[1];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing2, intervals)], -1);
    --thing2[0];
    EXPECT_EQ(collected[qdtsne::interpolator::hash<2>(thing2, intervals)], -1);
}

class InterpolateTester : public ::testing::TestWithParam<std::tuple<int, int> > {
protected:
    static constexpr int ndim = 2;

    template<class Param>
    void assemble(Param param) {
        N = std::get<0>(param);
        intervals = std::get<1>(param);
        Y.resize(N * ndim);

        std::mt19937_64 rng(42);
        std::normal_distribution<> dist(0, 1);
        for (auto& y : Y) {
            y = dist(rng);
        }

        // Constructing a grid.
        std::fill_n(mins.begin(), ndim, std::numeric_limits<double>::max());
        std::fill_n(maxs.begin(), ndim, std::numeric_limits<double>::lowest());
        const auto* copy = Y.data();
        for (size_t i = 0; i < N; ++i) {
            for (int d = 0; d < ndim; ++d, ++copy) {
                mins[d] = std::min(mins[d], *copy);
                maxs[d] = std::max(maxs[d], *copy);
            }
        }

        for (int d = 0; d < ndim; ++d) {
            step[d] = (maxs[d] - mins[d]) / intervals;
            if (step[d] == 0) {
                step[d] = 1e-8; 
            }
        }

        return;
    }

    int N;
    int intervals;
    std::vector<double> Y;
    qdtsne::interpolator::coords<ndim> mins, maxs, step;
};

TEST_P(InterpolateTester, ExactGrid) {
    assemble(GetParam());
    qdtsne::SPTree<ndim> tree(N, 20);
    tree.set(Y.data());

    // Constructing the grid.
    int npts = intervals + 1;
    std::vector<double> grid(npts * npts * ndim);
    for (int i = 0; i <= intervals; ++i) {
        for (int j = 0; j <= intervals; ++j) {
            double* g = grid.data() + (i * npts + j) * ndim;
            g[0] = mins[0] + step[0] * i;
            g[1] = mins[1] + step[1] * j;
        }
    }

    // Computing forces through the grid; results should be the same as the
    // forces computed directly on the grid points.
    std::vector<double> neg(npts * npts * ndim);
    qdtsne::interpolator::interpolate_non_edge_forces(tree, npts * npts, grid.data(), 0.1, neg.data(), intervals);

    for (int i = 0; i <= intervals; ++i) {
        for (int j = 0; j <= intervals; ++j) {
            std::vector<double> x(ndim);
            int offset = (i * npts + j) * ndim;
            const double* g = grid.data() + offset; 
            tree.compute_non_edge_forces(g, 0.1, x.data());
            EXPECT_FLOAT_EQ(x[0], neg[offset]);
            EXPECT_FLOAT_EQ(x[1], neg[offset + 1]);
        }
    }
}

TEST_P(InterpolateTester, MidGrid) {
    assemble(GetParam());
    qdtsne::SPTree<ndim> tree(N, 20);
    tree.set(Y.data());

    // Constructing the midpoints.
    std::vector<double> grid(intervals * intervals * ndim);
    for (int i = 0; i < intervals; ++i) {
        for (int j = 0; j < intervals; ++j) {
            double* g = grid.data() + (i * intervals + j) * ndim;
            g[0] = mins[0] + step[0] * i + step[0] / 2;
            g[1] = mins[1] + step[1] * j + step[1] / 2;
        }
    }

    // Adding the limits so that the interpolation is done right.
    grid.push_back(mins[0]);
    grid.push_back(mins[1]);
    grid.push_back(maxs[0]);
    grid.push_back(maxs[1]);

    // Computing forces through the grid; results should be the same as the
    // forces computed directly on the grid points.
    std::vector<double> neg(grid.size());
    qdtsne::interpolator::interpolate_non_edge_forces(tree, grid.size() / ndim, grid.data(), 0.1, neg.data(), intervals);

    for (int i = 0; i < intervals; ++i) {
        for (int j = 0; j < intervals; ++j) {
            int offset = (i * intervals + j) * ndim;

            double ref1 = 0, ref2 = 0;
            {
                std::vector<double> G{ mins[0] + step[0] * i, mins[1] + step[1] * j };
                std::vector<double> x(ndim);
                tree.compute_non_edge_forces(G.data(), 0.1, x.data());
                ref1 += x[0];
                ref2 += x[1];
            }
            {
                std::vector<double> G{ mins[0] + step[0] * i, mins[1] + step[1] * (j + 1) };
                std::vector<double> x(ndim);
                tree.compute_non_edge_forces(G.data(), 0.1, x.data());
                ref1 += x[0];
                ref2 += x[1];
            }
            {
                std::vector<double> G{ mins[0] + step[0] * (i + 1), mins[1] + step[1] * j };
                std::vector<double> x(ndim);
                tree.compute_non_edge_forces(G.data(), 0.1, x.data());
                ref1 += x[0];
                ref2 += x[1];
            }
            {
                std::vector<double> G{ mins[0] + step[0] * (i + 1), mins[1] + step[1] * (j + 1) };
                std::vector<double> x(ndim);
                tree.compute_non_edge_forces(G.data(), 0.1, x.data());
                ref1 += x[0];
                ref2 += x[1];
            }

            EXPECT_FLOAT_EQ(ref1/4, neg[offset]);
            EXPECT_FLOAT_EQ(ref2/4, neg[offset + 1]);
        }
    }
}


INSTANTIATE_TEST_CASE_P(
    Interpolate,
    InterpolateTester,
    ::testing::Combine(
        ::testing::Values(10, 100, 1000), // number of observations
        ::testing::Values(20, 100) // number of intervals 
    )
);
