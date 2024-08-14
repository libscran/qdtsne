#include <gtest/gtest.h>
#include "qdtsne/utils.hpp"

class InitTester : public ::testing::TestWithParam<std::tuple<int, int> > {};

TEST_P(InitTester, Normality) {
    auto param = GetParam();
    auto Y = qdtsne::initialize_random<2>(std::get<0>(param), std::get<1>(param));

    EXPECT_NE(Y.back(), 0); // last element should be filled correctly in the odd case.

    std::sort(Y.begin(), Y.end());

    EXPECT_TRUE(Y.front() < 0);
    EXPECT_TRUE(Y.back() > 0);

    double mean = std::accumulate(Y.begin(), Y.end(), 0.0) / Y.size();
    EXPECT_TRUE(mean > -0.1);
    EXPECT_TRUE(mean < 0.1);

    double var = 0;
    for (auto y : Y) {
        var += (y - mean) * (y - mean);
    }
    var /= Y.size() - 1;
    EXPECT_TRUE(var > 0.9);
    EXPECT_TRUE(var < 1.1);

    for (size_t i = 1; i < Y.size(); ++i) {
        EXPECT_TRUE(Y[i-1] < Y[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(
    InitTests,
    InitTester,
    ::testing::Combine(
        ::testing::Values(1000, 1001), // even AND odd
        ::testing::Values(42, 100, 0) // various seeds
    )
);
