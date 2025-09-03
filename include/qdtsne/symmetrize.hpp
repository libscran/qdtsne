#ifndef QDTSNE_SYMMETRIZE_HPP
#define QDTSNE_SYMMETRIZE_HPP

#include <vector>
#include <algorithm>

#include "sanisizer/sanisizer.hpp"

#include "utils.hpp"

namespace qdtsne {

namespace internal {

template<typename Index_, typename Float_>
void symmetrize_matrix(NeighborList<Index_, Float_>& x) {
    const Index_ num_points = x.size(); // assume that Index_ is sufficient to hold the number of observations.
    auto last = sanisizer::create<std::vector<Index_> >(num_points);
    auto original = sanisizer::create<std::vector<Index_> >(num_points);

    Float_ total = 0;
    for (Index_ i = 0; i < num_points; ++i) {
        auto& current = x[i];
        std::sort(current.begin(), current.end()); // sorting by ID, see below.

        original[i] = current.size();
        for (auto& y : current) {
            total += y.second;
        }
    }

    for (Index_ i = 0; i < num_points; ++i) {
        auto& current = x[i];

        // Looping through the neighbors and searching for self in each
        // neighbor's neighbors. Assuming that the each neighbor list is sorted
        // by index up to the original size of the list (i.e., excluding newly
        // appended elements from symmetrization), this should only require a
        // single pass through the entire set of neighbors as we do not need to
        // search previously searched hits.
        for (auto& y : current) {
            auto& target = x[y.first];
            auto& curlast = last[y.first];
            auto limits = original[y.first];
            while (curlast < limits && target[curlast].first < i) {
                ++curlast;
            }

            if (curlast < limits && target[curlast].first == i) {
                if (i < y.first) { 
                    // Adding the probabilities - but if i > y.first, then this
                    // would have already been done in a previous iteration of
                    // the outermost loop where i and y.first swap values. So
                    // we skip this to avoid adding it twice.
                    const Float_ combined = y.second + target[curlast].second;
                    y.second = combined;
                    target[curlast].second = combined;
                }
            } else {
                target.emplace_back(i, y.second);
            }
        }
    }

    // Divide the result by twice the total, so that it all sums to unity.
    total *= static_cast<Float_>(2);
    for (auto& current : x) {
        for (auto& y : current) {
            y.second /= total;
        }

        // Sorting to obtain increasing indices, which should be more cache
        // friendly in the edge force calculations in tsne.hpp.
        std::sort(current.begin(), current.end());
    }

    return;
}

}

}

#endif
