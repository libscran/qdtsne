/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#ifndef QDTSNE_SPTREE_HPP
#define QDTSNE_SPTREE_HPP

#include <cmath>
#include <array>
#include <algorithm>
#include <vector>

namespace qdtsne {

template<int ndim, typename Float = double>
class SPTree {
public:
    SPTree(size_t n, int max) : N(n), maxdepth(max), locations(N) {
        store.reserve(std::min(static_cast<Float>(n), std::pow(static_cast<Float>(4.0), static_cast<Float>(maxdepth))) * 2);
        return;
    }

public:
    struct Node {
        static constexpr int nchildren = (1 << ndim); 

        Node(const Float* point) {
            std::copy_n(point, ndim, center_of_mass.data());
            fill();
            return;
        }

        Node() {
            std::fill_n(center_of_mass.data(), ndim, 0);
            fill();
            return;
        }

        void fill() {
            std::fill_n(midpoint.begin(), ndim, 0);
            std::fill_n(halfwidth.begin(), ndim, 0);
            std::fill_n(children.begin(), nchildren, 0);
        }

        std::array<size_t, nchildren> children;
        std::array<Float, ndim> midpoint, halfwidth;
        std::array<Float, ndim> center_of_mass;

        int number = 1;
        bool is_leaf = true;
    };

private:
    const Float * data = NULL;
    size_t N;
    int maxdepth;
    std::vector<Node> store;
    std::vector<size_t> locations;
    std::vector<size_t> self;

public:
    void set(const Float* Y) {
        data = Y;

        store.clear();
        store.resize(1);
        store[0].is_leaf = false;
        store[0].number = N;

        self.clear();
        self.push_back(0); // placeholder for the root node.

        {
            std::array<Float, ndim> min_Y{}, max_Y{};
            std::fill_n(min_Y.begin(), ndim, std::numeric_limits<Float>::max());
            std::fill_n(max_Y.begin(), ndim, std::numeric_limits<Float>::lowest());

            auto& mean_Y = store[0].midpoint;
            auto copy = Y;
            for (size_t n = 0; n < N; ++n) {
                for (int d = 0; d < ndim; ++d, ++copy) {
                    mean_Y[d] += *copy;
                    min_Y[d] = std::min(min_Y[d], *copy);
                    max_Y[d] = std::max(max_Y[d], *copy);
                }
            }

            for (size_t d = 0; d < ndim; ++d) {
                mean_Y[d] /= N;
            }

            auto& halfwidth = store[0].halfwidth;
            for (int d = 0; d < ndim; ++d) {
                halfwidth[d] = std::max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + static_cast<Float>(1e-5);
            }
        }

        auto point = Y;
        for (size_t i = 0; i < N; ++i, point += ndim) {
            std::array<bool, ndim> side;
            size_t parent = 0;
            size_t child_loc = 0;

            for (int depth = 1; depth <= maxdepth; ++depth) {
                size_t child_idx = find_child(parent, point, side.data());
                child_loc = store[parent].children[child_idx];

                // Be careful with persistent references to store's contents,
                // as the vector may be reallocated when a push_back() occurs.
                if (child_loc == 0) { 
                    child_loc = store.size();
                    store[parent].children[child_idx] = child_loc;
                    store.push_back(Node(point));
                    set_child_boundaries(parent, child_loc, side.data());
                    self.push_back(i);
                    break;
                } 

                if (store[child_loc].is_leaf && depth < maxdepth) {
                    // Shifting the current child to become a child of itself.
                    size_t grandchild_loc = store.size();
                    store.push_back(Node(store[child_loc].center_of_mass.data())); 

                    std::array<bool, ndim> side2; 
                    size_t grandchild_idx = find_child(child_loc, store[grandchild_loc].center_of_mass.data(), side2.data());
                    set_child_boundaries(child_loc, grandchild_loc, side2.data());

                    self.push_back(self[child_loc]);
                    locations[self[grandchild_loc]] = grandchild_loc;

                    store[child_loc].children[grandchild_idx] = grandchild_loc;
                    store[child_loc].is_leaf = false;
                }

                // Online update of cumulative size and center-of-mass
                auto& node = store[child_loc];
                ++node.number;
                const Float cum_size = node.number;
                const Float mult1 = (cum_size - 1) / cum_size;

                for (int d = 0; d < ndim; ++d) {
                    node.center_of_mass[d] *= mult1;
                    node.center_of_mass[d] += point[d] / cum_size;
                }

                parent = child_loc;
            }

            locations[i] = child_loc;
        }

        return;
    }

private:
    size_t find_child (size_t parent, const Float* point, bool * side) const {
        int multiplier = 1;
        size_t child = 0;
        for (int c = 0; c < ndim; ++c) {
            side[c] = (point[c] >= store[parent].midpoint[c]);
            child += multiplier * side[c];
            multiplier *= 2;
        }
        return child;
    }

    void set_child_boundaries(size_t parent, size_t child, const bool* keep) {
        auto& current = store[child];
        auto& parental = store[parent];
        for (int c = 0; c < ndim; ++c) {
            current.halfwidth[c] = parental.halfwidth[c] / static_cast<Float>(2);
            if (keep[c]) {
                current.midpoint[c] = parental.midpoint[c] + current.halfwidth[c];
            } else {
                current.midpoint[c] = parental.midpoint[c] - current.halfwidth[c];
            }
        }
        return;
    }

public:
    Float compute_non_edge_forces(size_t index, Float theta, Float* neg_f) const {
        Float result_sum = 0;
        const Float * point = data + index * ndim;
        const auto& cur_children = store[0].children;
        std::fill_n(neg_f, ndim, 0);

        for (int i = 0; i < cur_children.size(); ++i) {
            if (cur_children[i]) {
                result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
            }
        }

        return result_sum;
    }

    Float compute_non_edge_forces(const Float * point, Float theta, Float* neg_f) const {
        Float result_sum = 0;
        const auto& cur_children = store[0].children;
        std::fill_n(neg_f, ndim, 0);

        for (int i = 0; i < cur_children.size(); ++i) {
            if (cur_children[i]) {
                result_sum += compute_non_edge_forces(N, point, theta, neg_f, cur_children[i]);
            }
        }

        return result_sum;
    }

private:
    Float compute_non_edge_forces(size_t index, const Float* point, Float theta, Float* neg_f, size_t position) const {
        const auto& node = store[position];
        std::array<Float, ndim> temp;
        const Float * center = node.center_of_mass.data();

        if (index < N && position == locations[index]) {
            if (node.number == 1) {
                return 0; // skipping self.
            } else if (node.is_leaf) {
                for (int d = 0; d < ndim; ++d) { // subtracting self from the box for the force calculations.
                    temp[d] = (node.center_of_mass[d] * node.number - point[d]) / (node.number - 1);
                }
                center = temp.data();
            }
        }

        // Compute squared distance between point and center-of-mass
        Float sqdist = 0;
        for (int d = 0; d < ndim; ++d) {
            sqdist += (point[d] - center[d]) * (point[d] - center[d]);
        }

        // Check whether we can use this node as a "summary"
        bool skip_children = node.is_leaf;
        if (!skip_children) {
            Float max_halfwidth = *std::max_element(node.halfwidth.begin(), node.halfwidth.end());
            skip_children = (max_halfwidth < theta * std::sqrt(sqdist));
        }

        Float result_sum = 0;
        if (skip_children) {
            // Compute and add t-SNE force between point and current node.
            const Float div = static_cast<Float>(1) / (static_cast<Float>(1) + sqdist);
            Float mult = node.number * div;
            result_sum += mult;
            mult *= div;

            for (int d = 0; d < ndim; ++d) {
                neg_f[d] += mult * (point[d] - center[d]);
            }
        } else {
            // Recursively apply Barnes-Hut to children
            const auto& cur_children = node.children;
            for (int i = 0; i < cur_children.size(); ++i) {
                if (cur_children[i]) {
                    result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
                }
            }
        }

        return result_sum;
    }

public:
    // For testing purposes only.
    const auto& get_store() {
        return store;
    }

    const auto& get_locations() {
        return locations;
    }
};

}

#endif
