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

namespace internal {

template<int ndim_, typename Float_>
class SPTree {
public:
    SPTree(size_t npts, int maxdepth) : my_npts(npts), my_maxdepth(maxdepth), my_locations(my_npts) {
        my_store.reserve(std::min(static_cast<Float_>(my_npts), std::pow(static_cast<Float_>(4.0), static_cast<Float_>(my_maxdepth))) * 2);
        return;
    }

public:
    struct Node {
        Node(size_t i, const Float_* point) : index(i) {
            std::copy_n(point, ndim_, center_of_mass.data());
            fill();
            return;
        }

        Node() {
            std::fill(center_of_mass.begin(), center_of_mass.end(), 0);
            fill();
            return;
        }

    public:
        static constexpr int nchildren = (1 << ndim_); 

        std::array<size_t, nchildren> children;
        std::array<Float_, ndim_> midpoint, halfwidth;
        std::array<Float_, ndim_> center_of_mass;
        Float_ max_halfwidth = 0;

        size_t number = 1;

        // This should only be used when is_leaf = true. In cases where multiple
        // points are assigned to the same leaf node (e.g., duplicates, max depth
        // truncation), it is the index of the first point for this Node.
        size_t index = -1; 

        bool is_leaf = true;

    private:
        void fill() {
            std::fill_n(midpoint.begin(), ndim_, 0);
            std::fill_n(halfwidth.begin(), ndim_, 0);
            std::fill_n(children.begin(), nchildren, 0);
        }
    };

private:
    const Float_ * my_data = NULL;
    size_t my_npts;
    int my_maxdepth;
    std::vector<Node> my_store;

    // We need to store the on-tree locations for each point separately as each
    // point may not have a 1:1 mapping to a single Node if there are
    // duplicates or we truncate the tree at 'maxdepth'.
    std::vector<size_t> my_locations;

    std::vector<size_t> my_first_assignment;

public:
    void set(const Float_* Y) {
        my_data = Y;

        {
            my_store.clear();
            my_store.resize(1);
            my_store[0].is_leaf = false;
            my_store[0].number = my_npts;

            std::array<Float_, ndim_> min_Y{}, max_Y{};
            std::fill_n(min_Y.begin(), ndim_, std::numeric_limits<Float_>::max());
            std::fill_n(max_Y.begin(), ndim_, std::numeric_limits<Float_>::lowest());

            // Setting the initial midpoint to the center of mass of all points
            // so that the partitioning effectively divides up the points.
            auto& mean_Y = my_store[0].midpoint;
            auto copy = Y;
            for (size_t n = 0; n < my_npts; ++n) {
                for (int d = 0; d < ndim_; ++d) {
                    auto curval = copy[d];
                    mean_Y[d] += curval;
                    min_Y[d] = std::min(min_Y[d], curval);
                    max_Y[d] = std::max(max_Y[d], curval);
                }
                copy += ndim_;
            }

            for (int d = 0; d < ndim_; ++d) {
                mean_Y[d] /= my_npts;
            }

            auto& halfwidth = my_store[0].halfwidth;
            for (int d = 0; d < ndim_; ++d) {
                auto mean = mean_Y[d];
                halfwidth[d] = std::max(max_Y[d] - mean, mean - min_Y[d]) + static_cast<Float_>(1e-5);
            }
        }

        auto point = Y;
        my_first_assignment.resize(my_npts);
        for (size_t i = 0; i < my_npts; ++i, point += ndim_) {
            std::array<bool, ndim_> side;
            size_t parent = 0;

            for (int depth = 1; depth <= my_maxdepth; ++depth) {
                size_t child_idx = find_child(parent, point, side.data());

                // Be careful with persistent references to my_store's contents,
                // as the vector may be reallocated when a push_back() occurs.
                size_t current_loc = my_store[parent].children[child_idx];

                // If current_loc refers to the root, this means that there is no
                // existing child, so we make a new one.
                if (current_loc == 0) { 
                    current_loc = my_store.size();
                    my_store[parent].children[child_idx] = current_loc;
                    my_store.emplace_back(i, point);
                    my_first_assignment[i] = i;
                    break;
                } 

                if (my_store[current_loc].is_leaf) {
                    // Check if it's a duplicate, in which case we quit immediately.
                    // No need to update the center of mass because it's the same point.
                    int nsame = 0;
                    const auto& center = my_store[current_loc].center_of_mass;
                    for (int d = 0; d < ndim_; ++d) {
                        nsame += (center[d] == point[d]);
                    }
                    if (nsame == ndim_) {
                        ++(my_store[current_loc].number);
                        my_first_assignment[i] = my_store[current_loc].index;
                        break;
                    }

                    if (depth == my_maxdepth) {
                        my_first_assignment[i] = my_store[current_loc].index;
                    } else {
                        // Otherwise, we convert the current node into a non-leaf node to
                        // accommodate further recursion.
                        size_t new_loc = my_store.size();

                        // Push the entire Node! Don't emplace_back() with the
                        // pointer for the child's center of mass, as any potential
                        // re-allocation of the vector would invalidate the pointer
                        // to the center of mass stored inside the vector.
                        my_store.push_back(my_store[current_loc]);

                        my_store[current_loc].is_leaf = false;
                        set_child_boundaries(parent, current_loc, side.data());

                        size_t new_child_idx = find_child(current_loc, my_store[new_loc].center_of_mass.data(), side.data());
                        my_store[current_loc].children[new_child_idx] = new_loc;
                    }
                }

                // Online update of non-leaf node's cumulative size and center-of-mass.
                auto& node = my_store[current_loc];
                ++node.number;

                const Float_ cum_size = node.number;
                const Float_ mult1 = (cum_size - 1) / cum_size;

                for (int d = 0; d < ndim_; ++d) {
                    node.center_of_mass[d] *= mult1;
                    node.center_of_mass[d] += point[d] / cum_size;
                }

                parent = current_loc;
            }
        }

        // Populating the on-tree locations for each node.
        my_locations.resize(my_npts);
        size_t nnodes = my_store.size();
        for (size_t n = 0; n < nnodes; ++n) {
            const auto& node = my_store[n];
            if (node.is_leaf) {
                my_locations[node.index] = n;
            }
        }

        for (size_t i = 0; i < my_npts; ++i) {
            auto tmp = my_locations[my_first_assignment[i]]; // break it up to avoid unsequencing errors when assigning to self.
            my_locations[i] = tmp;
        } 

        return;
    }

private:
    size_t find_child (size_t parent, const Float_* point, bool * side) const {
        int multiplier = 1;
        size_t child = 0;
        for (int d = 0; d < ndim_; ++d) {
            side[d] = (point[d] >= my_store[parent].midpoint[d]);
            child += multiplier * side[d];
            multiplier *= 2;
        }
        return child;
    }

    void set_child_boundaries(size_t parent, size_t child, const bool* keep) {
        auto& current = my_store[child];
        auto& parental = my_store[parent];
        for (int d = 0; d < ndim_; ++d) {
            current.halfwidth[d] = parental.halfwidth[d] / static_cast<Float_>(2);
            if (keep[d]) {
                current.midpoint[d] = parental.midpoint[d] + current.halfwidth[d];
            } else {
                current.midpoint[d] = parental.midpoint[d] - current.halfwidth[d];
            }
        }

        // Compute once for the theta calculations.
        current.max_halfwidth = *std::max_element(current.halfwidth.begin(), current.halfwidth.end());
    }

public:
    Float_ compute_non_edge_forces(size_t index, Float_ theta, Float_* neg_f) const {
        Float_ result_sum = 0;
        const Float_ * point = my_data + index * static_cast<size_t>(ndim_); // cast to avoid overflow.
        const auto& cur_children = my_store[0].children;
        std::fill_n(neg_f, ndim_, 0);

        for (int i = 0; i < Node::nchildren; ++i) {
            if (cur_children[i]) {
                result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
            }
        }

        return result_sum;
    }

private:
    Float_ compute_non_edge_forces(size_t index, const Float_* point, Float_ theta, Float_* neg_f, size_t position) const {
        const auto& node = my_store[position];

        std::array<Float_, ndim_> temp;
        const Float_ * center = node.center_of_mass.data();
        size_t count = node.number;

        if (position == my_locations[index]) { // check if we're at the leaf node containing the 'index' point.
            if (count == 1) {
                return 0; // skipping node that only contains self.
            }

            for (int d = 0; d < ndim_; ++d) { // removing self from the center of mass for the force calculations.
                temp[d] = (node.center_of_mass[d] * count - point[d]) / (count - 1);
            }
            center = temp.data();
            --count;
        }

        // Compute squared distance between point and center-of-mass
        Float_ sqdist = 0;
        for (int d = 0; d < ndim_; ++d) {
            Float_ delta = point[d] - center[d];
            sqdist += delta * delta;
        }

        // Check whether we can use this node as a "summary"
        bool skip_children = node.is_leaf || (node.max_halfwidth < theta * std::sqrt(sqdist));

        Float_ result_sum = 0;
        if (skip_children) {
            // Compute and add t-SNE force between point and current node.
            const Float_ div = static_cast<Float_>(1) / (static_cast<Float_>(1) + sqdist);
            Float_ mult = count * div;
            result_sum += mult;
            mult *= div;

            for (int d = 0; d < ndim_; ++d) {
                neg_f[d] += mult * (point[d] - center[d]);
            }
        } else {
            // Recursively apply Barnes-Hut to children
            const auto& cur_children = node.children;
            for (int i = 0; i < Node::nchildren; ++i) {
                if (cur_children[i]) {
                    result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
                }
            }
        }

        return result_sum;
    }

public:
#ifndef NDEBUG
    // For testing purposes only.
    const auto& get_store() const {
        return my_store;
    }

    const auto& get_locations() const {
        return my_locations;
    }
#endif
};

}

}

#endif
