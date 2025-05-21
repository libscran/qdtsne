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
#include <cstddef>

#include "utils.hpp"

namespace qdtsne {

namespace internal {

// Assume that size_t is big enough to hold positional indices to all of the nodes.
typedef std::size_t SPTreeIndex;

template<std::size_t num_dim_, typename Float_>
class SPTree {
public:
    SPTree(SPTreeIndex npts, int maxdepth) : my_npts(npts), my_maxdepth(maxdepth), my_locations(my_npts) {
        my_store.reserve(std::min(static_cast<Float_>(my_npts), std::pow(static_cast<Float_>(4.0), static_cast<Float_>(my_maxdepth))) * 2);
        return;
    }

public:
    struct Node {
        Node(SPTreeIndex i, const Float_* point) : index(i) {
            std::copy_n(point, num_dim_, center_of_mass.data());
            fill();
            return;
        }

        Node() {
            std::fill(center_of_mass.begin(), center_of_mass.end(), 0);
            fill();
            return;
        }

    public:
        static constexpr int nchildren = (1 << num_dim_); 

        std::array<Float_, num_dim_> center_of_mass;

        // This should only be used when is_leaf = false, such that the values
        // have been set by set_child_boundaries().
        std::array<SPTreeIndex, nchildren> children;
        std::array<Float_, num_dim_> midpoint, halfwidth;
        Float_ max_width = 0;

        // This should only be used when is_leaf = true. In cases where multiple
        // points are assigned to the same leaf node (e.g., duplicates, max depth
        // truncation), it is the index of the first point for this Node.
        SPTreeIndex index = -1; 

        SPTreeIndex number = 1;
        bool is_leaf = true;

    private:
        void fill() {
            std::fill_n(midpoint.begin(), num_dim_, 0);
            std::fill_n(halfwidth.begin(), num_dim_, 0);
            std::fill_n(children.begin(), nchildren, 0);
        }
    };

private:
    const Float_ * my_data = NULL;
    SPTreeIndex my_npts;
    int my_maxdepth;
    std::vector<Node> my_store;

    // We need to store the on-tree locations for each point separately as each
    // point may not have a 1:1 mapping to a single Node if there are
    // duplicates or we truncate the tree at 'maxdepth'.
    std::vector<SPTreeIndex> my_locations;

    std::vector<SPTreeIndex> my_first_assignment;

    /****************************
     *** Construction methods ***
     ****************************/
public:
    void set(const Float_* Y) {
        my_data = Y;

        {
            my_store.clear();
            my_store.resize(1);
            my_store[0].is_leaf = false;
            my_store[0].number = my_npts;

            std::array<Float_, num_dim_> min_Y{}, max_Y{};
            std::fill_n(min_Y.begin(), num_dim_, std::numeric_limits<Float_>::max());
            std::fill_n(max_Y.begin(), num_dim_, std::numeric_limits<Float_>::lowest());

            // Setting the initial midpoint to the center of mass of all points
            // so that the partitioning effectively divides up the points.
            auto& mean_Y = my_store[0].midpoint;
            auto copy = Y;
            for (SPTreeIndex n = 0; n < my_npts; ++n) {
                for (std::size_t d = 0; d < num_dim_; ++d) {
                    auto curval = copy[d];
                    mean_Y[d] += curval;
                    min_Y[d] = std::min(min_Y[d], curval);
                    max_Y[d] = std::max(max_Y[d], curval);
                }
                copy += num_dim_;
            }

            for (std::size_t d = 0; d < num_dim_; ++d) {
                mean_Y[d] /= my_npts;
            }

            auto& halfwidth = my_store[0].halfwidth;
            for (std::size_t d = 0; d < num_dim_; ++d) {
                auto mean = mean_Y[d];
                halfwidth[d] = std::max(max_Y[d] - mean, mean - min_Y[d]) + static_cast<Float_>(1e-5);
            }
        }

        auto point = Y;
        my_first_assignment.resize(my_npts);
        for (SPTreeIndex i = 0; i < my_npts; ++i, point += num_dim_) {
            std::array<bool, num_dim_> side;
            SPTreeIndex parent = 0;

            for (int depth = 1; depth <= my_maxdepth; ++depth) {
                SPTreeIndex child_idx = find_child(parent, point, side.data());

                // Be careful with persistent references to my_store's contents,
                // as the vector may be reallocated when a push_back() occurs.
                SPTreeIndex current_loc = my_store[parent].children[child_idx];

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
                    for (std::size_t d = 0; d < num_dim_; ++d) {
                        nsame += (center[d] == point[d]);
                    }
                    if (nsame == num_dim_) {
                        ++(my_store[current_loc].number);
                        my_first_assignment[i] = my_store[current_loc].index;
                        break;
                    }

                    if (depth == my_maxdepth) {
                        my_first_assignment[i] = my_store[current_loc].index;
                    } else {
                        // Otherwise, we convert the current node into a non-leaf node to
                        // accommodate further recursion.
                        SPTreeIndex new_loc = my_store.size();

                        // Push the entire Node! Don't emplace_back() with the
                        // pointer for the child's center of mass, as any potential
                        // re-allocation of the vector would invalidate the pointer
                        // to the center of mass stored inside the vector.
                        my_store.push_back(my_store[current_loc]);

                        my_store[current_loc].is_leaf = false;
                        set_child_boundaries(parent, current_loc, side.data());

                        SPTreeIndex new_child_idx = find_child(current_loc, my_store[new_loc].center_of_mass.data(), side.data());
                        my_store[current_loc].children[new_child_idx] = new_loc;
                    }
                }

                // Online update of non-leaf node's cumulative size and center-of-mass.
                auto& node = my_store[current_loc];
                ++node.number;

                const Float_ cum_size = node.number;
                const Float_ mult1 = (cum_size - 1) / cum_size;

                for (std::size_t d = 0; d < num_dim_; ++d) {
                    node.center_of_mass[d] *= mult1;
                    node.center_of_mass[d] += point[d] / cum_size;
                }

                parent = current_loc;
            }
        }

        // Populating the on-tree locations for each node.
        my_locations.resize(my_npts);
        SPTreeIndex nnodes = my_store.size();
        for (SPTreeIndex n = 0; n < nnodes; ++n) {
            const auto& node = my_store[n];
            if (node.is_leaf) {
                my_locations[node.index] = n;
            }
        }

        for (SPTreeIndex i = 0; i < my_npts; ++i) {
            auto tmp = my_locations[my_first_assignment[i]]; // break it up to avoid unsequencing errors when assigning to self.
            my_locations[i] = tmp; // It is guaranteed that 'my_first_assignment[i] <= i', so there isn't any effect on future iterations of 'i'.
        } 

        return;
    }

private:
    SPTreeIndex find_child (SPTreeIndex parent, const Float_* point, bool * side) const {
        int multiplier = 1;
        SPTreeIndex child = 0;
        for (std::size_t d = 0; d < num_dim_; ++d) {
            side[d] = (point[d] >= my_store[parent].midpoint[d]);
            child += multiplier * side[d];
            multiplier *= 2;
        }
        return child;
    }

    void set_child_boundaries(SPTreeIndex parent, SPTreeIndex child, const bool* keep) {
        auto& current = my_store[child];
        auto& parental = my_store[parent];
        for (std::size_t d = 0; d < num_dim_; ++d) {
            current.halfwidth[d] = parental.halfwidth[d] / static_cast<Float_>(2);
            if (keep[d]) {
                current.midpoint[d] = parental.midpoint[d] + current.halfwidth[d];
            } else {
                current.midpoint[d] = parental.midpoint[d] - current.halfwidth[d];
            }
        }

        // Compute once for the theta calculations. Note that the max width is
        // twice the maximum current.halfwidth, so we just take the max of the
        // parental halfwidth to avoid loss of precision from an unnecessary
        // multiplication. I suspect that the original sptree.cpp code in Rtsne
        // was not correct as their 'width' was really a half width. For us it
        // doesn't matter as we double the default 'theta' to match their
        // default 'theta=0.5', so it all ends up being the same anyway.
        current.max_width = *std::max_element(parental.halfwidth.begin(), parental.halfwidth.end());
    }

    /***********************************
     *** Non-edge force calculations ***
     ***********************************/
private:
    static Float_ compute_sqdist(const Float_* point, const std::array<Float_, num_dim_>& center, std::array<Float_, num_dim_>& diffs) {
        Float_ sqdist = 0;
        for (std::size_t d = 0; d < num_dim_; ++d) {
            Float_ delta = point[d] - center[d]; // note that 'diffs' expects to hold 'point - center' on output, not the other way around.
            diffs[d] = delta;
            sqdist += delta * delta;
        }
        return sqdist;
    }

    static void add_non_edge_forces(const std::array<Float_, num_dim_>& diffs, Float_ sqdist, Float_ count, Float_& result_sum, Float_* neg_f) {
        const Float_ div = static_cast<Float_>(1) / (static_cast<Float_>(1) + sqdist);
        Float_ mult = count * div;
        result_sum += mult;
        mult *= div;
        for (std::size_t d = 0; d < num_dim_; ++d) {
            neg_f[d] += mult * diffs[d]; // remember, 'diff[d] := point[d] - center[d]' from compute_sqdist().
        }
    }

    static void remove_self_from_center(const Float_* point, const std::array<Float_, num_dim_>& center, Float_ count, std::array<Float_, num_dim_>& temp) {
        for (std::size_t d = 0; d < num_dim_; ++d) { 
            temp[d] = (center[d] * count - point[d]) / (count - 1);
        }
    }

public:
    Float_ compute_non_edge_forces(SPTreeIndex index, Float_ theta, Float_* neg_f) const {
        Float_ result_sum = 0;
        const Float_ * point = my_data + static_cast<std::size_t>(index) * num_dim_; // cast to avoid overflow.
        const auto& cur_children = my_store[0].children;
        std::fill_n(neg_f, num_dim_, 0);

        for (int i = 0; i < Node::nchildren; ++i) {
            if (cur_children[i]) {
                result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
            }
        }

        return result_sum;
    }

private:
    Float_ compute_non_edge_forces(SPTreeIndex index, const Float_* point, Float_ theta, Float_* neg_f, SPTreeIndex position) const {
        const auto& node = my_store[position];

        std::array<Float_, num_dim_> temp;
        auto center = &(node.center_of_mass);
        SPTreeIndex count = node.number;

        // Check if we're at the leaf node containing the 'index' point. We
        // skip it if the leaf only contains that point, otherwise we remove
        // the point from the center of mass for repulsive calculations.
        if (position == my_locations[index]) {
            if (count == 1) {
                return 0; 
            }
            remove_self_from_center(point, *center, count, temp);
            center = &temp;
            --count;
        }

        std::array<Float_, num_dim_> diffs;
        Float_ sqdist = compute_sqdist(point, *center, diffs);

        // Check whether we can use skip this node's children, either because
        // it's already a leaf or because we can use the BH approximation.
        bool skip_children = node.is_leaf || (node.max_width < theta * std::sqrt(sqdist));

        Float_ result_sum = 0;
        if (skip_children) {
            add_non_edge_forces(diffs, sqdist, count, result_sum, neg_f);
        } else {
            const auto& cur_children = node.children;
            for (int i = 0; i < Node::nchildren; ++i) {
                if (cur_children[i]) {
                    result_sum += compute_non_edge_forces(index, point, theta, neg_f, cur_children[i]);
                }
            }
        }

        return result_sum;
    }

    /*************************************************************
     *** Non-edge force calculations, using leaf approximation ***
     *************************************************************/
public:
    struct LeafApproxWorkspace {
        std::vector<SPTreeIndex> leaf_indices;
        std::vector<std::array<Float_, num_dim_> > leaf_neg_f;
        std::vector<Float_> leaf_sums;
    };

    void compute_non_edge_forces_for_leaves(Float_ theta, LeafApproxWorkspace& workspace, [[maybe_unused]] int num_threads) const {
        SPTreeIndex nnodes = my_store.size();
        workspace.leaf_neg_f.resize(nnodes);
        workspace.leaf_sums.resize(nnodes);

        auto process_leaf_node = [&](SPTreeIndex leaf) -> void {
            Float_ result_sum = 0;
            auto neg_f = workspace.leaf_neg_f[leaf].data();
            std::fill_n(neg_f, num_dim_, 0);

            const auto& cur_children = my_store[0].children;
            for (int i = 0; i < Node::nchildren; ++i) {
                if (cur_children[i] && cur_children[i] != leaf) {
                    result_sum += compute_non_edge_forces_for_leaves(leaf, theta, neg_f, cur_children[i]);
                }
            }

            workspace.leaf_sums[leaf] = result_sum;
        };

        if (num_threads == 1) {
            for (SPTreeIndex n = 0; n < nnodes; ++n) {
                if (my_store[n].is_leaf) {
                    process_leaf_node(n);
                }
            }

        } else {
            // Identifying the indices of leaf nods so that the processing is
            // more balanced between threads, otherwise the last thread is
            // going to have to process all the leaf nodes.
            workspace.leaf_indices.clear();
            workspace.leaf_indices.reserve(nnodes);
            for (SPTreeIndex n = 0; n < nnodes; ++n) {
                if (my_store[n].is_leaf) {
                    workspace.leaf_indices.push_back(n);
                }
            }

            SPTreeIndex nleaves = workspace.leaf_indices.size();
            parallelize(num_threads, nleaves, [&](int, SPTreeIndex start, SPTreeIndex length) -> void {
                for (SPTreeIndex n = start, end = start + length; n < end; ++n) {
                    process_leaf_node(workspace.leaf_indices[n]);
                }
            });
        }
    }

    Float_ compute_non_edge_forces_from_leaves(SPTreeIndex index, Float_* neg_f, const LeafApproxWorkspace& workspace) const {
        auto node_loc = my_locations[index];
        Float_ result_sum = workspace.leaf_sums[node_loc];
        const auto& leaf_neg_f = workspace.leaf_neg_f[node_loc];
        std::copy(leaf_neg_f.begin(), leaf_neg_f.end(), neg_f);

        const auto& node = my_store[node_loc];
        if (node.number != 1) {
            const Float_ * point = my_data + static_cast<std::size_t>(index) * num_dim_; // cast to avoid overflow.
            std::array<Float_, num_dim_> temp;
            remove_self_from_center(point, node.center_of_mass, node.number, temp);

            std::array<Float_, num_dim_> diffs;
            Float_ sqdist = compute_sqdist(point, temp, diffs);
            add_non_edge_forces(diffs, sqdist, node.number - 1, result_sum, neg_f);
        }

        return result_sum;
    }

private:
    Float_ compute_non_edge_forces_for_leaves(SPTreeIndex self_position, Float_ theta, Float_* neg_f, SPTreeIndex position) const {
        const auto& self_node = my_store[self_position];
        auto point = self_node.center_of_mass.data();

        const auto& node = my_store[position];
        std::array<Float_, num_dim_> diffs;
        Float_ sqdist = compute_sqdist(point, node.center_of_mass, diffs);

        bool skip_children = node.is_leaf || (node.max_width < theta * std::sqrt(sqdist));

        Float_ result_sum = 0;
        if (skip_children) {
            add_non_edge_forces(diffs, sqdist, node.number, result_sum, neg_f);
        } else {
            const auto& cur_children = node.children;
            for (int i = 0; i < Node::nchildren; ++i) {
                if (cur_children[i] && cur_children[i] != self_position) {
                    result_sum += compute_non_edge_forces_for_leaves(self_position, theta, neg_f, cur_children[i]);
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
