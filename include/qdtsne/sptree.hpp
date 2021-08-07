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

namespace qdtsne {

template <int ndim>
struct SPTreeNode {
    static constexpr int nchildren = (1 << ndim); 

    SPTreeNode(const double* point, int d) : depth(d) {
        std::copy_n(point, ndim, center_of_mass.data());
        fill();
        return;
    }

    SPTreeNode() {
        std::fill_n(center_of_mass.data(), ndim, 0);
        fill();
        return;
    }

    void fill() {
        std::fill_n(midpoint.begin(), ndim, 0);
        std::fill_n(halfwidth.begin(), ndim, 0);
        std::fill_n(children.begin(), nchildren, 0);
    }

    std::array<double, ndim> midpoint, halfwidth;
    std::array<double, ndim> center_of_mass;
    std::array<size_t, nchildren> children;
    int number = 1;
    int depth = 0;
    bool is_leaf = true;
};

template<int ndim, int maxdepth>
class SPTree {
public:
    SPTree(size_t n) : N(n) {
        store.reserve(std::min(static_cast<double>(n), std::pow(4.0, static_cast<double>(maxdepth))) * 2);
        return;
    }

private:
    size_t N;
    std::vector<SPTreeNode<ndim> > store;

public:
    void set(const double* Y) {
        store.resize(1);
        store[0].is_leaf = false;
        {
            std::array<double, ndim> min_Y{}, max_Y{};
            std::fill_n(min_Y.begin(), ndim, std::numeric_limits<double>::lowest());
            std::fill_n(max_Y.begin(), ndim, std::numeric_limits<double>::max());

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
                halfwidth[d] = std::max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
            }
        }

        auto point = Y;
        for (size_t i = 0; i < N; ++i, point += ndim) {
            std::array<bool, ndim> side;
            size_t parent = 0;

            for (int depth = 1; depth <= maxdepth; ++depth) {
                size_t child_idx = find_child(parent, point, side.data());
                size_t child_loc = store[parent].children[child_idx];

                // Be careful with persistent references to store's contents,
                // as the vector may be reallocated when a push_back() occurs.
                if (child_loc == 0) { 
                    child_loc = store.size();
                    store[parent].children[child_idx] = child_loc;
                    store.push_back(SPTreeNode<ndim>(point, depth));
                    set_child_boundaries(parent, child_loc, side.data());
                    break;
                } 

                if (store[child_loc].is_leaf && depth < maxdepth) {
                    // Shifting the current child to become a child of itself.
                    size_t grandchild_loc = store.size();
                    store.push_back(SPTreeNode<ndim>(store[child_loc].center_of_mass.data(), depth + 1)); 

                    std::array<bool, ndim> side2; 
                    size_t grandchild_idx = find_child(child_loc, store[grandchild_loc].center_of_mass.data(), side2.data());
                    set_child_boundaries(child_loc, grandchild_loc, side2.data());

                    store[child_loc].children[grandchild_idx] = grandchild_loc;
                    store[child_loc].is_leaf = false;
                }

                // Online update of cumulative size and center-of-mass
                auto& node = store[child_loc];
                ++node.number;
                const double cum_size = node.number;
                const double mult1 = (cum_size - 1) / cum_size;

                for (int d = 0; d < ndim; ++d) {
                    node.center_of_mass[d] *= mult1;
                    node.center_of_mass[d] += point[d] / cum_size;
                }

                parent = child_loc;
            }
        }

        return;
    }

private:
    size_t find_child (size_t parent, const double* point, bool * side) const {
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
            current.halfwidth[c] = parental.halfwidth[c] / 2.0;
            if (keep[c]) {
                current.midpoint[c] = parental.midpoint[c] + current.halfwidth[c];
            } else {
                current.midpoint[c] = parental.midpoint[c] - current.halfwidth[c];
            }
        }
        return;
    }

public:
    double compute_non_edge_forces(const double* point, double theta, double* neg_f, size_t position = 0) const {
        // Compute squared distance between point and center-of-mass
        double sqdist = 0;
        const auto& node = store[position];
        for (int d = 0; d < ndim; ++d) {
            sqdist += (point[d] - node.center_of_mass[d]) * (point[d] - node.center_of_mass[d]);
        }
      
        // Check whether we can use this node as a "summary"
        bool skip_children = node.is_leaf;
        if (!skip_children) {
            double max_halfwidth = *std::max_element(node.width.begin(), node.width.end());
            skip_children = (max_halfwidth < theta * std::sqrt(sqdist));
        }

        double result_sum = 0;
        if (skip_children) {
            // Compute and add t-SNE force between point and current node
            sqdist = 1.0 / (1.0 + sqdist);
            double mult = node.number * sqdist;
            result_sum += mult;
            mult *= sqdist;
            for (int d = 0; d < ndim; d++) {
                neg_f[d] += mult * (point[d] - node.center_of_mass[d]);
            }
        } else {
            // Recursively apply Barnes-Hut to children
            for (int i = 0; i < SPTreeNode<ndim>::nchildren; ++i) {
                result_sum += compute_non_edge_forces(point, theta, neg_f, node.children[i]);
            }
        }

        return result_sum;
    }
};

//// Print out tree
//template<int NDims>
//void SPTree<NDims>::print()
//{
//  if(cum_size == 0) {
//    Rprintf("Empty node\n");
//    return;
//  }
//
//  if(is_leaf) {
//    Rprintf("Leaf node; data = [");
//    for(unsigned int i = 0; i < size; i++) {
//      double* point = data + index[i] * NDims;
//      for(int d = 0; d < NDims; d++) Rprintf("%f, ", point[d]);
//      Rprintf(" (index = %d)", index[i]);
//      if(i < size - 1) Rprintf("\n");
//      else Rprintf("]\n");
//    }
//  }
//  else {
//    Rprintf("Intersection node with center-of-mass = [");
//    for(int d = 0; d < NDims; d++) Rprintf("%f, ", center_of_mass[d]);
//    Rprintf("]; children are:\n");
//    for(int i = 0; i < no_children; i++) children[i]->print();
//  }
//}

}

#endif
