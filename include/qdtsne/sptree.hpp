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

    std::array<double, ndim> corner, width;
    std::array<double, ndim> center_of_mass;
    std::array<size_t, nchildren> children;
    int depth = 0;
    bool is_leaf = true;
    int number = 0;

    SPTreeNode() {
        std::fill_n(corner.begin(), ndim, 0);
        std::fill_n(width.begin(), ndim, 0);
        std::fill_n(center_of_mass.begin(), ndim, 0);
        std::fill_n(children.begin(), ndim, 0);
        return;
    }
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

private:
    void set(const double* Y) {
        store.resize(1);
        store[0].is_leaf = false;
        {
            std::array<double, ndim> min_Y{}, max_Y{};
            std::fill_n(min_Y.begin(), ndim, std::numeric_limits<double>::lowest());
            std::fill_n(max_Y.begin(), ndim, std::numeric_limits<double>::max());

            auto& mean_Y = store[0].corner;
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

            auto& width = store[0].width;
            for (int d = 0; d < ndim; ++d) {
                width[d] = std::max(max_Y[d] - mean_Y[d], mean_Y[d] - min_Y[d]) + 1e-5;
            }
        }

        auto point = Y;
        for (size_t i = 0; i < N; ++i, copy += ndim) {
            size_t position = 0, depth = 0;
            do {
                std::array<bool, ndim> side;
                size_t child = find_child(point, side.data());
                ++depth;

                // Be careful with persistent references to store's contents,
                // as the vector may be reallocated when a push_back() occurs.
                if (store[position].children[child] == 0) { 
                    size_t newloc = store.size();
                    store[position].children[child] = newloc;
                    store.resize(newloc + 1);

                    set_child_boundaries(position, child, side.data());
                    auto& latest = store.back();
                    latest.number = 1;
                    latest.depth = depth;
                    break;
                } 

                if (store[child].is_leaf) {
                    size_t newloc = store.size();
                    store.push_back(store[child]); // making a copy, so we copy the center of mass for free.
                    auto& latest = store.back();

                    std::array<bool, ndim> side2; 
                    size_t child2 = find_child(latest.center_of_mass.data(), side2.data());
                    store[child].children[child2] = newloc;
                    store[child].is_leaf = false;

                    set_child_boundaries(child, child2, side2.data());
                    ++latest.depth;
                }

                // Online update of cumulative size and center-of-mass
                ++node.number;
                const double cum_size = node.number;
                const double mult1 = (cum_size - 1) / cum_size;

                for (int d = 0; d < ndim; ++d) {
                    node.center_of_mass[d] *= mult1;
                    node.center_of_mass[d] += point[d] / cum_size;
                }
            } while (depth <= maxdepth);

        }
    }

private:
    size_t find_child (size_t parent, const double* point, bool * side) {
        int multiplier = 1;
        size_t child = 0;
        for (int c = 0; c < ndim; ++c) {
            side[c] = (point[c] >= store[parent].corner[c]);
            child += multiplier * side[c];
            multiplier *= 2;
        }
        return child;
    }

    void set_child_boundaries(size_t parent, size_t child, const bool* keep) const {
        auto& current = store[child];
        auto& parental = store[parent];
        for (int c = 0; c < ndim; ++c) {
            current.width[c] = parental.width[c] / 2.0;
            if (keep[c]) {
                current.corner[c] = parental.corner[c] + current.width[c];
            } else {
                current.corner[c] = parental.corner[c] - current.width[c];
            }
        }
        return;
    }

    void update_node(SPTreeNode& node, const double* point) const {
        return;
    }

      
      // Otherwise, we need to subdivide the current cell
      if(is_leaf) subdivide();
      
      // Find out where the point can be inserted
      for(unsigned int i = 0; i < no_children; i++) {
        if(children[i]->insert(new_index)) return true;
      }
      
      // Otherwise, the point cannot be inserted (this should never happen)
      return false;
    }


    // Create four children which fully divide this cell into four quads of equal area
    template<int NDims>
    void SPTree<NDims>::subdivide() {
      
      // Create new children
      double new_corner[NDims];
      double new_width[NDims];
      for(unsigned int i = 0; i < no_children; i++) {
        unsigned int div = 1;
        for(unsigned int d = 0; d < NDims; d++) {
          new_width[d] = .5 * boundary.getWidth(d);
          if((i / div) % 2 == 1) new_corner[d] = boundary.getCorner(d) - .5 * boundary.getWidth(d);
          else                   new_corner[d] = boundary.getCorner(d) + .5 * boundary.getWidth(d);
          div *= 2;
        }
        children[i] = new SPTree(this, data, new_corner, new_width);
      }
      
      // Move existing points to correct children
      for(unsigned int i = 0; i < size; i++) {
        bool success = false;
        for(unsigned int j = 0; j < no_children; j++) {
          if(!success) success = children[j]->insert(index[i]);
        }
        index[i] = -1;
      }
      
      // Empty parent node
      size = 0;
      is_leaf = false;
    }

public:
};

// Compute non-edge forces using Barnes-Hut algorithm
template<int NDims>
double SPTree<NDims>::computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[]) const
{
  double resultSum = 0;
  double buff[NDims];  // make buff local for parallelization
  
  // Make sure that we spend no time on empty nodes or self-interactions
  if(cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return resultSum;
  
  // Compute distance between point and center-of-mass
  double sqdist = .0;
  unsigned int ind = point_index * NDims;
  
  for(unsigned int d = 0; d < NDims; d++) {
    buff[d] = data[ind + d] - center_of_mass[d];
    sqdist += buff[d] * buff[d];
  }
  
  // Check whether we can use this node as a "summary"
  double max_width = 0.0;
  double cur_width;
  for(unsigned int d = 0; d < NDims; d++) {
    cur_width = boundary.getWidth(d);
    max_width = (max_width > cur_width) ? max_width : cur_width;
  }
  if(is_leaf || max_width / sqrt(sqdist) < theta) {
    
    // Compute and add t-SNE force between point and current node
    sqdist = 1.0 / (1.0 + sqdist);
    double mult = cum_size * sqdist;
    resultSum += mult;
    mult *= sqdist;
    for(unsigned int d = 0; d < NDims; d++) neg_f[d] += mult * buff[d];
  }
  else {
    
    // Recursively apply Barnes-Hut to children
    for(unsigned int i = 0; i < no_children; i++){
      resultSum += children[i]->computeNonEdgeForces(point_index, theta, neg_f);
    }
  }
  return resultSum;
}


// Computes edge forces
template<int NDims>
void SPTree<NDims>::computeEdgeForces(unsigned int* row_P, unsigned int* col_P, double* val_P, unsigned int N, double* pos_f, int num_threads) const
{
  
  // Loop over all edges in the graph
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for(unsigned int n = 0; n < N; n++) {
    unsigned int ind1 = n * NDims;
    for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {
      
      double buff[NDims]; // make buff local for parallelization
      
      // Compute pairwise distance and Q-value
      double sqdist = 1.0;
      unsigned int ind2 = col_P[i] * NDims;
      
      for(unsigned int d = 0; d < NDims; d++) {
        buff[d] = data[ind1 + d] - data[ind2 + d];
        sqdist += buff[d] * buff[d];
      }
      
      sqdist = val_P[i] / sqdist;
      
      // Sum positive force
      for(unsigned int d = 0; d < NDims; d++) pos_f[ind1 + d] += sqdist * buff[d];
    }
  }
}

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
