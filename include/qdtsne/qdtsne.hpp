#ifndef QDTSNE_QDTSNE_HPP
#define QDTSNE_QDTSNE_HPP

/**
 * @file qdtsne.hpp
 *
 * @brief Umbrella header for all **qdtsne** files.
 */

#include "tsne.hpp"
#include "utils.hpp"

/**
 * @namespace qdtsne
 * @brief Quick and dirty t-SNE functions.
 *
 * The t-distributed stochastic neighbor embedding (t-SNE) algorithm is a non-linear dimensionality reduction technique for visualizing high-dimensional datasets.
 * It places each observation in a low-dimensional map (usually 2D) in a manner that preserves the identity of its neighbors in the original space, thus preserving the local structure of the dataset.
 * This is achieved by converting the distances between neighbors in high-dimensional space to probabilities via a Gaussian kernel;
 * creating a low-dimensional representation where the distances between neighbors can be converted to similar probabilities (in this case, with a t-distribution);
 * and then iterating such that the Kullback-Leiber divergence between the two probability distributions is minimized.
 * In practice, this involves balancing the attractive forces between neighbors and repulsive forces between all points.
 *
 * @see
 * van der Maaten, L.J.P. and Hinton, G.E. (2008). 
 * Visualizing high-dimensional data using t-SNE. 
 * _Journal of Machine Learning Research_, 9, 2579-2605.
 *
 * @see 
 * van der Maaten, L.J.P. (2014). 
 * Accelerating t-SNE using tree-based algorithms. 
 * _Journal of Machine Learning Research_, 15, 3221-3245.
 */
namespace qdtsne {}

#endif
