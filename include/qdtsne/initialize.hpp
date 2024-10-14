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

#ifndef QDTSNE_INITIALIZE_HPP
#define QDTSNE_INITIALIZE_HPP

#include "knncolle/knncolle.hpp"

#include <vector>
#include <stdexcept>

#include "Status.hpp"
#include "Options.hpp"
#include "gaussian.hpp"
#include "symmetrize.hpp"

/**
 * @file initialize.hpp
 * @brief Initialize the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @cond
 */
namespace internal {

template<int num_dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(NeighborList<Index_, Float_> nn, Float_ perp, const Options& options) {
    compute_gaussian_perplexity(nn, perp, options.num_threads);
    symmetrize_matrix(nn);
    return Status<num_dim_, Index_, Float_>(std::move(nn), options);
}

}
/**
 * @endcond
 */

/**
 * Initialize the data structures for t-SNE algorithm, given the nearest neighbors of each observation.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param neighbors List of indices and distances to nearest neighbors for each observation. 
 * Each observation should have the same number of neighbors, sorted by increasing distance, which should not include itself.
 * @param options Further options.
 * If `Options::infer_perplexity = true`, the perplexity is determined from `neighbors` and the value in `Options::perplexity` is ignored.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<int num_dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(NeighborList<Index_, Float_> neighbors, const Options& options) {
    Float_ perp;
    if (options.infer_perplexity && neighbors.size()) {
        perp = static_cast<Float_>(neighbors.front().size())/3;
    } else {
        perp = options.perplexity;
    }
    return internal::initialize<num_dim_>(std::move(neighbors), perp, options);
}

/**
 * Overload that accepts a neighbor search index and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Dim_ Integer type for the dataset dimensions.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param prebuilt A `knncolle::Prebuilt` instance containing a neighbor search index built on the dataset of interest.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<int num_dim_, typename Dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(const knncolle::Prebuilt<Dim_, Index_, Float_>& prebuilt, const Options& options) { 
    const Index_ K = perplexity_to_k(options.perplexity);
    Index_ N = prebuilt.num_observations();
    if (K >= N) {
        throw std::runtime_error("number of observations should be greater than 3 * perplexity");
    }

    auto neighbors = find_nearest_neighbors(prebuilt, K, options.num_threads);
    return internal::initialize<num_dim_>(std::move(neighbors), options.perplexity, options);
}

/**
 * Overload that accepts a column-major matrix of coordinates and computes the nearest neighbors for each observation,
 * before proceeding with the initialization of the t-SNE algorithm.
 *
 * @tparam num_dim_ Number of dimensions of the final embedding.
 * @tparam Dim_ Integer type for the dataset dimensions.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param data_dim Number of rows of the matrix at `data`, corresponding to the dimensions of the input dataset.
 * @param num_points Number of columns of the matrix at `data`, corresponding to the points of the input dataset.
 * @param[in] data Pointer to an array containing a column-major matrix with `data_dim` rows and `num_points` columns.
 * @param builder A `knncolle::Builder` instance specifying the nearest-neighbor algorithm to use.
 * @param options Further options.
 *
 * @return A `Status` object representing an initial state of the t-SNE algorithm.
 */
template<int num_dim_, typename Dim_, typename Index_, typename Float_>
Status<num_dim_, Index_, Float_> initialize(
    Dim_ data_dim,
    Index_ num_points,
    const Float_* data,
    const knncolle::Builder<knncolle::SimpleMatrix<Dim_, Index_, Float_>, Float_>& builder,
    const Options& options) 
{
    auto index = builder.build_unique(knncolle::SimpleMatrix<Dim_, Index_, Float_>(data_dim, num_points, data));
    return initialize<num_dim_>(*index, options);
}

}

#endif
