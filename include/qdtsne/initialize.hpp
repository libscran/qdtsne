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

#ifndef QDTSNE_TSNE_HPP
#define QDTSNE_TSNE_HPP

#include "Status.hpp"
#include "Options.hpp"

#include "knncolle/knncolle.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <limits>

/**
 * @file tsne.hpp
 *
 * @brief Initialize the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @cond
 */
namespace internal {

template<int ndim_, typename Index_, typename Float_>
Status<ndim_, Index_, Float_> initialize_internal(NeighborList<Index_, Float_> nn, Float_ perp, const Options& options) {
    compute_gaussian_perplexity(nn, perp, options.num_threads);
    symmetrize_matrix(nn);
    return Status<ndim_, Index_, Float_>(std::move(nn), options);
}

}
/**
 * @endcond
 */

/**
 * @brief Initialize the t-SNE algorithm.
 *
 * @tparam ndim_ Number of dimensions of the final embedding.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @param nn List of indices and distances to nearest neighbors for each observation. 
 * Each observation should have the same number of neighbors, sorted by increasing distance, which should not include itself.
 * @param options Further options.
 * If `Options::infer_perplexity = true`, the perplexity is determined from `nn` and the value in `Options::perplexity` is ignored.
 *
 * @return A `Status` object containing various precomputed structures required for iterations.
 */
template<int ndim_, typename Index_, Float_>
Status<ndim_, Index_, Float_> initialize(NeighborList<Index, Float> nn, const Options& options) {
    Float_ perp;
    if (infer_perplexity && nn.size()) {
        perp = static_cast<Float_>(nn.front().size())/3;
    } else {
        perp = perplexity;
    }
    return internal::initialize(std::move(nn), perp, options);
}

/**
 * @param prebuilt A `knncolle::Prebuilt` instance containing a neighbor search index built on the dataset of interest.
 *
 * @tparam ndim_ Number of dimensions of the final embedding.
 * @tparam Dim_ Integer type for the dataset dimensions.
 * @tparam Index_ Integer type for the neighbor indices.
 * @tparam Float_ Floating-point type to use for the calculations.
 *
 * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
 */
template<typename ndim_, typename Dim_, typename Index_, typename Float_>
Status<ndim_, Index_, Float_> initialize(const knncolle::Prebuilt<Dim_, Index_, Float_>& prebuilt, const Options& options) { 
    const Index_ K = perplexity_to_k(perplexity);
    Index_ N = prebuilt.num_observations();
    if (K >= N) {
        throw std::runtime_error("number of observations should be greater than 3 * perplexity");
    }

    NeighborList<Index_, Float> neighbors(N);

#ifndef QDTSNE_CUSTOM_PARALLEL
#ifdef _OPENMP
    #pragma omp parallel num_threads(iparams.nthreads)
#endif
    {
#else
    QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
#endif

        std::vector<Index_> indices;
        std::vector<Float_> distances;
        auto searcher = prebuilt.initialize();

#ifdef _OPENMP
        #pragma omp for 
#endif
        for (size_t i = 0; i < N; ++i) {
#else
        for (size_t i = first_; i < last_; ++i) {
#endif

            searcher->find_nearest_neighbors(i, K, &indices, &distances);
            size_t actual_k = indices.size();
            for (size_t x = 0; x < actual_k; ++x) {
                neighbors[i].emplace_back(indices[x], distances[x]);
            }

#ifndef QDTSNE_CUSTOM_PARALLEL
    }
#else
    }
    }, iparams.nthreads);
#endif

    return internal::initialize(std::move(neighbors), perplexity, options);
}

/**
 * @tparam Input Floating point type for the input data.
 * 
 * @param[in] input Pointer to a 2D array containing the input high-dimensional data, with number of rows and columns equal to `D` and `N`, respectively.
 * The array is treated as column-major where each row corresponds to a dimension and each column corresponds to an observation.
 * @param D Number of input dimensions.
 * @param N Number of observations.
 *
 * @return A `Status` object containing various pre-computed structures required for the iterations in `run()`.
 *
 * This differs from the other `run()` methods in that it will internally compute the nearest neighbors for each observation.
 * As with the original t-SNE implementation, it will use vantage point trees for the search.
 * See the other `initialize()` methods to specify a custom search algorithm.
 */
template<typename Input = Float>
auto initialize(const Input* input, size_t D, size_t N, const knncolle::Builder<knncolle::SimpleMatrixt sta<) { 
    knncollek::Vptree<> searcher(D, N, input); 
    return initialize(&searcher);
}

}

#endif
