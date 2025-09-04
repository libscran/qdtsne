#ifndef QDTSNE_OPTIONS_HPP
#define QDTSNE_OPTIONS_HPP

/**
 * @file Options.hpp
 * @brief Options for the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @brief Options for `initialize()`.
 */
struct Options {
    /**
     * Perplexity value that determines how distances between neighbors are converted into conditional probabilities.
     * Increasing the perplexity will reduce the changes in probability with respect to distance (i.e., greater entropy).
     * In practice, this controls the balance between local and global structure in the final embedding.
     * Higher perplexities will focus on global structure at the cost of increased runtime and decreased local resolution.
     *
     * This option affects all `initialize()` functions except for the overload where precomputed neighbor search results are supplied _and_ `Options::infer_perplexity = true`.
     * In that scenarios, the perplexity is inferred from the number of neighbors in the precomputed search results and this setting is ignored.
     */
    double perplexity = 30;

    /**
     * Whether to infer the perplexity in the `initialize()` overload that accepts a `NeighborList` object.
     * If true, the value in `Options::perplexity` is ignored.
     * The perplexity is instead defined from the `NeighborList` as the number of nearest neighbors for the first observation divided by 3.
     * (It is assumed that all observations have the same number of neighbors.)
     * This effectively performs the inverse calculation of `perplexity_to_k()`.
     */
    bool infer_perplexity = true;

    /**
     * Approximation level for the Barnes-Hut calculation of repulsive forces.
     *
     * Consider the calculation of repulsive forces applied to an observation \f$i\f$.
     * `theta` is defined as the maximum \f$s/d\f$ at which a group of observations can be approximated by their center of mass,
     * where \f$s\f$ is the maximum width of the box containing all observations in the group (i.e., the longest side across all dimensions)
     * and \f$d\f$ is the distance from \f$i\f$ to that center of mass.
     * Lower values increase accuracy at the cost of increased compute time.
     */
    double theta = 1;

    /**
     * Maximum number of iterations to perform.
     * Larger values improve convergence of the algorithm at the cost of increased compute time.
     */
    int max_iterations = 1000;

    /**
     * Number of iterations of the early exaggeration phase, where the conditional probabilities are multiplied by `Options::exaggeration_factor`.
     * This increases the attractive forces between neighboring observations and encourages the formation of compact clusters of nearest neighbors.
     * As a result, the embedding will have a lot of empty space so that clusters can easily relocate to find a good global organization.
     * Larger values improve convergence within this phase at the cost of reducing the remaining iterations in `Options::max_iterations`.
     */
    int early_exaggeration_iterations = 250;

    /**
     * Exaggeration factor to scale the probabilities during the early exaggeration phase (see `Options::early_exaggeration_iterations`).
     * Larger values increase the attraction between nearest neighbors to favor local structure during this phase.
     */
    double exaggeration_factor = 12;

    /**
     * Number of iterations to perform before switching from the starting momentum to the final momentum.
     *
     * At each iteration, the update to each observation's position includes a small step in the direction of its previous update, i.e., some "momentum" is preserved.
     * Greater momentum can improve convergence by increasing the step size and smoothing over local oscillations, at the risk of potentially skipping over relevant minima.
     * The magnitude of this momentum switches from `Options::start_momentum` to `Options::final_momentum` at the specified number of iterations.
     */
    int momentum_switch_iterations = 250;

    /**
     * Starting momentum in \f$[0, 1)\$f, to be used in the iterations before the momentum switch at `Options::momentum_switch_iterations`.
     * This is usually lower than `Options::final_momentum` to avoid skipping over suitable local minima.
     */
    double start_momentum = 0.5;

    /**
     * Final momentum in \f$[0, 1)\$f, to be used in the iterations after the momentum switch at `Options::momentum_switch_iterations`.
     * This is usually higher than `Options::start_momentum` to accelerate convergence to the local minima once the observations are moderately well-organized.
     */
    double final_momentum = 0.8;

    /**
     * The learning rate, used to scale the updates to the coordinates at each iteration.
     * Larger values can speed up convergence at the cost of potentially skipping over local minima.
     */
    double eta = 200;

    /**
     * Maximum depth of the tree used in the Barnes-Hut approximation of the repulsive forces.
     * Setting `max_depth` to a smaller value (typically 7 - 10) improves speed by bounding the depth of recursion, at the cost of some accuracy.
     *
     * If neighboring observations cannot be separated before the maximum depth is reached, they will be assigned to the same leaf node.
     * This effectively approximates each observation's coordinates with the center of mass of its leaf node.
     * Repulsive forces can then be calculated once against the leaf's center of mass instead of each individual observation assigned to that leaf.
     *
     * The default is to use a large value, which means that the tree's depth is unbounded for most practical applications.
     * This aims to be consistent with the original implementation of the BH search,
     * but with some protection against near-duplicate observations that would otherwise result in unnecessary recursion.
     */
    int max_depth = 20;

    /**
     * Whether to replace a observation with the center of mass of its leaf node when computing the repulsive forces to all other observations.
     * This speeds up the calculation at the cost of some accuracy.
     *
     * By default, repulsive forces applied to an observation are computed separately for each observation,
     * even if the repulsive forces exerted by observation might be approximated by its assigned node's center of mass.
     * If the leaf approximation is used, the repulsive forces applied to each leaf node are used to approximate the respulsive forces applied to each observation in that leaf node.
     * This reduces compute time as it skips near-redundant calculations for neighboring observations assigned to the same leaf node.
     * The effectiveness of this option depends on `Options::max_depth`, which needs to be small enough so that many leaf nodes have multiple assigned observations.
     */
    bool leaf_approximation = false;

    /**
     * Number of threads to use.
     * The parallelization scheme is determined by `parallelize()` for most calculations.
     * The exception is the nearest-neighbor search in some of the `initialize()` overloads, where the scheme is determined by `knncolle::parallelize()` instead.
     */
    int num_threads = 1;
};

}

#endif
