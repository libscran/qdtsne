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
     * Perplexity value that determines the balance between local and global structure.
     * Higher perplexities will focus on global structure, at the cost of increased runtime and decreased local resolution.
     *
     * This option affects all `initialize()` methods except if precomputed neighbor search results are supplied _and_ `Options::infer_perplexity = true`.
     * In such cases, the perplexity is inferred from the number of neighbors in the supplied search results.
     */
    double perplexity = 30;

    /**
     * Whether to infer the perplexity in `initialize()` methods that accept a `NeighborList` object.
     * In such cases, the value in `Options::perplexity` is ignored.
     * The perplexity is instead defined from the `NeighborList` as the number of nearest neighbors for the first point divided by 3.
     * (It is assumed that all points have the same number of neighbors.)
     */
    bool infer_perplexity = true;

    /**
     * Amount of approximation to use in the Barnes-Hut calculation of repulsive forces.
     * This is defined as the maximum \f$s/d\f$ at which a group of points can be approximated by their center of mass,
     * where \f$s\f$ is the maximum width of the box containing all points in the group (i.e., the longest side across all dimensions)
     * and \f$d\f$ is the distance from a point to the center of mass.
     * Lower values increase accuracy at the cost of computational time.
     */
    double theta = 1;

    /**
     * Maximum number of iterations to perform.
     */
    int max_iterations = 1000;

    /**
     * Number of iterations to perform with exaggerated probabilities, as part of the early exaggeration phase.
     *
     * In the early exaggeration phase, the probabilities are multiplied by `Options::exaggeration_factor`.
     * This forces the algorithm to minimize the distances between neighbors, creating an embedding containing tight, well-separated clusters of neighboring cells.
     * Because there is so much empty space, these clusters have an opportunity to move around to find better global positions before the phase ends and they are forced to settle down.
     */
    int stop_lying_iter = 250;

    /**
     * Number of iterations to perform before switching from the starting momentum to the final momentum.
     *
     * The update to each point includes a small step in the direction of its previous update, i.e., there is some "momentum" from the previous update.
     * This aims to speed up the optimization and to avoid local minima by effectively smoothing the updates.
     * The starting momentum is usually smaller than the final momentum,
     * to give a chance for the points to improve their organization before encouraging iteration to a specific local minima.
     */
    int mom_switch_iter = 250;

    /**
     * Starting momentum, to be used in the early iterations before the momentum switch.
     */
    double start_momentum = 0.5;

    /**
     * Final momentum, to be used in the later iterations after the momentum switch.
     */
    double final_momentum = 0.8;

    /** 
     * The learning rate, used to scale the updates.
     * Larger values yield larger updates that speed up convergence to a local minima at the cost of stability.
     */
    double eta = 200;

    /** 
     * Factor to scale the probabilities during the early exaggeration phase (see `Options::stop_lying_iter`).
     */
    double exaggeration_factor = 12;

    /**
     * Maximum depth of the tree used in the Barnes-Hut approximation of the repulsive forces.
     * This effectively replaces each point with the center of mass of the most fine-grained partition at the leaves of the tree.
     * Setting this to a smaller value (e.g., 7 - 10) improves speed by bounding the depth of recursion, at the cost of some accuracy.
     *
     * The default is to use a large value, which means that the tree's depth is unbounded for most practical applications.
     * This aims to be consistent with the original implementation of the BH search,
     * but with some protection against near-duplicate points that would otherwise result in unnecessary recursion.
     */
    int max_depth = 20;

    /**
     * Whether to replace a point with the center of mass of its leaf node when computing the repulsive forces to all other points.
     * This allows the repulsive forces to be computed once per leaf node and then re-used across all points in that leaf node.
     * The effectiveness of this option depends on `Options::max_depth`, which needs to be small enough so that many leaf nodes have multiple assigned points.
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
