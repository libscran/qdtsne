#ifndef QDTSNE_OPTIONS_HPP
#define QDTSNE_OPTIONS_HPP

/**
 * @file Options.hpp
 *
 * @brief Options for the t-SNE algorithm.
 */

namespace qdtsne {

/**
 * @param Options for `initialize()`.
 */
struct Options {
    /**
     * Perplexity value that determines the balance between local and global structure.
     * Higher perplexities will focus on global structure, at the cost of increased runtime and decreased local resolution.
     *
     * This option affects all methods except if precomputed neighbor search results are supplied _and_ `set_infer_perplexity()` is `true`.
     * In such cases, the perplexity is inferred from the number of neighbors per observation in the supplied search results.
     */
    double perplexity = 30;

    /**
     * Whether to infer the perplexity in `initialize()` methods that accept a `NeighborList` object.
     * In such cases, the value in `Options::perplexity` is ignored.
     * The perplexity is instead defined from the `NeighborList` as the number of nearest neighbors per observation divided by 3.
     */
    bool infer_perplexity = true;

    /**
     * Level of the approximation to use in the Barnes-Hut tree calculation of repulsive forces.
     * Lower values increase accuracy at the cost of computational time.
     */
    double theta = 0.5;

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
     * The idea here is that the update to each point includes a small step in the direction of its previous update, i.e., there is some "momentum" from the previous update.
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
     * @param m Maximum depth of the Barnes-Hut tree.
     * Larger values improve the quality of the approximation for the repulsive force calculation, at the cost of computational time.
     * A value of 7 is a good compromise for most applications.
     *
     * The default is to use a large value, which means that the tree's depth is unbounded for most practical applications.
     * This aims to be consistent with the original implementation of the BH search,
     * but with some protection against duplicate points that would otherwise result in infinite recursion during tree construction.
     * If users are confident that their data contains no duplicates, they can set the depth to arbitrarily large values.
     */
    int max_depth = 20;

    /**
     * Number of threads to use.
     */
    int num_threads = 1;
};

}

#endif
