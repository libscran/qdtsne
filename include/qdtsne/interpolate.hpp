#ifndef QDTSNE_INTERPOLATE_HPP
#define QDTSNE_INTERPOLATE_HPP

#include "utils.hpp"
#include "sptree.hpp"

#include <unordered_map>
#include <array>
#include <algorithm>
#include <vector>

namespace qdtsne {

template<int ndim = 2, typename Float = double>
struct Interpolator {
public:
    typedef std::array<Float, ndim> Coords;

    static constexpr int ndim_p1 = ndim + 1;

    static constexpr int ncorners = (1 << ndim);

public:
    static std::array<size_t, ndim> encode(const Float* data, const Coords& mins, const Coords& step, int intervals) {
        std::array<size_t, ndim> current;
        size_t limit = intervals - 1;
        for (int d = 0; d < ndim; ++d, ++data) {
            current[d] = std::min(static_cast<size_t>((*data - mins[d]) / step[d]), limit); // set upper bound to catch the value at the max.
        }
        return current;
    }

    static size_t hash(const std::array<size_t, ndim>& index, int intervals) {
        size_t counter = 0;
        for (int d = 0; d < ndim; ++d) {
            counter *= intervals + 1; // need to +1 as we can get indices == intervals when populating corners.
            counter += index[d];
        }
        return counter;
    }

    static Coords unhash(size_t hash, const Coords& mins, const Coords& step, int intervals) {
        Coords current;
        for (int d = 0; d < ndim; ++d) {
            int d0 = ndim - d - 1;
            current[d0] = (hash % (intervals + 1)) * step[d0] + mins[d0];
            hash /= intervals + 1;
        }
        return current;
    }

    static std::array<size_t, ndim> unhash(size_t hash, int intervals) {
        std::array<size_t, ndim> current;
        for (int d = 0; d < ndim; ++d) {
            current[ndim - d - 1] = (hash % (intervals + 1));
            hash /= intervals + 1;
        }
        return current;
    }

public:
    template<int d = 0>
    static void populate_corners(std::unordered_map<size_t, size_t>& collected, std::array<size_t, ndim> current, int intervals, int shifted = 0) {
        if constexpr(d == ndim - 1) {
            auto fill = [&](const std::array<size_t, ndim>& x) -> void {
                size_t h = hash(x, intervals);
                auto it = collected.find(h);
                if (it == collected.end()) {
                    collected[h] = -1;
                }
            };

            // Must have at least one hit.
            if (shifted) {
                fill(current);
                ++current[d];
                fill(current);
            } else {
                ++current[d];
                fill(current);
            }
        } else {
            populate_corners<d + 1>(collected, current, intervals, shifted);
            ++current[d];
            populate_corners<d + 1>(collected, current, intervals, shifted + 1);
        }
    }

public:
    Interpolator& set_num_threads(int n) {
        nthreads = n;
        return *this;
    }

    int nthreads = 1;

public:
    void compute_waypoint_non_edge_forces( 
        const SPTree<ndim, Float>& tree, 
        const Coords& mins,
        const Coords& step,
        std::unordered_map<size_t, size_t>& waypoints,
        std::unordered_map<size_t, size_t>& has_zero,
        Float theta,
        std::vector<Float>& collected,
        int intervals
    ) const {
        size_t i = 0;
        
#if defined(_OPENMP) || defined(QDTSNE_CUSTOM_PARALLEL)
        if (nthreads > 1) {
            std::vector<size_t> indices(waypoints.size());
            for (auto wIt = waypoints.begin(); wIt != waypoints.end(); ++wIt, ++i) {
                auto h = wIt->first;
                if (wIt->second == 0) {
                    has_zero[h] = has_zero.size();
                }
                wIt->second = i;
                indices[i] = h;
            }

#ifndef QDTSNE_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (size_t i = 0; i < indices.size(); ++i) {
#else
            QDTSNE_CUSTOM_PARALLEL(indices.size(), [&](size_t first_, size_t last_) -> void {
            for (size_t i = first_; i < last_; ++i) {
#endif
                auto h = indices[i];
                auto current = unhash(h, mins, step, intervals);
                Float* curcollected = collected.data() + ndim_p1 * i;
                curcollected[ndim] = tree.compute_non_edge_forces(current.data(), theta, curcollected);

#ifndef QDTSNE_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads); 
#endif

            return;
        }
#endif

        for (auto wIt = waypoints.begin(); wIt != waypoints.end(); ++wIt, ++i) {
            auto h = wIt->first;
            if (wIt->second == 0) {
                has_zero[h] = has_zero.size();
            }
            wIt->second = i;

            auto current = unhash(h, mins, step, intervals);
            Float* curcollected = collected.data() + ndim_p1 * i;
            curcollected[ndim] = tree.compute_non_edge_forces(current.data(), theta, curcollected);
        }

        return;
    }

    static Float interpolate_non_edge_forces(
        const Float* position,
        const Coords& mins,
        const Coords& step,
        const std::unordered_map<size_t, size_t>& has_zero,
        const std::vector<Float>& interpolants,
        size_t blocksize,
        Float* neg,
        int intervals)
    {
        auto current = encode(position, mins, step, intervals);
        std::array<Float, ndim> delta;
        for (int d = 0; d < ndim; ++d, ++position) {
            delta[d] = *position - (current[d] * step[d] + mins[d]);
        }

        size_t h = hash(current, intervals);
        size_t counter = has_zero.find(h)->second; // this had better not miss!
        Float current_sum = 0;

        for (int d = 0; d <= ndim; ++d) {
            size_t offset = counter * blocksize + d * ncorners;
            Float slope = interpolants[offset] * delta[1] + interpolants[offset + 1];
            Float intercept = interpolants[offset + 2] * delta[1] + interpolants[offset + 3];
            auto& output = (d == ndim ? current_sum : neg[d]);
            output = slope * delta[0] + intercept;
        }

        return current_sum;
    }

    Float sum_interpolated_non_edge_forces(
        size_t N,
        const Float* Y,
        const Coords& mins,
        const Coords& step,
        const std::unordered_map<size_t, size_t>& has_zero,
        const std::vector<Float>& interpolants,
        size_t blocksize,
        Float* neg,
        int intervals,
        std::vector<Float>& parallel_buffer
    ) const {

#if defined(_OPENMP) || defined(QDTSNE_CUSTOM_PARALLEL)
        if (nthreads > 1) {

#ifndef QDTSNE_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (size_t i = 0; i < N; ++i) {
#else
            QDTSNE_CUSTOM_PARALLEL(N, [&](size_t first_, size_t last_) -> void {
            for (size_t i = first_; i < last_; ++i) {
#endif        

                auto offset = i * ndim;
                parallel_buffer[i] = interpolate_non_edge_forces(
                    Y + offset,
                    mins, 
                    step, 
                    has_zero, 
                    interpolants, 
                    blocksize, 
                    neg + offset,
                    intervals
                );

#ifndef QDTSNE_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif

            return std::accumulate(parallel_buffer.begin(), parallel_buffer.end(), static_cast<Float>(0));
        }
#endif

        Float output_sum = 0;

        for (size_t i = 0; i < N; ++i) {
            auto offset = i * ndim;
            output_sum += interpolate_non_edge_forces(
                Y + offset,
                mins, 
                step, 
                has_zero, 
                interpolants, 
                blocksize, 
                neg + offset,
                intervals
            );
        }

        return output_sum;
    }

public:
    Float compute_non_edge_forces(
        const SPTree<ndim, Float>& tree, 
        size_t N, 
        const Float* Y, 
        Float theta, 
        Float* neg, 
        int intervals,
        std::vector<Float>& parallel_buffer
    ) const {
        // Get the limits of the existing coordinates.
        Coords mins, maxs;
        std::fill_n(mins.begin(), ndim, std::numeric_limits<Float>::max());
        std::fill_n(maxs.begin(), ndim, std::numeric_limits<Float>::lowest());
        {
            const auto* copy = Y;
            for (size_t i = 0; i < N; ++i) {
                for (int d = 0; d < ndim; ++d, ++copy) {
                    mins[d] = std::min(mins[d], *copy);
                    maxs[d] = std::max(maxs[d], *copy);
                }
            }
        }

        Coords step;
        for (int d = 0; d < ndim; ++d) {
            step[d] = (maxs[d] - mins[d]) / intervals;
            if (step[d] == 0) {
                step[d] = 1e-8; 
            }
        }

        // First pass to identify all occupied waypoints.
        std::unordered_map<size_t, size_t> waypoints;
        for (size_t i = 0; i < N; ++i) {
            auto current = encode(Y + i * ndim, mins, step, intervals); 

            size_t counter = hash(current, intervals);
            auto wIt = waypoints.find(counter);
            bool redo = false;
            if (wIt == waypoints.end()) {
                waypoints[counter] = 0;
                redo = true;
            } else if (wIt->second > 0) {
                wIt->second = 0;
                redo = true;
            }

            if (redo) {
                populate_corners(waypoints, current, intervals);
            }
        }

        // Second pass to compute forces for the waypoints. 
        std::unordered_map<size_t, size_t> has_zero;
        has_zero.reserve(waypoints.size());
        constexpr int ndim_p1 = ndim + 1;
        std::vector<Float> collected(ndim_p1 * waypoints.size());

        compute_waypoint_non_edge_forces( 
            tree, 
            mins,
            step,
            waypoints,
            has_zero,
            theta,
            collected,
            intervals
        );

        // Third pass to precompute the interpolating structures. 
        if constexpr(ndim != 2) {
            throw std::runtime_error("interpolation is not yet supported for ndim != 2");
        }
        constexpr int ncorners = (1 << ndim);
        size_t blocksize = ncorners * ndim_p1;
        std::vector<Float> interpolants(blocksize * has_zero.size());

        for (const auto& y : has_zero) {
            auto current = unhash(y.first, intervals);

            // Finding the other points in the same box, by traversing the corners.
            std::array<size_t, ncorners> others; 
            others[0] = waypoints[hash(current, intervals)];
            ++current[0];
            others[1] = waypoints[hash(current, intervals)];
            ++current[1];
            others[3] = waypoints[hash(current, intervals)];
            --current[0];
            others[2] = waypoints[hash(current, intervals)];

            // Computing the slopes and intercepts.
            for (int d = 0; d <= ndim; ++d) {
                std::array<Float, ncorners> obs;
                for (size_t o = 0; o < others.size(); ++o) {
                    obs[o] = collected[ndim_p1 * others[o] + d];
                }

                Float slope0 = (obs[1] - obs[0]) / step[0];
                Float intercept0 = obs[0];
                Float slope1 = (obs[3] - obs[2]) / step[0];
                Float intercept1 = obs[2];

                size_t offset = y.second * blocksize + d * (1 << ndim);
                interpolants[offset + 0] = (slope1 - slope0) / step[1]; // slope of the slope.
                interpolants[offset + 1] = slope0; // intercept of the slope.
                interpolants[offset + 2] = (intercept1 - intercept0) / step[1]; // slope of the intercept.
                interpolants[offset + 3] = intercept0; // intercept of the intercept.
            }
        }

        // Final pass for the actual interpolation. 
        auto output_sum = sum_interpolated_non_edge_forces(
            N,
            Y,
            mins,
            step,
            has_zero,
            interpolants,
            blocksize,
            neg,
            intervals,
            parallel_buffer
        );

        return output_sum;
    }
};

}

#endif
