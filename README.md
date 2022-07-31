# C++ library for t-SNE

![Unit tests](https://github.com/LTLA/qdtsne/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/qdtsne/actions/workflows/doxygenate.yaml/badge.svg)
![Rtsne comparison](https://github.com/LTLA/qdtsne/actions/workflows/compare-Rtsne.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/qdtsne/branch/master/graph/badge.svg?token=CX6G39BM7B)](https://codecov.io/gh/LTLA/qdtsne)

## Overview

This repository contains a header-only C++ library implementing the Barnes-Hut t-distributed stochastic neighbor embedding (t-SNE) algorithm (van der Maaten and Hinton, 2008; van der Maaten, 2014).
t-SNE is a non-linear dimensionality reduction technique that enables visualization of complex datasets by placing each observation on a low-dimensional (usually 2D) map based on its neighbors.
The general idea is that neighboring observations are placed next to each other, thus preserving the local structure of the dataset from its original high-dimensional form.
The non-linear nature of the algorithm gives it the flexibility to accommodate complicated structure that is not possible in linear techniques like PCA.

The code here is derived from the C++ code in the [**Rtsne** R package](https://github.com/jkrijthe/Rtsne/), itself taken from the 2014 paper.
It has been updated to use more modern C++, along with some additional options to sacrifice accuracy for speed - see below for details.

## Quick start

Using this library is as simple as including the header file in your source code:

```cpp
#include "qdtsne/qdtsne.hpp"

/** ... boilerplate here... **/

auto Y = qdtsne::initialize_random(N); // initial coordinates

/* Assuming `data` contains high-dimensional data in column-major format,
 * i.e., each column is a observation and each row is a dimension.
 * This fills `Y` in column-major format where each row is a dimension 
 * (defaulting to 2 dimensions) and each column is an observation.
 */
qdtsne::Tsne thing;
auto ref = thing.run(data.data(), D, N, Y.data());
```

You can change the parameters with the relevant setters:

```cpp
thing.set_perplexity(10).set_mom_switch_iter(100);
thing.run(data.data(), D, N, Y.data());
```

You can also stop and start the algorithm:

```cpp
qdtsne::Tsne thing;
auto ref = thing.initialize(data.data(), D, N);
ref.run(Y.data(), 200); // run up to 200 iterations
ref.run(Y.data(), 500); // run up to 500 iterations
```

See the [reference documentation](https://ltla.github.io/qdtsne/) for more details.

## Building projects

If you're already using CMake, you can add something like this to your `CMakeLists.txt`:

```
include(FetchContent)

FetchContent_Declare(
  qdtsne 
  GIT_REPOSITORY https://github.com/LTLA/qdtsne
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(qdtsne)
```

And then:

```
# For executables:
target_link_libraries(myexe qdtsne)

# For libaries
target_link_libraries(mylib INTERFACE qdtsne)
```

Otherwise, you can just copy the header files in `include/` into some location that is visible to your compiler.
This requires the manual inclusion of a few dependencies:

- The [**knncolle**](https://github.com/LTLA/knncolle) library for nearest neighbor search.
If you are instead supplying your own neighbor search, this dependency can be eliminated by defining the `QDTSNE_CUSTOM_NEIGHBORS` macro.
- The [**aarand**](https://github.com/LTLA/aarand) library for random distribution functions.

## Approximations for speed

The Barnes-Hut implementation (inherited from the 2014 paper) speeds up the algorithm by approximating the repulsive forces.
Specifically, the algorithm consolidates a group of distant points into a single center of mass, avoiding the need to calculate forces between individual points. 
The definition of "distant" is determined by the `theta` parameter, where larger values sacrifice accuracy for speed.
However, it must be said that the default `theta` is already appropriate in most settings; further increases may result in a noticeable deterioration in the visualization.

We have introduced an extra `max_depth` parameter that bounds the depth of the quad trees used for the Barnes-Hut force calculations.
All nodes at the maximum depth are designated as leaf nodes; if multiple observations are present, they are aggregated into the center of mass for that node.
This provides an upper bound on the runtime of the force calculation for each observation, e.g., a maximum depth of 7 means that there can be no more than 16384 leaf nodes.
Some timings with `gallery/speedtest.cpp` suggest that a moderate improvement in run-time is possible;
without any approximation, the iterations take 114 ± 1 seconds, while with `max_depth = 7`, the iterations take 85 ± 2 seconds.

We have also added an `interpolation` parameter that instructs the library to interpolate the repulsive forces for each observations.
Specifically, it divides up the bounding box for the current embedding into a grid of length equal to `interpolation` in each dimension.
We compute the repulsive forces at each grid point and perform linear interpolation to obtain the force at each point inside the grid.
This offers a large speed-up when the number of observations is much greater than the number of possible grid points.
We suggest using a grid resolution of 200 to ensure that artificial "edge effects" from interpolation are not visible.
(This implies that the dataset must have more than 40000 observations for the interpolation to provide any speed benefit.)

Of course, users can enable OpenMP to throw more threads at the problem.
With 4 threads and `max_depth` set as above, the iterations go down to 27 seconds.
The results are agnostic to the choice of parallelization, i.e., you can get the same results with or without using OpenMP.

## References

van der Maaten, L.J.P. and Hinton, G.E. (2008). 
Visualizing high-dimensional data using t-SNE. 
_Journal of Machine Learning Research_, 9, 2579-2605.

van der Maaten, L.J.P. (2014). 
Accelerating t-SNE using tree-based algorithms. 
_Journal of Machine Learning Research_, 15, 3221-3245.

