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
thing.run(ref, Y.data());
thing.set_max_iter(1100).run(ref, Y.data()); // run for more iterations
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

We have introduced an extra `max_depth` parameter that bounds the depth of the quad trees used for the Barnes-Hut force calculations.
All nodes at the maximum depth are designated as leaf nodes; if multiple observations are present, they are aggregated into the center of mass for that node.
This provides an upper bound on the runtime of the force calculation for each observation, e.g., a maximum depth of 7 means that there can be no more than 16384 leaf nodes 

We have also added an `interpolation` parameter that instructs the library to use interpolation to compute the repulsive forces.
Specifically, it divides up the bounding box for the current embedding into a grid of length equal to `interpolation` in each dimension.
We compute the repulsive forces at each grid point and then perform linear interpolation to obtain the force at each point inside the grid.

Some timings with the `gallery/speedtest.cpp` code indicate that the improvements can be considerable:

- Without any approximation, the iterations take 114 ± 1 seconds.
- With `max_depth = 7`, the iterations take 85 ± 2 seonds.
- With `max_depth = 7` and `interpolation = 100`, the iterations take 46 ± 1 seconds.

And of course, users can enable OpenMP to throw more threads at the problem.
With 4 threads and `max_depth` and `interpolation` set as above, the iterations go down to 23 seconds.

## References

van der Maaten, L.J.P. and Hinton, G.E. (2008). 
Visualizing high-dimensional data using t-SNE. 
_Journal of Machine Learning Research_, 9, 2579-2605.

van der Maaten, L.J.P. (2014). 
Accelerating t-SNE using tree-based algorithms. 
_Journal of Machine Learning Research_, 15, 3221-3245.

