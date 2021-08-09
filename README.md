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

The code here is derived from the C++ code in the [**Rtsne** R package](https://github.com/jkrijthe/Rtsne/), updated for more modern C++.
The only modification is that we have introduced an extra `max_depth` parameter that bounds the depth of the quad trees used for the Barnes-Hut force calculations.
All nodes at the maximum depth are designated as leaf nodes; if multiple observations are present, they are aggregated into the center of mass for that node.
This provides an upper bound on the runtime of the force calculation for each observation, improving performance for large datasets at the cost of some approximation.

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
  WeightedLowess 
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

Otherwise, you can just copy the header file into some location that is visible to your compiler.

## References

van der Maaten, L.J.P. and Hinton, G.E. (2008). 
Visualizing high-dimensional data using t-SNE. 
_Journal of Machine Learning Research_, 9, 2579-2605.

van der Maaten, L.J.P. (2014). 
Accelerating t-SNE using tree-based algorithms. 
_Journal of Machine Learning Research_, 15, 3221-3245.

