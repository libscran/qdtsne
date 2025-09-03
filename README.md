# Quick and dirty t-SNE in C++

![Unit tests](https://github.com/libscran/qdtsne/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/libscran/qdtsne/actions/workflows/doxygenate.yaml/badge.svg)
![Rtsne comparison](https://github.com/libscran/qdtsne/actions/workflows/compare-Rtsne.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/libscran/qdtsne/branch/master/graph/badge.svg?token=CX6G39BM7B)](https://codecov.io/gh/libscran/qdtsne)

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

// Assuming `data` contains high-dimensional data in column-major format,
// i.e., each column is a observation and each row is a dimension.
int nrow = 10;
int ncol = 2000;
std::vector<double> data(nrow * ncol);

// Configuring the neighbor search algorithm; here, we'll be using an exact
// search based on VP trees with a Euclidean distance metric.
knncolle::VptreeBuilder<int, double, double> nnalg(
    std::make_shared<knncolle::EuclideanDistance<double, double> >()
);

// Run the t-SNE algorithm for a 2-dimensional embedding.
qdtsne::Options opt;
auto status = qdtsne::initialize<2>(nrow, ncol, data.data(), nnalg, opt);
auto Y = qdtsne::initialize_random<2>(ncol); // starting from random 2D coordinates.
status.run(Y.data()); // Run through the iterations to obtain the t-SNE embedding.
```

You can change the parameters with the relevant setters:

```cpp
opt.perplexity = 10;
opt.mom_switch_iter = 100;
auto status2 = qdtsne::initialize<2>(nrow, ncol, data.data(), nnalg, opt);
```

You can also stop and start the algorithm:

```cpp
status2.run(Y.data(), 200); // run up to 200 iterations
status2.run(Y.data(), 500); // run up to 500 iterations
```

See the [reference documentation](https://libscran.github.io/qdtsne/) for more details.

## Approximations for speed

van der Maaten (2014) proposed the use of the Barnes-Hut approximation for the repulsive force calculations in t-SNE.
The algorithm consolidates a group of distant points into a single center of mass, avoiding the need to calculate forces between individual points. 
The definition of "distant" is determined by the `theta` parameter, where larger values sacrifice accuracy for speed.

In **qdtsne**, we introduce an extra `max_depth` parameter that bounds the depth of the tree used for the Barnes-Hut force calculations.
Setting a maximum depth of $m$ is equivalent to the following procedure:

1. Define the bounding box/hypercube for our dataset and partition it along each dimension into $2^m$ intervals, forming a high-dimensional grid.
2. In each grid cell, move all data points in that cell to the cell's center of mass.
3. Construct a standard Barnes-Hut tree on this modified dataset.
4. Use the tree to compute repulsive forces for each point $i$ using its original coordinates.

The approximation is based on ignoring the distribution within each grid cell, which should be acceptable at large $m$ where the intervals are small.
Smaller values of $m$ reduce computational time by limiting the depth of the recursion, at the cost of approximation quality for the repulsive force calculation.
A value of 7 to 10 seems to be a good compromise for most applications.

We can go even further by using the center of mass for $i$'s leaf node to approximate $i$'s repulsive forces with all other leaf nodes.
We compute the repulsive forces once per leaf node, and then re-use those values for all points assigned to the same node.
This eliminates near-redundant searches through the tree for neighboring points; the only extra calculation per point $i$ is the repulsion between $i$ and its leaf node.
We call this approach the "leaf approximation", which is enabled through the `leaf_approximation` parameter.
Note that this only has an effect in `max_depth`-bounded trees where multiple points are assigned to a leaf node.

Some testing indicates that both approximations can significantly speed up calculation of the embeddings.
Timings are shown below in seconds, based on a mock dataset containing 50,000 points (see [`tests/R/examples/basic.R`](tests/R/examples/basic.R) for details).

|Strategy|Serial|Parallel (OpenMP, n = 4)|
|----|----|---|
|Default|136|42| 
|`max_depth = 7`|85|26| 
|`max_depth = 7`, `leaf_approximation = true`|46|16| 

## Building projects

### CMake with `FetchContent`

If you're already using CMake, you can add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  qdtsne 
  GIT_REPOSITORY https://github.com/libscran/qdtsne
  GIT_TAG master # replace with a pinned release
)

FetchContent_MakeAvailable(qdtsne)
```

And then:

```cmake
# For executables:
target_link_libraries(myexe qdtsne)

# For libaries
target_link_libraries(mylib INTERFACE qdtsne)
```

By default, this will use `FetchContent` to fetch all external dependencies.
Applications should probably pin the version of each dependency themselves - see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for suggested versions.
If you want to install them manually, use `-DQDTSNE_FETCH_EXTERN=OFF`.

### CMake with `find_package()`

```cmake
find_package(libscran_qdtsne CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE libscran::qdtsne)
```

To install the library, use:

```sh
mkdir build && cd build
cmake .. -DQDTSNE_TESTS=OFF
cmake --build . --target install
```

Again, this will use `FetchContent` to fetch all external dependencies, see recommendations above.

### Manual

If you're not using CMake, you can just copy the header files in `include/` into some location that is visible to your compiler.
This requires the manual inclusion of the dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt).

## Comments on licensing

Most of the code in this repository is MIT-licensed (see the [`LICENSE`](LICENSE)).
The exception is the code that was derived from the **Rtsne** R package, which in turn was taken from the van der Maaten (2014) paper.
Readers are referred to the relevant source files ([`SPTree.hpp`](include/qdtsne/SPTree.hpp) and [`Status.hpp`](include/qdtsne/Status.hpp)) for their licensing conditions.

## References

van der Maaten, L.J.P. and Hinton, G.E. (2008). 
Visualizing high-dimensional data using t-SNE. 
_Journal of Machine Learning Research_, 9, 2579-2605.

van der Maaten, L.J.P. (2014). 
Accelerating t-SNE using tree-based algorithms. 
_Journal of Machine Learning Research_, 15, 3221-3245.
