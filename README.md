mlpack: a scalable C++ machine learning library
===============================================

mlpack is an intuitive, fast, scalable C++ machine learning library, meant to be
a machine learning analog to LAPACK. It aims to implement a wide array of
machine learning methods and functions as a "swiss army knife" for machine
learning researchers.

**Download [current stable version (2.0.1)](http://www.mlpack.org/files/mlpack-2.0.1.tar.gz).**

[![Build Status](http://big.mlpack.org/job/mlpack%20-%20git%20commit%20test/badge/icon)](http://big.mlpack.org/job/mlpack%20-%20git%20commit%20test/) ("https://ci.appveyor.com/project/mlpack/mlpack"><img src="https://ci.appveyor.com/api/projects/status/lmbfc78wi16agx4q?svg=true" alt="Build status" height="18"></a>

0. Contents
-----------

  1. [Introduction](#1-introduction)
  2. [Citation details](#2-citation-details)
  3. [Dependencies](#3-dependencies)
  4. [Building mlpack from source](#4-building-mlpack-from-source)
  5. [Running mlpack programs](#5-running-mlpack-programs)
  6. [Further documentation](#6-further-documentation)
  7. [Bug reporting](#7-bug-reporting)

1. Introduction
---------------

The mlpack website can be found at http://www.mlpack.org and contains numerous
tutorials and extensive documentation.  This README serves as a guide for what
mlpack is, how to install it, how to run it, and where to find more
documentation. The website should be consulted for further information:

  - [mlpack homepage](http://www.mlpack.org/)
  - [Tutorials](http://www.mlpack.org/docs/mlpack-git/doxygen.php?doc=tutorials.html)
  - [Development Site (Github)](http://www.github.com/mlpack/mlpack/)
  - [API documentation](http://www.mlpack.org/docs/mlpack-git/doxygen.php)

Below is a high-level list of the available functionality contained
within [mlpack](http://mlpack.org), along with relevant links to
papers, API documentation, tutorials, or other references.

  - AdaBoost
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1adaboost.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_adaboost.html)
      [[wiki]](https://en.wikipedia.org/wiki/AdaBoost)
      [[paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.91.9277&rep=rep1&type=pdf)
  - Alternating Matrix Factorization (AMF)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1amf.html)
  - Collaborative Filtering (with many decomposition techniques) (CF)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1cf.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_cf.html)
      [[wiki]](http://en.wikipedia.org/wiki/Collaborative_filtering)
  - Decision Stumps (one-level decision trees)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1decision__stump.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_decision_stump.html)
      [[paper]](http://www.mlpack.org/papers/ds.pdf)
  - Density Estimation Trees (DET)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1det.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_det.html)
      [[tutorial]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=dettutorial.html)
      [[paper]](http://www.mlpack.org/papers/det.pdf)
  - Euclidean Minimum Spanning Tree Calculation (EMST)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1emst.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_emst.html)
      [[tutorial]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=emst_tutorial.html) 
      [[paper]](http://www.mlpack.org/papers/emst.pdf)
  - Fast Exact Max-Kernel Search (FaskMKS)
      [[api]]()
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_fastmks.html)
  - Gaussian Mixture Models (GMM)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1gmm.html) 
      [cli-executable
        [(train)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_train.html)
        [(generate)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_generate.html)
        [(probability)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_probability.html)
      ]
      [[wiki]](http://en.wikipedia.org/wiki/Gaussian_mixture_model)
  - Hidden Markov Models (HMM)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1hmm.html) 
      [cli-executable
        [(train)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_train.html)
        [(generate)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_generate.html)
        [(loglik)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_loglik.html)
        [(viterbi)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_viterbi.html)
      ]
      [[wiki]](http://en.wikipedia.org/wiki/Hidden_Markov_Model)
  - Hoeffding Tree
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=classmlpack_1_1tree_1_1HoeffdingTree.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_hoeffding_tree.html)
  - Kernel Principal Components Analysis (optionally with sampling) (KernelPCA)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1kpca.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_kernel_pca.html)
      [[paper]](http://www.mlpack.org/papers/kpca.pdf)
      [[wiki]](http://en.wikipedia.org/wiki/Kernel_principal_component_analysis)
  - k-Means Clustering (with several accelerated algorithms)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1kmeans.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_kmeans.html)
      [[tutorial]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=kmtutorial.html) 
      [[wiki]](http://en.wikipedia.org/wiki/K-means_clustering)
  - Least-Angle Regression (LARS/LASSO) 
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1regression.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_lars.html)
      [[paper]](http://www.mlpack.org/papers/lars.pdf)
      [[wiki]](http://en.wikipedia.org/wiki/Least-angle_regression)
  - Linear Regression (simple least-squares) 
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1regression.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_linear_regression.html)
      [[wiki]](http://en.wikipedia.org/wiki/Linear_regression)
  - Local Coordinate coding 
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1lcc.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_local_coordinate_coding.html)
      [[paper]](http://www.mlpack.org/papers/lcc.pdf)
  - Locality-Sensitive Hashing for Approximate Nearest Neighbor Search (LSH)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1neighbor.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_lsh.html)
      [[paper]](http://www.mlpack.org/papers/lsh.pdf)
      [[wiki]](http://en.wikipedia.org/wiki/Locality-sensitive_hashing)
  - Logistic Regression
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1regression.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_logistic_regression.html)
      [[wiki]](http://en.wikipedia.org/wiki/Logistic_regression)
  - Matrix Completion (nuclear norm minimization)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=classmlpack_1_1matrix__completion_1_1MatrixCompletion.html)
      [[cli-executable]]()
  - Max-Kernel Search
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1fastmks.html)
      [[cli-executable]]()
      [[tutorial]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=fmkstutorial.html) 
      [[paper]](http://www.mlpack.org/papers/fmks.pdf)
  - Mean Shift Clustering
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=classmlpack_1_1meanshift_1_1MeanShift.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_mean_shift.html)
  - Maximum Variance Unfolding (MVU)
      [[api]]()
      [[cli-executable]]()
  - Naive Bayes Classifier 
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1naive__bayes.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_nbc.html)
      [[wiki]](http://en.wikipedia.org/wiki/Naive_Bayes_classifier)
  - Nearest Neighbor Search with Dual-Tree Algorithms 
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1neighbor.html) 
      [cli-executable
        [(k-furthest)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_train.html)
        [(k-nearest)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_generate.html)
        [(loglik)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_loglik.html)
        [(viterbi)](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_gmm_viterbi.html)
      ]
      [[tutorial]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=nstutorial.html) 
      [[paper]](http://www.mlpack.org/papers/ns.pdf) 
      [[wiki]](http://en.wikipedia.org/wiki/Nearest_neighbor_search)
  - Neighborhood Components Analysis (NCA)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1nca.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_nca.html)
      [[paper]](http://www.mlpack.org/papers/nca.pdf) 
      [[wiki]](http://en.wikipedia.org/wiki/Neighborhood_components_analysis)
  - Neural Network (NN)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1nn.html)
      [[wiki]](https://en.wikipedia.org/wiki/Artificial_neural_network)
  - Non-Negative Matrix Factorization (NMF)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1amf.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_nmf.html)
      [[paper]](http://www.mlpack.org/papers/nmf.pdf) 
      [[wiki]](http://en.wikipedia.org/wiki/Nonnegative_matrix_factorization)
  - Perceptrons
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1perceptron.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_perceptron.html)
      [[wiki]](http://en.wikipedia.org/wiki/Perceptron)
  - Principal Components Analysis (PCA)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1pca.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_pca.html)
      [[wiki]](http://en.wikipedia.org/wiki/Principal_components_analysis)
  - RADICAL (independent components analysis)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1radical.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_radical.html)
      [[paper]](http://www.mlpack.org/papers/radical.pdf)
  - Range Search with Dual-Tree Algorithms
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1range.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_range_search.html)
      [[tutorial]](doxygen.php?doc=rstutorial.html)
      [[paper]](http://www.mlpack.org/papers/rs.pdf)
      [[wiki]](http://en.wikipedia.org/wiki/Range_search)
  - Rank-Approximate Nearest Neighbor Search (RANN)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=classRASearch.html) 
      [[cli-executable]]()
      [[paper]](http://www.mlpack.org/papers/rann.pdf)
  - Regularized Singular Value Decomposition (SVD)
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1svd.html)
      [[cli-executable]]()
  - Recurrent Model for Visual Attention (RMVA)
      [[api]]()
      [[cli-executable]]()
  - Softmax Regression
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=classmlpack_1_1regression_1_1SoftmaxRegression.html)
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_softmax_regression.html)
  - Sparse Autoencoder
      [[api]]()
      [[cli-executable]]()
  - Sparse Coding with Dictionary Learning
      [[api]](http://www.mlpack.org/docs/mlpack-2.0.3/doxygen.php?doc=namespacemlpack_1_1sparse__coding.html) 
      [[cli-executable]](http://www.mlpack.org/docs/mlpack-2.0.3/man/mlpack_sparse_coding.html)
      [[paper]](http://www.mlpack.org/papers/sparse_coding.pdf) 
      [[wiki]](http://en.wikipedia.org/wiki/Sparse_coding)

2. Citation details
-------------------

If you use mlpack in your research or software, please cite mlpack using the
citation below (given in BiBTeX format):

    @article{mlpack2013,
      title     = {{mlpack}: A Scalable {C++} Machine Learning Library},
      author    = {Curtin, Ryan R. and Cline, James R. and Slagle, Neil P. and
                   March, William B. and Ram, P. and Mehta, Nishant A. and Gray,
                   Alexander G.},
      journal   = {Journal of Machine Learning Research},
      volume    = {14},
      pages     = {801--805},
      year      = {2013}
    }

Citations are beneficial for the growth and improvement of mlpack.

3. Dependencies
---------------

mlpack has the following dependencies:

      Armadillo     >= 4.100.0
      Boost (program_options, math_c99, unit_test_framework, serialization)
      CMake         >= 2.8.5

All of those should be available in your distribution's package manager.  If
not, you will have to compile each of them by hand.  See the documentation for
each of those packages for more information.

If you are compiling Armadillo by hand, ensure that LAPACK and BLAS are enabled.

4. Building mlpack from source
------------------------------

(see also [Building mlpack From Source](http://www.mlpack.org/doxygen.php?doc=build.html))

mlpack uses CMake as a build system and allows several flexible build
configuration options. One can consult any of numerous CMake tutorials for
further documentation, but this tutorial should be enough to get mlpack built
and installed.

First, unpack the mlpack source and change into the unpacked directory.  Here we
use mlpack-x.y.z where x.y.z is the version.

    $ tar -xzf mlpack-x.y.z.tar.gz
    $ cd mlpack-x.y.z

Then, make a build directory.  The directory can have any name, not just
'build', but 'build' is sufficient.

    $ mkdir build
    $ cd build

The next step is to run CMake to configure the project.  Running CMake is the
equivalent to running `./configure` with autotools. If you run CMake with no
options, it will configure the project to build with no debugging symbols and no
profiling information:

    $ cmake ../

You can specify options to compile with debugging information and profiling
information:

    $ cmake -D DEBUG=ON -D PROFILE=ON ../

Options are specified with the -D flag.  A list of options allowed:

    DEBUG=(ON/OFF): compile with debugging symbols
    PROFILE=(ON/OFF): compile with profiling symbols
    ARMA_EXTRA_DEBUG=(ON/OFF): compile with extra Armadillo debugging symbols
    BOOST_ROOT=(/path/to/boost/): path to root of boost installation
    ARMADILLO_INCLUDE_DIR=(/path/to/armadillo/include/): path to Armadillo headers
    ARMADILLO_LIBRARY=(/path/to/armadillo/libarmadillo.so): Armadillo library

Other tools can also be used to configure CMake, but those are not documented
here.

Once CMake is configured, building the library is as simple as typing 'make'.
This will build all library components as well as 'mlpack_test'.

    $ make

You can specify individual components which you want to build, if you do not
want to build everything in the library:

    $ make mlpack_pca mlpack_knn mlpack_kfn

If the build fails and you cannot figure out why, register an account on Github
and submit an issue; the mlpack developers will quickly help you figure it out:

[mlpack on Github](https://www.github.com/mlpack/mlpack/)

Alternately, mlpack help can be found in IRC at `#mlpack` on irc.freenode.net.

If you wish to install mlpack to `/usr/local/include/mlpack/` and `/usr/local/lib/`
and `/usr/local/bin/`, once it has built, make sure you have root privileges (or
write permissions to those three directories), and simply type

    $ make install

You can now run the executables by name; you can link against mlpack with
    `-lmlpack`
and the mlpack headers are found in
    `/usr/local/include/mlpack/`.

If running the programs (i.e. `$ mlpack_knn -h`) gives an error of the form

    error while loading shared libraries: libmlpack.so.2: cannot open shared object file: No such file or directory

then be sure that the runtime linker is searching the directory where
`libmlpack.so` was installed (probably `/usr/local/lib/` unless you set it
manually).  One way to do this, on Linux, is to ensure that the
`LD_LIBRARY_PATH` environment variable has the directory that contains
`libmlpack.so`.  Using bash, this can be set easily:

    export LD_LIBRARY_PATH=/usr/local/lib/

(or whatever directory `libmlpack.so` is installed in.)

5. Running mlpack programs
--------------------------

After building mlpack, the executables will reside in `build/bin/`.  You can call
them from there, or you can install the library and (depending on system
settings) they should be added to your PATH and you can call them directly.  The
documentation below assumes the executables are in your PATH.

Consider the 'mlpack_knn' program, which finds the k nearest neighbors in a
reference dataset of all the points in a query set.  That is, we have a query
and a reference dataset. For each point in the query dataset, we wish to know
the k points in the reference dataset which are closest to the given query
point.

Alternately, if the query and reference datasets are the same, the problem can
be stated more simply: for each point in the dataset, we wish to know the k
nearest points to that point.

Each mlpack program has extensive help documentation which details what the
method does, what each of the parameters are, and how to use them:

    $ mlpack_knn --help

Running `mlpack_knn` on one dataset (that is, the query and reference
datasets are the same) and finding the 5 nearest neighbors is very simple:

    $ mlpack_knn -r dataset.csv -n neighbors_out.csv -d distances_out.csv -k 5 -v

The `-v (--verbose)` flag is optional; it gives informational output.  It is not
unique to `mlpack_knn` but is available in all mlpack programs.  Verbose
output also gives timing output at the end of the program, which can be very
useful.

6. Further documentation
------------------------

The documentation given here is only a fraction of the available documentation
for mlpack.  If doxygen is installed, you can type `make doc` to build the
documentation locally.  Alternately, up-to-date documentation is available for
older versions of mlpack:

  - [mlpack homepage](http://www.mlpack.org/)
  - [Tutorials](http://www.mlpack.org/tutorials.html)
  - [Development Site (Github)](https://www.github.com/mlpack/mlpack/)
  - [API documentation](http://www.mlpack.org/doxygen.php)

7. Bug reporting
----------------
   (see also [mlpack help](http://www.mlpack.org/help.html))

If you find a bug in mlpack or have any problems, numerous routes are available
for help.

Github is used for bug tracking, and can be found at
https://github.com/mlpack/mlpack/.
It is easy to register an account and file a bug there, and the mlpack
development team will try to quickly resolve your issue.

In addition, mailing lists are available.  The mlpack discussion list is
available at

  [mlpack discussion list](https://lists.cc.gatech.edu/mailman/listinfo/mlpack)

and the git commit list is available at

  [commit list](https://lists.cc.gatech.edu/mailman/listinfo/mlpack-git)

Lastly, the IRC channel ```#mlpack``` on Freenode can be used to get help.
