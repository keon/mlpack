/**
 * @file binarize_test.cpp
 * @author Keon Kim
 *
 * Test the Binarzie method.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/binarize.hpp>
#include <mlpack/core/math/random.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace arma;
using namespace mlpack::data;

BOOST_AUTO_TEST_SUITE(BinarizeTest);

BOOST_AUTO_TEST_CASE(BinerizeOneDimension)
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;
  const size_t dimension = 1;
  Binarize<double>(input, output, threshold, dimension);

  BOOST_REQUIRE_CLOSE(input(0, 0), 1, 1e-5); // 1
  BOOST_REQUIRE_CLOSE(input(0, 1), 2, 1e-5); // 2
  BOOST_REQUIRE_CLOSE(input(0, 2), 3, 1e-5); // 3
  BOOST_REQUIRE_SMALL(input(1, 0), 1e-5); // 4 target
  BOOST_REQUIRE_SMALL(input(1, 1), 1e-5); // 5 target
  BOOST_REQUIRE_CLOSE(input(1, 2), 1, 1e-5); // 6 target
  BOOST_REQUIRE_CLOSE(input(2, 0), 7, 1e-5); // 7
  BOOST_REQUIRE_CLOSE(input(2, 1), 8, 1e-5); // 8
  BOOST_REQUIRE_CLOSE(input(2, 2), 9, 1e-5); // 9
}

BOOST_AUTO_TEST_CASE(BinerizeAll)
{
  mat input;
  input << 1 << 2 << 3 << endr
        << 4 << 5 << 6 << endr // this row will be tested
        << 7 << 8 << 9;

  mat output;
  const double threshold = 5.0;
  const size_t dimension = 1;
  Binarize<double>(input, output, threshold);

  BOOST_REQUIRE_SMALL(input(0, 0), 1e-5); // 1
  BOOST_REQUIRE_SMALL(input(0, 1), 1e-5); // 2
  BOOST_REQUIRE_SMALL(input(0, 2), 1e-5); // 3
  BOOST_REQUIRE_SMALL(input(1, 0), 1e-5); // 4
  BOOST_REQUIRE_SMALL(input(1, 1), 1e-5); // 5
  BOOST_REQUIRE_CLOSE(input(1, 2), 1.0, 1e-5); // 6
  BOOST_REQUIRE_CLOSE(input(2, 0), 1.0, 1e-5); // 7
  BOOST_REQUIRE_CLOSE(input(2, 1), 1.0, 1e-5); // 8
  BOOST_REQUIRE_CLOSE(input(2, 2), 1.0, 1e-5); // 9
}

BOOST_AUTO_TEST_SUITE_END();