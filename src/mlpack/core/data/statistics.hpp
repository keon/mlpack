/**
 * @file statistics.hpp
 * @author Keon Kim
 *
 * Defines the Statistics class, which calculates various statistics on a given
 * data.
 */
#ifndef MLPACK_CORE_DATA_STATISTICS_HPP
#define MLPACK_CORE_DATA_STATISTICS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

// Statistics class, it calculates most of the statistical elements in its
// constructor.
template <typename T>
class Statistics
{
 public:

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  Statistics(const arma::Mat<T>& input, const bool population = false,
      const bool columnMajor = true);

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Min(const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Max(const size_t dimension) const;

  std::pair<double, double> MinMax(const size_t dimension) const;
  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Range(const size_t dimension) const;

  /**
   * Mean, also known as average.
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Mean(const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Median(const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Variance(const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double StandardDeviation(const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double Skewness(const size_t dimension) const;

  /**
   * The sum of deviations to the Nth Power
   *
   * @param n power
   * @return sum of nth power deviations
   */
  double Kurtosis(const size_t dimension) const;

  /**
   * The sum of deviations to the Nth Power
   *
   * @param n power
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double SumNthPowerDeviations(const size_t n, const size_t dimension) const;

  /**
   * StandardError of Mean
   *
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double StandardError(const size_t dimension) const;

 private:
  // Is a copy of the original data.
  arma::Mat<T> data;

  // Determines if the dataset is considered as a sample or a population.
  bool population;

  // Determines if the dataset is considered as a column major or not.
  bool columnMajor;
};

} // namespace data
} // namespace mlpack

#include "statistics_impl.hpp"

#endif
