/**
 * @file statistics.hpp
 * @author Keon Kim
 *
 * Defines the Statistics class, which calculates various statistics on a given
 * data.
 */
#ifndef MLPACK_METHODS_PREPROCESS_STATISTICS_HPP
#define MLPACK_METHODS_PREPROCESS_STATISTICS_HPP

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
   * Create the Statistics object with the given input and options.
   *
   * @param input Dataset a user wishes to analyze.
   * @param population Determines if the input is considered sample or
   *        population data.
   * @param columnMajor Determines if the input is columnMajor or not.
   */
  Statistics(const arma::Mat<T>& input, const bool population = false,
      const bool columnMajor = true);

  /**
   * Calculates minimum of the input matrix with the given dimension.
   *
   * @param dimension Dimension
   * @return Minimum value of the given dimension.
   */
  double Min(const size_t dimension) const;

  /**
   * Calculates maximum of the input matrix with the given dimension.
   *
   * @param dimension Dimension
   * @return Maximum value of the given dimension.
   */
  double Max(const size_t dimension) const;

  /**
   * Calculates minimum and maximum of the input matrix with the given
   * dimension. It is equivalent function as std::minmax, but it applies
   * to the armadillo matrix inputs.
   *
   * @param dimension Dimension
   * @return A pair of minimum and maximum values of the given dimension.
   */
  std::pair<double, double> MinMax(const size_t dimension) const;

  /**
   * Calculates minimum and maximum and uses those values to derive range of
   * the dataset of the given dimension.
   *
   * @param dimension Dimension
   * @return Range value of the given dimension.
   */
  double Range(const size_t dimension) const;

  /**
   * Calculates mean, also known as average, of the given dimension of the input
   * matrix.
   *
   * @param dimension Dimension
   * @return Mean value of the given dimension.
   */
  double Mean(const size_t dimension) const;

  /**
   * Calculates median of the given dimension of the input matrix.
   *
   * @param dimension Dimension
   * @return Median of the given dimension.
   */
  double Median(const size_t dimension) const;

  /**
   * Calculates varianc e of the given dimension of the input matrix.
   *
   * @param dimension Dimension
   * @return Variance value of the given dimension.
   */
  double Variance(const size_t dimension) const;

  /**
   * Calculates standard deviation of the given dimension.
   *
   * @param dimension Dimension
   * @return Standard deviation.
   */
  double StandardDeviation(const size_t dimension) const;

  /**
   * Calculates Skewness of the given dimension.
   *
   * @param dimension Dimension
   * @return Skewness
   */
  double Skewness(const size_t dimension) const;

  /**
   * Calculates kurtosis of the given dimension.
   *
   * @param n power
   * @return Kurtosis.
   */
  double Kurtosis(const size_t dimension) const;

  /**
   * Calculates the sum of deviations to the Nth Power
   *
   * @param n power
   * @param dimension Dimension
   * @return sum of nth power deviations
   */
  double SumNthPowerDeviations(const size_t n, const size_t dimension) const;

  /**
   * Calculates standard error.
   *
   * @param dimension Dimension
   * @return Standard error.
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
