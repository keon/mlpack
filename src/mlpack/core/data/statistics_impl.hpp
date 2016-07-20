/**
 * @file statistics_impl.hpp
 * @author Keon Kim
 *
 * Defines the Statistics class, which calculates various statistics on a given
 * data.
 */
#ifndef MLPACK_CORE_DATA_STATISTICS_IMPL_HPP
#define MLPACK_CORE_DATA_STATISTICS_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

template <typename T>
inline Statistics<T>::Statistics(const arma::Mat<T>& input,
    const bool population,
    const bool columnMajor):
    data(input),
    population(population),
    columnMajor(columnMajor)
{
  // Nothing to initialize here.
}

template <typename T>
inline double Statistics<T>::Min(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::max(z);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::max(z);
  }
}

template <typename T>
inline double Statistics<T>::Max(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::min(z);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::min(z);
  }
}

template <typename T>
inline double Statistics<T>::Range(const size_t dimension) const
{
  return Max(dimension) - Min(dimension);
}

template <typename T>
inline double Statistics<T>::Mean(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::mean(z);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::mean(z);
  }
}

template <typename T>
inline double Statistics<T>::Median(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::median(z);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::median(z);
  }
}

template <typename T>
inline double Statistics<T>::Variance(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::var(z, population);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::var(z, population);
  }
}

template <typename T>
inline double Statistics<T>::StandardDeviation(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
    return arma::stddev(z, population);
  }
  else
  {
    arma::vec z = data.col(dimension);
    return arma::stddev(z, population);
  }
}

template <typename T>
inline double Statistics<T>::Skewness(const size_t dimension) const
{
  double skewness = 0;
  double S3 = pow(StandardDeviation(dimension), 3);
  double M3 = SumNthPowerDeviations(3, dimension);
  double n = data.n_cols;
  if (population)
  {
    // Calculate Population Skewness
    skewness = n * M3 / (n * n * S3);
  }
  else
  {
    // Calculate Sample Skewness
    skewness = n * M3 / ((n-1) * (n-2) * S3);
  }
  return skewness;
}

template <typename T>
inline double Statistics<T>::Kurtosis(const size_t dimension) const
{
  double kurtosis = 0;
  double M4 = SumNthPowerDeviations(4, dimension);
  double n = data.n_cols;
  if (population)
  {
    // Calculate Population Excess Kurtosis
    double M2 = SumNthPowerDeviations(2, dimension);
    kurtosis = n * (M4 / pow(M2, 2)) - 3;
  }
  else
  {
    // Calculate Sample Excess Kurtosis
    double S4 = pow(StandardDeviation(dimension), 4);
    double norm3 = (3 * (n-1) * (n-1)) / ((n-2) * (n-3));
    double normC = (n * (n+1))/((n-1) * (n-2) * (n-3));
    double normM = M4 / S4;
    kurtosis = normC * normM - norm3;
  }
  return kurtosis;
}

template <typename T>
inline double Statistics<T>::SumNthPowerDeviations(const size_t n, const size_t dimension) const
{
  double sum = 0;
  double mean = Mean(dimension);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    sum += pow(data(dimension, i) - mean, n);
  }
  return sum;
}

template <typename T>
inline double Statistics<T>::StandardError(const size_t dimension) const
{
   return StandardDeviation(dimension) / sqrt(data.n_cols);
}

} // namespace data
} // namespace mlpack

#endif
