/**
 * @file statistics_impl.hpp
 * @author Keon Kim
 *
 * Defines the Descriptive DescriptiveStatistics class, which calculates various
 * statistics on a given data.
 */
#ifndef MLPACK_METHODS_PREPROCESS_DESCRIPTIVE_STATISTICS_IMPL_HPP
#define MLPACK_METHODS_PREPROCESS_DESCRIPTIVE_STATISTICS_IMPL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

template <typename T>
inline DescriptiveStatistics<T>::DescriptiveStatistics(
    arma::Mat<T>& input,
    const bool population,
    const bool columnMajor):
    data(input)
{
  // Nothing to initialize here.
}

template <typename T>
inline double DescriptiveStatistics<T>::Min(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::max(z);
}

template <typename T>
inline double DescriptiveStatistics<T>::Max(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::min(z);
}

template <typename T>
inline double DescriptiveStatistics<T>::Range(const size_t dimension) const
{
  std::pair<double, double> minmax = MinMax(dimension);
  return minmax.second - minmax.first;
}

template <typename T>
inline std::pair<double, double> DescriptiveStatistics<T>::MinMax(
    const size_t dimension) const
{
  double min = 0;
  double max = 0;
  if (columnMajor)
  {
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      if (data(dimension, i) > max)
      {
        max = data(dimension, i);
      }
      if (data(dimension, i) < min)
      {
        min = data(dimension, i);
      }
    }
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      if (data(i, dimension) > max)
      {
        max = data(i, dimension);
      }
      if (data(i, dimension) < min)
      {
        min = data(i, dimension);
      }
    }
  }
  return std::make_pair(min, max);
}

template <typename T>
inline double DescriptiveStatistics<T>::Mean(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::mean(z);
}

template <typename T>
inline double DescriptiveStatistics<T>::Median(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::median(z);
}

template <typename T>
inline double DescriptiveStatistics<T>::Variance(const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::var(z, population);
}

template <typename T>
inline double DescriptiveStatistics<T>::StandardDeviation(
    const size_t dimension) const
{
  if (columnMajor)
  {
    arma::rowvec z = data.row(dimension);
  }
  else
  {
    arma::vec z = data.col(dimension);
  }
  return arma::stddev(z, population);
}

template <typename T>
inline double DescriptiveStatistics<T>::Skewness(const size_t dimension) const
{
  double skewness = 0;
  double S3 = pow(StandardDeviation(dimension), 3);
  double M3 = SumNthPowerDeviations(3, dimension);
  double n = columnMajor ? data.n_cols : data.n_rows;
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
inline double DescriptiveStatistics<T>::Kurtosis(const size_t dimension) const
{
  double kurtosis = 0;
  double M4 = SumNthPowerDeviations(4, dimension);
  double n = columnMajor ? data.n_cols : data.n_rows;
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
inline double DescriptiveStatistics<T>::SumNthPowerDeviations(const size_t n,
    const size_t dimension) const
{
  double sum = 0;
  double mean = Mean(dimension);
  if (columnMajor)
  {
    for (size_t i = 0; i < data.n_cols; ++i)
    {
      sum += pow(data(dimension, i) - mean, n);
    }
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      sum += pow(data(i, dimension) - mean, n);
    }
  }
  return sum;
}

template <typename T>
inline double DescriptiveStatistics<T>::StandardError(
    const size_t dimension) const
{
   return StandardDeviation(dimension) / sqrt(data.n_cols);
}

} // namespace data
} // namespace mlpack

#endif
