/**
 * @file one_hot_encoder.hpp
 * @author Keon Kim
 *
 * Defines OneHotEncoder()
 */
#ifndef MLPACK_CORE_DATA_SPLIT_DATA_HPP
#define MLPACK_CORE_DATA_SPLIT_DATA_HPP

#include <mlpack/core.hpp>
#include <vector>

namespace mlpack {
namespace data {

template<typename T>
void OneHotEncoder(const arma::Mat<T>& input // array-like
                   const arma::Mat<T>& output
                   const size_t n_unique) // number of unique categories
{

}


// automatically detect number of unique categories
template<typename T>
void OneHotEncoder(const arma::Mat<T>& input
                   const arma::Mat<T>& output)
{
  // calculate n_uniqe for now, it is just 5
  size_t n_unique = 5;
  OneHotEncoder(input, output, n_unique);
}

} // namespace data
} // namespace mlpack

#endif
