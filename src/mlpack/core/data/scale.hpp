/**
 * @file scale.hpp
 * @author Keon Kim
 *
 * Defines Scale functions.
 */
#ifndef MLPACK_CORE_DATA_SCALE_HPP
#define MLPACK_CORE_DATA_SCALE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

/**
 * StandardScale
 * Standardize by removing the mean and scaling to unit variance.
 */
template<typename T>
void StandardScale();

/**
 * Minmaxscale
 * Data is scaled to a fixed range
 * Smaller standard deviations,
 * which can suppress the effect of outliers
 * Xnorm = (X - Xmin)/(Xmax-Xmin)
 */
template<typename T>
void MinMaxScale();

/**
 * Maxabsscale
 */
template<typename T>
void MaxAbsScale();

/**
 * robust scale
 */
template<typename T>
void RobustScale();


} // namespace data
} // namespace mlpack

#include "scale_impl.hpp"

#endif
