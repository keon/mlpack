/**
 * @file scalar.hpp
 * @author Keon Kim
 *
 * Defines the DatasetInfo class, which holds information about a dataset.  This
 * is useful when the dataset contains categorical non-numeric features that
 * needs to be mapped to categorical numeric features.
 */
#ifndef MLPACK_CORE_DATA_SCALAR_HPP
#define MLPACK_CORE_DATA_SCALAR_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace data {

/**
 * The Datatype enum specifies the types of data mlpack algorithms can use.  The
 * vast majority of mlpack algorithms can only use numeric data (i.e.
 * float/double/etc.), but some algorithms can use categorical data, specified
 * via this Datatype enum and the DatasetInfo class.
 */
enum Datatype : bool /* bool is all the precision we need for two types */
{
  numeric = 0,
  categorical = 1
};

/**
 * Auxiliary information for a dataset, including mappings to/from strings and
 * the datatype of each dimension.  DatasetInfo objects are optionally produced
 * by data::Load(), and store the type of each dimension (Datatype::numeric or
 * Datatype::categorical) as well as mappings from strings to unsigned integers
 * and vice versa.
 */
class Scalar
{
 public:
  /**
   * Create the DatasetInfo object with the given dimensionality.  Note that the
   * dimensionality cannot be changed later; you will have to create a new
   * DatasetInfo object.
   */
  DatasetInfo(const size_t dimensionality = 0);

  /**
   * Given the string and the dimension to which it belongs, return its numeric
   * mapping.  If no mapping yet exists, the string is added to the list of
   * mappings for the given dimension.  The dimension parameter refers to the
   * index of the dimension of the string (i.e. the row in the dataset).
   *
   * @param string String to find/create mapping for.
   * @param dimension Index of the dimension of the string.
   */
  size_t MapString(const std::string& string, const size_t dimension);

  /**
   * Return the string that corresponds to a given value in a given dimension.
   * If the string is not a valid mapping in the given dimension, a
   * std::invalid_argument is thrown.
   *
   * @param value Mapped value for string.
   * @param dimension Dimension to unmap string from.
   */
  const std::string& UnmapString(const size_t value, const size_t dimension);

  //! Return the type of a given dimension (numeric or categorical).
  Datatype Type(const size_t dimension) const;
  //! Modify the type of a given dimension (be careful!).
  Datatype& Type(const size_t dimension);

  /**
   * Get the number of mappings for a particular dimension.  If the dimension
   * is numeric, then this will return 0.
   */
  size_t NumMappings(const size_t dimension) const;

  /**
   * Get the dimensionality of the DatasetInfo object (that is, how many
   * dimensions it has information for).  If this object was created by a call
   * to mlpack::data::Load(), then the dimensionality will be the same as the
   * number of rows (dimensions) in the dataset.
   */
  size_t Dimensionality() const;

  /**
   * Serialize the dataset information.
   */
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(types, "types");
    ar & data::CreateNVP(maps, "maps");
  }

 private:
  std::vector<Datatype> types;

  //! Mappings from strings to integers.  Map entries will only exist for
  //! dimensions that are categorical.
  std::unordered_map<size_t, std::pair<boost::bimap<std::string, size_t>,
      size_t>> maps;

};

} // namespace data
} // namespace mlpack

#include "dataset_info_impl.hpp"

#endif
