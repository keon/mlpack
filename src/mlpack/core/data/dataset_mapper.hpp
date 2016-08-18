/**
 * @file dataset_mapper.hpp
 * @author Ryan Curtin
 * @author Keon Kim
 *
 * Defines the DatasetMapper class, which holds information about a dataset.
 * This is useful when the dataset contains categorical non-numeric features
 * that needs to be mapped to categorical numeric features.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>

#include "map_policies/increment_policy.hpp"

namespace mlpack {
namespace data {
/**
 * Auxiliary information for a dataset, including mappings to/from strings and
 * the datatype of each dimension.  DatasetMapper objects are optionally
 * produced by data::Load(), and store the type of each dimension
 * (Datatype::numeric or Datatype::categorical) as well as mappings from strings
 * to unsigned integers and vice versa.
 *
 * @tparam PolicyType Mapping policy used to specify MapString();
 */
template <typename PolicyType>
class DatasetMapper
{
 public:
  /**
   * BiMapType definition
   */
  using MappedObjectType = std::pair<size_t, size_t>;
  using BiMapType =
      boost::bimap<std::string, typename PolicyType::MappedType>;

  using ObjectBiMapType =
      boost::bimap<std::string, MappedObjectType>;
  /**
   * Mappings from strings to integers.
   * Map entries will only exist for dimensions that are categorical.
   * MapType = map<dimension, pair<bimap<string, MappedType>, numMappings>>
   */
  using MapType = std::unordered_map<size_t, std::pair<BiMapType, size_t>>;
  using ObjectMapType =
      std::unordered_map<size_t, std::pair<ObjectBiMapType, size_t>>;

  /**
   * Create the DatasetMapper object with the given dimensionality.  Note that
   * the dimensionality cannot be changed later; you will have to create a new
   * DatasetMapper object.
   */
  explicit DatasetMapper(const size_t dimensionality = 0);

  /**
   * Create the DatasetMapper object with the given policy and dimensionality.
   * Note that the dimensionality cannot be changed later; you will have to
   * create a new DatasetMapper object. Policy can be modified by the modifier.
   */
  explicit DatasetMapper(PolicyType& policy, const size_t dimensionality = 0);

  /**
   * Given the string and the dimension to which it belongs, return its numeric
   * mapping.  If no mapping yet exists, the string is added to the list of
   * mappings for the given dimension.  The dimension parameter refers to the
   * index of the dimension of the string (i.e. the row in the dataset).
   *
   * @param string String to find/create mapping for.
   * @param dimension Index of the dimension of the string.
   */
  typename PolicyType::MappedType MapString(const std::string& string,
                                            const size_t dimension);

  double MapInvalidValue(const std::string& string,
                         const size_t dimension,
                         const size_t point)
  {

    //return policy.template MapString<MapType, ObjectMapType>(string, dimension,
        //maps, invalidMaps, types);
/***/
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    Log::Debug << "<INVALID MAPPING>" << std::endl;
    // This string does not exist yet.
    typedef boost::bimap<std::string, MappedObjectType>::value_type PairType;
    MappedObjectType coordinates(dimension, point);
    invalidMaps[dimension].first.insert(PairType(string, coordinates));

    size_t& numMappings = invalidMaps[dimension].second;
    ++numMappings;
    return NaN;
  }

  std::pair<size_t, size_t> UnmapInvalidValue(const std::string& string,
                                              const size_t dimension)
  {
    // Throw an exception if the value doesn't exist.
    if (maps[dimension].first.left.count(string) == 0)
    {
      std::ostringstream oss;
      oss << "DatasetMapper<PolicyType>::UnmapValue(): string '" << string
          << "' unknown for dimension " << dimension;
      throw std::invalid_argument(oss.str());
    }

    return maps[dimension].first.left.at(string);
  }

  /**
   * Return the string that corresponds to a given value in a given dimension.
   * If the string is not a valid mapping in the given dimension, a
   * std::invalid_argument is thrown.
   *
   * @param value Mapped value for string.
   * @param dimension Dimension to unmap string from.
   */
  const std::string& UnmapString(const size_t value, const size_t dimension);


  /**
   * Return the value that corresponds to a given string in a given dimension.
   * If the value is not a valid mapping in the given dimension, a
   * std::invalid_argument is thrown.
   *
   * @param string Mapped string for value.
   * @param dimension Dimension to unmap string from.
   */
  template <typename eT = typename PolicyType::MappedType>
  eT UnmapValue(const std::string& string, const size_t dimension);

  /**
   * MapTokens turns vector of strings into numeric variables and puts them
   * into a given matrix. It is uses mapping policy to store categorical values
   * to maps. How it determines whether a value is categorical and how it
   * stores the categorical value into map and replaces with the numerical value
   * all depends on the mapping policy object's MapTokens() funciton.
   *
   * @tparam eT Type of armadillo matrix.
   * @param tokens Vector of variables inside a dimension.
   * @param row Position of the given tokens.
   * @param matrix Matrix to save the data into.
   */
  template <typename eT>
  void MapTokens(const std::vector<std::string>& tokens,
                 size_t& row,
                 arma::Mat<eT>& matrix)
  {
    std::stringstream token;
    for (size_t i = 0; i != tokens.size(); ++i)
    {
      token.str(tokens[i]);
      token>>matrix.at(row, i);
      eT val;
      // if a token is a number or categorical value,
      // but is included in the missingSet, map it to invalidMaps.
      // if the token is just not a number, map it to maps.
      if (missingSet.find(tokens[i]) != std::end(missingSet))
      {
        val = static_cast<eT>(this->MapInvalidValue(tokens[i], row, i));
        Log::Warn << "Invalid value at point " <<i<< ", numerical feature "
            << row << " : " << tokens[i] << std::endl;
        matrix.at(row, i) = val;
      }
      else if (token.fail())
      {
        Log::Warn << "Categorical value at point " <<i<< ", cat feature "
            << row << " : " << tokens[i] << std::endl;
        val = static_cast<eT>(this->MapString(tokens[i], row));
      }
      matrix.at(row, i) = val;
      token.clear();
    }
  }

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
   * Get the dimensionality of the DatasetMapper object (that is, how many
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

  //! Return the policy of the mapper.
  const PolicyType& Policy() const;

  //! Modify the policy of the mapper (be careful!).
  PolicyType& Policy();

  //! Modify (Replace) the policy of the mapper with a new policy
  void Policy(PolicyType&& policy);

  // Note that missingSet and maps are different.
  // missingSet specifies which value/string should be mapped.
  std::set<std::string> missingSet;

 private:
  //! Types of each dimension.
  std::vector<Datatype> types;

  //! maps object stores string and numerical pairs.
  MapType maps;

  ObjectMapType invalidMaps;

  //! policy object tells dataset mapper how the categorical values should be
  //  mapped to the maps object. It is used in MapString() and MapTokens().
  PolicyType policy;

  //! number of invalid values.
  //size_t numInvalidMappings;
  //size_t numMappings;
};

/**
 * Use typedef to provide backward compatibility
 */
using DatasetInfo = DatasetMapper<data::IncrementPolicy>;

} // namespace data
} // namespace mlpack

#include "dataset_mapper_impl.hpp"

#endif
