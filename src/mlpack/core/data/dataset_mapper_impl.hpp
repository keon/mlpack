/**
 * @file dataset_mapper_impl.hpp
 * @author Ryan Curtin
 * @author Keon Kim
 *
 * An implementation of the DatasetMapper<PolicyType> class.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP

// In case it hasn't already been included.
#include "dataset_mapper.hpp"

namespace mlpack {
namespace data {

// Default constructor.
template<typename PolicyType>
inline DatasetMapper<PolicyType>::DatasetMapper(const size_t dimensionality) :
    types(dimensionality, Datatype::numeric)
{
  // Nothing to initialize here.
}

template<typename PolicyType>
inline DatasetMapper<PolicyType>::DatasetMapper(PolicyType& policy,
    const size_t dimensionality) :
    types(dimensionality, Datatype::numeric),
    policy(std::move(policy))
{
  // Nothing to initialize here.
}

// When we want to insert value into the map,
// we could use the policy to map the string
template<typename PolicyType>
inline typename PolicyType::MappedType DatasetMapper<PolicyType>::MapString(
    const std::string& string,
    const size_t dimension)
{
  return policy.template MapString<MapType, ObjectMapType>(string, dimension,
      maps, invalidMaps, types);
}

// Return the string corresponding to a value in a given dimension.
template<typename PolicyType>
inline const std::string& DatasetMapper<PolicyType>::UnmapString(
    const size_t value,
    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.right.count(value) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType>::UnmapString(): value '" << value
        << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.right.at(value);
}

// Return the value corresponding to a string in a given dimension.
template<typename PolicyType>
template<typename eT>
inline eT DatasetMapper<PolicyType>::UnmapValue(
    const std::string& string,
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

//template<typename PolicyType>
//template<typename eT>
//inline void DatasetMapper<PolicyType>::MapTokens(
    //const std::vector<std::string>& tokens,
    //size_t& row,
    //arma::Mat<eT>& matrix)
//{
  ////return policy.template MapTokens<eT, MapType, ObjectMapType>(tokens, row,
      ////matrix, maps, invalidMaps, types);
    //// ValidatePolicy allows double type matrix only, because it uses NaN.
    ////static_assert(std::is_same<eT, double>::value, "You must use double type "
        ////" matrix in order to apply ValidatePolicy");

    //auto notNumber = [](const std::string& str)
    //{
      //eT val(0);
      //std::stringstream token;
      //token.str(str);
      //token >> val;
      //return token.fail();
    //};
  ////template <typename MapType, typename ObjectMapType>
  ////MappedType MapString(const std::string& string,
                       ////const size_t dimension,
                       ////MapType& maps,
                       ////ObjectMapType& invalidMaps,
                       ////std::vector<Datatype>& types,
                       ////const size_t point = 0,
                       ////const bool invalid = false)

    //// determine if all of the values are categorical values
    //const size_t categoricalValues = std::count_if(std::begin(tokens),
                                                   //std::end(tokens), notNumber);

    //if (categoricalValues > tokens.size() / 2)
    //{
      //std::stringstream token;
      //for (size_t i = 0; i != tokens.size(); ++i)
      //{
        //double numeric;
        //token.str(tokens[i]);

        //eT val;
        //if (token >> numeric)
        //{
          //Log::Debug << "TOKEN: Possibly problematic value at point " << i
              //<< ", categorical feature " << row << " : " << tokens[i]
              //<< " (numeric value in categorical feature)"<< std::endl;
          //val = static_cast<eT>(this->MapInvalidValue(tokens[i], row, i));
        //}
        //else if (missingSet.find(tokens[i]) != std::end(missingSet))
        //{
          //Log::Debug << "TOKEN: Invalid value at point " << i
              //<< ", categorical feature "
              //<< row << " : " << tokens[i] << std::endl;
          //val = static_cast<eT>(this->MapInvalidValue(tokens[i], row, i));
        //}
        //else
        //{
          //val = static_cast<eT>(this->MapString(tokens[i], row));
        //}
        //matrix.at(row, i) = val;
        //token.clear();
      //}
    //}
    //else
    //{
      //std::stringstream token;
      //for (size_t i = 0; i != tokens.size(); ++i)
      //{
        //token.str(tokens[i]);
        //token>>matrix.at(row, i);
        //// if the token is not number, map it.
        //// or if token is a number, but is included in the missingSet, map it.
        //if (token.fail() || missingSet.find(tokens[i]) != std::end(missingSet))
        //{
          //const eT val = static_cast<eT>(this->MapInvalidValue(tokens[i], row, i));
          //Log::Warn << "Invalid value at point " <<i<< ", numerical feature "
              //<< row << " : " << tokens[i] << std::endl;
          //matrix.at(row, i) = val;
        //}
        //token.clear();
      //}
    //}

//}

// Get the type of a particular dimension.
template<typename PolicyType>
inline Datatype DatasetMapper<PolicyType>::Type(const size_t dimension) const
{
  if (dimension >= types.size())
  {
    std::ostringstream oss;
    oss << "requested type of dimension " << dimension << ", but dataset only "
        << "has " << types.size() << " dimensions";
    throw std::invalid_argument(oss.str());
  }

  return types[dimension];
}

template<typename PolicyType>
inline Datatype& DatasetMapper<PolicyType>::Type(const size_t dimension)
{
  if (dimension >= types.size())
    types.resize(dimension + 1, Datatype::numeric);

  return types[dimension];
}

template<typename PolicyType>
inline
size_t DatasetMapper<PolicyType>::NumMappings(const size_t dimension) const
{
  return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).second;
}

template<typename PolicyType>
inline size_t DatasetMapper<PolicyType>::Dimensionality() const
{
  return types.size();
}

template<typename PolicyType>
inline const PolicyType& DatasetMapper<PolicyType>::Policy() const
{
  return this->policy;
}

template<typename PolicyType>
inline PolicyType& DatasetMapper<PolicyType>::Policy()
{
  return this->policy;
}

template<typename PolicyType>
inline void DatasetMapper<PolicyType>::Policy(PolicyType&& policy)
{
  this->policy = std::forward<PolicyType>(policy);
}

} // namespace data
} // namespace mlpack

#endif
