/**
 * @file validate_policy.hpp
 * @author Keon Kim
 *
 * Dataset validation map policy for dataset info.
 */
#ifndef MLPACK_CORE_DATA_MAP_POLICIES_VALIDATE_POLICY_HPP
#define MLPACK_CORE_DATA_MAP_POLICIES_VALIDATE_POLICY_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>
#include <mlpack/core/data/map_policies/datatype.hpp>
#include <limits>

namespace mlpack {
namespace data {
/**
 * ValidatePolicy is used as a helper class for DatasetMapper. It tells how the
 * strings should be mapped. Purpose of this policy is to map all user-defined
 * missing variables into maps so that users can decide what to do with the
 * corrupted data. User-defined missing variables are given by the missingSet.
 * Note that ValidatePolicy does not change type of features.
 */
class ValidatePolicy
{
 public:
  // typedef of MappedType
  // This policy maps strings to coordinates, pair of row and column coordinates
  // of each invalid data.
  using MappedType = double;
  using MappedObjectType = std::pair<size_t, size_t>;

  ValidatePolicy()
  {
    // Nothing to initialize here.
  }

  /**
   * Create the ValidatePolicy object with the given missingSet. Note that the
   * missingSet cannot be changed later; you will have to create a new
   * ValidatePolicy object.
   *
   * @param missingSet Set of strings that should be mapped.
   */
  explicit ValidatePolicy(std::set<std::string> missingSet) :
      missingSet(std::move(missingSet))
  {
    // Nothing to initialize here.
  }

  /**
   * Given the string and the dimension to which it belongs by the user, and
   * the maps and types given by the DatasetMapper class, returns its numeric
   * mapping. If no mapping yet exists and the string is included in the
   * missingSet, the string is added to the list of mappings for the given
   * dimension. This function is used as a helper function for DatasetMapper
   * class.
   *
   * @tparam MapType Type of unordered_map that contains mapped value pairs
   * @param string String to find/create mapping for.
   * @param dimension Index of the dimension of the string.
   * @param maps Unordered map given by the DatasetMapper.
   * @param types Vector containing the type information about each dimensions.
   */
  template <typename MapType, typename ObjectMapType>
  MappedType MapString(const std::string& string,
                       const size_t dimension,
                       MapType& maps,
                       ObjectMapType& invalidMaps,
                       std::vector<Datatype>& types,
                       const size_t point = 0,
                       const bool invalid = false)
  {
    // If this condition is true, either we have no mapping for the given string
    // or we have no mappings for the given dimension at all.  In either case,
    // we create a mapping.
    Log::Debug << "MapString coordinates: dimension " << dimension << ", point:"
        << point << std::endl;
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (!invalid && (maps.count(dimension) == 0 ||
        maps[dimension].first.left.count(string) == 0))
    {
      Log::Debug << "<INCREMENT MAPPING>" << std::endl;
      // This string does not exist yet.
      size_t& numMappings = maps[dimension].second;

      // change type of the feature to categorical
      if (numMappings == 0)
        types[dimension] = Datatype::categorical;

      typedef boost::bimap<std::string, MappedType>::value_type PairType;
      maps[dimension].first.insert(PairType(string, numMappings));
      return numMappings++;
    }
    if (invalid)
    {
      Log::Debug << "<INVALID MAPPING>" << std::endl;
      // This string does not exist yet.
      typedef boost::bimap<std::string, MappedObjectType>::value_type PairType;
      MappedObjectType coordinates(dimension, point);
      invalidMaps[dimension].first.insert(PairType(string, coordinates));

      size_t& numMappings = invalidMaps[dimension].second;
      ++numMappings;
      return NaN;
    }
    else
    {
      Log::Debug << "<NOT CATEGORIZED MAPPING OCCURED>" << std::endl;
      // This string already exists in the mapping or not included in
      // the missingSet.
      return maps[dimension].first.left.at(string);
    }
  }

  /**
   * MapTokens turns vector of strings into numeric variables and puts them
   * into a given matrix. It is used as a helper function when trying to load
   * files. Each dimension's tokens are given in to this function. If one of the
   * tokens turns out to be a string or one of the missingSet's variables, only
   * the token responsible for it should be mapped using the MapString()
   * funciton.
   *
   * @tparam eT Type of armadillo matrix.
   * @tparam MapType Type of unordered_map that contains mapped value pairs.
   * @param tokens Vector of variables inside a dimension.
   * @param row Position of the given tokens.
   * @param matrix Matrix to save the data into.
   * @param maps Maps given by the DatasetMapper class.
   * @param types Types of each dimensions given by the DatasetMapper class.
   */
  template <typename eT, typename MapType, typename ObjectMapType>
  void MapTokens(const std::vector<std::string>& tokens,
                 size_t& row,
                 arma::Mat<eT>& matrix,
                 MapType& maps,
                 ObjectMapType& invalidMaps,
                 std::vector<Datatype>& types)
  {
    // ValidatePolicy allows double type matrix only, because it uses NaN.
    static_assert(std::is_same<eT, double>::value, "You must use double type "
        " matrix in order to apply ValidatePolicy");

    auto notNumber = [](const std::string& str)
    {
      eT val(0);
      std::stringstream token;
      token.str(str);
      token >> val;
      return token.fail();
    };

    // determine if all of the values are categorical values
    const size_t categoricalValues = std::count_if(std::begin(tokens),
                                                   std::end(tokens), notNumber);

    if (categoricalValues > tokens.size() / 2)
    {
      std::stringstream token;
      for (size_t i = 0; i != tokens.size(); ++i)
      {
        double numeric;
        token.str(tokens[i]);

        eT val;
        if (token >> numeric)
        {
          Log::Debug << "TOKEN: Possibly problematic value at point " << i
              << ", categorical feature " << row << " : " << tokens[i]
              << " (numeric value in categorical feature)"<< std::endl;
          val = static_cast<eT>(this->MapString(tokens[i], row, maps,
              invalidMaps, types, i, true));
        }
        else if (missingSet.find(tokens[i]) != std::end(missingSet))
        {
          Log::Debug << "TOKEN: Invalid value at point " << i
              << ", categorical feature "
              << row << " : " << tokens[i] << std::endl;
          val = static_cast<eT>(this->MapString(tokens[i], row, maps,
              invalidMaps, types, i, true));
        }
        else
        {
          val = static_cast<eT>(this->MapString(tokens[i], row, maps,
              invalidMaps, types, i, false));
        }
        matrix.at(row, i) = val;
        token.clear();
      }
    }
    else
    {
      std::stringstream token;
      for (size_t i = 0; i != tokens.size(); ++i)
      {
        token.str(tokens[i]);
        token>>matrix.at(row, i);
        // if the token is not number, map it.
        // or if token is a number, but is included in the missingSet, map it.
        if (token.fail() || missingSet.find(tokens[i]) != std::end(missingSet))
        {
          const eT val = static_cast<eT>(this->MapString(tokens[i], row, maps,
                invalidMaps, types, i, true));
          Log::Warn << "Invalid value at point " <<i<< ", numerical feature "
              << row << " : " << tokens[i] << std::endl;
          matrix.at(row, i) = val;
        }
        token.clear();
      }
    }
  }

 private:
  // Note that missingSet and maps are different.
  // missingSet specifies which value/string should be mapped.
  std::set<std::string> missingSet;
}; // class ValidatePolicy

} // namespace data
} // namespace mlpack

#endif
