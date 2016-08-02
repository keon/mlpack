/**
 * @file preprocess_validate_main.cpp
 * @author Keon Kim
 *
 * a utility that monitors missing variables in a dataset.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/imputer.hpp>
#include <mlpack/core/data/dataset_mapper.hpp>
#include <mlpack/core/data/map_policies/validate_policy.hpp>

PROGRAM_INFO("Validate Data", "This utility takes a dataset and warns user "
    "defined missing variable to another to provide more meaningful analysis "
    "\n\n"
    "The program does not modify the original file, but instead makes a "
    "separate file to save the output data; You can save the output by "
    "specifying the file name with --output_file (-o)."
    "\n\n"
    "For example, if we consider 'NULL' in dimension 0 to be a missing "
    "variable and want to delete whole row containing the NULL in the "
    "column-wise dataset, and save the result to result.csv, we could run"
    "\n\n"
    "$ mlpack_preprocess_imputer -i dataset.csv -o result.csv -m NULL -d 0 \n"
    "> -s listwise_deletion");

PARAM_STRING_IN_REQ("input_file", "File containing data,", "i");
PARAM_STRING_IN("invalid_value", "User defined invalid value", "I", "");
PARAM_FLAG("no_predefined", "Do not use predefined set of invalid values, which"
    "includes 'nan', 'NaN', 'null', 'Null', and 'NULL'.","n");
PARAM_INT_IN("dimension", "the dimension to apply imputation", "d", 0);
PARAM_DOUBLE_IN("minimum", "Minimun threshold. The program will ", "m", 0);
PARAM_DOUBLE_IN("maximum", "Maximum threshold. The program will ", "M", 0);
using namespace mlpack;
using namespace arma;
using namespace std;
using namespace data;

int main(int argc, char** argv)
{
  CLI::ParseCommandLine(argc, argv);

  const string inputFile = CLI::GetParam<string>("input_file");
  const string missingValue = CLI::GetParam<string>("missing_value");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  const double minimum = CLI::GetParam<double>("minimum");
  const double maximum = CLI::GetParam<double>("maximum");

  // The program needs user-defined missing values.
  // Missing values can be any list of strings such as "1", "a", "NULL".
  if (!CLI::HasParam("missing_value"))
  {
    Log::Warn << "--missing_value is not specifiec, the validation will be "
        << "applied to all dimensions."<< endl;
  }

  if (!CLI::HasParam("dimension"))
    Log::Warn << "--dimension is not specified, the imputation will be "
        << "applied to all dimensions."<< endl;

  arma::mat input;
  // Policy tells how the DatasetMapper should map the values.
  std::set<std::string> missingSet;
  if (!CLI::HasParam("no_predefined"))
  {
    missingSet.insert("nan");
    missingSet.insert("NaN");
    missingSet.insert("null");
    missingSet.insert("Null");
    missingSet.insert("NULL");
  }
  missingSet.insert(missingValue);
  ValidatePolicy policy(missingSet);
  using MapperType = DatasetMapper<ValidatePolicy>;
  DatasetMapper<ValidatePolicy> info(policy);
  std::vector<size_t> dirtyDimensions;

  Load(inputFile, input, info, true, true);

  // print how many mapping exist in each dimensions
  for (size_t i = 0; i < input.n_rows; ++i)
  {
    size_t numMappings = info.NumMappings(i);
    Log::Info << numMappings << " mappings in dimension " << i << "."
        << endl;
    if (numMappings > 0)
    {
      dirtyDimensions.push_back(i);
    }
  }

  Timer::Start("validation");
  if (CLI::HasParam("minimum") || CLI::HasParam("maximum"))
  {
    if (CLI::HasParam("dimension"))
    {
      // when --dimension is specified,
      // the program will apply the changes to only the given dimension.
      for (size_t i = 0; i < input.n_cols; ++i)
      {
        if (CLI::HasParam("minimum") && input(dimension, i) < minimum)
        {
          Log::Warn << "Smaller than Minimum" << input(dimension, i)
              << std::endl;
        }
        if (CLI::HasParam("maximum") && input(dimension, i) > maximum)
        {
          Log::Warn << "Larger than Maximum " << input(dimension, i)
              << std::endl;
        }
      }
    }
    else
    {
      // when --dimension is not specified,
      // the program will apply the changes to all dimensions.
      for (size_t i = 0; i < input.n_rows; ++i)
      {
        for (size_t j = 0; j < input.n_cols; ++j)
        {
          if (CLI::HasParam("minimum") && input(i, j) < minimum)
          {
            Log::Warn << "Smaller than Minimum" << input(i, j)
                << std::endl;
          }
          if (CLI::HasParam("maximum") && input(i, j) > maximum)
          {
            Log::Warn << "Larger than Maximum " << input(i, j)
                << std::endl;
          }
        }
      }
    }
  }
  Timer::Stop("validation");
}

