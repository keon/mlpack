/**
 * @file preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and CLI executable.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/data/statistics.hpp>

#include <boost/format.hpp>

using namespace mlpack;
using namespace mlpack::data;
using namespace std;
using namespace boost;

PROGRAM_INFO("Descriptive Statistics", "This utility takes a dataset prints "
    "out the statistical facts about the data.");

// Define parameters for data.
PARAM_STRING_REQ("input_file", "File containing data,", "i");
PARAM_INT("dimension", "Dimension of the data", "d", 0);
PARAM_INT("precision", "preferred precision of the result", "p", 2);

/**
 * Make sure a CSV is loaded correctly.
 */
int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t dimension = (size_t) CLI::GetParam<int>("dimension");
  const size_t precision = (size_t) CLI::GetParam<int>("precision");

  // Load the data
  arma::mat data;
  data::Load(inputFile, data);

  Statistics<double> stats(data);

  // Headers
  Log::Info << boost::format("%-s	%-s	%-s	%-s	%-s	%-s	%-s	%-s	%-s	%-s	%-s")
      % "dim"
      % "var"
      % "mean"
      % "std"
      % "median"
      % "min"
      % "max"
      % "range"
      % "skew"
      % "kurt"
      % "SE"
      << endl;

  // If the user specified dimension, describe statistics of the given
  // dimension. If it dimension not specified, describe all dimensions.
  if (CLI::HasParam("dimension"))
  {
    Log::Info << boost::format("%-6i	%-.4f	%-.4f	%-.4f	%-.4f	%-.4f	%-.4f	%-.4f"
        "	%-.4f	%-.4f	%-.4f")
        % dimension
        % stats.Variance(dimension)
        % stats.Mean(dimension)
        % stats.StandardDeviation(dimension)
        % stats.Median(dimension)
        % stats.Min(dimension)
        % stats.Max(dimension)
        % stats.Range(dimension)
        % stats.Skewness(dimension)
        % stats.Kurtosis(dimension)
        % stats.StandardError(dimension)
        << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      Log::Info << boost::format("%-6i	%-.4f	%-.4f	%-.4f	%-.4f	%-.4f	%-.4f	"
          "%-.4f	%-.4f	%-.4f	%-.4f")
          % i
          % stats.Variance(i)
          % stats.Mean(i)
          % stats.StandardDeviation(i)
          % stats.Median(i)
          % stats.Min(i)
          % stats.Max(i)
          % stats.Range(i)
          % stats.Skewness(i)
          % stats.Kurtosis(i)
          % stats.StandardError(i)
          << endl;
    }
  }
}

