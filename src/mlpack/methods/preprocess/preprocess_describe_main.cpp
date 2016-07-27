/**
 * @file preprocess_describe_main.cpp
 * @author Keon Kim
 *
 * Descriptive Statistics Class and CLI executable.
 */
#include <mlpack/core.hpp>
#include "descriptive_statistics.hpp"

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
PARAM_FLAG("population", "If specified, the program will calculate statistics "
    "assuming the dataset is the population. By default, the program will "
    "assume the dataset as a sample.", "P")

/**
* Calculates the sum of deviations to the Nth Power
*
* @param n power
* @param dimension Dimension
* @return sum of nth power deviations
*/
double SumNthPowerDeviations(const arma::rowvec& input,
    const double& rowMean,
    const size_t Nth) // Degree of Power
{
  double sum = 0;
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    sum += pow(input(i) - rowMean, Nth);
  }
  return sum;
}
/**
 * Calculates Skewness of the given dimension.
 *
 * @param dimension Dimension
 * @return Skewness
 */
double Skewness(const arma::rowvec& input,
    const double& rowStd,
    const double& rowMean,
    const bool population)
{
  double skewness = 0;
  double S3 = pow(rowStd, 3);
  double M3 = SumNthPowerDeviations(input, rowMean, 3);
  double n = input.n_elem;
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
/**
 * Calculates kurtosis of the given dimension.
 *
 * @param n power
 * @return Kurtosis.
 */
double Kurtosis(const arma::rowvec& input,
    const double& rowStd,
    const double& rowMean,
    const bool population)
{
  double kurtosis = 0;
  double M4 = SumNthPowerDeviations(input, rowMean, 4);
  double n = input.n_elem;
  if (population)
  {
    // Calculate Population Excess Kurtosis
    double M2 = SumNthPowerDeviations(input, rowMean, 2);
    kurtosis = n * (M4 / pow(M2, 2)) - 3;
  }
  else
  {
    // Calculate Sample Excess Kurtosis
    double S4 = pow(rowStd, 4);
    double norm3 = (3 * (n-1) * (n-1)) / ((n-2) * (n-3));
    double normC = (n * (n+1))/((n-1) * (n-2) * (n-3));
    double normM = M4 / S4;
    kurtosis = normC * normM - norm3;
  }
  return kurtosis;
}
/**
 * Calculates standard error.
 *
 * @param dimension Dimension
 * @return Standard error.
 */
double StandardError(const arma::rowvec& input, const double rowStd)
{
   return rowStd / sqrt(input.n_elem);
}

int main(int argc, char** argv)
{
  // Parse command line options.
  CLI::ParseCommandLine(argc, argv);
  const string inputFile = CLI::GetParam<string>("input_file");
  const size_t dimension = static_cast<size_t>(CLI::GetParam<int>("dimension"));
  const bool population = CLI::HasParam("population");

  // Load the data
  arma::mat data;
  data::Load(inputFile, data, false, true /*transpose*/);

  Timer::Start("statistics");
  // Headers
  Log::Info << boost::format("%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s\t%-s"
      "\t%-s") % "dim" % "var" % "mean" % "std" % "median" % "min" % "max"
      % "range" % "skew" % "kurt" % "SE" << endl;

  // If the user specified dimension, describe statistics of the given
  // dimension. If it dimension not specified, describe all dimensions.
  if (CLI::HasParam("dimension"))
  {
    arma::rowvec row = data.row(dimension);
    double rowMax = arma::max(row);
    double rowMin = arma::min(row);
    double rowRange = rowMax - rowMin;
    double rowMean = arma::mean(row);
    double rowMedian = arma::median(row);
    double rowVar = arma::var(row, population);
    double rowStd = arma::stddev(row, population);

    Log::Info << boost::format("%-6i\t%-.4f\t%-.4f\t%-.4f\t%-.4f\t%-.4f\t%-.4f"
        "\t%-.4f\t%-.4f\t%-.4f\t%-.4f")
        % dimension
        % rowVar
        % rowMean
        % rowStd
        % rowMedian
        % rowMin
        % rowMax
        % rowRange
        % Skewness(row, rowStd, rowMean, population)
        % Kurtosis(row, rowStd, rowMean, population)
        % StandardError(row, rowStd)
        << endl;
  }
  else
  {
    for (size_t i = 0; i < data.n_rows; ++i)
    {
      arma::rowvec row = data.row(i);
      double rowMax = arma::max(row);
      double rowMin = arma::min(row);
      double rowRange = rowMax - rowMin;
      double rowMean = arma::mean(row);
      double rowMedian = arma::median(row);
      double rowVar = arma::var(row, population);
      double rowStd = arma::stddev(row, population);

      Log::Info << boost::format("%-6i\t%-.4f\t%-.4f\t%-.4f\t%-.4f\t%-.4f\t"
          "%-.4f\t%-.4f\t%-.4f\t%-.4f\t%-.4f")
          % i
          % rowVar
          % rowMean
          % rowStd
          % rowMedian
          % rowMin
          % rowMax
          % rowRange
          % Skewness(row, rowStd, rowMean, population)
          % Kurtosis(row, rowStd, rowMean, population)
          % StandardError(row, rowStd)
          << endl;
    }
  }
  Timer::Stop("statistics");
}

