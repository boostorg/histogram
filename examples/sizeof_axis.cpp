#include <cstdio>
#include <boost/preprocessor.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/container/vector.hpp>
#include <boost/scoped_array.hpp>
#include <vector>
#include <string>
#include <valarray>

int main(int argc, char** argv) {
  using namespace boost::histogram;
  #define SIZEOF(x) std::printf("%32s: %lu\n", BOOST_STRINGIZE(x), sizeof(x))
  SIZEOF(char);
  SIZEOF(int);
  SIZEOF(long);  
  SIZEOF(float);
  SIZEOF(double);
  SIZEOF(void*);
  SIZEOF(std::string);
  SIZEOF(std::vector<double>);
  SIZEOF(std::valarray<double>);
  SIZEOF(boost::container::vector<double>);
  SIZEOF(boost::scoped_array<double>);
  SIZEOF(regular_axis);
  SIZEOF(polar_axis);
  SIZEOF(variable_axis);
  SIZEOF(category_axis);
  SIZEOF(integer_axis);
  SIZEOF(axis_type);
}
