#include <cstdio>
#include <boost/preprocessor.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/wtype.hpp>
#include <boost/container/vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/scoped_array.hpp>
#include <boost/multiprecision/cpp_int.hpp>
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
  SIZEOF(detail::wtype);
  SIZEOF(boost::multiprecision::int128_t);
  SIZEOF(boost::multiprecision::int512_t);
  SIZEOF(boost::multiprecision::cpp_int);
  SIZEOF(std::string);
  SIZEOF(std::vector<double>);
  SIZEOF(std::valarray<double>);
  SIZEOF(boost::container::vector<double>);
  typedef boost::container::static_vector<axis_type,16> static_vector_a16;
  SIZEOF(static_vector_a16);
  SIZEOF(boost::scoped_array<double>);
  SIZEOF(regular_axis);
  SIZEOF(polar_axis);
  SIZEOF(variable_axis);
  SIZEOF(category_axis);
  SIZEOF(integer_axis);
  SIZEOF(axis_type);
}
