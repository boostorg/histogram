//[ guide_custom_modified_axis

#include <boost/histogram.hpp>
#include <sstream>
#include <cassert>

namespace bh = boost::histogram;

// custom axis, which adapts builtin integer axis
struct custom_axis : public bh::axis::integer<> {
  using value_type = const char*; // type that is fed to the axis

  using integer::integer; // inherit ctors of base

  // the customization point
  // - accept const char* and convert to int
  // - then call index method of base class
  int operator()(value_type s) const { return integer::operator()(std::atoi(s)); }
};

int main() {
  auto h = bh::make_histogram(custom_axis(3, 6));
  h("-10");
  h("3");
  h("4");
  h("9");

  std::ostringstream os;
  for (auto xi : h.axis()) {
    os << "bin " << xi.idx()
       << " [" << xi.lower() << ", " << xi.upper() << ") "
       << h.at(xi) << "\n";
  }

  std::cout << os.str() << std::endl;

  assert(os.str() ==
         "bin 0 [3, 4) 1\n"
         "bin 1 [4, 5) 1\n"
         "bin 2 [5, 6) 0\n");
}

//]
