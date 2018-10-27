//[ guide_histogram_serialization

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp> // includes serialization code
#include <sstream>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  auto a = bh::make_histogram(bh::axis::regular<>(3, -1, 1, "axis 0"),
                              bh::axis::integer<>(0, 2, "axis 1"));
  a(0.5, 1);

  std::string buf; // to hold persistent representation

  // store histogram
  {
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << a;
    buf = os.str();
  }

  auto b = decltype(a)(); // create a default-constructed second histogram

  assert(b != a); // b is empty, a is not

  // load histogram
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }

  assert(b == a); // now b is equal to a
}

//]
