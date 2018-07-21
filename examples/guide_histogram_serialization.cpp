//[ guide_histogram_serialization

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp> // includes serialization code
#include <sstream>

namespace bh = boost::histogram;

int main() {
  auto a = bh::make_static_histogram(bh::axis::regular<>(3, -1, 1, "r"),
                                     bh::axis::integer<>(0, 2, "i"));
  a(0.5, 1);

  std::string buf; // holds persistent representation

  // store histogram
  {
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << a;
    buf = os.str();
  }

  auto b = decltype(a)(); // create a default-constructed second histogram

  std::cout << "before restore " << (a == b) << std::endl;
  // prints: before restore 0

  // load histogram
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }

  std::cout << "after restore " << (a == b) << std::endl;
  // prints: after restore 1
}

//]
