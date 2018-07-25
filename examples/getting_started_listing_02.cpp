//[ getting_started_listing_02

#include <boost/histogram.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cstdlib>
#include <string>

namespace br = boost::random;
namespace bh = boost::histogram;

int main() {
  /*
      create a dynamic histogram with the factory `make_dynamic_histogram`
      - axis can be passed directly just like for `make_static_histogram`
      - in addition, the factory also accepts iterators over a sequence of
        axis::any, the polymorphic type that can hold concrete axis types
  */
  std::vector<bh::axis::any_std> axes;
  axes.emplace_back(bh::axis::category<std::string>({"red", "blue"}));
  axes.emplace_back(bh::axis::regular<>(5, -5, 5, "x"));
  axes.emplace_back(bh::axis::regular<>(5, -5, 5, "y"));
  auto h = bh::make_dynamic_histogram(axes.begin(), axes.end());

  // fill histogram with random numbers
  br::mt19937 gen;
  br::normal_distribution<> norm;
  for (int i = 0; i < 1000; ++i)
    h(i % 2 ? "red" : "blue", norm(gen), norm(gen));

  /*
      print dynamic histogram by iterating over bins
      - for most axis types, the for loop looks just like for a static
        histogram, except that we can pass runtime numbers, too
      - if the [bin type] of the axis is not convertible to a
        double interval, one needs to cast axis::any before looping;
        this is here the case for the category axis
  */
  using cas = bh::axis::category<std::string>;
  for (auto cbin : static_cast<const cas&>(h.axis(0))) {
    std::printf("%s\n", cbin.value().c_str());
    for (auto ybin : h.axis(2)) {   // rows
      for (auto xbin : h.axis(1)) { // columns
        std::printf("%3.0f ", h.at(cbin, xbin, ybin).value());
      }
      std::printf("\n");
    }
  }
}

//]
