//[ getting_started_listing_02

#include <boost/histogram.hpp>
#include <cstdlib>
#include <string>

namespace bh = boost::histogram;

int main() {
  /*
      create a dynamic histogram with the factory `make_dynamic_histogram`
      - axis can be passed directly just like for `make_static_histogram`
      - in addition, the factory also accepts iterators over a sequence of
        axis::any, the polymorphic type that can hold concrete axis types
  */
  std::vector<
    bh::axis::any<
      bh::axis::regular<>,
      bh::axis::category<std::string>
    >
  > axes;
  axes.emplace_back(bh::axis::category<std::string>({"red", "blue"}));
  axes.emplace_back(bh::axis::regular<>(5, 0, 1, "x"));
  axes.emplace_back(bh::axis::regular<>(5, 0, 1, "y"));
  auto h = bh::make_dynamic_histogram(axes.begin(), axes.end());

  // fill histogram with data, usually this happens in a loop
  h("red", 0.1, 0.2);
  h("blue", 0.3, 0.4);
  h("red", 0.5, 0.6);
  h("red", 0.7, 0.8);

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
    for (auto ybin : h.axis(2)) {   // rows
      for (auto xbin : h.axis(1)) { // columns
        const auto v = h.at(cbin, xbin, ybin).value();
        if (v)
          std::printf("%4s [%3.1f, %3.1f) [%3.1f, %3.1f) %3.0f\n",
                      cbin.value().c_str(),
                      xbin.lower(), xbin.upper(),
                      ybin.lower(), ybin.upper(),
                      v);
      }
    }
  }
}

//]
