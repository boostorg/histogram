//[ getting_started_listing_02

#include <boost/histogram.hpp>
#include <cstdlib>
#include <string>

namespace bh = boost::histogram;

int main() {
  /*
      Create a dynamic histogram with the factory `make_dynamic_histogram`.
      - axis can be passed directly just like for `make_static_histogram`
      - in addition, the factory also accepts iterators over a sequence of
        axis::variant, the polymorphic type that can hold concrete axis types
  */
  std::vector<bh::axis::variant<bh::axis::regular<>, bh::axis::category<std::string> > >
      axes;
  axes.emplace_back(bh::axis::category<std::string>({"red", "blue"}));
  axes.emplace_back(bh::axis::regular<>(3, 0, 1, "x"));
  axes.emplace_back(bh::axis::regular<>(3, 0, 1, "y"));
  auto h = bh::make_histogram(axes.begin(), axes.end());

  // fill histogram with data, usually this happens in a loop
  h("red", 0.1, 0.2);
  h("blue", 0.7, 0.3);
  h("red", 0.3, 0.7);
  h("red", 0.7, 0.7);

  /*
      Print dynamic histogram by iterating over bins.
      If the [bin type] of the axis is not convertible to a
      double interval, you need to cast axis::variant before looping;
      this is here the case for the category axis.
  */
  using cas = bh::axis::category<std::string>;
  for (auto cbin : bh::axis::get<cas>(h.axis(0))) {
    for (auto ybin : h.axis(2)) {   // rows
      for (auto xbin : h.axis(1)) { // columns
        const auto v = h.at(cbin, xbin, ybin);
        if (v)
          std::printf("(%i, %i, %i) %4s [%3.1f, %3.1f) [%3.1f, %3.1f) %3.0f\n",
                      cbin.idx(), xbin.idx(), ybin.idx(), cbin.value().c_str(),
                      xbin.lower(), xbin.upper(), ybin.lower(), ybin.upper(), v);
      }
    }
  }

  assert(h.at(0, 0, 0) == 1);
  assert(h.at(0, 0, 2) == 1);
  assert(h.at(0, 2, 2) == 1);
  assert(h.at(1, 2, 0) == 1);
}

//]
