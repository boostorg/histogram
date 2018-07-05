//[ guide_custom_axis

#include <boost/histogram.hpp>
#include <iostream>

namespace bh = boost::histogram;

// custom axis, which adapts builtin integer axis
struct custom_axis : public bh::axis::integer<> {
    using value_type = const char*; // type that is fed to the axis

    using integer::integer; // inherit ctors of base

    // the customization point
    // - accept const char* and convert to int
    // - then call index method of base class
    int index(value_type s) const {
      return integer::index(std::atoi(s));
    }
};

int main() {
    auto h = bh::make_static_histogram(custom_axis(0, 3));
    h("-10");
    h("0");
    h("1");
    h("9");

    for (auto xi : h.axis()) {
        std::cout << "bin " << xi.idx() << " [" << xi.lower() << ", "
                  << xi.upper() << ") " << h.at(xi).value()
                  << std::endl;
    }

    /* prints:
        bin 0 [0, 1) 1
        bin 1 [1, 2] 1
        bin 2 [2, 3] 0
    */
}

//]
