//[ guide_make_dynamic_histogram

#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
    // create vector of axes, axis::any is a polymorphic axis type
    auto v = std::vector<bh::axis::any<>>();
    v.push_back(bh::axis::regular<>(100, -1, 1));
    v.push_back(bh::axis::integer<>(1, 7));

    // create dynamic histogram (make_static_histogram be used with iterators)
    auto h = bh::make_dynamic_histogram(v.begin(), v.end());

    // do something with h
}

//]
