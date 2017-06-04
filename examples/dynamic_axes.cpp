#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
    using dhist = decltype(bh::make_dynamic_histogram());
    auto v = std::vector<dhist::axis_type>();
    v.push_back(bh::axis::regular<>(100, -1, 1));
    v.push_back(bh::axis::integer(1, 6));
    auto h = dhist(v.begin(), v.end());
    // do something with h
}
