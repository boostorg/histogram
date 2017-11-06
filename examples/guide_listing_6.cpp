// also see examples/create_dynamic_histogram.cpp
#include <boost/histogram.hpp>
#include <algorithm>
#include <vector>

namespace bh = boost::histogram;

int main() {
	auto h = bh::make_static_histogram(bh::axis::integer<>(0, 9));
	std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	std::for_each(v.begin(), v.end(), [&h](int x) { h.fill(x, bh::weight(2.0)); });
	// h is now filled
}
