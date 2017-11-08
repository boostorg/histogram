// also see examples/create_dynamic_histogram.cpp
#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

int main() {
	using hist_type = bh::histogram<bh::Dynamic, bh::axis::builtins>;
	auto v = std::vector<bh::axis::any<>>();
	v.push_back(bh::axis::regular<>(100, -1, 1));
	v.push_back(bh::axis::integer<>(1, 7));
	auto h = hist_type(v.begin(), v.end());
	// do something with h
}
