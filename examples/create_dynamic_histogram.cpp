#include <boost/histogram.hpp>
#include <boost/histogram/histogram_ostream_operators.hpp>
#include <vector>
#include <cstdlib>

namespace bh = boost::histogram;

/*
  command line usage: cmd [i a b|r n x y]
    a,b,n: integers
    x,y  : floats
*/

int main(int argc, char** argv) {
    using Histogram = bh::histogram<bh::Dynamic, bh::builtin_axes>;

    auto v = std::vector<Histogram::any_axis_type>();

    // parse arguments
    auto argi = 1;
    while (argi < argc) {
        switch (argv[argi][0]) {
            case 'i': {
                ++argi;
                auto a = std::atoi(argv[argi]);
                ++argi;
                auto b = std::atoi(argv[argi]);
                ++argi;
                v.push_back(bh::axis::integer<>(a, b));
            }
            break;
            case 'r': {
                ++argi;
                auto n = std::atoi(argv[argi]);
                ++argi;
                auto x = std::atof(argv[argi]);
                ++argi;
                auto y = std::atof(argv[argi]);
                ++argi;
                v.push_back(bh::axis::regular<>(n, x, y));
            }
            break;
            default:
                std::cerr << "unknown argument " << argv[argi] << std::endl;
                return 1;
        }
    }

    auto h = Histogram(v.begin(), v.end());

    // do something with h
    std::cout << "you created the following histogram:\n" << h << std::endl;
}
