#ifndef _BOOST_HISTOGRAM_DETAIL_ZERO_SUPPRESSION_HPP_
#define _BOOST_HISTOGRAM_DETAIL_ZERO_SUPPRESSION_HPP_

#include <cstddef>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

bool
zero_suppression_encode(std::vector<char>& output, std::size_t output_max,
                        const char* input, std::size_t input_len);

void
zero_suppression_decode(char* output, std::size_t output_len,
                        const std::vector<char>& input);

}
}
}

#endif
