#include <boost/histogram/detail/zero_suppression.hpp>

namespace boost {
namespace histogram {
namespace detail {

template <>
bool
zero_suppression_encode(std::vector<wtype>& output, const wtype* input,
                        uintptr_t size)
{
    #define BOOST_HISTOGRAM_ZERO_SUPPRESSION_FILL { \
        if ((size - output.size()) < 2) \
            return false;                     \
        output.push_back(0);                  \
        output.push_back(nzero);              \
        nzero = 0;                            \
    }

    const wtype* input_end = input + size;
    uint32_t nzero = 0;
    for (; input != input_end; ++input) {
        if (*input != 0) {
            if (nzero)
                BOOST_HISTOGRAM_ZERO_SUPPRESSION_FILL
            if (output.size() == size)
                return false;
            output.push_back(*input);
        }
        else {
            ++nzero;
            if (nzero == 0) // overflowed to zero
                BOOST_HISTOGRAM_ZERO_SUPPRESSION_FILL
        }
    }
    if (nzero)
        BOOST_HISTOGRAM_ZERO_SUPPRESSION_FILL
    return true;
}

template <>
void
zero_suppression_decode(wtype* output, uintptr_t size,
                        const std::vector<wtype>& input)
{
    const wtype* inp = &input[0];
    const wtype* output_end = output + size;
    while (output != output_end) {
        *output = *inp;
        if (*inp == 0) {
            const uintptr_t nzero = (++inp)->w;
            for (uintptr_t j = 1; j != nzero; ++j) {
                *(++output) = 0;
                if (output == output_end)
                    return;
            }
        }
        ++inp;
        ++output;
    }
}

}
}
}
