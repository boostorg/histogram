#include <boost/histogram/detail/zero_suppression.hpp>

namespace boost {
namespace histogram {
namespace detail {

bool
zero_suppression_encode(std::vector<char>& output, std::size_t output_max,
                        const char* input, std::size_t input_len)
{
    #define FILL_OUTPUT {                     \
        if ((output_max - output.size()) < 2) \
            return false;                     \
        output.push_back(0);                  \
        output.push_back(nzero);              \
        nzero = 0;                            \
    }

    const char* input_end = input + input_len;
    unsigned nzero = 0;
    for (; input != input_end; ++input) {
        if (*input != 0) {
            if (nzero)
                FILL_OUTPUT
            if (output.size() == output_max)
                return false;
            output.push_back(*input);
        }
        else {
            ++nzero;
            if (nzero == 256)
                FILL_OUTPUT
        }
    }
    if (nzero)
        FILL_OUTPUT
    return true;
}

void
zero_suppression_decode(char* output, std::size_t output_len,
                        const std::vector<char>& input)
{
    const char* inp = &input[0];
    const char* output_end = output + output_len;
    while (output != output_end) {
        *output = *inp;
        if (*inp == 0) {
            const char nzero = *(++inp);
            for (char j = 1; j != nzero; ++j) {
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
