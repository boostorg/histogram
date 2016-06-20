// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_ZERO_SUPPRESSION_HPP_
#define _BOOST_HISTOGRAM_DETAIL_ZERO_SUPPRESSION_HPP_

#include <boost/cstdint.hpp>
#include <boost/histogram/detail/wtype.hpp>
#include <limits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <typename T>
bool
zero_suppression_encode(std::vector<T>& output, const T* input,
                        uintptr_t size)
{
    const T* input_end = input + size;
    T nzero = 0;
    for (; input != input_end; ++input) {
        if (*input != 0) {
            if (nzero) {
				if ((size - output.size()) < 2)
					return false;
				output.push_back(0);
				output.push_back(nzero);
				nzero = 0;
            }
            if (output.size() == size)
                return false;
            output.push_back(*input);
        }
        else {
            ++nzero;
            if (nzero == 0) { // overflowed to zero
					if ((size - output.size()) < 2)
						return false;
					output.push_back(0);
					output.push_back(nzero);
					nzero = 0;
				}
        }
    }
    if (nzero){
		if ((size - output.size()) < 2)
			return false;
		output.push_back(0);
		output.push_back(nzero);
		nzero = 0;
	}
    return true;
}

template <typename T>
void
zero_suppression_decode(T* output, uintptr_t size,
                        const std::vector<T>& input)
{
    const T* inp = &input[0];
    const T* output_end = output + size;
    while (output != output_end) {
        *output = *inp;
        if (*inp == 0) {
            const uintptr_t nzero = *(++inp);
            for (T j = 1; j != nzero; ++j) {
                *(++output) = 0;
                if (output == output_end)
                    return;
            }
        }
        ++inp;
        ++output;
    }
}

template <>
void
zero_suppression_decode(wtype* output, uintptr_t size,
                        const std::vector<wtype>& input);

template <>
bool
zero_suppression_encode(std::vector<wtype>& output, const wtype* input,
                        uintptr_t size);

}
}
}

#endif
