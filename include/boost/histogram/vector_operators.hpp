#ifndef _BOOST_HISTOGRAM_VECTOR_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_VECTOR_OPERATORS_HPP_

#include <stdexcept>

namespace boost {
namespace histogram {

    // vectors: generic ==
    template<typename T>
    bool operator==(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size())
            return false;
        // pointer arithmetric is faster
        const T* da = &a.front();
        const T* db = &b.front();
        for (unsigned i = 0, n = a.size(); i < n; ++i)
            if (da[i] != db[i])
                return false;
        return true;
    }

    // vectors: generic +=
    template<typename T>
    std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size())
            throw std::invalid_argument("sizes do not match");
        // pointer arithmetric is faster
        T* da = &a.front();
        const T* db = &b.front();
        for (unsigned i = 0, n = a.size(); i < n; ++i)
            da[i] += db[i];
        return a;
    }

    // vectors: generic -=
    template<typename T>
    std::vector<T>& operator-=(std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size())
            throw std::invalid_argument("sizes do not match");
        // pointer arithmetric is faster
        T* da = &a.front();
        const T* db = &b.front();
        for (unsigned i = 0, n = a.size(); i < n; ++i)
            da[i] -= db[i];
        return a;
    }

}
}

#endif
