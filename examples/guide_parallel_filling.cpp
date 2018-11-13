// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_custom_storage

#include <array>
#include <atomic>
#include <boost/histogram.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <cassert>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

namespace bh = boost::histogram;

template <typename Histogram>
void fill(Histogram& h) {
  for (unsigned i = 0; i < 1000; ++i) { h(i % 10); }
}

/*
  std::atomic has deleted copy ctor, we need to wrap it in a type with a
  potentially unsafe copy ctor. It can be used in a thread-safe way if some
  rules are followed, see below.
*/
template <typename T>
class copyable_atomic : public std::atomic<T> {
public:
  using std::atomic<T>::atomic;

  // this is potentially not thread-safe
  copyable_atomic(const copyable_atomic& rhs) { this->operator=(rhs); }

  // this is potentially not thread-safe
  copyable_atomic& operator=(const copyable_atomic& rhs) {
    if (this != &rhs) { std::atomic<T>::operator=(rhs.load()); }
    return *this;
  }
};

int main() {
  /*
    Create histogram with array<std::atomic<std::size_t>, 100> as counter storage
    for parallel filling in several threads. You cannot use std::vector here,
    because std::atomic types are not copyable.
  */
  auto h = bh::make_histogram_with(std::vector<copyable_atomic<unsigned>>(),
                                   bh::axis::integer<>(0, 10));

  /*
    The histogram storage may not be resized in either thread. This is the case
    if you do not use growing axis types. Some notes regarding std::thread.
    - The templated fill function must be instantiated when passed to std::thread
      that why we pass fill<decltype(h)>.
    - std::thread copies the argument. To avoid filling two copies of the
      histogram, we need to pass it via std::ref.
  */
  std::thread t1(fill<decltype(h)>, std::ref(h));
  std::thread t2(fill<decltype(h)>, std::ref(h));
  t1.join();
  t2.join();

  assert(bh::algorithm::sum(h) == 2000);
}

//]
