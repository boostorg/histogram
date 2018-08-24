// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_PYTHON_SERIALIZATION_SUITE_HPP_
#define _BOOST_HISTOGRAM_PYTHON_SERIALIZATION_SUITE_HPP_

#include <algorithm>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/assert.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/python.hpp>
#include <cstdlib>
#include <iosfwd>

namespace boost {
namespace histogram {
namespace detail {

#if PY_MAJOR_VERSION < 3
#define PyBytes_FromStringAndSize PyString_FromStringAndSize
#define PyBytes_AS_STRING PyString_AS_STRING
#define PyBytes_Size PyString_Size
#define _PyBytes_Resize _PyString_Resize
#endif

class python_bytes_sink : public iostreams::sink {
public:
  python_bytes_sink(PyObject** pstr) : pstr_(pstr), len_(0), pos_(0) {
    BOOST_ASSERT(*pstr == 0);
  }

  std::streamsize write(const char* s, std::streamsize n) {
    if (len_ == 0) {
      *pstr_ = PyBytes_FromStringAndSize(s, n);
      if (*pstr_ == nullptr) // no point trying to recover from allocation error
        std::terminate();
      len_ = n;
    } else {
      if (pos_ + n > len_) {
        len_ = pos_ + n;
        if (_PyBytes_Resize(pstr_, len_) == -1)
          std::terminate(); // no point trying to recover from allocation error
      }
      char* b = PyBytes_AS_STRING(*pstr_);
      std::copy(s, s + n, b + pos_);
    }
    pos_ += n;
    return n;
  }

private:
  PyObject** pstr_;
  std::streamsize len_, pos_;
};
} // namespace detail

template <class T>
struct serialization_suite : python::pickle_suite {
  static python::tuple getstate(python::object obj) {
    PyObject* pobj = nullptr;
    iostreams::stream<detail::python_bytes_sink> os(&pobj);
    archive::text_oarchive oa(os);
    oa << python::extract<const T&>(obj)();
    os.flush();
    return python::make_tuple(obj.attr("__dict__"),
                              python::object(python::handle<>(pobj)));
  }

  static void setstate(python::object obj, python::tuple state) {
    // restore the object's __dict__
    python::dict d = python::extract<python::dict>(obj.attr("__dict__"));
    d.update(state[0]);

    // restore the C++ object
    python::object o = state[1];
    iostreams::stream<iostreams::array_source> is(PyBytes_AS_STRING(o.ptr()),
                                                  PyBytes_Size(o.ptr()));
    archive::text_iarchive ia(is);
    ia >> python::extract<T&>(obj)();
  }

  static bool getstate_manages_dict() { return true; }
};

#undef PyBytes_FromStringAndSize
#undef PyBytes_AS_STRING
#undef PyBytes_Size
#undef _PyBytes_Resize
} // histogram
} // boost

#endif
