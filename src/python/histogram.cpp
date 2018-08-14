// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/mp11.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#include "serialization_suite.hpp"
#include "utility.hpp"
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#endif
#include <memory>

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
#define BOOST_HISTOGRAM_AXIS_LIMIT 16
#endif

namespace bh = boost::histogram;
namespace bp = boost::python;
namespace mp11 = boost::mp11;

using pyhistogram = bh::histogram<>;

#ifdef HAVE_NUMPY
namespace boost {
namespace histogram {
class python_access {
public:
  using mp_int = detail::mp_int;
  using wcount = detail::wcount;

  struct dtype_visitor {
    template <typename T, typename Buffer>
    bp::str operator()(T*, const Buffer&, bp::list&, bp::list& strides) {
      strides.append(sizeof(T));
      return bp::dtype_typestr<T>();
    }
    template <typename Buffer>
    bp::str operator()(void*, const Buffer&, bp::list&, bp::list& strides) {
      strides.append(sizeof(uint8_t));
      return bp::dtype_typestr<uint8_t>();
    }
    template <typename Buffer>
    bp::str operator()(mp_int*, const Buffer&, bp::list&, bp::list& strides) {
      strides.append(sizeof(double));
      return bp::dtype_typestr<double>();
    }
    template <typename Buffer>
    bp::str operator()(wcount*, const Buffer&, bp::list& shapes, bp::list& strides) {
      strides.append(sizeof(double));
      strides.append(strides[-1] * 2);
      shapes.append(2);
      return bp::dtype_typestr<double>();
    }
  };

  struct data_visitor {
    template <typename T, typename Buffer>
    bp::object operator()(T* tp, const Buffer&, const bp::list&, const bp::list&) const {
      return bp::make_tuple(reinterpret_cast<uintptr_t>(tp), true);
    }
    template <typename Buffer>
    bp::object operator()(void*, const Buffer&, const bp::list& shapes,
                          const bp::list&) const {
      // cannot pass non-existent memory to numpy; make new
      // zero-initialized uint8 array, and pass it
      return np::zeros(bp::tuple(shapes), np::dtype::get_builtin<uint8_t>());
    }
    template <typename Buffer>
    bp::object operator()(mp_int* tp, const Buffer& b, const bp::list& shapes,
                          const bp::list& strides) const {
      // cannot pass cpp_int to numpy; make new
      // double array, fill it and pass it
      auto a = np::empty(bp::tuple(shapes), np::dtype::get_builtin<double>());
      for (std::size_t i = 0, n = bp::len(shapes); i < n; ++i)
        const_cast<Py_intptr_t*>(a.get_strides())[i] = bp::extract<int>(strides[i]);
      auto* buf = (double*)a.get_data();
      for (std::size_t i = 0; i < b.size; ++i) { buf[i] = static_cast<double>(tp[i]); }
      return a;
    }
  };

  static bp::object array_interface(const pyhistogram& self) {
    bp::dict d;
    bp::list shapes;
    bp::list strides;
    auto& b = self.storage_.buffer_;
    d["typestr"] = bh::detail::apply(dtype_visitor(), b, shapes, strides);
    for (std::size_t i = 0; i < self.dim(); ++i) {
      if (i > 0) strides.append(strides[-1] * shapes[-1]);
      shapes.append(self.axis(i).shape());
    }
    if (self.dim() == 0) shapes.append(0);
    d["shape"] = bp::tuple(shapes);
    d["strides"] = bp::tuple(strides);
    d["data"] = bh::detail::apply(data_visitor(), b, shapes, strides);
    return d;
  }
};
} // namespace histogram
} // namespace boost
#endif

struct axis_visitor : public boost::static_visitor<bp::object> {
  template <typename T>
  bp::object operator()(const T& t) const {
    return bp::object(t);
  }
};

struct axes_appender {
  bp::object obj;
  pyhistogram::axes_type& axes;
  bool& success;
  axes_appender(bp::object o, pyhistogram::axes_type& a, bool& s)
      : obj(o), axes(a), success(s) {}
  template <typename A>
  void operator()(const A&) const {
    if (success) return;
    bp::extract<const A&> get_axis(obj);
    if (get_axis.check()) {
      axes.emplace_back(get_axis());
      success = true;
    }
  }
};

bp::object histogram_axis(const pyhistogram& self, int i) {
  if (i < 0) i += self.dim();
  if (i < 0 || i >= int(self.dim())) throw std::out_of_range("axis index out of range");
  return boost::apply_visitor(axis_visitor(), self.axis(i));
}

bp::object histogram_init(bp::tuple args, bp::dict kwargs) {

  bp::object self = args[0];

  if (kwargs) { throw std::invalid_argument("no keyword arguments allowed"); }

  const unsigned dim = bp::len(args) - 1;

  // normal constructor
  pyhistogram::axes_type axes;
  for (unsigned i = 0; i < dim; ++i) {
    bp::object pa = args[i + 1];
    bool success = false;
    boost::mp11::mp_for_each<bh::axis::any_std>(axes_appender(pa, axes, success));
    if (!success) {
      std::string msg = "require an axis object, got ";
      msg += bp::extract<std::string>(bp::str(pa));
      PyErr_SetString(PyExc_TypeError, msg.c_str());
      bp::throw_error_already_set();
    }
  }
  pyhistogram h(axes, typename pyhistogram::storage_type());
  return self.attr("__init__")(std::move(h));
}

template <typename T>
struct fetcher {
  long n = -1;
  union {
    T value = 0;
    const T* carray;
  };
  bp::object keep_alive;

  void assign(bp::object o) {
    // skipping check for currently held type, since it is always value
    bp::extract<T> get_value(o);
    if (get_value.check()) {
      value = get_value();
      n = 0;
      return;
    }
#ifdef HAVE_NUMPY
    np::ndarray a = np::from_object(o, np::dtype::get_builtin<T>(), 1);
    carray = reinterpret_cast<const T*>(a.get_data());
    n = a.shape(0);
    keep_alive = a; // this may be a temporary object
    return;
#endif
    throw std::invalid_argument("argument must be a number");
  }

  const T& operator[](long i) const noexcept {
    if (n > 0) return carray[i];
    return value;
  }
};

template <typename T>
struct span {
  T* data;
  std::size_t size;
  const T* begin() const { return data; }
  const T* end() const { return data + size; }
};

bp::object histogram_call(bp::tuple args, bp::dict kwargs) {
  const auto nargs = bp::len(args);
  pyhistogram& self = bp::extract<pyhistogram&>(args[0]);

  const unsigned dim = nargs - 1;
  if (dim != self.dim()) {
    throw std::invalid_argument("number of arguments and dimension do not match");
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    throw std::invalid_argument(
        bh::detail::cat("too many arguments, maximum is ", BOOST_HISTOGRAM_AXIS_LIMIT)
            .c_str());
  }

  fetcher<double> fetch[BOOST_HISTOGRAM_AXIS_LIMIT];
  long n = 0;
  for (auto d = 0u; d < dim; ++d) {
    fetch[d].assign(args[1 + d]);
    if (fetch[d].n > 0) {
      if (n > 0 && fetch[d].n != n) {
        throw std::invalid_argument("lengths of sequences do not match");
      }
      n = fetch[d].n;
    }
  }

  fetcher<double> fetch_weight;
  const auto nkwargs = bp::len(kwargs);
  if (nkwargs > 0) {
    const bool use_weight = kwargs.has_key("weight");
    if (nkwargs > use_weight) { // only one keyword allowed: weight
      throw std::invalid_argument("only keyword weight allowed");
    }

    if (use_weight) {
      fetch_weight.assign(kwargs.get("weight"));
      if (fetch_weight.n > 0) {
        if (n > 0 && fetch_weight.n != n) {
          throw std::invalid_argument("length of weight sequence does not match");
        }
        n = fetch_weight.n;
      }
    }
  }

  if (!n) ++n;
  if (dim == 1) {
    if (fetch_weight.n >= 0) {
      for (auto i = 0l; i < n; ++i) self(bh::weight(fetch_weight[i]), fetch[0][i]);
    } else {
      for (auto i = 0l; i < n; ++i) self(fetch[0][i]);
    }
  } else {
    double v[BOOST_HISTOGRAM_AXIS_LIMIT];
    if (fetch_weight.n >= 0) {
      for (auto i = 0l; i < n; ++i) {
        for (auto d = 0u; d < dim; ++d) v[d] = fetch[d][i];
        self(bh::weight(fetch_weight[i]), span<double>{v, dim});
      }
    } else {
      for (auto i = 0l; i < n; ++i) {
        for (auto d = 0u; d < dim; ++d) v[d] = fetch[d][i];
        self(span<double>{v, dim});
      }
    }
  }

  return bp::object();
}

bp::object histogram_getitem(const pyhistogram& self, bp::object args) {
  bp::extract<int> get_int(args);
  if (get_int.check()) {
    if (self.dim() == 1) { return bp::object(self[get_int()]); }
    throw std::invalid_argument("wrong number of arguments");
  }

  const unsigned dim = bp::len(args);
  if (self.dim() != dim) { throw std::invalid_argument("wrong number of arguments"); }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    throw std::invalid_argument(
        bh::detail::cat("too many arguments, maximum is ", BOOST_HISTOGRAM_AXIS_LIMIT)
            .c_str());
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (std::size_t i = 0; i < dim; ++i) idx[i] = bp::extract<int>(args[i]);

  return bp::object(self.at(span<int>{idx, self.dim()}));
}

bp::object histogram_at(bp::tuple args, bp::dict kwargs) {
  if (kwargs) { throw std::invalid_argument("no keyword arguments allowed"); }
  const pyhistogram& self = bp::extract<const pyhistogram&>(args[0]);

  bp::object a = args.slice(1, bp::_);
  return histogram_getitem(self, bp::extract<bp::tuple>(a));
}

bp::object histogram_reduce_to(bp::tuple args, bp::dict kwargs) {
  if (kwargs) { throw std::invalid_argument("no keyword arguments allowed"); }
  const pyhistogram& self = bp::extract<const pyhistogram&>(args[0]);

  const auto nargs = bp::len(args) - 1;
  if (nargs == 0) { throw std::invalid_argument("at least one argument required"); }

  if (nargs > BOOST_HISTOGRAM_AXIS_LIMIT) {
    throw std::invalid_argument(
        bh::detail::cat("too many arguments, maximum is ", BOOST_HISTOGRAM_AXIS_LIMIT)
            .c_str());
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (auto i = 0u; i < nargs; ++i) {
    idx[i] = bp::extract<int>(args[1 + i]);
    if (i > 0 && idx[i] <= idx[i - 1])
      throw std::invalid_argument("indices must be strictly ascending");
  }

  return bp::object(self.reduce_to(idx, idx + nargs));
}

std::string histogram_repr(const pyhistogram& h) { return bh::detail::cat(h); }

double element_value(const pyhistogram::element_type& b) { return b.value(); }

double element_variance(const pyhistogram::element_type& b) { return b.variance(); }

double element_getitem(const pyhistogram::element_type& e, int i) {
  if (i < 0 || i > 1) throw std::out_of_range("element_getitem");
  return i == 0 ? e.value() : e.variance();
}

int element_len(const pyhistogram::element_type&) { return 2; }

std::string element_repr(const pyhistogram::element_type& e) {
  return bh::detail::cat("histogram.element(", e.value(), ", ", e.variance(), ")");
}

void register_histogram() {
  bp::docstring_options dopt(true, true, false);

  bp::scope s =
      bp::class_<pyhistogram, boost::shared_ptr<pyhistogram>>(
          "histogram", "N-dimensional histogram for real-valued data.", bp::no_init)
          .def("__init__", bp::raw_function(histogram_init),
               ":param axis args: axis objects"
               "\nPass one or more axis objects to configure the histogram.")
          // shadowed C++ ctors
          .def(bp::init<const pyhistogram&>())
// .def(bp::init<pyhistogram &&>()) move-ctor doesn't work with boost.python
#ifdef HAVE_NUMPY
          .add_property("__array_interface__", &bh::python_access::array_interface)
#endif
          .add_property("dim", &pyhistogram::dim)
          .def("axis", histogram_axis, bp::arg("i") = 0,
               ":param int i: axis index"
               "\n:return: corresponding axis object")
          .def("__call__", bp::raw_function(histogram_call),
               ":param double args: values (number must match dimension)"
               "\n:keyword double weight: optional weight"
               "\n"
               "\nIf Numpy support is enabled, 1d-arrays can be passed "
               "instead of"
               "\nvalues, which must be equal in lenght. Arrays and values can"
               "\nbe mixed arbitrarily in the same call.")
          .def("__len__", &pyhistogram::size,
               ":return: total number of bins, including under- and overflow")
          .def("at", bp::raw_function(histogram_at),
               ":param int args: indices of the bin (number must match "
               "dimension)"
               "\n:return: bin content")
          .def("__getitem__", histogram_getitem,
               ":param int args: indices of the bin (number must match "
               "dimension)"
               "\n:return: bin content")
          .def("reduce_to", bp::raw_function(histogram_reduce_to),
               ":param int args: indices of the axes in the reduced histogram"
               "\n:return: reduced histogram with subset of axes")
          .def("__iter__", bp::iterator<const pyhistogram>())
          .def("__repr__", histogram_repr,
               ":return: string representation of the histogram")
          .def(bp::self == bp::self)
          .def(bp::self != bp::self)
          .def(bp::self += bp::self)
          .def(bp::self *= double())
          .def(bp::self * double())
          .def(double() * bp::self)
          .def(bp::self + bp::self)
          .def_pickle(bh::serialization_suite<pyhistogram>());

  bp::class_<pyhistogram::element_type>(
      "element", "Holds value and variance of bin count.", bp::init<double, double>())
      .add_property("value", element_value)
      .add_property("variance", element_variance)
      .def("__getitem__", element_getitem)
      .def("__len__", element_len)
      .def(bp::self == bp::self)
      .def(bp::self != bp::self)
      .def(bp::self += bp::self)
      .def(bp::self += double())
      .def(bp::self + bp::self)
      .def(bp::self + double())
      .def(double() + bp::self)
      .def("__repr__", element_repr);
}
