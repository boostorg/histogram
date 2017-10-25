// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "serialization_suite.hpp"
#include <boost/histogram/axis.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/histogram_ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#ifdef HAVE_NUMPY
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL boost_histogram_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
#define BOOST_HISTOGRAM_AXIS_LIMIT 32
#endif

namespace boost {

namespace histogram {
using dynamic_histogram = histogram<Dynamic, builtin_axes, adaptive_storage<>>;
} // namespace histogram

namespace python {

#ifdef HAVE_NUMPY
class access {
public:
  using mp_int = histogram::adaptive_storage<>::mp_int;
  using weight = histogram::adaptive_storage<>::weight;
  template <typename T>
  using array = histogram::adaptive_storage<>::array<T>;

  struct dtype_visitor : public static_visitor<std::pair<int, object>> {
    template <typename Array>
    std::pair<int, object> operator()(const Array& /*unused*/) const {
      std::pair<int, object> p;
      p.first = sizeof(typename Array::value_type);
      p.second = str("|u") + str(p.first);
      return p;
    }
    std::pair<int, object> operator()(const array<void>& /*unused*/) const {
      std::pair<int, object> p;
      p.first = sizeof(uint8_t);
      p.second = str("|u") + str(p.first);
      return p;
    }
    std::pair<int, object> operator()(const array<mp_int>& /*unused*/) const {
      std::pair<int, object> p;
      p.first = sizeof(double);
      p.second = str("|f") + str(p.first);
      return p;
    }
    std::pair<int, object> operator()(const array<weight>& /*unused*/) const {
      std::pair<int, object> p;
      p.first = 0; // communicate that the type was array<weight>
      p.second = str("|f") + str(sizeof(double));
      return p;
    }
  };

  struct data_visitor : public static_visitor<object> {
    const list& shapes;
    const list& strides;
    data_visitor(const list& sh, const list& st) : shapes(sh), strides(st) {}
    template <typename Array>
    object operator()(const Array& b) const {
      return make_tuple(reinterpret_cast<uintptr_t>(b.begin()), true);
    }
    object operator()(const array<void>& b) const {
      // cannot pass non-existent memory to numpy; make new
      // zero-initialized uint8 array, and pass it
      auto dim = len(shapes);
      npy_intp shapes2[BOOST_HISTOGRAM_AXIS_LIMIT];
      for (auto i = 0; i < dim; ++i) {
        shapes2[i] = extract<npy_intp>(shapes[i]);
      }
      auto *a = (PyArrayObject*)PyArray_SimpleNew(dim, shapes2, NPY_UINT8);
      for (auto i = 0; i < dim; ++i) {
        PyArray_STRIDES(a)[i] = extract<npy_intp>(strides[i]);
      }
      auto *buf = (uint8_t *)PyArray_DATA(a);
      std::fill(buf, buf + b.size, uint8_t(0));
      PyArray_CLEARFLAGS(a, NPY_ARRAY_WRITEABLE);
      return object(handle<>((PyObject*)a));
    }
    object operator()(const array<mp_int>& b) const {
      // cannot pass cpp_int to numpy; make new
      // double array, fill it and pass it
      auto dim = len(shapes);
      npy_intp shapes2[BOOST_HISTOGRAM_AXIS_LIMIT];
      for (auto i = 0; i < dim; ++i) {
        shapes2[i] = extract<npy_intp>(shapes[i]);
      }
      auto *a = (PyArrayObject*)PyArray_SimpleNew(dim, shapes2, NPY_DOUBLE);
      for (auto i = 0; i < dim; ++i) {
        PyArray_STRIDES(a)[i] = extract<npy_intp>(strides[i]);
      }
      auto *buf = (double *)PyArray_DATA(a);
      for (auto i = 0ul; i < b.size; ++i) {
        buf[i] = static_cast<double>(b[i]);
      }
      PyArray_CLEARFLAGS(a, NPY_ARRAY_WRITEABLE);
      return object(handle<>((PyObject*)a));
    }
  };

  static object array_interface(const histogram::dynamic_histogram &self) {
    dict d;

    list shapes;
    list strides;
    auto &b = self.storage_.buffer_;
    auto dtype = apply_visitor(dtype_visitor(), b);
    auto stride = dtype.first;
    if (stride == 0) { // buffer is weight, needs special treatment
      stride = sizeof(double);
      strides.append(stride);
      stride *= 2;
      shapes.append(2);
    }
    for (unsigned i = 0; i < self.dim(); ++i) {
      const auto s = histogram::shape(self.axis(i));
      shapes.append(s);
      strides.append(stride);
      stride *= s;
    }
    if (self.dim() == 0) {
      shapes.append(0);
      strides.append(stride);
    }
    d["shape"] = tuple(shapes);
    d["strides"] = tuple(strides);
    d["typestr"] = dtype.second;
    d["data"] = apply_visitor(data_visitor(shapes, strides), b);
    return d;
  }
};
#endif

} // namespace python

namespace histogram {

struct axis_visitor : public static_visitor<python::object> {
  template <typename T> python::object operator()(const T &t) const {
    return python::object(t);
  }
};

python::object histogram_axis(const dynamic_histogram &self, int i) {
  if (i < 0)
    i += self.dim();
  if (i < 0 || i >= int(self.dim())) {
    PyErr_SetString(PyExc_IndexError, "axis index out of range");
    python::throw_error_already_set();
  }
  return apply_visitor(axis_visitor(), self.axis(i));
}

python::object histogram_init(python::tuple args, python::dict kwargs) {

  python::object self = args[0];
  python::object pyinit = self.attr("__init__");

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    python::throw_error_already_set();
  }

  const unsigned dim = len(args) - 1;

  // normal constructor
  std::vector<dynamic_histogram::axis_type> axes;
  for (unsigned i = 0; i < dim; ++i) {
    python::object pa = args[i + 1];
    python::extract<axis::regular<>> er(pa);
    if (er.check()) {
      axes.push_back(er());
      continue;
    }
    python::extract<axis::circular<>> ep(pa);
    if (ep.check()) {
      axes.push_back(ep());
      continue;
    }
    python::extract<axis::variable<>> ev(pa);
    if (ev.check()) {
      axes.push_back(ev());
      continue;
    }
    python::extract<axis::integer<>> ei(pa);
    if (ei.check()) {
      axes.push_back(ei());
      continue;
    }
    python::extract<axis::category> ec(pa);
    if (ec.check()) {
      axes.push_back(ec());
      continue;
    }
    std::string msg = "require an axis object, got ";
    msg += python::extract<std::string>(pa.attr("__class__").attr("__name__"))();
    PyErr_SetString(PyExc_TypeError, msg.c_str());
    python::throw_error_already_set();
  }
  dynamic_histogram h(axes.begin(), axes.end());
  return pyinit(h);
}

struct fetcher {
  char type = 0;
  union {
#ifdef HAVE_NUMPY
    PyArrayObject* a;
#endif
    double value;
  };

#ifdef HAVE_NUMPY
  ~fetcher() {
    if (type == 2) Py_DECREF((PyObject*)a);
  }
#endif

  long connect(python::object o) {
    python::extract<double> get_double(o);
    if (get_double.check()) {
      type = 1;
      value = get_double();
      return 0;
    }
#ifdef HAVE_NUMPY
    a = (PyArrayObject*)PyArray_FROMANY(o.ptr(), NPY_DOUBLE, 1, 1, NPY_ARRAY_CARRAY);
    if (a) {
      type = 2;
      return PyArray_SHAPE(a)[0];
    }
#endif
    PyErr_SetString(PyExc_ValueError, "cannot convert argument");
    python::throw_error_already_set();
    return 0;
  }

  bool empty() const { return type == 0; }

  double operator[](long i) const {
#ifdef HAVE_NUMPY
    if (type == 2) {
      return *static_cast<double*>(PyArray_GETPTR1(a, i));
    }
#endif
    return value;
  };
};

python::object histogram_fill(python::tuple args, python::dict kwargs) {
  const auto nargs = python::len(args);
  dynamic_histogram &self = python::extract<dynamic_histogram &>(args[0]);

  const unsigned dim = nargs - 1;
  if (dim != self.dim()) {
    PyErr_SetString(PyExc_ValueError, "number of arguments and dimension do not match");
    python::throw_error_already_set();
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    python::throw_error_already_set();
  }

  fetcher fetch[BOOST_HISTOGRAM_AXIS_LIMIT];
  long n = 0;
  for (auto d = 0u; d < dim; ++d) {
    python::object o = python::extract<python::object>(args[1 + d]);
    const auto on = fetch[d].connect(o);
    if (on > 0) {
      if (n && on != n) {
        PyErr_SetString(PyExc_ValueError, "lengths of sequences do not match");
        python::throw_error_already_set();
      }
      n = on;
    }
  }

  fetcher fetch_weight;
  const auto nkwargs = python::len(kwargs);
  if (nkwargs > 0) {
    if (nkwargs > 1 || !kwargs.has_key("weight")) {
      PyErr_SetString(PyExc_RuntimeError, "only keyword weight allowed");
      python::throw_error_already_set();
    }
    python::object o = kwargs.get("weight");
    const auto on = fetch_weight.connect(o);
    if (on > 0) {
      if (n && on != n) {
        PyErr_SetString(PyExc_ValueError, "length of weight sequence does not match");
        python::throw_error_already_set();
      }
      n = on;
    }
  }

  double v[BOOST_HISTOGRAM_AXIS_LIMIT];
  if (!n) ++n;
  for (auto i = 0l; i < n; ++i) {
    for (auto d = 0u; d < dim; ++d)
      v[d] = fetch[d][i];
    if (fetch_weight.empty()) {
      self.fill(v, v + dim);
    } else {
      self.fill(v, v + dim, weight(fetch_weight[i]));
    }
  }

  return python::object();
}

python::object histogram_value(python::tuple args, python::dict kwargs) {
  const dynamic_histogram & self = python::extract<const dynamic_histogram &>(args[0]);

  const unsigned dim = len(args) - 1;
  if (self.dim() != dim) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    python::throw_error_already_set();
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    python::throw_error_already_set();
  }

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    python::throw_error_already_set();
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = python::extract<int>(args[1 + i]);

  return python::object(self.value(idx + 0, idx + self.dim()));
}

python::object histogram_variance(python::tuple args, python::dict kwargs) {
  const dynamic_histogram &self =
      python::extract<const dynamic_histogram &>(args[0]);

  const unsigned dim = len(args) - 1;
  if (self.dim() != dim) {
    PyErr_SetString(PyExc_RuntimeError, "wrong number of arguments");
    python::throw_error_already_set();
  }

  if (dim > BOOST_HISTOGRAM_AXIS_LIMIT) {
    std::ostringstream os;
    os << "too many axes, maximum is " << BOOST_HISTOGRAM_AXIS_LIMIT;
    PyErr_SetString(PyExc_RuntimeError, os.str().c_str());
    python::throw_error_already_set();
  }

  if (kwargs) {
    PyErr_SetString(PyExc_RuntimeError, "no keyword arguments allowed");
    python::throw_error_already_set();
  }

  int idx[BOOST_HISTOGRAM_AXIS_LIMIT];
  for (unsigned i = 0; i < self.dim(); ++i)
    idx[i] = python::extract<int>(args[1 + i]);

  return python::object(self.variance(idx + 0, idx + self.dim()));
}

std::string histogram_repr(const dynamic_histogram &h) {
  std::ostringstream os;
  os << h;
  return os.str();
}

void register_histogram() {
  python::docstring_options dopt(true, true, false);

  python::class_<dynamic_histogram, boost::shared_ptr<dynamic_histogram>>(
      "histogram", "N-dimensional histogram for real-valued data.", python::no_init)
      .def("__init__", python::raw_function(histogram_init),
           ":param axis args: axis objects"
           "\nPass one or more axis objects to configure the histogram.")
      // shadowed C++ ctors
      .def(python::init<const dynamic_histogram &>())
#ifdef HAVE_NUMPY
      .add_property("__array_interface__", &python::access::array_interface)
#endif
      .def("__len__", &dynamic_histogram::dim)
      .def("axis", histogram_axis, python::arg("i")=0,
           ":param index: integer"
           "\nAccess axis object as index.")
      .def("fill", python::raw_function(histogram_fill),
           "Pass N values where N is equal to the dimensions"
           "\nof the histogram, and optionally another value with the keyword"
           "\n*weight*. All values must be convertible to double."
           "\n"
           "\nIf Numpy support is enabled, 1d-arrays can be passed instead of"
           "\nvalues, which must be equal in lenght. Arrays and values can"
           "\nbe mixed in the same call.")
      .add_property("sum", &dynamic_histogram::sum,
           "Returns sum of all entries, including under- and overflow bins.")
      .def("value", python::raw_function(histogram_value),
           ":param int args: indices of the bin"
           "\n:return: count for the bin")
      .def("variance", python::raw_function(histogram_variance),
           ":param int args: indices of the bin"
           "\n:return: variance estimate for the bin")
      .def("__repr__", histogram_repr,
           ":returns: string representation of the histogram")
      .def(python::self == python::self)
      .def(python::self += python::self)
      .def_pickle(serialization_suite<dynamic_histogram>());
}

} // NS histogram
} // NS boost
