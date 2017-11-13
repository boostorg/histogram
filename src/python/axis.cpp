// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/to_python_converter.hpp>
#ifdef HAVE_NUMPY
#include <boost/python/numpy.hpp>
namespace np = boost::python::numpy;
#endif
#include <sstream>
#include <type_traits>
#include <vector>
#include <utility>
#include <iostream>

namespace boost {
namespace histogram {

namespace {

template <typename T> python::str generic_repr(const T &t) {
  std::ostringstream os;
  os << t;
  return os.str().c_str();
}

struct generic_iterator {
  generic_iterator(python::object o) : iterable(o), size(python::len(iterable)) {}
  python::object next() {
    if (idx == size) {
      PyErr_SetString(PyExc_StopIteration, "No more items.");
      python::throw_error_already_set();
    }
    return iterable[idx++];
  }
  python::object self() { return python::object(*this); }
  python::object iterable;
  unsigned idx = 0;
  unsigned size = 0;
};

generic_iterator make_generic_iterator(python::object self) {
  return generic_iterator(self);
}

template <typename T>
struct axis_interval_to_python
{
  static PyObject* convert(const axis::interval<T> &i)
  {
    return python::incref(python::make_tuple(i.lower(), i.upper()).ptr());
  }
};

template <typename T>
struct pair_int_axis_interval_to_python
{
  static PyObject* convert(const std::pair<int, axis::interval<T>> &p)
  {
    return python::incref(python::make_tuple(
      p.first, python::make_tuple(p.second.lower(), p.second.upper())
    ).ptr());
  }
};

python::object variable_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) < 2) {
    PyErr_SetString(PyExc_TypeError, "require at least two arguments");
    throw_error_already_set();
  }

  std::vector<double> v;
  for (int i = 1, n = len(args); i < n; ++i) {
    v.push_back(extract<double>(args[i]));
  }

  string_view label;
  auto uo = axis::uoflow::on;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    string_view k = extract<const char*>(kv[0])();
    object v = kv[1];
    if (k == "label")
      label = extract<const char*>(v)();
    else if (k == "uoflow") {
      if (!extract<bool>(v))
        uo = axis::uoflow::off;
    }
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  return self.attr("__init__")(
      axis::variable<>(v.begin(), v.end(), label, uo));
}

python::object category_init(python::tuple args, python::dict kwargs) {
  using namespace python;

  object self = args[0];

  if (len(args) == 1) {
    PyErr_SetString(PyExc_TypeError, "require at least one argument");
    throw_error_already_set();
  }

  string_view label;
  while (len(kwargs) > 0) {
    python::tuple kv = kwargs.popitem();
    string_view k = extract<const char*>(kv[0])();
    object v = kv[1];
    if (k == "label")
      label = extract<const char*>(v)();
    else {
      std::stringstream s;
      s << "keyword " << k << " not recognized";
      PyErr_SetString(PyExc_KeyError, s.str().c_str());
      throw_error_already_set();
    }
  }

  std::vector<int> c;
  for (int i = 1, n = len(args); i < n; ++i)
    c.push_back(extract<int>(args[i]));

  return self.attr("__init__")(axis::category<>(c.begin(), c.end(), label));
}

template <typename A> python::object axis_getitem(const A &a, int i) {
  if (i < -1 * a.uoflow() || i >= a.size() + 1 * a.uoflow()) {
    PyErr_SetString(PyExc_IndexError, "index out of bounds");
    python::throw_error_already_set();
  }
  return python::object(a[i]);
}

template <typename T> void axis_set_label(T& t, python::str s) {
  t.label({python::extract<const char*>(s)(),
           static_cast<std::size_t>(python::len(s))});
}

template <typename T> python::str axis_get_label(const T& t) {
  auto s = t.label();
  return {s.data(), s.size()};
}

#ifdef HAVE_NUMPY
template <typename Axis> python::object axis_array_interface(const Axis& axis) {
  using T = typename std::decay<decltype(axis[0].lower())>::type;
  python::dict d;
  auto shape = python::make_tuple(axis.size()+1);
  d["shape"] = shape;
  d["typestr"] = python::dtype_typestr<T>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<T>());
  auto buf = reinterpret_cast<T*>(a.get_data());
  for (auto i = 0; i < axis.size()+1; ++i)
    buf[i] = axis[i].lower();
  d["data"] = a;
  d["version"] = 3;
  return d;
}

template <> python::object axis_array_interface<axis::category<>>(const axis::category<>& axis) {
  python::dict d;
  auto shape = python::make_tuple(axis.size());
  d["shape"] = shape;
  d["typestr"] = python::dtype_typestr<int>();
  // make new array, and pass it to Python
  auto a = np::empty(shape, np::dtype::get_builtin<int>());
  auto buf = reinterpret_cast<int*>(a.get_data());
  for (auto i = 0; i < axis.size(); ++i)
    buf[i] = axis[i];
  d["data"] = a;
  d["version"] = 3;
  return d;
}
#endif

template <class T>
struct axis_suite : public python::def_visitor<axis_suite<T>> {
  template <class Class> static void visit(Class &cl) {
    cl.add_property(
        "shape", &T::shape,
        "Number of bins, including over-/underflow bins if they are present.");
    cl.add_property(
        "label", axis_get_label<T>, axis_set_label<T>,
        "Name or description for the axis.");
    cl.def("index", &T::index, ":param float x: value"
                               "\n:returns: bin index for the passed value",
           python::args("self", "x"));
    cl.def("__len__", &T::size,
           ":returns: number of bins, excluding over-/underflow bins.",
           python::arg("self"));
    cl.def("__getitem__", axis_getitem<T>,
           ":param integer i: bin index"
           "\n:returns: bin corresponding to index",
           python::args("self", "i"));
    cl.def("__iter__", make_generic_iterator);
    cl.def("__repr__", generic_repr<T>,
           ":returns: string representation of this axis", python::arg("self"));
    cl.def(python::self == python::self);
#ifdef HAVE_NUMPY
    cl.add_property("__array_interface__", &axis_array_interface<T>);
#endif
  }
};

template <typename Transform>
axis::regular<double, Transform>* regular_init(
  unsigned bin, double lower, double upper,
  python::str pylabel, bool with_uoflow)
{
  using namespace ::boost::python;
  const auto uo = with_uoflow ? axis::uoflow::on : axis::uoflow::off;
  return new axis::regular<double, Transform>(bin, lower, upper,
      {extract<const char*>(pylabel)(),
       static_cast<std::size_t>(len(pylabel))},
      uo);
}

axis::regular<double, axis::transform::pow>* regular_pow_init(
  unsigned bin, double lower, double upper, double power,
  python::str pylabel, bool with_uoflow)
{
  using namespace ::boost::python;
  const auto uo = with_uoflow ? axis::uoflow::on : axis::uoflow::off;
  return new axis::regular<double, axis::transform::pow>(
      bin, lower, upper,
      {extract<const char*>(pylabel)(),
       static_cast<std::size_t>(len(pylabel))},
      uo, power);
}

axis::integer<>* integer_init(int lower, int upper,
                              python::str pylabel, bool with_uoflow)
{
  using namespace ::boost::python;
  const auto uo = with_uoflow ? axis::uoflow::on : axis::uoflow::off;
  return new axis::integer<>(lower, upper,
      {extract<const char*>(pylabel)(),
       static_cast<std::size_t>(len(pylabel))},
      uo);
}

} // namespace

void register_axis_types() {
  using namespace ::boost::python;
  using namespace ::boost::histogram::axis;
  using ::boost::python::arg; // resolve ambiguity
  docstring_options dopt(true, true, false);

  to_python_converter<
    interval<int>,
    axis_interval_to_python<int>
  >();

  to_python_converter<
    interval<double>,
    axis_interval_to_python<double>
  >();

  to_python_converter<
    std::pair<int, interval<int>>,
    pair_int_axis_interval_to_python<int>
  >();

  to_python_converter<
    std::pair<int, interval<double>>,
    pair_int_axis_interval_to_python<double>
  >();

  class_<generic_iterator>("generic_iterator", init<object>())
    .def("__iter__", &generic_iterator::self)
    .def("next", &generic_iterator::next);

  class_<regular<>>(
      "regular",
      "Axis for real-valued data and bins of equal width."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", make_constructor(regular_init<axis::transform::identity>,
        default_call_policies(),
        (arg("bin"), arg("lower"), arg("upper"),
         arg("label")="", arg("uoflow")=true)))
      .def(axis_suite<regular<>>());

#define BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(x)                           \
  class_<regular<double, axis::transform::x>>(                            \
      "regular_"#x,                                                       \
      "Axis for real-valued data and bins of equal width in "#x"-space."  \
      "\nBinning is a O(1) operation.",                                   \
      no_init)                                                            \
      .def("__init__", make_constructor(regular_init<axis::transform::x>, \
        default_call_policies(),                                          \
        (arg("bin"), arg("lower"), arg("upper"),                          \
         arg("label")="", arg("uoflow")=true)))                           \
      .def(axis_suite<regular<double, axis::transform::x>>())

  BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(log);
  BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(sqrt);
  BOOST_HISTOGRAM_PYTHON_REGULAR_CLASS(cos);

  class_<regular<double, axis::transform::pow>>(
      "regular_pow",
      "Axis for real-valued data and bins of equal width in power-space."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", make_constructor(regular_pow_init,
        default_call_policies(),
        (arg("bin"), arg("lower"), arg("upper"), arg("power"),
         arg("label")="", arg("uoflow")=true)))
      .def(axis_suite<regular<double, axis::transform::pow>>());

  class_<circular<>>(
      "circular",
      "Axis for real-valued angles."
      "\nThere are no overflow/underflow bins for this axis,"
      "\nsince the axis is circular and wraps around after reaching"
      "\nthe perimeter value. Binning is a O(1) operation.",
      no_init)
      .def(init<unsigned, double, double, const char*>(
          (arg("self"), arg("bin"), arg("phase") = 0.0,
           arg("perimeter") = math::double_constants::two_pi,
           arg("label") = "")))
      .def(axis_suite<circular<>>());

  class_<variable<>>(
      "variable",
      "Axis for real-valued data and bins of varying width."
      "\nBinning is a O(log(N)) operation. If speed matters and"
      "\nthe problem domain allows it, prefer a regular axis.",
      no_init)
      .def("__init__", raw_function(variable_init))
      .def(init<const variable<> &>())
      .def(axis_suite<variable<>>());

  class_<integer<>>(
      "integer",
      "An axis for a contiguous range of integers with bins"
      "\nthat are one integer wide. Faster than a regular axis."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", make_constructor(integer_init,
           default_call_policies(),
           (arg("lower"), arg("upper"), arg("label") = "",
            arg("uoflow") = true)))
      .def(axis_suite<integer<>>());

  class_<category<>>(
      "category",
      "An axis for set of unique integer values. Each value is mapped to"
      "\na corresponding bin, following the order of the arguments in"
      "\nthe constructor."
      "\nBinning is a O(1) operation.",
      no_init)
      .def("__init__", raw_function(category_init))
      .def(init<const category<> &>())
      .def(axis_suite<category<>>());
}
}
}
