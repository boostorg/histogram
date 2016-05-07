#include <boost/histogram/axis.hpp>
#include <boost/histogram/basic_histogram.hpp>
#include <boost/python.hpp>
#include <boost/python/def_visitor.hpp>

namespace boost {
namespace histogram {

namespace {

struct axis_visitor : public static_visitor<python::object>
{
  template <typename T>
  python::object operator()(const T& t) const { return python::object(T(t)); }
};

python::object
basic_histogram_axis(const basic_histogram& self, unsigned i)
{
  return apply_visitor(axis_visitor(), self.axis<axis_type>(i));
}

} // namespace

void register_basic_histogram() {
  using namespace python;
  using python::arg;
  docstring_options dopt(true, true, false);

  class_<basic_histogram>("basic_histogram", no_init)
    .add_property("dim", &basic_histogram::dim,
                  "dimensions of the histogram")
    .def("shape", &basic_histogram::shape,
         ":param int i: index of the axis\n"
         ":returns: number of count fields for axis i\n"
         "  (bins + 2 if underflow and overflow"
         " bins are enabled, otherwise equal to bins",
         args("self", "i"))
    .def("axis", basic_histogram_axis,
         ":param int i: index of the axis\n"
         ":returns: axis object for axis i",
         args("self", "i"))
    ;
}

}
}
