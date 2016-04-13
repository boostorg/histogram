Types
=====

The library consists of a single :cpp:class:`histogram` and several axis types which are stored in a ``boost::variant`` called :cpp:type:`axis_type`. The axis types are created and passed to the constructor of the histogram to define its binning scheme. All following types are embedded in the ``boost::histogram`` namespace, which is omitted for brevity.

The Histogram Class
-------------------

C++ interface
^^^^^^^^^^^^^

.. cpp:namespace:: boost::histogram

.. cpp:class:: histogram : public basic_histogram

    The class implements an n-dimensional histogram, managing counts in bins.

    It inherits from :cpp:class:`basic_histogram`, which manages the stored axis instances and the conversion of an n-dimensional tuple or index into an internal linear offset that is used to address the bin count. How the bin count is stored is an encapsulated implementation detail.

.. cpp:function:: histogram::histogram(const axis_type& a0, ...)

    Constructors for a variable number of axis types, each defining the binning scheme for its dimension. Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` axis types can be passed to the constructor, yielding the same number of dimensions.

.. cpp:function:: histogram::fill_c(unsigned n, const double* v)

    Fills the histogram with a c-array ``v`` of length ``n``. A checks at run-time asserts that ``n`` agrees with the dimensions of the histogram. Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` dimensions are supported.

.. cpp:function:: histogram::fill(double x0, ...)

    Same as :cpp:func:`histogram::fill_c`, but passing the values directly.  Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` dimensions are supported.

.. cpp:function:: histogram::wfill_c(unsigned n, const double* v, double w)

    Fills the histogram with a c-array ``v`` of length ``n``, using weight ``w``. A checks at run-time asserts that ``n`` agrees with the dimensions of the histogram. Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` dimensions are supported.

Axis Types
----------

C++ interface
^^^^^^^^^^^^^

.. cpp:type:: boost::variant\<regular_axis, polar_axis, variable_axis, \
                              category_axis, integer_axis> axis_type

    A variant template which stores one of several axis objects.

Python interface
^^^^^^^^^^^^^^^^
