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

.. cpp:namespace-push:: histogram

.. cpp:function:: histogram(const axis_type& a0, ...)

    Constructors for a variable number of axis types, each defining the binning scheme for its dimension. Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` axis types can be passed to the constructor, yielding the same number of dimensions.

.. cpp:function:: void fill_c(unsigned n, const double* v)

    Fills the histogram with a c-array ``v`` of length ``n``. A checks at run-time asserts that ``n`` agrees with the dimensions of the histogram.

    Allocation of internal memory is delayed until the first call to this function.

.. cpp:function:: void fill(double x0, ...)

    Same as :cpp:func:`fill_c`, but passing the values of the tuple directly.

.. cpp:function:: void wfill_c(unsigned n, const double* v, double w)

    Fills the histogram with a c-array ``v`` of length ``n``, using weight ``w``. A checks at run-time asserts that ``n`` agrees with the dimensions of the histogram.

    Allocation of internal memory is delayed until the first call to this function. If the histogram was filled with :cpp:func:`fill_c` before, the internal memory is converted to the wide format used for storing weighted counts.

    If the data is not weighted (all weights are 1.0), using :cpp:func:`fill` is much more space-efficient. In the most extreme case, storing of weighted counts consumes 16x more memory.

.. cpp:function:: void wfill(unsigned n, const double* v, double w)

    Same as :cpp:func:`wfill_c`, but passing the values of the tuple directly.

.. cpp:function:: double value_c(unsigned n, const int* idx) const

    Returns the count of the bin addressed by the supplied index. Just like in Python, negative indices like ``-1`` are allowed and count from the end. So if an axis has ``k`` bins, ``-1`` points to ``k-1``.

.. cpp:function:: double value(int i0, ...) const

    Same as :cpp:func:`value_c`, but passing the values of the index directly.

.. cpp:function:: double variance_c(unsigned n, const int* idx) const

    Returns the variance estimate for the count of the bin addressed by the supplied index. Negative indices are allowed just like in case of :cpp:func:`value_c`.

    Note that it does not return the standard deviation :math:`\sigma`, commonly called "error", but the variance :math:`\sigma^2`.

    In case of unweighted counts, the variance estimate returned is :math:`n`, if the count is :math:`n`. This is a common estimate for the variance based on the theory of the `Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_.

    In case of weighted counts, the variance estimate returned is :math:`\sum_i w_i^2`, if the individual weights are :math:`w_i`. This estimate can be derived from the estimate above using `uncertainty propagation <https://en.wikipedia.org/wiki/Propagation_of_uncertainty>`_. The extra storage needed for keeping track of the this sum is the reason why a histogram with weighted counts consumes more memory.

.. cpp:function:: double variance(int i0, ...) const

    Same as :cpp:func:`variance_c`, but passing the values of the index directly.

.. cpp:function:: unsigned depth() const

    Returns the current size of a count in the internal memory buffer in number of bytes.

.. cpp:function:: double sum() const

    Returns the sum of bin counts, including overflow and underflow bins. This could be implemented as a free function.
    
.. cpp:function:: operator==(const histogram& other) const

    Returns true if the two histograms have the dimension, same axis types, and same data content. Two otherwise identical histograms are not considered equal, if they do not have the same depth, even if counts and variances are the same, for example if one is filled entirely using :cpp:func:`fill`, and the other using :cpp:func:`wfill` using weights = 1.

Axis Types
----------

C++ interface
^^^^^^^^^^^^^

.. cpp:type:: boost::variant\<regular_axis, polar_axis, variable_axis, \
                              category_axis, integer_axis> axis_type

    A variant template which stores one of several axis objects.

Python interface
^^^^^^^^^^^^^^^^
