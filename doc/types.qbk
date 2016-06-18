Types
=====

The library consists of a single :cpp:class:`histogram` and several axis types which are stored in a ``boost::variant`` called :cpp:type:`axis_type`. The axis types are created and passed to the constructor of the histogram to define its binning scheme. All following types are embedded in the ``boost::histogram`` namespace, which is omitted for brevity.

Histogram type
--------------

``#include <boost/histogram/histogram.hpp>``

C++ interface
^^^^^^^^^^^^^

.. cpp:class:: histogram : public basic_histogram

    The class implements an n-dimensional histogram, managing counts in bins.

    It inherits from :cpp:class:`basic_histogram`, which manages the stored axis instances and the conversion of an n-dimensional tuple or index into an internal linear offset that is used to address the bin count. How the bin count is stored is an encapsulated implementation detail.

    .. cpp:function:: histogram(const axis_type& a0, ...)

        Constructors for a variable number of axis types, each defining the binning scheme for its dimension. Up to :cpp:var:`BOOST_HISTOGRAM_AXIS_LIMIT` axis types can be passed to the constructor, yielding the same number of dimensions.

    .. cpp:function:: void fill_c(unsigned n, const double* v)

        Fills the histogram with a c-array :cpp:var:`v` of length :cpp:var:`n`. A checks at run-time asserts that :cpp:var:`n` agrees with the dimensions of the histogram.

        Allocation of internal memory is delayed until the first call to this function.

    .. cpp:function:: void fill(double x0, ...)

        Same as :cpp:func:`fill_c`, but passing the values of the tuple directly.

    .. cpp:function:: void wfill_c(unsigned n, const double* v, double w)

        Fills the histogram with a c-array :cpp:var:`v` of length :cpp:var:`n`, using weight :cpp:var:`w`. A checks at run-time asserts that `n` agrees with the dimensions of the histogram.

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
        
    .. cpp:function:: bool operator==(const histogram& other) const

        Returns true if the two histograms have the dimension, same axis types, and same data content. Two otherwise identical histograms are not considered equal, if they do not have the same depth, even if counts and variances are the same. This case only occurs if one histogram is filled using :cpp:func:`fill` and the other with :cpp:func:`wfill`, using weights of 1.

    .. cpp:function:: histogram& operator+=(const histogram& other) const

        Adds the counts of the histogram on the right hand side to this histogram, if the two histograms have the same signature. Otherwise, a :cpp:type:`std::logic_error` is thrown. Returns itself.

    The following member functions are inherited from :cpp:class:`basic_histogram`:

    .. cpp:function:: unsigned dim() const

        Returns the number of dimensions of the histogram, how many axis it has.

    .. cpp:function:: unsigned bins(unsigned i) const

        Returns the number of bins for axis :cpp:var:`i`.

    .. cpp:function:: unsigned shape(unsigned i) const

        Returns the actual number of fields used by the axis. If the axis has no underflow and overflow bins, this is equal to :cpp:func:`bins`. Otherwise, the number is larger by 2.

    .. cpp:function:: template <typename T> T& axis(unsigned i)

        Returns the axis object at index :cpp:var:`i`, casted to type :cpp:type:`T`. A runtime exception is thrown if the type cast is invalid.

    .. cpp:function:: template <typename T> const T& axis(unsigned i) const

        The ``const``-version of the previous member function.

Python interface
^^^^^^^^^^^^^^^^

The operators ``==``, ``+=``, and ``+`` are defined for histograms. They are also pickable.

.. py:module:: histogram

.. autoclass:: histogram

    .. py:method:: __init__(*axes)

        Pass one or more axis objects as arguments to define the dimensions of the histogram.

    .. autoattribute:: dim

    .. automethod:: shape

    .. automethod:: axis

    .. py:method:: fill(*values, w=None)

        Pass a sequence of values with a length ``n`` is equal to the dimensions of the histogram, and optionally a weight :py:obj:`w` for this fill (*int* or *float*).

        If Numpy support is enabled, :py:obj:`values` my also be a 2d-array of shape ``(m, n)``, where ``m`` is the number of tuples to pass at once, and optionally another a second 1d-array :py:obj:`w` of shape ``(m,)``.

    .. py:method:: value(*indices)

        :param int indices: indices of the bin
        :return: count for the bin

    .. py:method:: variance(*indices)

        :param int indices: indices of the bin
        :return: variance estimate for the bin

Axis Types
----------

``#include <boost/histogram/axis.hpp>``

C++ interface
^^^^^^^^^^^^^

Axis types have a similar and often common interface, but have no common base type. To increase performance, axis types are internally stored by :cpp:class:`basic_histogram` in a :cpp:class:`boost::variant`.

.. cpp:type:: boost::variant\<regular_axis, polar_axis, variable_axis, \
                              category_axis, integer_axis> axis_type

    A variant which stores one of several axis objects. It needs to be cast to the type it currently holds to be useful or passed to a visitor.

All axis types support the ``==`` operator.

.. cpp:class:: regular_axis

    An axis for real-valued data and bins of equal width. Binning is a O(1) operation.

    .. cpp:function:: regular_axis(unsigned n, double min, double max, \
                                   const std::string& label=std::string(), bool uoflow=true)

        :param n: number of bins
        :param min: low edge of first bin
        :param max: high edge of last bin
        :param label: description of the axis
        :param uoflow: add underflow and overflow bins to the histogram for this axis or not

.. cpp:class:: polar_axis

    An axis for real-valued angles. There are no overflow/underflow bins for this axis, since the axis is circular and wraps around after :math:`2 \pi`. Binning is a O(1) operation.

    .. cpp:function:: polar_axis(unsigned n, double start, \
                                 const std::string& label=std::string())

        :param n: number of bins
        :param start: starting phase of the angle
        :param label: description of the axis

.. cpp:class:: variable_axis

    An axis for real-valued data and bins of varying width. Binning is a O(log(N)) operation. If speed matters and the problem domain allows it, prefer a regular_axis.

    .. cpp:function:: variable_axis(const std::vector<double>& x, \
                                    const std::string& label = std::string(), bool uoflow=true)

        :param x: bin edges, the number of bins is one less the size of this vector
        :param label: description of the axis
        :param uoflow: add underflow and overflow bins to the histogram for this axis or not

.. cpp:class:: category_axis

    An axis for enumerated categories. The axis stores the category labels, and expects that they are addressed using an integer from ``0`` to ``n-1``. There are no underflow/overflow bins for this axis.  Binning is a O(1) operation.

    .. cpp:function:: category_axis(const std::vector<std::string>& categories)

        :param categories: an ordered sequence of categories that this axis discriminates

    .. cpp:function:: category_axis(const std::string& categories)

        :param categories: a string of categories separated by the character ``;``

    .. cpp:function:: const std::string& operator[](int index) const

        Returns the category for the bin index.

.. cpp:class:: integer_axis

    An axis for a contiguous range of integers. There are no underflow/overflow bins for this axis. Binning is a O(1) operation.

    .. cpp:function:: integer_axis(int min, int max, \
                                   const std::string& label=std:string(), bool uoflow=true)

    .. cpp:function:: int operator[](int index) const

        Returns the integer that is mapped to the bin index.

Common interface among axis types:

.. cpp:function:: unsigned bins() const

    Returns the number of bins.

.. cpp:function:: bool uoflow() const

    Returns whether overflow and underflow bins will be added in the histogram.

.. cpp:function:: const std::string& label() const

    Returns the axis label, which is a name or description (not implemented for category_axis).

.. cpp:function:: void label(const std::string& label)

    Change the label of an axis (not implemented for category_axis).

.. cpp:function:: int index(const double x) const

    Returns the bin index for the passed argument.

.. cpp:function:: double operator[](int index) const

    Returns the low edge of the bin (not implemented for category_axis and integer_axis).

Python interface
^^^^^^^^^^^^^^^^

All axis types support the operators ``==`` and ``[]``. They support the :py:func:`len` and :py:func:`repr` calls, and the iterator protocol.

.. autoclass:: regular_axis
    :members:

.. autoclass:: polar_axis
    :members:

.. autoclass:: variable_axis
    :members:

.. autoclass:: category_axis
    :members:

.. autoclass:: integer_axis
    :members:
