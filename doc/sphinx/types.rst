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

