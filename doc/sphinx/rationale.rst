Rationale
=========

I designed the histogram based on a decade of experience collected in working with Big Data, more precisely in the field of particle physics and astroparticle physics. In many ways, the `ROOT <https://root.cern.ch>`_ histograms served as an example of *not to do it*.

Design principles
-----------------

* "Do one thing and do it well", Doug McIlroy
* The `Zen of Python <https://www.python.org/dev/peps/pep-0020>`_ (also applies to other languages).

Interface convenience
---------------------

A histogram should have the same consistent interface whatever the dimension. Like ``std::vector`` it should *just work*, users shouldn't be forced to make *a priori* choices among several histogram classes and options everytime they encounter a new data set.

Language transparency
---------------------

Python is a great language for data analysis, so the histogram needs Python bindings. The histogram should be usable as an interface between a complex simulation or data-storage system written in C++ and data-analysis/plotting in Python: define the histogram in Python, let it be filled on the C++ side, and then get it back for further data analysis or plotting.

Data analysis in Python is Numpy-based, so Numpy support is a must.  

Powerful binning strategies
---------------------------

The histogram supports half a dozent different binning strategies, conveniently encapsulated in axis objects. There is the standard sorting of real-valued data into bins of equal or varying width, but also binning of angles or integer values.

Extra bins that count over- and underflow values are added by default. This feature can be turned off individually for each axis. The extra bins do not disturb normal bin counting. On an axis with ``n`` bins, the first bin has the index ``0``, the last bin ``n-1``, while the under- and overflow bins are accessible at ``-1`` and ``n``, respectively.

Performance and memory-efficiency
---------------------------------

Dense storage in memory is a must for high performance. Unfortunately, the `curse of dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`_ quickly become a problem as the number of dimensions grows, leading to histograms which consume large amounts (up to GBs) of memory.

Fortunately, having many dimensions typically reduces the number of counts per bin, since counts get spread over many dimensions. The histogram uses an adaptive count size per bin to be as memory-efficient as possible, by starting with the smallest integer size per bin of 1 byte and increasing as needed to up to 8 byte. A ``std::vector`` grows in *length* as new elements are added, while the count storage grows in *depth*.

Weighted counts and variance estimates
--------------------------------------

A histogram categorizes and counts, so the natural choice for the data type of the counts are integers. However, in particle physics, histograms are also often filled with weighted events, for example, to make sure that two histograms look the same in one variable, while the distribution of another, correlated variable is a subject of study.

The histogram can be filled with either weighted or unweighted counts. In the weighted case, the sum of weights is stored in a ``double``. The histogram provides a variance estimate is both cases. In the unweighted case, the estimate is computed from the count itself, using Poisson-theory. In the weighted case, the sum of squared weights is stored alongside the sum of weights in a second ``double``, and used to compute a variance estimate.

Serialization and zero-suppression
----------------------------------

Serialization is implemented using ``boost::serialization``. Pickling in Python is implemented based on the C++ serialization code. To ensure portability of the pickled histogram, the pickle string is an ASCII representation of the histogram, generated with the ``boost::archive::text_oarchive``. It would be great to switch to a portable binary representation in the future, when that becomes available.

To reduce the size of the string, run-length encoding is applied (zero-suppression) to sequences of zeros. Partly filled histograms often contain large sequences of zeros.
