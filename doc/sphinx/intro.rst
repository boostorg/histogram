Introduction
============

Histograms are a basic tool in every statistical analysis. They compactly represent a data set of one or several random variables with acceptable loss of information. It is often more convenient to work with a histogram of a data distribution, instead of the original distribution which may consume a lot of memory or disc space. Interesting quantities like the mean, variance, or mode may be extracted from the histogram instead of the original data set.

This library implements a single histogram class with a simple interface, which can be used with 1, 2, 3 or N-dimensional data sets (the internal limit is set to 16 dimensions). It supports normal counting and weighted counting, and its provides a data-driven variance estimate for the sum of the counts in either case. The histogram was written so that it *just works*, to be efficient and safe to use in its application domain, completely hiding the implementation details on how it does its counting (*you*, the user, have more important things to worry about).

The histogram is implemented in C++ and has Python-bindings. It passes the language barrier without copying its internal (possibly large) data buffer. The language transparency allows users who do data analysis in Python to create an empty histogram instance in Python, pass it over to a complex C++ code for filling, then analyse the results::

    import histogram as hg
    import complex_cpp_module

    h = hg.histogram(hg.regular_axis(100, 0, 1))

    complex_cpp_module.run_filling(h)

    # h is now filled with data,
    # continue with statistical analysis of h

Histograms can be added if they have the same signature. This is convenient if histograms are filled in parallel on a cluster and then merged (added).

The histogram can be serialized to disk for persistent storage from C++ and pickled in Python. It comes with Numpy support, too. The histogram can be fast-filled with Numpy arrays for maximum performance, and viewed as a Numpy array without copying its memory buffer.
