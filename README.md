# histogram
Fast n-dimensional histogram with convenient interface for C++ and Python

This project contains an easy-to-use powerful n-dimensional histogram class implemented in C++0x, optimized for convenience and excellent performance under heavy duty. The histogram has a complete C++ and a [Python](http://www.python.org) interface, and can be moved over the language boundary with ease. [Numpy](http://www.numpy.org) is fully supported; histograms can be filled with Numpy arrays at C speeds and are convertible into Numpy arrays without copying data. Histograms can be streamed from/to files and pickled in Python. 

This project only depends on [Boost](http://www.boost.org). Optional dependencies are Python and Numpy.

My goal is to submit this project to the Boost libraries, that's why it uses the boost directory structure and namespace. The code is released under the MIT License, making it free to use in open- and closed-source projects.

## Design rationale

### No candidate
There is a lack of a widely-used free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin. The main competitor is the [ROOT framework](https://root.cern.ch). This histogram class is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of [Meyers](http://www.aristeia.com/books.html) and [Sutter and Alexandrescu](http://www.gotw.ca/publications/c++cs.htm).

### Interface convenience, language transparency
A histogram should have the same consistent interface whatever the dimension. Like `std::vector` it should *just work*, users shouldn't be forced to make *a priori* choices among several histogram classes and options everytime they encounter a new data set. Python is a great language for data analysis, so the histogram should have Python bindings. Data analysis in Python is Numpy-based, so Numpy support is a must. The histogram should be usable as an interface between a complex simulation or data-storage system written in C++ and data-analysis/plotting in Python: define the histogram in Python, let it be filled on the C++ side, and then get it back for further data analysis or plotting. 

### Powerful binning strategies
The histogram supports half a dozent different binning strategies, conveniently encapsulated in axis objects. There is the standard sorting of real-valued data into bins of equal or varying width, but also binning of angles or integer values. Extra bins that count over- and underflow values are added by default. This feature can be turned off individually for each dimension. The extra bins do not disturb normal counting. On an axis with n-bins, the first bin has the index `0`, the last bin `n-1`, while the under- and overflow bins are accessible at `-1` and `n`, respectively.

### Performance, cache-friendliness and memory-efficiency
Dense storage in memory is a must for high performance. Unfortunately, the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) quickly become a problem as the number of dimensions grows, leading to histograms which consume large amounts (up to GBs) of memory. Fortunately, having many dimensions typically reduces the number of counts per bin, since tuples get spread over many dimensions. The histogram uses an adaptive count size per bin to exploit this, which starts with the smallest size per bin of 1 byte and increases transparently as needed up to 8 byte per bin. A `std::vector` grows in length as new elements are added, while the count storage grows in *depth*.
