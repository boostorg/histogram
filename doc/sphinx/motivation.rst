Motivation
==========

There is a lack of a widely-used, free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin.

The main competitor is the `ROOT framework <https://root.cern.ch>`_. The histogram in this project is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of `Meyers <http://www.aristeia.com/books.html>`_ and `Sutter and Alexandrescu <http://www.gotw.ca/publications/c++cs.htm>`_.

Two of the main design goals are to conveniently hide the internal details on how things are counted, and to use the same interface for 1-dimensional and n-dimensional histograms. The count storage is an implementation detail, chosen automatically to be fast and efficient. The histogram should *just work*, users shouldn't be forced to make choices among several storage options everytime they encounter a new data set.
