Motivation
==========

There is a lack of a widely-used, free histogram class. While it is easy to write an 1-dimensional histogram, writing an n-dimensional histogram poses more of a challenge. If you add serialization and Python/Numpy support onto the wish-list, the air becomes thin.

The main competitor is the `ROOT framework <https://root.cern.ch>`_. The histogram in this project is designed to be more convenient to use, and as fast or faster than the equivalent ROOT histograms. It comes without heavy baggage, instead it has a clean and modern C++ design which follows the advice given in popular C++ books, like those of `Meyers <http://www.aristeia.com/books.html>`_ and `Sutter and Alexandrescu <http://www.gotw.ca/publications/c++cs.htm>`_.
