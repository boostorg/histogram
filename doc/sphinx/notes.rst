Notes
=====

Dependencies
------------

* `Boost <http://www.boost.org>`_
* `CMake <https://cmake.org>`_

* **Optional dependencies**

  * `Python <http://www.python.org>`_ for Python bindings
  * `Numpy <http://www.numpy.org>`_ for Numpy support
  * `Sphinx <http://www.sphinx-doc.org>`_ to (re)build this documentation

How to build and install
------------------------
::

    git clone git@github.com:HDembinski/histogram.git
    mkdir build; cd build
    cmake ../histogram.git/CMake
    make install

Do ``make test`` to run the tests, or ``ctest -V`` for more output.

**Caveat**: I couldn't figure out a proper way to install the Python module with CMake, so for the time being, CMake will print a message with manual instructions instead. The main problem is how to pick the right dist-packages path in a platform-independent way, and such that it respects the ``CMAKE_INSTALL_PREFIX``.

Tests
-----

Most of the C++ interface is implicitly tested in the tests of the Python interface, which in turn calls the C++ interface.

Checks
------

Some checks are included in ``test/check``. These are not strictly tests, and not strictly examples, yet they provide useful information that belongs with the library code. They are not build by default, building can be activated with the CMake flag ``BUILD_CHECKS``.

Congruence of C++ and Python interface
--------------------------------------

The Python and C++ interface are indentical - except when they are not. The exceptions are minor and concern cases where a more elegant and pythonic way of implementing exists.

* getter/setter-like functions are wrapped as properties
* C++ functions :cpp:func:`histogram::fill` and :cpp:func:`histogram::wfill` are wrapped by the single Python function :py:func:`histogram.fill`
