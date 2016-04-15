# VARIABLES set by this module
# - NUMPY_INCLUDE_DIR
# - NUMPY_FOUND
find_package(PythonInterp)

set(NUMPY_FOUND FALSE)

if(PYTHONINTERP_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(EXISTS ${NUMPY_INCLUDE_DIR})
    set(NUMPY_FOUND TRUE)
    if (NOT NUMPY_FIND_QUIETLY)
      message (STATUS "Found numpy")
    endif (NOT NUMPY_FIND_QUIETLY)
  endif()
endif()

if(NOT NUMPY_FOUND AND NUMPY_FIND_REQUIRED)
  message (FATAL_ERROR "numpy missing")
endif()
