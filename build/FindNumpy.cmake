# VARIABLES set by this module
# - NUMPY_INCLUDE_DIR
# - NUMPY_VERSION
# - NUMPY_FOUND
find_package(PythonInterp)

set(NUMPY_FOUND FALSE)

if(PYTHONINTERP_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, sys; sys.stdout.write(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)

  if(EXISTS ${NUMPY_INCLUDE_DIR})
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, sys; sys.stdout.write(numpy.version.version)"
      OUTPUT_VARIABLE NUMPY_VERSION
    )
    set(NUMPY_FOUND TRUE)
    if (NOT NUMPY_FIND_QUIETLY)
      message(STATUS "Numpy version: ${NUMPY_VERSION}")
    endif (NOT NUMPY_FIND_QUIETLY)
  endif()
endif()

if(NOT NUMPY_FOUND AND NUMPY_FIND_REQUIRED)
  message (FATAL_ERROR "numpy missing")
endif()
