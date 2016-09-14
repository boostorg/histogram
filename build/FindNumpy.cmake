# VARIABLES set by this module
# - NUMPY_INCLUDE_DIR
# - NUMPY_VERSION
# - NUMPY_FOUND
set(NUMPY_FOUND FALSE)

find_package(PythonInterp REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, sys; sys.stdout.write(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import numpy, sys; sys.stdout.write(numpy.version.version)"
  OUTPUT_VARIABLE NUMPY_VERSION
)

if(EXISTS ${NUMPY_INCLUDE_DIR})
  if(NOT NUMPY_FIND_QUIETLY)
    message(STATUS "Numpy version: ${NUMPY_VERSION} (required >= ${Numpy_FIND_VERSION})")
  endif()
  if("${NUMPY_VERSION}" VERSION_GREATER "${Numpy_FIND_VERSION}")
    set(NUMPY_FOUND TRUE)
  else()
    set(NUMPY_FOUND FALSE)
  endif()
endif()

if(NOT NUMPY_FOUND AND NUMPY_FIND_REQUIRED)
  message (FATAL_ERROR "numpy missing")
endif()
