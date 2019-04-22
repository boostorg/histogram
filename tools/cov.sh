#!/bin/sh
# must be executed in project root folder
if [ -z $GCOV ]; then
  GCOV=gcov-5
fi

LCOV_VERSION="1.13"
LCOV_DIR="tools/lcov-${LCOV_VERSION}"

if [ ! -e $LCOV_DIR ]; then
  cd tools
  wget -O - https://github.com/linux-test-project/lcov/releases/download/v${LCOV_VERSION}/lcov-${LCOV_VERSION}.tar.gz | tar zxf -
  cd ..
fi

# LCOV="$LCOV_EXE --gcov-tool=${GCOV} --rc lcov_branch_coverage=1"
LCOV="${LCOV_DIR}/bin/lcov --gcov-tool=${GCOV}" # no branch coverage

# collect raw data
$LCOV --base-directory `pwd` \
  --directory `pwd`/../../bin.v2/libs/histogram/test \
  --directory `pwd`/../../bin.v2/libs/histogram/examples \
  --capture --output-file coverage.info

# remove uninteresting entries
$LCOV --extract coverage.info "*/boost/histogram/*" --output-file coverage.info

if [ $CI ]; then
  # upload if on CI
  curl -s https://codecov.io/bash | bash -s - -f coverage.info -X gcov -x $GCOV
else
  # otherwise just print
  $LCOV_DIR/bin/genhtml coverage.info -o coverage-report
fi
