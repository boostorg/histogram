#!/bin/bash

# must be executed in project root folder

# Copyright Hans Dembinski 2018-2019
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

set -ex

if [ -z $GCOV ]; then
  for i in 8 5; do
    if test $(which gcov-$i); then
      GCOV=gcov-$i
      break;
    fi;
  done
fi

LCOV_VERSION="1.15"
LCOV_DIR="tools/lcov-${LCOV_VERSION}"

if [ ! -e $LCOV_DIR ]; then
  cd tools
  curl -L https://github.com/linux-test-project/lcov/releases/download/v${LCOV_VERSION}/lcov-${LCOV_VERSION}.tar.gz | tar zxf -
  cd ..
fi

# For clang
if [ -n "$LLVM_COV_PATH" ]; then
  sudo ln -s $LLVM_COV_PATH /usr/bin/llvm-cov || true
else
  sudo ln -s /usr/bin/llvm-cov-11 /usr/bin/llvm-cov || true
fi

mkdir -p ~/.local/bin
echo -e '#!/bin/bash\nexec llvm-cov gcov "$@"' > ~/.local/bin/gcov_for_clang.sh
chmod 755 ~/.local/bin/gcov_for_clang.sh

# --rc lcov_branch_coverage=1 doesn't work on travis
# LCOV="${LCOV_DIR}/bin/lcov --gcov-tool=${GCOV} --rc lcov_branch_coverage=1"
LCOV="${LCOV_DIR}/bin/lcov --gcov-tool=${GCOV}"

echo "collect raw data"
$LCOV --base-directory `pwd`/../../ \
  --directory `pwd`/../../bin.v2/libs/histogram/test \
  --capture --output-file coverage.info
echo "done with first run of lcov"

echo "remove uninteresting entries"
$LCOV --extract coverage.info "*/boost/histogram/*" --output-file coverage.info
echo "done with extraction"

if [ $1 ]; then
  echo "Uploading to cpp-coveralls"
  # upload if on CI or when token is passed as argument
  which cpp-coveralls || echo "Error: you need to install cpp-coveralls"
  cpp-coveralls -l coverage.info -r ../.. -n -t $1
elif [ ! $CI ]; then
  echo "Generating html report"
  # otherwise generate html report
  $LCOV_DIR/bin/genhtml coverage.info --demangle-cpp -o coverage-report
fi
