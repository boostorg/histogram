#!/bin/bash

#################
#
# Instructions
#
# Run this script in a docker container (such as ubuntu:20.04), a local machine, or a cloud instance.
# It will build the library, and run lcov.
# The repository will initially be placed at /opt/github/$LIBRARY
# and the final coverage.info file /opt/github/boost-root/libs/$LIBRARY/coverage.info
#
# If you would like to upload the results to coveralls.io, set an env variable
# export COVERALLS_REPO_TOKEN=__

set -xe
LIBRARY_URL=https://github.com/boostorg/histogram
LIBRARY=$(basename $LIBRARY_URL)
GITHUB_WORKSPACE=/opt/github/$LIBRARY
TOOLSET=gcc-8
GCOV=gcov-8
CXXSTD="03,11,14,17,2a"
BASE_PACKAGES="git python3 build-essential g++ curl wget mlocate vim"
PACKAGES="g++-8 $BASE_PACKAGES"
# LCOV_COV_PATH=
UBSAN_OPTIONS="print_stacktrace=1"
B2_OPTS="-q -j2 warnings-as-errors=on"

apt-get update
which sudo | apt-get install -y sudo
echo "Install packages"
sudo apt install -y $PACKAGES
mkdir -p /opt/github
cd /opt/github
if [ ! -d histogram ]; then
  git clone $LIBRARY_URL
  cd $LIBRARY
else
  cd $LIBRARY
  git fetch origin
fi

echo "Setup Boost"
# echo GITHUB_REPOSITORY: $GITHUB_REPOSITORY
# LIBRARY=${GITHUB_REPOSITORY#*/}
echo LIBRARY: $LIBRARY
# echo "LIBRARY=$LIBRARY" >> $GITHUB_ENV
# echo GITHUB_BASE_REF: $GITHUB_BASE_REF
# echo GITHUB_REF: $GITHUB_REF
# REF=${GITHUB_BASE_REF:-$GITHUB_REF}
# REF=${REF#refs/heads/}
# echo REF: $REF
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
BOOST_BRANCH=develop && [ "$GIT_BRANCH" == "master" ] && BOOST_BRANCH=master || true
echo BOOST_BRANCH: $BOOST_BRANCH
cd ..
if [ ! -d boost-root ]; then
  git clone -b $BOOST_BRANCH --depth 1 --no-single-branch https://github.com/boostorg/boost.git boost-root
  cd boost-root
else
  cd boost-root
  git checkout $BOOST_BRANCH
  git pull
fi
BOOST_ROOT=`pwd`

cp -r $GITHUB_WORKSPACE/* libs/$LIBRARY
git submodule update --init tools/boostdep
python3 tools/boostdep/depinst/depinst.py --git_args "--jobs 3" $LIBRARY
./bootstrap.sh
./b2 -d0 headers

if [ -n "$COMPILER" ]; then
  echo "Create user-config.jam"
  echo "using $TOOLSET : : $COMPILER ;" > ~/user-config.jam
fi

./b2 -j3 libs/$LIBRARY/test//all toolset=$TOOLSET cxxstd=latest coverage=on

echo "Process coverage data"
cd libs/$LIBRARY
tools/cov.sh

if [ -n "$COVERALLS_REPO_TOKEN" ]; then
  echo "Uploading results to coveralls.io"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get install -y npm
  npm install coveralls
  cat coverage.info | node_modules/.bin/coveralls
fi
