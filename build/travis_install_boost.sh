# Must be run from project directory, can be run locally for testing.
# Uses default python interpreter and patches project-config.jam as a
# workaround for a bug in the boost build scripts.
BOOST_VERSION=1.66.0
if [[ -z "${TRAVIS_BUILD_DIR}" ]]; then
  TRAVIS_BUILD_DIR=/tmp
fi
BOOST_DIR=${TRAVIS_BUILD_DIR}/deps/boost-${BOOST_VERSION}
PROJECT_DIR=$(pwd)
echo "Boost: ${BOOST_DIR}"
mkdir -p ${BOOST_DIR}
if [[ -z "$(ls -A ${BOOST_DIR})" ]]; then
  BOOST_URL="http://sourceforge.net/projects/boost/files/boost/${BOOST_VERSION}/boost_${BOOST_VERSION//\./_}.tar.gz"
  { wget --quiet -O - ${BOOST_URL} | tar --strip-components=1 -xz -C ${BOOST_DIR}; } || exit 1
  ( cd ${BOOST_DIR}
    ./bootstrap.sh > /dev/null
    (./b2 install --prefix=${BOOST_DIR} --with-serialization | grep -v -e common\.copy -e common\.mkdir) )
fi
ls ${BOOST_DIR}/lib | grep libboost || exit 1
