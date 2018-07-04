from distutils import sysconfig
import os.path
import sys
import glob
pj = os.path.join

pyver = sysconfig.get_config_var('VERSION')
getvar = sysconfig.get_config_var

libname = "python" + pyver

lib = glob.glob(pj(getvar('LIBPL'), "*" + libname + ".*"))[0]
sys.stdout.write(lib)