from distutils import sysconfig
import os.path
import sys
import glob
pj = os.path.join

pyver = sysconfig.get_config_var('VERSION')
getvar = sysconfig.get_config_var

libname = "python" + pyver

for ext in ('so', 'dylib', 'dll'):
    lib = glob.glob(pj(getvar('LIBPL'), "*" + libname + "." + ext))
    if lib:
        assert len(lib) == 1
        sys.stdout.write(lib[0])
        break
