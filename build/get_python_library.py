from distutils import sysconfig
import os.path
import sys
import glob
pj = os.path.join

pyver = sysconfig.get_config_var('VERSION')
getvar = sysconfig.get_config_var

libname = "python" + pyver

for libvar in ('LIBDIR', 'LIBPL'):
    for ext in ('so', 'dylib', 'dll'):
        match = pj(getvar(libvar), "*" + libname + "*." + ext)
        lib = glob.glob(match)
        if lib:
            assert len(lib) == 1
            sys.stdout.write(lib[0])
            break
