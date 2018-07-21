from distutils import sysconfig
import os
import sys
from pprint import pprint
from glob import glob
pj = os.path.join

LIB_KEYS = ('LIBDEST', 'LIBDIR', 'LIBPL')

if sys.platform == "darwin":
    so_ext = "dylib"
elif sys.platform.startswith("linux"):
    so_ext = "so"
else:
    so_ext = "dll"

config = sysconfig.get_config_vars()

library = "*python%s*%s" % (sysconfig.get_python_version(), so_ext) 
for libpath in LIB_KEYS:
    p = pj(config[libpath], library)
    cand = glob(p)
    if cand and len(cand) == 1:
        sys.stdout.write(cand[0])
        raise SystemExit

pprint("no library found, dumping library pattern, config, and directory contents:")
pprint(library)
pprint(config)

for libpath in LIB_KEYS:
    pprint(libpath)
    p = config[libpath]
    if os.path.exists(p):
        pprint(os.listdir(p))

raise SystemExit(1)
