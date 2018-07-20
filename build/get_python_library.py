from distutils import sysconfig
import subprocess as subp
import os
import sys
from glob import glob
from pprint import pprint
pj = os.path.join
ex = os.path.exists

config = sysconfig.get_config_vars()

def fail():
    pprint("no library found, dumping config:")
    pprint(config)
    raise SystemExit(1)

bindir = config.get("BINDIR")

python_config = pj(bindir, "python-config")
if not ex(python_config):
    pprint(python_config + " not found")
    fail()

args = subp.check_output([python_config, "--ldflags"]).split()

libdir = []
lib = []

so_ext = config.get("SO")

for arg in args:
    if arg.startswith("-L"):
        libdir.append(arg[2:])
    if arg.startswith("-l"):
        lib.append(arg[2:])

for d in libdir:
    for l in lib:
        pattern = pj(d, "*" + l + "*" + so_ext)
        match = glob(pattern)
        if match:
            assert len(match) == 1
            sys.stdout.write(match[0])
            raise SystemExit

fail()
