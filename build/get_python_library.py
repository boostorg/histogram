from distutils import sysconfig
import os.path as op
import sys
v = sysconfig.get_config_vars()
fpaths = [op.join(v[pv], v['LDLIBRARY']) for pv in ('LIBDIR', 'LIBPL')]
sys.stdout.write(list(filter(op.exists, fpaths))[0])