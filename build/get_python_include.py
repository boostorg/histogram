from distutils import sysconfig
import sys
sys.stdout.write(sysconfig.get_python_inc())
