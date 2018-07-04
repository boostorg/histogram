import subprocess as subp
import sys
import os

args = sys.argv[1:]

if not os.path.isabs(args[0]):
    args[0] = os.path.abspath(args[0])

exit_code = subp.call(args)

sys.exit(not exit_code)
