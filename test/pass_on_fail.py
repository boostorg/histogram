# -*- coding: utf-8 -*-
#
# Copyright 2018 Hans Dembinski
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt
# or copy at http://www.boost.org/LICENSE_1_0.txt)

import subprocess as subp
import sys
import os

args = sys.argv[1:]

if not os.path.isabs(args[0]):
    args[0] = os.path.abspath(args[0])

exit_code = subp.call(args)

sys.exit(not exit_code)
