#  tests for cudatoolkit-10.0.130-0 (this is a generated file);
print('===== testing package: cudatoolkit-10.0.130-0 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import sys
import os
from numba.cuda.cudadrv.libs import test, get_cudalib
from numba.cuda.cudadrv.nvvm import NVVM


def run_test():
    # on windows only nvvm is available to numba
    if sys.platform.startswith('win'):
        nvvm = NVVM()
        print("NVVM version", nvvm.get_version())
        return nvvm.get_version() is not None
    if not test():
        return False
    nvvm = NVVM()
    print("NVVM version", nvvm.get_version())
    # check pkg version matches lib pulled in
    gotlib = get_cudalib('cublas')
    lookfor = os.environ['PKG_VERSION']
    if sys.platform.startswith('win'):
        # windows libs have no dot
        lookfor = lookfor.replace('.', '')
    return lookfor in gotlib


sys.exit(0 if run_test() else 1)
#  --- run_test.py (end) ---

print('===== cudatoolkit-10.0.130-0 OK =====');
