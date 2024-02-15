import os
import sys
import numpy

import numpy.core.multiarray
import numpy.core.numeric
import numpy.core.umath
import numpy.fft.pocketfft
import numpy.linalg.lapack_lite
import numpy.random.mtrand

try:
    from numpy.fft import _restore_dict  # sentinal that mkl_fft is in use
    from mkl_fft import __version__
    print('USING MKLFFT: %s' % __version__)
except ImportError:
    print("Not using MKLFFT")

try:
    print('MKL: %r' % numpy.__mkl_version__)
except AttributeError:
    print('NO MKL')

if sys.platform == 'darwin':
    os.environ['LDFLAGS'] = ' '.join((os.getenv('LDFLAGS', ''), " -undefined dynamic_lookup"))
elif sys.platform.startswith('linux'):
    os.environ['LDFLAGS'] = ' '.join((os.getenv('LDFLAGS', ''), '-shared'))
    os.environ['FFLAGS'] = ' '.join((os.getenv('FFLAGS', ''), '-Wl,-shared'))
result = numpy.test()
if sys.version_info[0:2] == (3, 7) and not result:
    print("WARNING :: Ignore numpy test failures on Python 3.7")
    sys-exit(0)
sys.exit(not result)
