#  tests for ipython-7.32.0-py37h89c1867_0 (this is a generated file);
print('===== testing package: ipython-7.32.0-py37h89c1867_0 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import subprocess
import platform
import os
import sys

WIN = platform.system() == "Windows"
LINUX = platform.system() == "Linux"
PYPY = "__pypy__" in sys.builtin_module_names
PPC = "ppc" in platform.machine()

# Environment variable should be set in the meta.yaml
MIGRATING = eval(os.environ.get("MIGRATING", "None"))

# this is generally failing, for whatever reason
NOSE_EXCLUDE = ["recursion"]

if WIN:
    NOSE_EXCLUDE += ["home_dir_3", "home_dir_5", "store_restore", "storemagic"]
else:
    NOSE_EXCLUDE += ["history"]

if LINUX:
    # https://github.com/ipython/ipython/issues/12164
    NOSE_EXCLUDE += ["system_interrupt"]

if PPC:
    NOSE_EXCLUDE += ["ipython_dir_8", "audio_data"]

IPTEST_ARGS = []

if PYPY:
    # TODO: figure out a better way to skip doctests, so the 500+ `core` tests
    #       that _do_ work are executed
    IPTEST_ARGS = [
        "autoreload",
        "extensions",
        "lib",
        "terminal",
        "testing",
        "utils",
    ]
    NOSE_EXCLUDE += [
        "audio",
        "check_complete",
        "longer",
        "memory_error",
        "nest_embed",
        "obj_del",
        "reset_del",
        "tclass",
        "ultratb",
        "xdel"
    ]

if __name__ == "__main__":
    print("Building on Windows?", WIN)
    print("Building on Linux?", LINUX)
    print("Building for PyPy?", PYPY)
    print("Is this a migration?", MIGRATING)

    if MIGRATING:
        print("This is a migration, skipping test suite! Put it back later!", flush=True)
    else:
        env = dict(os.environ)
        env["NOSE_EXCLUDE"] = "|".join(sorted(NOSE_EXCLUDE))
        print("NOSE_EXCLUDE is {NOSE_EXCLUDE}".format(**env), flush=True)
        print("ipytest3 args", *IPTEST_ARGS, flush=True)
        sys.exit(subprocess.call(["iptest3", *IPTEST_ARGS], env=env))
#  --- run_test.py (end) ---

print('===== ipython-7.32.0-py37h89c1867_0 OK =====');
print("import: 'IPython'")
import IPython

print("import: 'IPython.core'")
import IPython.core

print("import: 'IPython.core.magics'")
import IPython.core.magics

print("import: 'IPython.core.tests'")
import IPython.core.tests

print("import: 'IPython.extensions'")
import IPython.extensions

print("import: 'IPython.extensions.tests'")
import IPython.extensions.tests

print("import: 'IPython.external'")
import IPython.external

print("import: 'IPython.external.decorators'")
import IPython.external.decorators

print("import: 'IPython.lib'")
import IPython.lib

print("import: 'IPython.lib.tests'")
import IPython.lib.tests

print("import: 'IPython.sphinxext'")
import IPython.sphinxext

print("import: 'IPython.terminal'")
import IPython.terminal

print("import: 'IPython.terminal.pt_inputhooks'")
import IPython.terminal.pt_inputhooks

print("import: 'IPython.terminal.tests'")
import IPython.terminal.tests

print("import: 'IPython.testing'")
import IPython.testing

print("import: 'IPython.testing.plugin'")
import IPython.testing.plugin

print("import: 'IPython.testing.tests'")
import IPython.testing.tests

print("import: 'IPython.utils'")
import IPython.utils

print("import: 'IPython.utils.tests'")
import IPython.utils.tests

