"""Launch Owl executables with the environment captured before Python imports."""
import os
import subprocess

_environment = os.environ.copy()
_environment['OPENBLAS_NUM_THREADS'] = '1'

def run(command):
    return subprocess.run(command, text=True, capture_output=True, check=True, env=_environment)
