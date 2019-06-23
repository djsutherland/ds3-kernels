import sys
assert sys.version_info >= (3, 6)

from enum import Enum

Engine = Enum("Engine", "AUTOGRAD PYTORCH TENSORFLOW")
