import sys

assert sys.version_info >= (3, 6)

from .kernels import LazyKernel  # noqa: F401
from .utils import as_tensors  # noqa: F401
