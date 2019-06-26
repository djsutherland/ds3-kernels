import sys

assert sys.version_info >= (3, 6)

from .kernels import LazyKernel  # noqa: F401
from .omniglot import CombinedOmniglot  # noqa: F401
from .r2d2_featurizer import R2D2Featurizer  # noqa: F401
from .utils import as_tensors, plot_confusion_matrix  # noqa: F401
