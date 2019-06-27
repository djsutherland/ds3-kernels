# flake8: noqa
import sys
assert sys.version_info >= (3, 6)

from .kernels import LazyKernel
from .mmd import mmd2_u_stat_variance
from .omniglot import CombinedOmniglot
from .r2d2_featurizer import R2D2Featurizer
from .utils import as_tensors, pil_grid, plot_confusion_matrix
