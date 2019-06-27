"""
Some probably over-engineered infrastructure for lazily computing kernel
matrices, allowing for various sums / means / etc used by MMD-related estimators.
"""
from functools import wraps

import numpy as np
import torch

from .utils import as_tensors


def _cache(f):
    # Only works when the function takes no or simple arguments!
    @wraps(f)
    def wrapper(self, *args):
        key = (f.__name__,) + tuple(args)
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = val = f(self, *args)
        return val

    return wrapper


################################################################################
# Kernel base class

_name_map = {"X": 0, "Y": 1, "Z": 2}


class LazyKernel(torch.nn.Module):
    """
    Base class that allows computing kernel matrices among a bunch of datasets,
    only computing the matrices when we use them.

    Constructor arguments:
        - A bunch of matrices we'll compute the kernel among.
          2d tensors, with second dimension agreeing, or None;
          None is a special value meaning to use the first entry X.
          (This is more efficient than passing the same tensor again.)

    Access the results with:
      - K[0, 1] to get the Tensor between parts 0 and 1.
      - K.XX, K.XY, K.ZY, etc: shortcuts, with X=0, Y=1, Z=2.
      - K.matrix(0, 1) or K.XY_m: returns a Matrix subclass (see below).
    """

    def __init__(self, X, *rest):
        super().__init__()
        self._parts = X, *rest = as_tensors(X, *rest)
        for p in self._parts:
            if p is not None:
                assert len(p.shape) >= 2
                assert p.shape[1:] == X.shape[1:]

        self._cache = {}
        if not hasattr(self, "const_diagonal"):
            self.const_diagonal = False

    @property
    def X(self):
        return self._parts[0]

    @property
    def ns(self):
        return [self.X.shape[0] if p is None else p.shape[0] for p in self._parts]

    @property
    def n_parts(self):
        return len(self._parts)

    @property
    def parts(self):
        return [self.X if p is None else p for p in self._parts]

    def __repr__(self):
        return f"<{type(self).__name__}({', '.join(str(n) for n in self.ns)})>"

    def _precompute(self, A):
        return ()

    @_cache
    def _precompute_i(self, i):
        p = self._parts[i]
        if p is None:
            return self._precompute_i(0)
        return self._precompute(p)

    @_cache
    def __getitem__(self, k):
        try:
            i, j = k
        except ValueError:
            raise KeyError("You should index kernels with pairs")

        A = self._parts[i]
        if A is None:
            return self[0, j]

        B = self._parts[j]
        if B is None:
            return self[i, 0]

        if i > j:
            return self[j, i].t()

        A_info = self._precompute_i(i)
        B_info = self._precompute_i(j)
        return self._compute(A, *A_info, B, *B_info)

    @_cache
    def matrix(self, i, j):
        A = self._parts[i]
        if A is None:
            return self.matrix(0, j)

        B = self._parts[j]
        if B is None:
            return self.matrix(i, 0)

        k = self[i, j]
        if i == j:
            return as_matrix(k, const_diagonal=self.const_diagonal, symmetric=True)
        else:
            return as_matrix(k)

    @_cache
    def joint(self, *inds):
        if not inds:
            return self.joint(*range(self.n_parts))
        return torch.cat([torch.cat([self[i, j] for j in inds], 1) for i in inds], 0)

    @_cache
    def joint_m(self, *inds):
        if not inds:
            return self.joint(*range(self.n_parts))
        return as_matrix(
            self.joint(*inds), const_diagonal=self.const_diagonal, symmetric=True
        )

    def __getattr__(self, name):
        # self.X, self.Y, self.Z
        if len(name) == 1:
            i = _name_map.get(name, np.inf)
            if i < self.n_parts:
                return self.parts[i]
            raise AttributeError()

        # self.XX, self.XY, self.YZ, etc; also self.XX_m
        ret_matrix = False
        if len(name) == 4 and name.endswith("_m"):
            ret_matrix = True
            name = name[:2]

        if len(name) == 2:
            i = _name_map.get(name[0], np.inf)
            j = _name_map.get(name[1], np.inf)
            if i < self.n_parts and j < self.n_parts:
                return self.matrix(i, j) if ret_matrix else self[i, j]
            else:
                raise AttributeError(f"have {self.n_parts} parts, asked for {i}, {j}")

        return super().__getattr__(name)

    def _invalidate_cache(self, i):
        for k in list(self._cache.keys()):
            if (
                i in k[1:]
                or any(isinstance(arg, tuple) and i in arg for arg in k[1:])
                or k in [("joint",), ("joint_m",)]
            ):
                del self._cache[k]

    def drop_last_part(self):
        self._invalidate_cache(self.n_parts - 1)
        del self._parts[-1]

    def change_part(self, i, new):
        assert i < self.n_parts
        assert len(new.shape) == 2
        assert new.shape[1:] == self.X.shape[1:]
        self._invalidate_cache(i)
        self._parts[i] = new

    def append_part(self, new):
        assert len(new.shape) == 2
        assert new.shape[1:] == self.X.shape[1:]
        self._parts.append(new)

    def as_tensors(self, *args, **kwargs):
        "Helper that makes everything a tensor with self.X's type."
        kwargs.setdefault("device", self.X.device)
        kwargs.setdefault("dtype", self.X.dtype)
        return tuple(None if r is None else torch.as_tensor(r, **kwargs) for r in args)


################################################################################
# Matrix wrappers that cache sums / etc. Including various subclasses; see
# as_matrix() to pick between them appropriately.

# TODO: could support a matrix transpose that shares the cache appropriately


class Matrix:
    def __init__(self, M, const_diagonal=False):
        self.mat = M = torch.as_tensor(M)
        self.m, self.n = self.shape = M.shape
        self._cache = {}

    @_cache
    def row_sums(self):
        return self.mat.sum(0)

    @_cache
    def col_sums(self):
        return self.mat.sum(1)

    @_cache
    def row_sums_sq_sum(self):
        sums = self.row_sums()
        return sums @ sums

    @_cache
    def col_sums_sq_sum(self):
        sums = self.col_sums()
        return sums @ sums

    @_cache
    def sum(self):
        if "row_sums" in self._cache:
            return self.row_sums().sum()
        elif "col_sums" in self._cache:
            return self.col_sums().sum()
        else:
            return self.mat.sum()

    def mean(self):
        return self.sum() / (self.m * self.n)

    @_cache
    def sq_sum(self):
        flat = self.mat.view(-1)
        return flat @ flat

    def __repr__(self):
        return f"<{type(self).__name__}, {self.m} by {self.n}>"


class SquareMatrix(Matrix):
    def __init__(self, M):
        super().__init__(M)
        assert self.m == self.n

    @_cache
    def diagonal(self):
        return self.mat.diagonal()

    @_cache
    def trace(self):
        return self.mat.trace()

    @_cache
    def sq_trace(self):
        diag = self.diagonal()
        return diag @ diag

    @_cache
    def offdiag_row_sums(self):
        return self.row_sums() - self.diagonal()

    @_cache
    def offdiag_col_sums(self):
        return self.col_sums() - self.diagonal()

    @_cache
    def offdiag_row_sums_sq_sum(self):
        sums = self.offdiag_row_sums()
        return sums @ sums

    @_cache
    def offdiag_col_sums_sq_sum(self):
        sums = self.offdiag_col_sums()
        return sums @ sums

    @_cache
    def offdiag_sum(self):
        return self.offdiag_row_sums().sum()

    def offdiag_mean(self):
        return self.offdiag_sum() / (self.n * (self.n - 1))

    @_cache
    def offdiag_sq_sum(self):
        return self.sq_sum() - self.sq_trace()


class SymmetricMatrix(SquareMatrix):
    def col_sums(self):
        return self.row_sums()

    def sums(self):
        return self.row_sums()

    def offdiag_col_sums(self):
        return self.offdiag_row_sums()

    def offdiag_sums(self):
        return self.offdiag_row_sums()

    def col_sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def offdiag_col_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()

    def offdiag_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()


class ConstDiagMatrix(SquareMatrix):
    def __init__(self, M, diag_value):
        super().__init__(M)
        self.diag_value = diag_value

    @_cache
    def diagonal(self):
        return self.mat.new_full((1,), self.diag_value)

    def trace(self):
        return self.n * self.diag_value

    def sq_trace(self):
        return self.n * (self.diag_value ** 2)


class SymmetricConstDiagMatrix(ConstDiagMatrix, SymmetricMatrix):
    pass


def as_matrix(M, const_diagonal=False, symmetric=False):
    if symmetric:
        if const_diagonal is not False:
            return SymmetricConstDiagMatrix(M, diag_value=const_diagonal)
        else:
            return SymmetricMatrix(M)
    elif const_diagonal is not False:
        return ConstDiagMatrix(M, diag_value=const_diagonal)
    elif M.shape[0] == M.shape[1]:
        return SquareMatrix(M)
    else:
        return Matrix(M)
