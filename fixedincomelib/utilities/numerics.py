import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    r"""Monotone, convex interpolator for instantaneous forward rates (IFRs).

    Nodes ``axis1 = [t_1, ..., t_n]`` (strictly increasing, ``t_1 > 0``) carry
    IFR values ``values = [f_1, ..., f_n]``.  The curve is built interval by
    interval on ``[t_i, t_{i+1}]`` in the local coordinate
    ``s = (x - t_i) / (t_{i+1} - t_i)`` and has these properties.

      * It is continuous and passes through every supplied IFR point.
      * On each node interval it is monotone, and it is convex whenever the
        neighbouring nodes show convex data (see "Incompatible data").
      * ``integrate`` is exact (closed form), so ``exp(-integrate(0, T))`` is
        consistent with ``interpolate``.

    Hagan & West ingredients (Wilmott 2008, section 6; AMF 2006)
    ------------------------------------------------------------
    Hagan & West write the forward on an interval as ``f = fd + g`` where
    ``fd`` is the interval's mean forward ("discrete forward") and ``g`` has
    zero mean, ``g(0) = f_left - fd``, ``g(1) = f_right - fd``.  The quadratic
    through these three conditions is (eq. 27)

        g(s) = g(0) (1 - 4 s + 3 s^2) + g(1) (-2 s + 3 s^2).

    It is monotone when ``g'(0)`` and ``g'(1)`` have the same sign.  When they
    do not, Hagan & West insert a flat segment that turns into a quadratic at
    exactly the point ``eta`` that keeps the mean equal to ``fd``:

        flat at the left  (eq. 28-29):  eta = (g(1) + 2 g(0)) / (g(1) - g(0))
        flat at the right (eq. 30-31):  eta = 3 g(1) / (g(1) - g(0)).

    Both are implemented here, so the interval mean ``fd`` is always honoured
    and the interval is monotone.

    Difference from Hagan & West
    ----------------------------
    Hagan & West take *discrete forwards* as input and derive node forwards
    from them (eq. 22-24).  Here the node values are the inputs, so the node
    forwards are fixed and the interval mean ``fd`` is the free quantity.  It
    is chosen from the local curvature of the data:

        fd = (f_left + f_right)/2 - q/6,   q = kappa h^2 / 2,

    where ``kappa`` is the curvature of the parabola through the neighbouring
    nodes (the smaller of the two node estimates, floored at 0) and ``q`` is
    capped at ``2 |f_right - f_left|``.  ``q >= 0`` puts the mean below the
    chord, which makes the interval convex, and the cap keeps ``fd`` strictly
    between the two node values, so the Hagan-West construction is monotone
    (sectors (i) to (iii) only; the same-sign sector (iv) cannot occur).

    Incompatible data
    -----------------
    If the neighbouring nodes do not show convexity (``kappa <= 0``, a
    concave kink, or only two nodes) then ``q = 0`` and the interval is the
    straight chord: it is still continuous, monotone and weakly convex, and
    nothing overshoots.  At a local maximum or minimum of the data each
    interval is still monotone, and the extremum sits at the node, because
    the construction never leaves the band between the two node values.

    Extrapolation
    -------------
    ``FLAT`` holds the first value before ``t_1`` and the last value after
    ``t_n``.  A one-node curve (the funding curve) is that constant.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT, \
            "MONOTONE_CONVEX supports FLAT extrapolation only."

        self.x_ = np.asarray(self.axis1_, dtype=float)
        self.f_ = np.asarray(self.values_, dtype=float)
        self.n_ = len(self.x_)
        # pieces_[i] lists (s_lo, s_hi, c0, c1, c2) with f(s) = c0 + c1 s + c2 s^2.
        self.pieces_ = []
        if self.n_ >= 2:
            assert np.all(np.diff(self.x_) > 0), "node times must be strictly increasing"
            self._build()

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        x, f, n = self.x_, self.f_, self.n_
        h = np.diff(x)
        slope = np.diff(f) / h

        # Curvature of the parabola through three neighbouring nodes.
        kappa_node = np.zeros(n)
        for j in range(1, n - 1):
            kappa_node[j] = 2.0 * (slope[j] - slope[j - 1]) / (x[j + 1] - x[j - 1])

        self.h_ = h
        self.fd_ = np.empty(n - 1)
        for i in range(n - 1):
            candidates = []
            if i >= 1:
                candidates.append(kappa_node[i])
            if i + 1 <= n - 2:
                candidates.append(kappa_node[i + 1])
            kappa = max(0.0, min(candidates)) if candidates else 0.0

            fl, fr = f[i], f[i + 1]
            delta = fr - fl
            q = min(0.5 * kappa * h[i] ** 2, 2.0 * abs(delta))
            self.fd_[i] = 0.5 * (fl + fr) - q / 6.0
            self.pieces_.append(self._interval_pieces(fl, fr, self.fd_[i]))

    @staticmethod
    def _interval_pieces(fl: float, fr: float, fd: float):
        """Hagan-West pieces ``(s_lo, s_hi, c0, c1, c2)`` for one interval."""
        if fl == fr:
            return [(0.0, 1.0, fl, 0.0, 0.0)]

        g0, g1 = fl - fd, fr - fd
        d0 = -4.0 * g0 - 2.0 * g1        # g'(0)
        d1 = 2.0 * g0 + 4.0 * g1         # g'(1)

        if g0 * g1 >= 0.0:
            # Sector (iv): cannot occur because fd is strictly between fl and fr.
            raise ValueError("interval mean must lie between the node values")

        if d0 * d1 >= 0.0:
            # Sector (i): the quadratic of eq. 27 is already monotone.
            c0 = fd + g0
            c1 = -4.0 * g0 - 2.0 * g1
            c2 = 3.0 * (g0 + g1)
            return [(0.0, 1.0, c0, c1, c2)]

        if abs(g1) > 2.0 * abs(g0):
            # Sector (ii), eq. 28-29: flat at the left, then a quadratic.
            eta = (g1 + 2.0 * g0) / (g1 - g0)
            c2 = (g1 - g0) / (1.0 - eta) ** 2
            c1 = -2.0 * eta * c2
            c0 = fd + g0 + c2 * eta ** 2
            return [(0.0, eta, fd + g0, 0.0, 0.0), (eta, 1.0, c0, c1, c2)]

        # Sector (iii), eq. 30-31: a quadratic, then flat at the right.
        eta = 3.0 * g1 / (g1 - g0)
        c2 = (g0 - g1) / eta ** 2
        c1 = -2.0 * eta * c2
        c0 = fd + g1 + c2 * eta ** 2
        return [(0.0, eta, c0, c1, c2), (eta, 1.0, fd + g1, 0.0, 0.0)]

    # -------------------------------------------------------------- utilities
    def _locate(self, x: float) -> int:
        """Interval index ``i`` with ``x`` in ``[x[i], x[i+1])`` (clipped)."""
        i = int(np.searchsorted(self.x_, x, side='right')) - 1
        return min(max(i, 0), self.n_ - 2)

    @staticmethod
    def _primitive(piece, s: float) -> float:
        _, _, c0, c1, c2 = piece
        return c0 * s + 0.5 * c1 * s * s + c2 * s ** 3 / 3.0

    # ------------------------------------------------------------- interface
    def interpolate(self, x: float) -> float:
        if self.n_ == 1 or x <= self.x_[0]:
            return float(self.f_[0])
        if x >= self.x_[-1]:
            return float(self.f_[-1])

        i = self._locate(x)
        s = (x - self.x_[i]) / self.h_[i]
        pieces = self.pieces_[i]
        for piece in pieces:
            if s < piece[1]:
                break
        else:
            piece = pieces[-1]
        _, _, c0, c1, c2 = piece
        return float(c0 + c1 * s + c2 * s * s)

    def integrate(self, start_x: float, end_x: float) -> float:
        """Exact integral of the IFR curve from ``start_x`` to ``end_x``."""
        if end_x == start_x:
            return 0.0
        if end_x < start_x:
            return -self.integrate(end_x, start_x)
        if self.n_ == 1:
            return float(self.f_[0]) * (end_x - start_x)

        xs = self.x_
        total = 0.0
        if start_x < xs[0]:
            total += float(self.f_[0]) * (min(end_x, xs[0]) - start_x)
        if end_x > xs[-1]:
            total += float(self.f_[-1]) * (end_x - max(start_x, xs[-1]))

        lo, hi = max(start_x, xs[0]), min(end_x, xs[-1])
        if hi > lo:
            for i in range(self._locate(lo), self._locate(hi) + 1):
                seg_lo, seg_hi = max(lo, xs[i]), min(hi, xs[i + 1])
                if seg_hi <= seg_lo:
                    continue
                s_lo = (seg_lo - xs[i]) / self.h_[i]
                s_hi = (seg_hi - xs[i]) / self.h_[i]
                acc = 0.0
                for piece in self.pieces_[i]:
                    a, b = max(s_lo, piece[0]), min(s_hi, piece[1])
                    if b > a:
                        acc += self._primitive(piece, b) - self._primitive(piece, a)
                total += acc * self.h_[i]
        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
