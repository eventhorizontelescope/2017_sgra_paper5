# Copyright (C) 2017 Chi-kwan Chan
# Copyright (C) 2017 Steward Observatory
#
# This file is part of `mockservation`.
#
# `Mockservation` is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# `Mockservation` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `mockservation`.  If not, see <http://www.gnu.org/licenses/>.

from math    import isclose
from numbers import Number

import numpy as np


def almostreal(c, tolerance=1e-7):
    return np.all(abs(c.imag) * tolerance <= abs(c.real))

def evendim(spec):
    nyquist = spec[...,-1]
    N       = nyquist.shape[-1]
    H       = N // 2

    ck0i = almostreal(nyquist[...,H])
    if N % 2 == 0:
        ck0H = almostreal(nyquist[...,    0])
        ckiH = np.allclose(nyquist[...,  1:H],
                   np.flip(nyquist[...,H+1: ].conj(), axis=-1))
    else:
        ck0H = True # N is odd; nothing to check
        ckiH = np.allclose(nyquist[...,   :H],
                   np.flip(nyquist[...,H+1: ].conj(), axis=-1))

    return ck0i and ck0H and ckiH


def upfft(imgs, width, height, N=None):

    if not isclose(width, height):
        print("WARNING: image FOV is anisotropic")

    U = imgs.shape[-2] / width  # U in unit of lambda when width  in rad
    V = imgs.shape[-1] / height # V in unit of lambda when height in rad
    if not isclose(U, V):
        print("WARNING: image resolution is anisotropic")

    if isinstance(N, Number):
        N   = np.array([N, N])
    elif isinstance(N, (list, tuple)) and len(N) == 2 and all(isinstance(n, Number) for n in N):
        N   = np.array(N)
    else: # try to make visibility resolution as isotropic as possible
        fov = max(abs(width), abs(height))
        W   = round(imgs.shape[-2] * abs(fov / width ))
        H   = round(imgs.shape[-1] * abs(fov / height))
        N   = np.array([W, H])
        print(f"{imgs.shape[-2:]} -> {N}") # note padding does not affect U and V

    assert all(N >= imgs.shape[-2:])

    if any(N > imgs.shape[-2:]):
        pi = np.zeros(imgs.ndim, dtype='i')
        pf = np.zeros(imgs.ndim, dtype='i')
        D  = N - np.array(imgs.shape[-2:])
        pi[-2:] = D // 2
        pf[-2:] = D - pi[-2:]
        imgs = np.pad(imgs, tuple(zip(pi, pf)))

    # We use fftshift() to move the zeroth Fourier component to the index N // 2
    spec = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(imgs, axes=(-2,-1)), norm='backward'), axes=-2)
    return spec, U, V
