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

from copy    import copy
from math    import isclose, ceil
from numbers import Number

import numpy as np
from astropy import units

from . import dalt


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

    if not isclose(abs(width), abs(height)):
        print("WARNING: image FOV is anisotropic")

    U = imgs.shape[-2] / width  # U in unit of lambda when width  in rad
    V = imgs.shape[-1] / height # V in unit of lambda when height in rad
    if not isclose(abs(U), abs(V)):
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


def downifft(spec, U, V, N=None, show=False):

    Nu =      spec.shape[-2]
    Nv = 2 * (spec.shape[-1] - 1)
    if not evendim(spec):
        Nv += 1
        print("WARNING: the last dimension should be odd; but ifft is making it even")

    if not isclose(abs(U), abs(V)):
        print("WARNING: image resolution is anisotropic")

    width  = Nu / U
    height = Nv / V
    if not isclose(abs(width), abs(height)):
        print("WARNING: image FOV is anisotropic")

    if isinstance(N, Number):
        N   = np.array([N, N])
    elif isinstance(N, (list, tuple)) and len(N) == 2 and all(isinstance(n, Number) for n in N):
        N   = np.array(N)
    else: # try to make FoV as isotropic as possible
        uvd = min(abs(U), abs(V))
        W   = round(Nu * abs(uvd / U))
        H   = round(Nv * abs(uvd / V))
        N   = np.array([W, H])
        print(f"{(Nu, Nv)} -> {N}") # note padding does not affect FoV

    assert N[0] <= Nu and N[1] <= Nv

    if N[0] < Nu or N[1] < Nv:

        i = Nu//2 - N[0]//2
        f = i + N[0]

        trun = spec[...,i:f,:N[1]//2+1] / (Nu * Nv) # makes a copy so we can write to spec below
        norm = 'forward'

        if N[0] % 2 == 0 and N[0] < Nu:
            trun[...,0,:] += spec[...,f,:N[1]//2+1].conj() / (Nu * Nv)

        #if N[1] % 2 == 0:
        if True: # because imgs.shape[-1] is always made even; see first warning in this function
            H = N[0]//2
            I = H + 1

            trun[...,H,-1].real *= 2
            trun[...,H,-1].imag  = 0

            if N[0] % 2 == 0:
                trun[...,0,-1].real *= 2
                trun[...,0,-1].imag  = 0
                trun[...,1:H,-1] += np.flip(trun[...,I: ,-1].conj(), axis=-1)
                trun[...,I: ,-1]  = np.flip(trun[...,1:H,-1].conj(), axis=-1)
            else:
                trun[..., :H,-1] += np.flip(trun[...,I: ,-1].conj(), axis=-1)
                trun[...,I: ,-1]  = np.flip(trun[..., :H,-1].conj(), axis=-1)
    else:
        trun = spec
        norm = 'backward'

    if show:
        from matplotlib import pyplot as plt
        plt.imshow(abs(trun.T), origin='lower')

    imgs = np.fft.fftshift(np.fft.irfft2(np.fft.fftshift(trun, axes=-2), norm=norm), axes=(-2,-1))
    return imgs, width, height


def crop(imgs, width, height):
    meta = copy(imgs.meta)
    Nw = 2 * ceil(imgs.shape[-2] * width  / meta.width  / 2)
    Nh = 2 * ceil(imgs.shape[-1] * height / meta.height / 2)
    iw = imgs.shape[-2]//2 - Nw//2
    ih = imgs.shape[-1]//2 - Nh//2
    meta.width  = meta.width  * Nw / imgs.shape[-2]
    meta.height = meta.height * Nh / imgs.shape[-1]
    return dalt.Image(copy(imgs.value)[...,iw:iw+Nw,ih:ih+Nh], meta=meta)


def mockserve(imgs, N=None):
    px = imgs.fov / imgs.shape[-2:]
    pa = abs(px[0] * px[1])
    m  = imgs.meta
    return dalt.Visibility(*upfft(imgs*pa, *imgs.fov.to(units.rad).value, N=N),
                           freq=m.freq, time=m.time)

def compress(imgs, N=None, cutoff=15e9):
    fov = imgs.fov.to(units.rad).value
    spec, U, V = upfft(imgs, *fov, N=N)

    Nu =    spec.shape[-2]
    Nv = 2*(spec.shape[-1]-1)

    I, W, H = downifft(spec, U, V, N=[
        2 * ceil(abs(Nu * cutoff / U)),
        2 * ceil(abs(Nv * cutoff / V)),
    ])

    meta = copy(imgs.meta)
    meta.width  = abs(W * meta.dist).to(meta.rg)
    meta.height = abs(H * meta.dist).to(meta.rg)
    return dalt.Image(I, meta=meta)
