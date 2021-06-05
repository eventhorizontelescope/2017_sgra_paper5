# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of `blackholepy`.
#
# `Blackholepy` is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# `Blackholepy` is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `blackholepy`.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

def moments(img, width, height, FWHM=False):

    from math import pi, sqrt, log, atan2

    f  = sqrt(8 * log(2)) if FWHM else 1
    s  = np.sum(img)
    w  = width  * ((np.arange(img.shape[-2]) + 0.5) / img.shape[-2] - 0.5)
    h  = height * ((np.arange(img.shape[-1]) + 0.5) / img.shape[-1] - 0.5)

    w0 = np.sum(w * np.sum(img, axis=-1)) / s # axis -2, so sum -1 first
    h0 = np.sum(h * np.sum(img, axis=-2)) / s # axis -1, so sum -2 first
    w -= w0
    h -= h0

    ww = np.sum(w * w * np.sum(img, axis=-1)) / s
    hh = np.sum(h * h * np.sum(img, axis=-2)) / s
    wh = np.sum(w[...,:,np.newaxis] * h[...,np.newaxis,:] * img) / s

    cc = 0.5 * (ww + hh)
    dd = 0.5 * (ww - hh)
    D  = sqrt(dd * dd + wh * wh)

    try:
        major = sqrt(cc + D)
    except ValueError:
        print('Warning! img has negative value')
        major = float('nan')

    try:
        minor = sqrt(cc - D)
    except ValueError:
        print('Warning! img has negative value')
        minor = float('nan')

    return (
        s,
        w0,
        h0,
        major * f,
        minor * f,
        atan2(wh, D - dd) * 180 / pi,
    )
