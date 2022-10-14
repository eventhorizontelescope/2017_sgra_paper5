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
from .convolveSquareImage import *
import pdb

def moments(img, width, height, FWHM=False):

    from math import pi, sqrt, log, atan2

    #img contains many parameters.  Only look at Stokes I.
    StokesI = img[:,:,0]

    f  = sqrt(8 * log(2)) if FWHM else 1
    s  = np.sum(StokesI)
    w  = width  * ((np.arange(StokesI.shape[-2]) + 0.5) / StokesI.shape[-2] - 0.5)
    h  = height * ((np.arange(StokesI.shape[-1]) + 0.5) / StokesI.shape[-1] - 0.5)

    w0 = np.sum(w * np.sum(StokesI, axis=-1)) / s # axis -2, so sum -1 first
    h0 = np.sum(h * np.sum(StokesI, axis=-2)) / s # axis -1, so sum -2 first
    w -= w0
    h -= h0

    ww = np.sum(w * w * np.sum(StokesI, axis=-1)) / s
    hh = np.sum(h * h * np.sum(StokesI, axis=-2)) / s
    wh = np.sum(w[...,:,np.newaxis] * h[...,np.newaxis,:] * StokesI) / s

    cc = 0.5 * (ww + hh)
    dd = 0.5 * (ww - hh)
    D  = sqrt(dd * dd + wh * wh)

    try:
        major = sqrt(cc + D)
    except ValueError:
        print('Warning! StokesI has negative value')
        major = float('nan')

    try:
        minor = sqrt(cc - D)
    except ValueError:
        print('Warning! StokesI has negative value')
        minor = float('nan')

    return (
        s / StokesI.size,
        w0,
        h0,
        major * f,
        minor * f,
        atan2(wh, D - dd) * 180 / pi,
    )

def unresolvedFractionalPolarizations(img):

    if img.shape[2] <= 4:
        #Return nan if there is no polarization data, which we check by just looking at the number of 2d arrays
        return np.nan, np.nan
    else:
        #Otherwise, compute unresolved linear and circular polarization fractions
        totalFlux = np.sum(img[:,:,0])
        unresolvedLinear = np.sqrt(np.sum(img[:,:,1])**2 + np.sum(img[:,:,2])**2)
        unresolvedCircular = np.sum(img[:,:,3])
        return unresolvedLinear/totalFlux, unresolvedCircular/totalFlux

def resolvedFractionalPolarizations(img, blurring_fwhm_muas=20.0):

    if img.shape[2] <= 4:
        #Return nan if there is no polarization data, which we check by just looking at the number of 2d arrays
        return np.nan, np.nan
    else:
        assert np.isclose(np.abs(img.fov.value[0]), np.abs(img.fov.value[1]))
        blurredStokesImages = [convolveSquareImage(img.value[:,:,stokes], np.abs(img.fov.value[0]), blurring_fwhm_muas) for stokes in range(4)]
        resolvedLinear = np.sqrt(blurredStokesImages[1]**2 + blurredStokesImages[2]**2)
        return np.nanmean(resolvedLinear/blurredStokesImages[0]), np.nanmean(blurredStokesImages[3]/blurredStokesImages[0])
