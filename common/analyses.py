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

    #Otherwise, compute unresolved linear and circular polarization fractions
    totalFlux = np.sum(img[:,:,0])
    unresolvedLinear = np.sqrt(np.sum(img[:,:,1])**2 + np.sum(img[:,:,2])**2)
    unresolvedCircular = np.sum(img[:,:,3])
    return unresolvedLinear/totalFlux, unresolvedCircular/totalFlux

def resolvedFractionalPolarizations(img, blurring_fwhm_muas=20.0):

    if img.shape[2] <= 4:
        #Return nan if there is no polarization data, which we check by just looking at the number of 2d arrays
        return np.nan, np.nan

    assert np.isclose(np.abs(img.fov.value[0]), np.abs(img.fov.value[1]))
    blurredStokesImages = [convolveSquareImage(img.value[:,:,stokes], np.abs(img.fov.value[0]), blurring_fwhm_muas) for stokes in range(4)]
    resolvedLinear = np.sqrt(blurredStokesImages[1]**2 + blurredStokesImages[2]**2)
    return np.nanmean(resolvedLinear/blurredStokesImages[0]), np.nanmean(blurredStokesImages[3]/blurredStokesImages[0])

def computeBetaCoefficient(img, m=2, r_min=0, r_max=np.inf, norm_in_int=False, norm_with_StokesI=True):
    """Based on pmodes.py by Daniel Palumbo"""

    if img.shape[2] <= 4:
        #Return nan if there is no polarization data, which we check by just looking at the number of 2d arrays
        return np.nan, np.nan

    assert np.isclose(np.abs(img.fov.value[0]), np.abs(img.fov.value[1]))
    fov_muas = np.abs(img.fov.value[0])
    iarr = np.flip(np.transpose(img.value[:,:,0], (1,0)), axis=0)
    qarr = np.flip(np.transpose(img.value[:,:,1], (1,0)), axis=0)
    uarr = np.flip(np.transpose(img.value[:,:,2], (1,0)), axis=0)
    assert iarr.shape[0] == iarr.shape[1]
    npix = iarr.shape[0]

    parr = qarr + 1j*uarr
    normparr = np.abs(parr)
    marr = parr/iarr
    phatarr = parr/normparr
    area = (r_max*r_max - r_min*r_min) * np.pi
    pxi = (np.arange(npix)-0.01)/npix-0.5
    pxj = np.arange(npix)/npix-0.5
    mui = pxi*fov_muas
    muj = pxj*fov_muas
    MUI,MUJ = np.meshgrid(mui,muj)
    MUDISTS = np.sqrt(np.power(MUI,2.)+np.power(MUJ,2.))

    # get angles measured East of North
    PXI,PXJ = np.meshgrid(pxi,pxj)
    angles = np.arctan2(-PXJ,PXI) - np.pi/2.
    angles[angles<0.] += 2.*np.pi

    # get flux in annulus
    tf = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

    # get total polarized flux in annulus
    pf = normparr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()

    #get number of pixels in annulus
    npix = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].size

    #get number of pixels in annulus with flux >= some % of the peak flux
    ann_iarr = iarr [ (MUDISTS<=r_max) & (MUDISTS>=r_min) ]
    peak = np.max(ann_iarr)
    num_above5 = ann_iarr[ann_iarr > .05* peak].size
    num_above10 = ann_iarr[ann_iarr > .1* peak].size

    # compute betas
    qbasis = np.cos(-angles*m)
    ubasis = np.sin(-angles*m)
    pbasis = qbasis + 1.j*ubasis
    if norm_in_int:
        if norm_with_StokesI:
            prod = marr * pbasis
        else:
            prod = phatarr * pbasis
        coeff = prod[ (MUDISTS <= r_max) & (MUDISTS >= r_min) ].sum()
        coeff /= npix
    else:
        prod = parr * pbasis
        coeff = prod[ (MUDISTS<=r_max) & (MUDISTS>=r_min) ].sum()
        if norm_with_StokesI:
            coeff /= tf
        else:
            coeff /= pf

    return np.abs(coeff), np.angle(coeff) * 180.0 / np.pi

def computeOpticalDepth(img):
    """Intensity-weighted average optical depth"""

    I = img.value[:,:,0]
    tau = img.value[:,:,-1]
    return np.sum(tau * I) / np.sum(I)

def computeFaradayDepth(img):
    if img.shape[2] <= 4:
        #Return nan if there is no polarization data, which we check by just looking at the number of 2d arrays
        return np.nan

    I = img.value[:,:,0]
    tauF = img.value[:,:,-2]
    return np.sum(tauF * I) / np.sum(I)

