# Copyright (C) 2021 Ben Prather
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
from astropy import units
from astropy.io import fits
import pdb

from . import scale as s
from . import dalt  as d

def load_fits(f, pol=True, **kwargs):
    """Load the contents of a fits file just like an ipole image.
    Requires just the astropy.io.fits.hdu.image.PrimaryHDU object,
    i.e. fits.open(fname)[0]
    Note has to ASSUME mass and distance as they are not recorded
    """

    h = f.header
    # These are unreliable/nonexistent in A.C.-O. FITS files
    #MBH  = h['MBH'] * 1e6 * units.M_sun
    #dist = h['dsource'] * units.cm
    # This is needlessly precise so as to exactly
    # match the effective value when converting
    # ipole's L_unit in io_ipole.py
    MBH = 4141166.64059526 * units.M_sun
    L_unit = MBH.to(units.cm, equivalencies=s.GR).value
    dsource = 2.5077305106e+22
    dist = dsource * units.cm
    freq = h['FREQ'] * units.Hz

    try:
        time  = h['TIME']
    except KeyError:
        time = 0

    nx = h['NAXIS1']
    ny = h['NAXIS2']
    fov_to_d = dsource / L_unit / 2.06265e11 # latter is muas_per_rad
    width = abs(nx*h['CDELT1']*3600*1e6*fov_to_d)
    height = abs(ny*h['CDELT2']*3600*1e6*fov_to_d)
    scale = (width * L_unit / nx) * (height * L_unit / ny) / (dsource * dsource) / 1e-23

    if len(f.data.shape) == 2:
        img = f.data / scale
    elif len(f.data.shape) == 3:
        img = np.transpose(f.data[:5,:,:], (2,1,0)) / scale
        try:
            tauI = f.data[4,:,:].T / scale
        except IndexError:
            tauI = None
        try:
            tauF = f.data[5,:,:].T / scale
        except IndexError:
            tauF = None
        
    return d.Image(img, MBH, dist, freq, time, width, height, tauI, tauF, scale, **kwargs)

def load_img(f, **kwargs):
    if isinstance(f, list):
        return load_fits(f[0], **kwargs)
    with fits.open(f) as g:
        img = load_fits(g[0], **kwargs)

        if float(img.meta.dict()['time']) == 0:
            #ARR:  This part must have only applied to some specific file format.  Replacing with something else.
            '''
            if len(f.split('/')[-1].split('_')) > 8:
                time = float(f.split('/')[-1].split('_')[8][1:])
            else:
                time = float(f.split('/')[-1].split('_')[-1].split('.')[0])
            '''
            time = float(f.split('/')[-1].split('_')[1][1:])
            img.set_time(time)
        return img

def load_summ(f, **kwargs):

    """Most info will be missing.  Returning nan for those."""
    
    img = load_img(f, **kwargs)
    Ftot = np.nansum(img[:,:,0]).value * img.scale
    Mdot = np.nan
    Ladv = np.nan
    nuLnu = np.nan
    return Mdot, Ladv, nuLnu, Ftot, img

def load_mov(fs, **kwargs):
    if isinstance(fs, str):
        fs = [fs]

    times = []
    imgs  = [] # collect arrays in list and then cast to np.array() in
               # d.Image() all at once is faster than concatenate
    for f in fs:
        img = load_img(f, **kwargs)
        times.append(img.meta.time)
        imgs.append(img)

    meta = img.meta
    meta.time = units.Quantity(times)
    return d.Image(imgs, meta=meta)
