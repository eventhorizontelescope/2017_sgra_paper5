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

from copy import copy

import numpy as np
import h5py
from astropy import units

from . import scale  as s
from . import datlas as d

def load_hdf5(f, pol=False, **kwargs):

    def get(u, k):
        return u[k][()]

    h = f['header']
    c = h['camera']
    u = h['units']

    img  = f['pol'][:,:,0] if pol else f['unpol'][:,:]
    MBH  = (get(u, 'L_unit') * units.cm).to(units.M_sun, equivalencies=s.GR)
    dist = get(h, 'dsource') * units.cm
    freq = get(h, 'freqcgs') * units.Hz

    time  = get(h, 't')
    width = get(c, 'dx')
    try:
        height = get(c, 'dy')
    except:
        height = width

    return d.Image(img, MBH, dist, freq, time, width, height, **kwargs)

def load_img(f, **kwargs):
    if isinstance(f, h5py.File):
        return load_hdf5(f, **kwargs)
    with h5py.File(f, "r") as g:
        return load_hdf5(g, **kwargs)

def load_mov(fs, **kwargs):
    if isinstance(fs, str):
        fs = [fs]

    times = []
    imgs  = [] # collect arrays in list and then cast to np.array()
               # all at once is faster than concatenate
    for f in fs:
        img = load_img(f, **kwargs)
        times.append(img.meta.time)
        imgs.append(img)

    meta = img.meta
    meta.time = units.Quantity(times)
    return d.Image(imgs, meta=meta)