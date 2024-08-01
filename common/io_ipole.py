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

import h5py
from astropy import units

from . import scale as s
from . import dalt  as d
import numpy as np

def load_hdf5(f, pol=True, **kwargs):

    def get(u, k):
        return u[k][()]

    h = f['header']
    c = h['camera']
    u = h['units']

    if pol:
        #nx ny Stokes+Tau_F
        img  = f['pol'][:,:,0:4]
        tauF = f['pol'][:,:,4]
    else:
        # img = np.atleast_3d(f['unpol'][()])
        img  = f['unpol'][()]
        tauF = None

    tauI = f['tau'][()]

    #Note that no flips or transposes have been made.  This may need to occur in analysis scripts.

    MBH  = (get(u, 'L_unit') * units.cm).to(units.M_sun, equivalencies=s.GR)
    dist = get(h, 'dsource') * units.cm
    freq = get(h, 'freqcgs') * units.Hz

    time  = get(h, 't')
    width = get(c, 'dx')
    try:
        height = get(c, 'dy')
    except:
        height = width
    # print(MBH, dist, freq, time, width, height)
    return d.Image(img, MBH, dist, freq, time, width, height, tauI, tauF, **kwargs)

def load_img(f, **kwargs):
    if isinstance(f, h5py.File):
        return load_hdf5(f, **kwargs)
    with h5py.File(f, "r") as g:
        return load_hdf5(g, **kwargs)

def load_summ(f, **kwargs):
    with h5py.File(f, "r") as h:
        Mdot  = h['Mdot'][()]
        Ladv  = h['Ladv'][()]
        nuLnu = h['nuLnu'][()]
        Ftot  = h['Ftot'][()]
        img   = load_img(h, **kwargs)
    return Mdot, Ladv, nuLnu, Ftot, img

def load_mov(fs, mean=False, **kwargs):
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

    #from scipy import ndimage
    #import numpy as np
    #imgs = [ndimage.rotate(im, 140, reshape=False) for im in imgs]

    if mean:
        import numpy as np
        imgs = np.mean(imgs, axis=0)

    return d.Image(imgs, meta=meta)
