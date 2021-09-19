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

import numpy  as np
import pandas as pd

from astropy import units
from parse   import parse

from . import scale as s
from . import dalt  as d

def load_img(f, **kwargs):

    pf = pd.read_csv(f, delim_whitespace=True)
    x  = pf.x.unique()
    y  = pf.y.unique()
    z  = np.zeros((len(x), len(y)))
    z[pf.y-1, pf.x-1] = pf.z

    # TODO: for EHT work, we need file dependent freq and time (and
    # probably width and height).  One possible hack is to read these
    # information off from the file path `f` using the parse package.
    # Here's an example on how to do it...
    f    = 'model/HAMR/230GHz/img_00000.txt'

    fmt  = 'model/HAMR/{freq:g}GHz/img_{snapshot:d}.txt'
    par  = parse(fmt, f).named
    freq = par['freq']          # assume freq is already in GHz
    time = par['snapshot'] * 10 # assume snapshot every 10 M

    return d.Image(z,       # erg/s/Hz/sr/cm2
                   4.140e6, # MBH    in Msun
                   8.127e3, # dist   in parsec
                   freq,    # freq   in GHz
                   time,    # time   in MBH
                   64,      # width  in MBH
                   64,      # height in MBH
                   **kwargs)

def load_summ(f, **kwargs):
    img = load_img(f, **kwargs)
    return None, None, img.nuLnu, img.Fnu, img

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
