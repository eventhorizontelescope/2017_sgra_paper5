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

from math import pi, floor, cos

import numpy as np
import h5py

def load_hdf5(f, i=70, di=10):

    def get(u, k):
        return u[k][()]

    # TODO: use astropy constant
    ME   = 9.1093897e-28
    CL   = 2.99792458e10
    HPL  = 6.6260755e-27
    LSUN = 3.827e33

    nu    = np.power(10.0, f['output/lnu'])   * ME * CL * CL / HPL
    nuLnu = np.array(      f['output/nuLnu']) * LSUN
    # Note that nuLnu dimension is [N_TYPEBINS, N_EBINS, N_THBINS]
    # The theta bins range from 0 to 90 deg.

    nth = nuLnu.shape[-1]
    dth = pi / 2 / nth
    mth = list(range(nth))
    mth = mth + mth[::-1]

    ji = (i - di/2) * nth / 90
    jf = (i + di/2) * nth / 90

    W, T = 0, 0
    for j in range(floor(ji), floor(jf)+1):
        w  = abs(+ cos(max(j,  ji ) * dth)
                 - cos(min(jf, j+1) * dth))
        W += w
        T += w * nuLnu[:,:,mth[j]]

    return get(f, 'params/t'), nu, T.T / W

def load_one(f, **kwargs):
    if isinstance(f, h5py.File):
        return load_hdf5(f, **kwargs)
    with h5py.File(f, 'r') as g:
        return load_hdf5(g, **kwargs)

def load_sed(files, **kwargs):
    nuLnu = []
    for f in files:
        data = load_one(f, **kwargs)
        t    = data[0]

        try:
            if all(nu != data[1]):
                raise ValueError('Frequency bins `nu` do not match')
        except NameError:
            nu = data[1]

        nuLnu += [np.column_stack((np.sum(data[2], axis=-1), data[2]))]

    nuLnu = np.array(nuLnu)
    return (t, nu,
            np.mean(nuLnu, axis=0),
            np.std (nuLnu, axis=0))
