#!/usr/bin/env python3
#
# Copyright (C) 2021 Chi-kwan Chan
# Copyright (C) 2021 Steward Observatory
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pathlib   import Path
from itertools import product
from importlib import import_module

import numpy  as np
import pandas as pd
import h5py

from tqdm import tqdm
from yaml import safe_load

from scipy.interpolate import RegularGridInterpolator
from scipy.signal      import argrelextrema

from common import hallmark      as hm
from common import dalt
from common import mockservation as mk

def cache_null(src_fmt, dst_fmt,
               params=None, order=['mag', 'aspin', 'Rhigh', 'inc'],
               dpa=45, nmin=2.5e9, nmax=3.5e9, bmin=6e9,  bmax=8e9, lamp=0.04, scat=0.5,
               **kwargs):

    dlen = 0 # for pretty format in `tqdm`

    # Find input models using hallmark `ParaFrame`
    pf = hm.ParaFrame(src_fmt)
    if len(pf) == 0:
        print('No input found; please try different options')
        exit(1)

    # Automatically determine parameters if needed, turn `params` into
    # a dict of parameters and their unique values
    if params is None:
        params = list(pf.keys())
        params.remove('path')
        for k in order:
            params.remove(k)
    params = {p:np.unique(pf[p]) for p in params}

    # Main loop for generating multiple image caches
    for values in product(*params.values()):
        criteria = {p:v for p, v in zip(params.keys(), values)}

        # Check output file
        dst = Path(dst_fmt.format(**criteria))
        if dst.is_file():
            print(f'  "{dst}" exists; SKIP')
            continue

        # Select models according to `criteria`
        sel = pf
        for p, v in criteria.items():
            sel = sel(**{p:v})
        if len(sel) == 0:
            print(f'  No input found for {criteria}; SKIP')
            continue

        # Pretty format in `tqdm`
        desc = f'* "{dst}"'
        desc = f'{desc:<{dlen}}'
        dlen = len(desc)

        # Make sure that the summary table is sorted correctly
        sel = sel.sort_values(order)

        # Actually load the cache and perform the analysis
        tab = pd.DataFrame(columns=order+['score'])
        uvd = np.linspace(0, 1.024e10, 1024)
        for i, row in tqdm(list(sel.iterrows()), desc=desc):
            with h5py.File(row.path) as h:
                m    = h['meta']
                meta = dalt.ImageMeta(**{k:m[k][()] for k in m.keys()})
                data = h['data'][:]

            mov = dalt.Image(data, meta=meta)
            vis = mk.mockserve(mov, N=256)

            U, V = vis.uvd

            u = np.linspace( 0,   U/2, num=vis.shape[-1])
            v = np.linspace(-V/2, V/2, num=vis.shape[-2], endpoint=False)
            t = vis.meta.time.value

            # Ugly hack...
            if t[0] == t[1]:
                print('WARNING: t[0] == t[1]')
                t[0] = t[1] - (t[2] - t[1])

            amp = RegularGridInterpolator((t, v, u[::-1]), abs     (vis[...,::-1]))
            #phi = RegularGridInterpolator((t, v, u[::-1]), np.angle(vis[...,::-1]))

            good = 0
            for t0 in t:

                null_pass = False
                lamp_pass = False

                for j in range(-90,90,dpa):
                    phi = np.pi * j / 180

                    u = uvd * np.cos(phi)
                    v = uvd * np.sin(phi)

                    mask = u <= 0

                    p = np.array([np.repeat(t0, np.sum( mask)),  v[ mask],  u[ mask]]).T
                    m = np.array([np.repeat(t0, np.sum(~mask)), -v[~mask], -u[~mask]]).T

                    s = np.zeros(len(uvd))
                    s[ mask] = amp(p)
                    s[~mask] = amp(m)

                    lc = argrelextrema(s, np.less)[0]
                    for ni in lc:
                        if nmin <= uvd[ni] and uvd[ni] <= nmax:
                            null_pass = True

                    la = np.median(s[(bmin <= uvd) & (uvd <= bmax)])
                    if la * scat < lamp:
                        lamp_pass = True

                    # print(t0, j, uvd[ni], s[ni], la)

                    if null_pass and lamp_pass:
                        good += 1
                        break

            out = {k:row[k] for k in order}
            out['score'] = good / len(t)

            tab = tab.append(out, ignore_index=True)

        # Only touch file system if everything works
        dst.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(dst, sep='\t', index=False)


#==============================================================================
# Make cache_null() callable as a script

import click

@click.command()
@click.argument('args', nargs=-1)
def cmd(args):

    confs  = []
    params = {}
    for arg in args:
        if '=' in arg:
            p = arg.split('=')
            params[p[0]] = p[1]
        else:
            confs.append(arg)

    for c in confs:
        with open(c) as f:
            cache_null(**safe_load(f), **params)

if __name__ == '__main__':
    cmd()
