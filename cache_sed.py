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

from astropy import units as u
from tqdm    import tqdm
from yaml    import safe_load

from common import hallmark    as hm
from common import io_igrmonty as io

def cache_sed(src_fmt, dst_fmt,
              params=None, order=['snapshot', 'realization'], **kwargs):

    dlen = 0 # for pretty format in `tqdm`

    # Find input models using hallmark `ParaFrame`
    pf = hm.ParaFrame(src_fmt, **kwargs)
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

    # Main loop for generating multiple SEDs
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
        for k in order:
            sel = sel.sort_values(k)

        # Actually combine the SEDs
        time, avgs, errs, lens = [], [], [], []
        for s in tqdm(np.unique(sel.snapshot), desc=desc):
            paths = sel(snapshot=s).path
            t, nu, avg, err = io.load_sed(paths, i=criteria['inc'])
            time.append(t)
            avgs.append(avg)
            errs.append(err)
            lens.append(len(paths))

        knd = np.array([
            "total",
            "(synch) base", "(synch) once", "(synch) twice", "(synch) > twice",
            "(brems) base", "(brems) once", "(brems) twice", "(brems) > twice",
        ], dtype='a16')

        # Only touch file system if everything works
        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, 'w') as f:
            f['time'] = time
            f['nu']   = nu
            f['knd']  = knd
            f['avg']  = np.array(avgs)
            f['err']  = np.array(errs)
            f['len']  = np.array(lens)

#==============================================================================
# Make cache_sed() callable as a script

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
            cache_sed(**safe_load(f), **params)

if __name__ == '__main__':
    cmd()
