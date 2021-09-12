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

import numpy as np
import h5py

from tqdm import tqdm
from yaml import safe_load

from common import hallmark as hm

def cache_mov(src_fmt, dst_fmt, img_fmt='ipole',
              params=None, order=['snapshot'], **kwargs):

    io = import_module('common.io_' + img_fmt)

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
        for k in order:
            sel = sel.sort_values(k)

        # Actually load the images
        mov = load_mov(tqdm(sel.path, desc=desc))

        # Only touch file system if everything works
        dst.parent.mkdir(parents=True, exist_ok=True)
        # mov.to_csv(dst, sep='\t', index=False)

#==============================================================================
# Make cache_mov() callable as a script

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
            cache_mov(**safe_load(f), **params)

if __name__ == '__main__':
    cmd()
