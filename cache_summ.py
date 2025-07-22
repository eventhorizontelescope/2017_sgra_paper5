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

from common import hallmark as hm
from common import analyses as mm
import pdb

def cache_summ(src_fmt, dst_fmt, 
               params=None, order=['snapshot'], FWHM=True, **kwargs):

    file_ending = src_fmt.split('.')[-1]
    if file_ending == 'fits':
        img_fmt = 'fits'
    elif file_ending == 'h5':
        img_fmt = 'ipole'
    else:
        raise ValueError("You are asking for neither h5 nor fits files, which are not implemented.")

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

    # Main loop for generating multiple summary tables
    for values in product(*params.values()):
        criteria = {p:v for p, v in zip(params.keys(), values)}

        if 'mag' in criteria.keys():
            if not (criteria['mag'] in ['MAD', 'SANE', 'M', 'S', 'mad', 'sane']):
                #This is helping me catch folders representing old runs.  Skip them.
                continue

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

        # Actually creating the table
        tab = []
        for p in tqdm(sel.path, desc=desc):
            Mdot, Ladv, nuLnu, Ftot, img = io.load_summ(p)

            moments = mm.moments(img.value, *img.fov.value, FWHM=FWHM)
            unresolvedPolarizationFractions = mm.unresolvedFractionalPolarizations(img)
            resolvedPolarizationFractions = mm.resolvedFractionalPolarizations(img)
            betas = mm.computeBetaCoefficients(img, m_list=[1,2,3,4,5])
            opticalDepth = mm.computeOpticalDepth(img)
            faradayDepth = mm.computeFaradayDepth(img)
            time    = img.meta.time.value
            time_hr = img.meta.time.to(u.hr).value
            tab.append([
                p, time, time_hr,
                Ladv, Mdot, nuLnu, Ftot, np.min(img.value), np.max(img.value),
                *moments, *unresolvedPolarizationFractions, *resolvedPolarizationFractions, 
                *betas[0], *betas[1], *betas[2], *betas[3], *betas[4], 
                opticalDepth, faradayDepth])

        # Turn list of of list into pandas data frame
        tab = pd.DataFrame(tab, columns=[
            'file_path', 
            'time', 'time_hr',
            'Mdot', 'Ladv', 'nuLnu', 'Ftot',
            'Imin', 'Imax', 'Imean',
            'alpha0', 'beta0', 'major_FWHM', 'minor_FWHM', 'PA', 
            'mnet', 'vnet', 'mavg', 'vavg', 
            'b1_amp', 'b1_phase', 
            'b2_amp', 'b2_phase', 
            'b3_amp', 'b3_phase', 
            'b4_amp', 'b4_phase', 
            'b5_amp', 'b5_phase', 
            'tauI', 'tauF']
        )

        # Only touch file system if everything works
        dst.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(dst, sep='\t', index=False)

#==============================================================================
# Make cache_summ() callable as a script

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
            cache_summ(**safe_load(f), **params)

if __name__ == '__main__':
    cmd()
