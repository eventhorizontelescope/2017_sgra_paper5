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

from pathlib import Path

import numpy  as np
import pandas as pd
import h5py

from astropy import units as u
from tqdm    import tqdm

from common import hallmark as hm
from common import io
from common import analyses as mm

def cache_sum(repo, mag, aspin, window):

    print(f'data/{repo}/{mag}a{aspin}_w{window}/'+
                       'img_s{snapshot:d}_Rh{Rhigh:g}_i{inc:g}.h5')

    pf = hm.ParaFrame(f'data/{repo}/{mag}a{aspin}_w{window}/'+
                       'img_s{snapshot:d}_Rh{Rhigh:g}_i{inc:g}.h5')
    if len(pf) == 0:
        print('No input file found; try passing in different options')
        exit(1)

    for Rh in np.unique(pf['Rhigh']):
        for i in np.unique(pf['inc']):
            sel  = pf(Rhigh=Rh)(inc=i).sort_values('snapshot')
            desc = f'* {repo} {mag} {aspin} {window}: {Rh:g} {i:g}'

            tab = []
            for p in tqdm(sel.path, desc=desc):
                with h5py.File(p, "r") as f:
                    Mdot  = f['Mdot'][()]
                    Ladv  = f['Ladv'][()]
                    nuLnu = f['nuLnu'][()]
                    Ftot  = f['Ftot'][()]
                    img   = io.load_img(f)

                moments = mm.moments(img.value, *img.fov.value, FWHM=True)
                time    = img.meta.time.value
                time_hr = img.meta.time.to(u.hr).value
                tab.append([
                    time, time_hr,
                    Ladv, Mdot, nuLnu, Ftot, np.min(img.value), np.max(img.value),
                    *moments])

            tab = pd.DataFrame(tab, columns=[
                'time', 'time_hr',
                'Mdot', 'Ladv', 'nuLnu', 'Ftot', 'Fmin', 'Fmax',
                'Fsum', 'ra', 'dec', 'major_FWHM', 'minor_FWHM', 'PA'])

            d = f'cache/{repo}/{mag}a{aspin}'
            Path(d).mkdir(parents=True, exist_ok=True)

            f = f'sum_Rh{Rh:g}_i{i:g}_w{window}.tsv'
            tab.to_csv(Path(d).joinpath(f), sep='\t', index=False)

#==============================================================================
# Make cache_sum() callable as a script

import click

@click.command()
@click.option('-r','--repo',  default='Illinois_230GHz',help='Data repository')
@click.option('-m','--mag',   default='M',              help='Magnetization')
@click.option('-a','--aspin', default='0',              help='Black hole spin')
@click.option('-w','--window',default='5',              help='Time window')
def cmd(repo, mag, aspin, window):
    cache_sum(repo, mag, aspin, window)

if __name__ == '__main__':
    cmd()
