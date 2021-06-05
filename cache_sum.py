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

    pf = hm.ParaFrame(f'data/{repo}/{mag}a{aspin}_w{window}/'+
                       'img_s{snapshot:d}_Rh{Rhigh:g}_i{inc:g}.h5')
    if len(pf) == 0:
        print('No input file found; try passing in different options')
        exit(1)

    Rhigh = np.unique(pf['Rhigh'])
    inc   = np.unique(pf['inc'])
    dlen  = 0 # for pretty format

    for Rh in Rhigh:
        for i in inc:
            d = Path(f'cache/{repo}/{mag}a{aspin}')
            f = d.joinpath(f'sum_Rh{Rh:g}_i{i:g}_w{window}.tsv')

            if f.is_file():
                print(f'  "{f}" exists; SKIP')
                continue
            else:
                desc = f'* "{f}"'
                desc = f'{desc:<{dlen}}' # pretty format
                dlen = len(desc)

            sel = pf(Rhigh=Rh)(inc=i).sort_values('snapshot')
            tab = []

            for p in tqdm(sel.path, desc=desc):
                with h5py.File(p, "r") as h:
                    Mdot  = h['Mdot'][()]
                    Ladv  = h['Ladv'][()]
                    nuLnu = h['nuLnu'][()]
                    Ftot  = h['Ftot'][()]
                    img   = io.load_img(h)

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

            d.mkdir(parents=True, exist_ok=True)
            tab.to_csv(f, sep='\t', index=False)

#==============================================================================
# Make cache_sum() callable as a script

import click

@click.command()
@click.option('-r','--repo',   default=None, help='Data repository')
@click.option('-m','--mag',    default=None, help='Magnetization')
@click.option('-a','--aspin',  default=None, help='Black hole spin')
@click.option('-w','--window', default=None, help='Time window')
def cmd(**kwargs):
    pf = hm.ParaFrame('data/{repo}/{mag}a{aspin}_w{window}')
    for k, v in kwargs.items():
        if v is not None:
            pf = pf(**{k:v})

    for row in pf.itertuples(index=False):
        print(f'Source repo "{row[0]}":')
        cache_sum(*row[1:])

if __name__ == '__main__':
    cmd()
