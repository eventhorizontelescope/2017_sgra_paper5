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

from pathlib    import Path
from subprocess import run

import ehtplot
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import numpy  as np
import pandas as pd
import h5py

from astropy import units as u
from tqdm    import tqdm

from common import hallmark as hm
from common import io
from common import analyses as mm
from common import viz

def mov_simple(repo, mag, aspin, window):

    pf = hm.ParaFrame(f'data/{repo}/{mag}a{aspin}_w{window}/'+
                       'img_s{snapshot:d}_Rh{Rhigh:g}_i{inc:g}.h5')
    if len(pf) == 0:
        print('No input file found; try passing in different options')
        exit(1)

    Rhigh    = np.unique(pf['Rhigh'])
    inc      = np.unique(pf['inc'])
    snapshot = np.unique(pf['snapshot'])
    dlen     = 0 # for pretty format

    for Rh in Rhigh:
        for i in inc:
            sel = pf(Rhigh=Rh)(inc=i).sort_values('snapshot')
            mov = io.load_mov(sel.path)

            d  = Path(f'cache/{repo}/{mag}a{aspin}')
            f  = d.joinpath(f'summ_Rh{Rh:g}_i{i:g}_w{window}.tsv')
            df = pd.read_csv(f, sep='\t')
            vmax = np.max(df.Fmax)

            d = Path(f'mov/{repo}/{mag}a{aspin}')
            d.mkdir(parents=True, exist_ok=True)
            for s in tqdm(snapshot):
                f = d.joinpath(f'img_Rh{Rh:g}_i{i:g}_w{window}.{s:06d}.png')
                if f.is_file():
                    print(f'  "{f}" exists; SKIP')
                    continue

                fig, ax = plt.subplots(1, 1, figsize=(6,6))
                viz.show(mov, s=s-snapshot[0], ax=ax, vmin=0, vmax=vmax, cmap='afmhot_us')
                fig.savefig(f)
                plt.close()

            f = d.joinpath(f'mov_Rh{Rh:g}_i{i:g}_w{window}.mpg')
            if f.is_file():
                print(f'  "{f}" exists; SKIP')
                continue

            cmd = ['ffmpeg',
                   '-start_number', f'{snapshot[0]}',
                   '-i', f'img_Rh{Rh:g}_i{i:g}_w{window}.%06d.png',
                   '-qmax', '2', f'mov_Rh{Rh:g}_i{i:g}_w{window}.mov']
            print(cmd)
            run(cmd, cwd=d)
                
#==============================================================================
# Make mov_simple() callable as a script

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
        mov_simple(*row[1:])

if __name__ == '__main__':
    cmd()
