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

from astropy import units as u, constants as c
from tqdm    import tqdm
from yaml    import safe_load

from scipy.interpolate import interp2d

from common import hallmark as hm
from common import analyses as mm

types = ['lc',   'loglc',
         'sed',  'logsed',
         'mi1',  'logmi1',
         'mi3',  'logmi3',
         'mi10', 'logmi10',
         'major','logmajor',
         'minor','logminor']

def Fnu_to_nuLnu(nu, Fnu):
    d = 8.127e3 * u.pc
    S = 4 * np.pi * d * d
    return (Fnu*u.Jy * S * nu*u.Hz).to(u.erg/u.second).value

def nuLnu_to_Fnu(nu, nuLnu):
    d = 8.127e3 * u.pc
    S = 4 * np.pi * d * d
    return (nuLnu*(u.erg/u.second) / (S * nu*u.Hz)).to(u.Jy).value

def t_to_hr(t):
    M = 4.14e+6 * u.Msun
    T = c.G * M / c.c**3
    return (t * T).to(u.hr).value

def mi(hr, vals, T=3):
    mis = []

    t = hr - hr[0]
    n = int(np.max(t) // T)
    for i in range(n):
        mask = (i * T <= t) & (t < (i+1)*T)
        m = np.mean(vals[mask])
        s = np.std (vals[mask])
        mis.append(s / m)

    return mis

stat_keys = ['count','mean','std','min','q1','med','q3','max']
def stat(vals):
    m  = np.mean(vals)
    s  = np.std (vals)
    qs = np.percentile(vals, [0, 25, 50, 75, 100])
    return dict(zip(stat_keys, [len(vals),m,s]+list(qs)))

def cache_stat(src_fmt, dst_fmt, freqs,
               params=None, order=['mag', 'aspin', 'Rhigh', 'inc'], **kwargs):

    freq_out = ['86GHz', '230GHz', 'NIR', 'xray']
    freq_map = dict(zip(freq_out, freqs))
    freq_val = dict(zip(freq_out, [86e9, 230e9, 1.4141e+14, 1.45e18]))

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
        print(params, order)
        params.remove('path')
        for k in order:
            params.remove(k)
    params = {p:np.unique(pf[p]) for p in params}

    # Main loop for generating multiple summary tables
    for values in product(*params.values()):

        criteria = {p:v for p, v in zip(params.keys(), values)}

        # Check output file
        dst_generic = dst_fmt.format(freq='*', type='*', **criteria)

        dst = {
            f'{f}_{t}':Path(dst_fmt.format(freq=f, type=t, **criteria))
            for f, t in product(freq_map, types)
        }

        if all(d.is_file() for d in dst.values()):
            print(f'  "{dst_generic}" exists')
            continue

        # Select models according to `criteria`
        sel = pf
        for p, v in criteria.items():
            sel = sel(**{p:v})
        if len(sel) == 0:
            print(f'  No input found for {criteria}; SKIP')
            continue

        # Pretty format in `tqdm`
        desc = f'* "{dst_generic}"'
        desc = f'{desc:<{dlen}}'
        dlen = len(desc)

        # Make sure that the summary table is sorted correctly
        sel = sel.sort_values(order)

        # Actually creating the table
        tab = {
            f'{f}_{t}':pd.DataFrame(columns=order+stat_keys)
            for f, t in product(freq_map, types)
        }

        for i, row in tqdm(list(sel.iterrows()), desc=desc):
            suffix = f"_{freq_map['230GHz']}.tsv"
            path   = row['path']
            if path.endswith(suffix):
                prefix = path[:-len(suffix)]
            else:
                raise ValueError(f'path "{path}" does not end with suffix "{suffix}"')

            summ = {}
            for k, v in list(freq_map.items())[:3]:
                try:
                    summ[k] = pd.read_csv(prefix + f'_{v}.tsv', sep='\t')
                except:
                    pass

            try:
                with h5py.File(prefix.replace('/summ_', '/sed_') + '.h5') as h:
                    time = h['time'][()]
                    nu   = h['nu'  ][()]
                    avg  = h['avg' ][()]
                sed = interp2d(nu, time, avg[:,:,0]) # in nuLnu
            except:
                pass

            for f, t in product(freq_map, types):
                key = f'{f}_{t}'

                # Now we need to do the actual works...
                try:
                    if   t.endswith('lc'):
                        vals = summ[f].Ftot
                    elif t.endswith('sed'):
                        vals = sed(freq_val[f], time)
                        if f != 'xray':
                            vals = nuLnu_to_Fnu(freq_val[f], vals)
                    elif t.endswith('mi1'):
                        vals = mi(summ[f].time_hr, summ[f].Ftot, T=1)
                    elif t.endswith('mi3'):
                        vals = mi(summ[f].time_hr, summ[f].Ftot, T=3)
                    elif t.endswith('mi10'):
                        vals = mi(summ[f].time_hr, summ[f].Ftot, T=10)
                    elif t.endswith('major'):
                        vals = summ[f].major_FWHM
                    elif t.endswith('minor'):
                        vals = summ[f].minor_FWHM
                    else:
                        raise KeyError(f'unknown key "{t}"')
                except:
                    continue

                if t.startswith('log'):
                    vals = np.log10(vals)

                out = {k:row[k] for k in order}
                out.update(stat(vals))

                tab[key] = tab[key].append(out, ignore_index=True)

        # Only touch file system if everything works
        for f, t in product(freq_map, types):
            key = f'{f}_{t}'
            if len(tab[key]) != 0:
                dst[key].parent.mkdir(parents=True, exist_ok=True)
                tab[key].to_csv(dst[key], sep='\t', index=False)

#==============================================================================
# Make cache_stat() callable as a script

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
            cache_stat(**safe_load(f), **params)

if __name__ == '__main__':
    cmd()
