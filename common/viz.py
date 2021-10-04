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

import numpy as np

from matplotlib  import pyplot as plt
from scipy.stats import norm, poisson


def show(vs, s=None, f=None, ax=None, **kwargs):

    if vs.ndim != 2:
        if s is None:
            raise ValueError('must specify snapshot number for movie')
        v = vs[s,:,:]
    else:
        v = vs

    if f is None:
        f = lambda x: x

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(f(v.T), origin='lower', extent=vs.extent, **kwargs)

    labels = vs.extent_labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    return ax


def ellipse(a0, b0, major, minor, PA, ax=None, diameter=False, **kwargs):
    phi = (np.pi / 180) * np.arange(361)
    PA *=  np.pi / 180

    if diameter:
        major /= 2
        minor /= 2

    h = major * np.sin(phi)
    w = minor * np.cos(phi)

    # position angle is measured from vertical axis (North), toward
    # left (East in sky).
    b = h * np.cos(PA) - w * np.sin(PA) + b0
    a = h * np.sin(PA) + w * np.cos(PA) + a0

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(a, b, linewidth=1, **kwargs)
    ax.plot([a[90],a[270]], [b[90],b[270]], '--', linewidth=1, **kwargs)

    return ax


def interval(avg, std, sigma=1):
    """We model the different realizations of grmonty results follow a
    Poisson distribution, which has mean and variance both equal to
    `mu`.  Therefore,

        avg = dnuLnu * mu
        std = dnuLnu * sqrt(mu)

        mu = (avg/std)**2

    One we obtain `mu`, we can estimate the lower and upper intervals
    according to `sigma`.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = (avg/std)**2

    lower = poisson.ppf(norm.cdf(-sigma), mu)
    upper = poisson.ppf(norm.cdf(+sigma), mu)
    units = avg / mu

    return lower * units, upper * units

def step_one(ax, nu, avg, std=None, sigma=1,
             step=True, shade=True, ylog=True, **kwargs):

    if step:
        p = ax.step(nu, avg, where='mid', **kwargs)
    else:
        p = ax.plot(nu, avg, **kwargs)

    if std is not None and shade:
        l, u = interval(avg, std, sigma=sigma)
        ax.fill_between(nu, l, u, step='mid' if step else None,
                        color=p[0].get_color(), alpha=1/3, linewidth=0)

    # x-axis must be in log scale; otherwise the bin boundaries are wrong
    ax.set_xscale('log')
    # optionally we may set y-axis to log sacle
    if ylog:
        ax.set_yscale('log')

def step(ax, nu, avg, std=None, color=None, shade=None, label=None, **kwargs):
    n   = len(nu)
    avg = avg.reshape(n,-1)
    for i in range(avg.shape[-1]):
        stdi   = std.reshape(n,-1)[:,i] if std is not None else None
        colori = color if color is not None else ('k'  if i == 0 else None)
        widthi = 2 if i == 0 else 1
        shadei = shade if shade is not None else (True if i == 0 else False)
        labeli = label[i] if label is not None else None
        step_one(ax, nu, avg[:,i], stdi,
                 color=colori, linewidth=widthi,
                 shade=shadei, label=labeli,
                 **kwargs)


def grid(pf, plot,
         fout   = None,
         title  = None,
         xtitle = None,
         ytitle = None,
         xlabel = None,
         ylabel = None,
         xspace = 0.05,
         yspace = 0,
         legend = None,
         **kwargs):


    keys   = list(kwargs.keys())
    colkey = keys[0]
    cols   = kwargs.pop(keys[0])
    rowkey = keys[1]
    rows   = kwargs.pop(keys[1])

    fig, axes = plt.subplots(len(rows), len(cols),
                             #sharex=True, sharey=True,
                             **kwargs)
    if len(rows) == 1:
        axes = [axes]
    if len(cols) == 1:
        axes = [[a] for a in axes]

    for i, c in enumerate(cols):
        for j, r in enumerate(rows):

            plot(axes[j][i], pf(**{colkey:c})(**{rowkey:r}))

            if i == 0:
                axes[j][i].set_ylabel(ylabel)
            if i == len(cols)-1 and ytitle is not None:
                ax_r = axes[j][i].twinx()
                ax_r.set_ylabel(ytitle.format(r))
                ax_r.tick_params(axis='both',
                                 direction='in',
                                 labelright=False)

            if j == 0 and xtitle is not None:
                    axes[j][i].set_title(xtitle.format(c))
            if j == len(rows)-1:
                axes[j][i].set_xlabel(xlabel)

            axes[j][i].tick_params(axis='both',
                                   direction='in',
                                   top=True,
                                   right=True)

    if legend is not None:
        axes[0][-1].legend(loc=legend)

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(wspace=xspace, hspace=yspace)
    if fout:
        fig.savefig(fout+'.pdf')
        fig.savefig(fout+'.png', dpi=300)

    return fig
