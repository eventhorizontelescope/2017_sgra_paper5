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
from matplotlib import pyplot as plt


def show(imgs, s=None, ax=None, **kwargs):

    if imgs.ndim != 2:
        if s is None:
            raise ValueError('must specify snapshot number for movie')
        img = imgs[s,:,:]
    else:
        img = imgs

    u    = imgs.fov.unit
    r, t = imgs.fov.value / 2

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.imshow(img.T, origin='lower', extent=[-r, r, -t, t], **kwargs)
    ax.set_xlabel(f'Relative R.A. [{u:latex}]')
    ax.set_ylabel(f'Relative Declination [{u:latex}]')

    return ax


def ellipse(ax, a0, b0, major, minor, theta, diameter=False, **kwargs):
    phi    = np.arange(361) * np.pi / 180
    theta *=                  np.pi / 180

    if diameter:
        major /= 2
        minor /= 2

    x = major * np.cos(phi)
    y = minor * np.sin(phi)

    a = x * np.cos(theta) - y * np.sin(theta) + a0
    b = x * np.sin(theta) + y * np.cos(theta) + b0

    ax.plot(a, b, linewidth=1, **kwargs)
    ax.plot([a[0],a[180]], [b[0], b[180]], '--', linewidth=1, **kwargs)
