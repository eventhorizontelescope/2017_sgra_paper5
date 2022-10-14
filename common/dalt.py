# Copyright 2017 Chi-kwan Chan
# Copyright 2017 Harvard-Smithsonian Center for Astrophysics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math      import pi
from functools import cached_property

import numpy as np
from astropy import units as u

from . import scale as s


def du(q, d):
    try:
        return q.to(d)
    except:
        return q * d


class ImageMeta:

    def __init__(self,
                 mass=None, dist=None, freq=None,
                 time=None, width=None, height=None, 
                 tauI=None, tauF=None):

        self.mass   = du(mass, u.M_sun )
        self.dist   = du(dist, u.parsec)
        self.freq   = du(freq, u.GHz   )

        self.GR = s.GR
        with u.set_enabled_equivalencies(self.GR):
            self.tg = u.def_unit('M', self.mass.to(u.s))
            self.rg = u.def_unit('M', self.mass.to(u.cm))

        self.time   = du(time,   self.tg)
        self.width  = du(width,  self.rg)
        self.height = du(height, self.rg)

        self.tauI = tauI
        self.tauF = tauF

        self.geom = [ # geometry equivalencies
            (self.rg, u.radian, lambda L: L / self.dist.to(self.rg),
                                lambda a: a * self.dist.to(self.rg)),
        ]

    def dict(self):
        return {
            'mass':   self.mass  .to(u.M_sun ).value,
            'dist':   self.dist  .to(u.parsec).value,
            'freq':   self.freq  .to(u.GHz   ).value,
            'time':   self.time  .to(self.tg ).value,
            'width':  self.width .to(self.rg ).value,
            'height': self.height.to(self.rg ).value,
            'tauI':   self.tauI,
            'tauF':   self.tauF
        }

    def set_time(self, time):
        self.time = du(time,   self.tg)


class Image(u.SpecificTypeQuantity):

    @staticmethod
    def angle(y, x):
        return (u.Quantity(y) / x) * u.rad

    _equivalent_unit = (u.erg / u.s) / (u.cm * u.cm * u.sr * u.Hz)

    def __new__(cls,
                img, *args,
                meta=None, unit=None, dtype=None, copy=True, **kwargs):
        if unit is None:
            unit = cls._equivalent_unit

        if meta is None:
            meta = ImageMeta(*args, **kwargs)

        self = super().__new__(cls, img, unit, dtype, copy)
        self.meta = meta
        return self

    def set_time(self, time):
        self.meta.set_time(time)

    @cached_property
    def Fnu(self):
        m = self.meta
        w = self.angle(m.width  / self.shape[-2], m.dist)
        h = self.angle(m.height / self.shape[-1], m.dist)
        return (w * h * np.sum(self, axis=(-2,-1))).to(u.Jy)

    @property
    def Lnu(self):
        d = self.meta.dist
        return (4 * pi * d * d * self.Fnu).to(u.erg / u.s / u.Hz)

    @property
    def nuLnu(self):
        return (self.meta.freq * self.Lnu).to(u.erg / u.s)

    @property
    def fov(self):
        m = self.meta
        return self.angle([-m.width, m.height], m.dist).to(u.uas)

    @property
    def tauI(self):
        return self.meta.tauI

    @property
    def tauF(self):
        return self.meta.tauF

    @property
    def extent(self):
        return np.tensordot(self.fov.value, [-0.5, 0.5], 0).flatten()

    @property
    def extent_labels(self):
        u = self.fov.unit
        return [f'$x$ [{u:latex}]',
                f'$y$ [{u:latex}]']


class VisibilityMeta:

    def __init__(self, U=None, V=None, freq=None, time=None):

        self.freq = du(freq, u.GHz)
        self.time = du(time, u.hr)
        self.U    = U
        self.V    = V

    def dict(self):
        return {
            'freq': self.freq.to(u.GHz).value,
            'time': self.time.to(u.hr ).value,
            'U'   : self.U,
            'V'   : self.V,
        }

class Visibility(u.SpecificTypeQuantity):

    _equivalent_unit = u.Jy

    def __new__(cls,
                vis, *args,
                meta=None, unit=None, dtype=None, copy=True, **kwargs):

        if unit is None:
            unit = cls._equivalent_unit

        if meta is None:
            meta = VisibilityMeta(*args, **kwargs)

        self = super().__new__(cls, vis, unit, dtype, copy)
        self.meta = meta
        return self

    @property
    def uvd(self):
        m = self.meta
        return np.array([m.U, m.V])

    @property
    def extent(self):
        U, V = self.uvd
        return [-U/2, U/2, 0, V/2]

    @property
    def extent_labels(self):
        return ['u [G$\lambda$]', 'v [G$\lambda$]']
