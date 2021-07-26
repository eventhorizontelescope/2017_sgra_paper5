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


class ImageMeta:

    @staticmethod
    def du(q, d):
        try:
            return q.to(d)
        except:
            return q * d

    def __init__(self,
                 mass=None, dist=None, freq=None,
                 time=None, width=None, height=None, scale='cgs'):
        if scale == 'AGN':
            T, L = u.hr, u.au
        else:
            T, L = u.s,  u.cm

        self.mass   = self.du(mass, u.M_sun )
        self.dist   = self.du(dist, u.parsec)
        self.freq   = self.du(freq, u.GHz   )

        tg = u.def_unit('M', self.mass.to(u.s,  s.GR))
        rg = u.def_unit('M', self.mass.to(u.cm, s.GR))

        self.time   = self.du(time,   self.mass.to(tg, s.GR))
        self.width  = self.du(width,  self.mass.to(rg, s.GR))
        self.height = self.du(height, self.mass.to(rg, s.GR))


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
