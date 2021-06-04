# Copyright (C) 2019 Chi-kwan Chan
# Copyright (C) 2019 Steward Observatory
#
# This file is part of `foci`.
#
# `Foci` is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `Foci` is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `foci`.  If not, see <http://www.gnu.org/licenses/>.

from astropy import units as u, constants as c

GR = [
    (u.kg, u.m, lambda M: (c.G * M) / c.c**2, lambda R: (R * c.c**2) / c.G),
    (u.kg, u.s, lambda M: (c.G * M) / c.c**3, lambda T: (T * c.c**3) / c.G),
    (u.s,  u.m, lambda T: T * c.c,            lambda R: R / c.c),
]
