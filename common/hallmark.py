# Copyright 2019 Chi-kwan Chan
# Copyright 2019 Steward Observatory
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

from glob import glob

import re
import parse
import pandas as pd

def filter(self, **kwargs):
    mask = [False] * len(self)
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            mask |= self[k].isin(v)
        else:
            mask |= self[k] == v
    return self[mask]

pd.DataFrame.__call__ = filter # monkey patch pandas DataFrame

def ParaFrame(fmt, *args, debug=False, **kwargs):
    pattern = fmt

    for i in range(len(fmt) // 3):
        if debug:
            print(i, pattern, args, kwargs)
        try:
            pattern = pattern.format(*args, **kwargs)
            break
        except KeyError as e:
            k = e.args[0]
            pattern = re.sub(r'\{'+k+':?.*?\}', '{'+k+':s}', pattern)
            kwargs[e.args[0]] = '*'

    files = sorted(glob(pattern))
    if debug:
        print(f'Pattern: "{pattern}"')
        n = len(files)
        if n > 1:
            print(f'{n} matches, e.g., "{files[0]}"')
        elif n > 0:
            print(f'{n} match, i.e., "{files[0]}"')
        else:
            print(f'No match; please check format string')

    parser = parse.compile(fmt)

    l = []
    for f in files:
        r = parser.parse(f)
        if r is None:
            print(f'Failed to parse "{f}"')
        else:
            l.append({'path':f, **r.named})
    return pd.DataFrame(l)
